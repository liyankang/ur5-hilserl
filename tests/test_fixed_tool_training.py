"""固定工具训练链路的离线回归；不导入 JAX、相机或机械臂驱动。"""
import ast
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from examples.collection_utils import (
    apply_manual_success, ensure_training_schema, make_transition,
    validate_action_space, validate_parameter_shapes, validate_transition_shapes,
)
from tests.test_collection import load_definition


class Box:
    """测试只需要 Box 的形状、边界和采样接口。"""
    def __init__(self, low=-1, high=1, shape=None, dtype=np.float32):
        self.shape = tuple(shape) if shape is not None else tuple(np.asarray(low).shape)
        self.dtype = dtype
        self.low = np.full(self.shape, low, dtype=dtype)
        self.high = np.full(self.shape, high, dtype=dtype)

    def sample(self):
        return np.zeros(self.shape, dtype=self.dtype)


class DictSpace:
    def __init__(self, spaces):
        self.spaces = spaces


def spaces():
    observations = DictSpace({"state": Box(shape=(1, 19)),
                              "wrist": Box(0, 255, shape=(1, 2, 2, 3), dtype=np.uint8)})
    return observations, Box(shape=(6,))


def transition():
    obs = {"state": np.zeros((1, 19)), "wrist": np.zeros((1, 2, 2, 3), dtype=np.uint8)}
    return make_transition(obs, np.zeros(6), obs, 0, False, True, {})


def replay_classes():
    """执行真实经验池实现，仅替换 Gym 空间与 Dataset 基类。"""
    class Dataset:
        def __init__(self, dataset_dict):
            self.dataset_dict = dataset_dict

    namespace = dict(
        np=np, copy=copy, gym=types.SimpleNamespace(Space=object, spaces=types.SimpleNamespace(Box=Box, Dict=DictSpace)),
        Dataset=Dataset, DatasetDict=dict, Union=Union, Optional=Optional,
        Iterable=Iterable, Tuple=Tuple, Box=Box,
        frozen_dict=types.SimpleNamespace(FrozenDict=dict),
    )
    for name in ("_init_replay_dict", "_insert_recursively", "ReplayBuffer"):
        namespace[name] = load_definition("serl_launcher/data/replay_buffer.py", name, namespace)
    memory = load_definition("serl_launcher/data/memory_efficient_replay_buffer.py", "MemoryEfficientReplayBuffer", namespace)
    return namespace["ReplayBuffer"], memory


class FixedToolTrainingTests(unittest.TestCase):
    def test_action_dimension_and_setup_are_checked_before_building_network(self):
        config = types.SimpleNamespace(action_dim=6, setup_mode="single-arm-fixed-gripper")
        env = types.SimpleNamespace(action_space=Box(shape=(6,)))
        self.assertEqual(validate_action_space(config, env).shape, (6,))
        env.action_space = Box(shape=(7,))
        with self.assertRaisesRegex(ValueError, "without a gripper"):
            validate_action_space(config, env)
        env.action_space = Box(shape=(6,))
        config.setup_mode = "single-arm-learned-gripper"
        with self.assertRaisesRegex(ValueError, "without gripper critic"):
            validate_action_space(config, env)

    def test_actual_applied_action_and_intervention_survive_copy(self):
        info = {"applied_action": np.full(6, 0.1), "intervene_action": np.full(6, 0.8), "is_intervention": True}
        sample = make_transition(0, np.ones(6), 1, 0, False, True, info)
        np.testing.assert_allclose(sample["actions"], 0.1)
        self.assertTrue(sample["is_intervention"])
        self.assertEqual((sample["dones"], sample["masks"]), (True, 1))
        info["applied_action"][:] = 9
        np.testing.assert_allclose(sample["actions"], 0.1)

    def test_manual_success_holds_robot_and_synchronizes_outcome(self):
        held = []
        env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(hold_position=lambda: held.append(True)))
        info = {"manual_success": True, "succeed": False}
        reward, done, truncated, info = apply_manual_success(0, False, True, info, env=env)
        self.assertEqual((reward, done, truncated, info["succeed"]), (1, True, False, True))
        self.assertEqual(held, [True])
        held.clear()
        info = {"manual_success": True, "safety_stop": True, "succeed": False}
        result = apply_manual_success(0, True, False, info, env=env)
        self.assertEqual(result[:3], (0, True, False))
        self.assertFalse(info["succeed"])
        self.assertEqual(held, [])

    def test_replay_and_compressed_replay_store_intervention_flag(self):
        replay_cls, memory_cls = replay_classes()
        obs_space, action_space = spaces()
        for cls in (replay_cls, memory_cls):
            with self.subTest(cls=cls.__name__):
                kwargs = {"pixel_keys": ("wrist",)} if cls is memory_cls else {}
                buffer = cls(obs_space, action_space, capacity=10, include_intervention=True, **kwargs)
                sample = transition()
                sample["is_intervention"] = True
                buffer.insert(sample)
                index = buffer._insert_index - 1
                self.assertTrue(buffer.dataset_dict["is_intervention"][index])
                sample.pop("is_intervention")
                buffer.insert(sample)
                self.assertFalse(buffer.dataset_dict["is_intervention"][buffer._insert_index - 1])

    def test_old_demo_dimensions_are_rejected_without_silent_slicing(self):
        obs_space, action_space = spaces()
        sample = transition()
        validate_transition_shapes(sample, obs_space, action_space)
        for field in ("actions", "state", "image"):
            with self.subTest(field=field):
                bad = copy.deepcopy(sample)
                if field == "actions":
                    bad["actions"] = np.zeros(7)
                elif field == "state":
                    bad["observations"]["state"] = np.zeros((1, 18))
                else:
                    bad["next_observations"]["wrist"] = np.zeros((2, 2, 3))
                with self.assertRaisesRegex(ValueError, "recollect demos"):
                    validate_transition_shapes(bad, obs_space, action_space, "old.pkl")

    def test_schema_records_input_output_and_rejects_legacy_checkpoints(self):
        obs_space, action_space = spaces()
        env = types.SimpleNamespace(observation_space=obs_space, action_space=action_space)
        config = types.SimpleNamespace(action_dim=6, proprio_keys=["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"],
                                       setup_mode="single-arm-fixed-gripper")
        with tempfile.TemporaryDirectory() as directory:
            ensure_training_schema(config, env, directory, "ram_insertion")
            payload = json.loads((Path(directory) / "training_schema.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["action_shape"], [6])
            self.assertEqual(payload["observation_shapes"]["state"], [1, 19])
            ensure_training_schema(config, env, directory, "ram_insertion", has_checkpoint=True)
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                ensure_training_schema(config, env, directory, "ram_insertion", has_checkpoint=True, algorithm="bc")
            config.proprio_keys.remove("gripper_pose")
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                ensure_training_schema(config, env, directory, "ram_insertion", has_checkpoint=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "schema is unverified"):
                ensure_training_schema(config, env, directory, "ram_insertion", has_checkpoint=True)

    def test_actual_checkpoint_parameter_shapes_are_checked(self):
        expected = {"actor": {"head": np.zeros((32, 6))}, "encoder": np.zeros((19, 32))}
        validate_parameter_shapes(expected, copy.deepcopy(expected))
        for wrong in ({"actor": {"head": np.zeros((32, 7))}, "encoder": np.zeros((19, 32))},
                      {"actor": {"head": np.zeros((32, 6))}, "encoder": np.zeros((18, 32))}):
            with self.assertRaisesRegex(ValueError, "network parameter shapes"):
                validate_parameter_shapes(expected, wrong)

    def test_auto_translation_does_not_create_rotation_at_offset_tcp(self):
        class ZRotation:
            def __init__(self, angle):
                self.angle = angle
            @classmethod
            def from_quat(cls, q):
                return cls(2 * np.arctan2(q[2], q[3]))
            @classmethod
            def from_rotvec(cls, value):
                np.testing.assert_allclose(np.asarray(value)[:2], 0, atol=1e-9)
                return cls(value[2])
            def inv(self):
                return ZRotation(-self.angle)
            def __mul__(self, other):
                return ZRotation(self.angle + other.angle)
            def as_rotvec(self):
                return np.array([0., 0., self.angle])
            def as_mrp(self):
                return np.array([0., 0., np.tan(self.angle / 4)])
            def as_matrix(self):
                c, s = np.cos(self.angle), np.sin(self.angle)
                return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        compute = load_definition("examples/auto_trajectory.py", "compute_auto_step_action", {"np": np, "Rot": ZRotation})
        base = types.SimpleNamespace(currpos=np.array([.2, -.4, .8, 0, 0, np.sqrt(.5), np.sqrt(.5)]),
                                     action_scale=(.01, .01, 1))
        target = np.array([.21, -.4, .8, 0, 0, np.pi / 2])
        action = compute(base, target)
        np.testing.assert_allclose(action[:3], [0, -.2, 0], atol=1e-7)
        np.testing.assert_allclose(action[3:], 0, atol=1e-7)

    def test_auto_manual_success_uses_applied_action_and_stops(self):
        class Env:
            action_scale = (.01, .01, 1)
            def __init__(self):
                self.unwrapped, self.holds = self, 0
            def hold_position(self):
                self.holds += 1
            def step(self, action):
                return 1, 0, False, False, {"manual_success": True, "is_intervention": True,
                                            "applied_action": np.full(6, .1)}
        move = load_definition("examples/auto_trajectory.py", "auto_move_to_target", dict(
            np=np, copy=copy, make_transition=make_transition, apply_manual_success=apply_manual_success,
            compute_auto_step_action=lambda *args, **kwargs: np.ones(6)))
        env = Env()
        samples, interrupted = move(env, None, initial_obs=0)
        self.assertFalse(interrupted)
        self.assertEqual(env.holds, 1)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["rewards"], 1)
        self.assertTrue(samples[0]["infos"]["succeed"])
        np.testing.assert_allclose(samples[0]["actions"], .1)

    def test_real_learner_requests_half_human_half_online_batches(self):
        class Buffer:
            def __init__(self):
                self.requests = []
            def __len__(self):
                return 4
            def get_iterator(self, **kwargs):
                self.requests.append(kwargs)
                return iter([])
        class Server:
            def __init__(self, *args, **kwargs):
                pass
            def register_data_store(self, *args):
                pass
            def start(self, **kwargs):
                pass
            def publish_network(self, params):
                pass
        class Progress:
            def __init__(self, iterable=None, **kwargs):
                self.iterable, self.n = iterable or [], kwargs.get("initial", 0)
            def __iter__(self):
                return iter(self.iterable)
            def update(self, count):
                self.n += count
            def close(self):
                pass
        class Agent:
            state = types.SimpleNamespace(params={})
        config = types.SimpleNamespace(batch_size=64, training_starts=1, max_steps=0)
        learner = load_definition("examples/train_rlpd.py", "learner", dict(
            config=config, FLAGS=types.SimpleNamespace(checkpoint_path=None),
            TrainerServer=Server, make_trainer_config=lambda: None, print_green=lambda text: None,
            tqdm=types.SimpleNamespace(tqdm=Progress), sharding=types.SimpleNamespace(replicate=lambda: None),
            Timer=lambda: None, SACAgent=Agent))
        online, human = Buffer(), Buffer()
        learner(None, Agent(), online, human)
        self.assertEqual(online.requests[0]["sample_args"]["batch_size"], 32)
        self.assertEqual(human.requests[0]["sample_args"]["batch_size"], 32)


if __name__ == "__main__":
    unittest.main()
