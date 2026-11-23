"""无硬件回归：原生六维动作、TCP 坐标变换、物理步长与成功停推。"""
import ast
import copy
from pathlib import Path
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Rotation:
    """独立的四元数测试实现，避免测试机依赖 SciPy 或机器人驱动。"""
    def __init__(self, quat):
        self.q = np.asarray(quat, dtype=float)
        self.q = self.q / np.linalg.norm(self.q)

    @classmethod
    def from_quat(cls, quat):
        return cls(quat)

    @classmethod
    def from_rotvec(cls, vector):
        vector = np.asarray(vector, dtype=float)
        angle = np.linalg.norm(vector)
        return cls([0, 0, 0, 1]) if angle < 1e-14 else cls(np.r_[vector / angle * np.sin(angle / 2), np.cos(angle / 2)])

    @classmethod
    def from_mrp(cls, p):
        p = np.asarray(p, dtype=float)
        square = p @ p
        return cls(np.r_[2 * p / (1 + square), (1 - square) / (1 + square)])

    def as_quat(self):
        return self.q.copy()

    def inv(self):
        return Rotation(np.r_[-self.q[:3], self.q[3]])

    def __mul__(self, other):
        a, b = self.q[:3], other.q[:3]
        return Rotation(np.r_[self.q[3] * b + other.q[3] * a + np.cross(a, b),
                              self.q[3] * other.q[3] - a @ b])

    def as_rotvec(self):
        q = self.q if self.q[3] >= 0 else -self.q
        norm = np.linalg.norm(q[:3])
        return np.zeros(3) if norm < 1e-14 else q[:3] / norm * (2 * np.arctan2(norm, q[3]))

    def as_mrp(self):
        return self.q[:3] / (1 + self.q[3])

    def as_matrix(self):
        x, y, z, w = self.q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shape) if shape else np.asarray(low, dtype=dtype)
        self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shape) if shape else np.asarray(high, dtype=dtype)
        self.shape = self.low.shape


class Dict:
    def __init__(self, spaces):
        self.spaces = spaces

    def __getitem__(self, key):
        return self.spaces[key]


class Wrapper:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)


def definition(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def gym_stub():
    return types.SimpleNamespace(Env=object, Wrapper=Wrapper, ActionWrapper=Wrapper,
                                 spaces=types.SimpleNamespace(Box=Box, Dict=Dict))


def robot_env(gripper=False):
    namespace = dict(gym=gym_stub(), np=np, Rotation=Rotation, Dict=dict, copy=copy,
                     DefaultEnvConfig=object,
                     time=types.SimpleNamespace(time=lambda: 0.0, sleep=lambda dt: None))
    cls = definition("ur_env/envs/ur_env.py", "urEnv", namespace)
    config = types.SimpleNamespace(
        ACTION_SCALE=(0.005, 0.01, 1.0), ACTION_ROTATION_REPR="rotvec", POSE_ROTATION_REPR="rotvec",
        TARGET_POSE=np.array([0, 0, 0, 0, 0, 0], dtype=float), RESET_POSE=np.array([0, 0, 0.1, 0, 0, 0]),
        REWARD_THRESHOLD=np.zeros(6), SERVER_URL="http://unused/", MAX_EPISODE_LENGTH=20,
        DISPLAY_IMAGE=False, GRIPPER_SLEEP=0, GRIPPER_ENABLED=gripper, CAMERAS={"wrist": {}},
        RANDOM_RESET=False, RANDOM_XY_RANGE=0, RANDOM_RZ_RANGE=0,
        JOINT_RESET_PERIOD=0, ABS_POSE_LIMIT_LOW=np.array([-1, -1, -1, -4, -4, -4]),
        ABS_POSE_LIMIT_HIGH=np.array([1, 1, 1, 4, 4, 4]),
        MAX_TRANSLATION_STEP_M=0.005, MAX_ROTATION_STEP_RAD=0.01)
    env = cls(config=config, fake_env=True)
    env.curr_path_length = 0  # 测试已直接注入初始状态，相当于完成一次 reset。
    env.use_pose_reward = False
    env.commands = []
    env.gripper_commands = []
    env.camera_reads = 0
    env._send_pos_command = lambda pose, source: env.commands.append(pose.copy())
    env._send_gripper_command = lambda value: env.gripper_commands.append(value)
    env._update_currpos = lambda: setattr(env, "currpos", env.nextpos.copy())
    def get_im():
        env.camera_reads += 1
        return {"wrist": np.ones((2, 2, 3)) * env.camera_reads}
    env.get_im = get_im
    # 与 Gym Env 的 unwrapped 属性提供同样接口。
    env.unwrapped = env
    return env


class FixedToolControlTests(unittest.TestCase):
    def test_native_six_dimensions_never_send_gripper_and_limit_vector_norms(self):
        env = robot_env()
        self.assertEqual(env.action_space.shape, (6,))
        before = env.currpos.copy()
        _, _, _, _, info = env.step(np.ones(6))
        self.assertEqual(env.gripper_commands, [])
        self.assertLessEqual(np.linalg.norm(env.currpos[:3] - before[:3]), 0.005 + 1e-9)
        actual_rotation = Rotation.from_quat(env.currpos[3:]) * Rotation.from_quat(before[3:]).inv()
        self.assertLessEqual(np.linalg.norm(actual_rotation.as_rotvec()), 0.01 + 1e-9)
        np.testing.assert_allclose(info["applied_action"][:3] * 0.005, env.currpos[:3] - before[:3])
        with self.assertRaises(ValueError):
            env.step(np.zeros(7))
        with self.assertRaises(ValueError):
            env.step(np.full(6, np.nan))

    def test_legacy_gripper_wrapper_is_identity_for_native_six_dimensions(self):
        cls = definition("ur_env/envs/wrappers.py", "GripperCloseEnv", dict(gym=gym_stub(), np=np, Box=Box))
        for gripper in (False, True):
            env = robot_env(gripper=gripper)
            wrapper = cls(env)
            self.assertEqual(wrapper.action_space.shape, (6,))
            self.assertEqual(wrapper.action(np.ones(6)).shape, (7,) if gripper else (6,))
            if gripper:
                self.assertEqual(wrapper.action(np.ones(6))[-1], 0)

    def test_interpolation_does_not_multiply_physical_step_budget(self):
        env = robot_env()
        cls = definition("ur_env/envs/wrappers.py", "ActionInterpolationWrapper",
                         dict(gym=gym_stub(), np=np, KeyBoardIntervention=type("UnusedKeyboard", (), {})))
        wrapper = cls(env, policy_freq=20, control_freq=100)
        start = env.currpos.copy()
        env._get_obs()
        _, _, _, _, info = wrapper.step(np.ones(6))
        previous = start
        distance = angle = 0
        for pose in env.commands:
            distance += np.linalg.norm(pose[:3] - previous[:3])
            angle += np.linalg.norm((Rotation.from_quat(pose[3:]) * Rotation.from_quat(previous[3:]).inv()).as_rotvec())
            previous = pose
        self.assertEqual(len(env.commands), 5)
        self.assertEqual(env.camera_reads, 2)  # 初始帧加策略末帧。
        self.assertLessEqual(distance, 0.005 + 1e-9)
        self.assertLessEqual(angle, 0.01 + 1e-9)
        self.assertEqual(info["applied_action"].shape, (6,))
        self.assertFalse(info["is_intervention"])

    def test_relative_frame_keeps_translation_and_rotation_independent(self):
        cls = definition("ur_env/envs/relative_env.py", "RelativeFrame",
                         dict(gym=gym_stub(), np=np, R=Rotation, Env=object))
        pose = np.r_[[0.4, -0.6, 0.2], Rotation.from_rotvec([0, 0, np.pi / 2]).as_quat()]
        wrapper = cls(types.SimpleNamespace(action_space=Box(-np.ones(6), np.ones(6))), include_relative_pose=False)
        wrapper.adjoint_matrix = wrapper._tcp_frame_matrix(pose)
        translated = wrapper.transform_action(np.array([1., 0, 0, 0, 0, 0]))
        np.testing.assert_allclose(translated, [0, 1, 0, 0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(wrapper.transform_action_inv(translated), [1, 0, 0, 0, 0, 0], atol=1e-12)

    def test_relative_rotation_box_works_near_pi(self):
        env = robot_env()
        reference = np.array([np.pi - 0.01, 0, 0])
        env.config.ROTATION_REFERENCE_ROTVEC = reference
        env.config.ROTATION_LIMITS_RAD = np.deg2rad([5, 5, 2])
        desired = Rotation.from_rotvec(reference) * Rotation.from_rotvec(np.deg2rad([10, 0, 0]))
        clipped = env.clip_safety_box(np.r_[[0, 0, .1], desired.as_quat()])
        offset = (Rotation.from_rotvec(reference).inv() * Rotation.from_quat(clipped[3:])).as_rotvec()
        np.testing.assert_allclose(offset, np.deg2rad([5, 0, 0]), atol=1e-12)

    def test_visual_success_holds_and_stops_further_substeps(self):
        namespace = dict(gym=gym_stub(), np=np, KeyBoardIntervention=type("UnusedKeyboard", (), {}))
        interpolation = definition("ur_env/envs/wrappers.py", "ActionInterpolationWrapper", namespace)
        reward = definition("ur_env/envs/wrappers.py", "MultiCameraBinaryRewardClassifierWrapper",
                            dict(gym=gym_stub(), Env=object, time=types.SimpleNamespace(time=lambda: 0, sleep=lambda dt: None)))
        env = robot_env()
        env.holds = 0
        env.hold_position = lambda: setattr(env, "holds", env.holds + 1)
        wrapped = interpolation(reward(env, lambda obs: 0.99, reward_threshold=.95), policy_freq=20, control_freq=100)
        _, rew, done, truncated, _ = wrapped.step(np.ones(6))
        self.assertEqual((rew, done, truncated), (1.0, True, False))
        self.assertEqual(len(env.commands), 1)
        self.assertEqual(env.holds, 1)

    def test_ram_reset_only_resumes_acknowledged_hold_at_fresh_tcp(self):
        robot = robot_env()
        keyboard = types.SimpleNamespace(register=lambda callback: lambda: None)
        cls = definition(
            "examples/experiments/ram_insertion/wrapper.py", "RAMEnv",
            dict(urEnv=type(robot), R=Rotation, np=np, copy=copy,
                 TERMINAL_KEYBOARD_HUB=keyboard, LOGGER=types.SimpleNamespace(info=lambda *args: None),
                 _fmt_pose=lambda *args: "", time=types.SimpleNamespace(sleep=lambda delay: None)),
        )
        env = cls(config=robot.config, fake_env=True)
        env.config.PRECISION_PARAM = {}
        env.config.COMPLIANCE_PARAM = {}
        measured = np.array([0.05, -0.02, 0.01, 0.0, 0.0, 0.0, 1.0])
        calls = []

        def read_state():
            env.currpos = measured.copy()
            calls.append("state")

        def post(route, payload=None):
            calls.append(route)
            return types.SimpleNamespace(raise_for_status=lambda: None)

        env._update_currpos = read_state
        env._post_debug = post
        env._send_pos_command = lambda pose, source: calls.append((source, pose.copy()))
        env.interpolate_move = lambda goal, timeout: calls.append(("move", goal.copy()))

        env.hold_position()
        self.assertEqual(calls, [])
        env._fake_env = False  # Real network behaviour is stubbed from here onward.
        env.hold_position()
        env.go_to_reset()
        self.assertEqual(calls[:2], ["hold", "state"])
        self.assertEqual(calls[2][0], "policy")
        np.testing.assert_array_equal(calls[2][1], measured)
        np.testing.assert_array_equal(calls[5][1], [*measured[:2], env.resetpos[2], *measured[3:]])

        calls.clear()
        env.go_to_reset()
        self.assertEqual(calls[1][0], "reset")

        calls.clear()
        env.hold_position()
        measured[0] = 2.0
        with self.assertRaisesRegex(RuntimeError, "inside the workspace"):
            env.go_to_reset()
        self.assertEqual(calls, ["hold", "state"])
        self.assertTrue(env._held_for_reset)

    def test_applied_action_round_trip_uses_pre_action_tcp_axes_and_mrp(self):
        env = robot_env()
        env.currpos[3:] = Rotation.from_rotvec([0, 0, np.pi / 2]).as_quat()
        relative_cls = definition("ur_env/envs/relative_env.py", "RelativeFrame",
                                  dict(gym=gym_stub(), np=np, R=Rotation, Env=object, copy=copy))
        mrp_cls = definition("ur_env/envs/wrappers.py", "MrpToRotvecActionWrapper",
                             dict(gym=gym_stub(), np=np, R=Rotation, Env=object))
        relative = relative_cls(env, include_relative_pose=False)
        relative.adjoint_matrix = relative._tcp_frame_matrix(env.currpos)
        matrix_before = relative.adjoint_matrix.copy()
        wrapper = mrp_cls(relative, rotation_action_scale=env.action_scale[1])
        pose_before = env.currpos.copy()
        *_, info = wrapper.step(np.ones(6))
        base_action = matrix_before @ wrapper.action(info["applied_action"])
        np.testing.assert_allclose(base_action[:3] * env.action_scale[0], env.currpos[:3] - pose_before[:3], atol=1e-8)
        delta = Rotation.from_quat(env.currpos[3:]) * Rotation.from_quat(pose_before[3:]).inv()
        np.testing.assert_allclose(base_action[3:] * env.action_scale[1], delta.as_rotvec(), atol=1e-8)

    def test_safety_stop_and_outside_workspace_send_no_new_command(self):
        env = robot_env()
        with self.assertRaisesRegex(RuntimeError, "Robot safety stop: TCP orientation is outside"):
            env._parse_state_response({"safety_stop": True, "safety_reason": "TCP orientation is outside"})
        with self.assertRaisesRegex(RuntimeError, "latched"):
            env.step(np.zeros(6))
        self.assertEqual(env.commands, [])
        env = robot_env()
        env.currpos[0] = 2.0
        with self.assertRaisesRegex(RuntimeError, "outside the workspace"):
            env.step(np.zeros(6))
        self.assertEqual(env.commands, [])


if __name__ == "__main__":
    unittest.main()
