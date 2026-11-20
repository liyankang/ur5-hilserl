"""Offline collector regressions, without camera/robot/JAX dependencies."""
import ast
import copy
import pickle
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
from examples.collection_utils import (
    make_transition, label_success, classifier_sample, save_pickle,
)

ROOT = Path(__file__).resolve().parents[1]


def load_definition(path, name, namespace):
    # Isolate the real definition from optional hardware imports at module scope.
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


class CollectionTests(unittest.TestCase):
    def test_timeout_preserves_bootstrap_and_copies_observation(self):
        obs = {"state": [0]}
        transition = make_transition(obs, [1], {"state": [1]}, 0, False, True, {})
        obs["state"][0] = 9
        self.assertEqual(transition["observations"]["state"], [0])
        self.assertTrue(transition["dones"])
        self.assertEqual(transition["masks"], 1)

    def test_success_and_intervention_labels(self):
        transition = make_transition(0, [0], 1, 0, False, False, {"intervene_action": [2]})
        label_success(transition)
        self.assertEqual(transition["actions"], [2])
        self.assertEqual((transition["rewards"], transition["masks"], transition["dones"]), (1, 0, True))
        self.assertTrue(transition["infos"]["succeed"])

    def test_classifier_frames_are_independent_labeled_examples(self):
        original = make_transition(0, [0], 1, 0, False, False, {})
        for success in (False, True):
            sample = classifier_sample(original, success)
            self.assertTrue(sample["dones"])
            self.assertEqual(sample["labels"], int(success))
        self.assertFalse(original["dones"])

    def test_atomic_snapshot_replaces_resume_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.pkl"
            save_pickle(path, [1])
            save_pickle(path, [1, 2])
            self.assertEqual(pickle.loads(path.read_bytes()), [1, 2])
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_auto_temporal_alignment_timeout_and_speed(self):
        class Env:
            def __init__(self):
                self.unwrapped = self
                self.action_scale = [0.01, 0.01]
                self.index = 0

            def step(self, action):
                self.index += 1
                return {"state": self.index}, 0, False, self.index == 2, {"succeed": False}

        speeds = []
        def controller(base, target, speed):
            speeds.append(speed)
            return np.array([speed])
        move = load_definition("examples/auto_trajectory.py", "auto_move_to_target", dict(
            np=np, copy=copy, make_transition=make_transition, compute_auto_step_action=controller))
        transitions, interrupted = move(Env(), None, initial_obs={"state": 0})
        self.assertFalse(interrupted)
        self.assertEqual([(t["observations"]["state"], t["next_observations"]["state"])
                          for t in transitions], [(0, 1), (1, 2)])
        self.assertEqual(speeds, [0.2, 0.2])
        self.assertEqual(transitions[-1]["masks"], 1)
        self.assertTrue(transitions[-1]["dones"])
        self.assertEqual(move(Env(), None, initial_obs={}, interrupt_check=lambda: True), ([], True))
        with self.assertRaises(ValueError):
            move(Env(), None, speed_mm=0, initial_obs={})

    def test_classifier_accepts_both_image_layouts(self):
        normalize = load_definition("examples/train_reward_classifier.py", "_normalize_classifier_obs", {})
        for obs in ({"state": 0, "wrist": 1}, {"state": 0, "images": {"wrist": 1}}):
            self.assertEqual(normalize(obs, ["wrist"]), {"state": 0, "wrist": 1})

    def test_interpolation_aggregates_executed_intervention(self):
        class Wrapper:
            def __init__(self, env):
                self.env, self.unwrapped = env, env
        class Keyboard:
            pass
        class Env(Keyboard):
            action_space = types.SimpleNamespace(shape=(6,))
            def step(self, action):
                executed = np.ones(6) * self._interpolation_weight
                return 0, 0, False, False, {"intervene_action": executed}
        cls = load_definition("ur_env/envs/wrappers.py", "ActionInterpolationWrapper", dict(
            gym=types.SimpleNamespace(Wrapper=Wrapper), np=np, KeyBoardIntervention=Keyboard))
        env = Env()
        wrapper = cls(env, policy_freq=20, control_freq=100)
        *_, info = wrapper.step(np.zeros(6))
        np.testing.assert_allclose(info["intervene_action"], np.ones(6))
        self.assertEqual(env._interpolation_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
