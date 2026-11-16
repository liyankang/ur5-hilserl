"""Offline checks for learned reward activation; no JAX or hardware required."""
import ast
import os
from pathlib import Path
import time
import types
import unittest
from typing import Callable, Dict, List

import numpy as np
from tests.test_collection import load_definition, ROOT


class ClassifierRewardTests(unittest.TestCase):
    def test_missing_checkpoint_does_not_initialize_random_classifier(self):
        def unexpected(*args, **kwargs):
            self.fail("Must reject missing checkpoint before network initialization")
        load = load_definition("serl_launcher/networks/reward_classifier.py", "load_classifier_func", dict(
            jnp=np, Dict=Dict, List=List, Callable=Callable, os=os,
            checkpoints=types.SimpleNamespace(latest_checkpoint=lambda path: None),
            create_classifier=unexpected))
        with self.assertRaisesRegex(FileNotFoundError, "train_reward_classifier"):
            load(None, {}, [], "missing")

    def test_visual_success_and_timeout_flags(self):
        class Wrapper:
            def __init__(self, env):
                self.env = env
        class Env:
            def step(self, action):
                return {}, 0, False, True, {}
        cls = load_definition("ur_env/envs/wrappers.py", "MultiCameraBinaryRewardClassifierWrapper", dict(
            gym=types.SimpleNamespace(Wrapper=Wrapper), Env=Env, time=time))
        for probability, expected in [(0.99, (1.0, True, False)), (0.1, (0.0, False, True))]:
            env = cls(Env(), lambda obs: probability, reward_threshold=0.95)
            _, reward, done, truncated, info = env.step(None)
            self.assertEqual((reward, done, truncated), expected)
            self.assertEqual(info["succeed"], bool(reward))

    def test_pose_reward_can_be_disabled(self):
        path = "ur_env/envs/ur_env.py"
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if getattr(n, "name", None) == "urEnv")
        method = next(n for n in cls.body if getattr(n, "name", None) == "compute_reward")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), path, "exec"), namespace)
        self.assertFalse(namespace["compute_reward"](types.SimpleNamespace(use_pose_reward=False), {}))


if __name__ == "__main__":
    unittest.main()
