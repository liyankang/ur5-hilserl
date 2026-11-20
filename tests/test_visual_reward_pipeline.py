"""视觉奖励数据和验收门槛的离线测试；不导入相机、Gym 或 JAX。"""

import ast
import json
import datetime
import glob
import os
from pathlib import Path
import random
import tempfile
import time
import types
import unittest
import uuid

import numpy as np

from examples.classifier_data import (
    sample_group, split_grouped_examples, validation_metrics, checkpoint_fingerprint,
    validate_classifier_artifact,
)
from examples.collection_utils import make_transition, classifier_sample, save_pickle


ROOT = Path(__file__).resolve().parents[1]


class VisualRewardPipelineTests(unittest.TestCase):
    def test_episode_and_legacy_batch_groups_do_not_depend_on_class_file(self):
        sample = {"infos": {"episode_id": "episode-1"}}
        self.assertEqual(sample_group(sample, "positive.pkl"), sample_group(sample, "negative.pkl"))
        self.assertEqual(
            sample_group({}, "ram_insertion_20_success_images_2026-11-17_10-00-00.pkl"),
            sample_group({}, "ram_insertion_failure_images_2026-11-17_10-00-00.pkl"),
        )

    def test_split_keeps_complete_episodes_and_both_labels_in_each_partition(self):
        examples = [dict(validation_group=str(episode), labels=label, value=index)
                    for episode in range(8) for label in (0, 1) for index in range(3)]
        train, validation = split_grouped_examples(examples, 0.25, random.Random(1))
        self.assertFalse({e["validation_group"] for e in train} & {e["validation_group"] for e in validation})
        self.assertEqual({e["labels"] for e in train}, {0, 1})
        self.assertEqual({e["labels"] for e in validation}, {0, 1})
        self.assertEqual(len(train) + len(validation), len(examples))

    def test_old_single_capture_cannot_claim_held_out_accuracy(self):
        examples = [dict(validation_group="only-capture", labels=label) for label in (0, 1)]
        with self.assertRaisesRegex(ValueError, "independent"):
            split_grouped_examples(examples, 0.2, random.Random(0))
        for ratio in (0, 1, -0.1):
            with self.assertRaises(ValueError):
                split_grouped_examples(examples, ratio, random.Random(0))

    def test_metrics_use_deployment_threshold_and_reject_imbalanced_false_success(self):
        metrics = validation_metrics([1, 1, 0, 0], [0.96, 0.94, 0.94, 0.10], 0.95)
        self.assertEqual(metrics["positive_accuracy"], 0.5)
        self.assertEqual(metrics["negative_accuracy"], 1)
        self.assertEqual(metrics["validation_accuracy"], 0.75)
        self.assertFalse(metrics["passed"])
        self.assertTrue(validation_metrics([1, 0], [0.99, 0.01], 0.95)["passed"])
        with self.assertRaises(ValueError):
            validation_metrics([1, 0], [float("nan"), 0], 0.95)

    def test_artifact_gate_ties_accuracy_to_latest_real_weights_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            checkpoint = path / "checkpoint_150"
            checkpoint.write_bytes(b"saved classifier parameters")
            report = dict(validation_metrics([1, 0], [0.99, 0.01], 0.95),
                          exp_name="ram_insertion", image_keys=["wrist", "global_2"],
                          use_proprio=False, threshold=0.95, checkpoint_step=150,
                          checkpoint_sha256=checkpoint_fingerprint(checkpoint),
                          split_method="grouped_by_episode_or_capture")
            report_path = path / "validation.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            args = (directory, "ram_insertion", ["wrist", "global_2"], False, 0.95)
            self.assertTrue(validate_classifier_artifact(*args)["passed"])
            for field, invalid in (("validation_accuracy", 0.95), ("positive_accuracy", 0.95),
                                   ("negative_accuracy", 0.95), ("passed", False),
                                   ("use_proprio", True), ("threshold", 0.5),
                                   ("checkpoint_step", 149), ("split_method", "random_frames")):
                with self.subTest(field=field):
                    report_path.write_text(json.dumps(dict(report, **{field: invalid})), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        validate_classifier_artifact(*args)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            checkpoint.write_bytes(b"different unvalidated weights")
            with self.assertRaisesRegex(ValueError, "weights changed"):
                validate_classifier_artifact(*args)
            (path / "checkpoint_151").write_bytes(b"newer model")
            with self.assertRaisesRegex(ValueError, "latest"):
                validate_classifier_artifact(*args)

    def test_collection_requires_explicit_labels_and_consumes_requests_once(self):
        tree = ast.parse((ROOT / "examples/record_success_fail.py").read_text(encoding="utf-8"))
        consume = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_consume_request")
        namespace = {"_requests": {"success": False, "failure": False, "reset": False}}
        exec(compile(ast.Module(body=[consume], type_ignores=[]), "collector", "exec"), namespace)
        self.assertIsNone(namespace["_consume_request"]())
        namespace["_requests"]["failure"] = True
        self.assertEqual(namespace["_consume_request"](), "failure")
        self.assertIsNone(namespace["_consume_request"]())
        namespace["_requests"].update(success=True, failure=True)
        self.assertEqual(namespace["_consume_request"](), "success")
        namespace["_requests"].update(success=True, reset=True)
        self.assertEqual(namespace["_consume_request"](), "reset")

    def test_visual_only_input_keeps_camera_shape_and_excludes_robot_state(self):
        tree = ast.parse((ROOT / "examples/test_classifier.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_images_and_optional_state")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "classifier-input", "exec"), namespace)
        image = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        obs = {"state": np.zeros((1, 19)), "images": {"wrist_2": image}}
        result = namespace["_images_and_optional_state"](obs, ["wrist"], False)
        self.assertEqual(set(result), {"wrist"})
        self.assertEqual(result["wrist"].shape, (1, 128, 128, 3))

    def test_collector_drops_unlabeled_frames_and_holds_after_positive(self):
        # 执行真实采集主循环，用三个假控制步检查：未标注、负样本、正样本。
        class Progress:
            def __init__(self, **kwargs):
                pass
            def update(self, amount):
                pass
            def close(self):
                pass
            def set_postfix(self, **kwargs):
                pass

        class Env:
            action_space = types.SimpleNamespace(shape=(6,))
            def __init__(self):
                self.unwrapped = self
                self.steps = self.resets = self.holds = 0
                self.closed = False
            def reset(self):
                self.resets += 1
                return {"state": 0}, {}
            def step(self, action):
                self.steps += 1
                if self.steps == 2:
                    namespace["_requests"]["failure"] = True
                elif self.steps == 3:
                    namespace["_requests"]["success"] = True
                # applied_action 必须优先于原有 intervene_action。
                return {"state": self.steps}, 0, False, False, {
                    "applied_action": np.ones(6) * 0.1,
                    "intervene_action": np.ones(6) * 0.2,
                }
            def hold_position(self):
                self.holds += 1
            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as directory:
            env = Env()
            config = types.SimpleNamespace(classifier_data_dir=directory,
                                           get_environment=lambda **kwargs: env)
            namespace = dict(
                FLAGS=types.SimpleNamespace(auto=False, successes_needed=1, resume=False, exp_name="ram_insertion"),
                CONFIG_MAPPING={"ram_insertion": lambda: config},
                TERMINAL_KEYBOARD_HUB=types.SimpleNamespace(register=lambda handler: lambda: None),
                _requests={"success": False, "failure": False, "reset": False}, _key_times={},
                _kb_handler=lambda key: None, np=np, Path=Path, glob=glob, os=os,
                pkl=__import__("pickle"), tqdm=Progress, time=time, uuid=uuid,
                datetime=datetime, make_transition=make_transition,
                classifier_sample=classifier_sample, save_pickle=save_pickle,
            )
            tree = ast.parse((ROOT / "examples/record_success_fail.py").read_text(encoding="utf-8"))
            definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                           n.name in {"main", "_consume_request", "_get_manual_action"}]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), "collector", "exec"), namespace)
            namespace["main"](None)
            self.assertEqual((env.steps, env.resets, env.holds), (3, 1, 1))
            self.assertTrue(env.closed)
            import pickle
            positive = pickle.loads(next(Path(directory).glob("*success*.pkl")).read_bytes())
            negative = pickle.loads(next(Path(directory).glob("*failure*.pkl")).read_bytes())
            self.assertEqual((len(positive), len(negative)), (1, 1))
            self.assertEqual(positive[0]["episode_id"], negative[0]["episode_id"])
            self.assertEqual(positive[0]["next_observations"]["state"], 3)
            self.assertEqual(negative[0]["next_observations"]["state"], 2)
            np.testing.assert_allclose(positive[0]["actions"], np.ones(6) * 0.1)


if __name__ == "__main__":
    unittest.main()
