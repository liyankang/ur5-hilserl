"""用模拟状态和相机反馈验证跨回合目标更新与失效处理。"""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from examples.experiments.ram_insertion.port_approach import PortApproach


class SimulatedEnvironment:
    def __init__(self):
        self.config = SimpleNamespace(TARGET_POSE=np.array([0.5, 0, 0.2, 0, 0, 0]),
                                      RESET_POSE=np.array([0.5, 0, 0.3, 0, 0, 0]), PRECISION_PARAM={})
        self._TARGET_POSE = self.config.TARGET_POSE.copy()
        self.xyz_bounding_box = SimpleNamespace(low=np.array([0.4, -0.1, 0.19]),
                                                high=np.array([0.6, 0.1, 0.4]))
        self.camera_config = {"global_2": {}, "wrist": {}}
        self.cap = {name: SimpleNamespace(read=lambda: None) for name in self.camera_config}
        self.pose = np.array([0.5, 0, 0.3, 0, 0, 0, 1])
        self.events = []
        self.held = False

    def _post_debug(self, route, payload=None, timeout=None):
        response = Mock()
        if route == "getstate":
            response.json.return_value = {"pose": self.pose.tolist(), "vel": [0] * 6}
        else:
            self.events.append((route, payload))
            if route == "hold":
                self.held = True
            if route == "pose":
                self.held = False
                self.pose = np.array(payload["arr"])
        return response

    def _pose6_to_quat(self, pose):
        return np.r_[pose[:3], [0, 0, 0, 1]]


class PortApproachTests(unittest.TestCase):
    def setUp(self):
        self.env = SimulatedEnvironment()
        self.coarse_xy = np.array([0.519, 0.01])
        self.actual_port_xy = np.array([0.5223, 0.01])
        vision = SimpleNamespace(fixed_camera="global_2", wrist_camera="wrist", approach_z_m=0.3,
                                 estimate_port_xy=lambda frame: SimpleNamespace(xy_m=self.coarse_xy),
                                 wrist_correction=lambda frame: SimpleNamespace(xy_m=self.actual_port_xy - self.env.pose[:2]))
        self.approach = PortApproach(self.env, vision)

        def confirm_fixture():
            self.assertTrue(self.env.held)
            self.assertGreaterEqual(self.env.pose[2], 0.3 - 0.0008)
            self.env.events.append(("ready", self.env.pose.copy()))

        self.approach._wait_for_fixture = confirm_fixture
        self.sleep = patch("examples.experiments.ram_insertion.port_approach.time.sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def test_aligns_both_positions_and_retracts_before_second_placement(self):
        first = self.approach.align()
        self.assertGreater(first["port_alignment_iterations"], 0)
        np.testing.assert_allclose(self.env._TARGET_POSE[:2], self.actual_port_xy)
        self.assertTrue(self.env.held)
        np.testing.assert_array_equal(self.env.config.TARGET_POSE, [0.5, 0, 0.2, 0, 0, 0])

        # 模拟完成第一轮插入后的位置。第二次确认前只允许垂直退出。
        self.env.pose[:3] = self.env._TARGET_POSE[:3]
        old_xy = self.env.pose[:2].copy()
        self.env.events.clear()
        self.coarse_xy = np.array([0.48, -0.021])
        self.actual_port_xy = np.array([0.478, -0.021])
        self.approach.align()
        before_ready = next(i for i, (route, _) in enumerate(self.env.events) if route == "ready")
        for route, payload in self.env.events[:before_ready]:
            if route == "pose":
                np.testing.assert_allclose(payload["arr"][:2], old_xy)
        np.testing.assert_allclose(self.env._TARGET_POSE[:2], self.actual_port_xy)
        self.assertEqual(self.env._TARGET_POSE[2], 0.2)
        self.assertEqual(self.env.resetpos[2], 0.3)

    def test_rejects_coarse_target_outside_workspace_before_motion(self):
        self.coarse_xy = np.array([0.611, 0])
        with self.assertRaisesRegex(RuntimeError, "outside RAM workspace"):
            self.approach.align()
        self.assertNotIn("pose", [route for route, _ in self.env.events])
        self.assertTrue(self.env.held)

    def test_bad_wrist_frame_leaves_coarse_approach_held_and_old_target_intact(self):
        self.approach.vision.wrist_correction = Mock(side_effect=ValueError("port not detected"))
        with self.assertRaisesRegex(ValueError, "port not detected"):
            self.approach.align()
        self.assertEqual(self.env.events[-1][0], "hold")
        np.testing.assert_array_equal(self.env._TARGET_POSE, self.env.config.TARGET_POSE)
        self.assertFalse(self.approach.previous_round_aligned)
