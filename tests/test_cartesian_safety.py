"""用真实监督器与控制方法验证停推逻辑，不导入 Flask 或连接机械臂。"""
import ast
import copy
import logging
from pathlib import Path
import threading
import time
import types
import unittest
from typing import Any, Callable, Optional

import numpy as np

from serl_robot_ur.robot_servers.cartesian_safety import (
    CartesianSafetyGuard, SafetyViolation, _relative_rotvec, _quaternion,
)


ROOT = Path(__file__).resolve().parents[1]


def load_controller():
    # 抽取实际的坐标转换函数与控制类；替代的只是 RTDE I/O，不是安全实现。
    path = ROOT / "serl_robot_ur/robot_servers/ur_server.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    wanted = {"quat_to_rotvec", "rotvec_to_quat", "ensure_quat_pose", "quat_pose_to_rotvec_pose",
              "_call_optional", "_optional_or_default", "_format_vector", "URStreamingController"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in wanted]
    namespace = dict(np=np, logging=logging, threading=threading, time=time, Any=Any,
                     Optional=Optional, Callable=Callable, CartesianSafetyGuard=CartesianSafetyGuard,
                     SafetyViolation=SafetyViolation, LOGGER_NAME="test.safety")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["URStreamingController"]


Controller = load_controller()


def safety_config(wrench=False):
    # 测试阈值是人为构造的断言输入，不是推荐用于真实 USB 插接的力阈值。
    return dict(enabled=True, latch_on_violation=True, control_hz=50,
                workspace_low=[0.4, -0.1, 0.1], workspace_high=[0.6, 0.1, 0.3],
                rotation_reference_rotvec=[0, 0, 0], rotation_limits_rad=[0.1, 0.1, 0.1],
                wrench_monitor_enabled=wrench, wrench_bias=[0]*6,
                force_norm_limit_n=5 if wrench else None,
                torque_norm_limit_nm=0.5 if wrench else None)


class FakeReceive:
    def __init__(self):
        self.pose = np.array([0.5, 0, 0.2, 0, 0, 0.])
        self.wrench = np.zeros(6)
        self.force_reads = 0
        self.forbid_force = False
        self.fail_pose = False

    def getActualTCPPose(self):
        if self.fail_pose:
            raise ConnectionError("simulated RTDE read failure")
        return self.pose.copy()

    def getActualTCPForce(self):
        self.force_reads += 1
        if self.forbid_force:
            raise AssertionError("Position-only mode must not read a force channel")
        return self.wrench.copy()

    def getActualTCPSpeed(self):
        return np.zeros(6)

    def getActualQ(self):
        return np.zeros(6)

    def getActualQd(self):
        return np.zeros(6)


class FakeControl:
    def __init__(self):
        self.servo_calls = []
        self.stop_calls = []
        self.joint_calls = []
        self.reuploads = 0
        self.servo_result = True

    def initPeriod(self):
        return 0

    def servoL(self, *args):
        self.servo_calls.append(args)
        return self.servo_result

    def waitPeriod(self, period):
        pass

    def servoStop(self):
        self.stop_calls.append("servoStop")

    def stopL(self):
        self.stop_calls.append("stopL")

    def speedStop(self):
        self.stop_calls.append("speedStop")

    def moveJ(self, *args):
        self.joint_calls.append(args)
        return True

    def reuploadScript(self):
        self.reuploads += 1
        return True


def controller():
    # 跳过会创建真实控制线程的构造函数，显式给真实方法准备全部状态。
    result = Controller.__new__(Controller)
    result.rtde_r, result.rtde_c = FakeReceive(), FakeControl()
    result.control_hz = 50
    result.default_speed = result.target_speed = 0.05
    result.default_accel = result.target_accel = 0.05
    result._safety_guard = None
    result._safety_stop_sent = False
    result._servo_paused = False
    result.start_paused_until_pose = False
    result.disable_motion = False
    result._motion_lock, result._lock = threading.RLock(), threading.Lock()
    result._stop_event = threading.Event()
    result.servo_gain, result.lookahead_time = 300, 0.1
    result.target_pose = np.r_[result.rtde_r.pose[:3], [0, 0, 0, 1]]
    result.last_state = {"pose": result.target_pose.copy()}
    result.logger = logging.getLogger("test.safety")
    result.logger.addHandler(logging.NullHandler())
    result.reset_joint_target = np.zeros(6)
    result._filtered_pose_rtde = result._filtered_vel_rtde = result._filtered_force_rtde = None
    result.pose_filter_alpha, result.vel_filter_alpha, result.force_filter_alpha = 0.2, 0.1, 0.1
    return result


class CartesianSafetyTests(unittest.TestCase):
    def test_position_only_mode_does_not_require_or_read_wrench(self):
        robot = controller()
        robot.rtde_r.forbid_force = True
        status = robot.configure_safety(safety_config())
        self.assertFalse(status["wrench_monitor_enabled"])
        self.assertTrue(robot._servo_paused)
        robot.set_target_pose([0.51, 0, 0.2, 0, 0, 0])
        self.assertTrue(robot._servo_cycle(0.02))
        state = robot.get_state()
        self.assertFalse(state["safety_stop"])
        self.assertFalse(state["wrench_monitor_enabled"])
        np.testing.assert_array_equal(state["force"], np.zeros(3))
        robot.hold_position()
        robot.reset_safety(acknowledge=True)
        self.assertEqual(robot.rtde_r.force_reads, 0)

    def test_control_feedback_uses_raw_pose_without_ema_lag(self):
        robot = controller()
        robot.configure_safety(safety_config())
        robot.get_state()
        robot.rtde_r.pose[0] = 0.52
        state = robot.get_state()
        self.assertEqual(state["pose"][0], 0.52)
        self.assertAlmostEqual(state["pose_filtered"][0], 0.504)

    def test_disabling_optional_wrench_does_not_return_stale_filtered_force(self):
        robot = controller()
        robot.configure_safety(safety_config(True))
        robot.rtde_r.wrench[0] = 3
        state = robot.get_state()
        self.assertEqual(state["force"][0], 3)
        robot.configure_safety(safety_config(False))
        robot.rtde_r.forbid_force = True
        state = robot.get_state()
        np.testing.assert_array_equal(state["force"], np.zeros(3))

    def test_monitoring_requires_valid_limits_and_control_rate(self):
        for key, value in (("force_norm_limit_n", None), ("torque_norm_limit_nm", 0),
                           ("force_norm_limit_n", float("nan")), ("control_hz", 100),
                           ("wrench_monitor_enabled", "false"), ("latch_on_violation", False)):
            config = safety_config(True)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                CartesianSafetyGuard(config, 50)

    def test_invalid_targets_latch_and_do_not_reach_servo(self):
        for pose in ([0.7, 0, 0.2, 0, 0, 0], [0.5, 0, 0.2, 0.2, 0, 0],
                     [float("nan"), 0, 0.2, 0, 0, 0], [0, 1],
                     [0.5, 0, 0.2, 0, 0, 0, 0]):
            robot = controller()
            robot.configure_safety(safety_config())
            with self.subTest(pose=pose), self.assertRaises((ValueError, SafetyViolation)):
                robot.set_target_pose(pose)
            self.assertTrue(robot._safety_guard.latched)
            self.assertFalse(robot._servo_cycle(0.02))
            self.assertFalse(robot.rtde_c.servo_calls)

    def test_background_detects_force_torque_and_missing_data_before_motion(self):
        for wrench in ([6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0.6, 0], [float("nan")]*6):
            robot = controller()
            robot.configure_safety(safety_config(True))
            robot.set_target_pose([0.51, 0, 0.2, 0, 0, 0])
            robot.rtde_c.stop_calls.clear()
            robot.rtde_r.wrench = np.array(wrench)
            self.assertFalse(robot._servo_cycle(0.02))
            self.assertTrue(robot._safety_guard.latched)
            self.assertIn("servoStop", robot.rtde_c.stop_calls)
            self.assertFalse(robot.rtde_c.servo_calls)
            # 反馈恢复正常也不能自动解除，后续周期继续不发送运动指令。
            robot.rtde_r.wrench = np.zeros(6)
            self.assertFalse(robot._servo_cycle(0.02))
            self.assertFalse(robot.rtde_c.servo_calls)

    def test_background_position_or_connection_failure_latches_without_wrench(self):
        for failure in ("outside", "nan", "connection"):
            robot = controller()
            robot.configure_safety(safety_config())
            robot.set_target_pose([0.51, 0, 0.2, 0, 0, 0])
            if failure == "outside":
                robot.rtde_r.pose[0] = 0.8
            elif failure == "nan":
                robot.rtde_r.pose[0] = float("nan")
            else:
                robot.rtde_r.fail_pose = True
            self.assertFalse(robot._servo_cycle(0.02))
            self.assertTrue(robot._safety_guard.latched)
            self.assertFalse(robot.rtde_c.servo_calls)

    def test_clearerr_reconfigure_and_hold_cannot_acknowledge_stop(self):
        robot = controller()
        config = safety_config(True)
        robot.configure_safety(config)
        robot.rtde_r.wrench[0] = 6
        self.assertFalse(robot._servo_cycle(0.02))
        robot.rtde_r.wrench[0] = 0
        with self.assertRaises(SafetyViolation):
            robot.clear_error()
        with self.assertRaises(SafetyViolation):
            robot.configure_safety(config)
        robot.hold_position()
        self.assertTrue(robot._safety_guard.latched)
        self.assertEqual(robot.rtde_c.reuploads, 0)

    def test_reset_requires_ack_and_safe_feedback_then_waits_for_new_target(self):
        robot = controller()
        robot.configure_safety(safety_config(True))
        robot.rtde_r.wrench[0] = 6
        robot._servo_cycle(0.02)
        with self.assertRaises(ValueError):
            robot.reset_safety()
        with self.assertRaises(SafetyViolation):
            robot.reset_safety(acknowledge=True)
        self.assertTrue(robot._safety_guard.latched)
        robot.rtde_r.wrench[0] = 0
        robot.reset_safety(acknowledge=True)
        self.assertFalse(robot._safety_guard.latched)
        self.assertFalse(robot._servo_cycle(0.02))
        robot.set_target_pose([0.51, 0, 0.2, 0, 0, 0])
        self.assertTrue(robot._servo_cycle(0.02))

    def test_hold_stops_old_target_and_uses_raw_actual_pose(self):
        robot = controller()
        robot.configure_safety(safety_config())
        robot.set_target_pose([0.55, 0, 0.2, 0, 0, 0])
        robot.rtde_r.pose[0] = 0.52
        robot.last_state["pose"][0] = 0.49
        robot.rtde_c.stop_calls.clear()
        target = robot.hold_position()
        self.assertEqual(target[0], 0.52)
        self.assertTrue(robot._servo_paused)
        self.assertIn("servoStop", robot.rtde_c.stop_calls)
        self.assertFalse(robot._servo_cycle(0.02))

    def test_stop_failure_does_not_skip_other_available_stop_interfaces(self):
        robot = controller()
        robot.configure_safety(safety_config())
        robot.rtde_c.stop_calls.clear()
        def failed_stop():
            raise ConnectionError("first interface failed")
        robot.rtde_c.servoStop = failed_stop
        robot.hold_position()
        self.assertIn("stopL", robot.rtde_c.stop_calls)
        self.assertIn("speedStop", robot.rtde_c.stop_calls)
        self.assertTrue(robot._servo_paused)

    def test_all_stop_interfaces_fail_keeps_latch_and_raises(self):
        robot = controller()
        robot.configure_safety(safety_config())
        def failed_stop():
            raise ConnectionError("controller unavailable")
        robot.rtde_c.servoStop = robot.rtde_c.stopL = robot.rtde_c.speedStop = failed_stop
        with self.assertRaises(SafetyViolation):
            robot._trip_safety("test limit exceeded")
        self.assertTrue(robot._safety_guard.latched)
        self.assertTrue(robot._servo_paused)

    def test_guarded_joint_motion_is_rejected_and_old_unguarded_mode_remains(self):
        robot = controller()
        robot.configure_safety(safety_config())
        with self.assertRaises(SafetyViolation):
            robot.move_joints([0]*6)
        with self.assertRaises(SafetyViolation):
            robot.joint_reset()
        self.assertFalse(robot.rtde_c.joint_calls)
        legacy = controller()
        legacy.move_joints([0]*6)
        self.assertEqual(len(legacy.rtde_c.joint_calls), 1)

    def test_relative_rotation_uses_base_axes_and_rejects_overflow(self):
        reference = np.array([2.9348, -1.0839, 0.0372])
        delta = np.array([0.02, -0.03, 0.0])
        # 用独立四元数乘法构造 R_delta * R_ref，再转回旋转向量。
        a, b = _quaternion(delta), _quaternion(reference)
        q = np.r_[a[3]*b[:3] + b[3]*a[:3] + np.cross(a[:3], b[:3]), a[3]*b[3] - a[:3]@b[:3]]
        if q[3] < 0:
            q = -q
        norm = np.linalg.norm(q[:3])
        actual = q[:3]/norm * (2*np.arctan2(norm, q[3]))
        np.testing.assert_allclose(_relative_rotvec(reference, actual), delta, atol=1e-12)
        config = safety_config()
        config["rotation_reference_rotvec"] = [1e308]*3
        guard = CartesianSafetyGuard(config, 50)
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises((ValueError, SafetyViolation)):
            guard.observe([0.5, 0, 0.2, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
