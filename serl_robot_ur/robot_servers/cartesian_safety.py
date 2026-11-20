"""UR5 笛卡尔监督停止保护，可选监测力反馈；不实现阻抗或力模式控制。"""
import threading

import numpy as np


class SafetyViolation(RuntimeError):
    """安全域、力反馈或安全锁存不允许继续运动。"""


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return array.copy()


def _quaternion(rotvec):
    angle = np.linalg.norm(rotvec)
    if not np.isfinite(angle):
        raise ValueError("Rotation-vector norm must be finite")
    if angle < 1e-14:
        return np.array([0., 0., 0., 1.])
    return np.r_[rotvec / angle * np.sin(angle / 2), np.cos(angle / 2)]


def _relative_rotvec(reference, actual):
    # 基坐标轴相对朝向 q_actual * q_ref^-1，与 reset 左乘扰动和客户端限幅一致。
    ref, pose = _quaternion(reference), _quaternion(actual)
    a, b = pose[:3], -ref[:3]
    q = np.r_[pose[3] * b + ref[3] * a + np.cross(a, b), pose[3] * ref[3] - a @ b]
    if q[3] < 0:
        q = -q
    length = np.linalg.norm(q[:3])
    return np.zeros(3) if length < 1e-14 else q[:3] / length * (2 * np.arctan2(length, q[3]))


class CartesianSafetyGuard:
    """保存标定参数并锁存违规；清锁存必须显式确认且重新检查原始读数。"""

    def __init__(self, config, control_hz):
        if config.get("enabled") is not True or config.get("latch_on_violation") is not True:
            raise ValueError("Safety configuration requires enabled=true and latch_on_violation=true")
        requested_hz = float(config["control_hz"])
        if not np.isfinite(requested_hz) or requested_hz <= 0 or not np.isclose(requested_hz, control_hz, rtol=0, atol=1e-9):
            raise ValueError("Safety control_hz must equal the running robot server control_hz")
        self.workspace_low = _vector(config["workspace_low"], 3, "workspace_low")
        self.workspace_high = _vector(config["workspace_high"], 3, "workspace_high")
        if np.any(self.workspace_low >= self.workspace_high):
            raise ValueError("Each workspace_low must be strictly less than workspace_high")
        self.rotation_reference = _vector(config["rotation_reference_rotvec"], 3, "rotation_reference_rotvec")
        self.rotation_limits = _vector(config["rotation_limits_rad"], 3, "rotation_limits_rad")
        if np.any(self.rotation_limits < 0) or np.any(self.rotation_limits > np.pi):
            raise ValueError("rotation_limits_rad must lie in [0, pi]")
        # UR5 无力控仍可使用位置监督。只有经现场验证的力反馈才开启额外监测。
        self.wrench_monitor_enabled = config.get("wrench_monitor_enabled", False)
        if not isinstance(self.wrench_monitor_enabled, bool):
            raise ValueError("wrench_monitor_enabled must be a boolean")
        self.wrench_bias = _vector(config.get("wrench_bias", [0.0] * 6), 6, "wrench_bias")
        self.force_limit = None
        self.torque_limit = None
        if self.wrench_monitor_enabled:
            self.force_limit = self._positive(config.get("force_norm_limit_n"), "force_norm_limit_n")
            self.torque_limit = self._positive(config.get("torque_norm_limit_nm"), "torque_norm_limit_nm")
        self.control_hz = requested_hz
        self._lock = threading.RLock()
        self.latched = False
        self.reason = None

    @staticmethod
    def _positive(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive finite number")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive finite number") from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return value

    def validate_pose(self, pose):
        pose = _vector(pose, 6, "raw_tcp_pose")
        if np.any(pose[:3] < self.workspace_low - 1e-9) or np.any(pose[:3] > self.workspace_high + 1e-9):
            raise SafetyViolation("TCP position is outside the configured Cartesian workspace")
        offset = _relative_rotvec(self.rotation_reference, pose[3:])
        if np.any(np.abs(offset) > self.rotation_limits + 1e-9):
            raise SafetyViolation("TCP orientation is outside the relative rotation workspace")
        return pose

    def validate_readings(self, pose, wrench=None):
        self.validate_pose(pose)
        if not self.wrench_monitor_enabled:
            return
        corrected = _vector(wrench, 6, "raw_tcp_wrench") - self.wrench_bias
        if np.linalg.norm(corrected[:3]) > self.force_limit:
            raise SafetyViolation("TCP force norm exceeds force_norm_limit_n")
        if np.linalg.norm(corrected[3:]) > self.torque_limit:
            raise SafetyViolation("TCP torque norm exceeds torque_norm_limit_nm")

    def latch(self, reason):
        with self._lock:
            self.latched = True
            if self.reason is None:
                self.reason = str(reason)

    def observe(self, pose, wrench=None):
        with self._lock:
            if self.latched:
                raise SafetyViolation(f"Safety stop is latched: {self.reason}")
            try:
                self.validate_readings(pose, wrench)
            except (ValueError, SafetyViolation) as exc:
                self.latch(exc)
                raise SafetyViolation(str(exc)) from exc

    def validate_target(self, pose):
        with self._lock:
            if self.latched:
                raise SafetyViolation(f"Safety stop is latched: {self.reason}")
            try:
                return self.validate_pose(pose)
            except (ValueError, SafetyViolation) as exc:
                self.latch(exc)
                raise SafetyViolation(str(exc)) from exc

    def acknowledge(self, pose, wrench=None):
        with self._lock:
            # 先检查再解锁；仍越界或（可选的）超力时，重复确认也不恢复运动。
            self.validate_readings(pose, wrench)
            self.latched = False
            self.reason = None

    def status(self):
        with self._lock:
            return {"enabled": True, "wrench_monitor_enabled": self.wrench_monitor_enabled,
                    "latched": self.latched, "reason": self.reason,
                    "control_hz": self.control_hz}
