"""
UR5 control server using RTDE + Flask.

The HTTP surface intentionally stays close to the old Franka server so the
existing Gym environments can keep using `requests.post(...)` without
robot-specific branching.
"""

from __future__ import annotations

import argparse
import atexit
import logging
import threading
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from flask import Flask, g, jsonify, request
from scipy.spatial.transform import Rotation as R

try:
    import rtde_control
    import rtde_receive
except Exception as exc:  # pragma: no cover - exercised in environments without RTDE
    rtde_control = None
    rtde_receive = None
    RTDE_IMPORT_ERROR = exc
else:  # pragma: no cover - only hit on the real robot PC
    RTDE_IMPORT_ERROR = None


LOGGER_NAME = "serl_robot_ur.robot_servers.ur_server"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(threadName)s | %(message)s"
logger = logging.getLogger(LOGGER_NAME)
logger.addHandler(logging.NullHandler())


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    logger_name: str = LOGGER_NAME,
) -> logging.Logger:
    app_logger = logging.getLogger(logger_name)
    app_logger.handlers.clear()
    app_logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    app_logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    app_logger.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        app_logger.addHandler(file_handler)

    return app_logger


def quat_to_rotvec(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if quat.size != 4:
        raise ValueError(f"Quaternion must have shape (4,), got {quat.shape}")

    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.zeros((3,), dtype=np.float64)
    x, y, z, w = quat / norm
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return np.zeros((3,), dtype=np.float64)
    axis = np.array([x, y, z], dtype=np.float64) / s
    return axis * angle


def rotvec_to_quat(rotvec) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(-1)
    if rotvec.size != 3:
        raise ValueError(f"Rotation vector must have shape (3,), got {rotvec.shape}")

    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rotvec / angle
    half = angle / 2.0
    sin_half = np.sin(half)
    return np.concatenate([axis * sin_half, [np.cos(half)]])


def quat_to_euler_xyz(quat) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(-1)

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def ensure_quat_pose(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.size == 7:
        return pose
    if pose.size == 6:
        return np.concatenate([pose[:3], rotvec_to_quat(pose[3:])])
    raise ValueError(f"Pose must have 6 or 7 elements, got {pose.shape}")


def quat_pose_to_rotvec_pose(pose) -> np.ndarray:
    pose = ensure_quat_pose(pose)
    return np.concatenate([pose[:3], quat_to_rotvec(pose[3:])])


def _call_optional(target: Any, method_names: list[str], *args, **kwargs):
    for method_name in method_names:
        method = getattr(target, method_name, None)
        if not callable(method):
            continue
        try:
            return method(*args, **kwargs)
        except TypeError:
            continue
    return None


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    return value


def _optional_or_default(value, default):
    return default if value is None else value


def _format_vector(label: str, value, precision: int = 6) -> str:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return f"{label}={np.array2string(arr, precision=precision, separator=', ')}"


class DummyGripperServer:
    """No-op gripper used for the current no-gripper UR5 deployment."""

    def __init__(self, gripper_ip=None):
        self.gripper_ip = gripper_ip
        self.gripper_pos = 0.0

    def activate_gripper(self):
        self.gripper_pos = 0.0

    def reset_gripper(self):
        self.gripper_pos = 0.0

    def open(self):
        self.gripper_pos = 1.0

    def close(self):
        self.gripper_pos = 0.0

    def close_slow(self):
        self.gripper_pos = 0.0

    def move(self, pos):
        pos = float(np.clip(pos, 0.0, 255.0))
        self.gripper_pos = pos / 255.0 if pos > 1.0 else pos


class URStreamingController:
    """Background RTDE controller with servo and forceMode backends."""

    def __init__(
        self,
        robot_ip: str,
        reset_joint_target,
        controller_mode: str = "forcemode",
        control_hz: float = 20.0,
        default_speed: float = 0.2,
        default_accel: float = 0.2,
        disable_motion: bool = False,
        start_paused_until_pose: bool = False,
        rtde_control_cls=None,
        rtde_receive_cls=None,
        logger_instance: Optional[logging.Logger] = None,
    ):
        if rtde_control_cls is None or rtde_receive_cls is None:
            if rtde_control is None or rtde_receive is None:
                raise RuntimeError(
                    "rtde_control / rtde_receive not installed or not available. "
                    "Install the Universal Robots RTDE client libraries on the robot PC."
                ) from RTDE_IMPORT_ERROR
            rtde_control_cls = rtde_control.RTDEControlInterface
            rtde_receive_cls = rtde_receive.RTDEReceiveInterface

        self.robot_ip = robot_ip
        self.logger = logger_instance or logging.getLogger(LOGGER_NAME)
        self.controller_mode = controller_mode
        self.control_hz = 100.0 if controller_mode == "forcemode" else control_hz
        self.default_speed = default_speed
        self.default_accel = default_accel
        self.disable_motion = disable_motion
        self.start_paused_until_pose = start_paused_until_pose
        self.servo_gain = 300
        self.lookahead_time = 0.1
        self.translational_stiffness = 2000.0
        self.translational_damping = 89.0
        self.rotational_stiffness = 150.0
        self.rotational_damping = 7.0
        self.translational_clip = np.array([0.01, 0.01, 0.01], dtype=np.float64)
        self.translational_clip_neg = np.array([0.01, 0.01, 0.01], dtype=np.float64)
        self.rotational_clip = np.array([0.05, 0.05, 0.05], dtype=np.float64)
        self.rotational_clip_neg = np.array([0.05, 0.05, 0.05], dtype=np.float64)
        self.force_mode_damping = 0.05
        self.force_mode_task_frame = np.zeros((6,), dtype=np.float64)
        self.force_mode_selection_vector = np.ones((6,), dtype=np.int32)
        self.force_mode_limits = np.array([2.0, 2.0, 2.0, 1.5, 1.5, 1.5], dtype=np.float64)
        self.force_mode_idle_timeout = 0.2
        self.force_mode_pos_deadband = np.array([5e-4, 5e-4, 5e-4], dtype=np.float64)
        self.force_mode_rot_deadband = np.array([5e-3, 5e-3, 5e-3], dtype=np.float64)
        self.force_mode_vel_deadband = np.array([5e-3, 5e-3, 5e-3, 1e-2, 1e-2, 1e-2], dtype=np.float64)
        self.pose_filter_alpha = 0.2
        self.vel_filter_alpha = 0.1
        self.force_filter_alpha = 0.1
        self.last_policy_target_time = 0.0
        self._force_mode_active = False
        self._filtered_pose_rtde = None
        self._filtered_vel_rtde = None
        self._filtered_force_rtde = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._motion_lock = threading.Lock()
        self._servo_paused = bool(start_paused_until_pose)
        self._thread = threading.Thread(target=self._run, daemon=True)

        self.rtde_c = rtde_control_cls(robot_ip)
        self.rtde_r = rtde_receive_cls(robot_ip)

        self.reset_joint_target = np.asarray(reset_joint_target, dtype=np.float64).reshape(6)
        state = self.get_state()
        self.target_pose = np.array(state["pose"], dtype=np.float64)
        self.target_speed = default_speed
        self.target_accel = default_accel
        self.last_state = state
        self._thread.start()
        self.logger.info(
            "controller_initialized robot_ip=%s mode=%s control_hz=%.1f disable_motion=%s start_paused_until_pose=%s",
            self.robot_ip,
            self.controller_mode,
            self.control_hz,
            self.disable_motion,
            self.start_paused_until_pose,
        )

    def close(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.stop_servo()
        _call_optional(self.rtde_c, ["disconnect"])
        _call_optional(self.rtde_r, ["disconnect"])
        self.logger.info("controller_closed")

    def update_param(self, params: dict[str, Any]) -> dict[str, Any]:
        params = dict(params or {})
        applied = {}
        consumed_keys = set()

        translational_stiffness = params.get("translational_stiffness")
        if translational_stiffness is not None:
            stiffness = float(np.clip(float(translational_stiffness), 100.0, 2000.0))
            self.servo_gain = int(stiffness)
            self.translational_stiffness = stiffness
            applied["translational_stiffness"] = stiffness
            consumed_keys.add("translational_stiffness")

        translational_damping = params.get("translational_damping")
        if translational_damping is not None:
            damping = float(translational_damping)
            self.translational_damping = damping
            if self.controller_mode == "servo":
                self.lookahead_time = float(np.clip(damping / 1000.0, 0.03, 0.2))
            applied["translational_damping"] = damping
            consumed_keys.add("translational_damping")

        rotational_stiffness = params.get("rotational_stiffness")
        if rotational_stiffness is not None:
            rotational_stiffness = float(rotational_stiffness)
            self.rotational_stiffness = rotational_stiffness
            applied["rotational_stiffness"] = rotational_stiffness
            consumed_keys.add("rotational_stiffness")

        rotational_damping = params.get("rotational_damping")
        if rotational_damping is not None:
            rotational_damping = float(rotational_damping)
            self.rotational_damping = rotational_damping
            applied["rotational_damping"] = rotational_damping
            consumed_keys.add("rotational_damping")

        translational_clip, translational_clip_neg, translational_clip_keys = self._update_axis_clip_params(
            params,
            "translational",
            self.translational_clip,
            self.translational_clip_neg,
        )
        if translational_clip is not None:
            self.translational_clip = translational_clip
            self.translational_clip_neg = translational_clip_neg
            applied["translational_clip"] = translational_clip.tolist()
            applied["translational_clip_neg"] = translational_clip_neg.tolist()
        consumed_keys.update(translational_clip_keys)

        rotational_clip, rotational_clip_neg, rotational_clip_keys = self._update_axis_clip_params(
            params,
            "rotational",
            self.rotational_clip,
            self.rotational_clip_neg,
        )
        if rotational_clip is not None:
            self.rotational_clip = rotational_clip
            self.rotational_clip_neg = rotational_clip_neg
            applied["rotational_clip"] = rotational_clip.tolist()
            applied["rotational_clip_neg"] = rotational_clip_neg.tolist()
        consumed_keys.update(rotational_clip_keys)

        if "force_mode_damping" in params:
            self.force_mode_damping = float(np.clip(float(params["force_mode_damping"]), 0.0, 1.0))
            if self.controller_mode == "forcemode":
                _call_optional(self.rtde_c, ["forceModeSetDamping"], self.force_mode_damping)
            applied["force_mode_damping"] = self.force_mode_damping
            consumed_keys.add("force_mode_damping")

        ignored = sorted([key for key in params if key not in consumed_keys])
        return {"applied": applied, "ignored": ignored}

    def set_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload or {})
        mass = float(payload.get("mass", 0.0))
        center_of_mass = payload.get("center_of_mass", payload.get("F_x_center_load", [0.0, 0.0, 0.0]))
        center_of_mass = list(center_of_mass)[:3]
        while len(center_of_mass) < 3:
            center_of_mass.append(0.0)
        applied = callable(getattr(self.rtde_c, "setPayload", None))
        if applied:
            self.rtde_c.setPayload(mass, center_of_mass)
        return {"mass": mass, "center_of_mass": center_of_mass, "applied": applied}

    def clear_error(self) -> str:
        self.logger.info("clear_error requested")
        self.stop_servo()
        if self.controller_mode == "forcemode":
            _call_optional(self.rtde_c, ["zeroFtSensor"])
        if _call_optional(self.rtde_c, ["reuploadScript"]) is not None:
            return "Cleared and reuploaded RTDE script"
        return "Cleared (best effort)"

    def set_target_pose(self, pose, speed=None, accel=None, source: str = "policy") -> np.ndarray:
        pose = ensure_quat_pose(pose)
        state = self.get_state()
        current_pose = state["pose"]
        self.logger.debug(
            "set_target_pose source=%s %s %s speed=%s accel=%s",
            source,
            _format_vector("current_pose", current_pose),
            _format_vector("target_pose", pose),
            _optional_or_default(speed, self.target_speed),
            _optional_or_default(accel, self.target_accel),
        )
        with self._lock:
            self.target_pose = pose
            if speed is not None:
                self.target_speed = float(speed)
            if accel is not None:
                self.target_accel = float(accel)
            if source == "policy":
                self.last_policy_target_time = time.monotonic()
            if self._servo_paused and self.start_paused_until_pose and source == "policy":
                self._servo_paused = False
                self.start_paused_until_pose = False
                self.logger.info("motion_resumed_after_first_pose")
            return self.target_pose.copy()

    def move_joints(self, q, speed=None, accel=None) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != 6:
            raise ValueError("q must have length 6 for UR5")
        speed = float(self.default_speed if speed is None else speed)
        accel = float(self.default_accel if accel is None else accel)
        state = self.get_state()
        current_q = state["q"]
        current_pose = state["pose"]
        self.logger.info(
            "move_joints %s %s speed=%.6f accel=%.6f %s",
            _format_vector("current_q", current_q),
            _format_vector("q", q),
            speed,
            accel,
            _format_vector("current_pose", current_pose),
        )
        if self.disable_motion:
            self.logger.warning("move_joints skipped: disable_motion=True")
            return np.asarray(current_q, dtype=np.float64)
        with self._motion_lock:
            self._servo_paused = True
            try:
                self.stop_servo()
                _call_optional(self.rtde_c, ["moveJ"], q.tolist(), speed, accel)
            finally:
                self._servo_paused = False

        state = self.get_state()
        with self._lock:
            self.target_pose = np.array(state["pose"], dtype=np.float64)
        return q

    def joint_reset(self, target=None, speed=None, accel=None) -> np.ndarray:
        target = self.reset_joint_target if target is None else np.asarray(target, dtype=np.float64).reshape(-1)
        if target.size != 6:
            raise ValueError("Reset joint target must have length 6")
        self.logger.info(
            "joint_reset %s speed=%s accel=%s",
            _format_vector("target_q", target),
            _optional_or_default(speed, self.default_speed),
            _optional_or_default(accel, self.default_accel),
        )
        return self.move_joints(target, speed=speed, accel=accel)

    def get_state(self) -> dict[str, Any]:
        raw_pose_value = _optional_or_default(_call_optional(self.rtde_r, ["getActualTCPPose"]), np.zeros((6,)))
        raw_vel_value = _optional_or_default(_call_optional(self.rtde_r, ["getActualTCPSpeed"]), np.zeros((6,)))
        raw_force_value = _optional_or_default(_call_optional(self.rtde_r, ["getActualTCPForce"]), np.zeros((6,)))
        q_value = _optional_or_default(_call_optional(self.rtde_r, ["getActualQ"]), np.zeros((6,)))
        dq_value = _optional_or_default(_call_optional(self.rtde_r, ["getActualQd"]), np.zeros((6,)))
        jacobian_value = _optional_or_default(
            _call_optional(self.rtde_r, ["getActualJacobian", "getJacobian"]),
            np.zeros((36,)),
        )

        raw_pose = np.asarray(
            raw_pose_value,
            dtype=np.float64,
        ).reshape(-1)
        raw_vel = np.asarray(
            raw_vel_value,
            dtype=np.float64,
        ).reshape(-1)
        raw_force = np.asarray(
            raw_force_value,
            dtype=np.float64,
        ).reshape(-1)
        q = np.asarray(
            q_value,
            dtype=np.float64,
        ).reshape(-1)
        dq = np.asarray(
            dq_value,
            dtype=np.float64,
        ).reshape(-1)
        jacobian = np.asarray(
            jacobian_value,
            dtype=np.float64,
        ).reshape(-1)
        if jacobian.size % 6 != 0:
            jacobian = np.zeros((36,), dtype=np.float64)

        # RTDE tcp pose is already xyz + rotvec, which is convenient for smoothing
        # before converting to the env-facing xyz + quat convention.
        if self._filtered_pose_rtde is None:
            self._filtered_pose_rtde = raw_pose.copy()
        else:
            self._filtered_pose_rtde = (
                self.pose_filter_alpha * raw_pose
                + (1.0 - self.pose_filter_alpha) * self._filtered_pose_rtde
            )

        if self._filtered_vel_rtde is None:
            self._filtered_vel_rtde = raw_vel.copy()
        else:
            self._filtered_vel_rtde = (
                self.vel_filter_alpha * raw_vel
                + (1.0 - self.vel_filter_alpha) * self._filtered_vel_rtde
            )

        if self._filtered_force_rtde is None:
            self._filtered_force_rtde = raw_force.copy()
        else:
            self._filtered_force_rtde = (
                self.force_filter_alpha * raw_force
                + (1.0 - self.force_filter_alpha) * self._filtered_force_rtde
            )

        state = {
            "pose": ensure_quat_pose(self._filtered_pose_rtde).tolist(),
            "vel": self._filtered_vel_rtde.tolist(),
            "force": self._filtered_force_rtde[:3].tolist(),
            "torque": self._filtered_force_rtde[3:6].tolist(),
            "q": q.tolist(),
            "dq": dq.tolist(),
            "jacobian": jacobian.reshape(6, jacobian.size // 6).tolist(),
            "gripper_pos": [0.0],
        }
        self.last_state = state
        return state

    def stop_servo(self):
        if self.controller_mode == "forcemode":
            _call_optional(self.rtde_c, ["forceModeStop"])
            self._force_mode_active = False
        _call_optional(self.rtde_c, ["servoStop"])
        _call_optional(self.rtde_c, ["stopL"])
        _call_optional(self.rtde_c, ["speedStop"])

    def _update_axis_clip_params(self, params, prefix, current_pos, current_neg):
        updated = False
        new_pos = np.array(current_pos, dtype=np.float64)
        new_neg = np.array(current_neg, dtype=np.float64)
        consumed_keys = set()
        for index, axis in enumerate("xyz"):
            pos_key = f"{prefix}_clip_{axis}"
            neg_key = f"{prefix}_clip_neg_{axis}"
            if pos_key in params:
                new_pos[index] = abs(float(params[pos_key]))
                updated = True
                consumed_keys.add(pos_key)
            if neg_key in params:
                new_neg[index] = abs(float(params[neg_key]))
                updated = True
                consumed_keys.add(neg_key)
        if not updated:
            return None, None, consumed_keys
        return new_pos, new_neg, consumed_keys

    def _compute_force_mode_wrench(self, state: dict[str, Any], target_pose: np.ndarray) -> np.ndarray:
        curr_pose = np.asarray(state["pose"], dtype=np.float64)
        curr_vel = np.asarray(state["vel"], dtype=np.float64)
        pos_error = target_pose[:3] - curr_pose[:3]
        pos_error = np.where(np.abs(pos_error) < self.force_mode_pos_deadband, 0.0, pos_error)
        pos_error = np.clip(
            pos_error,
            -self.translational_clip_neg,
            self.translational_clip,
        )
        vel = np.where(np.abs(curr_vel) < self.force_mode_vel_deadband, 0.0, curr_vel)
        force = self.translational_stiffness * pos_error - self.translational_damping * vel[:3]

        rot_error = (R.from_quat(target_pose[3:]) * R.from_quat(curr_pose[3:]).inv()).as_rotvec()
        rot_error = np.where(np.abs(rot_error) < self.force_mode_rot_deadband, 0.0, rot_error)
        rot_error = np.clip(rot_error, -self.rotational_clip_neg, self.rotational_clip)
        torque = self.rotational_stiffness * rot_error - self.rotational_damping * vel[3:]
        return np.concatenate([force, torque]).astype(np.float64)

    def _run(self):
        dt = 1.0 / self.control_hz
        if self.controller_mode == "forcemode":
            _call_optional(self.rtde_c, ["forceModeSetDamping"], self.force_mode_damping)
            _call_optional(self.rtde_c, ["zeroFtSensor"])
        while not self._stop_event.is_set():
            if self.disable_motion:
                # Dry-run mode for debugging command generation without moving the robot.
                self.get_state()
                time.sleep(dt)
                continue
            if getattr(self, '_servo_paused', False):
                time.sleep(0.01)
                continue
            
            cycle_start = time.monotonic()
            with self._lock:
                target_pose = self.target_pose.copy()
                speed = self.target_speed
                accel = self.target_accel

            state = self.get_state()
            with self._motion_lock:
                period = _call_optional(self.rtde_c, ["initPeriod"])
                if self.controller_mode == "forcemode":
                    policy_is_fresh = (time.monotonic() - self.last_policy_target_time) <= self.force_mode_idle_timeout
                    if not policy_is_fresh:
                        if self._force_mode_active:
                            _call_optional(self.rtde_c, ["forceModeStop"])
                            self._force_mode_active = False
                        if period is not None:
                            _call_optional(self.rtde_c, ["waitPeriod"], period)
                        remaining = dt - (time.monotonic() - cycle_start)
                        if remaining > 0:
                            time.sleep(remaining)
                        continue
                    wrench = self._compute_force_mode_wrench(state, target_pose)
                    if np.allclose(wrench, 0.0, atol=1e-6):
                        if self._force_mode_active:
                            _call_optional(self.rtde_c, ["forceModeStop"])
                            self._force_mode_active = False
                        if period is not None:
                            _call_optional(self.rtde_c, ["waitPeriod"], period)
                        remaining = dt - (time.monotonic() - cycle_start)
                        if remaining > 0:
                            time.sleep(remaining)
                        continue
                    _call_optional(
                        self.rtde_c,
                        ["forceMode"],
                        self.force_mode_task_frame.tolist(),
                        self.force_mode_selection_vector.tolist(),
                        wrench.tolist(),
                        2,
                        self.force_mode_limits.tolist(),
                    )
                    self._force_mode_active = True
                else:
                    target_rotvec = quat_pose_to_rotvec_pose(target_pose).tolist()
                    servo_result = _call_optional(
                        self.rtde_c,
                        ["servoL"],
                        target_rotvec,
                        speed,
                        accel,
                        dt,
                        self.lookahead_time,
                        self.servo_gain,
                    )
                    if servo_result is None:
                        _call_optional(self.rtde_c, ["moveL"], target_rotvec, speed, accel)
                if period is not None:
                    _call_optional(self.rtde_c, ["waitPeriod"], period)

            remaining = dt - (time.monotonic() - cycle_start)
            if remaining > 0:
                time.sleep(remaining)


def create_app(
    robot_ip,
    gripper_type=None,
    gripper_ip=None,
    reset_joint_target=None,
    controller_mode: str = "forcemode",
    control_hz: float = 20.0,
    default_speed: float = 0.6,
    default_accel: float = 1.2,
    disable_motion: bool = False,
    start_paused_until_pose: bool = False,
    backend_factory: Optional[Callable[..., Any]] = None,
    logger_instance: Optional[logging.Logger] = None,
):
    app = Flask(__name__)
    app_logger = logger_instance or logging.getLogger(LOGGER_NAME)

    reset_joint_target = reset_joint_target or [-1.25,-1.74,-1.79,-1.17,1.58,1.90]
    if backend_factory is None:
        robot_backend = URStreamingController(
            robot_ip,
            reset_joint_target=reset_joint_target,
            controller_mode=controller_mode,
            control_hz=control_hz,
            default_speed=default_speed,
            default_accel=default_accel,
            disable_motion=disable_motion,
            start_paused_until_pose=start_paused_until_pose,
            logger_instance=app_logger,
        )
    else:
        robot_backend = backend_factory(robot_ip=robot_ip, reset_joint_target=reset_joint_target)

    if callable(getattr(robot_backend, "close", None)):
        atexit.register(robot_backend.close)

    if gripper_type is None or gripper_type == "None":
        gripper_server = DummyGripperServer()
    else:
        gripper_server = DummyGripperServer(gripper_ip)

    app.config["robot_backend"] = robot_backend
    app.config["gripper_server"] = gripper_server
    app.config["logger"] = app_logger

    @app.before_request
    def _log_request_start():
        g.request_id = uuid.uuid4().hex[:8]
        g.request_started_at = time.monotonic()

    @app.after_request
    def _log_request_end(response):
        request_id = getattr(g, "request_id", "-")
        started_at = getattr(g, "request_started_at", None)
        duration_ms = -1.0
        if started_at is not None:
            duration_ms = (time.monotonic() - started_at) * 1000.0
        app_logger.info(
            "http_request id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    def _request_id() -> str:
        return getattr(g, "request_id", "-")

    @app.route("/pose", methods=["POST"])
    def pose():
        try:
            body = request.get_json(silent=True) or {}
            arr = body.get("arr")
            if arr is None:
                return "Missing 'arr' field", 400
            app_logger.debug(
                "route_pose id=%s %s speed=%s accel=%s",
                _request_id(),
                _format_vector("arr", arr),
                body.get("speed"),
                body.get("accel"),
            )
            pose_cmd = robot_backend.set_target_pose(
                arr,
                speed=body.get("speed"),
                accel=body.get("accel"),
                source=body.get("source", "policy"),
            )
            return jsonify({"status": "Moved", "target_pose": _to_jsonable(pose_cmd)})
        except Exception as exc:
            app_logger.exception("route_pose failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/movel_async", methods=["POST"])
    def movel_async():
        return pose()

    @app.route("/movej", methods=["POST"])
    def movej():
        try:
            body = request.get_json(silent=True) or {}
            q = body.get("q", [])
            app_logger.info(
                "route_movej id=%s %s speed=%s accel=%s",
                _request_id(),
                _format_vector("q", q),
                body.get("speed"),
                body.get("accel"),
            )
            robot_backend.move_joints(q, speed=body.get("speed"), accel=body.get("accel"))
            return "MovedJ", 200
        except Exception as exc:
            app_logger.exception("route_movej failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/getpos", methods=["POST"])
    def getpos():
        try:
            return jsonify({"pose": _to_jsonable(robot_backend.get_state()["pose"])})
        except Exception as exc:
            app_logger.exception("route_getpos failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/getpos_euler", methods=["POST"])
    def getpos_euler():
        try:
            state = robot_backend.get_state()
            pose = np.asarray(state["pose"], dtype=np.float64)
            pose_euler = np.concatenate([pose[:3], quat_to_euler_xyz(pose[3:])])
            return jsonify({"pose": pose_euler.tolist()})
        except Exception as exc:
            app_logger.exception("route_getpos_euler failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/getvel", methods=["POST"])
    def getvel():
        try:
            return jsonify({"vel": robot_backend.get_state()["vel"]})
        except Exception as exc:
            app_logger.exception("route_getvel failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/getforce", methods=["POST"])
    def getforce():
        try:
            return jsonify({"force": robot_backend.get_state()["force"]})
        except Exception as exc:
            app_logger.exception("route_getforce failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/gettorque", methods=["POST"])
    def gettorque():
        try:
            return jsonify({"torque": robot_backend.get_state()["torque"]})
        except Exception as exc:
            app_logger.exception("route_gettorque failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/getq", methods=["POST"])
    def getq():
        try:
            return jsonify({"q": robot_backend.get_state()["q"]})
        except Exception as exc:
            app_logger.exception("route_getq failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/getdq", methods=["POST"])
    def getdq():
        try:
            return jsonify({"dq": robot_backend.get_state()["dq"]})
        except Exception as exc:
            app_logger.exception("route_getdq failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/getjacobian", methods=["POST"])
    def getjacobian():
        try:
            return jsonify({"jacobian": robot_backend.get_state()["jacobian"]})
        except Exception as exc:
            app_logger.exception("route_getjacobian failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": gripper_server.gripper_pos})

    @app.route("/getstate", methods=["POST"])
    def getstate():
        try:
            state = robot_backend.get_state()
            state["gripper_pos"] = [float(gripper_server.gripper_pos)]
            return jsonify(_to_jsonable(state))
        except Exception as exc:
            app_logger.exception("route_getstate failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/jointreset", methods=["POST"])
    def jointreset():
        try:
            body = request.get_json(silent=True) or {}
            target = body.get("target", reset_joint_target)
            app_logger.info(
                "route_jointreset id=%s %s speed=%s accel=%s",
                _request_id(),
                _format_vector("target", target),
                body.get("speed"),
                body.get("accel"),
            )
            robot_backend.joint_reset(target=target, speed=body.get("speed"), accel=body.get("accel"))
            return "Reset Joint", 200
        except Exception as exc:
            app_logger.exception("route_jointreset failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        gripper_server.activate_gripper()
        return "Activated", 200

    @app.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        gripper_server.reset_gripper()
        return "Reset", 200

    @app.route("/open_gripper", methods=["POST"])
    def open_gripper():
        gripper_server.open()
        return "Opened", 200

    @app.route("/close_gripper", methods=["POST"])
    def close_gripper():
        gripper_server.close()
        return "Closed", 200

    @app.route("/close_gripper_slow", methods=["POST"])
    def close_gripper_slow():
        gripper_server.close_slow()
        return "ClosedSlow", 200

    @app.route("/move_gripper", methods=["POST"])
    def move_gripper():
        body = request.get_json(silent=True) or {}
        gripper_server.move(body.get("gripper_pos", 0))
        return "Moved", 200

    @app.route("/set_load", methods=["POST"])
    @app.route("/set_payload", methods=["POST"])
    def set_payload():
        try:
            result = robot_backend.set_payload(request.get_json(silent=True) or {})
            return jsonify(_to_jsonable(result))
        except Exception as exc:
            app_logger.exception("route_set_payload failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/clearerr", methods=["POST"])
    def clearerr():
        try:
            return robot_backend.clear_error(), 200
        except Exception as exc:
            app_logger.exception("route_clearerr failed id=%s", _request_id())
            return f"Error: {exc}", 500

    @app.route("/update_param", methods=["POST"])
    def update_param():
        try:
            result = robot_backend.update_param(request.get_json(silent=True) or {})
            return jsonify(_to_jsonable(result))
        except Exception as exc:
            app_logger.exception("route_update_param failed id=%s", _request_id())
            return f"Error: {exc}", 500

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", type=str, default="192.168.0.10", help="UR controller IP")
    parser.add_argument("--gripper_ip", type=str, default=None, help="Gripper IP if any")
    parser.add_argument("--gripper_type", type=str, default="None", help="Gripper type or None")
    parser.add_argument(
        "--controller_mode",
        choices=["forcemode", "servo"],
        default="forcemode",
        help="Control backend: forceMode for impedance-like control, or servo for streamed pose servoing.",
    )
    parser.add_argument("--control_hz", type=float, default=20.0, help="Base control loop frequency for servo mode.")
    parser.add_argument("--default_speed", type=float, default=0.6, help="Default Cartesian/servo speed.")
    parser.add_argument("--default_accel", type=float, default=1.2, help="Default Cartesian/servo acceleration.")
    parser.add_argument(
        "--disable_motion",
        action="store_true",
        help="Debug mode: accept/log motion commands but do not execute robot motion.",
    )
    parser.add_argument(
        "--start_paused_until_pose",
        action="store_true",
        help="Start the streaming controller paused, then resume after the first /pose command.",
    )
    parser.add_argument("--flask_host", type=str, default="127.0.0.1")
    parser.add_argument("--flask_port", type=int, default=5000)
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Application log level.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional file path for rotating logs.",
    )
    parser.add_argument(
        "--log_max_bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="Max bytes per log file before rotation.",
    )
    parser.add_argument(
        "--log_backup_count",
        type=int,
        default=5,
        help="Number of rotated log files to keep.",
    )
    parser.add_argument(
        "--reset_joint_target",
        nargs=6,
        type=float,
        default=[0.0, -1.57, 1.57, 0.0, 1.57, 0.0],
        help="Default reset joint target (6 joints)",
    )
    args = parser.parse_args()
    app_logger = setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        max_bytes=args.log_max_bytes,
        backup_count=args.log_backup_count,
    )

    app = create_app(
        args.robot_ip,
        gripper_type=args.gripper_type,
        gripper_ip=args.gripper_ip,
        reset_joint_target=args.reset_joint_target,
        controller_mode=args.controller_mode,
        control_hz=args.control_hz,
        default_speed=args.default_speed,
        default_accel=args.default_accel,
        disable_motion=args.disable_motion,
        start_paused_until_pose=args.start_paused_until_pose,
        logger_instance=app_logger,
    )
    app_logger.info(
        "server_starting host=%s port=%s robot_ip=%s mode=%s disable_motion=%s",
        args.flask_host,
        args.flask_port,
        args.robot_ip,
        args.controller_mode,
        args.disable_motion,
    )
    app.run(host=args.flask_host, port=args.flask_port)
