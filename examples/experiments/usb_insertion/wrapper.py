"""USB insertion environment with a calibrated insertion target."""

import logging
import math
import time

import numpy as np
import requests
from scipy.spatial.transform import Rotation, Slerp

from experiments.ram_insertion.wrapper import RAMEnv

LOGGER = logging.getLogger(__name__)


class USBInsertionEnv(RAMEnv):
    """RAM-style UR5 environment for a fixed-tool USB plug.

    The plug orientation and target position are fixed by the task config.
    At reset, the configured target is selected and all target-dependent state
    in ``urEnv`` is updated before the reset motion starts. This keeps reward
    evaluation and safety clipping aligned with the physical insertion point.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # USB 任务从一开始就关闭坐标成功奖励；采集标签也不能由目标坐标代替。
        self.use_pose_reward = False
        self._reset_rng = np.random.default_rng(getattr(self.config, "RESET_SEED", 42))
        self.current_target_index = 0
        self.current_target_id = 0
        self.current_target_pose = np.asarray(self._TARGET_POSE, dtype=np.float64).copy()
        self._episode_count = 0
        self._set_target_pose(self.current_target_pose)
        if not kwargs.get("fake_env", False):
            try:
                response = requests.post(self.url + "configure_safety", json=self.config.TASK_SAFETY_CONFIG, timeout=5.0)
                response.raise_for_status()
            except Exception:
                self.close()
                raise

    def hold_position(self):
        """停止旧伺服目标并保持实际位置；成功、复位或报错均不清安全锁存。"""
        response = requests.post(self.url + "hold", timeout=5.0)
        response.raise_for_status()

    def _validate_reset_pose(self, pose):
        # 工作区是 TCP 的包围盒，不等价于整个机械臂连杆的碰撞检查。
        if not np.all(np.isfinite(pose)) or not np.all(pose[:3] >= self.xyz_bounding_box.low) or not np.all(pose[:3] <= self.xyz_bounding_box.high):
            raise ValueError("Reset/retreat pose is outside calibrated workspace_bounds")
        relative = Rotation.from_quat(pose[3:]) * Rotation.from_rotvec(self.config.ROTATION_REFERENCE_ROTVEC).inv()
        if np.any(np.abs(relative.as_rotvec()) > self.config.ROTATION_LIMITS_RAD + 1e-8):
            raise ValueError("Reset orientation is outside calibrated relative rotation bounds")

    def sample_reset_pose(self):
        """只随机化起始位姿，不移动真实母口或把目标也随机化。"""
        pose = self._pose6_to_quat(self.config.RESET_POSE)
        if self.randomreset:
            pose[:3] += self._reset_rng.uniform(-self.config.RESET_TRANSLATION_RANGE_M, self.config.RESET_TRANSLATION_RANGE_M)
            delta = self._reset_rng.uniform(-self.config.RESET_ROTATION_RANGE_RAD, self.config.RESET_ROTATION_RANGE_RAD)
            pose[3:] = (Rotation.from_rotvec(delta) * Rotation.from_quat(pose[3:])).as_quat()
        self._validate_reset_pose(pose)
        return pose

    def interpolate_move(self, goal, timeout):
        """复位使用平移插值和球面旋转插值，每条命令也受步长上限约束。"""
        goal = np.asarray(goal, dtype=float)
        if goal.shape == (6,):
            goal = self._pose6_to_quat(goal)
        self._update_currpos()
        start = self.currpos.copy()
        self._validate_reset_pose(start)
        self._validate_reset_pose(goal)
        distance = np.linalg.norm(goal[:3] - start[:3])
        angle = (Rotation.from_quat(goal[3:]) * Rotation.from_quat(start[3:]).inv()).magnitude()
        steps = max(1, math.ceil(timeout*self.hz), math.ceil(distance/self.config.MAX_TRANSLATION_STEP_M), math.ceil(angle/self.config.MAX_ROTATION_STEP_RAD))
        orientation = Slerp([0.0, 1.0], Rotation.from_quat([start[3:], goal[3:]]))
        for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
            pose = np.concatenate([start[:3] + fraction*(goal[:3]-start[:3]), orientation(fraction).as_quat()])
            self._validate_reset_pose(pose)
            self._send_pos_command(pose, source="reset")
            time.sleep(timeout/steps)
        # 等实际状态到位后才开始下一阶段，避免机械臂尚未拔出就横向移动。
        deadline = time.monotonic() + max(1.0, timeout)
        while True:
            self._update_currpos()
            position_error = np.linalg.norm(self.currpos[:3] - goal[:3])
            rotation_error = (Rotation.from_quat(self.currpos[3:]) * Rotation.from_quat(goal[3:]).inv()).magnitude()
            if position_error <= 0.003 and rotation_error <= 0.02:
                break
            if time.monotonic() >= deadline:
                self.hold_position()
                raise RuntimeError("Reset tracking timeout; task not started")
            time.sleep(1.0/self.hz)

    def go_to_reset(self, joint_reset=False):
        # 失败/超时回合也可能仍有旧目标；读状态或校验失败之前必须先停下。
        self.hold_position()
        if joint_reset:
            raise ValueError("Fixed-tool USB task uses Cartesian reset, not jointreset")
        self._update_currpos()
        sampled = self.sample_reset_pose()
        # 先沿插入轴反方向退出到已标定的安全平面，保持当前姿态；再横移/旋转。
        outward = -self.config.INSERTION_AXIS_BASE
        depth = float(np.dot(self.currpos[:3] - self._TARGET_POSE[:3], outward))
        retreat = self.currpos.copy()
        # 到位容差为3mm，因此目标额外退出4mm；随后仍检查实际位姿，不能靠目标推断已拔出。
        retreat[:3] += max(0.0, self.config.RESET_CLEARANCE_M + 0.004-depth) * outward
        self._validate_reset_pose(retreat)
        duration = self.config.RESET_INTERPOLATE_TIMEOUT
        self.interpolate_move(retreat, timeout=duration)
        self._update_currpos()
        actual_clearance = float(np.dot(self.currpos[:3] - self._TARGET_POSE[:3], outward))
        if actual_clearance < self.config.RESET_CLEARANCE_M:
            self.hold_position()
            raise RuntimeError("Retreat did not reach the calibrated clearance plane; no lateral reset executed")
        self.interpolate_move(sampled, timeout=duration)
        self.resetpos = sampled.copy()

    @property
    def target_pose(self):
        """Return the active target pose without exposing mutable state."""
        return self.current_target_pose.copy()

    def _sample_target_index(self, target_index=None):
        targets = np.asarray(self.config.TARGET_POSES)
        if targets.ndim != 2 or targets.shape[1] != 6 or len(targets) == 0:
            raise ValueError("config.TARGET_POSES must have shape (N, 6) with N > 0")

        if target_index is not None:
            index = int(target_index)
            if index < 0 or index >= len(targets):
                raise IndexError(
                    f"target_index {index} is outside [0, {len(targets) - 1}]"
                )
            return index

        selection = getattr(self.config, "TARGET_SELECTION", "random")
        if selection in {"fixed", "single"}:
            return 0
        if selection == "cycle":
            return self._episode_count % len(targets)
        if selection != "random":
            raise ValueError(
                f"Unsupported TARGET_SELECTION={selection!r}; use 'fixed', 'random', or 'cycle'"
            )
        return int(np.random.randint(len(targets)))

    def _set_target_pose(self, target_pose):
        """Synchronize the active target, reset pose, and reward target."""
        target_pose = np.asarray(target_pose, dtype=np.float64).reshape(-1)
        if target_pose.shape != (6,):
            raise ValueError(f"target_pose must have shape (6,), got {target_pose.shape}")
        base_orientation = np.asarray(self.config.BASE_TARGET_POSE, dtype=np.float64)[3:]
        if not np.allclose(target_pose[3:], base_orientation):
            raise ValueError(
                "USB insertion targets may translate the plug, but may not "
                "change its orientation"
            )

        # The final three components are copied from the configured target
        # pose; this method never introduces a rotation perturbation.
        self.current_target_pose = target_pose.copy()
        self._TARGET_POSE = target_pose.copy()
        self.config.TARGET_POSE = target_pose.copy()
        self.config.GRASP_POSE = target_pose.copy()

        reset_offset = np.asarray(
            getattr(self.config, "RESET_TARGET_OFFSET", [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        if reset_offset.shape != (6,):
            raise ValueError("RESET_TARGET_OFFSET must have shape (6,)")
        reset_pose = target_pose + reset_offset
        self._RESET_POSE = reset_pose.copy()
        self.config.RESET_POSE = reset_pose.copy()
        self.resetpos = self._pose6_to_quat(reset_pose)

        # The static safety box must contain the target and its reset pose.
        # Fail early if a custom calibrated pose violates it.
        target_in_box = np.all(target_pose[:3] >= self.xyz_bounding_box.low) and np.all(
            target_pose[:3] <= self.xyz_bounding_box.high
        )
        reset_in_box = np.all(reset_pose[:3] >= self.xyz_bounding_box.low) and np.all(
            reset_pose[:3] <= self.xyz_bounding_box.high
        )
        if not target_in_box or not reset_in_box:
            raise ValueError(
                "Active USB target or reset pose is outside "
                f"ABS_POSE_LIMIT_LOW/HIGH: target={target_pose[:3]}, "
                f"reset={reset_pose[:3]}"
            )

    def select_target_pose(self, target_index=None):
        """Select and activate a target without starting a reset motion."""
        index = self._sample_target_index(target_index)
        self.current_target_index = index
        self.current_target_id = index
        self._set_target_pose(np.asarray(self.config.TARGET_POSES[index]))
        self._episode_count += 1
        return self.target_pose

    def reset(self, joint_reset=False, target_index=None, options=None, **kwargs):
        """Select a target, then run the RAM-compatible reset sequence.

        ``target_index`` is useful for evaluation and calibration.  Training
        can omit it to use the configured selection mode. The USB
        single-position configuration uses ``TARGET_SELECTION='fixed'``.
        """
        # 保存视频或选择目标也可能抛错，所以不能等开始运动时才撤销旧目标。
        # go_to_reset 可独立调用，其入口还会再做一次幂等 hold。
        self.hold_position()
        if target_index is None and options is not None:
            target_index = options.get("target_index")

        self.select_target_pose(target_index)
        LOGGER.info(
            "USB target index=%d pose=%s",
            self.current_target_index,
            np.array2string(self.current_target_pose, precision=5),
        )
        seed = kwargs.pop("seed", None)
        if seed is not None:
            self._reset_rng = np.random.default_rng(seed)
        if self.save_video:
            self.save_video_recording()
        # 不在 reset 中调用 clearerr：安全停止必须由现场确认后显式解除。
        self.go_to_reset(joint_reset=joint_reset)
        self.curr_path_length = 0
        self.terminate = False
        self.should_regrasp = False
        self._update_currpos()
        return self._get_obs(), {"succeed": False, "target_index": self.current_target_index}


# Short alias matching the existing USB task naming in older experiments.
USBEnv = USBInsertionEnv

