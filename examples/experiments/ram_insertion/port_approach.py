"""回合开始时用固定相机定位，用腕部相机在接近高度对准母口。"""
import logging
import math
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation


LOGGER = logging.getLogger(__name__)

APPROACH_SPEED_MPS = 0.01
COMMAND_PERIOD_S = 0.05
POSITION_TOLERANCE_M = 0.0008
MAX_TRACKING_LAG_M = 0.003
ORIENTATION_TOLERANCE_RAD = 0.02
WRIST_TOLERANCE_M = 0.001
WRIST_STEP_M = 0.001
WRIST_SEARCH_RADIUS_M = 0.008
WRIST_ITERATIONS = 12


class PortApproach:
    def __init__(self, env, vision):
        self.env = env
        self.vision = vision
        self.nominal_target = np.asarray(env.config.TARGET_POSE, dtype=float).copy()
        self.nominal_reset = np.asarray(env.config.RESET_POSE, dtype=float).copy()
        self.nominal_rotation = Rotation.from_rotvec(self.nominal_target[3:])
        self.ready = threading.Event()
        self.waiting_for_fixture = False
        self.previous_round_aligned = False

        # 上方的接近高度仍由已有装置标定决定。视觉对准段不进入插接区。
        if not self.nominal_reset[2] <= vision.approach_z_m <= env.xyz_bounding_box.high[2]:
            raise ValueError("Wrist calibration height must lie between RAM RESET_POSE Z and the workspace top")
        for name in (vision.fixed_camera, vision.wrist_camera):
            if name not in env.camera_config:
                raise ValueError(f"Camera {name!r} from port vision calibration is missing from RAM CAMERAS")

    def on_key(self, key):
        if self.waiting_for_fixture and key.lower() == "u":
            self.ready.set()

    def hold(self):
        response = self.env._post_debug("hold", timeout=5.0)
        response.raise_for_status()

    def _wait_for_fixture(self):
        self.ready.clear()
        self.waiting_for_fixture = True
        try:
            print("母口换到本轮位置并固定后，确认工作区清空，然后在采集或 Actor 终端按 U。", flush=True)
            while not self.ready.wait(0.2):
                pass
        finally:
            self.waiting_for_fixture = False

    def _validate_xyz(self, xyz, roundoff=0.0):
        box = self.env.xyz_bounding_box
        if not np.all(np.isfinite(xyz)) or np.any(xyz < box.low - roundoff) or np.any(xyz > box.high + roundoff):
            raise RuntimeError(f"Port approach XYZ outside RAM workspace: {xyz.tolist()}")

    def read_state(self):
        response = self.env._post_debug("getstate", timeout=1.0)
        response.raise_for_status()
        state = response.json()
        if state.get("safety_stop"):
            raise RuntimeError(f"Robot safety stop: {state.get('safety_reason') or 'latched'}")
        pose = np.asarray(state.get("pose"), dtype=float)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)) or np.linalg.norm(pose[3:]) < 1e-6:
            raise RuntimeError("Robot /getstate did not return a valid TCP pose")
        return state, pose

    def _read_pose(self):
        _, pose = self.read_state()
        self._validate_xyz(pose[:3])
        if (Rotation.from_quat(pose[3:]) * self.nominal_rotation.inv()).magnitude() > ORIENTATION_TOLERANCE_RAD:
            raise RuntimeError("TCP orientation differs from the calibrated insertion orientation")
        return pose

    def send_pose(self, pose, source="policy"):
        pose = np.array(pose, dtype=float)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("Port motion requires a finite XYZ and XYZW quaternion")
        # 原策略使用 float32，裁剪后的边界值可能出现亚微米舍入误差。
        self._validate_xyz(pose[:3], roundoff=1e-7)
        box = self.env.xyz_bounding_box
        pose[:3] = np.clip(pose[:3], box.low, box.high)
        response = self.env._post_debug("pose", payload={"arr": pose.tolist(), "source": source}, timeout=1.0)
        response.raise_for_status()

    def _move(self, goal_xyz):
        start = self._read_pose()
        self._validate_xyz(goal_xyz)
        distance = np.linalg.norm(goal_xyz - start[:3])
        if distance <= POSITION_TOLERANCE_M:
            return
        steps = math.ceil(distance / (APPROACH_SPEED_MPS * COMMAND_PERIOD_S))
        minimum_z = min(start[2], goal_xyz[2]) - POSITION_TOLERANCE_M
        previous = start[:3]
        for step in range(1, steps + 1):
            actual = self._read_pose()
            if actual[2] < minimum_z:
                raise RuntimeError("TCP moved below the planned approach segment")
            if np.linalg.norm(actual[:3] - previous) > MAX_TRACKING_LAG_M:
                raise RuntimeError("TCP tracking error exceeded 3 mm during the vision approach")
            point = start[:3] + (goal_xyz - start[:3]) * (step / steps)
            self.send_pose(np.r_[point, start[3:]])
            previous = point
            time.sleep(COMMAND_PERIOD_S)
        deadline = time.monotonic() + 3.0
        while True:
            actual = self._read_pose()
            if actual[2] < minimum_z:
                raise RuntimeError("TCP moved below the planned approach segment")
            error = np.linalg.norm(actual[:3] - goal_xyz)
            if error > MAX_TRACKING_LAG_M:
                raise RuntimeError("TCP tracking error exceeded 3 mm during settling")
            if error <= POSITION_TOLERANCE_M:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Port approach timed out waiting for the actual TCP to settle")
            # 旧力控后端需要连续刷新合法位置目标。相机读图在停下后进行。
            self.send_pose(np.r_[goal_xyz, start[3:]])
            time.sleep(COMMAND_PERIOD_S)

    def _fresh_frame(self, camera_name):
        capture = self.env.cap[camera_name]
        capture.read()
        return capture.read()

    def _wait_for_stop(self):
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            state, _ = self.read_state()
            velocity = np.asarray(state.get("vel"), dtype=float)
            if velocity.shape != (6,) or not np.all(np.isfinite(velocity)):
                raise RuntimeError("Robot /getstate did not return a valid TCP velocity")
            if np.linalg.norm(velocity[:3]) <= 0.002 and np.linalg.norm(velocity[3:]) <= 0.02:
                return
            time.sleep(COMMAND_PERIOD_S)
        raise RuntimeError("TCP did not stop before the fixture replacement prompt")

    def _align(self):
        self.hold()
        current = self._read_pose()
        self.env._post_debug("update_param", payload=self.env.config.PRECISION_PARAM).raise_for_status()
        approach_z = self.vision.approach_z_m
        if current[2] < approach_z - POSITION_TOLERANCE_M:
            if not self.previous_round_aligned:
                raise RuntimeError("Start the vision approach with the TCP at or above the calibrated approach height")
            self._move(np.r_[current[:2], approach_z])
            self.hold()
        self._wait_for_stop()
        if self._read_pose()[2] < approach_z - POSITION_TOLERANCE_M:
            raise RuntimeError("TCP did not remain above the fixture before the placement prompt")

        self._wait_for_fixture()
        coarse = self.vision.estimate_port_xy(self._fresh_frame(self.vision.fixed_camera))
        current = self._read_pose()
        if current[2] < approach_z - POSITION_TOLERANCE_M:
            raise RuntimeError("TCP is below approach height after fixture placement")

        # 先在安全高度横移，再到腕部标定高度。两条路径均保持实测朝向。
        self._validate_xyz(np.r_[coarse.xy_m, self.nominal_target[2]])
        self._validate_xyz(np.r_[coarse.xy_m, approach_z])
        self._move(np.r_[coarse.xy_m, max(current[2], approach_z)])
        self._move(np.r_[coarse.xy_m, approach_z])
        self.hold()

        for iteration in range(WRIST_ITERATIONS + 1):
            self._wait_for_stop()
            current = self._read_pose()
            if abs(current[2] - approach_z) > POSITION_TOLERANCE_M:
                raise RuntimeError("TCP left the wrist calibration height")
            fine = self.vision.wrist_correction(self._fresh_frame(self.vision.wrist_camera))
            refined_xy = current[:2] + fine.xy_m
            if np.linalg.norm(refined_xy - coarse.xy_m) > WRIST_SEARCH_RADIUS_M:
                raise RuntimeError("Wrist correction disagrees with the fixed-camera estimate by more than 8 mm")
            self._validate_xyz(np.r_[refined_xy, self.nominal_target[2]])
            self._validate_xyz(np.r_[refined_xy, approach_z])
            error = np.linalg.norm(fine.xy_m)
            if error <= WRIST_TOLERANCE_M:
                return coarse, fine, refined_xy, iteration
            if iteration == WRIST_ITERATIONS:
                raise RuntimeError("Wrist alignment did not converge")
            next_xy = current[:2] + fine.xy_m * min(1, WRIST_STEP_M / error)
            self._move(np.r_[next_xy, approach_z])
            self.hold()

    def align(self):
        try:
            coarse, fine, refined_xy, iterations = self._align()
        except BaseException:
            try:
                self.hold()
            except Exception:
                LOGGER.exception("Port approach failed and the server did not confirm hold")
            raise

        # 只在两级定位都完成后提交新目标，每次从标称位姿构建。
        target = self.nominal_target.copy()
        target[:2] = refined_xy
        reset = self.nominal_reset.copy()
        reset[:3] = np.r_[refined_xy, self.vision.approach_z_m]
        self.env._TARGET_POSE = target
        self.env._RESET_POSE = reset
        self.env.resetpos = self.env._pose6_to_quat(reset)
        self.previous_round_aligned = True
        LOGGER.info("Port aligned at %s after %d wrist corrections", refined_xy.tolist(), iterations)
        return {
            "port_coarse_xy_m": coarse.xy_m.tolist(),
            "port_target_xy_m": refined_xy.tolist(),
            "port_wrist_residual_m": fine.xy_m.tolist(),
            "port_alignment_iterations": iterations,
        }
