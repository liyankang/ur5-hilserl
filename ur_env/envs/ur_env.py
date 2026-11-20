"""Gym Interface for ur"""
import logging
import os
import numpy as np
import gymnasium as gym
import cv2
import copy
from scipy.spatial.transform import Rotation
import time
import requests
import queue
import threading
from datetime import datetime
from collections import OrderedDict
from typing import Dict

from ur_env.camera.video_capture import VideoCapture
from ur_env.camera.rs_capture import RSCapture
from ur_env.camera.v4l_capture import V4LCapture
from ur_env.utils.rotations import euler_2_quat

LOGGER = logging.getLogger(__name__)


class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name, display_size=(400, 400)):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name
        self.display_size = tuple(display_size)
    def run(self):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.display_size[0] * 3, self.display_size[1])
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frames = [v for k, v in img_array.items() if "full" not in k]
            # Display the original camera frames without task cropping or model resizing.
            heights = {frame.shape[0] for frame in frames}
            if len(heights) == 1:
                frame = np.concatenate(frames, axis=1)
            else:
                # Keep mixed camera resolutions displayable while preserving each frame's pixels.
                height = max(heights)
                padded = []
                for image in frames:
                    pad = np.zeros((height - image.shape[0], image.shape[1], image.shape[2]), dtype=image.dtype)
                    padded.append(np.concatenate([image, pad], axis=0))
                frame = np.concatenate(padded, axis=1)

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


class DefaultEnvConfig:
    """Default configuration for urEnv. Fill in the values below."""

    SERVER_URL: str = "http://127.0.0.1:5000/"
    CAMERAS: Dict = {
        "wrist_1": {
            "video_device": "/dev/video0",
            "dim": (640, 480),
            "fps": 10,
        },
        "wrist_2": {
            "video_device": "/dev/video2",
            "dim": (640, 480),
            "fps": 10,
        },
    }
    REALSENSE_CAMERAS: Dict = {
        "wrist_1": "130322274175",
        "wrist_2": "127122270572",
    }
    IMAGE_CROP: dict[str, callable] = {}
    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    # 每个策略步允许请求的平移/旋转范数；None 保持旧任务行为。
    MAX_TRANSLATION_STEP_M = None
    MAX_ROTATION_STEP_RAD = None
    # 新任务用“相对基准朝向”的旋转边界，避免逐分量裁剪绝对 rotvec。
    ROTATION_REFERENCE_ROTVEC = None
    ROTATION_LIMITS_RAD = None
    ACTION_ROTATION_REPR: str = "euler"
    POSE_ROTATION_REPR: str = "euler"
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))
    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    DISPLAY_IMAGE: bool = True
    # Display-only size; model observations remain 128x128.
    DISPLAY_IMAGE_SIZE = (400, 400)
    GRIPPER_SLEEP: float = 0.6
    GRIPPER_ENABLED: bool = True
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0


##############################################################################


class urEnv(gym.Env):
    def __init__(
        self,
        hz=10,
        fake_env=False,
        save_video=False,
        config: DefaultEnvConfig = None,
        set_load=False,
    ):
        self.action_scale = config.ACTION_SCALE
        self.action_rotation_repr = getattr(config, "ACTION_ROTATION_REPR", "euler")
        self.pose_rotation_repr = getattr(config, "POSE_ROTATION_REPR", "euler")
        self._TARGET_POSE = config.TARGET_POSE
        self._RESET_POSE = config.RESET_POSE
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.url = config.SERVER_URL
        self.config = config
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP
        self.gripper_enabled = config.GRIPPER_ENABLED
        # 无夹爪时只产生六维末端动作，内部也不拼接虚假的夹爪控制量。
        self.terminate = False
        self._safety_stop = False
        self.camera_config = getattr(config, "CAMERAS", None) or config.REALSENSE_CAMERAS

        if self.pose_rotation_repr not in {"euler", "rotvec"}:
            raise ValueError(
                f"Unsupported POSE_ROTATION_REPR={self.pose_rotation_repr!r}. "
                "Expected 'euler' or 'rotvec'."
            )

        # convert configured pose from xyz + rotation-vector-parameterization to xyz + quat
        self.resetpos = self._pose6_to_quat(config.RESET_POSE)
        if fake_env:
            self._init_fake_state()
        else:
            self._update_currpos()
        self.last_gripper_act = time.time()
        self.lastsent = time.time()
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.hz = hz
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD  # reset the robot joint every 200 cycles

        self.save_video = save_video
        if self.save_video:
            LOGGER.info("Saving videos!")
            self.recording_frames = []

        # boundary box
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rot_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )
        # Action/Observation Space
        action_dim = 7 if self.gripper_enabled else 6
        self.action_space = gym.spaces.Box(
            np.ones((action_dim,), dtype=np.float32) * -1,
            np.ones((action_dim,), dtype=np.float32),
        )

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),  # xyz + quat
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {key: gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
                                for key in self.camera_config}
                ),
            }
        )
        self.cycle_count = 0

        if fake_env:
            return

        self.cap = None
        self.init_cameras(self.camera_config)
        if self.display_image:
            self.img_queue = queue.Queue(maxsize=1)
            self.displayer = ImageDisplayer(
                self.img_queue,
                self.url,
                display_size=getattr(config, "DISPLAY_IMAGE_SIZE", (400, 400)),
            )
            self.displayer.start()

        if set_load:
            input("Put arm into programing mode and press enter.")
            requests.post(self.url + "set_load", json=self.config.LOAD_PARAM)
            input("Put arm into execution mode and press enter.")
            for _ in range(2):
                self._recover()
                time.sleep(1)

        if not fake_env and getattr(config, "ENABLE_ESC_LISTENER", False):
            from pynput import keyboard
            self.terminate = False

            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        LOGGER.info("Initialized ur")

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        )
        reference = getattr(self.config, "ROTATION_REFERENCE_ROTVEC", None)
        relative_limits = getattr(self.config, "ROTATION_LIMITS_RAD", None)
        if reference is not None and relative_limits is not None:
            # 用基坐标轴表达相对插入姿态的小角度误差，与 reset 的左乘扰动一致。
            reference_rotation = Rotation.from_rotvec(np.asarray(reference, dtype=np.float64))
            limits = np.asarray(relative_limits, dtype=np.float64)
            if limits.shape != (3,) or not np.all(np.isfinite(limits)) or np.any(limits < 0):
                raise ValueError("ROTATION_LIMITS_RAD must contain three finite non-negative values")
            relative = (Rotation.from_quat(pose[3:]) * reference_rotation.inv()).as_rotvec()
            relative = np.clip(relative, -limits, limits)
            pose[3:] = (Rotation.from_rotvec(relative) * reference_rotation).as_quat()
        elif self.pose_rotation_repr == "euler":
            euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
            # Clip first euler angle separately due to discontinuity from pi to -pi
            sign = np.sign(euler[0])
            euler[0] = sign * (
                np.clip(
                    np.abs(euler[0]),
                    self.rot_bounding_box.low[0],
                    self.rot_bounding_box.high[0],
                )
            )
            euler[1:] = np.clip(
                euler[1:], self.rot_bounding_box.low[1:], self.rot_bounding_box.high[1:]
            )
            pose[3:] = Rotation.from_euler("xyz", euler).as_quat()
        else:
            rotvec = Rotation.from_quat(pose[3:]).as_rotvec()
            rotvec = np.clip(rotvec, self.rot_bounding_box.low, self.rot_bounding_box.high)
            pose[3:] = Rotation.from_rotvec(rotvec).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """standard gym step function."""
        start_time = time.time()
        if getattr(self, "_safety_stop", False):
            raise RuntimeError("Robot safety stop is latched; acknowledge it on the robot server before continuing")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError(f"Expected finite action shape {self.action_space.shape}, got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 插值分配给每个子步的物理预算之和为一个策略步的预算。
        weight = float(getattr(self, "_policy_step_weight", 1.0))
        if not np.isfinite(weight) or weight <= 0 or weight > 1:
            raise ValueError("_policy_step_weight must be in (0, 1]")
        config = getattr(self, "config", None)
        if hasattr(self, "xyz_bounding_box") and (
            np.any(self.currpos[:3] < self.xyz_bounding_box.low - 1e-6)
            or np.any(self.currpos[:3] > self.xyz_bounding_box.high + 1e-6)
        ):
            # 不能用“裁剪到边界”把一个已越界的当前位置突然拉回很远的位置。
            raise RuntimeError("Current TCP position is outside the workspace; reposition manually before policy control")
        xyz_delta = action[:3].astype(np.float64) * self.action_scale[0]
        max_translation = getattr(config, "MAX_TRANSLATION_STEP_M", None)
        if max_translation is not None:
            if not np.isfinite(max_translation) or max_translation <= 0:
                raise ValueError("MAX_TRANSLATION_STEP_M must be finite and positive")
            norm = np.linalg.norm(xyz_delta)
            if norm > max_translation * weight:
                xyz_delta *= max_translation * weight / norm

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta

        rot_delta = action[3:6] * self.action_scale[1]
        if self.action_rotation_repr == "euler":
            delta_rotation = Rotation.from_euler("xyz", rot_delta)
        elif self.action_rotation_repr == "rotvec":
            delta_rotation = Rotation.from_rotvec(rot_delta)
        else:
            raise ValueError(
                f"Unsupported ACTION_ROTATION_REPR={self.action_rotation_repr!r}. "
                "Expected 'euler' or 'rotvec'."
            )
        max_rotation = getattr(config, "MAX_ROTATION_STEP_RAD", None)
        if max_rotation is not None:
            if not np.isfinite(max_rotation) or max_rotation <= 0:
                raise ValueError("MAX_ROTATION_STEP_RAD must be finite and positive")
            # 在物理旋转角上限幅，而不是把 MRP/欧拉角的各分量当作角度范数。
            rotation_vector = delta_rotation.as_rotvec()
            norm = np.linalg.norm(rotation_vector)
            if norm > max_rotation * weight:
                rotation_vector *= max_rotation * weight / norm
            delta_rotation = Rotation.from_rotvec(rotation_vector)
        self.nextpos[3:] = (delta_rotation * Rotation.from_quat(self.currpos[3:])).as_quat()
        requested_orientation = self.nextpos[3:].copy()
        self.nextpos = self.clip_safety_box(self.nextpos)
        applied_action = action.copy()
        if self.action_scale[0] > 0:
            applied_action[:3] = (self.nextpos[:3] - self.currpos[:3]) / self.action_scale[0]
        if self.action_scale[1] > 0 and (max_rotation is not None or not np.allclose(requested_orientation, self.nextpos[3:])):
            applied_rotation = Rotation.from_quat(self.nextpos[3:]) * Rotation.from_quat(self.currpos[3:]).inv()
            if max_rotation is not None:
                # 对最终目标再核验一次；旋转工作区投影也不能突破单步物理角度预算。
                applied_vector = applied_rotation.as_rotvec()
                applied_norm = np.linalg.norm(applied_vector)
                if applied_norm > max_rotation * weight + 1e-10:
                    # 当前姿态不在允许旋转域内时拒绝继续，不用一次大角度动作拉回。
                    raise RuntimeError("Workspace rotation correction exceeds this step's rotation budget")
            applied_action[3:6] = (applied_rotation.as_rotvec() if self.action_rotation_repr == "rotvec" else applied_rotation.as_euler("xyz")) / self.action_scale[1]
        if getattr(self, "gripper_enabled", self.action_space.shape == (7,)):
            self._send_gripper_command(action[6] * self.action_scale[2])
        self._send_pos_command(self.nextpos, source="policy")

        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        self._update_currpos()
        defer_images = getattr(self, "_defer_step_images", False)
        ob = self._get_obs(refresh_images=not defer_images)
        reward = self.compute_reward(ob)
        terminated = bool(reward or self.terminate)
        truncated = bool(self.curr_path_length >= self.max_episode_length and not terminated)
        if defer_images and (terminated or truncated):
            ob = self._get_obs()
        return ob, int(reward), terminated, truncated, {"succeed": bool(reward), "applied_action": applied_action}

    def compute_reward(self, obs) -> bool:
        # An outer learned reward wrapper owns success during classifier-based RL.
        if not getattr(self, "use_pose_reward", True):
            return False
        current_pose = obs["state"]["tcp_pose"]
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        if self.pose_rotation_repr == "euler":
            target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
            diff_rot = current_rot.T  @ target_rot
            diff_rot_repr = Rotation.from_matrix(diff_rot).as_euler("xyz")
        else:
            target_rot = Rotation.from_rotvec(self._TARGET_POSE[3:]).as_matrix()
            diff_rot = current_rot.T  @ target_rot
            diff_rot_repr = Rotation.from_matrix(diff_rot).as_rotvec()
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_rot_repr]))
        # print(f"Delta: {delta}")
        if np.all(delta < self._REWARD_THRESHOLD):
            return True
        else:
            # print(f'Goal not reached, the difference is {delta}, the desired threshold is {self._REWARD_THRESHOLD}')
            return False

    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                images[key] = resized[..., ::-1]
                display_images[key] = rgb
                display_images[key + "_full"] = cropped_rgb
                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.camera_config)
                return self.get_im()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            try:
                self.img_queue.put_nowait(display_images)
            except queue.Full:
                try:
                    self.img_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.img_queue.put_nowait(display_images)
                except queue.Full:
                    pass
        return images

    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = self._pose6_to_quat(goal)
        steps = int(timeout * self.hz)
        self._update_currpos()
        path = np.linspace(self.currpos, goal, steps)
        for p in path:
            self._send_pos_command(p, source="reset")
            time.sleep(1 / self.hz)
        self.nextpos = p
        self._update_currpos()

    def go_to_reset(self, joint_reset=False):
        """
        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.
        """
        # Change to precision mode for reset        # Use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos, source="reset")
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # Perform joint reset if needed
        if joint_reset:
            LOGGER.info("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Perform Carteasian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            rz_noise = np.random.uniform(-self.random_rz_range, self.random_rz_range)
            noise_rot = Rotation.from_rotvec(np.array([0.0, 0.0, rz_noise], dtype=np.float64))
            base_rot = Rotation.from_quat(self.resetpos[3:])
            reset_pose[3:] = (noise_rot * base_rot).as_quat()
            self.interpolate_move(reset_pose, timeout=1)
        else:
            reset_pose = self.resetpos.copy()
            self.interpolate_move(reset_pose, timeout=1)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle!=0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    LOGGER.info("Saved video for camera %s at %s", camera_key, video_path)
                
            self.recording_frames.clear()
        except Exception as e:
            LOGGER.exception("Failed to save video: %s", e)

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            if "video_device" in kwargs:
                raw_cap = V4LCapture(name=cam_name, **kwargs)
            else:
                raw_cap = RSCapture(name=cam_name, **kwargs)
            cap = VideoCapture(raw_cap)
            self.cap[cam_name] = cap

    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            LOGGER.exception("Failed to close cameras: %s", e)

    def _recover(self):
        """Internal function to recover the robot from error state."""
        # 先读取安全锁存；自动 reset 不允许悄悄清除超力等保护状态。
        self._update_currpos()
        if getattr(self, "_safety_stop", False):
            raise RuntimeError("Safety stop requires explicit server acknowledgement")
        response = requests.post(self.url + "clearerr", timeout=5.0)
        response.raise_for_status()

    def _send_pos_command(self, pos: np.ndarray, source: str = "policy"):
        """Internal function to send position command to the robot."""
        arr = np.array(pos).astype(np.float32)
        data = {"arr": arr.tolist(), "source": source}
        response = requests.post(self.url + "pose", json=data, timeout=5.0)
        # 服务端拒绝越界/超力命令时必须中断，不能把未执行的命令写入经验池。
        response.raise_for_status()

    def _send_gripper_command(self, pos: float, mode="binary"):
        """Internal function to send gripper command to the robot."""
        if not self.gripper_enabled:
            return
        if mode == "binary":
            if (pos <= -0.5) and (self.curr_gripper_pos > 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # close gripper
                requests.post(self.url + "close_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            elif (pos >= 0.5) and (self.curr_gripper_pos < 0.85) and (time.time() - self.last_gripper_act > self.gripper_sleep):  # open gripper
                requests.post(self.url + "open_gripper")
                self.last_gripper_act = time.time()
                time.sleep(self.gripper_sleep)
            else: 
                return
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _normalize_pose(self, pose) -> np.ndarray:
        pose = np.array(pose, dtype=np.float32).reshape(-1)
        if pose.size == 7:
            return pose
        if pose.size == 6:
            quat = Rotation.from_rotvec(pose[3:]).as_quat().astype(np.float32)
            return np.concatenate([pose[:3], quat])
        raise ValueError(f"Unsupported pose shape from server: {pose.shape}")

    def _pose6_to_quat(self, pose) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.size != 6:
            raise ValueError(f"Expected pose shape (6,), got {pose.shape}")
        if self.pose_rotation_repr == "euler":
            quat = euler_2_quat(pose[3:])
        else:
            quat = Rotation.from_rotvec(pose[3:]).as_quat()
        return np.concatenate([pose[:3], quat])

    def _normalize_jacobian(self, jacobian) -> np.ndarray:
        jacobian = np.array(jacobian, dtype=np.float32).reshape(-1)
        if jacobian.size == 0:
            return np.zeros((6, 6), dtype=np.float32)
        if jacobian.size % 6 != 0:
            return np.zeros((6, 6), dtype=np.float32)
        return jacobian.reshape(6, jacobian.size // 6)

    def _parse_state_response(self, ps: dict):
        if ps.get("safety_stop"):
            self._safety_stop = True
            self.terminate = True
            reason = ps.get("safety_reason") or "server did not report a reason; check the robot server log"
            raise RuntimeError(f"Robot safety stop: {reason}")
        pose = ps.get("pose", np.zeros((7,), dtype=np.float32))
        self.currpos = self._normalize_pose(pose)
        self.currvel = np.array(ps.get("vel", np.zeros((6,))), dtype=np.float32).reshape(-1)
        self.currforce = np.array(ps.get("force", np.zeros((3,))), dtype=np.float32).reshape(-1)
        self.currtorque = np.array(ps.get("torque", np.zeros((3,))), dtype=np.float32).reshape(-1)
        self.currjacobian = self._normalize_jacobian(ps.get("jacobian", np.zeros((36,))))
        self.q = np.array(ps.get("q", np.zeros((6,))), dtype=np.float32).reshape(-1)
        self.dq = np.array(ps.get("dq", np.zeros((6,))), dtype=np.float32).reshape(-1)
        gripper_pos = np.array(ps.get("gripper_pos", [0.0]), dtype=np.float32).reshape(-1)
        if gripper_pos.size == 0:
            gripper_pos = np.zeros((1,), dtype=np.float32)
        self.curr_gripper_pos = gripper_pos[:1]

    def _init_fake_state(self):
        self.currpos = self.resetpos.copy()
        self.currvel = np.zeros((6,), dtype=np.float32)
        self.currforce = np.zeros((3,), dtype=np.float32)
        self.currtorque = np.zeros((3,), dtype=np.float32)
        self.currjacobian = np.zeros((6, 6), dtype=np.float32)
        self.q = np.zeros((6,), dtype=np.float32)
        self.dq = np.zeros((6,), dtype=np.float32)
        self.curr_gripper_pos = np.zeros((1,), dtype=np.float32)

    def _update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        response = requests.post(self.url + "getstate", timeout=5.0)
        response.raise_for_status()
        ps = response.json()
        self._parse_state_response(ps)

    def update_currpos(self):
        """
        Internal function to get the latest state of the robot and its gripper.
        """
        ps = requests.post(self.url + "getstate").json()
        self._parse_state_response(ps)

    def _get_obs(self, refresh_images=True) -> dict:
        # 低帧率相机只在策略步末刷新；中间子步仍读取实时机器人状态。
        if refresh_images or not hasattr(self, "_cached_obs_images"):
            self._cached_obs_images = self.get_im()
        images = self._cached_obs_images
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self):
        if hasattr(self, 'listener'):
            self.listener.stop()
        self.close_cameras()
        if self.display_image:
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()
