"""UR5 固定 U 盘插入任务：固定目标、受限六维增量动作和随机初始位姿。

所有现场坐标、动作上限和随机范围均从同目录 train_config.json 加载。
"""

import os

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from ur_env.envs.ur_env import DefaultEnvConfig
from ur_env.envs.wrappers import (
    ActionInterpolationWrapper,
    KeyboardIntervention,
    MrpToRotvecActionWrapper,
    MultiCameraBinaryRewardClassifierWrapper,
    Quat2MrpWrapper,
)
from ur_env.envs.relative_env import RelativeFrame
from serl_launcher.networks.reward_classifier import load_classifier_func
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from experiments.config import DefaultTrainingConfig
from experiments.usb_insertion.wrapper import USBInsertionEnv
from experiments.usb_insertion.runtime_config import (
    load_config, resolve_path, require_hardware_ready, controller_safety_payload,
)


def build_target_poses(base_pose: np.ndarray, translation_offsets: np.ndarray) -> np.ndarray:
    """Build [x, y, z, rx, ry, rz] targets with one shared orientation."""
    base_pose = np.asarray(base_pose, dtype=np.float64).reshape(-1)
    offsets = np.asarray(translation_offsets, dtype=np.float64)
    if base_pose.shape != (6,):
        raise ValueError(f"base_pose must have shape (6,), got {base_pose.shape}")
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError(
            f"translation_offsets must have shape (N, 3), got {offsets.shape}"
        )
    if offsets.shape[0] == 0:
        raise ValueError("translation_offsets must contain at least one target")

    targets = np.repeat(base_pose[None, :], offsets.shape[0], axis=0)
    targets[:, :3] += offsets
    return targets


def _flattened_state_index(proprio_keys, field_name, field_offset, field_shapes):
    """Return the flattened index used by SERLObsWrapper for one state value."""
    proprio_space = gym.spaces.Dict(
        {
            key: gym.spaces.Box(-np.inf, np.inf, shape=field_shapes[key])
            for key in proprio_keys
        }
    )
    marker_value = np.float32(12345.0)
    obs = {
        key: np.zeros(proprio_space[key].shape, dtype=np.float32)
        for key in proprio_space.spaces
    }
    obs[field_name][field_offset] = marker_value
    flat_obs = gym.spaces.flatten(proprio_space, obs)
    matches = np.where(flat_obs == marker_value)[0]
    if matches.size != 1:
        raise ValueError(f"Could not derive flattened index for {field_name}[{field_offset}]")
    return int(matches[0])


STATE_FIELD_SHAPES = {
    "tcp_pose": (6,),
    "tcp_vel": (6,),
    "tcp_force": (3,),
    "tcp_torque": (3,),
    "gripper_pose": (1,),
}


class EnvConfig(DefaultEnvConfig):
    """Hardware and task parameters for USB insertion on the UR5."""

    def __init__(self, task_config=None):
        # JSON 是运动参数的唯一来源；下方类属性只提供格式与显示默认值。
        settings = task_config or load_config()
        bounds, reset = settings["workspace_bounds"], settings["reset"]
        action, training = settings["action"], settings["training"]
        self.task_config = settings
        self.SERVER_URL = settings["robot"]["server_url"].rstrip("/") + "/"
        self.POLICY_HZ, self.CONTROL_HZ = training["policy_hz"], training["control_hz"]
        self.TARGET_POSE = np.asarray(settings["task"]["nominal_inserted_tcp_pose_m_rotvec"], dtype=float)
        self.BASE_TARGET_POSE = self.TARGET_POSE.copy()
        self.TARGET_POSES = self.TARGET_POSE[None, :].copy()
        self.GRASP_POSE = self.TARGET_POSE.copy()
        self.TARGET_SELECTION = "fixed"
        self.RESET_POSE = np.asarray(reset["nominal_tcp_pose_m_rotvec"], dtype=float)
        self.RESET_TARGET_OFFSET = self.RESET_POSE - self.TARGET_POSE
        self.ABS_POSE_LIMIT_LOW = np.asarray(bounds["position_low_m"] + [-np.pi] * 3)
        self.ABS_POSE_LIMIT_HIGH = np.asarray(bounds["position_high_m"] + [np.pi] * 3)
        self.ROTATION_REFERENCE_ROTVEC = np.asarray(bounds["reference_rotvec_rad"])
        self.ROTATION_LIMITS_RAD = np.deg2rad(bounds["relative_rotation_limit_deg"])
        self.MAX_TRANSLATION_STEP_M = action["max_translation_step_m"]
        self.MAX_ROTATION_STEP_RAD = np.deg2rad(action["max_rotation_step_deg"])
        # MRP 的模长是 tan(旋转角/4)，与欧拉角或 rotvec 不能直接混用。
        self.ACTION_SCALE = (self.MAX_TRANSLATION_STEP_M, float(np.tan(self.MAX_ROTATION_STEP_RAD / 4)), 0.0)
        self.GRIPPER_ENABLED = False
        self.RANDOM_RESET = reset["randomize"]
        self.RESET_TRANSLATION_RANGE_M = np.asarray(reset["translation_range_m"])
        self.RESET_ROTATION_RANGE_RAD = np.deg2rad(reset["rotation_range_deg"])
        self.INSERTION_AXIS_BASE = np.asarray(settings["task"]["insertion_axis_base"])
        self.RESET_CLEARANCE_M = reset["clearance_along_axis_m"]
        self.RESET_INTERPOLATE_TIMEOUT = reset["motion_duration_s"]
        self.RESET_SEED = reset["seed"]
        self.MAX_EPISODE_LENGTH = reset["max_episode_steps"]
        self.TASK_SAFETY_CONFIG = controller_safety_payload(settings)
        self.CAMERAS, self.IMAGE_CROP = {}, {}
        for key in settings["observation"]["image_keys"]:
            camera = dict(settings["observation"]["cameras"][key])
            y1, y2, x1, x2 = camera.pop("crop_yxyx")
            camera["dim"] = tuple(camera["dim"])
            self.CAMERAS[key] = camera
            self.IMAGE_CROP[key] = lambda image, y1=y1, y2=y2, x1=x1, x2=x2: image[y1:y2, x1:x2]
        self.REALSENSE_CAMERAS = self.CAMERAS

    # 运动参数全部由 JSON 实例加载；这里只保留格式与显示默认值。
    ACTION_ROTATION_REPR = "rotvec"
    POSE_ROTATION_REPR = "rotvec"
    DISPLAY_IMAGE = True
    DISPLAY_IMAGE_SIZE = (500, 500)
    GRIPPER_SLEEP = 0.0
    GRIPPER_ENABLED = False
    ENABLE_CLASSIFIER_REWARD = True

class TrainConfig(DefaultTrainingConfig):
    def __init__(self, config_path=None):
        self.task_config = load_config(config_path)
        self.env_config = EnvConfig(self.task_config)
        reward, training = self.task_config["reward_classifier"], self.task_config["training"]
        self.action_dim = self.task_config["action"]["action_dim"]
        self.image_keys = list(self.task_config["observation"]["image_keys"])
        self.proprio_keys = list(self.task_config["observation"]["proprio_keys"])
        self.classifier_keys = list(reward["image_keys"])
        self.classifier_use_proprio = reward["use_proprio"]
        self.classifier_success_threshold = reward["success_threshold"]
        self.classifier_checkpoint_path = str(resolve_path(reward["checkpoint_path"]))
        self.classifier_validation_path = os.path.join(self.classifier_checkpoint_path, "validation.json")
        self.checkpoint_path = str(resolve_path(training["checkpoint_path"]))
        for name in ("batch_size", "training_starts", "max_steps", "checkpoint_period"):
            setattr(self, name, training[name])
        self.tcp_pose_z_index = _flattened_state_index(self.proprio_keys, "tcp_pose", 2, STATE_FIELD_SHAPES)

    # 固定工具使用普通 SAC，不创建夹爪评价网络或夹爪惩罚。
    buffer_period = 1000
    actor_update_period = 2
    steps_per_update = 10
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        # 在打开相机、连接控制器之前拒绝未标定配置或未通过验证的模型。
        if not fake_env:
            require_hardware_ready(self.task_config, classifier=classifier)
        env_config = self.env_config
        env = USBInsertionEnv(
            hz=env_config.CONTROL_HZ,
            fake_env=fake_env,
            save_video=save_video,
            config=env_config,
        )
        # 基础环境在无夹爪模式原生输出六维，完全不附加虚构的夹爪动作。
        if env.action_space.shape != (self.action_dim,):
            env.close()
            raise ValueError("USB environment/action_dim mismatch")
        if not fake_env:
            env = KeyboardIntervention(env)
        env = RelativeFrame(env)
        env = MrpToRotvecActionWrapper(
            env, rotation_action_scale=env_config.ACTION_SCALE[1]
        )
        env = Quat2MrpWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        env = ActionInterpolationWrapper(
            env,
            policy_freq=env_config.POLICY_HZ,
            control_freq=env_config.CONTROL_HZ,
        )

        if classifier and not fake_env and EnvConfig.ENABLE_CLASSIFIER_REWARD:
            env.unwrapped.use_pose_reward = False
            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=self.classifier_checkpoint_path,
                use_proprio=self.classifier_use_proprio,
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return sigmoid(classifier_fn(obs)).item()

            env = MultiCameraBinaryRewardClassifierWrapper(
                env, reward_func, reward_threshold=self.classifier_success_threshold
            )
        return env
