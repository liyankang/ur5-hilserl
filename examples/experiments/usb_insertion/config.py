"""UR5 USB insertion configuration.

The USB plug is fixed to the tool, so every target uses the same orientation.
Only the insertion point is translated in the XY plane.  The candidate set is
kept explicit so demonstrations can be balanced across positions and the
workspace can be checked before the robot is enabled.
"""

import os

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from ur_env.envs.ur_env import DefaultEnvConfig
from ur_env.envs.wrappers import (
    ActionInterpolationWrapper,
    GripperCloseEnv,
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

    POLICY_HZ = 20
    CONTROL_HZ = 100
    SERVER_URL = "http://127.0.0.1:5000/"

    CAMERAS = {
        "global_1": {"video_device": "/dev/video2", "dim": (1280, 720)},
        "wrist": {"video_device": "/dev/video4", "dim": (1280, 720)},
        "global_2": {"video_device": "/dev/video0", "dim": (1280, 720)},
    }
    REALSENSE_CAMERAS = CAMERAS
    IMAGE_CROP = {
        "global_1": lambda img: img[79:550, 195:836],
        "wrist": lambda img: img[195:720, 358:977],
        "global_2": lambda img: img[410:656, 354:649],
    }

    # Replace this calibrated point with the USB socket pose on the real rig.
    # The final three values are the fixed plug insertion orientation.
    BASE_TARGET_POSE = np.array(
        [0.1092, -0.6608, -0.2163, 2.9348, -1.0839, 0.0372],
        dtype=np.float64,
    )
    TARGET_TRANSLATION_OFFSETS = np.array(
        [
            [-0.020, -0.020, 0.0],
            [-0.020, 0.000, 0.0],
            [-0.020, 0.020, 0.0],
            [0.000, -0.020, 0.0],
            [0.000, 0.000, 0.0],
            [0.000, 0.020, 0.0],
            [0.020, -0.020, 0.0],
            [0.020, 0.000, 0.0],
            [0.020, 0.020, 0.0],
        ],
        dtype=np.float64,
    )
    TARGET_POSES = build_target_poses(BASE_TARGET_POSE, TARGET_TRANSLATION_OFFSETS)
    # Keep the original single-target API as the default/reference pose.
    TARGET_POSE = BASE_TARGET_POSE.copy()
    TARGET_SELECTION = "random"

    RESET_TARGET_OFFSET = np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0])
    RESET_POSE = TARGET_POSE + RESET_TARGET_OFFSET
    RESET_POSES = TARGET_POSES + RESET_TARGET_OFFSET
    GRASP_POSE = TARGET_POSE.copy()
    REWARD_THRESHOLD = np.array(
        [0.005, 0.005, 0.005, 0.02, 0.02, 0.02], dtype=np.float64
    )

    # Bounds cover every translated target and the XY reset perturbation.
    # Orientation is intentionally tight because this task does not randomize
    # the plug angle.
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.03
    # Keep reset orientation identical across positions; only translation is
    # randomized for this task.
    RANDOM_RZ_RANGE = 0.0
    _translation_extent = np.max(np.abs(TARGET_TRANSLATION_OFFSETS), axis=0)
    _position_extent = _translation_extent + np.array(
        [RANDOM_XY_RANGE, RANDOM_XY_RANGE, 0.0]
    )
    ABS_POSE_LIMIT_LOW = BASE_TARGET_POSE - np.array(
        [
            _position_extent[0],
            _position_extent[1],
            _position_extent[2] + 0.02,
            0.01,
            0.01,
            0.01,
        ]
    )
    ABS_POSE_LIMIT_HIGH = BASE_TARGET_POSE + np.array(
        [
            _position_extent[0],
            _position_extent[1],
            _position_extent[2] + 0.15,
            0.01,
            0.01,
            0.01,
        ]
    )

    ACTION_SCALE = (0.01, 0.01, 1.0)
    ACTION_ROTATION_REPR = "rotvec"
    POSE_ROTATION_REPR = "rotvec"
    DISPLAY_IMAGE = True
    DISPLAY_IMAGE_SIZE = (500, 500)
    GRIPPER_SLEEP = 0.0
    GRIPPER_ENABLED = False
    MAX_EPISODE_LENGTH = 100
    ENABLE_CLASSIFIER_REWARD = False
    RESET_INTERPOLATE_TIMEOUT = 2.0

class TrainConfig(DefaultTrainingConfig):
    image_keys = ["global_1", "wrist", "global_2"]
    classifier_keys = ["wrist", "global_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    tcp_pose_z_index = _flattened_state_index(
        proprio_keys, "tcp_pose", 2, STATE_FIELD_SHAPES
    )
    buffer_period = 1000
    checkpoint_period = 5000
    training_starts = 200
    actor_update_period = 2
    steps_per_update = 10
    batch_size = 64
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = USBInsertionEnv(
            hz=EnvConfig.CONTROL_HZ,
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            env = KeyboardIntervention(env)
        env = RelativeFrame(env)
        env = MrpToRotvecActionWrapper(
            env, rotation_action_scale=EnvConfig.ACTION_SCALE[1]
        )
        env = Quat2MrpWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        env = ActionInterpolationWrapper(
            env,
            policy_freq=EnvConfig.POLICY_HZ,
            control_freq=EnvConfig.CONTROL_HZ,
        )

        if classifier and EnvConfig.ENABLE_CLASSIFIER_REWARD:
            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return float(sigmoid(classifier_fn(obs)))

            env = MultiCameraBinaryRewardClassifierWrapper(
                env, reward_func, reward_threshold=0.95
            )
        return env
