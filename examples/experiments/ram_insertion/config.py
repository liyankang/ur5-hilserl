import os
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from ur_env.envs.wrappers import (
    Quat2MrpWrapper,
    MrpToRotvecActionWrapper,
    KeyboardIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv,
    ActionInterpolationWrapper
)
from ur_env.envs.relative_env import RelativeFrame
from ur_env.envs.ur_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.ram_insertion.wrapper import RAMEnv


def _flattened_state_index(proprio_keys, field_name, field_offset, field_shapes):
    """Derive a stable flattened index using the same Dict flattening logic as SERLObsWrapper."""
    proprio_space = gym.spaces.Dict(
        {key: gym.spaces.Box(-np.inf, np.inf, shape=field_shapes[key]) for key in proprio_keys}
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
    # Current UR5 RAM-insertion deployment assumes no physical gripper.
    SERVER_URL = "http://127.0.0.1:5000/"
    CAMERAS = {
        # USB 相机只需要写摄像头 ID，一般 0 或 1，不需要序列号
        "global_1": {
            "video_device": "/dev/video2",
            "dim": (1280, 720),
        },
        "wrist": {
            "video_device": "/dev/video1",
            "dim": (640, 480),
        },
        "global_2": {
            "video_device": "/dev/video4",
            "dim": (1280, 720),
        },
    }

    REALSENSE_CAMERAS = CAMERAS

    IMAGE_CROP = {
        "global_1": lambda img: img[191:408, 547:753],
        "wrist": lambda img: img[130:480, 159:409],
        "global_2": lambda img: img[425:653, 421:555],

    }
    # Pose convention in this config: xyz + rotvec (UR RTDE native orientation).
    TARGET_POSE = np.array([0.0849, -0.6295, -0.2698, 2.8370, 1.1641, 0.1422])
    GRASP_POSE = np.array([0.0849, -0.6295, -0.2698, 2.8370, 1.1641, 0.1422])
    # Sparse task reward: reward=1 when TCP pose (xyz + rotvec) reaches TARGET_POSE within tolerance.
    REWARD_THRESHOLD = np.array([0.005, 0.005, 0.005, 0.02, 0.02, 0.02], dtype=np.float64)
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.1, 0, 0, 0])
    # Keep translation bounds tight, no rotation allowed.
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.05, 0.05, 0.02, 0, 0, 0])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.05, 0.05, 0.15, 0, 0, 0])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.02
    ACTION_SCALE = (0.003, 0.001, 1)
    ACTION_ROTATION_REPR = "rotvec"
    POSE_ROTATION_REPR = "rotvec"
    DISPLAY_IMAGE = True
    GRIPPER_SLEEP = 0.0
    GRIPPER_ENABLED = False
    MAX_EPISODE_LENGTH = 200
    # Keep classifier integration available in code, but disabled for RAM sparse target reward mode.
    ENABLE_CLASSIFIER_REWARD = False

    RESET_INTERPOLATE_TIMEOUT =2.0
    # 空载（无法兰夹具）
    # 安装夹具后，请用秤称质量，并运行 height.py 标定重心，然后更新以下参数：
    LOAD_PARAM = {
        "mass": 1.0,
        "F_x_center_load": [0.0, 0.0, 0.07],
    }
    # 解释：合规性参数，用于控制机器人在执行任务时的合规性
    COMPLIANCE_PARAM = {
        "translational_stiffness": 1200,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    #解释：精度参数，用于控制机器人在执行任务时的精度和稳定性
    PRECISION_PARAM = {
        "translational_stiffness": 1200,
        "translational_damping": 89,
        "rotational_stiffness": 100,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["global_1", "wrist", "global_2"]
    classifier_keys = ["wrist", "global_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    tcp_pose_z_index = _flattened_state_index(
        proprio_keys,
        field_name="tcp_pose",
        field_offset=2,
        field_shapes=STATE_FIELD_SHAPES,
    )
    buffer_period = 1000
    checkpoint_period = 5000
    training_starts = 200
    actor_update_period = 2
    steps_per_update = 10
    batch_size = 128  # 减小 batch 加速训练
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = RAMEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            env = KeyboardIntervention(env)
        env = RelativeFrame(env)
        env = MrpToRotvecActionWrapper(env, rotation_action_scale=EnvConfig.ACTION_SCALE[1])
        env = Quat2MrpWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # Apply action interpolation to smooth motion between policy steps
        env = ActionInterpolationWrapper(env, policy_freq=10, control_freq=50)
        if classifier and EnvConfig.ENABLE_CLASSIFIER_REWARD:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )
            
            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                return float(sigmoid(classifier(obs)))

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func, reward_threshold=0.95)
        return env
