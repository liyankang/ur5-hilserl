from logging import exception
import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
from ur_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import requests
from scipy.spatial.transform import Rotation as R
from ur_env.envs.ur_env import urEnv
from typing import List
from ur_env.utils.terminal_keyboard import TERMINAL_KEYBOARD_HUB


sigmoid = lambda x: 1 / (1 + np.exp(-x))


class ActionInterpolationWrapper(gym.Wrapper):
    """
    Wrapper to interpolate actions between policy steps to match robot control frequency.
    
    This wrapper smooths out the motion by interpolating between consecutive actions
    when the policy frequency is lower than the robot control frequency.
    
    Args:
        env: The environment to wrap
        policy_freq: Frequency of policy updates (default: 10 Hz)
        control_freq: Frequency of robot control (default: 50 Hz)
        interpolation_type: Type of interpolation ('linear' or 'spline', default: 'linear')
    """
    
    def __init__(self, env, policy_freq=10, control_freq=50, interpolation_type='linear'):
        super().__init__(env)
        self.policy_freq = policy_freq
        self.control_freq = control_freq
        self.interpolation_type = interpolation_type
        self.interpolation_steps = control_freq // policy_freq
        
        # Initialize action buffers
        self.prev_action = None
        self.current_action = None
        self.step_count = 0
        self.action_space = env.action_space
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = np.zeros(self.action_space.shape)
        self.current_action = np.zeros(self.action_space.shape)
        self.step_count = 0
        return obs, info
    
    def step(self, action):
        # Update current action when starting a new interpolation cycle
        if self.step_count == 0:
            self.prev_action = self.current_action
            self.current_action = action
        
        # Linear interpolation between prev_action and current_action
        alpha = (self.step_count + 1) / self.interpolation_steps
        
        if self.interpolation_type == 'linear':
            interpolated_action = (1 - alpha) * self.prev_action + alpha * self.current_action
        else:
            # For more advanced interpolation, could implement spline here
            interpolated_action = (1 - alpha) * self.prev_action + alpha * self.current_action
        
        # Increment step counter and wrap around
        self.step_count = (self.step_count + 1) % self.interpolation_steps
        
        return self.env.step(interpolated_action)

class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = done or rew
        info['succeed'] = bool(rew)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
    
class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received)) # either environment done or all rewards satisfied
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class Quat2MrpWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to MRP.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + mrp
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_mrp())
        )
        return observation


class MrpToRotvecActionWrapper(gym.ActionWrapper):
    """
    Convert policy rotational actions from MRP parameterization into rotvec
    while preserving the existing action scaling in the wrapped env.
    """

    def __init__(self, env: Env, rotation_action_scale: float):
        super().__init__(env)
        if rotation_action_scale <= 0:
            raise ValueError("rotation_action_scale must be positive")
        self.rotation_action_scale = float(rotation_action_scale)

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        # The wrapped env applies `rot = action[3:6] * rotation_action_scale`.
        # We map MRP->rotvec on the scaled quantity so downstream scaling stays valid.
        scaled_mrp = action[3:6] * self.rotation_action_scale
        scaled_rotvec = R.from_mrp(scaled_mrp).as_rotvec().astype(np.float32)
        action[3:6] = scaled_rotvec / self.rotation_action_scale
        return action

    def _rotvec_action_to_mrp_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        scaled_rotvec = action[3:6] * self.rotation_action_scale
        scaled_mrp = R.from_rotvec(scaled_rotvec).as_mrp().astype(np.float32)
        action[3:6] = scaled_mrp / self.rotation_action_scale
        return action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = self._rotvec_action_to_mrp_action(info["intervene_action"])
        return obs, rew, done, truncated, info


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    
class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)
        intervened = False
        
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)
            expert_a[:6] += np.random.uniform(-0.5, 0.5, size=6)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            return expert_a, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

class DualSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled

        self.expert = SpaceMouseExpert()
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        intervened = False
        expert_a, buttons = self.expert.get_action()
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons)


        if self.gripper_enabled:
            if self.left1:  # close gripper
                left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.left2:  # open gripper
                left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                left_gripper_action = np.zeros((1,))

            if self.right1:  # close gripper
                right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right2:  # open gripper
                right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                right_gripper_action = np.zeros((1,))
            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        else:
            return reward

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info

class DualGripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
        self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info


class KeyBoardIntervention(gym.ActionWrapper):
    """
    Keyboard-based intervention for demo collection without a SpaceMouse.
    """

    def __init__(self, env, action_indices=None, action_magnitude=0.6):
        super().__init__(env)

        self.gripper_enabled = self.action_space.shape != (6,)
        self.action_indices = action_indices
        self.action_magnitude = action_magnitude
        self.left, self.right = False, False
        self.intervened = False
        self.gripper_state = "close"

        self.axis_bindings = {
            "a": (0, 1.0),
            "d": (0, -1.0),
            "w": (1, 1.0),
            "s": (1, -1.0),
            "j": (2, 1.0),
            "k": (2, -1.0),
        }
        self.pending_action = np.zeros((6,), dtype=np.float32)
        self._gripper_toggle_requested = False
        self._unsubscribe = None
        self._last_toggle_time = 0.0
        self._toggle_debounce_s = 0.25

        print(
            "Keyboard intervention enabled: "
            "press W/S/A/D/J/K for one-step XYZ nudges, ; to toggle intervention"
            + (", L to toggle gripper" if self.gripper_enabled else "")
        )

        self._unsubscribe = TERMINAL_KEYBOARD_HUB.register(self._on_key)

    def _on_key(self, key_char: str):
        key_str = key_char.lower()
        if key_str in self.axis_bindings:
            axis, direction = self.axis_bindings[key_str]
            self.pending_action[axis] += direction
        elif key_str == ";":
            now = time.monotonic()
            if now - self._last_toggle_time >= self._toggle_debounce_s:
                self.intervened = not self.intervened
                self._last_toggle_time = now
                print(f"Keyboard intervention toggled: {self.intervened}")
        elif self.gripper_enabled and key_str == "l":
            self._gripper_toggle_requested = True

    def _keyboard_action(self) -> np.ndarray:
        action_dim = 7 if self.gripper_enabled else 6
        expert_a = np.zeros((action_dim,), dtype=np.float32)

        if np.linalg.norm(self.pending_action) > 0:
            expert_a[:6] = np.clip(self.pending_action, -1.0, 1.0) * self.action_magnitude
            self.pending_action[:] = 0.0

        if self.gripper_enabled:
            if self._gripper_toggle_requested and self.gripper_state == "open":
                expert_a[6] = -1.0
                self.left = True
                self.right = False
                self.gripper_state = "close"
            elif self._gripper_toggle_requested and self.gripper_state == "close":
                expert_a[6] = 1.0
                self.left = False
                self.right = True
                self.gripper_state = "open"
            else:
                self.left = False
                self.right = False

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        self._gripper_toggle_requested = False
        return expert_a

    def action(self, action: np.ndarray) -> np.ndarray:
        expert_a = self._keyboard_action()
        # When intervention mode is on, always block policy actions.
        # If no keyboard input is pending, this returns zero action (hold still).
        if self.intervened:
            return expert_a, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._gripper_toggle_requested = False
        self.pending_action[:] = 0.0
        self._last_toggle_time = 0.0
        return obs, info

    def close(self):
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        return super().close()


try:
    import glfw
except Exception:  # pragma: no cover - optional legacy dependency
    glfw = None

class KeyBoardIntervention2(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        if glfw is None:
            raise ImportError("glfw is required for KeyBoardIntervention2")

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.left, self.right = False, False
        self.action_indices = action_indices

        self.gripper_state = 'close'
        self.intervened = False
        self.action_length = 0.3
        self.current_action = np.array([0, 0, 0, 0, 0, 0])  # 分别对应 W, A, S, D 的状态
        self.flag = False
        self.key_states = {
            'w': False,
            'a': False,
            's': False,
            'd': False,
            'j': False,
            'k': False,
            'l': False,
            ';': False,
        }

        # 延迟设置回调，不在 __init__ 中直接访问 _viewer
        self._callback_set = False
        self._window = None

    def _try_set_key_callback(self):
        """尝试获取渲染窗口并设置键盘回调（仅在尚未设置且窗口存在时执行）"""
        if self._callback_set:
            return

        window = None
        # 尝试多种方式获取窗口，兼容新旧 gymnasium 和 mujoco-py
        try:
            # 新 gymnasium 渲染器 (>=0.26)
            if hasattr(self.env, 'renderer') and self.env.renderer is not None:
                window = self.env.renderer.window
            # 旧 mujoco-py 风格
            elif hasattr(self.env, '_viewer') and hasattr(self.env._viewer, 'viewer'):
                window = self.env._viewer.viewer.window
            # 尝试从 unwrapped 中获取（有时私有属性可通过 unwrapped 绕过）
            elif hasattr(self.env, 'unwrapped'):
                unwrapped = self.env.unwrapped
                if hasattr(unwrapped, 'renderer') and unwrapped.renderer is not None:
                    window = unwrapped.renderer.window
                elif hasattr(unwrapped, '_viewer') and hasattr(unwrapped._viewer, 'viewer'):
                    window = unwrapped._viewer.viewer.window
        except AttributeError:
            # 访问被保护，忽略并继续尝试其他方法
            pass

        if window is not None:
            glfw.set_key_callback(window, self.glfw_on_key)
            self._callback_set = True
            self._window = window
        else:
            # 无法获取窗口，键盘干预暂时失效（后续 reset/step 会继续尝试）
            # 可选：打印一次警告（避免刷屏）
            if not hasattr(self, '_warned'):
                print("Warning: Could not set key callback, keyboard intervention disabled until renderer is ready")
                self._warned = True

    def glfw_on_key(self, window, key, scancode, action, mods):
        # 原回调实现保持不变
        if action == glfw.PRESS:
            if key == glfw.KEY_W:
                self.key_states['w'] = True
            elif key == glfw.KEY_A:
                self.key_states['a'] = True
            elif key == glfw.KEY_S:
                self.key_states['s'] = True
            elif key == glfw.KEY_D:
                self.key_states['d'] = True
            elif key == glfw.KEY_J:
                self.key_states['j'] = True
            elif key == glfw.KEY_K:
                self.key_states['k'] = True
            elif key == glfw.KEY_L:
                self.key_states['l'] = True
                self.flag = True
            elif key == glfw.KEY_SEMICOLON:
                self.intervened = not self.intervened
                self.env.intervened = self.intervened
                print(f"Intervention toggled: {self.intervened}")

        elif action == glfw.RELEASE:
            if key == glfw.KEY_W:
                self.key_states['w'] = False
            elif key == glfw.KEY_A:
                self.key_states['a'] = False
            elif key == glfw.KEY_S:
                self.key_states['s'] = False
            elif key == glfw.KEY_D:
                self.key_states['d'] = False
            elif key == glfw.KEY_J:
                self.key_states['j'] = False
            elif key == glfw.KEY_K:
                self.key_states['k'] = False
            elif key == glfw.KEY_L:
                self.key_states['l'] = False

        self.current_action = [
            int(self.key_states['w']) - int(self.key_states['s']), 
            int(self.key_states['a']) - int(self.key_states['d']), 
            int(self.key_states['j']) - int(self.key_states['k']),  
            0,
            0,
            0,
        ]
        self.current_action = np.array(self.current_action, dtype=np.float64)
        self.current_action *= self.action_length

    def action(self, action: np.ndarray) -> np.ndarray:
        expert_a = self.current_action.copy()

        if self.gripper_enabled:
            if self.flag and self.gripper_state == 'open':  # close gripper
                self.gripper_state = 'close'
                self.flag = False
            elif self.flag and self.gripper_state == 'close':  # open gripper
                self.gripper_state = 'open'
                self.flag = False

            # 根据夹爪状态生成动作
            if self.gripper_state == 'close':
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
            else:
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if self.intervened:
            return expert_a, True
        else:
            return action, False

    def step(self, action):
        # 每次 step 都尝试设置回调（如果尚未设置）
        self._try_set_key_callback()

        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.gripper_state = 'open'
        self._try_set_key_callback()  # reset 时尝试设置回调
        return obs, info



KeyboardIntervention = KeyBoardIntervention
