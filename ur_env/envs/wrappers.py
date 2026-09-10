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
        interpolation_type: Type of interpolation ('linear', default: 'linear')
    """
    
    def __init__(
        self,
        env,
        policy_freq=10,
        control_freq=50,
        interpolation_type='linear',
        preserve_policy_horizon=True,
    ):
        super().__init__(env)
        self.policy_freq = float(policy_freq)
        self.control_freq = float(control_freq)
        self.interpolation_type = interpolation_type
        if self.policy_freq <= 0 or self.control_freq <= 0:
            raise ValueError("policy_freq and control_freq must be positive")
        if self.control_freq < self.policy_freq:
            raise ValueError("control_freq must be >= policy_freq")
        ratio = self.control_freq / self.policy_freq
        if not np.isclose(ratio, round(ratio)):
            raise ValueError(
                "control_freq / policy_freq must be an integer so substep timing is stable"
            )
        self.interpolation_steps = int(round(ratio))
        self.preserve_policy_horizon = preserve_policy_horizon
        
        # Initialize action buffer
        self.current_action = None
        self.action_space = env.action_space
        self._configure_timing()

    def _configure_timing(self):
        # Drive the underlying robot env at control_freq so each interpolation substep
        # runs at the intended high-rate control loop.
        base_env = self.unwrapped
        if hasattr(base_env, "hz"):
            base_env.hz = self.control_freq

        # Keep episode length in policy-step units (instead of shrinking by interpolation_steps).
        if self.preserve_policy_horizon and hasattr(base_env, "max_episode_length"):
            base_env.max_episode_length = int(base_env.max_episode_length) * self.interpolation_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_action = np.zeros(self.action_space.shape, dtype=np.float32)
        return obs, info

    def step(self, action):
        self.current_action = np.asarray(action, dtype=np.float32).copy()

        if self.interpolation_type != "linear":
            raise ValueError(
                f"Unsupported interpolation_type={self.interpolation_type!r}; only 'linear' is supported."
            )

        obs = None
        rew = 0.0
        done = False
        truncated = False
        merged_info = {}

        for substep in range(self.interpolation_steps):
            # Linear interpolation: velocity increases linearly from slow to fast.
            # Each sub-step k gets weight 2*(k+1) / (N*(N+1)), so the cumulative
            # sum equals 1.0 after all sub-steps.
            #   sub-step 0: 2/(N*(N+1)), sub-step 1: 4/(N*(N+1)), ..., sub-step N-1: 2N/(N*(N+1))
            # Sum = (2+4+...+2N) / (N*(N+1)) = N*(N+1) / (N*(N+1)) = 1.0 ✓
            weight = 2 * (substep + 1) / (self.interpolation_steps * (self.interpolation_steps + 1))
            sub_action = self.current_action * weight
            obs, step_rew, step_done, step_truncated, step_info = self.env.step(sub_action)
            merged_info.update(step_info)
            rew += float(step_rew)
            done = done or step_done
            truncated = truncated or step_truncated
            if done or truncated:
                break

        return obs, rew, done, truncated, merged_info

class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz=None, reward_threshold=0.5):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz
        self.reward_threshold = reward_threshold

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        prob = self.compute_reward(obs)
        info['classifier_prob'] = float(prob)
        rew = 1.0 if prob >= self.reward_threshold else 0.0
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
        if rotation_action_scale < 0:
            raise ValueError("rotation_action_scale must be non-negative")
        self.rotation_action_scale = float(rotation_action_scale)

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        if self.rotation_action_scale <= 0:
            # Rotation disabled: zero out rotational action
            action[3:6] = 0.0
            return action
        # The wrapped env applies `rot = action[3:6] * rotation_action_scale`.
        # We map MRP->rotvec on the scaled quantity so downstream scaling stays valid.
        scaled_mrp = action[3:6] * self.rotation_action_scale
        scaled_rotvec = R.from_mrp(scaled_mrp).as_rotvec().astype(np.float32)
        action[3:6] = scaled_rotvec / self.rotation_action_scale
        return action

    def _rotvec_action_to_mrp_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).copy()
        if self.rotation_action_scale <= 0:
            action[3:6] = 0.0
            return action
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

    def __init__(self, env, action_indices=None, action_magnitude=1.0):
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
        self._last_key_time = 0.0
        self._key_timeout_s = 0.3  # clear action if no key event within this window
        self._go_to_target = False
        self._manual_success = False  # press 'f' to mark current step as success

        print(
            "Keyboard intervention enabled (OFF by default): "
            "press W/S/A/D/J/K for XYZ movement, ; to toggle intervention, G for go-to-target, F to mark success"
            + (", L to toggle gripper" if self.gripper_enabled else "")
        )

        self._unsubscribe = TERMINAL_KEYBOARD_HUB.register(self._on_key)

    def _on_key(self, key_char: str):
        key_str = key_char.lower()
        if key_str in self.axis_bindings:
            axis, direction = self.axis_bindings[key_str]
            self.pending_action[axis] = direction
            self._last_key_time = time.monotonic()
            self._go_to_target = False  # cancel go-to-target on manual input
        elif key_str == ";":
            now = time.monotonic()
            if now - self._last_toggle_time >= self._toggle_debounce_s:
                self.intervened = not self.intervened
                self._last_toggle_time = now
                self._go_to_target = False  # cancel go-to-target on toggle
                print(f"Keyboard intervention toggled: {self.intervened}")
        elif key_str == "g":
            self._go_to_target = not self._go_to_target
            print(f"Go-to-target: {self._go_to_target}")
        elif key_str == "f":
            self._manual_success = True
            print("Manual success marked!")
        elif self.gripper_enabled and key_str == "l":
            self._gripper_toggle_requested = True

    def _go_to_target_action(self) -> np.ndarray:
        """Compute action towards target pose using proportional control.

        In the current RAM wrapper chain, KeyboardIntervention sits inside
        RelativeFrame. When it replaces the policy action, that replacement
        bypasses RelativeFrame.transform_action and is sent directly to the
        base environment. Therefore go-to-target must emit base-frame actions.
        """
        action_dim = 7 if self.gripper_enabled else 6
        expert_a = np.zeros((action_dim,), dtype=np.float32)
        try:
            from scipy.spatial.transform import Rotation as Rot
            base_env = self.env.unwrapped
            base_env._update_currpos()
            current_xyz = base_env.currpos[:3]
            current_quat = base_env.currpos[3:7]  # xyzw
            target_xyz = base_env._TARGET_POSE[:3]
            target_rotvec = base_env._TARGET_POSE[3:6]

            # Position error in base frame.
            pos_err_base = target_xyz - current_xyz
            current_rot = Rot.from_quat(current_quat).as_matrix()

            # Rotation error expressed in base frame so the downstream base env
            # can apply it directly.
            target_rot = Rot.from_rotvec(target_rotvec).as_matrix()
            rot_err_base = Rot.from_matrix(target_rot @ current_rot.T).as_rotvec()

            # Proportional control with gain
            kp = 4.0
            delta = np.concatenate([pos_err_base, rot_err_base]) * kp
            expert_a[:6] = np.clip(delta, -1.0, 1.0)

            # Auto-stop if close enough (check in base frame)
            if np.linalg.norm(pos_err_base) < 0.005 and np.linalg.norm(rot_err_base) < 0.05:
                self._go_to_target = False
                print("Go-to-target: reached target, stopped.")
        except Exception as e:
            print(f"Go-to-target error: {e}")
            self._go_to_target = False
        return expert_a

    def _keyboard_action(self) -> np.ndarray:
        action_dim = 7 if self.gripper_enabled else 6
        expert_a = np.zeros((action_dim,), dtype=np.float32)

        # Go-to-target mode
        if self._go_to_target:
            return self._go_to_target_action()

        # Auto-stop: clear pending action if no key event received recently
        if time.monotonic() - self._last_key_time > self._key_timeout_s:
            self.pending_action[:] = 0.0

        if np.linalg.norm(self.pending_action) > 0:
            expert_a[:6] = np.clip(self.pending_action, -1.0, 1.0) * self.action_magnitude

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
        has_keyboard_input = np.linalg.norm(expert_a) > 1e-6
        # Intervention mode: completely disable policy, use keyboard only
        if self.intervened:
            return expert_a, True
        # Go-to-target also works without manual intervention toggle
        if self._go_to_target and has_keyboard_input:
            return expert_a, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        if self._manual_success:
            info["manual_success"] = True
            self._manual_success = False  # one-shot, reset after marking
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._gripper_toggle_requested = False
        self.pending_action[:] = 0.0
        self._last_toggle_time = 0.0
        self._go_to_target = False
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
