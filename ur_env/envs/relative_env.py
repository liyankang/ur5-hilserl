import copy
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gymnasium import Env
from ur_env.utils.transformations import (
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)


class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)
        info['original_state_obs'] = copy.deepcopy(obs['state'])

        # this is to convert the spacemouse intervention action
        for field in ("intervene_action", "applied_action"):
            if field in info:
                info[field] = self.transform_action_inv(info[field])

        # Update adjoint matrix
        self.adjoint_matrix = self._tcp_frame_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['original_state_obs'] = copy.deepcopy(obs['state'])

        # Update adjoint matrix
        self.adjoint_matrix = self._tcp_frame_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    @staticmethod
    def _tcp_frame_matrix(pose):
        """Rotate TCP-local increments; do not change the point of application.

        动作是同一个 TCP 点的位移和旋转增量，不是从基坐标原点搬移过来的
        空间 twist。两部分分别乘 R 即可；加入 p×R 会把纯平移混成伪旋转。
        """
        rotation = R.from_quat(np.asarray(pose)[3:]).as_matrix()
        transform = np.zeros((6, 6), dtype=np.float64)
        transform[:3, :3] = rotation
        transform[3:, 3:] = rotation
        return transform

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Rotate translation and rotation increments from TCP-local to base axes.
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Rotate increments from base axes back to the same TCP-local axes.
        """
        action = np.array(action)
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action


class DualRelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "left/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ...
                "right/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ...
            }
        ),
        ......
    }, and at least 12 DoF action space
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.left_adjoint_matrix = np.zeros((6, 6))
        self.right_adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.left_T_r_o_inv = np.zeros((4, 4))
            self.right_T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

        # Update adjoint matrix
        self.left_adjoint_matrix = construct_adjoint_matrix(obs["state"]["left/tcp_pose"])
        self.right_adjoint_matrix = construct_adjoint_matrix(obs["state"]["right/tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Update adjoint matrix
        self.left_adjoint_matrix = construct_adjoint_matrix(obs["state"]["left/tcp_pose"])
        self.right_adjoint_matrix = construct_adjoint_matrix(obs["state"]["right/tcp_pose"])

        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.left_T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["left/tcp_pose"])
            )
            self.right_T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["right/tcp_pose"])
            )
        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix
        """
        left_adjoint_inv = np.linalg.inv(self.left_adjoint_matrix)
        obs["state"]["left/tcp_vel"] = left_adjoint_inv @ obs["state"]["left/tcp_vel"]

        right_adjoint_inv = np.linalg.inv(self.right_adjoint_matrix)
        obs["state"]["right/tcp_vel"] = right_adjoint_inv @ obs["state"]["right/tcp_vel"]

        if self.include_relative_pose:
            left_T_b_o = construct_homogeneous_matrix(obs["state"]["left/tcp_pose"])
            left_T_b_r = self.left_T_r_o_inv @ left_T_b_o

            # Reconstruct transformed tcp_pose vector
            left_p_b_r = left_T_b_r[:3, 3]
            left_theta_b_r = R.from_matrix(left_T_b_r[:3, :3]).as_quat()
            obs["state"]["left/tcp_pose"] = np.concatenate((left_p_b_r, left_theta_b_r))

            right_T_b_o = construct_homogeneous_matrix(obs["state"]["right/tcp_pose"])
            right_T_b_r = self.right_T_r_o_inv @ right_T_b_o

            # Reconstruct transformed tcp_pose vector
            right_p_b_r = right_T_b_r[:3, 3]
            right_theta_b_r = R.from_matrix(right_T_b_r[:3, :3]).as_quat()
            obs["state"]["right/tcp_pose"] = np.concatenate((right_p_b_r, right_theta_b_r))


        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the adjoint matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        if len(action) == 12:
            action[:6] = self.left_adjoint_matrix @ action[:6]
            action[6:] = self.right_adjoint_matrix @ action[6:]
        elif len(action) == 14:
            action[:6] = self.left_adjoint_matrix @ action[:6]
            action[7:13] = self.right_adjoint_matrix @ action[7:13]
        else:
            raise ValueError("Action dimension not supported")
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the adjoint matrix.
        """
        action = np.array(action)
        if len(action) == 12:
            action[:6] = np.linalg.inv(self.left_adjoint_matrix) @ action[:6]
            action[6:] = np.linalg.inv(self.right_adjoint_matrix) @ action[6:]
        elif len(action) == 14:
            action[:6] = np.linalg.inv(self.left_adjoint_matrix) @ action[:6]
            action[7:13] = np.linalg.inv(self.right_adjoint_matrix) @ action[7:13]
        else:
            raise ValueError("Action dimension not supported")
        return action
