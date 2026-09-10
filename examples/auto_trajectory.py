"""Shared auto-trajectory utilities for data collection scripts."""

import copy
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def _homogeneous(pose_quat):
    """Build 4x4 homogeneous transform from xyz+quat pose."""
    T = np.eye(4)
    T[:3, 3] = pose_quat[:3]
    T[:3, :3] = Rot.from_quat(pose_quat[3:7]).as_matrix()
    return T


def _convert_obs_for_training(raw_obs, T_r_o_inv):
    """Convert raw base-env observation to match the wrapper chain output.

    Applies: relative frame transform -> quat->MRP -> SERL flattening.
    """
    state = raw_obs["state"]
    pose_quat = state["tcp_pose"]  # xyz + quat (7D)

    # Relative frame: express pose relative to reset pose
    T_b_o = _homogeneous(pose_quat)
    T_b_r = T_r_o_inv @ T_b_o
    rel_xyz = T_b_r[:3, 3]
    rel_quat = Rot.from_matrix(T_b_r[:3, :3]).as_quat()

    # Quat -> MRP
    rel_mrp = Rot.from_quat(rel_quat).as_mrp()
    pose_out = np.concatenate([rel_xyz, rel_mrp]).astype(np.float32)  # (6,)

    # Build flattened state matching SERLObsWrapper output
    # proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    proprio = np.concatenate([
        pose_out,                                    # tcp_pose (6,)
        state["tcp_vel"].astype(np.float32),         # tcp_vel (6,)
        state["tcp_force"].astype(np.float32),       # tcp_force (3,)
        state["tcp_torque"].astype(np.float32),      # tcp_torque (3,)
        state["gripper_pose"].flatten().astype(np.float32),  # gripper_pose (1,)
    ])  # total: 19D

    obs = {
        "state": proprio,
        "images": raw_obs["images"],
    }
    return obs


def auto_move_to_target(env, target_pose, speed_mm=2.0, interrupt_check=None):
    """Move robot to target using direct position commands (bypasses ACTION_SCALE).

    Collects observations at each step for trajectory recording.
    Observation format matches the wrapper chain output (MRP + relative frame).

    Args:
        env: The gym environment.
        target_pose: Target pose (xyz + rotvec, 6D).
        speed_mm: Step size in mm for interpolation.
        interrupt_check: Optional callable that returns True to abort auto movement.
            Checked each step so keyboard intervention can break out immediately.

    Returns:
        (transitions, interrupted) where interrupted is True if abort was triggered.
    """
    base_env = env.unwrapped
    base_env._update_currpos()
    start_pos = base_env.currpos.copy()
    target_quat = base_env._pose6_to_quat(target_pose)

    distance = np.linalg.norm(target_pose[:3] - start_pos[:3])
    num_steps = max(int(distance / (speed_mm / 1000.0)), 10)

    # Use precision mode for accurate movement
    base_env._post_debug("update_param", payload=base_env.config.PRECISION_PARAM)
    time.sleep(0.3)

    path = np.linspace(start_pos, target_quat, num_steps)
    transitions = []

    # Get reset-pose reference for relative frame conversion
    base_env._update_currpos()
    reset_pose_quat = base_env.currpos.copy()
    T_r_o_inv = np.linalg.inv(_homogeneous(reset_pose_quat))

    prev_obs_raw = None
    for p in path:
        # Check interrupt BEFORE moving (so ; can break out immediately)
        if interrupt_check is not None and interrupt_check():
            return transitions, True

        base_env._send_pos_command(p)
        time.sleep(1.0 / base_env.hz)
        base_env._update_currpos()

        # Build raw observation matching urEnv._get_obs format
        images = base_env.get_im()
        state = {
            "tcp_pose": base_env.currpos.copy(),
            "tcp_vel": base_env.currvel.copy(),
            "gripper_pose": np.atleast_1d(base_env.curr_gripper_pos).flatten().astype(np.float32),
            "tcp_force": base_env.currforce.copy(),
            "tcp_torque": base_env.currtorque.copy(),
        }
        raw_obs = {"images": images, "state": state}

        # Convert to wrapper-chain format
        wrapped_obs = _convert_obs_for_training(raw_obs, T_r_o_inv)

        if prev_obs_raw is not None:
            prev_wrapped = _convert_obs_for_training(prev_obs_raw, T_r_o_inv)
            transition = copy.deepcopy(dict(
                observations=prev_wrapped,
                actions=np.zeros(6, dtype=np.float32),
                next_observations=wrapped_obs,
                rewards=0,
                masks=1.0,
                dones=False,
                infos={},
            ))
            transitions.append(transition)

        prev_obs_raw = copy.deepcopy(raw_obs)

    # Final transition
    if prev_obs_raw is not None:
        base_env._update_currpos()
        final_state = {
            "tcp_pose": base_env.currpos.copy(),
            "tcp_vel": np.zeros(6, dtype=np.float32),
            "gripper_pose": np.atleast_1d(base_env.curr_gripper_pos).flatten().astype(np.float32),
            "tcp_force": base_env.currforce.copy(),
            "tcp_torque": base_env.currtorque.copy(),
        }
        final_raw = {"images": base_env.get_im(), "state": final_state}
        prev_wrapped = _convert_obs_for_training(prev_obs_raw, T_r_o_inv)
        final_wrapped = _convert_obs_for_training(final_raw, T_r_o_inv)
        transitions.append(copy.deepcopy(dict(
            observations=prev_wrapped,
            actions=np.zeros(6, dtype=np.float32),
            next_observations=final_wrapped,
            rewards=0,
            masks=0.0,
            dones=True,
            infos={},
        )))

    base_env.nextpos = target_quat
    return transitions, False


def compute_auto_step_action(base_env, target_pose, speed=0.5):
    """Compute action in EE frame to move TCP toward target pose.

    Reads current pose directly from base_env (like _compute_auto_action in record_demos.py).
    The returned action flows through RelativeFrame which transforms EE->base.
    """
    base_env._update_currpos()
    base_pose = base_env.currpos.copy()  # xyz + quat (7D)
    current_xyz = base_pose[:3]
    current_quat = base_pose[3:7]

    # --- translation: base-frame error -> EE-frame action ---
    pos_err = target_pose[:3] - current_xyz
    R_ee = Rot.from_quat(current_quat).as_matrix()
    R_ee_inv = R_ee.T
    pos_action_ee = R_ee_inv @ pos_err

    # --- rotation: base-frame rotvec error -> EE-frame action ---
    target_rot = Rot.from_rotvec(target_pose[3:6]).as_matrix()
    current_rot = Rot.from_quat(current_quat).as_matrix()
    diff_rot = current_rot.T @ target_rot
    rot_err_rotvec = Rot.from_matrix(diff_rot).as_rotvec()
    rot_action_ee = R_ee_inv @ rot_err_rotvec

    # Normalise so that the largest component ~ 1 when far away, then scale
    combined = np.concatenate([pos_action_ee, rot_action_ee])
    norm = np.linalg.norm(combined)
    if norm < 1e-6:
        return np.zeros(6, dtype=np.float32)
    action_ee = (combined / norm) * speed

    full_action = np.zeros(6, dtype=np.float32)
    full_action[:3] = np.clip(action_ee[:3], -1.0, 1.0)
    full_action[3:6] = np.clip(action_ee[3:6], -1.0, 1.0)
    return full_action
