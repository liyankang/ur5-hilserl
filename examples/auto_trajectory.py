"""Automatic demonstrations collected through the complete policy wrapper chain."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from examples.collection_utils import make_transition
from ur_env.utils.transformations import construct_adjoint_matrix


def auto_move_to_target(env, target_pose, speed_mm=2.0, interrupt_check=None,
                        *, initial_obs):
    """Start at the caller's latest observation; preserve actual episode flags."""
    if not np.isfinite(speed_mm) or speed_mm <= 0:
        raise ValueError("speed_mm must be finite and positive")
    base_env = env.unwrapped
    scale = float(base_env.action_scale[0])
    if scale <= 0:
        raise ValueError("translation action scale must be positive")
    speed = min(speed_mm / 1000.0 / scale, 1.0)
    obs = copy.deepcopy(initial_obs)
    transitions = []
    while True:
        if interrupt_check is not None and interrupt_check():
            return transitions, True
        action = compute_auto_step_action(base_env, target_pose, speed=speed)
        next_obs, reward, terminated, truncated, info = env.step(action)
        transitions.append(make_transition(
            obs, action, next_obs, reward, terminated, truncated, info))
        obs = copy.deepcopy(next_obs)
        if terminated or truncated:
            return transitions, False


def compute_auto_step_action(base_env, target_pose, speed=0.2, rot_speed=0.05):
    """Invert RelativeFrame and scaled MRP conversion for a bounded base delta."""
    pose = base_env.currpos.copy()
    target_pose = np.asarray(target_pose)
    scale_xyz, scale_rot = base_env.action_scale[:2]
    pos_delta = (target_pose[:3] - pose[:3]) / scale_xyz
    norm = np.linalg.norm(pos_delta)
    if norm > speed:
        pos_delta *= speed / norm
    current_rot = Rot.from_quat(pose[3:])
    target_rot = Rot.from_rotvec(target_pose[3:6])
    rot_delta = (target_rot * current_rot.inv()).as_rotvec()
    norm = np.linalg.norm(rot_delta)
    if norm > rot_speed:
        rot_delta *= rot_speed / norm
    base_action = np.concatenate([
        pos_delta, rot_delta / scale_rot if scale_rot > 0 else np.zeros(3)])
    # RelativeFrame uses the full adjoint, including translation/rotation coupling.
    ee_action = np.linalg.solve(construct_adjoint_matrix(pose), base_action)
    if scale_rot > 0:
        ee_action[3:] = Rot.from_rotvec(ee_action[3:] * scale_rot).as_mrp() / scale_rot
    else:
        ee_action[3:] = 0.0
    ee_action /= max(1.0, np.max(np.abs(ee_action)))
    return ee_action.astype(np.float32)
