from scipy.spatial.transform import Rotation as R
import numpy as np


def quat_2_euler(quat):
    """calculates and returns: yaw, pitch, roll from given quaternion"""
    return R.from_quat(quat).as_euler("xyz")


def euler_2_quat(xyz):
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1)
    if xyz.size != 3:
        raise ValueError(f"Expected xyz euler with shape (3,), got {xyz.shape}")
    # Match Quat2EulerWrapper exactly: SciPy intrinsic xyz convention.
    return R.from_euler("xyz", xyz).as_quat()
