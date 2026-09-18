#!/usr/bin/env python3
"""类似 ur_server 的阻抗力控测试脚本。

核心做法与 serl_robot_ur/robot_servers/ur_server.py 的 _compute_force_mode_wrench 一致：
    wrench = 刚度 * (目标位置 - 当前位置) - 阻尼 * 速度
再把这个 wrench 下发给 forceMode。forceMode 只当"柔顺执行器"，不依赖力传感器，
所以 CB3（无内置 F/T 传感器）也能正常动。

用法：
    python impedance_force_test.py                    # 默认 z 轴向下 20mm
    AXIS=z DELTA=-0.02 python impedance_force_test.py
    AXIS=x DELTA=0.01 python impedance_force_test.py  # x 轴 +10mm
    AXIS=z DELTA=0 python impedance_force_test.py     # 原地柔顺，可手推验证
"""

import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import rtde_control
    import rtde_receive
except ImportError:
    print("缺少 ur-rtde：pip install ur-rtde")
    sys.exit(1)

# ── 可改参数（默认值与 ur_server.py 对齐）────────────────────────────────────
ROBOT_IP = os.environ.get("ROBOT_IP", "192.168.25.18")
AXIS = os.environ.get("AXIS", "z")                  # 目标偏移轴 x/y/z/rx/ry/rz
DELTA = float(os.environ.get("DELTA", "-0.02"))     # 目标点偏移（m；旋转轴为 rad）

KP = float(os.environ.get("KP", "2000"))            # 平移刚度 N/m
KD = float(os.environ.get("KD", "89"))              # 平移阻尼 Ns/m
KP_ROT = float(os.environ.get("KP_ROT", "150"))     # 旋转刚度 Nm/rad
KD_ROT = float(os.environ.get("KD_ROT", "7"))       # 旋转阻尼 Nms/rad

CLIP = float(os.environ.get("CLIP", "0.01"))        # 平移误差限幅（m），最大力 = KP*CLIP
CLIP_ROT = float(os.environ.get("CLIP_ROT", "0.05"))  # 旋转误差限幅（rad）
POS_DEADBAND = float(os.environ.get("POS_DEADBAND", "5e-4"))  # 位置死区（m）
VEL_DEADBAND = float(os.environ.get("VEL_DEADBAND", "5e-3"))  # 速度死区

DAMPING = float(os.environ.get("DAMPING", "0.05"))  # UR force mode 自身阻尼 [0,1]
VEL_LIMIT = float(os.environ.get("VEL_LIMIT", "0.15"))  # 柔顺轴速度上限
HZ = 100.0
# ────────────────────────────────────────────────────────────────────────────

AXIS_IDX = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}


def main():
    idx = AXIS_IDX[AXIS]

    # 只柔顺 AXIS 轴，其余轴（xy 和旋转）保持刚性，由 UR 位置保持锁住
    selection = np.zeros(6, dtype=np.int32)
    selection[idx] = 1
    task_frame = np.zeros(6)
    limits = np.array([VEL_LIMIT] * 3 + [0.3] * 3, dtype=np.float64)

    c = rtde_control.RTDEControlInterface(ROBOT_IP)
    r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
    print(f"已连接 {ROBOT_IP} | 轴={AXIS} 偏移={DELTA} 刚度={KP} 阻尼={KD}")

    c.forceModeSetDamping(DAMPING)
    c.forceModeSetGainScaling(float(os.environ.get("GAIN_SCALING", "1.0")))

    # 锚点 = 当前位置（xyz + 旋转向量）；目标 = 锚点 + 偏移
    anchor = np.asarray(r.getActualTCPPose(), dtype=np.float64)
    target = anchor.copy()
    target[idx] += DELTA
    print(f"锚点   {np.round(anchor, 4)}")
    print(f"目标点 {np.round(target, 4)}")

    force_active = False
    last_print = 0.0
    try:
        print("进入力控，Ctrl+C 退出 ...")
        next_tick = time.monotonic()
        while True:
            pos = np.asarray(r.getActualTCPPose(), dtype=np.float64)   # xyz + rotvec
            vel = np.asarray(r.getActualTCPSpeed(), dtype=np.float64)  # 6 维

            # 只对 AXIS 轴算力，其余轴 wrench=0（交给 UR 位置保持锁住）
            wrench = np.zeros(6)
            if idx < 3:  # 平移轴
                err = target[idx] - pos[idx]
                err = 0.0 if abs(err) < POS_DEADBAND else err
                err = float(np.clip(err, -CLIP, CLIP))
                v = 0.0 if abs(vel[idx]) < VEL_DEADBAND else vel[idx]
                wrench[idx] = KP * err - KD * v
            else:  # 旋转轴
                rot_err = (R.from_rotvec(target[3:]) * R.from_rotvec(pos[3:]).inv()).as_rotvec()
                err = rot_err[idx - 3]
                err = 0.0 if abs(err) < VEL_DEADBAND else err
                err = float(np.clip(err, -CLIP_ROT, CLIP_ROT))
                wrench[idx] = KP_ROT * err - KD_ROT * vel[idx]

            if np.allclose(wrench, 0.0, atol=1e-6):
                # 已在死区内：退出力控，回到位置保持，避免自由下垂
                if force_active:
                    c.forceModeStop()
                    force_active = False
            else:
                c.forceMode(
                    task_frame.tolist(),
                    selection.tolist(),
                    wrench.tolist(),
                    2,
                    limits.tolist(),
                )
                force_active = True

            now = time.monotonic()
            if now - last_print >= 0.5:
                last_print = now
                print(
                    f"pos[{AXIS}]={pos[idx]:+.4f}  err={target[idx]-pos[idx]:+.4f}  "
                    f"F_req={wrench[idx]:+7.2f}  v={vel[idx]:+.4f}  active={int(force_active)}"
                )

            next_tick += 1.0 / HZ
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        if force_active:
            c.forceModeStop()
        c.disconnect()
        r.disconnect()
        print("已停止力控并断开")


if __name__ == "__main__":
    main()
