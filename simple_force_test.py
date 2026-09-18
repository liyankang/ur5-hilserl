import rtde_control
import time

ROBOT_IP = "192.168.25.18"

c = rtde_control.RTDEControlInterface(ROBOT_IP)

# Z 方向柔顺
task_frame = [0, 0, 0, 0, 0, 0]
selection = [1, 1, 1, 0, 0, 0]

# 先用较小力测试，避免突然运动
wrench = [0, 0, -6, 0, 0, 0]

# 柔顺轴的速度限制
limits = [0.05, 0.05, 0.15, 0.2, 0.2, 0.2]

# ↓↓↓ 这两个参数最关键 ↓↓↓

# 越小越“轻”，越容易被推动
c.forceModeSetDamping(0.05)
c.forceModeSetGainScaling(0.1)

try:
    while True:
        c.forceMode(
            task_frame,
            selection,
            wrench,
            2,
            limits
        )

        time.sleep(0.01)  # 100 Hz

finally:
    c.forceModeStop()
    c.disconnect()