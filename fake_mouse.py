#!/usr/bin/env python3
"""
键盘模拟 3D 空间鼠标（SpaceMouse）控制 UR 机械臂
W/S：前后（Y轴）
A/D：左右（X轴）
J/K：上下（Z轴）
右 Ctrl：按钮触发
Q：退出
"""

import threading
import numpy as np
from typing import Tuple
from pynput import keyboard
import rtde_control
import rtde_receive
import time

# Robot setup
ROBOT_IP = "192.168.25.18"
STEP_DISTANCE = 0.001    # 步长
MAX_SPEED = 0.01         # 最大速度 (符合UR限制)
MAX_ACCEL = 0.01          # 最大加速度
CONTROL_LOOP_INTERVAL = 0.01  # 100Hz控制环


def format_values(values, precision=4):
    return ", ".join(f"{float(v):.{precision}f}" for v in values)


def print_robot_state(rtde_r):
    current_q = np.asarray(rtde_r.getActualQ(), dtype=np.float64)
    current_tcp = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)

    print("\n")
    print("-" * 80)
    print(f"当前关节角 q (rad): [{format_values(current_q)}]")
    print(f"当前 TCP 位姿 (xyz + rotvec): [{format_values(current_tcp)}]")
    print("")
    print(f"launch 参数: --reset_joint_target {format_values(current_q)}")
    print(f"TARGET_POSE = np.array([{format_values(current_tcp)}])")
    print(f"GRASP_POSE = np.array([{format_values(current_tcp)}])")
    print("RESET_POSE = TARGET_POSE + np.array([0, 0, 0.1, 0, 0, 0])")
    print("-" * 80)


class FakeSpaceMouseExpert:
    """键盘模拟空间鼠标"""
    def __init__(self):
        self.state_lock = threading.Lock()
        self.latest_data = {
            "action": np.zeros(6, dtype=np.float32),
            "buttons": [0, 1],
            "print_requested": False,
        }
        self.running = True

        self.thread = threading.Thread(
            target=self._listen_keyboard,
            daemon=True
        )
        self.thread.start()

    def _on_press(self, key):
        try:
            with self.state_lock:
                # Y轴 前后(W/S)
                if key == keyboard.KeyCode.from_char('s'):
                    self.latest_data["action"][1] = 1.0
                elif key == keyboard.KeyCode.from_char('w'):
                    self.latest_data["action"][1] = -1.0
                # X轴 左右(A/D)
                elif key == keyboard.KeyCode.from_char('d'):
                    self.latest_data["action"][0] = -1.0
                elif key == keyboard.KeyCode.from_char('a'):
                    self.latest_data["action"][0] = 1.0
                # Z轴 上下(J/K)
                elif key == keyboard.KeyCode.from_char('k'):
                    self.latest_data["action"][2] = 1.0
                elif key == keyboard.KeyCode.from_char('j'):
                    self.latest_data["action"][2] = -1.0
                # 右键
                elif key == keyboard.Key.ctrl_r:
                    self.latest_data["buttons"] = [1, 0]
                elif key == keyboard.KeyCode.from_char('p'):
                    self.latest_data["print_requested"] = True
                # 退出
                elif key == keyboard.KeyCode.from_char('q'):
                    self.running = False
        except:
            pass

    def _on_release(self, key):
        try:
            with self.state_lock:
                # 松开按键 轴归零
                if key in [keyboard.KeyCode.from_char('w'), keyboard.KeyCode.from_char('s')]:
                    self.latest_data["action"][1] = 0.0
                elif key in [keyboard.KeyCode.from_char('a'), keyboard.KeyCode.from_char('d')]:
                    self.latest_data["action"][0] = 0.0
                elif key in [keyboard.KeyCode.from_char('k'), keyboard.KeyCode.from_char('j')]:
                    self.latest_data["action"][2] = 0.0
                elif key == keyboard.Key.ctrl_r:
                    self.latest_data["buttons"] = [0, 1]
        except:
            pass

    def _listen_keyboard(self):
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

    def get_action(self) -> Tuple[np.ndarray, list]:
        with self.state_lock:
            return self.latest_data["action"].copy(), self.latest_data["buttons"].copy()

    def consume_print_requested(self) -> bool:
        with self.state_lock:
            requested = self.latest_data["print_requested"]
            self.latest_data["print_requested"] = False
            return requested


def main():
    print(f"连接机械臂: {ROBOT_IP}...")
    rtde_c, rtde_r = None, None
    try:
        # 初始化RTDE
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        time.sleep(0.5)
        print("✅ 机械臂连接成功")

        # 获取初始位姿
        current_pose = rtde_r.getActualTCPPose()
        print(f"初始TCP位姿: {[round(x,4) for x in current_pose]}")
        print_robot_state(rtde_r)

        print("\n" + "-"*80)
        print("W/S:前后 | A/D:左右 | J/K:上下 | P:打印当前q和TCP | 右Ctrl:按钮 | Q:退出")
        print("-"*80)

        mouse = FakeSpaceMouseExpert()
        # 起始目标 = 机械臂当前位置：启动时不移动，直接可在当前位置人工干预
        target_pose = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)
        print(f"起始目标(当前 TCP 位姿) = {[round(x, 4) for x in target_pose]}")

        while mouse.running:
            action, buttons = mouse.get_action()
            if mouse.consume_print_requested():
                print_robot_state(rtde_r)
                target_pose = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)
            print(f"\rXYZ控制: [{action[0]:.1f}, {action[1]:.1f}, {action[2]:.1f}] 按钮:{buttons}", end="")

            # 实时更新目标位姿
            if np.any(action[:3] != 0):
                target_pose[0] += action[0] * STEP_DISTANCE  # X
                target_pose[1] += action[1] * STEP_DISTANCE  # Y
                target_pose[2] += action[2] * STEP_DISTANCE  # Z

            # ✅ 修复核心：严格遵循UR servoL参数范围要求
            rtde_c.servoL(
                target_pose,
                MAX_SPEED,       # 速度
                MAX_ACCEL,       # 加速度
                0.1,             # 运动时间 [0.03~0.2] 合规
                0.1,             # 前瞻时间 [0.03~0.2] 合规
                300             # 控制增益 [300~2000] 合规
            )

            time.sleep(CONTROL_LOOP_INTERVAL)

    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
    finally:
        print("\n断开连接中...")
        if rtde_c:
            rtde_c.servoStop()  # 安全停止伺服
            rtde_c.disconnect()
        if rtde_r:
            rtde_r.disconnect()
        print("✅ 已安全断开")


if __name__ == "__main__":
    main()
