#!/usr/bin/env python3
"""
独立测试 move_joints 函数（保留锁机制）
"""

import time
import threading
import numpy as np
import rtde_control
import rtde_receive

# ========== 辅助函数 ==========
def _call_optional(target, method_names, *args, **kwargs):
    """尝试调用 target 的第一个可用方法"""
    for method_name in method_names:
        method = getattr(target, method_name, None)
        if callable(method):
            try:
                return method(*args, **kwargs)
            except TypeError:
                continue
    return None

def stop_servo(rtde_c):
    """停止所有伺服控制（原始逻辑）"""
    # 注意：原始类中可能还有 forceModeStop 等，这里只做最简
    _call_optional(rtde_c, ["servoStop"])
    _call_optional(rtde_c, ["stopL"])
    _call_optional(rtde_c, ["speedStop"])

def move_joints(rtde_c, q, speed=None, accel=None,
                default_speed=0.2, default_accel=0.2,
                motion_lock=None, lock=None, target_pose=None):
    """
    移动关节（原始 move_joints 逻辑，独立版本）
    
    参数:
        rtde_c: RTDEControlInterface 实例
        q: 目标关节角度 (list or array of 6)
        speed: 速度 (rad/s)
        accel: 加速度 (rad/s²)
        default_speed, default_accel: 默认值
        motion_lock: threading.Lock 对象，用于运动互斥
        lock: 用于保护 target_pose 的锁
        target_pose: 可修改的列表/数组，用于存储更新后的目标位姿
    """
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size != 6:
        raise ValueError("q must have length 6 for UR5")

    speed = float(default_speed if speed is None else speed)
    accel = float(default_accel if accel is None else accel)

    # 使用锁（如果提供）
    if motion_lock:
        motion_lock.acquire()
    try:
        stop_servo(rtde_c)
        _call_optional(rtde_c, ["moveJ"], q.tolist(), speed, accel)
    finally:
        if motion_lock:
            motion_lock.release()

    # 这里原代码调用了 self.get_state() 并更新 self.target_pose
    # 由于没有 get_state 方法，我们用一个简单的 RTDE 读取代替
    # 注意：需要 rtde_r 实例来获取状态，此处作为全局变量传入不方便，因此我们在 main 中直接处理
    # 在调用 move_joints 后，由调用者自行更新 target_pose。
    # 但为了保持一致性，我们可以在函数内通过另一个 RTDE 接收接口获取状态
    # 这里为了最小改动，不在此函数内更新 target_pose，留到外部。
    return q

# ========== 主测试 ==========
def main():
    ROBOT_IP = "192.168.25.18"        # 修改为实际 IP
    TARGET_JOINTS = [0.0849, -0.6295, -0.2698, 2.8370, 1.1641, 0.1422]  # 归零位姿
    SPEED = 0.25
    ACCEL = 0.05

    # 创建锁（模拟类的锁）
    motion_lock = threading.Lock()
    target_pose_lock = threading.Lock()
    target_pose = None  # 用于存放目标位姿

    # 连接 RTDE
    print("连接到机器人...")
    try:
        rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
        rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        print("✅ RTDE 连接成功")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 检查 RTDE 脚本
    try:
        q_curr = rtde_r.getActualQ()
        print(f"当前关节角度: {[round(j, 3) for j in q_curr]}")
        print("RTDE 脚本已在运行")
    except Exception:
        print("⚠️  RTDE 脚本未运行，尝试上传...")
        try:
            rtde_c.reuploadScript()
            time.sleep(1)
            print("✅ RTDE 脚本上传成功")
        except Exception as e:
            print(f"❌ 上传失败: {e}")
            print("请在 UR 示教器上手动运行 RTDE 控制脚本")
            rtde_c.disconnect()
            rtde_r.disconnect()
            return

    # 调用 move_joints
    print(f"\n正在移动到目标关节: {TARGET_JOINTS}")
    try:
        move_joints(rtde_c, TARGET_JOINTS, SPEED, ACCEL,
                    default_speed=0.2, default_accel=0.2,
                    motion_lock=motion_lock, lock=target_pose_lock, target_pose=None)
        print("✅ move_joints 调用成功")
    except Exception as e:
        print(f"❌ move_joints 调用失败: {e}")
        rtde_c.disconnect()
        rtde_r.disconnect()
        return

    # 等待运动完成
    print("等待运动完成...")
    time.sleep(3)

    # 验证位置
    q_after = rtde_r.getActualQ()
    print(f"运动后关节角度: {[round(j, 3) for j in q_after]}")
    error = sum(abs(a - b) for a, b in zip(q_after, TARGET_JOINTS))
    if error < 0.01:
        print("✅ 运动精确到位")
    else:
        print(f"⚠️  位置误差: {error:.4f} rad")

    # 可选：更新 target_pose（如果需要）
    # 这里演示如何模拟原类的 target_pose 更新
    with target_pose_lock:
        # 获取当前 TCP 位姿（原类中通过 get_state() 获取）
        tcp_pose = rtde_r.getActualTCPPose()
        # tcp_pose 是 [x, y, z, rx, ry, rz] 欧拉角，原类存储为 7 维四元数，此处简单转换
        # 为了完整性，我们只打印
        print(f"当前 TCP 位姿 (位置+欧拉角): {[round(p, 4) for p in tcp_pose]}")
        target_pose = tcp_pose  # 示例

    # 断开连接
    rtde_c.disconnect()
    rtde_r.disconnect()
    print("\n测试结束")

if __name__ == "__main__":
    main()