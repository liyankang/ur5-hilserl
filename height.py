#!/usr/bin/env python3
"""
UR5 工具重心测量程序（关节扭矩 + 数值雅可比法）

原理：
  关节扭矩 tau = tau_robot(q) + J(q)^T @ F_tool
  其中 F_tool = [0, 0, -mg, -mg*cog_y, mg*cog_x, 0] 是工具重力在 TCP 产生的 wrench

  两个姿态做差，tau_robot 消掉：
  delta_tau = delta(J^T) @ F_tool

  用最小二乘求解 cog_x, cog_y（质量由用户输入）

  数值雅可比：对每个关节做微小扰动，用正运动学算 TCP 位移，
  J[:,i] = delta_TCP / delta_qi，不依赖 DH 参数精确性。

使用步骤：
  1. 用秤称量末端工具质量
  2. 运行脚本，输入质量
  3. 脚本自动移动机器人到多个姿态采集数据
  4. 输出 mass 和 center_of_mass，填入 config.py 的 LOAD_PARAM
"""

import numpy as np
import time
import json
from dataclasses import dataclass
from typing import List, Tuple
import sys

try:
    import rtde_control
    import rtde_receive
except ImportError:
    print("错误: 请安装 RTDE 库")
    print("pip install rtde-control rtde-receive")
    sys.exit(1)


GRAVITY = 9.81
# 数值雅可比扰动步长 (rad)
DELTA_Q = 1e-4


@dataclass
class MeasurementData:
    """测量数据容器"""
    joint_positions: np.ndarray    # 关节角度 (6,)
    joint_torques: np.ndarray      # 关节扭矩 (6,)
    tcp_pos: np.ndarray            # TCP 位置 (3,)
    rotation_matrix: np.ndarray    # 旋转矩阵 (3,3) TCP→Base
    jacobian: np.ndarray           # 几何雅可比 (6,6)


class UR5GravityCalibrator:
    """UR5 重心标定器 - 关节扭矩法"""

    def __init__(self, robot_ip: str = "192.168.25.18"):
        print(f"连接到 UR5: {robot_ip}")
        self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        print("RTDE 连接成功")

        # 使用 getActualCurrent() 获取电机电流，乘以扭矩常数得到关节扭矩
        # UR5 电机扭矩常数 (Nm/A)
        self.Kt = np.array([1.0, 1.0, 0.8, 0.5, 0.5, 0.5])  # 近似值，关节1-3较大，4-6较小
        print("使用 getActualCurrent() 获取电机电流")

    def _get_joint_torques(self) -> np.ndarray:
        """通过电机电流估算关节扭矩: tau = Kt * I"""
        current = np.array(self.rtde_r.getActualCurrent())
        return self.Kt * current

    def _fk(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """UR5 正运动学，返回 (位置, 旋转矩阵)"""
        dh = [
            [0, 0.089159, np.pi / 2],
            [-0.425, 0, 0],
            [-0.39225, 0, 0],
            [0, 0.10915, np.pi / 2],
            [0, 0.09465, -np.pi / 2],
            [0, 0.0823, 0],
        ]
        T = np.eye(4)
        for i in range(6):
            a, d, alpha = dh[i]
            ct, st = np.cos(q[i]), np.sin(q[i])
            ca, sa = np.cos(alpha), np.sin(alpha)
            T_i = np.array([
                [ct, -st * ca,  st * sa, a * ct],
                [st,  ct * ca, -ct * sa, a * st],
                [0,   sa,       ca,      d],
                [0,   0,        0,       1],
            ])
            T = T @ T_i
        return T[:3, 3].copy(), T[:3, :3].copy()

    def _numerical_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        数值几何雅可比 (6x6)
        J[:3, i] = d(position)/d(q_i)   线速度部分
        J[3:, i] = d(rotvec)/d(q_i)     角速度部分（近似）
        """
        J = np.zeros((6, 6))
        p0, R0 = self._fk(q)

        for i in range(6):
            q_pert = q.copy()
            q_pert[i] += DELTA_Q
            p1, R1 = self._fk(q_pert)

            # 线速度部分
            J[:3, i] = (p1 - p0) / DELTA_Q

            # 角速度部分：从旋转差异提取旋转向量
            dR = R1 @ R0.T
            # 从旋转矩阵提取旋转向量 (Rodrigues)
            trace = np.clip(np.trace(dR), -1.0, 3.0)
            angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
            if angle < 1e-10:
                J[3:, i] = np.zeros(3)
            else:
                axis = np.array([
                    dR[2, 1] - dR[1, 2],
                    dR[0, 2] - dR[2, 0],
                    dR[1, 0] - dR[0, 1],
                ]) / (2.0 * np.sin(angle))
                J[3:, i] = (angle * axis) / DELTA_Q

        return J

    def get_current_state(self) -> MeasurementData:
        """获取当前机器人状态"""
        q = np.array(self.rtde_r.getActualQ())
        tau = self._get_joint_torques()
        p, R = self._fk(q)
        J = self._numerical_jacobian(q)

        return MeasurementData(
            joint_positions=q,
            joint_torques=tau,
            tcp_pos=p,
            rotation_matrix=R,
            jacobian=J,
        )

    def move_to_pose(self, joint_angles: np.ndarray, speed: float = 0.06, accel: float = 0.06):
        """移动到目标关节角度"""
        print(f"  移动到: {joint_angles.round(3)} rad")
        self.rtde_c.moveJ(joint_angles.tolist(), speed, accel)
        time.sleep(2.0)

    def measure_at_pose(self, pose_name: str, joint_angles: np.ndarray) -> MeasurementData:
        """在特定位姿进行测量"""
        print(f"\n  测量: {pose_name}")
        self.move_to_pose(joint_angles)
        print("  等待稳定...")
        time.sleep(1.5)

        samples = []
        for _ in range(20):
            samples.append(self.get_current_state())
            time.sleep(0.1)

        avg = MeasurementData(
            joint_positions=np.mean([s.joint_positions for s in samples], axis=0),
            joint_torques=np.mean([s.joint_torques for s in samples], axis=0),
            tcp_pos=np.mean([s.tcp_pos for s in samples], axis=0),
            rotation_matrix=np.mean([s.rotation_matrix for s in samples], axis=0),
            jacobian=np.mean([s.jacobian for s in samples], axis=0),
        )
        std_tau = np.std([s.joint_torques for s in samples], axis=0)

        print(f"  关节扭矩(Nm): [{avg.joint_torques[0]:7.3f}, {avg.joint_torques[1]:7.3f}, "
              f"{avg.joint_torques[2]:7.3f}, {avg.joint_torques[3]:7.3f}, "
              f"{avg.joint_torques[4]:7.3f}, {avg.joint_torques[5]:7.3f}]")
        print(f"  扭矩std:      [{std_tau[0]:.3f}, {std_tau[1]:.3f}, {std_tau[2]:.3f}, "
              f"{std_tau[3]:.3f}, {std_tau[4]:.3f}, {std_tau[5]:.3f}]")
        return avg

    def calculate_cog(self, measurements: List[MeasurementData],
                      mass: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        通过姿态差 + 最小二乘计算重心。

        原理：
          tau_i = tau_robot(q_i) + J_i^T @ wrench_tool
          wrench_tool = [0, 0, -mg, -mg*cog_y, mg*cog_x, 0]

          做差消去 tau_robot：
          delta_tau = delta(J^T) @ wrench_tool

          已知 m，求解 cog_x, cog_y

        返回: (cog_tcp, tau_bias)
          cog_tcp: 重心在 TCP 坐标系中的位置
          tau_bias: 估计的关节扭矩偏置 (6,)
        """
        n = len(measurements)
        mg = mass * GRAVITY

        print(f"\n  输入质量: {mass:.3f} kg, mg = {mg:.2f} N")

        # ── Step 1: 用差值法建立方程 ──
        # 以第一个姿态为参考
        ref = measurements[0]
        A_rows = []
        b_rows = []

        for i in range(1, n):
            m_data = measurements[i]
            delta_tau = m_data.joint_torques - ref.joint_torques
            delta_JT = m_data.jacobian.T - ref.jacobian.T

            # wrench_tool = [0, 0, -mg, -mg*cog_y, mg*cog_x, 0]
            # J^T @ wrench = J^T[:,2]*(-mg) + J^T[:,3]*(-mg*cog_y) + J^T[:,4]*(mg*cog_x)
            # delta(J^T @ wrench) = delta_JT[:,2]*(-mg) + delta_JT[:,3]*(-mg*cog_y) + delta_JT[:,4]*(mg*cog_x)

            # 对每个关节 j:
            # delta_tau[j] = -mg * delta_JT[j,2] - mg*cog_y * delta_JT[j,3] + mg*cog_x * delta_JT[j,4]
            # 整理: delta_tau[j] + mg * delta_JT[j,2] = mg*cog_x * delta_JT[j,4] - mg*cog_y * delta_JT[j,3]

            for j in range(6):
                # 未知数: [cog_x, cog_y]
                # 方程: mg*delta_JT[j,4] * cog_x - mg*delta_JT[j,3] * cog_y = delta_tau[j] + mg*delta_JT[j,2]
                A_rows.append([mg * delta_JT[j, 4], -mg * delta_JT[j, 3]])
                b_rows.append(delta_tau[j] + mg * delta_JT[j, 2])

        A = np.array(A_rows)
        b = np.array(b_rows)

        # 最小二乘求解
        cog_base, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        cog_x, cog_y = cog_base[0], cog_base[1]

        print(f"\n  重心 (基坐标系): [{cog_x:.4f}, {cog_y:.4f}, 0.0000] m")
        print(f"  最小二乘 rank: {rank}")
        if len(residuals) > 0:
            print(f"  残差范数: {np.sqrt(np.sum(residuals)):.4f}")

        # ── Step 2: 估计关节扭矩偏置 ──
        # tau_bias = tau_i - J_i^T @ wrench_tool (对所有姿态取平均)
        tau_biases = []
        for m_data in measurements:
            wrench = np.array([0, 0, -mg, -mg * cog_y, mg * cog_x, 0])
            tau_predicted = m_data.jacobian.T @ wrench
            tau_bias = m_data.joint_torques - tau_predicted
            tau_biases.append(tau_bias)
        tau_bias = np.mean(tau_biases, axis=0)
        print(f"  估计扭矩偏置: [{tau_bias[0]:.2f}, {tau_bias[1]:.2f}, {tau_bias[2]:.2f}, "
              f"{tau_bias[3]:.2f}, {tau_bias[4]:.2f}, {tau_bias[5]:.2f}] Nm")

        # ── Step 3: 转换到 TCP 坐标系 ──
        R_avg = np.mean([m.rotation_matrix for m in measurements], axis=0)
        U, _, Vt = np.linalg.svd(R_avg)
        R_avg = U @ Vt
        cog_tcp = R_avg.T @ np.array([cog_x, cog_y, 0.0])
        cog_tcp[2] = 0.0  # z 不可辨识

        print(f"  重心 (TCP坐标系): [{cog_tcp[0]:.4f}, {cog_tcp[1]:.4f}, {cog_tcp[2]:.4f}] m")

        # ── Step 4: 合理性检查 ──
        cog_norm = np.linalg.norm(cog_tcp)
        if cog_norm > 0.2:
            print(f"\n  !! 警告: 重心偏移 {cog_norm:.3f}m 异常大!")
        elif cog_norm > 0.1:
            print(f"\n  注意: 重心偏移 {cog_norm:.3f}m 偏大，请确认是否合理")

        return cog_tcp, tau_bias

    def set_tool_data(self, mass: float, cog: np.ndarray):
        """设置工具数据到 UR"""
        print(f"\n设置工具数据:")
        print(f"  质量: {mass:.3f} kg")
        print(f"  重心 (TCP): [{cog[0]:.4f}, {cog[1]:.4f}, {cog[2]:.4f}] m")
        try:
            self.rtde_c.setPayload(float(mass), cog.tolist())
            print("  工具数据已设置到机器人")
        except Exception as e:
            print(f"  设置负载失败: {e}")
            print("  请在 UR 示教器上手动设置")

    def verify_calibration(self, mass: float, cog: np.ndarray):
        """验证标定效果 - 设置 Payload 后检查 TCP 力"""
        print("\n" + "=" * 50)
        print("验证标定效果")
        print("=" * 50)

        self.set_tool_data(mass, cog)
        time.sleep(2)

        verify_poses = [
            ("验证-中立", np.array([-1.250, -1.740, -1.790, -1.170, 1.580, 1.900])),
            ("验证-前倾", np.array([-1.205, -2.325, -1.122, -2.005, 1.581, 1.926])),
            ("验证-侧倾", np.array([-0.759, -2.420, -1.124, -2.006, 0.662, 1.926])),
        ]
        all_ok = True
        for name, joints in verify_poses:
            data = self.measure_at_pose(name, joints)
            # 读取 TCP 力来验证
            force_data = self.rtde_r.getActualTCPForce()
            if force_data and len(force_data) >= 6:
                tcp_force = np.array(force_data[:3])
                tcp_torque = np.array(force_data[3:6])
            else:
                tcp_force = np.zeros(3)
                tcp_torque = np.zeros(3)

            f_mag = np.linalg.norm(tcp_force)
            t_mag = np.linalg.norm(tcp_torque)
            status = "OK" if f_mag < 5.0 else "WARN"
            if f_mag >= 10.0:
                status = "FAIL"
                all_ok = False
            print(f"  {name}: TCP力={tcp_force.round(2)} |F|={f_mag:.2f}N |T|={t_mag:.3f}Nm [{status}]")

        if all_ok:
            print("\n  标定通过!")
        else:
            print("\n  标定效果不佳，请确认输入的质量是否准确")
        return all_ok

    def disconnect(self):
        try:
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
        except Exception:
            pass
        print("RTDE 已断开")


def main():
    print("=" * 60)
    print("UR5 工具重心测量程序 (关节扭矩 + 数值雅可比法)")
    print("=" * 60)
    print("""
注意事项：
1. 确保工具已牢固安装
2. 确保工作区域无障碍物
3. 【重要】请用秤称量末端工具（含夹爪）的质量
4. 机器人将移动到多个姿态进行测量，约需 3-5 分钟
""")

    ROBOT_IP = "192.168.25.18"

    # 测量姿态：尽量让末端朝向差异大，提高辨识精度
    measurement_poses = [
        {"name": "姿态1 - 前倾",
         "joints": np.array([-1.205, -2.325, -1.122, -2.005, 1.581, 1.926])},
        {"name": "姿态2 - 侧倾",
         "joints": np.array([-0.759, -2.420, -1.124, -2.006, 0.662, 1.926])},
        {"name": "姿态3 - 后仰",
         "joints": np.array([-1.269, -0.963, -1.871, -1.603, 1.508, 1.926])},
        {"name": "姿态4 - 右侧",
         "joints": np.array([-1.793, -2.182, -1.694, -0.590, 2.438, 1.926])},
        {"name": "姿态5 - 中立",
         "joints": np.array([-1.250, -1.740, -1.790, -1.170, 1.580, 1.900])},
    ]

    calibrator = None
    try:
        calibrator = UR5GravityCalibrator(ROBOT_IP)

        # 获取用户输入的质量
        print("\n" + "-" * 40)
        while True:
            try:
                mass_str = input("请输入末端工具质量 (kg，例如 0.8): ").strip()
                mass = float(mass_str)
                if mass <= 0 or mass > 20:
                    print("  质量应在 0.01 ~ 20 kg 之间，请重新输入")
                    continue
                break
            except ValueError:
                print("  输入无效，请输入数字")

        print(f"  使用质量: {mass:.3f} kg")

        input("\n按 Enter 开始测量，按 Ctrl+C 取消...")

        # 收集数据
        measurements = []
        for pose in measurement_poses:
            data = calibrator.measure_at_pose(pose["name"], pose["joints"])
            measurements.append(data)

        # 计算
        print("\n" + "=" * 50)
        print("计算结果")
        print("=" * 50)

        cog_tcp, tau_bias = calibrator.calculate_cog(measurements, mass)

        print(f"\n推荐配置 (填入 config.py 的 LOAD_PARAM):")
        print(f'  "mass": {mass:.3f},')
        print(f'  "F_x_center_load": [{cog_tcp[0]:.4f}, {cog_tcp[1]:.4f}, {cog_tcp[2]:.4f}],')

        # 验证
        calibrator.verify_calibration(mass, cog_tcp)

        # 保存
        result = {
            "mass_kg": float(mass),
            "center_of_mass_tcp": cog_tcp.tolist(),
            "joint_torque_bias": tau_bias.tolist(),
        }
        with open("tool_calibration_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n结果已保存到 tool_calibration_result.json")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if calibrator:
            calibrator.disconnect()


if __name__ == "__main__":
    main()
