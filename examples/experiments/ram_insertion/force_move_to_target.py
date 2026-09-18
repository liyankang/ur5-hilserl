#!/usr/bin/env python3
"""用 UR 力控（force mode 阻抗）把机械臂柔顺地移动到本目录 config.py 里的 TARGET_POSE。

特点：
  * 不需要外置六维力传感器，用控制器自身估计的 TCP 力；
  * 目标位姿直接读 config.py 的 TARGET_POSE，不在脚本里重复维护；
  * 虚拟弹簧-阻尼 wrench 与 ur_server.py / auto_force_test.py 保持一致；
  * 保护停止、forceMode 连续失败会自动中止；退出时一定 forceModeStop / servoStop / disconnect。

用法：
    python examples/experiments/ram_insertion/force_move_to_target.py          # 需输入 yes 确认
    python examples/experiments/ram_insertion/force_move_to_target.py --yes    # 跳过确认直接执行

注意：同一时刻只允许一个 RTDE 控制端连接。请先停掉 ur_server，否则连接会失败。
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import rtde_control
    import rtde_receive
except ImportError:
    print("错误: 缺少 RTDE 库，请先安装 (pip install ur-rtde)")
    sys.exit(1)


CONFIG_PATH = Path(__file__).resolve().parent / "config.py"

# 与 config.py 的 COMPLIANCE_PARAM 对齐
KP_TRANS, KD_TRANS = 1200.0, 89.0
KP_ROT, KD_ROT = 150.0, 7.0

# (平移正限幅, 平移负限幅, 旋转正限幅, 旋转负限幅)
CLIP_PRESETS = {
    "compliance": (
        np.array([0.0075, 0.0016, 0.0055]),
        np.array([0.0020, 0.0016, 0.0050]),
        np.array([0.010, 0.025, 0.005]),
        np.array([0.010, 0.025, 0.005]),
    ),
    "precision": (np.full(3, 0.1), np.full(3, 0.1), np.full(3, 0.5), np.full(3, 0.5)),
}

POS_DEADBAND = np.full(3, 5e-4)
ROT_DEADBAND = np.full(3, 5e-3)
VEL_DEADBAND = np.array([5e-3, 5e-3, 5e-3, 1e-2, 1e-2, 1e-2])
SEL_AXIS = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}


def _fmt(vec, precision=4):
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{v:.{precision}f}" for v in arr) + "]"


def load_target_pose(config_path=CONFIG_PATH):
    """从 config.py 文本里提取 TARGET_POSE。

    只为取一个常量，没必要 import 整套 jax/gym/ur_env 依赖。
    """
    text = Path(config_path).read_text(encoding="utf-8")
    match = re.search(r"^\s*TARGET_POSE\s*=\s*np\.array\(\s*\[([^\]]*)\]", text, re.M)
    if match is None:
        raise ValueError(f"在 {config_path} 里找不到 TARGET_POSE")
    values = [float(v) for v in match.group(1).replace("\n", " ").split(",") if v.strip()]
    pose = np.asarray(values, dtype=np.float64)
    if pose.size != 6:
        raise ValueError(f"TARGET_POSE 需要 6 个元素 (xyz+rotvec)，实际 {pose.size} 个")
    return pose


class ForceMoveToTarget:
    def __init__(self, args, target):
        self.args = args
        self.target = np.asarray(target, dtype=np.float64)
        self.abort_reason = None
        self.closed = False

        self.rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
        if not getattr(self.rtde_c, "isConnected", lambda: True)():
            raise RuntimeError(
                "RTDE 控制口未连接。检查网线/IP，并确认没有其它程序（如 ur_server）占用控制端。"
            )

        clip_pos, clip_neg, rot_clip_pos, rot_clip_neg = CLIP_PRESETS[args.clip_preset]
        self.clip_pos = clip_pos * args.clip_scale
        self.clip_neg = clip_neg * args.clip_scale
        self.rot_clip_pos = rot_clip_pos * args.rot_clip_scale
        self.rot_clip_neg = rot_clip_neg * args.rot_clip_scale

        self.dev_limits = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
        self.vel_limits = np.array(
            [args.vel_limit] * 3 + [args.rot_vel_limit] * 3, dtype=np.float64
        )
        self.task_frame = np.zeros(6, dtype=np.float64)
        self.selection = np.ones(6, dtype=np.int32)
        for name in args.rigid_axes:
            self.selection[SEL_AXIS[name]] = 0

        self.pos = np.zeros(6)
        self.vel = np.zeros(6)
        self.pos_initialized = False
        self.force_active = False
        self.fail_count = 0
        self.next_tick = 0.0

        self._try("forceModeSetDamping", args.damping)
        self._try("forceModeSetGainScaling", args.gain_scaling)
        if args.mass is not None:
            self._try("setPayload", float(args.mass), [float(v) for v in args.cog])
            print(f"已下发负载: mass={args.mass} kg, cog={args.cog}")
        if not args.no_tare:
            self._try("zeroFtSensor")

        self._read_state()
        print(f"已连接 {args.robot_ip} | 控制 {args.hz:.0f}Hz | 限幅预设 {args.clip_preset} "
              f"x{args.clip_scale:.2f} | 柔顺轴 {self.selection.tolist()}")

    # ── RTDE 基础 ───────────────────────────────────────────────────────────
    def _try(self, method_name, *args):
        method = getattr(self.rtde_c, method_name, None)
        if not callable(method):
            print(f"提示: 当前 ur_rtde 版本没有 {method_name}()，已跳过")
            return None
        try:
            return method(*args)
        except Exception as exc:
            print(f"警告: {method_name} 调用失败: {exc}")
            return None

    def _read_state(self):
        raw = np.asarray(self.rtde_r.getActualTCPPose(), dtype=np.float64).reshape(-1)
        alpha = self.args.filter_alpha
        if not self.pos_initialized:
            self.pos = raw
            self.pos_initialized = True
        elif alpha > 0.0:
            self.pos = alpha * raw + (1.0 - alpha) * self.pos
        else:
            self.pos = raw
        self.vel = np.asarray(self.rtde_r.getActualTCPSpeed(), dtype=np.float64).reshape(-1)

    def _protective_stopped(self):
        return bool(getattr(self.rtde_r, "isProtectiveStopped", lambda: False)())

    def _stop_force_mode(self):
        if self.force_active:
            try:
                self.rtde_c.forceModeStop()
            except Exception:
                pass
            self.force_active = False

    def _abort(self, reason):
        if self.abort_reason is None:
            self.abort_reason = reason
            print(f"\n!! 已中止: {reason}")
            self._stop_force_mode()

    def _limits(self):
        limits = self.dev_limits.copy()
        limits[self.selection == 1] = self.vel_limits[self.selection == 1]
        return limits

    def _send(self, wrench):
        try:
            ok = self.rtde_c.forceMode(
                self.task_frame.tolist(),
                self.selection.tolist(),
                wrench.tolist(),
                2,
                self._limits().tolist(),
            )
        except Exception as exc:
            print(f"\n!! forceMode 调用异常: {exc}")
            ok = False
        if ok is None:
            ok = True
        self.force_active = bool(ok)
        return bool(ok)

    # ── 虚拟弹簧-阻尼（与 ur_server.py 一致）────────────────────────────────
    def _impedance_wrench(self):
        pos_err = self.target[:3] - self.pos[:3]
        pos_err = np.where(np.abs(pos_err) < POS_DEADBAND, 0.0, pos_err)
        pos_err = np.clip(pos_err, -self.clip_neg, self.clip_pos)
        vel = np.where(np.abs(self.vel) < VEL_DEADBAND, 0.0, self.vel)
        force = self.args.kp * pos_err - self.args.kd * vel[:3]

        rot_err = (R.from_rotvec(self.target[3:]) * R.from_rotvec(self.pos[3:]).inv()).as_rotvec()
        rot_err = np.where(np.abs(rot_err) < ROT_DEADBAND, 0.0, rot_err)
        rot_err = np.clip(rot_err, -self.rot_clip_neg, self.rot_clip_pos)
        torque = self.args.kp_rot * rot_err - self.args.kd_rot * vel[3:]
        return np.concatenate([force, torque])

    def _errors(self):
        pos_err = np.linalg.norm(self.target[:3] - self.pos[:3])
        rot_err = np.linalg.norm(
            (R.from_rotvec(self.target[3:]) * R.from_rotvec(self.pos[3:]).inv()).as_rotvec()
        )
        return float(pos_err), float(rot_err)

    # ── 主流程 ──────────────────────────────────────────────────────────────
    def run(self):
        args = self.args
        self.next_tick = time.monotonic()
        t_end = self.next_tick + args.timeout
        stable_since = None

        pos_err, rot_err = self._errors()
        print(f"起点 {_fmt(self.pos)} -> 目标 {_fmt(self.target)} | "
              f"初始误差 {pos_err * 1000:.1f} mm / {np.degrees(rot_err):.2f}°")

        while time.monotonic() < t_end and self.abort_reason is None:
            self._read_state()

            if self._protective_stopped():
                self._abort("机器人处于保护停止")
                break

            pos_err, rot_err = self._errors()
            converged = pos_err < args.tol_pos and rot_err < args.tol_rot
            if converged:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= args.hold_s:
                    break
            else:
                stable_since = None

            wrench = self._impedance_wrench()
            if converged or np.allclose(wrench, 0.0, atol=1e-6):
                # 已在死区内：停止 force mode，让机器人回到位置保持，避免全柔顺下垂。
                self._stop_force_mode()
            else:
                if self._send(wrench):
                    self.fail_count = 0
                else:
                    self.fail_count += 1
                    if self.fail_count >= 20:
                        self._abort(
                            "forceMode 连续失败，请确认示教器处于 Remote Control、"
                            "无保护停止、无其它 RTDE 控制程序"
                        )
                        break

            self._sleep_to_rate()

        pos_err, rot_err = self._errors()
        if self.abort_reason is None:
            print(f"\n结束: {_fmt(self.pos)} | 误差 {pos_err * 1000:.1f} mm / "
                  f"{np.degrees(rot_err):.2f}° | 收敛={pos_err < args.tol_pos and rot_err < args.tol_rot}")
        return self.abort_reason is None

    def _sleep_to_rate(self):
        self.next_tick += 1.0 / self.args.hz
        remaining = self.next_tick - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:
            self.next_tick = time.monotonic()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._stop_force_mode()
        try:
            self.rtde_c.servoStop()
            self.rtde_c.stopL()
        except Exception:
            pass
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        print("已安全断开")


def confirm(target, current, yes):
    print("\n即将开启力控（force mode）把机械臂移动到本目录 config.py 的 TARGET_POSE：")
    print(f"  当前 TCP: {_fmt(current)}")
    print(f"  目标 TCP: {_fmt(target)}")
    print(f"  位置差  : {_fmt(target[:3] - current[:3], 6)} m")
    print("  力控期间机械臂处于柔顺状态，请确认工作空间内无干涉、可随时急停。")
    if yes:
        return True
    try:
        reply = input("输入 yes 继续，其它任意输入取消: ").strip().lower()
    except EOFError:
        return False
    if reply != "yes":
        print("已取消，未下发任何运动指令")
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="力控移动到 ram_insertion/config.py 的 TARGET_POSE")
    parser.add_argument("--robot_ip", default="192.168.25.18")
    parser.add_argument("--hz", type=float, default=100.0, help="控制下发频率")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认，直接执行")
    parser.add_argument("--timeout", type=float, default=20.0, help="最长执行时间 (s)")
    parser.add_argument("--tol_pos", type=float, default=0.002, help="位置收敛阈值 (m)")
    parser.add_argument("--tol_rot", type=float, default=0.02, help="姿态收敛阈值 (rad)")
    parser.add_argument("--hold_s", type=float, default=0.5, help="到达后需保持的时长 (s)")

    # 力控参数（默认与 config.py 的 COMPLIANCE_PARAM 对齐）
    parser.add_argument("--damping", type=float, default=0.05, help="UR force mode 阻尼 [0,1]")
    parser.add_argument("--gain_scaling", type=float, default=0.5, help="UR force mode 增益缩放 [0,2]")
    parser.add_argument("--kp", type=float, default=KP_TRANS, help="平移刚度 K (N/m)")
    parser.add_argument("--kd", type=float, default=KD_TRANS, help="平移阻尼 D (Ns/m)")
    parser.add_argument("--kp_rot", type=float, default=KP_ROT, help="旋转刚度 (Nm/rad)")
    parser.add_argument("--kd_rot", type=float, default=KD_ROT, help="旋转阻尼 (Nms/rad)")
    parser.add_argument("--clip_preset", choices=sorted(CLIP_PRESETS), default="compliance")
    parser.add_argument("--clip_scale", type=float, default=1.0, help="平移限幅整体缩放")
    parser.add_argument("--rot_clip_scale", type=float, default=1.0, help="旋转限幅整体缩放")
    parser.add_argument("--vel_limit", type=float, default=0.1, help="柔顺轴速度上限 (m/s)")
    parser.add_argument("--rot_vel_limit", type=float, default=0.3, help="柔顺轴角速度上限 (rad/s)")
    parser.add_argument("--filter_alpha", type=float, default=0.2, help="位姿低通 α（0=关闭）")
    parser.add_argument(
        "--rigid-axes", nargs="*", default=[], choices=["x", "y", "z", "rx", "ry", "rz"],
        help="这些轴保持刚性、不参与柔顺；默认空 = 6 轴全柔顺（与 ur_server 一致）",
    )
    parser.add_argument("--mass", type=float, default=None, help="负载质量 kg（给了就下发 setPayload）")
    parser.add_argument("--cog", type=float, nargs=3, default=[0.0, 0.0, 0.07], help="负载重心")
    parser.add_argument("--no_tare", action="store_true", help="跳过 zeroFtSensor")
    return parser.parse_args()


def main():
    args = parse_args()
    target = load_target_pose()
    print(f"读取 {CONFIG_PATH} 的 TARGET_POSE = {_fmt(target, 6)}")

    current = np.zeros(6)
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)
        current = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64).reshape(-1)
        rtde_r.disconnect()
    except Exception as exc:
        print(f"错误: 无法读取当前 TCP 位姿 ({exc})，检查网线/IP")
        return

    if not confirm(target, current, args.yes):
        return

    mover = None
    try:
        mover = ForceMoveToTarget(args, target)
        mover.run()
    except KeyboardInterrupt:
        print("\n中断退出")
    finally:
        if mover is not None:
            mover.close()


if __name__ == "__main__":
    main()
