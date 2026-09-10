#!/usr/bin/env python3
"""
UR5 Force Mode 独立测试脚本（不需要外置六维力传感器）

三种模式（按 1 / 2 / 3 切换）：
  1 监测：不发运动/力控指令，只读 RTDE 的 actual_TCP_force / actual_TCP_speed，
          统计静止噪声与零漂。用来判断这台控制器的力估计质量
          （e-Series 法兰内置 F/T：静止噪声通常 < 0.5 N；CB3 靠关节电流估算：常见 N 级漂移）。
  2 恒力：只让一个轴 compliant，其余轴保持位置，在该轴上施加恒定力。
          最直观的 force mode 测试：对着电子秤/桌面压，看实测值是否接近指令值。
  3 阻抗：6 轴全 compliant，wrench = K*Δp - D*v（复刻 ur_server.py 的虚拟弹簧-阻尼），
          用键盘挪开目标点，观察跟随误差、接触力上限和"推得动/推不动"的边界。

安全提示：
  * 力控默认关闭，必须按 f 才生效；空格随时立即退出力控。
  * 模式 3 在 wrench≈0 时等价于 freedrive（手臂会被外力和重力推走），首次测试请把
    --force-limit 调小，并确保周围无人、机器人处于安全状态。
  * 建议先在"贴近但不接触"的位置试恒力，再逐步贴近工件。

键盘：
  1 / 2 / 3   切换模式
  f           启用力控       空格  立即停止力控（退出 force mode）
  x / y / z   恒力模式：选择作用轴      r  反向      + / -  力大小 ±0.5N
  W/S A/D J/K 阻抗模式：挪动目标点（基座系 xyz，1mm/步）
  b           阻抗模式：把目标点重锚定到当前位置
  t           重新皮重（zeroFtSensor）  P  打印状态   q  退出

用法：
  python test_force_mode.py --robot_ip 192.168.25.18
  python test_force_mode.py --clip-scale 3 --force-limit 15 --csv force_test.csv
"""

import argparse
import csv
import sys
import threading
import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import rtde_control
    import rtde_receive
except ImportError:
    print("错误: 缺少 RTDE 库，请先安装 (pip install ur-rtde)")
    sys.exit(1)

try:
    from pynput import keyboard
except ImportError:
    print("错误: 缺少 pynput，请先安装 (pip install pynput)")
    sys.exit(1)


# ── 默认参数（位置限幅/刚度与 examples/experiments/ram_insertion/config.py 的
#    COMPLIANCE_PARAM 对齐，方便直接比较）────────────────────────────────────
KP_TRANS, KD_TRANS = 1200.0, 89.0
KP_ROT, KD_ROT = 150.0, 7.0

CLIP_PRESETS = {
    # (正/负位置限幅, 正/负姿态限幅)  单位 m / rad
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
JOG_STEP = 1e-3  # 阻抗模式目标点步长 (m)
KEY_TIMEOUT = 0.3  # 按键持续判定时间 (s)


def _fmt(vec, precision=3):
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{v:.{precision}f}" for v in arr) + "]"


class ForceModeTester:
    def __init__(self, args):
        self.args = args
        self.rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)

        # 刚度 / 阻尼
        self.kp, self.kd = args.kp, args.kd
        self.kp_rot, self.kd_rot = args.kp_rot, args.kd_rot

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

        # 状态
        self.lock = threading.Lock()
        self.running = True
        self.mode = 1
        self.enabled = False
        self.force_active = False
        self.axis = 2
        self.force_sign = -1.0
        self.force_mag = float(args.force_init)
        self.pos = np.zeros(6)
        self.vel = np.zeros(6)
        self.force = np.zeros(6)
        self.target = np.zeros(6)
        self.force_req = np.zeros(6)
        self.fail_count = 0
        self.force_window = deque(maxlen=200)
        self.key_times = {}
        self.force_max = np.zeros(6)
        self.pos_initialized = False
        self.closed = False

        # 初始化：阻尼/增益、皮重、负载
        self._try("forceModeSetDamping", args.damping)
        self._try("forceModeSetGainScaling", args.gain_scaling)
        if args.mass is not None:
            self._try("setPayload", float(args.mass), [float(v) for v in args.cog])
            print(f"已下发负载: mass={args.mass} kg, cog={args.cog}")
        if not args.no_tare:
            self._try("zeroFtSensor")
            print("已执行 zeroFtSensor（皮重）")

        self._read_state()
        self.target = self.pos.copy()
        print(
            f"已连接 {args.robot_ip} | 控制 {args.hz:.0f}Hz | "
            f"限幅缩放 {args.clip_scale:.2f}x（等效最大力 ≈ "
            f"{self.kp * self.clip_pos.max():.1f} N）"
        )

        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.listener.start()

    # ── RTDE ────────────────────────────────────────────────────────────────
    def _try(self, method_name, *args):
        """调用可选的 RTDE 接口，失败时只告警，不中断测试。"""
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
            # 与 ur_server.py 一致：对 RTDE 的 xyz+rotvec 做一阶低通
            self.pos = alpha * raw + (1.0 - alpha) * self.pos
        else:
            self.pos = raw
        self.vel = np.asarray(self.rtde_r.getActualTCPSpeed(), dtype=np.float64).reshape(-1)
        self.force = np.asarray(self.rtde_r.getActualTCPForce(), dtype=np.float64).reshape(-1)
        self.force_max = np.maximum(self.force_max, np.abs(self.force))
        self.force_window.append(self.force.copy())

    def _stop_force_mode(self):
        if self.force_active:
            self.rtde_c.forceModeStop()
            self.force_active = False
        self.force_req = np.zeros(6)

    # ── 键盘 ────────────────────────────────────────────────────────────────
    def _on_press(self, key):
        try:
            ch = getattr(key, "char", None)
            if ch is None:
                return
            ch = ch.lower()
            with self.lock:
                self.key_times[ch] = time.monotonic()
                if ch == "q":
                    self.running = False
                elif ch in ("1", "2", "3"):
                    self.mode = int(ch)
                    self.enabled = False
                    print(f"\n=== 切换到模式 {self.mode}（力控已关闭，按 f 启用）===")
                elif ch == "f":
                    self.enabled = True
                    print("\n=== 力控已启用 ===")
                elif ch == " ":
                    self.enabled = False
                    print("\n=== 力控已停止 ===")
                elif ch in ("x", "y", "z"):
                    self.axis = "xyz".index(ch)
                    print(f"\n恒力作用轴 → {ch.upper()}")
                elif ch == "r":
                    self.force_sign *= -1.0
                    print(f"\n恒力方向 → {'正向' if self.force_sign > 0 else '负向'}")
                elif ch in ("+", "="):
                    self.force_mag = min(self.args.force_limit, self.force_mag + 0.5)
                    print(f"\n恒力大小 → {self.force_mag:.1f} N")
                elif ch in ("-", "_"):
                    self.force_mag = max(0.0, self.force_mag - 0.5)
                    print(f"\n恒力大小 → {self.force_mag:.1f} N")
                elif ch == "b":
                    self.target = self.pos.copy()
                    print("\n目标点已重锚定到当前位置")
                elif ch == "t":
                    self._try("zeroFtSensor")
                    print("\n已重新皮重 (zeroFtSensor)")
        except Exception:
            pass

    def _on_release(self, key):
        try:
            ch = getattr(key, "char", None)
            if ch:
                with self.lock:
                    self.key_times.pop(ch.lower(), None)
        except Exception:
            pass

    def _key_active(self, ch):
        t = self.key_times.get(ch)
        return t is not None and (time.monotonic() - t) < KEY_TIMEOUT

    def _apply_jog(self):
        """阻抗模式：按 WASD/JK 挪动目标点（沿用 fake_mouse 的轴向映射）。"""
        jog = np.zeros(3)
        if self._key_active("a"):
            jog[0] += JOG_STEP
        if self._key_active("d"):
            jog[0] -= JOG_STEP
        if self._key_active("s"):
            jog[1] += JOG_STEP
        if self._key_active("w"):
            jog[1] -= JOG_STEP
        if self._key_active("k"):
            jog[2] += JOG_STEP
        if self._key_active("j"):
            jog[2] -= JOG_STEP
        if np.any(jog):
            with self.lock:
                self.target[:3] += jog

    # ── 力控指令 ────────────────────────────────────────────────────────────
    def _impedance_wrench(self):
        """虚拟弹簧-阻尼: wrench = K*Δp - D*v（与 ur_server.py 一致）。"""
        pos_err = self.target[:3] - self.pos[:3]
        pos_err = np.where(np.abs(pos_err) < POS_DEADBAND, 0.0, pos_err)
        pos_err = np.clip(pos_err, -self.clip_neg, self.clip_pos)
        vel = np.where(np.abs(self.vel) < VEL_DEADBAND, 0.0, self.vel)
        force = self.kp * pos_err - self.kd * vel[:3]

        rot_err = (
            R.from_rotvec(self.target[3:]) * R.from_rotvec(self.pos[3:]).inv()
        ).as_rotvec()
        rot_err = np.where(np.abs(rot_err) < ROT_DEADBAND, 0.0, rot_err)
        rot_err = np.clip(rot_err, -self.rot_clip_neg, self.rot_clip_pos)
        torque = self.kp_rot * rot_err - self.kd_rot * vel[3:]
        return np.concatenate([force, torque])

    def _build_command(self):
        """返回 (selection_vector, wrench)，按当前模式构造。"""
        if self.mode == 2:
            selection = np.zeros(6, dtype=np.int32)
            selection[self.axis] = 1
            wrench = np.zeros(6)
            wrench[self.axis] = self.force_sign * self.force_mag
        elif self.mode == 3:
            selection = np.ones(6, dtype=np.int32)
            wrench = self._impedance_wrench()
        else:
            selection = np.zeros(6, dtype=np.int32)
            wrench = np.zeros(6)
        return selection, wrench

    def _limits(self, selection):
        limits = self.dev_limits.copy()
        limits[selection == 1] = self.vel_limits[selection == 1]
        return limits

    # ── 主循环 ──────────────────────────────────────────────────────────────
    def run(self):
        dt = 1.0 / self.args.hz
        next_t = time.monotonic()
        last_print = 0.0
        last_csv = 0.0
        writer = None
        csv_file = None
        if self.args.csv:
            csv_file = open(self.args.csv, "w", newline="")
            writer = csv.writer(csv_file)
            writer.writerow(
                ["t", "mode", "enabled"]
                + [f"p{i}" for i in range(6)]
                + [f"f_meas{i}" for i in range(6)]
                + [f"wrench{i}" for i in range(6)]
            )

        print("\n" + "-" * 78)
        print("1/2/3 模式 | f 启用力控 | 空格 停止 | x/y/z 选轴 r 反向 +/- 调力")
        print("WASD/JK 挪目标(模式3) | b 重锚定 | t 皮重 | P 打印 | q 退出")
        print("-" * 78)

        try:
            while self.running:
                self._read_state()
                if self.mode == 3:
                    # 允许在力控关闭时先摆好目标点，再按 f 启用
                    self._apply_jog()
                if not self.enabled:
                    self._stop_force_mode()
                else:
                    selection, wrench = self._build_command()
                    stopped = getattr(self.rtde_r, "isProtectiveStopped", lambda: False)()
                    if stopped:
                        print("\n!! 机器人处于保护停止，已关闭力控")
                        self.enabled = False
                        self._stop_force_mode()
                    else:
                        try:
                            ok = self.rtde_c.forceMode(
                                self.task_frame.tolist(),
                                selection.tolist(),
                                wrench.tolist(),
                                2,
                                self._limits(selection).tolist(),
                            )
                        except Exception as exc:
                            print(f"\n!! forceMode 调用异常: {exc}")
                            ok = False
                        self.force_active = bool(ok)
                        self.force_req = wrench
                        self.fail_count = 0 if ok else self.fail_count + 1
                        if self.fail_count == 20:
                            print("\n!! forceMode 连续失败，已关闭力控（检查限幅/模式参数）")
                            self.enabled = False
                            self._stop_force_mode()

                now = time.monotonic()
                if now - last_print >= 0.2:
                    last_print = now
                    self._print_status()
                if writer is not None and now - last_csv >= 0.05:
                    last_csv = now
                    writer.writerow(
                        [f"{now:.4f}", self.mode, int(self.enabled)]
                        + [f"{v:.6f}" for v in self.pos]
                        + [f"{v:.4f}" for v in self.force]
                        + [f"{v:.4f}" for v in self.force_req]
                    )

                next_t += dt
                sleep = next_t - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_t = time.monotonic()
        finally:
            self.close(writer)
            if csv_file is not None:
                csv_file.close()

    def _print_status(self):
        std = np.std(np.array(self.force_window), axis=0) if self.force_window else np.zeros(6)
        tag = "ON " if self.force_active else "off"
        if self.mode == 1:
            print(
                f"\r[mode1 {tag}] F={_fmt(self.force[:3])} T={_fmt(self.force[3:])} "
                f"| 噪声σ(F)=[{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}] N   ",
                end="",
            )
        elif self.mode == 2:
            print(
                f"\r[mode2 {tag}] 轴={'xyz'[self.axis]} 指令={self.force_sign * self.force_mag:+.1f}N "
                f"该轴实测={self.force[self.axis]:+.2f}N F={_fmt(self.force[:3])}   ",
                end="",
            )
        else:
            err = self.target[:3] - self.pos[:3]
            print(
                f"\r[mode3 {tag}] Δp={_fmt(err)} F={_fmt(self.force[:3])} "
                f"wrench={_fmt(self.force_req[:3])} v={_fmt(self.vel[:3])}   ",
                end="",
            )

    def close(self, writer=None):
        if self.closed:
            return
        self.closed = True
        time.sleep(0.1)
        try:
            self.rtde_c.forceModeStop()
            self.rtde_c.servoStop()
            self.rtde_c.stopL()
        except Exception:
            pass
        if writer is not None:
            writer.writerow([])
            writer.writerow(["# max |F| (N):"] + [f"{v:.3f}" for v in self.force_max[:3]])
            writer.writerow(["# max |T| (Nm):"] + [f"{v:.3f}" for v in self.force_max[3:]])
            print(f"\n已写入 CSV: {self.args.csv}")
        try:
            self.listener.stop()
        except Exception:
            pass
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        print("\n力控已停止，RTDE 已断开。最大实测 |F| = " + _fmt(self.force_max[:3]) +
              " N, |T| = " + _fmt(self.force_max[3:]) + " Nm")


def parse_args():
    parser = argparse.ArgumentParser(description="UR5 Force Mode 独立测试")
    parser.add_argument("--robot_ip", default="192.168.25.18")
    parser.add_argument("--hz", type=float, default=100.0, help="力控下发频率")
    parser.add_argument("--damping", type=float, default=0.05, help="UR force mode 阻尼 [0,1]")
    parser.add_argument("--gain_scaling", type=float, default=0.5, help="UR force mode 增益缩放 [0,2]")
    parser.add_argument("--kp", type=float, default=KP_TRANS, help="平移刚度 K (N/m)")
    parser.add_argument("--kd", type=float, default=KD_TRANS, help="平移阻尼 D (Ns/m)")
    parser.add_argument("--kp_rot", type=float, default=KP_ROT, help="旋转刚度 (Nm/rad)")
    parser.add_argument("--kd_rot", type=float, default=KD_ROT, help="旋转阻尼 (Nms/rad)")
    parser.add_argument(
        "--clip_preset", choices=sorted(CLIP_PRESETS), default="compliance",
        help="位置限幅预设（等效最大力 = K × 限幅）",
    )
    parser.add_argument("--clip_scale", type=float, default=1.0, help="平移限幅整体缩放")
    parser.add_argument("--rot_clip_scale", type=float, default=1.0, help="姿态限幅整体缩放")
    parser.add_argument("--vel_limit", type=float, default=0.1, help="compliant 轴速度上限 (m/s)")
    parser.add_argument("--rot_vel_limit", type=float, default=0.3, help="compliant 轴角速度上限 (rad/s)")
    parser.add_argument("--force_init", type=float, default=5.0, help="恒力模式初始力 (N)")
    parser.add_argument("--force_limit", type=float, default=20.0, help="恒力模式可调上限 (N)")
    parser.add_argument("--filter_alpha", type=float, default=0.2, help="位姿低通 α（0=关闭）")
    parser.add_argument("--mass", type=float, default=None, help="负载质量 kg（给了就下发 setPayload）")
    parser.add_argument("--cog", type=float, nargs=3, default=[0.0, 0.0, 0.07], help="负载重心")
    parser.add_argument("--no_tare", action="store_true", help="跳过 zeroFtSensor")
    parser.add_argument("--csv", default=None, help="记录到 CSV 文件")
    return parser.parse_args()


def main():
    args = parse_args()
    tester = None
    try:
        tester = ForceModeTester(args)
        tester.run()
    except KeyboardInterrupt:
        print("\n中断退出")
    finally:
        if tester is not None:
            tester.close(None)


if __name__ == "__main__":
    main()
