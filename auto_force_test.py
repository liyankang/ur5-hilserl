#!/usr/bin/env python3
"""
UR5 柔顺/力控自动测试（不需要键盘，不需要图形界面）

柔顺测试（推荐，指标全部基于位置反馈，不依赖力传感器）：
  阶段 monitor     只读 actual_TCP_force，统计静止噪声 σ 与残余零漂（不发任何指令）
  阶段 compliance  对每个轴做目标点阶跃，测跟随误差/上升时间/超调/轴向串扰
  阶段 sweep       对同一动作扫描一组参数（kp/kd/clip_scale/gain_scaling），对比响应差异
  阶段 push        柔顺状态下由人手推动，记录推开距离与松手后回位偏差

力反馈测试（需要真实 F/T 传感器；UR5/CB3 无内置传感器，实测不可用）：
  阶段 constant    按预设轴/力值施加恒定力，保持后释放，统计稳态力误差
  阶段 impedance   按预设幅值/周期往复移动目标点，记录接触力与跟随误差

安全设计：
  * 机器人保护停止、forceMode 连续失败会自动中止
  * 无论正常结束还是异常中断，都会 forceModeStop / servoStop / disconnect
  * --max-force > 0 时，|实测力| 超过阈值立即停止力控并中断；
    但 UR5 无力传感器，该读数在 force mode 下会输出伪值，通常应设 --max-force 0
  * 真正的安全保证是指令力限幅（K × clip）与速度限幅（--vel_limit）

用法：
  python auto_force_test.py --robot_ip 192.168.25.18 --csv comp.csv          # 监测 + 柔顺跟随
  python auto_force_test.py --phases compliance --comp-axes z --comp-amp 0.002
  python auto_force_test.py --phases sweep --sweep-param clip_scale --sweep-values 0.5 1 2
  python auto_force_test.py --phases push --push-s 6
"""

import argparse
import csv
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

ROOT = Path(__file__).resolve().parent

# 与 examples/experiments/ram_insertion/config.py 的 COMPLIANCE_PARAM 对齐
KP_TRANS, KD_TRANS = 1200.0, 89.0
KP_ROT, KD_ROT = 150.0, 7.0

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
AXIS = {"x": 0, "y": 1, "z": 2}
SEL_AXIS = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}


def _fmt(vec, precision=3):
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{v:.{precision}f}" for v in arr) + "]"


def _num(value, unit=""):
    return "n/a" if value is None else f"{value:.2f}{unit}"


def step_metrics(rows, axis, anchor, target_delta, t_step):
    """从目标点阶跃的轨迹中提取柔顺跟随指标（全部基于位置，不依赖力传感器）。

    稳态量取保持段后 30% 的均值，避免用单个采样点被抖动带偏。
    """
    if len(rows) < 5:
        return None
    idx = AXIS[axis]
    t = np.array([r[0] for r in rows]) - t_step
    pos = np.array([r[1] for r in rows])
    disp = pos[:, idx] - anchor[idx]

    tail = disp[int(len(disp) * 0.7):]
    steady = float(np.mean(tail))
    ripple = float(np.std(tail))

    others = [i for i in range(3) if i != idx]
    crosstalk = float(np.max(np.abs(pos[:, others] - anchor[others]))) * 1000.0

    rise = None
    if abs(steady) > 1e-5:
        reach = np.where(np.abs(disp) >= 0.9 * abs(steady))[0]
        if len(reach):
            rise = float(t[reach[0]])

    return {
        "axis": axis,
        "target_mm": target_delta * 1000.0,
        "steady_mm": steady * 1000.0,
        "steady_error_mm": (target_delta - steady) * 1000.0,
        "ripple_mm": ripple * 1000.0,
        "peak_mm": float(np.max(np.abs(disp))) * 1000.0,
        "rise_s": rise,
        "crosstalk_mm": crosstalk,
        "correct_direction": bool(np.sign(steady) == np.sign(target_delta)) if abs(target_delta) > 1e-9 else None,
    }


class AutoForceTest:
    def __init__(self, args):
        self.args = args
        self.abort_reason = None
        self.phase = "init"

        self.rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)

        self.kp, self.kd = args.kp, args.kd
        self.kp_rot, self.kd_rot = args.kp_rot, args.kd_rot

        clip_pos, clip_neg, rot_clip_pos, rot_clip_neg = CLIP_PRESETS[args.clip_preset]
        self.base_clip_pos = clip_pos
        self.base_clip_neg = clip_neg
        self.base_rot_clip_pos = rot_clip_pos
        self.base_rot_clip_neg = rot_clip_neg
        self.base_kp, self.base_kd = args.kp, args.kd
        self.base_gain_scaling = args.gain_scaling
        self.base_damping = args.damping
        self.clip_pos = clip_pos * args.clip_scale
        self.clip_neg = clip_neg * args.clip_scale
        self.rot_clip_pos = rot_clip_pos * args.rot_clip_scale
        self.rot_clip_neg = rot_clip_neg * args.rot_clip_scale

        self.dev_limits = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1])
        self.vel_limits = np.array(
            [args.vel_limit] * 3 + [args.rot_vel_limit] * 3, dtype=np.float64
        )
        self.task_frame = np.zeros(6, dtype=np.float64)

        self.pos = np.zeros(6)
        self.vel = np.zeros(6)
        self.force_raw = np.zeros(6)
        self.force = np.zeros(6)
        self.force_bias = np.zeros(6)
        self.wrench_req = np.zeros(6)
        self.force_active = False
        self.force_max = np.zeros(6)
        self.pos_initialized = False
        self.fail_count = 0
        self.tick = 0.0
        self.closed = False
        self.results = {}

        self._try("forceModeSetDamping", args.damping)
        self._try("forceModeSetGainScaling", args.gain_scaling)
        if args.mass is not None:
            self._try("setPayload", float(args.mass), [float(v) for v in args.cog])
            print(f"已下发负载: mass={args.mass} kg, cog={args.cog}")
        if not args.no_tare:
            self._try("zeroFtSensor")
            print("已尝试 zeroFtSensor（注意：本控制器该指令无效，真正生效的是后面的软件置零）")

        self._read_state()
        connected = getattr(self.rtde_c, "isConnected", lambda: True)()
        print(
            f"已连接 {args.robot_ip} | RTDE控制口 connected={connected} | "
            f"控制 {args.hz:.0f}Hz | 限幅 {args.clip_preset} x{args.clip_scale:.2f} | "
            f"安全上限 {args.max_force:.1f} N"
        )
        if not connected:
            print("警告: RTDE 控制口未连接，检查网线/IP/是否被其它程序占用")

        self.csv_file = None
        self.writer = None
        if args.csv:
            self.csv_file = open(args.csv, "w", newline="")
            self.writer = csv.writer(self.csv_file)
            self.writer.writerow(
                ["t", "mode", "phase", "enabled"]
                + [f"p{i}" for i in range(6)]
                + [f"f_meas{i}" for i in range(6)]
                + [f"wrench{i}" for i in range(6)]
            )

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
        self.force_raw = np.asarray(self.rtde_r.getActualTCPForce(), dtype=np.float64).reshape(-1)
        self.force = self.force_raw - self.force_bias
        self.force_max = np.maximum(self.force_max, np.abs(self.force))

    def _stop_force_mode(self):
        if self.force_active:
            try:
                self.rtde_c.forceModeStop()
            except Exception:
                pass
            self.force_active = False
        self.wrench_req = np.zeros(6)

    def _protective_stopped(self):
        return bool(getattr(self.rtde_r, "isProtectiveStopped", lambda: False)())

    def _abort(self, reason):
        if self.abort_reason is None:
            self.abort_reason = reason
            print(f"\n!! 已中止: {reason}")
            self._stop_force_mode()

    def _selection(self):
        """柔顺轴选择向量：1=柔顺，0=保持刚性。

        --rigid-axes 指定的轴不参与柔顺。6 轴全柔顺时机械臂处于完全"松手"状态，
        没有哪个轴提供支撑，会以约 1Hz 缓慢摆动；保留部分刚性轴可以显著抑制。
        """
        selection = np.ones(6, dtype=np.int32)
        for name in self.args.rigid_axes:
            selection[SEL_AXIS[name]] = 0
        return selection

    def _limits(self, selection):
        limits = self.dev_limits.copy()
        limits[selection == 1] = self.vel_limits[selection == 1]
        return limits

    def _send(self, selection, wrench):
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
        if ok is None:
            ok = True
        self.force_active = bool(ok)
        return bool(ok)

    # ── 虚拟弹簧-阻尼（与 ur_server.py 一致）────────────────────────────────
    def _impedance_wrench(self, target):
        pos_err = target[:3] - self.pos[:3]
        pos_err = np.where(np.abs(pos_err) < POS_DEADBAND, 0.0, pos_err)
        pos_err = np.clip(pos_err, -self.clip_neg, self.clip_pos)
        vel = np.where(np.abs(self.vel) < VEL_DEADBAND, 0.0, self.vel)
        force = self.kp * pos_err - self.kd * vel[:3]

        rot_err = (R.from_rotvec(target[3:]) * R.from_rotvec(self.pos[3:]).inv()).as_rotvec()
        rot_err = np.where(np.abs(rot_err) < ROT_DEADBAND, 0.0, rot_err)
        rot_err = np.clip(rot_err, -self.rot_clip_neg, self.rot_clip_pos)
        torque = self.kp_rot * rot_err - self.kd_rot * vel[3:]
        return np.concatenate([force, torque])

    # ── 单拍执行 ────────────────────────────────────────────────────────────
    def _tick(self, mode, enabled, selection=None, wrench=None):
        self._read_state()
        wrench = np.zeros(6) if wrench is None else np.asarray(wrench, dtype=np.float64)

        if enabled and self.abort_reason is None:
            if self._protective_stopped():
                self._abort("机器人处于保护停止")
            else:
                ok = self._send(selection, wrench)
                if ok:
                    self.fail_count = 0
                    self.wrench_req = wrench
                else:
                    self.wrench_req = np.zeros(6)
                    self.fail_count += 1
                    if self.fail_count >= 20:
                        self._abort(
                            "forceMode 连续失败，请确认示教器处于 Remote Control、"
                            "无保护停止、无其它 RTDE 控制程序"
                        )
        else:
            self._stop_force_mode()

        if enabled and self.abort_reason is None and self.args.max_force > 0:
            # 用未扣除基线的原始读数，避免软件置零掩盖真实接触力。
            # 注意：本控制器的力是关节电流估算，进入 force mode 后会输出几十牛的伪值，
            # 这里的阈值很容易误触发；真正的安全保证是指令力限幅 + 速度限幅。
            # 需要连续跑完时可设 --max-force 0 关闭该检查。
            peak = float(np.max(np.abs(self.force_raw[:3])))
            if peak > self.args.max_force:
                self._abort(
                    f"实测力 {peak:.1f} N 超过安全上限 {self.args.max_force:.1f} N"
                    "（该读数可能是 force mode 下的估算伪值，可用 --max-force 0 关闭）"
                )

        if self.writer is not None:
            self.writer.writerow(
                [f"{time.monotonic():.4f}", mode, self.phase, int(bool(enabled and not self.abort_reason))]
                + [f"{v:.6f}" for v in self.pos]
                + [f"{v:.4f}" for v in self.force]
                + [f"{v:.4f}" for v in self.wrench_req]
            )

        self.tick += 1.0 / self.args.hz
        sleep = self.tick - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            self.tick = time.monotonic()

    def _hold(self, seconds, mode, enabled, selection=None, wrench=None, collect=False, target=None):
        """在指定时长内持续执行，可选收集稳态实测值。

        给 target 时按虚拟弹簧-阻尼实时计算 wrench，机械臂会被真正地拉到目标点；
        只给 wrench 则整个时长内下发同一个恒定力。
        """
        t_end = time.monotonic() + seconds
        samples = []
        while time.monotonic() < t_end and self.abort_reason is None:
            w = self._impedance_wrench(target) if target is not None else wrench
            self._tick(mode, enabled, selection, w)
            if collect:
                samples.append(self.force.copy())
        return np.array(samples) if samples else np.zeros((0, 6))

    def _hold_trace(self, seconds, mode, enabled, selection, target=None, wrench=None):
        """执行指定时长并记录完整轨迹，用于柔顺响应分析。

        target 不为空时按虚拟弹簧-阻尼实时计算 wrench（阻抗模式）。
        """
        t_end = time.monotonic() + seconds
        rows = []
        while time.monotonic() < t_end and self.abort_reason is None:
            w = self._impedance_wrench(target) if target is not None else wrench
            self._tick(mode, enabled, selection, w)
            rows.append(
                (time.monotonic(), self.pos.copy(), self.force.copy(), self.wrench_req.copy())
            )
        return rows

    def _wait_still(self, target, timeout, vel_eps=0.008, hold_s=0.4):
        """持续下发 target，直到机械臂速度低于阈值并保持一小段时间。

        返回是否在超时前静止。用于避开刚进入 force mode 时的瞬态摆动，
        否则会把摆动当成阶跃响应，读数完全失真。
        """
        selection = self._selection()
        t_end = time.monotonic() + timeout
        still_since = None
        while time.monotonic() < t_end and self.abort_reason is None:
            self._tick(3, True, selection, self._impedance_wrench(target))
            if float(np.max(np.abs(self.vel[:3]))) < vel_eps:
                if still_since is None:
                    still_since = time.monotonic()
                elif time.monotonic() - still_since >= hold_s:
                    return True
            else:
                still_since = None
        return False

    def _settle_at(self, target, selection=None):
        """先把机械臂拉到 target 并等它真正静止，再做测量。"""
        selection = self._selection() if selection is None else selection
        self._hold_trace(self.args.comp_settle_s, 3, True, selection, target=target)
        if not self._wait_still(target, self.args.comp_settle_max_s):
            print(f"    提示: 等待 {self.args.comp_settle_max_s:.0f}s 仍未静止，当前位置有持续摆动")

    def _apply_sweep(self, param, value):
        """应用一个待扫描的参数。"""
        if param == "kp":
            self.kp = float(value)
        elif param == "kd":
            self.kd = float(value)
        elif param == "clip_scale":
            self.clip_pos = self.base_clip_pos * value
            self.clip_neg = self.base_clip_neg * value
        elif param == "gain_scaling":
            self._try("forceModeSetGainScaling", float(value))
        elif param == "damping":
            self._try("forceModeSetDamping", float(value))

    def _restore_sweep(self, param):
        if param == "kp":
            self.kp = self.base_kp
        elif param == "kd":
            self.kd = self.base_kd
        elif param == "clip_scale":
            self.clip_pos = self.base_clip_pos * self.args.clip_scale
            self.clip_neg = self.base_clip_neg * self.args.clip_scale
        elif param == "gain_scaling":
            self._try("forceModeSetGainScaling", self.base_gain_scaling)
        elif param == "damping":
            self._try("forceModeSetDamping", self.base_damping)

    # ── 软件置零 ────────────────────────────────────────────────────────────
    def _capture_bias(self):
        """在当前位置采集力基线并扣除。

        这台控制器的 TCP 力来自关节电流估算，zeroFtSensor 对它无效，
        因此只能在软件里把当前姿态下的静态偏置记下来减掉。
        """
        if self.args.bias_mode != "auto":
            print("已跳过软件置零（--bias-mode none）")
            return
        self.force_bias = np.zeros(6)
        samples = []
        t_end = time.monotonic() + self.args.bias_s
        while time.monotonic() < t_end:
            self._read_state()
            samples.append(self.force_raw.copy())
            time.sleep(0.02)
        # 去掉前 1/4，避开残余抖动
        samples = samples[len(samples) // 4 :]
        if not samples:
            return
        data = np.array(samples)
        self.force_bias = data.mean(axis=0)
        # 基线采集期间累积的极值不算测试结果
        self.force_max = np.zeros(6)
        print(f"软件置零偏置 (N) = {_fmt(self.force_bias[:3])}")
        print(f"        噪声 (N) = {_fmt(data.std(axis=0)[:3])}")
        print(
            "  提示: 基线必须在「实际任务的姿态」下采集，且工具不接触任何物体；"
            "任务中姿态变化越大，这个置零越不准"
        )

    # ── 阶段一：只读监测 ────────────────────────────────────────────────────
    def phase_monitor(self):
        self.phase = "monitor"
        print(f"\n[阶段1/3] 只读监测 {self.args.monitor_s:.1f}s（不下发任何指令）")
        data = self._hold(self.args.monitor_s, 1, False, collect=True)
        if len(data) == 0:
            self.results["monitor"] = {"ok": False}
            return
        # 取后半段作为静止段，避开起始瞬态
        tail = data[len(data) // 2 :]
        offset = tail.mean(axis=0)
        noise = tail.std(axis=0)
        self.results["monitor"] = {
            "ok": True,
            "offset_N": offset[:3].tolist(),
            "noise_std_N": noise[:3].tolist(),
            "offset_T_Nm": offset[3:].tolist(),
            "noise_std_T_Nm": noise[3:].tolist(),
        }
        print(f"  静止零漂 (N)   = {_fmt(offset[:3])}")
        print(f"  噪声 σ     (N)   = {_fmt(noise[:3])}")
        print(f"  噪声 σ     (Nm)  = {_fmt(noise[3:])}")
        if float(np.max(noise[:3])) > self.args.noise_limit:
            print(
                f"  !! 噪声 σ 超过 {self.args.noise_limit:.2f} N，"
                "请检查皮重 / 负载参数 / 传感器状态"
            )
        if float(np.max(np.abs(offset[:3]))) > self.args.offset_limit:
            print(
                f"  !! 静止零漂超过 {self.args.offset_limit:.2f} N，"
                "恒力测试的绝对误差会带上这个偏置，建议先重新皮重或标定负载"
            )

    # ── 阶段二：恒力 ────────────────────────────────────────────────────────
    def phase_constant(self):
        self.phase = "constant"
        print(
            f"\n[阶段2/3] 恒力测试 轴={self.args.axes} 力值={self.args.levels} "
            f"保持{self.args.hold_s:.1f}s 稳定期{self.args.settle_s:.1f}s 重复{self.args.repeats}次"
        )
        records = []
        for axis in self.args.axes:
            idx = AXIS[axis]
            for level in self.args.levels:
                for rep in range(self.args.repeats):
                    if self.abort_reason is not None:
                        break
                    cmd = np.zeros(6)
                    cmd[idx] = self.args.sign * level
                    selection = np.zeros(6, dtype=np.int32)
                    selection[idx] = 1

                    # 抬升命令 + 稳定期
                    settle = self._hold(self.args.settle_s, 2, True, selection, cmd, collect=False)
                    hold = self._hold(self.args.hold_s, 2, True, selection, cmd, collect=True)

                    # 释放
                    self._stop_force_mode()
                    self._hold(self.args.pause_s, 2, False, selection, cmd, collect=False)

                    if len(hold):
                        measured = hold[:, idx]
                        err = measured - cmd[idx]
                        records.append(
                            {
                                "axis": axis,
                                "command_N": float(cmd[idx]),
                                "steady_mean_N": float(measured.mean()),
                                "steady_std_N": float(measured.std()),
                                "error_mean_N": float(err.mean()),
                                "error_p95_abs_N": float(np.percentile(np.abs(err), 95)),
                                "n": int(len(hold)),
                            }
                        )
                        print(
                            f"  轴 {axis} 指令 {cmd[idx]:+5.1f}N -> 实测均值 "
                            f"{measured.mean():+6.2f}N (σ={measured.std():.2f}) "
                            f"偏差 {err.mean():+6.2f}N"
                        )
        self.results["constant"] = records

    # ── 阶段三：阻抗 ────────────────────────────────────────────────────────
    def phase_impedance(self):
        self.phase = "impedance"
        idx = AXIS[self.args.imp_axis]
        print(
            f"\n[阶段3/3] 阻抗测试 轴={self.args.imp_axis} 幅值={self.args.imp_amp * 1000:.1f}mm "
            f"周期数={self.args.imp_cycles}"
        )
        selection = self._selection()
        anchor = self.pos.copy()
        records = []

        for cycle in range(self.args.imp_cycles):
            if self.abort_reason is not None:
                break
            prev_amp = 0.0
            for name, amp, duration in (
                ("逼近", self.args.imp_amp, self.args.imp_ramp_s),
                ("保压", self.args.imp_amp, self.args.imp_hold_s),
                ("回退", 0.0, self.args.imp_ramp_s),
                ("松开", 0.0, self.args.imp_pause_s),
            ):
                if self.abort_reason is not None:
                    break
                t0 = time.monotonic()
                t_end = t0 + duration
                collected = []
                while time.monotonic() < t_end and self.abort_reason is None:
                    frac = min(1.0, (time.monotonic() - t0) / max(duration, 1e-6))
                    target = anchor.copy()
                    target[idx] += self.args.sign * (prev_amp + (amp - prev_amp) * frac)
                    wrench = self._impedance_wrench(target)
                    self._tick(3, True, selection, wrench)
                    collected.append((self.force.copy(), self.wrench_req.copy(), target.copy()))
                prev_amp = amp

                if name == "保压" and collected:
                    force = np.array([item[0] for item in collected])
                    req = np.array([item[1] for item in collected])
                    records.append(
                        {
                            "cycle": cycle + 1,
                            "requested_N": float(req[:, idx].mean()),
                            "measured_N": float(force[:, idx].mean()),
                            "error_mean_N": float((force[:, idx] - req[:, idx]).mean()),
                            "error_p95_abs_N": float(
                                np.percentile(np.abs(force[:, idx] - req[:, idx]), 95)
                            ),
                            "n": int(len(force)),
                        }
                    )
                    print(
                        f"  第{cycle + 1}周期 保压: 请求 {req[:, idx].mean():+6.2f}N "
                        f"实测 {force[:, idx].mean():+6.2f}N "
                        f"偏差 {(force[:, idx] - req[:, idx]).mean():+6.2f}N"
                    )

        # 回到锚点附近再退出，避免残留偏移
        if self.abort_reason is None:
            self._hold(self.args.imp_ramp_s, 3, True, selection, target=anchor)
        self.results["impedance"] = records

    # ── 柔顺：目标点阶跃响应 ────────────────────────────────────────────────
    def _one_step(self, axis, amp):
        """对单个轴做一次目标点阶跃，返回跟随指标。"""
        idx = AXIS[axis]
        selection = self._selection()
        anchor = self.pos.copy()

        # 用实时弹簧把机械臂拉到锚点，并等它真正静止（不能下发恒定力，否则会自由下垂）
        self._settle_at(anchor)
        # 稳定后重新取基线，让阶跃测量从平衡点开始
        anchor = self.pos.copy()

        sign = self.args.comp_sign
        target_on = anchor.copy()
        target_on[idx] += sign * amp
        t_step = time.monotonic()
        rows = self._hold_trace(self.args.comp_hold_s, 3, True, selection, target=target_on)
        metrics = step_metrics(rows, axis, anchor, sign * amp, t_step)

        if metrics is not None:
            # 限幅是不对称的（负方向通常小得多），指令力一旦饱和机械臂就不会跟随，
            # 这是"阶跃加多大都不动"的典型原因，必须显式报出来
            wrench = np.array([r[3] for r in rows])
            limit = self.clip_pos[idx] if sign > 0 else self.clip_neg[idx]
            metrics["peak_commanded_N"] = float(np.max(np.abs(wrench[:, idx])))
            metrics["wrench_limit_N"] = float(self.kp * limit)
            metrics["saturated"] = metrics["peak_commanded_N"] >= 0.95 * metrics["wrench_limit_N"]

        self._hold_trace(self.args.comp_return_s, 3, True, selection, target=anchor)
        self._stop_force_mode()
        self._hold(self.args.comp_pause_s, 3, False, selection, np.zeros(6))
        return metrics

    def phase_compliance(self):
        self.phase = "compliance"
        print(
            f"\n[柔顺] 目标点阶跃响应 轴={self.args.comp_axes} "
            f"幅值={self.args.comp_amp * 1000:.1f}mm 方向={'+' if self.args.comp_sign > 0 else '-'}"
        )
        print("  指标全部来自位置反馈，不依赖力传感器")
        records = []
        for axis in self.args.comp_axes:
            if self.abort_reason is not None:
                break
            m = self._one_step(axis, self.args.comp_amp)
            if m:
                records.append(m)
                flag = "  !! 指令力已饱和，机械臂跟不动" if m["saturated"] else ""
                print(
                    f"  轴 {axis}: 目标 {m['target_mm']:+.2f}mm  稳态 {m['steady_mm']:+.2f}mm  "
                    f"稳态误差 {m['steady_error_mm']:+.2f}mm  抖动 {m['ripple_mm']:.2f}mm  "
                    f"指令力 {m['peak_commanded_N']:.2f}/{m['wrench_limit_N']:.2f}N  "
                    f"上升 {_num(m['rise_s'], 's')}  串扰 {m['crosstalk_mm']:.2f}mm  "
                    f"{'方向正确' if m['correct_direction'] else '!! 方向相反'}{flag}"
                )
        self.results["compliance"] = records

    # ── 柔顺：参数对比扫描 ──────────────────────────────────────────────────
    def phase_sweep(self):
        self.phase = "sweep"
        param = self.args.sweep_param
        values = self.args.sweep_values
        axis = self.args.comp_axes[0]
        print(f"\n[扫描] 参数 {param} 取值 {values}，测试轴 {axis}（其余参数固定）")
        records = []
        try:
            for value in values:
                if self.abort_reason is not None:
                    break
                self._apply_sweep(param, value)
                m = self._one_step(axis, self.args.comp_amp)
                if m:
                    m["param"] = param
                    m["value"] = value
                    records.append(m)
                    flag = "  !!饱和" if m["saturated"] else ""
                    print(
                        f"  {param}={value:<6}: 稳态 {m['steady_mm']:+.2f}mm  "
                        f"稳态误差 {m['steady_error_mm']:+.2f}mm  "
                        f"抖动 {m['ripple_mm']:.2f}mm  指令力 {m['peak_commanded_N']:.2f}N  "
                        f"上升 {_num(m['rise_s'], 's')}  串扰 {m['crosstalk_mm']:.2f}mm{flag}"
                    )
        finally:
            self._restore_sweep(param)
            print(f"  已还原 {param} 到默认值")
        self.results["sweep"] = records

    # ── 柔顺：外力推动 ──────────────────────────────────────────────────────
    def phase_push(self):
        self.phase = "push"
        selection = self._selection()
        anchor = self.pos.copy()
        print("\n[推动] 外力推动柔顺性测试（需要你用手配合）")
        print("  先拉到锚点并等它静止 ...")
        self._settle_at(anchor, selection)
        anchor = self.pos.copy()

        print(f"\n  >>> 请用手推动机械臂，持续 {self.args.push_s:.0f} 秒 <<<")
        rows_push = self._hold_trace(self.args.push_s, 3, True, selection, target=anchor)
        print(f"\n  >>> 请松手，观察回位 {self.args.push_return_s:.0f} 秒 <<<")
        rows_return = self._hold_trace(self.args.push_return_s, 3, True, selection, target=anchor)

        record = {}
        if rows_push:
            pos_push = np.array([r[1] for r in rows_push])
            disp = pos_push[:, :3] - anchor[:3]
            dist = np.linalg.norm(disp, axis=1)
            peak = int(dist.argmax())
            record["max_push_mm"] = float(dist.max() * 1000.0)
            record["max_push_axis_mm"] = [float(v * 1000.0) for v in disp[peak]]
        if rows_return:
            pos_ret = np.array([r[1] for r in rows_return])
            resid = pos_ret[-1, :3] - anchor[:3]
            record["residual_mm"] = float(np.linalg.norm(resid) * 1000.0)
            record["residual_axis_mm"] = [float(v * 1000.0) for v in resid]
        self.results["push"] = record

        if record:
            print(
                f"  最大推开距离 {record.get('max_push_mm', 0):.2f}mm "
                f"分量 {_fmt(record.get('max_push_axis_mm', [0, 0, 0]), 2)}"
            )
            print(
                f"  松手后残余偏差 {record.get('residual_mm', 0):.2f}mm "
                f"分量 {_fmt(record.get('residual_axis_mm', [0, 0, 0]), 2)}"
            )

    # ── 主流程 ──────────────────────────────────────────────────────────────
    def run(self):
        phases = [name.strip() for name in self.args.phases.split(",") if name.strip()]
        handlers = {
            "monitor": self.phase_monitor,
            "constant": self.phase_constant,
            "impedance": self.phase_impedance,
            "compliance": self.phase_compliance,
            "sweep": self.phase_sweep,
            "push": self.phase_push,
        }
        unknown = [name for name in phases if name not in handlers]
        if unknown:
            raise SystemExit(
                f"未知阶段: {unknown}，可选 "
                "monitor/constant/impedance/compliance/sweep/push"
            )

        print("=" * 78)
        print(f"自动化流程: {' -> '.join(phases)}")
        print("=" * 78)
        self.tick = time.monotonic()
        self._capture_bias()

        for name in phases:
            if self.abort_reason is not None:
                break
            handlers[name]()

        self._print_summary()

    def _print_summary(self):
        print("\n" + "=" * 78)
        print("测试汇总")
        print("=" * 78)
        monitor = self.results.get("monitor")
        if monitor and monitor.get("ok"):
            print(f"静止零漂 (N)  : {_fmt(monitor['offset_N'])}")
            print(f"噪声 σ   (N)  : {_fmt(monitor['noise_std_N'])}")

        records = self.results.get("constant") or []
        if records:
            print("\n恒力: 轴  指令(N)  实测均值(N)    偏差(N)  p95|偏差|(N)")
            for item in records:
                print(
                    f"      {item['axis']}  {item['command_N']:+7.1f}  "
                    f"{item['steady_mean_N']:+9.2f}  {item['error_mean_N']:+9.2f}  "
                    f"{item['error_p95_abs_N']:11.2f}"
                )

        records = self.results.get("impedance") or []
        if records:
            print("\n阻抗: 周期  请求(N)  实测(N)  偏差(N)")
            for item in records:
                print(
                    f"      {item['cycle']:>4}  {item['requested_N']:+7.2f}  "
                    f"{item['measured_N']:+7.2f}  {item['error_mean_N']:+7.2f}"
                )

        records = self.results.get("compliance") or []
        if records:
            print("\n柔顺跟随: 轴  目标(mm)  稳态(mm)  稳态误差(mm)  抖动(mm)  上升(s)  串扰(mm)")
            for item in records:
                print(
                    f"          {item['axis']}  {item['target_mm']:+8.2f}  {item['steady_mm']:+8.2f}  "
                    f"{item['steady_error_mm']:+12.2f}  {item['ripple_mm']:8.2f}  "
                    f"{_num(item['rise_s']):>7}  {item['crosstalk_mm']:8.2f}"
                )

        records = self.results.get("sweep") or []
        if records:
            print(f"\n参数扫描: {records[0]['param']}")
            print("    取值   稳态(mm)  稳态误差(mm)  抖动(mm)  上升(s)")
            for item in records:
                print(
                    f"    {item['value']:<6} {item['steady_mm']:+8.2f}  "
                    f"{item['steady_error_mm']:+12.2f}  {item['ripple_mm']:8.2f}  "
                    f"{_num(item['rise_s']):>7}"
                )

        push = self.results.get("push")
        if push:
            print("\n外力推动:")
            if "max_push_mm" in push:
                print(
                    f"        最大推开距离 {push['max_push_mm']:6.2f} mm  "
                    f"分量 {_fmt(push['max_push_axis_mm'], 2)}"
                )
            if "residual_mm" in push:
                print(
                    f"        松手后残余   {push['residual_mm']:6.2f} mm  "
                    f"分量 {_fmt(push['residual_axis_mm'], 2)}"
                )

        print(f"\n全程最大实测 |F| = {_fmt(self.force_max[:3])} N, |T| = {_fmt(self.force_max[3:])} Nm")
        if self.abort_reason:
            print(f"中止原因: {self.abort_reason}")
        else:
            print("测试正常结束")

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
        if self.writer is not None:
            self.writer.writerow([])
            self.writer.writerow(["# max |F| (N):"] + [f"{v:.3f}" for v in self.force_max[:3]])
            self.writer.writerow(["# max |T| (Nm):"] + [f"{v:.3f}" for v in self.force_max[3:]])
        if self.csv_file is not None:
            self.csv_file.close()
            print(f"已写入 CSV: {self.args.csv}")
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()


def run_analysis(csv_path):
    sys.path.insert(0, str(ROOT))
    import json

    from analyze_force_data import analyze, build_ai_prompt, load_csv

    source = Path(csv_path)
    samples, metadata = load_csv(source)
    result = analyze(samples, metadata)
    json_path = source.parent / f"{source.stem}_analysis.json"
    prompt_path = source.parent / f"{source.stem}_ai_prompt.txt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_path.write_text(build_ai_prompt(result), encoding="utf-8")
    print(f"\n分析结果: {json_path}")
    print(f"AI 分析文本: {prompt_path}")
    for warning in result["warnings"]:
        print(f"- {warning}")


def parse_args():
    parser = argparse.ArgumentParser(description="UR5 柔顺/力控全自动测试（无需键盘）")
    parser.add_argument("--robot_ip", default="192.168.25.18")
    parser.add_argument("--hz", type=float, default=100.0, help="控制下发频率")
    parser.add_argument(
        "--phases", default="monitor,compliance",
        help="要执行的阶段，逗号分隔："
             "monitor,compliance,sweep,push（柔顺相关）| constant,impedance（力反馈相关）",
    )
    parser.add_argument("--csv", default="force_auto.csv", help="CSV 输出路径")
    parser.add_argument("--analyze", action="store_true", help="结束后自动调用分析脚本")

    # 安全
    parser.add_argument(
        "--max-force", type=float, default=0.0,
        help="|实测力| 安全上限 (N)；默认 0=关闭。UR5/CB3 无力传感器，该读数在 force mode "
             "下会输出几十牛的伪值，开启后极易误触发；真正的安全保证是指令力限幅与速度限幅",
    )
    parser.add_argument("--noise_limit", type=float, default=0.5, help="静止噪声告警阈值 (N)")
    parser.add_argument("--offset_limit", type=float, default=1.0, help="静止零漂告警阈值 (N)")
    parser.add_argument(
        "--bias-mode", choices=["auto", "none"], default="auto",
        help="软件置零：auto=启动时在当前位置采集力基线并扣除（zeroFtSensor 对这类控制器无效）",
    )
    parser.add_argument("--bias-s", type=float, default=1.5, help="力基线采集时长 (s)")

    # 力控参数
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
    parser.add_argument("--filter_alpha", type=float, default=0.2, help="位姿低通 α（0=关闭）")
    parser.add_argument("--mass", type=float, default=None, help="负载质量 kg（给了就下发 setPayload）")
    parser.add_argument("--cog", type=float, nargs=3, default=[0.0, 0.0, 0.07], help="负载重心")
    parser.add_argument("--no_tare", action="store_true", help="跳过 zeroFtSensor")

    # 阶段一
    parser.add_argument("--monitor-s", type=float, default=5.0, help="只读监测时长 (s)")

    # 阶段二
    parser.add_argument("--axes", nargs="+", default=["z"], choices=["x", "y", "z"], help="恒力作用轴")
    parser.add_argument("--levels", nargs="+", type=float, default=[3.0, 5.0], help="恒力大小 (N)")
    parser.add_argument("--sign", type=float, default=-1.0, choices=[-1.0, 1.0], help="恒力方向")
    parser.add_argument("--hold-s", type=float, default=3.0, help="每档保持时长 (s)")
    parser.add_argument("--settle-s", type=float, default=1.0, help="每档稳定期，不计入统计 (s)")
    parser.add_argument("--pause-s", type=float, default=1.0, help="档位之间的释放间隔 (s)")
    parser.add_argument("--repeats", type=int, default=1, help="每档重复次数")

    # 阶段三
    parser.add_argument("--imp-axis", default="z", choices=["x", "y", "z"], help="阻抗作用轴")
    parser.add_argument("--imp-amp", type=float, default=0.002, help="目标点幅值 (m)")
    parser.add_argument("--imp-cycles", type=int, default=3, help="往复周期数")
    parser.add_argument("--imp-ramp-s", type=float, default=1.5, help="单程斜坡时长 (s)")
    parser.add_argument("--imp-hold-s", type=float, default=2.0, help="保压时长 (s)")
    parser.add_argument("--imp-pause-s", type=float, default=1.0, help="松开停留时长 (s)")

    # 柔顺测试
    parser.add_argument(
        "--rigid-axes", nargs="*", default=[],
        choices=["x", "y", "z", "rx", "ry", "rz"],
        help="这些轴保持刚性、不参与柔顺；默认空 = 6 轴全柔顺。"
             "全柔顺时机械臂完全松手，会以约 1Hz 缓慢摆动，保留部分刚性轴可抑制",
    )
    parser.add_argument(
        "--comp-axes", nargs="+", default=["x", "y", "z"], choices=["x", "y", "z"],
        help="柔顺跟随测试的轴",
    )
    parser.add_argument(
        "--comp-sign", type=float, default=1.0, choices=[-1.0, 1.0],
        help="阶跃方向；注意 CLIP_PRESETS 的负方向限幅小得多，-1 时很容易指令力饱和",
    )
    parser.add_argument("--comp-amp", type=float, default=0.002, help="目标点阶跃幅值 (m)")
    parser.add_argument("--comp-hold-s", type=float, default=2.5, help="阶跃后保持时长 (s)")
    parser.add_argument("--comp-return-s", type=float, default=2.0, help="回锚点保持时长 (s)")
    parser.add_argument("--comp-settle-s", type=float, default=1.5, help="每次阶跃前的最短稳定时长 (s)")
    parser.add_argument(
        "--comp-settle-max-s", type=float, default=8.0,
        help="等待机械臂静止的最长时间 (s)；超过了就带着摆动开始测量（会污染结果）",
    )
    parser.add_argument("--comp-pause-s", type=float, default=1.0, help="两次测试之间的刚性停留 (s)")
    parser.add_argument(
        "--sweep-param", choices=["kp", "kd", "clip_scale", "gain_scaling", "damping"],
        default="damping",
        help="参数扫描的对象；damping 是 UR force mode 自身阻尼，默认值 0.05 偏小易振荡",
    )
    parser.add_argument(
        "--sweep-values", nargs="+", type=float, default=[0.05, 0.2, 0.5],
        help="参数扫描取值列表",
    )
    parser.add_argument("--push-s", type=float, default=6.0, help="外力推动记录时长 (s)")
    parser.add_argument("--push-return-s", type=float, default=4.0, help="松手后回位观察时长 (s)")
    return parser.parse_args()


def main():
    args = parse_args()
    tester = None
    try:
        tester = AutoForceTest(args)
        tester.run()
    except KeyboardInterrupt:
        print("\n中断退出")
    finally:
        if tester is not None:
            tester.close()
    if args.analyze and args.csv:
        run_analysis(args.csv)


if __name__ == "__main__":
    main()
