#!/usr/bin/env python3
"""
最简 Z 轴力控测试：只在 Z 轴 compliant 并施加恒力，其余 5 轴保持位置。

用法:
  python test_force_z.py --force -5 --duration 5      # 向下压 5N，持续 5s
  python test_force_z.py --force 3                    # 向上拉 3N
按 Ctrl+C 随时停止。
"""
import argparse
import time

import rtde_control
import rtde_receive


def main():
    ap = argparse.ArgumentParser(description="UR5 Z 轴 forceMode 最简测试")
    ap.add_argument("--robot_ip", default="192.168.25.18")
    ap.add_argument("--force", type=float, default=-5.0, help="Z 轴恒力 (N)，负=向下")
    ap.add_argument("--duration", type=float, default=5.0, help="持续时间 (s)")
    ap.add_argument("--hz", type=float, default=100.0, help="下发频率")
    ap.add_argument("--no_tare", action="store_true", help="跳过 zeroFtSensor 皮重")
    args = ap.parse_args()

    force = max(-20.0, min(20.0, args.force))

    rtde_c = rtde_control.RTDEControlInterface(args.robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(args.robot_ip)

    task_frame = [0.0] * 6
    selection = [0, 0, 1, 0, 0, 0]              # 只放开 Z 轴
    wrench = [0.0, 0.0, force, 0.0, 0.0, 0.0]   # Z 轴恒力
    limits = [0.02, 0.02, 0.1, 0.1, 0.1, 0.1]   # Z 轴速度上限 0.1 m/s

    try:
        if not args.no_tare:
            rtde_c.zeroFtSensor()
            print("已执行 zeroFtSensor（皮重）")
        print(f"Z 轴恒力 {force:+.1f} N，持续 {args.duration:.1f}s，Ctrl+C 停止")

        t0 = time.monotonic()
        fz_max = 0.0
        while time.monotonic() - t0 < args.duration:
            ok = rtde_c.forceMode(task_frame, selection, wrench, 2, limits)
            fz = rtde_r.getActualTCPForce()[2]
            fz_max = max(fz_max, abs(fz))
            print(f"\r指令 Fz={force:+.1f}N | 实测 Fz={fz:+.2f}N | 峰值={fz_max:.2f}N | 接受={ok}", end="")
            time.sleep(1.0 / args.hz)
        print()
    except KeyboardInterrupt:
        print("\n手动中断")
    finally:
        rtde_c.forceModeStop()
        rtde_c.servoStop()
        rtde_c.disconnect()
        rtde_r.disconnect()
        print("力控已停止，RTDE 已断开")


if __name__ == "__main__":
    main()
