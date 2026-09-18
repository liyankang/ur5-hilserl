#!/usr/bin/env python3
"""最简测试：直接发 HTTP 让 server 沿单个轴动一下，看机械臂跟不跟。

不依赖键盘、不依赖 env/wrapper 链路，只发 /pose + 读 /getpos。
每次只动一个轴，动完立刻读回实际位移和指令对比。

用法:
    python test_pose_minimal.py                 # 先 Z 上抬 3mm，再 X 横向 +3mm
    python test_pose_minimal.py --delta_mm 0    # 空跑：只验证 HTTP 通路，不动机械臂
"""

import argparse
import time

import numpy as np
import requests

URL = "http://127.0.0.1:5000/"


def getpos() -> np.ndarray:
    resp = requests.post(URL + "getpos", json={}, timeout=2.0)
    resp.raise_for_status()
    return np.asarray(resp.json()["pose"], dtype=np.float64).reshape(-1)


def move_axis(axis: int, delta_mm: float, hold_s: float, hz: float = 10.0) -> np.ndarray:
    """把目标位姿沿单轴偏 delta_mm，持续下发 hold_s 秒，返回实际位移(mm)。"""
    start = getpos()
    target = start.copy()
    target[axis] += delta_mm / 1000.0

    for _ in range(max(int(hold_s * hz), 1)):
        # 不带 source 时 server 按 "policy" 处理，和 env 走的是同一条路
        requests.post(URL + "pose", json={"arr": target.tolist()}, timeout=2.0)
        time.sleep(1.0 / hz)
    time.sleep(0.5)

    actual = (getpos()[:3] - start[:3]) * 1000.0
    print(
        f"  {'xyz'[axis]} 轴指令 {delta_mm:+.1f} mm → 实际位移 "
        f"Δ=({actual[0]:+.1f}, {actual[1]:+.1f}, {actual[2]:+.1f}) mm"
    )
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta_mm", type=float, default=3.0,
                        help="单轴指令位移（0 = 空跑，不动机械臂）")
    parser.add_argument("--hold_s", type=float, default=2.0, help="每轴持续下发的时间")
    args = parser.parse_args()

    print(f"server = {URL}")
    print(f"起始 xyz = {np.round(getpos()[:3], 4).tolist()}")

    if args.delta_mm == 0.0:
        print("delta_mm=0：目标恒等于当前位姿，不会产生位移，只验通路")
        move_axis(2, 0.0, 0.5)
        print("完成：server 可访问，/pose 与 /getpos 通路正常")
        return

    print(f"\n① Z 轴（向上 +{args.delta_mm}mm，远离台面）—— 期望 Z 跟随")
    dz = move_axis(2, args.delta_mm, args.hold_s)

    print(f"\n② X 轴（横向 +{args.delta_mm}mm）—— 期望 XY 不动（selection 只开了 Z 时）")
    dx = move_axis(0, args.delta_mm, args.hold_s)

    thr = 0.5 * abs(args.delta_mm)
    print("\n结论:")
    print(f"  Z 位移 {dz[2]:+.1f} mm → " + ("Z 轴可控" if abs(dz[2]) > thr else "Z 轴没跟上"))
    print(f"  X 位移 {dx[0]:+.1f} mm → " + ("X 轴可控" if abs(dx[0]) > thr else "X 轴被冻结（selection 只开了 Z）"))


if __name__ == "__main__":
    main()
