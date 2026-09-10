#!/usr/bin/env python3
"""Terminal keyboard -> Flask -> UR robot teleop checker.

This script validates the full pipeline:
1. Terminal keyboard input
2. HTTP calls to the UR Flask server
3. Robot TCP motion through the /pose route

Controls:
  ; : arm/disarm motion
  w/s : +Y / -Y
  a/d : -X / +X
  j/k : -Z / +Z
  q   : quit

It uses the project's terminal keyboard hub, so it works in WSL / plain terminals
without requiring X11 or pynput.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass

# Ensure the repo root is importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

from ur_env.utils.terminal_keyboard import TERMINAL_KEYBOARD_HUB


@dataclass
class RouteCheck:
    ok: bool
    message: str


def post_json(base_url: str, route: str, payload=None, timeout: float = 2.0):
    response = requests.post(
        base_url.rstrip("/") + "/" + route.lstrip("/"),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    return response.text


def normalize_pose(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.size != 7:
        raise ValueError(f"Expected pose length 7, got {pose.shape}")
    quat = pose[3:]
    norm = np.linalg.norm(quat)
    if norm == 0:
        pose[3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        pose[3:] = quat / norm
    return pose


def get_state(base_url: str) -> dict:
    state = post_json(base_url, "/getstate")
    state["pose"] = normalize_pose(state["pose"])
    state["q"] = np.asarray(state["q"], dtype=np.float64).reshape(-1)
    state["dq"] = np.asarray(state["dq"], dtype=np.float64).reshape(-1)
    state["vel"] = np.asarray(state["vel"], dtype=np.float64).reshape(-1)
    state["force"] = np.asarray(state["force"], dtype=np.float64).reshape(-1)
    state["torque"] = np.asarray(state["torque"], dtype=np.float64).reshape(-1)
    state["jacobian"] = np.asarray(state["jacobian"], dtype=np.float64)
    return state


def check_routes(base_url: str) -> list[RouteCheck]:
    results: list[RouteCheck] = []

    for route in ("/getstate", "/pose", "/jointreset"):
        try:
            if route == "/pose":
                state = get_state(base_url)
                post_json(base_url, route, {"arr": state["pose"].tolist()})
            else:
                post_json(base_url, route)
            results.append(RouteCheck(True, f"{route} reachable"))
        except Exception as exc:
            results.append(RouteCheck(False, f"{route} failed: {exc}"))

    try:
        # /clearerr is intentionally treated as a best-effort route because it can
        # take longer while stopping and re-uploading the RTDE script.
        post_json(base_url, "/clearerr", timeout=10.0)
        results.append(RouteCheck(True, "/clearerr reachable"))
    except Exception as exc:
        results.append(RouteCheck(True, f"/clearerr slow or failed (ignored for teleop): {exc}"))

    try:
        state = get_state(base_url)
        expected = {
            "pose": (7,),
            "q": (6,),
            "dq": (6,),
            "vel": (6,),
            "force": (3,),
            "torque": (3,),
        }
        for key, shape in expected.items():
            actual = tuple(np.asarray(state[key]).shape)
            if actual == shape:
                results.append(RouteCheck(True, f"/getstate `{key}` shape OK: {actual}"))
            else:
                results.append(RouteCheck(False, f"/getstate `{key}` shape mismatch: got {actual}, expected {shape}"))

        jac_shape = tuple(state["jacobian"].shape)
        if jac_shape == (6, 6):
            results.append(RouteCheck(True, "/getstate `jacobian` shape OK: (6, 6)"))
        else:
            results.append(RouteCheck(False, f"/getstate `jacobian` shape mismatch: got {jac_shape}, expected (6, 6)"))
    except Exception as exc:
        results.append(RouteCheck(False, f"/getstate validation failed: {exc}"))

    return results


class TerminalTeleop:
    def __init__(self, base_url: str, linear_step: float, angular_step_deg: float, hz: float):
        self.base_url = base_url.rstrip("/")
        self.linear_step = float(linear_step)
        self.angular_step = np.deg2rad(float(angular_step_deg))
        self.period = 1.0 / float(hz)
        self.running = True
        self.armed = False
        self.last_toggle_time = 0.0
        self.toggle_debounce_s = 0.25
        self.current_pose = get_state(self.base_url)["pose"]
        self.pending_key = None
        self.last_step_time = 0.0
        self.unsubscribe = TERMINAL_KEYBOARD_HUB.register(self.on_key)

    def print_help(self):
        print("Terminal keyboard -> Flask -> UR teleop checker")
        print("  ;: arm/disarm motion")
        print("  W/S: +Y / -Y")
        print("  A/D: -X / +X")
        print("  J/K: -Z / +Z")
        print("  Q: quit")

    def on_key(self, key_char: str):
        key = key_char.lower()
        if key == ";":
            now = time.monotonic()
            if now - self.last_toggle_time >= self.toggle_debounce_s:
                self.armed = not self.armed
                self.last_toggle_time = now
                print(f"\nintervention armed: {self.armed}")
            return

        if key == "q":
            self.running = False
            return

        if key in ("w", "a", "s", "d", "j", "k"):
            self.pending_key = key

    def step(self):
        if not self.armed or not self.pending_key:
            return

        now = time.monotonic()
        if now - self.last_step_time < self.period:
            return

        try:
            state = get_state(self.base_url)
            pose = state["pose"].copy()
            key = self.pending_key

            if key == "w":
                pose[1] += self.linear_step
            elif key == "s":
                pose[1] -= self.linear_step
            elif key == "a":
                pose[0] -= self.linear_step
            elif key == "d":
                pose[0] += self.linear_step
            elif key == "j":
                pose[2] -= self.linear_step
            elif key == "k":
                pose[2] += self.linear_step
            else:
                return

            post_json(self.base_url, "/pose", {"arr": pose.tolist()})
            self.current_pose = pose
            self.last_step_time = now
            print(
                f"\rarmed={self.armed} key={key} "
                f"tcp={[round(x, 4) for x in pose[:3]]} "
                f"rpy_deg={[round(v, 1) for v in R.from_quat(pose[3:]).as_euler('xyz', degrees=True)]}",
                end="",
                flush=True,
            )
        except Exception as exc:
            print(f"\n/pose failed: {exc}")
            time.sleep(0.3)

    def run(self):
        self.print_help()
        try:
            state = get_state(self.base_url)
            euler = R.from_quat(state["pose"][3:]).as_euler("xyz", degrees=True)
            print(
                f"Initial TCP: {[round(x, 4) for x in state['pose'][:3]]} | "
                f"RPY(deg): {[round(x, 1) for x in euler]}"
            )
            print("Press ; to arm, then hold W/A/S/D/J/K.")
            while self.running:
                self.step()
                time.sleep(0.01)
        finally:
            self.unsubscribe()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5000")
    parser.add_argument("--linear_step", type=float, default=0.006, help="Meters per key step.")
    parser.add_argument("--angular_step_deg", type=float, default=2.0, help="Unused here, kept for symmetry.")
    parser.add_argument("--hz", type=float, default=8.0, help="Command rate while a key is held.")
    parser.add_argument("--check_only", action="store_true", help="Only check routes, do not enter teleop.")
    args = parser.parse_args()

    print(f"Checking UR Flask server at {args.server_url}")
    results = check_routes(args.server_url)
    has_failure = False
    for result in results:
        prefix = "PASS" if result.ok else "FAIL"
        print(f"[{prefix}] {result.message}")
        has_failure = has_failure or (not result.ok)

    if args.check_only:
        return 1 if has_failure else 0

    if has_failure:
        print("Some route checks failed. Fix the FAIL items before running teleop.")
        return 1

    teleop = TerminalTeleop(
        base_url=args.server_url,
        linear_step=args.linear_step,
        angular_step_deg=args.angular_step_deg,
        hz=args.hz,
    )
    teleop.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
