#!/usr/bin/env python3
"""Interactive keyboard checker for the UR Flask server.

This script is meant for migration debugging on the real UR5 path:
record_demos -> ram_insertion -> ur_env -> ur_server

It does not require the full env stack. It directly probes the Flask server and
lets the operator move the TCP with the keyboard to verify:
- `getstate` shape compatibility
- `pose` route behavior
- `jointreset` / `clearerr` reachability
- whether the current UR server behaves as expected after migrating from Franka
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import requests
from scipy.spatial.transform import Rotation as R


@dataclass
class ServerCheckResult:
    ok: bool
    message: str


class _DummyKeyboardEnv(gym.Env):
    """Minimal env stub so we can reuse KeyboardIntervention directly."""

    def __init__(self):
        self.action_space = gym.spaces.Box(
            low=-np.ones((6,), dtype=np.float32),
            high=np.ones((6,), dtype=np.float32),
        )
        self.observation_space = gym.spaces.Box(
            low=-np.ones((1,), dtype=np.float32),
            high=np.ones((1,), dtype=np.float32),
        )

    def reset(self, **kwargs):
        return np.zeros((1,), dtype=np.float32), {}

    def step(self, action):
        return np.zeros((1,), dtype=np.float32), 0.0, False, False, {}


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
        raise ValueError(f"Expected pose length 7 (xyz + quat), got {pose.shape}")
    quat = pose[3:]
    quat_norm = np.linalg.norm(quat)
    if quat_norm == 0:
        pose[3:] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        pose[3:] = quat / quat_norm
    return pose


def get_state(base_url: str) -> dict:
    state = post_json(base_url, "/getstate")
    state["pose"] = normalize_pose(state["pose"])
    state["vel"] = np.asarray(state["vel"], dtype=np.float64).reshape(-1)
    state["force"] = np.asarray(state["force"], dtype=np.float64).reshape(-1)
    state["torque"] = np.asarray(state["torque"], dtype=np.float64).reshape(-1)
    state["q"] = np.asarray(state["q"], dtype=np.float64).reshape(-1)
    state["dq"] = np.asarray(state["dq"], dtype=np.float64).reshape(-1)
    state["jacobian"] = np.asarray(state["jacobian"], dtype=np.float64)
    return state


def run_server_checks(base_url: str) -> list[ServerCheckResult]:
    results: list[ServerCheckResult] = []
    try:
        post_json(base_url, "/clearerr")
        results.append(ServerCheckResult(True, "/clearerr reachable"))
    except Exception as exc:
        results.append(ServerCheckResult(False, f"/clearerr failed: {exc}"))
        return results

    try:
        state = get_state(base_url)
    except Exception as exc:
        results.append(ServerCheckResult(False, f"/getstate failed: {exc}"))
        return results

    expected_shapes = {
        "pose": (7,),
        "vel": (6,),
        "force": (3,),
        "torque": (3,),
        "q": (6,),
        "dq": (6,),
    }
    for key, expected in expected_shapes.items():
        actual = tuple(np.asarray(state[key]).shape)
        if actual == expected:
            results.append(ServerCheckResult(True, f"/getstate `{key}` shape OK: {actual}"))
        else:
            results.append(ServerCheckResult(False, f"/getstate `{key}` shape mismatch: got {actual}, expected {expected}"))

    jacobian_shape = tuple(state["jacobian"].shape)
    if jacobian_shape == (6, 6):
        results.append(ServerCheckResult(True, "/getstate `jacobian` shape OK: (6, 6)"))
    else:
        results.append(ServerCheckResult(False, f"/getstate `jacobian` shape mismatch: got {jacobian_shape}, expected (6, 6)"))

    try:
        pose = state["pose"].tolist()
        post_json(base_url, "/pose", {"arr": pose})
        results.append(ServerCheckResult(True, "/pose reachable with current pose"))
    except Exception as exc:
        results.append(ServerCheckResult(False, f"/pose failed with current pose: {exc}"))

    try:
        post_json(base_url, "/update_param", {"translational_stiffness": 2000, "translational_damping": 89})
        results.append(ServerCheckResult(True, "/update_param reachable"))
    except Exception as exc:
        results.append(ServerCheckResult(False, f"/update_param failed: {exc}"))

    try:
        post_json(base_url, "/jointreset")
        results.append(ServerCheckResult(True, "/jointreset reachable"))
    except Exception as exc:
        results.append(ServerCheckResult(False, f"/jointreset failed: {exc}"))

    return results


class KeyboardServerTeleop:
    def __init__(self, base_url: str, linear_step: float, angular_step_deg: float, hz: float):
        self.base_url = base_url.rstrip("/")
        self.linear_step = float(linear_step)
        self.angular_step = np.deg2rad(float(angular_step_deg))
        self.period = 1.0 / float(hz)
        self.target_pose = get_state(self.base_url)["pose"]
        self.running = True
        self.keyboard_intervention = self._build_keyboard_intervention()
        self.print_help()

    def print_help(self):
        print("Keyboard UR server check")
        print("  Use the existing KeyboardIntervention mapping from wrappers.py")
        print("  ;: toggle intervention on/off")
        print("  W/S: X +/-")
        print("  A/D: Y +/-")
        print("  J/K: Z +/-")
        print("  P: print current state")
        print("  C: clear error")
        print("  H: joint reset")
        print("  Q or Esc: quit")

    def _build_keyboard_intervention(self):
        from ur_env.envs.wrappers import KeyboardIntervention

        return KeyboardIntervention(_DummyKeyboardEnv())

    def on_press(self, key):
        from pynput import keyboard

        if key == keyboard.Key.esc:
            self.running = False
            return False

        try:
            key_str = key.char.lower()
        except AttributeError:
            return

        if key_str == "q":
            self.running = False
            return False
        if key_str == "p":
            self.print_state()
            return
        if key_str == "c":
            self.clear_error()
            return
        if key_str == "h":
            self.joint_reset()
            return

    def clear_error(self):
        try:
            result = post_json(self.base_url, "/clearerr")
            print(f"clearerr: {result}")
        except Exception as exc:
            print(f"clearerr failed: {exc}")

    def joint_reset(self):
        try:
            result = post_json(self.base_url, "/jointreset")
            print(f"jointreset: {result}")
            time.sleep(0.5)
            self.target_pose = get_state(self.base_url)["pose"]
            self.print_state()
        except Exception as exc:
            print(f"jointreset failed: {exc}")

    def print_state(self):
        try:
            state = get_state(self.base_url)
            pose = state["pose"]
            euler = R.from_quat(pose[3:]).as_euler("xyz", degrees=True)
            print(
                "state:"
                f" xyz=({pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f})"
                f" rpy_deg=({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f})"
                f" q_shape={state['q'].shape}"
                f" jacobian_shape={state['jacobian'].shape}"
            )
        except Exception as exc:
            print(f"print state failed: {exc}")

    def compute_pose_delta(self) -> np.ndarray:
        action, replaced = self.keyboard_intervention.action(np.zeros((6,), dtype=np.float32))
        if not replaced:
            return np.zeros((6,), dtype=np.float64)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        return np.concatenate([action[:3] * self.linear_step, action[3:6] * self.angular_step])

    def step(self):
        delta = self.compute_pose_delta()
        if not np.any(delta):
            return

        try:
            current_state = get_state(self.base_url)
            current_pose = current_state["pose"]
            target_pose = current_pose.copy()
            target_pose[:3] += delta[:3]
            current_rot = R.from_quat(current_pose[3:])
            delta_rot = R.from_euler("xyz", delta[3:])
            target_pose[3:] = (delta_rot * current_rot).as_quat()
            post_json(self.base_url, "/pose", {"arr": target_pose.tolist()})
            self.target_pose = target_pose
        except Exception as exc:
            print(f"/pose failed: {exc}")
            time.sleep(0.5)

    def run(self):
        from pynput import keyboard

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        self.print_state()
        try:
            while self.running:
                loop_start = time.time()
                self.step()
                time.sleep(max(0.0, self.period - (time.time() - loop_start)))
        finally:
            listener.stop()
            if hasattr(self.keyboard_intervention, "listener"):
                self.keyboard_intervention.listener.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:5000")
    parser.add_argument("--linear_step", type=float, default=0.002, help="Meters per update while a key is held.")
    parser.add_argument("--angular_step_deg", type=float, default=2.0, help="Degrees per update while a key is held.")
    parser.add_argument("--hz", type=float, default=8.0, help="Command rate for held keys.")
    parser.add_argument("--check_only", action="store_true", help="Run route checks only, skip interactive control.")
    args = parser.parse_args()

    print(f"Checking UR server at {args.server_url}")
    results = run_server_checks(args.server_url)
    has_failure = False
    for result in results:
        prefix = "PASS" if result.ok else "FAIL"
        print(f"[{prefix}] {result.message}")
        has_failure = has_failure or (not result.ok)

    if args.check_only:
        return 1 if has_failure else 0

    if has_failure:
        print("Server checks failed. Fix the FAIL items before interactive motion.")
        return 1

    try:
        import pynput  # noqa: F401
    except ModuleNotFoundError:
        print("Interactive keyboard control requires `pynput`. Install it before running without --check_only.")
        return 1

    print("Starting keyboard control. Keep the robot in a safe area.")
    teleop = KeyboardServerTeleop(
        base_url=args.server_url,
        linear_step=args.linear_step,
        angular_step_deg=args.angular_step_deg,
        hz=args.hz,
    )
    teleop.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
