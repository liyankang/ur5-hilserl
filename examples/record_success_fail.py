import copy
import os
import sys
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from scipy.spatial.transform import Rotation as Rot

# Ensure the repo root is importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glob

from experiments.mappings import CONFIG_MAPPING
from ur_env.utils.terminal_keyboard import TERMINAL_KEYBOARD_HUB
from examples.auto_trajectory import auto_move_to_target

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "ram_insertion", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful transitions to collect.")
flags.DEFINE_boolean("auto", False, "Auto-move to target and collect success samples.")
flags.DEFINE_float("auto_speed", 2.0, "Auto mode step size in mm.")
flags.DEFINE_boolean("resume", False, "Resume from latest existing data files.")


# ── Keyboard state tracking ──────────────────────────────────────────────────
# Since TERMINAL_KEYBOARD_HUB only fires on key-press (no release events),
# we use a timeout to determine if a key is still "held".
_key_times = {}           # key -> last press timestamp
_auto_mode = True         # True = auto (go to target), False = manual (keyboard)
_kb_interrupted = False   # Set True when ; pressed during auto → break auto loop
_key_timeout = 0.3        # seconds before a key is considered "released"
_kb_wrapper = None        # direct reference to KeyBoardIntervention wrapper


def _kb_handler(key_char: str):
    """Track key press times; ';' toggles auto/manual."""
    global _auto_mode, _kb_interrupted
    k = key_char.lower()
    _key_times[k] = time.monotonic()
    if k == ";":
        if _auto_mode:
            _kb_interrupted = True
        _auto_mode = not _auto_mode
        print(f"\n=== Mode: {'AUTO' if _auto_mode else 'MANUAL'} ===")


def _is_key_active(k):
    """Return True if key was pressed within _key_timeout."""
    return k in _key_times and (time.monotonic() - _key_times[k]) < _key_timeout


def _get_manual_action():
    """Build a 6D action from current keyboard state (WASDJK)."""
    action = np.zeros(6, dtype=np.float32)
    if _is_key_active("w"): action[1] += 1.0
    if _is_key_active("s"): action[1] -= 1.0
    if _is_key_active("a"): action[0] += 1.0
    if _is_key_active("d"): action[0] -= 1.0
    if _is_key_active("j"): action[2] += 1.0
    if _is_key_active("k"): action[2] -= 1.0
    return action


def _check_target_reached(base_env, target_pose):
    """Check if TCP is within reward threshold of target."""
    base_env._update_currpos()
    final_pose = base_env.currpos.copy()
    final_rot = Rot.from_quat(final_pose[3:]).as_matrix()
    target_rot = Rot.from_rotvec(target_pose[3:6]).as_matrix()
    diff_rot = final_rot.T @ target_rot
    diff_rotvec = Rot.from_matrix(diff_rot).as_rotvec()
    delta = np.abs(np.hstack([final_pose[:3] - target_pose[:3], diff_rotvec]))
    return np.all(delta < base_env._REWARD_THRESHOLD), delta


def main(_):
    global _auto_mode, _kb_interrupted

    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    # Register our keyboard handler
    unsub_kb = TERMINAL_KEYBOARD_HUB.register(_kb_handler)

    # Find KeyBoardIntervention wrapper directly
    global _kb_wrapper
    w = env
    while w is not None:
        if type(w).__name__ == 'KeyBoardIntervention':
            _kb_wrapper = w
            break
        w = getattr(w, 'env', None)
    if _kb_wrapper:
        print(f"Found KeyBoardIntervention wrapper")
    else:
        print(f"WARNING: KeyBoardIntervention wrapper not found!")

    obs, _ = env.reset()

    successes = []
    failures = []
    success_needed = FLAGS.successes_needed

    # ── Resume: load existing data ──
    if FLAGS.resume:
        data_dir = "./classifier_data"
        # Find latest success file
        success_files = sorted(glob.glob(f"{data_dir}/{FLAGS.exp_name}_*success*.pkl"))
        failure_files = sorted(glob.glob(f"{data_dir}/{FLAGS.exp_name}_*failure*.pkl"))
        if success_files:
            with open(success_files[-1], "rb") as f:
                successes = pkl.load(f)
            print(f"Resumed {len(successes)} successes from {success_files[-1]}")
        if failure_files:
            with open(failure_files[-1], "rb") as f:
                failures = pkl.load(f)
            print(f"Resumed {len(failures)} failures from {failure_files[-1]}")

    pbar = tqdm(total=success_needed, initial=len(successes))
    target_pose = env.unwrapped._TARGET_POSE
    base_count = len(successes)  # number of successes loaded from resume

    print("\n" + "=" * 60)
    print("  AUTO mode: robot moves to target automatically")
    print("  Press ; to intervene → MANUAL control (WASD/JK)")
    print("  Press ; again → resume AUTO")
    print("  Press SPACE to mark success (non-auto mode only)")
    print("=" * 60 + "\n")

    try:
        while (len(successes) - base_count) < success_needed:

            # ════════════════ AUTO MODE ════════════════
            if FLAGS.auto and _auto_mode:
                _kb_interrupted = False

                def interrupt_check():
                    return _kb_interrupted

                transitions, interrupted = auto_move_to_target(
                    env, target_pose,
                    speed_mm=FLAGS.auto_speed,
                    interrupt_check=interrupt_check,
                )

                # Restore terminal for tqdm/print output
                TERMINAL_KEYBOARD_HUB.suspend()

                if interrupted:
                    # User pressed ; → switch to manual
                    _kb_interrupted = False
                    print(f"Auto interrupted → manual mode ({len(transitions)} transitions discarded)")
                    TERMINAL_KEYBOARD_HUB.resume()
                    continue

                # Auto completed: check if target reached
                within, delta = _check_target_reached(env.unwrapped, target_pose)
                print(f"Auto completed: delta={delta}, within={within}")

                if within and len(transitions) > 0:
                    # Intermediate transitions → failures (classifier negative samples)
                    for t in transitions[:-1]:
                        failures.append(copy.deepcopy(t))
                    # Only the final transition → success
                    successes.append(copy.deepcopy(transitions[-1]))
                    pbar.update(1)
                    print(f"  +1 success, {len(transitions)-1} failure samples")
                else:
                    # Did not reach target → all transitions are failures
                    for t in transitions:
                        failures.append(copy.deepcopy(t))
                    print(f"Auto: did not reach target, {len(transitions)} failure samples")

                obs, _ = env.reset()
                TERMINAL_KEYBOARD_HUB.resume()
                continue

            # ════════════════ MANUAL MODE ════════════════
            action = _get_manual_action()
            # Force KeyBoardIntervention.intervened=True so keyboard input is used
            if _kb_wrapper is not None:
                _kb_wrapper.intervened = True
            next_obs, rew, done, truncated, info = env.step(action)
            if "intervene_action" in info:
                action = info["intervene_action"]

            transition = copy.deepcopy(dict(
                observations=obs,
                actions=action,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            ))
            obs = next_obs

            # Record transitions
            if FLAGS.auto:
                # Auto mode + manual intervention → all frames are failures
                failures.append(transition)
                if len(failures) % 20 == 0:
                    print(f"  [manual] {len(failures)} failure frames collected, action_norm={np.linalg.norm(action):.2f}")
            else:
                # Non-auto mode: SPACE marks success
                if _is_key_active(" "):
                    successes.append(transition)
                    pbar.update(1)
                    _key_times.pop(" ", None)  # consume
                else:
                    failures.append(transition)

            if done or truncated:
                if FLAGS.auto:
                    # Auto mode: manual was just intervention; reset → back to auto
                    obs, _ = env.reset()
                    _auto_mode = True
                    print("\n=== Mode: AUTO (after reset) ===")
                else:
                    # Non-auto: grace window for terminal success
                    deadline = time.time() + 1.0
                    marked = False
                    while time.time() < deadline and not _is_key_active(" "):
                        time.sleep(0.02)
                    if _is_key_active(" "):
                        successes.append(failures.pop())
                        pbar.update(1)
                        _key_times.pop(" ", None)
                    obs, _ = env.reset()
    finally:
        unsub_kb()
        # ── Save results even if interrupted ──
        if successes or failures:
            if not os.path.exists("./classifier_data"):
                os.makedirs("./classifier_data")
            uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"./classifier_data/{FLAGS.exp_name}_{success_needed}_success_images_{uuid}.pkl"
            with open(file_name, "wb") as f:
                pkl.dump(successes, f)
                print(f"\nsaved {len(successes)} successful transitions to {file_name}")

            file_name = f"./classifier_data/{FLAGS.exp_name}_failure_images_{uuid}.pkl"
            with open(file_name, "wb") as f:
                pkl.dump(failures, f)
                print(f"saved {len(failures)} failure transitions to {file_name}")
        else:
            print("\nNo data collected, nothing saved.")


if __name__ == "__main__":
    app.run(main)
