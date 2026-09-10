# import debugpy
# debugpy.listen(10010)
# print('wait debugger')
# debugpy.wait_for_client()
# print("Debugger Attached")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import glob

from scipy.spatial.transform import Rotation as Rot

from experiments.mappings import CONFIG_MAPPING
from ur_env.utils.terminal_keyboard import TERMINAL_KEYBOARD_HUB

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", 'ram_insertion', "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_boolean("auto", False, "Auto-generate trajectories to target pose.")
flags.DEFINE_float("auto_speed", 2.0, "Auto mode step size in mm (default 2.0mm per step).")
flags.DEFINE_boolean("resume", False, "Resume from latest existing demo data.")

from examples.auto_trajectory import auto_move_to_target


def _compute_auto_action(obs, info, target_pose, speed=0.5):
    """Compute action in EE frame to move TCP toward target pose.

    The action flows through RelativeFrame which transforms EE→base, so we
    compute the desired base-frame delta first, then rotate it into the EE frame.
    """
    base_pose = info["original_state_obs"]["tcp_pose"]
    current_xyz = base_pose[:3]
    current_quat = base_pose[3:7]

    # --- translation: base-frame error → EE-frame action ---
    pos_err = target_pose[:3] - current_xyz
    R_ee = Rot.from_quat(current_quat).as_matrix()
    R_ee_inv = R_ee.T
    pos_action_ee = R_ee_inv @ pos_err

    # --- rotation: base-frame rotvec error → EE-frame action ---
    target_rot = Rot.from_rotvec(target_pose[3:6]).as_matrix()
    current_rot = Rot.from_quat(current_quat).as_matrix()
    diff_rot = current_rot.T @ target_rot
    rot_err_rotvec = Rot.from_matrix(diff_rot).as_rotvec()
    rot_action_ee = R_ee_inv @ rot_err_rotvec

    # Normalise so that the largest component ≈ 1 when far away, then scale
    combined = np.concatenate([pos_action_ee, rot_action_ee])
    norm = np.linalg.norm(combined)
    if norm < 1e-6:
        return np.zeros(6, dtype=np.float32)
    action_ee = (combined / norm) * speed

    full_action = np.zeros(6, dtype=np.float32)
    full_action[:3] = np.clip(action_ee[:3], -1.0, 1.0)
    full_action[3:6] = np.clip(action_ee[3:6], -1.0, 1.0)
    return full_action


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    restart_requested = False
    success_requested = False
    failure_requested = False

    def _on_key(key_char: str):
        nonlocal restart_requested, success_requested, failure_requested
        k = key_char.lower()
        if k == "r":
            restart_requested = True
        elif k == "f":
            success_requested = True
        elif k == "g":
            failure_requested = True

    unsubscribe_restart = TERMINAL_KEYBOARD_HUB.register(_on_key)
    print("Hotkeys: 'f' = mark success, 'g' = mark failure, 'r' = discard & reset")

    try:
        obs, info = env.reset()
        print("Reset done")
        transitions = []
        success_count = 0
        success_needed = FLAGS.successes_needed

        # ── Resume: load existing data ──
        if FLAGS.resume:
            data_dir = "./demo_data"
            demo_files = sorted(glob.glob(f"{data_dir}/{FLAGS.exp_name}_*demos*.pkl"))
            if demo_files:
                with open(demo_files[-1], "rb") as f:
                    transitions = pkl.load(f)
                success_count = len([t for t in transitions if t.get("infos", {}).get("succeed", False)])
                print(f"Resumed {len(transitions)} transitions ({success_count} successes) from {demo_files[-1]}")

        pbar = tqdm(total=success_needed, initial=success_count)
        base_success = success_count  # number of successes loaded from resume
        trajectory = []
        returns = 0
        
        while (success_count - base_success) < success_needed:
            if restart_requested:
                restart_requested = False
                success_requested = False
                failure_requested = False
                trajectory = []
                returns = 0
                obs, info = env.reset()
                pbar.set_description("Episode discarded (manual reset)")
                continue

            if FLAGS.auto:
                # Direct position control: move to target and collect trajectory
                target = env.unwrapped._TARGET_POSE
                trajectory, _interrupted = auto_move_to_target(env, target, speed_mm=FLAGS.auto_speed)

                # Restore terminal for tqdm/print output
                TERMINAL_KEYBOARD_HUB.suspend()

                # Check if final pose is within threshold
                base_env = env.unwrapped
                base_env._update_currpos()
                final_pose = base_env.currpos.copy()
                final_rot = Rot.from_quat(final_pose[3:]).as_matrix()
                target_rot = Rot.from_rotvec(target[3:6]).as_matrix()
                diff_rot = final_rot.T @ target_rot
                diff_rotvec = Rot.from_matrix(diff_rot).as_rotvec()
                delta = np.abs(np.hstack([final_pose[:3] - target[:3], diff_rotvec]))
                within_threshold = np.all(delta < base_env._REWARD_THRESHOLD)
                print(f"\nFinal delta: {delta}, threshold: {base_env._REWARD_THRESHOLD}, "
                      f"within: {within_threshold}")

                if within_threshold or success_requested:
                    success_requested = False
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                elif failure_requested:
                    failure_requested = False
                    pbar.set_description("Episode marked as failure")

                trajectory = []
                returns = 0
                obs, info = env.reset()

                # Re-enter raw mode for next iteration
                TERMINAL_KEYBOARD_HUB.resume()
                continue

            # ---- Manual mode ----
            actions = np.zeros(env.action_space.sample().shape)
            next_obs, rew, done, truncated, info = env.step(actions)

            # Manual override: key press takes priority over auto detection
            if success_requested:
                info["succeed"] = True
                done = True
                success_requested = False
            elif failure_requested:
                info["succeed"] = False
                done = True
                failure_requested = False

            if restart_requested:
                restart_requested = False
                success_requested = False
                failure_requested = False
                trajectory = []
                returns = 0
                obs, info = env.reset()
                pbar.set_description("Episode discarded (manual reset)")
                continue

            returns += rew
            if "intervene_action" in info:
                actions = info["intervene_action"]
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            trajectory.append(transition)
            
            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            if done:
                if info["succeed"]:
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                trajectory = []
                returns = 0
                obs, info = env.reset()
    finally:
        unsubscribe_restart()
        # ── Save results even if interrupted ──
        if transitions:
            if not os.path.exists("./demo_data"):
                os.makedirs("./demo_data")
            uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
            with open(file_name, "wb") as f:
                pkl.dump(transitions, f)
                print(f"\nsaved {len(transitions)} transitions ({success_count} successes) to {file_name}")
        else:
            print("\nNo transitions collected, nothing saved.")

if __name__ == "__main__":
    app.run(main)
