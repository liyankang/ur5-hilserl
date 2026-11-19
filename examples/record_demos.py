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
from examples.collection_utils import (
    make_transition, label_success, save_pickle, validate_action_space,
    validate_transition_shapes, apply_manual_success,
)


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # USB 示范以视觉分类器判定成功；采集前先训练并验证成功检测器。
    env = config.get_environment(fake_env=False, save_video=False,
                                 classifier=FLAGS.exp_name == "usb_insertion")
    validate_action_space(config, env)
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
        elif k == "h":
            failure_requested = True

    unsubscribe_restart = TERMINAL_KEYBOARD_HUB.register(_on_key)
    print("Hotkeys: 'f' = success, 'h' = failure, 'r' = discard & reset; 'g' = go to target")

    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    resume_path = None

    try:
        obs, info = env.reset()
        print("Reset done")
        transitions = []
        success_count = 0
        success_needed = FLAGS.successes_needed

        # ── Resume: load existing data ──
        if FLAGS.resume:
            data_dir = "./demo_data"
            demo_files = sorted(glob.glob(f"{data_dir}/{FLAGS.exp_name}_*demos*.pkl"), key=os.path.getmtime)
            if demo_files:
                resume_path = demo_files[-1]
                with open(demo_files[-1], "rb") as f:
                    transitions = pkl.load(f)
                for transition in transitions:
                    validate_transition_shapes(transition, env.observation_space, env.action_space, resume_path)
                success_count = len([t for t in transitions if t.get("infos", {}).get("succeed", False)])
                print(f"Resumed {len(transitions)} transitions ({success_count} successes) from {demo_files[-1]}")

        pbar = tqdm(total=success_needed, initial=success_count)
        trajectory = []
        returns = 0
        
        while success_count < success_needed:
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
                trajectory, _interrupted = auto_move_to_target(
                    env, target, speed_mm=FLAGS.auto_speed, initial_obs=obs,
                    interrupt_check=lambda: restart_requested or failure_requested or success_requested)

                # Restore terminal for tqdm/print output
                TERMINAL_KEYBOARD_HUB.suspend()

                # 自动控制仅产生动作，绝不根据“接近坐标”把 USB 标成成功。
                if trajectory and not restart_requested and not failure_requested and (
                    trajectory[-1]["infos"].get("succeed", False) or success_requested
                ):
                    if success_requested:
                        info = trajectory[-1]["infos"]
                        info["manual_success"] = True
                        apply_manual_success(0.0, False, False, info, env=env)
                    label_success(trajectory[-1])
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
                restart_requested = success_requested = failure_requested = False
                obs, info = env.reset()

                # Re-enter raw mode for next iteration
                TERMINAL_KEYBOARD_HUB.resume()
                continue

            # ---- Manual mode ----
            actions = np.zeros(env.action_space.sample().shape)
            next_obs, rew, done, truncated, info = env.step(actions)

            # Manual override: key press takes priority over auto detection
            if (success_requested or info.get("manual_success", False)) and not info.get("safety_stop", False):
                info["manual_success"] = True
                rew, done, truncated, info = apply_manual_success(rew, done, truncated, info, env=env)
                success_requested = False
            elif failure_requested:
                info["succeed"] = False
                rew = 0.0
                done = True
                truncated = False
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
            transition = make_transition(obs, actions, next_obs, rew, done, truncated, info)
            trajectory.append(transition)
            
            pbar.set_description(f"Return: {returns}")

            obs = next_obs
            if done or truncated:
                if info.get("succeed", False):
                    for transition in trajectory:
                        transitions.append(copy.deepcopy(transition))
                    success_count += 1
                    pbar.update(1)
                trajectory = []
                returns = 0
                obs, info = env.reset()
    finally:
        unsubscribe_restart()
        try:
            # ── Save results even if interrupted ──
            if transitions:
                if not os.path.exists("./demo_data"):
                    os.makedirs("./demo_data")
                uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_name = resume_path or f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
                save_pickle(file_name, transitions)
                print(f"\nsaved {len(transitions)} transitions ({success_count} successes) to {file_name}")
            else:
                print("\nNo transitions collected, nothing saved.")
        finally:
            env.close()

if __name__ == "__main__":
    app.run(main)
