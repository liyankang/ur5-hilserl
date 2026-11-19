"""人工明确标注插接成功/失败；每次成功后结束回合，避免继续挤压。"""

import datetime
import glob
import os
from pathlib import Path
import pickle as pkl
import sys
import time
import uuid

import numpy as np
from tqdm import tqdm
from absl import app, flags

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mappings import CONFIG_MAPPING
from ur_env.utils.terminal_keyboard import TERMINAL_KEYBOARD_HUB
from examples.collection_utils import make_transition, classifier_sample, save_pickle

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "usb_insertion", "要采集的任务名称。")
flags.DEFINE_integer("successes_needed", 100, "成功样本总数，包含续采文件中的样本。")
flags.DEFINE_boolean("resume", False, "在最新采集快照上续采。")
flags.DEFINE_boolean("auto", False, "已禁用：视觉奖励样本必须由人明确标注。")
flags.DEFINE_float("auto_speed", 2.0, "保留旧命令兼容；人工标注模式不使用此参数。")

# 键盘线程只提交请求；主循环负责在一次真实 step 后消费，避免等待相机时丢键。
_requests = {"success": False, "failure": False, "reset": False}
_key_times = {}


def _kb_handler(key_char):
    key = key_char.lower()
    _key_times[key] = time.monotonic()
    if key in ("f", " "):
        _requests["success"] = True
    elif key == "h":
        _requests["failure"] = True
    elif key == "r":
        _requests["reset"] = True


def _consume_request():
    """复位优先；未标注的图像不擅自当作负样本。"""
    result = None
    if _requests["reset"]:
        result = "reset"
    elif _requests["success"]:
        result = "success"
    elif _requests["failure"]:
        result = "failure"
    for key in _requests:
        _requests[key] = False
    return result


def _get_manual_action(action_dim):
    """没有键盘包装器时的备用输入，动作中始终没有夹爪维度。"""
    action = np.zeros(action_dim, dtype=np.float32)
    bindings = {"a": (0, 1), "d": (0, -1), "w": (1, 1), "s": (1, -1),
                "j": (2, 1), "k": (2, -1), "z": (3, 1), "x": (3, -1),
                "v": (4, 1), "b": (4, -1), "n": (5, 1), "m": (5, -1)}
    now = time.monotonic()
    for key, (axis, direction) in bindings.items():
        if axis < action_dim and now - _key_times.get(key, -1e9) < 0.3:
            action[axis] += direction
    return action


def main(_):
    if FLAGS.auto:
        raise ValueError("Reward data require manual labels; --auto must not infer success from TARGET_POSE")
    if FLAGS.successes_needed <= 0:
        raise ValueError("successes_needed must be positive")
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    successes, failures = [], []
    success_path = failure_path = None
    data_dir = Path(getattr(config, "classifier_data_dir", "classifier_data"))
    # 先加载文件再连接硬件；坏文件不能触发机器人复位。
    if FLAGS.resume:
        for kind, destination in (("success", successes), ("failure", failures)):
            paths = sorted(glob.glob(str(data_dir / f"{FLAGS.exp_name}_*{kind}*.pkl")), key=os.path.getmtime)
            if paths:
                with open(paths[-1], "rb") as stream:
                    destination.extend(pkl.load(stream))
                if kind == "success":
                    success_path = paths[-1]
                else:
                    failure_path = paths[-1]

    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    # 采集阶段仅人工确认成功，不能让坐标阈值抢先结束插入动作。
    env.unwrapped.use_pose_reward = False
    unsubscribe = TERMINAL_KEYBOARD_HUB.register(_kb_handler)
    keyboard = env
    while keyboard is not None and type(keyboard).__name__ not in ("KeyBoardIntervention", "KeyboardIntervention"):
        keyboard = getattr(keyboard, "env", None)
    pbar = tqdm(total=FLAGS.successes_needed, initial=len(successes))
    try:
        obs, _ = env.reset()
        episode_id = str(uuid.uuid4())
        _consume_request()
        print("WASD/JK 平移，ZX/VB/NM 旋转；H 明确标未插入（可长按）；F/空格标插入成功并复位；R 丢弃当前步并复位。")
        print("未明确标注的帧不会保存；相同回合通过 episode_id 保持在同一验证分组。")
        last_report = time.monotonic()
        while len(successes) < FLAGS.successes_needed:
            if keyboard is not None:
                keyboard.intervened = True
                # G 是自动追目标的旧快捷键，这个入口始终采用人工运动与标签。
                keyboard._go_to_target = False
            action = _get_manual_action(env.action_space.shape[0])
            started = time.monotonic()
            next_obs, reward, terminated, truncated, info = env.step(action)
            elapsed = time.monotonic() - started
            requested = _consume_request()
            if info.get("manual_success") and requested != "reset":
                requested = "success"
            if requested in ("success", "failure"):
                # 已执行动作优先，保证干预数据不会写成原来的策略动作。
                action = info.get("applied_action", info.get("intervene_action", action))
                info = dict(info, episode_id=episode_id)
                transition = make_transition(obs, action, next_obs, reward, terminated, truncated, info)
                transition["actions"] = np.asarray(action, dtype=np.float32).copy()
                sample = classifier_sample(transition, requested == "success")
                sample["episode_id"] = episode_id
                sample["label_source"] = "human_key"
                if requested == "success":
                    successes.append(sample)
                    pbar.update(1)
                else:
                    failures.append(sample)
            obs = next_obs
            if time.monotonic() - last_report >= 1:
                pbar.set_postfix(negative=len(failures), step_hz=f"{1 / max(elapsed, 1e-6):.1f}")
                last_report = time.monotonic()
            if requested in ("success", "reset") or terminated or truncated:
                if requested == "success":
                    hold = getattr(env.unwrapped, "hold_position", None)
                    if hold is not None:
                        hold()
                # 成功标注后先结束此回合，不再生成下一步向端口推进的动作。
                if len(successes) >= FLAGS.successes_needed:
                    break
                obs, _ = env.reset()
                episode_id = str(uuid.uuid4())
                _consume_request()
    finally:
        unsubscribe()
        pbar.close()
        try:
            if successes or failures:
                stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                success_path = success_path or data_dir / f"{FLAGS.exp_name}_{FLAGS.successes_needed}_success_images_{stamp}.pkl"
                failure_path = failure_path or data_dir / f"{FLAGS.exp_name}_failure_images_{stamp}.pkl"
                save_pickle(success_path, successes)
                save_pickle(failure_path, failures)
                print(f"Saved {len(successes)} positives to {success_path}; {len(failures)} negatives to {failure_path}")
        finally:
            env.close()


if __name__ == "__main__":
    app.run(main)
