"""用独立标注数据检查分类器；默认只检查报告中留出的完整回合。"""

import argparse
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import jax
import numpy as np
from experiments.mappings import CONFIG_MAPPING
from serl_launcher.networks.reward_classifier import load_classifier_func
from examples.classifier_data import (
    sample_group, validation_metrics, latest_classifier_checkpoint, checkpoint_fingerprint,
)


def _images_and_optional_state(obs, image_keys, use_proprio):
    """兼容旧版 images 嵌套字段和当前扁平相机字段，不手写时间维形状。"""
    result = {}
    images = obs.get("images", obs)
    aliases = {"global_1": "wrist_1", "wrist": "wrist_2", "global_2": "wrist_3"}
    for key in image_keys:
        if key in images:
            result[key] = images[key]
        elif aliases.get(key) in images:
            result[key] = images[aliases[key]]
        else:
            raise ValueError(f"Missing classifier camera: {key}")
    if use_proprio:
        result["state"] = obs["state"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", choices=CONFIG_MAPPING, default="ram_insertion")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--all_data", action="store_true", help="仅诊断：检查全部数据，不代表独立验证准确率")
    args = parser.parse_args()
    config = CONFIG_MAPPING[args.exp_name]()
    use_proprio = getattr(config, "classifier_use_proprio", True)
    threshold = args.threshold if args.threshold is not None else getattr(config, "classifier_success_threshold", 0.95)
    data_dir = Path(args.data_dir or getattr(config, "classifier_data_dir", "classifier_data"))
    # 缺少报告时拒绝把训练样本上的拟合准确率冒充离线泛化结果。
    held_out = None
    if not args.all_data:
        report_file = Path(config.classifier_checkpoint_path) / "validation.json"
        report = json.loads(report_file.read_text(encoding="utf-8"))
        step, checkpoint = latest_classifier_checkpoint(config.classifier_checkpoint_path)
        if report.get("checkpoint_step") != step or report.get("checkpoint_sha256") != checkpoint_fingerprint(checkpoint):
            raise ValueError("Validation groups do not describe the currently saved classifier")
        if report.get("exp_name") != args.exp_name or report.get("image_keys") != list(config.classifier_keys):
            raise ValueError("Validation report camera/task settings do not match this experiment")
        if report.get("use_proprio") is not bool(use_proprio):
            raise ValueError("Validation report network inputs do not match the current classifier")
        held_out = set(report["validation_groups"])
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)
    sample = env.observation_space.sample()
    classifier = load_classifier_func(
        jax.random.PRNGKey(0), sample, config.classifier_keys,
        config.classifier_checkpoint_path, use_proprio=use_proprio,
    )
    labels, probabilities = [], []
    for kind, label in (("success", 1), ("failure", 0)):
        for path in sorted(data_dir.glob(f"{args.exp_name}_*{kind}*.pkl")):
            with path.open("rb") as stream:
                records = pickle.load(stream)
            for record in records:
                if held_out is not None and sample_group(record, path) not in held_out:
                    continue
                obs = record.get("next_observations", record["observations"])
                obs = _images_and_optional_state(obs, config.classifier_keys, use_proprio)
                probability = float(np.asarray(jax.nn.sigmoid(classifier(obs))).item())
                labels.append(label)
                probabilities.append(probability)
    metrics = validation_metrics(labels, probabilities, threshold)
    print(json.dumps(dict(metrics, threshold=threshold, independent_validation=not args.all_data), indent=2))
    # fake_env 没有创建相机；无需调用会访问硬件清理字段的基类 close。


if __name__ == "__main__":
    main()
