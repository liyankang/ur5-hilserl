"""奖励样本分组、指标计算和模型验收；不导入 JAX，也不连接机器人。"""

import hashlib
import json
import math
import re
from pathlib import Path


def sample_group(transition, source_path):
    """同一回合始终留在同一数据分区；旧文件按整次采集批次分组。"""
    info = transition.get("infos", {})
    episode = transition.get("episode_id", info.get("episode_id"))
    if episode is not None:
        return "episode:" + str(episode)
    # 旧版正负文件共享 images_ 后的时间戳，必须一起进入训练或验证。
    stem = Path(source_path).stem
    batch = stem.rsplit("images_", 1)[-1] if "images_" in stem else str(Path(source_path).resolve())
    return "capture:" + batch


def split_grouped_examples(examples, val_ratio, rng):
    """同时划分两类样本，禁止正负文件中的同一回合跨训练/验证泄漏。"""
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    groups = {}
    for example in examples:
        group = example.get("validation_group")
        if group is None:
            raise ValueError("Each classifier example needs a validation_group")
        groups.setdefault(group, set()).add(int(example["labels"]))
    names = list(groups)
    if len(names) < 2:
        raise ValueError("Need multiple independent episodes/capture batches for held-out validation")
    if any(sum(label in values for values in groups.values()) < 2 for label in (0, 1)):
        raise ValueError("Both labels need samples in at least two independent episodes/capture batches")
    # 两个分区都必须有正负样本；只有一次采集的旧数据不能伪造独立验证集。
    target = max(1, min(len(names) - 1, round(len(names) * val_ratio)))
    sizes = sorted(range(1, len(names)), key=lambda n: abs(n - target))
    for size in sizes:
        for _ in range(128):
            shuffled = names.copy()
            rng.shuffle(shuffled)
            held_out = set(shuffled[:size])
            train_labels = set().union(*(groups[g] for g in names if g not in held_out))
            val_labels = set().union(*(groups[g] for g in names if g in held_out))
            if train_labels == {0, 1} and val_labels == {0, 1}:
                train = [e for e in examples if e["validation_group"] not in held_out]
                validation = [e for e in examples if e["validation_group"] in held_out]
                return train, validation
    raise ValueError("Cannot hold out complete groups with both classes; collect more labeled episodes")


def validation_metrics(labels, probabilities, threshold):
    """按实际部署阈值计算指标，尤其单独报告未插入却误报成功的比例。"""
    if not 0 < threshold < 1:
        raise ValueError("Success threshold must be between 0 and 1")
    labels, probabilities = list(labels), list(probabilities)
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("Labels and probabilities must be nonempty and equally sized")
    positive = negative = true_positive = true_negative = 0
    for label, probability in zip(labels, probabilities):
        probability = float(probability)
        if label not in (0, 1) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Validation needs binary labels and finite probabilities in [0, 1]")
        if label == 1:
            positive += 1
            true_positive += probability >= threshold
        else:
            negative += 1
            true_negative += probability < threshold
    if not positive or not negative:
        raise ValueError("Validation must contain both fully inserted and negative examples")
    positive_accuracy = true_positive / positive
    negative_accuracy = true_negative / negative
    accuracy = (true_positive + true_negative) / len(labels)
    return {
        "validation_accuracy": accuracy,
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "false_positive_rate": 1.0 - negative_accuracy,
        "positive_count": positive,
        "negative_count": negative,
        # 所有门槛都严格大于95%，不让类别不平衡掩盖误报。
        "passed": accuracy > 0.95 and positive_accuracy > 0.95 and negative_accuracy > 0.95,
    }


def checkpoint_fingerprint(path):
    """同时支持单文件 Flax 检查点和 Orbax 检查点目录。"""
    path = Path(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"Checkpoint has no saved parameter files: {path}")
    digest = hashlib.sha256()
    for file in files:
        # 加入相对文件名，避免不同目录布局恰好具有相同字节拼接。
        digest.update((file.name if path.is_file() else file.relative_to(path).as_posix()).encode("utf-8"))
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def latest_classifier_checkpoint(directory):
    """按检查点步数选择最新模型，不把临时文件或旧模型报告当成当前模型。"""
    candidates = []
    for path in Path(directory).glob("checkpoint_*"):
        match = re.fullmatch(r"checkpoint_(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ValueError(f"No trained classifier checkpoint in {directory}")
    return max(candidates, key=lambda item: item[0])


def validate_classifier_artifact(checkpoint_dir, exp_name, image_keys, use_proprio, threshold,
                                 min_accuracy=0.95):
    """在线运动前检查：部署配置、验证指标与真实最新权重必须一致。"""
    directory = Path(checkpoint_dir)
    report_path = directory / "validation.json"
    if not report_path.is_file():
        raise ValueError(f"Missing classifier validation report: {report_path}")
    with report_path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("exp_name") != exp_name:
        raise ValueError("Classifier report belongs to another experiment")
    if report.get("use_proprio") is not bool(use_proprio):
        raise ValueError("Classifier proprioception setting does not match deployment")
    if report.get("image_keys") != list(image_keys):
        raise ValueError("Classifier camera keys/order do not match deployment")
    if not math.isclose(float(report.get("threshold", -1)), float(threshold), abs_tol=1e-12):
        raise ValueError("Classifier was not validated at the deployment success threshold")
    if report.get("split_method") != "grouped_by_episode_or_capture":
        raise ValueError("Classifier validation must hold out independent episodes/capture groups")
    for field in ("validation_accuracy", "positive_accuracy", "negative_accuracy"):
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid classifier metric: {field}")
        if not min_accuracy < value <= 1:
            raise ValueError(f"Classifier {field} must be strictly greater than {min_accuracy}")
    false_positive = report.get("false_positive_rate")
    if not isinstance(false_positive, (int, float)) or not math.isfinite(false_positive):
        raise ValueError("Invalid classifier false_positive_rate")
    if not math.isclose(false_positive, 1 - report["negative_accuracy"], abs_tol=1e-9):
        raise ValueError("Classifier false-positive metric is inconsistent")
    if report.get("passed") is not True:
        raise ValueError("Classifier validation did not pass")
    step, checkpoint = latest_classifier_checkpoint(directory)
    if type(report.get("checkpoint_step")) is not int or report["checkpoint_step"] != step:
        raise ValueError("Validation report does not describe the latest saved checkpoint")
    if report.get("checkpoint_sha256") != checkpoint_fingerprint(checkpoint):
        raise ValueError("Classifier weights changed after validation")
    return report
