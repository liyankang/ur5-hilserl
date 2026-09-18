#!/usr/bin/env python3
"""自动分析 UR5 测试 CSV，并生成 AI 可读取的结构化结果。

按 CSV 的 phase 列自动区分两类测试：
  * 柔顺测试（compliance/sweep/push）：只有位置反馈可信，按力控启用段分析运动学表现
  * 力反馈测试（constant/impedance）：分析力的指令与实测偏差

UR5（CB3）没有力传感器，力读数由关节电流估算，在 force mode 下不可信，
因此柔顺测试里不把力作为判据。
"""

import argparse
import csv
import json
import math
from pathlib import Path

POSITION_PHASES = {"compliance", "sweep", "push"}
FORCE_PHASES = {"constant", "impedance"}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values, ratio):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * ratio
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def stats(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "p95_abs": percentile(sorted(abs(value) for value in values), 0.95),
    }


def load_csv(path):
    samples = []
    max_force = None
    max_torque = None
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if not row.get("t") or not row.get("mode"):
                if row.get("# max |F| (N):"):
                    max_force = row.get("# max |F| (N):")
                if row.get("# max |T| (Nm):"):
                    max_torque = row.get("# max |T| (Nm):")
                continue
            try:
                mode = int(row["mode"])
                sample = {
                    "t": float(row["t"]),
                    "mode": mode,
                    "phase": (row.get("phase") or "").strip(),
                    "enabled": int(row.get("enabled", 0)),
                    "pos": [float(row[f"p{i}"]) for i in range(6)],
                    "force": [float(row[f"f_meas{i}"]) for i in range(6)],
                    "wrench": [float(row[f"wrench{i}"]) for i in range(6)],
                }
            except (KeyError, TypeError, ValueError):
                continue
            samples.append(sample)
    return samples, {"max_force": max_force, "max_torque": max_torque}


def split_segments(samples):
    """按力控启用的连续区间切分样本。"""
    segments = []
    current = []
    for sample in samples:
        if sample["enabled"]:
            current.append(sample)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def analyze_compliance(samples, result):
    """柔顺测试分析：只依据位置反馈，力读数不参与判据。"""
    result["test_type"] = "compliance"
    segments = []
    for seg in split_segments(samples):
        if len(seg) < 5:
            continue
        base = seg[0]["pos"][:3]
        axes = {}
        for index, axis in enumerate(("x", "y", "z")):
            values = [s["pos"][index] - base[index] for s in seg]
            axes[axis] = {
                "range_mm": round((max(values) - min(values)) * 1000.0, 3),
                "end_offset_mm": round(values[-1] * 1000.0, 3),
            }
        segments.append(
            {
                "samples": len(seg),
                "duration_s": round(seg[-1]["t"] - seg[0]["t"], 3),
                "position_range_mm": axes,
                "mean_commanded_wrench_N": [
                    round(sum(s["wrench"][i] for s in seg) / len(seg), 3) for i in range(3)
                ],
            }
        )
    result["compliant_segments"] = segments

    if not segments:
        result["warnings"].append("没有识别到力控启用区间，请确认测试是否真正进入 force mode")
        return result

    widest = max(
        (seg["position_range_mm"][axis]["range_mm"], axis, seg["duration_s"])
        for seg in segments
        for axis in ("x", "y", "z")
    )
    result["max_position_range_mm"] = {
        "axis": widest[1],
        "range_mm": widest[0],
        "duration_s": widest[2],
    }
    if widest[0] > 10.0:
        result["warnings"].append(
            f"柔顺段在 {widest[1]} 轴位移范围达 {widest[0]:.1f} mm，"
            "可能是漂移或与工件碰触，建议结合现场确认"
        )

    notes = [
        "本文件来自柔顺测试：力读数由关节电流估算，不参与判据",
        "评价应看位置跟随、抖动、轴向串扰与松手回位精度",
    ]
    if any(max(abs(v) for v in s["force"]) > 50 for s in samples):
        notes.append(
            "文件中存在 >50 N 的力读数，这是该控制器在 force mode 下的估算伪值，并非真实接触力"
        )
    result["notes"] = notes
    return result


def analyze(samples, metadata):
    result = {
        "source_samples": len(samples),
        "time": {},
        "modes": {},
        "warnings": [],
    }
    if not samples:
        result["warnings"].append("没有读取到有效采样数据")
        return result

    result["time"] = {
        "start": samples[0]["t"],
        "end": samples[-1]["t"],
        "duration_s": samples[-1]["t"] - samples[0]["t"],
    }
    result["recorded_max_values"] = metadata

    phases = sorted({sample["phase"] for sample in samples if sample["phase"]})
    if phases:
        result["phases"] = phases
    if any(phase in POSITION_PHASES for phase in phases):
        return analyze_compliance(samples, result)

    for mode in sorted({sample["mode"] for sample in samples}):
        current = [sample for sample in samples if sample["mode"] == mode]
        mode_result = {
            "samples": len(current),
            "enabled_ratio": sum(s["enabled"] for s in current) / len(current),
            "measured_force_xyz_N": {
                axis: stats([sample["force"][index] for sample in current])
                for index, axis in enumerate(("x", "y", "z"))
            },
            "measured_torque_xyz_Nm": {
                axis: stats([sample["force"][index] for sample in current])
                for index, axis in enumerate(("rx", "ry", "rz"), start=3)
            },
        }

        if mode == 2:
            errors = []
            requested = []
            for sample in current:
                for measured, command in zip(sample["force"][:3], sample["wrench"][:3]):
                    if abs(command) > 1e-9:
                        requested.append(command)
                        errors.append(measured - command)
            mode_result["constant_force_command_N"] = stats(requested)
            mode_result["force_error_N"] = stats(errors)
            if errors and max(abs(error) for error in errors) > 2.0:
                result["warnings"].append("模式 2 存在超过 2 N 的恒力跟踪误差")

        if mode == 3:
            position_errors = []
            wrench_errors = []
            for sample in current:
                for index in range(3):
                    position_errors.append(sample["wrench"][index])
                    wrench_errors.append(sample["force"][index] - sample["wrench"][index])
            mode_result["requested_force_xyz_N"] = stats(position_errors)
            mode_result["force_tracking_error_xyz_N"] = stats(wrench_errors)
            if any(abs(error) > 5.0 for error in wrench_errors):
                result["warnings"].append("模式 3 存在超过 5 N 的力跟踪偏差")

        result["modes"][str(mode)] = mode_result

    mode1 = result["modes"].get("1")
    if mode1:
        noise = [mode1["measured_force_xyz_N"][axis]["std"] for axis in ("x", "y", "z")]
        result["monitoring"] = {"force_noise_std_xyz_N": noise}
        if max(noise, default=0.0) > 0.5:
            result["warnings"].append("模式 1 静止力噪声超过 0.5 N，建议检查皮重、负载参数或传感器漂移")

    if any(sample["enabled"] and max(abs(value) for value in sample["force"]) > 50 for sample in samples):
        result["warnings"].append("检测到超过 50 N 的实测力，请确认机器人和工件处于安全状态")
    return result


def build_ai_prompt(result):
    if result.get("test_type") == "compliance":
        return (
            "请分析下面这份 UR5 柔顺控制测试结果。注意：该机器人是 UR5（CB3 代），法兰内没有力传感器，"
            "文件中的力读数由关节电流估算，在 force mode 下会输出几十牛的伪值，不要用它判断接触力。"
            "请重点判断：1）各柔顺段的位移范围是否合理，有无漂移或与工件碰触的迹象；"
            "2）力控启用段的数量与时长是否符合预期，有没有段被意外中断；"
            "3）位置跟随表现是否稳定，下一步应调整哪些参数或补做哪些测试。"
            "不要臆造没有提供的数据。\n\n"
            + json.dumps(result, ensure_ascii=False, indent=2)
        )
    return (
        "请分析下面这份 UR5 力控测试结果。重点判断：1）力传感器噪声和零漂是否正常；"
        "2）恒力模式的指令与实测偏差；3）阻抗模式是否稳定；4）是否存在安全风险；"
        "5）下一步应该如何调整参数或重新测试。不要臆造没有提供的数据。\n\n"
        + json.dumps(result, ensure_ascii=False, indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description="分析 UR5 force mode CSV")
    parser.add_argument("csv", nargs="?", help="CSV 路径；省略时自动选择当前目录最新的 force*.csv")
    parser.add_argument("--out-dir", default=None, help="输出目录，默认与输入 CSV 相同")
    args = parser.parse_args()

    if args.csv:
        source = Path(args.csv)
    else:
        candidates = sorted(Path.cwd().glob("force*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not candidates:
            parser.error("当前目录没有找到 force*.csv，请指定 CSV 路径")
        source = candidates[0]
    if not source.exists():
        parser.error(f"CSV 文件不存在: {source}")

    samples, metadata = load_csv(source)
    result = analyze(samples, metadata)
    output_dir = Path(args.out_dir) if args.out_dir else source.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{source.stem}_analysis.json"
    prompt_path = output_dir / f"{source.stem}_ai_prompt.txt"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_path.write_text(build_ai_prompt(result), encoding="utf-8")

    print(f"已读取 {source}：{len(samples)} 条有效数据")
    print(f"结构化结果：{json_path}")
    print(f"AI 分析文本：{prompt_path}")

    if result.get("test_type") == "compliance":
        segments = result.get("compliant_segments") or []
        print(f"测试类型：柔顺（{', '.join(result.get('phases', []))}）")
        print(f"识别到 {len(segments)} 个力控启用段")
        for index, seg in enumerate(segments, start=1):
            rng = seg["position_range_mm"]
            print(
                f"  段{index} {seg['duration_s']:5.2f}s  位移范围 "
                f"x {rng['x']['range_mm']:6.2f}  y {rng['y']['range_mm']:6.2f}  "
                f"z {rng['z']['range_mm']:6.2f} mm"
            )
        for note in result.get("notes", []):
            print(f"  注: {note}")

    if result["warnings"]:
        print("发现问题：")
        for warning in result["warnings"]:
            print(f"- {warning}")
    else:
        print("未发现预设阈值问题，请继续结合机器人实际工况判断。")


if __name__ == "__main__":
    main()
