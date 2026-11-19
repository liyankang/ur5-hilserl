"""只读检查 USB 配置、实机状态与分类器验收。"""
import argparse
import json
import math
from pathlib import Path
import sys
from urllib.request import Request, urlopen

# 从任何工作目录运行，都使用同一仓库的纯标准库配置检查器。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.experiments.usb_insertion.runtime_config import (
    load_config,
    require_hardware_ready,
    resolve_config_path,
    resolve_path,
    vector,
)


def _probe_server(config):
    # getstate 仅查询。这里不下发运动，也不安装保护配置。
    url = config["robot"]["server_url"].rstrip("/") + "/getstate"
    request = Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=3.0) as response:
        state = json.load(response)
    if state.get("safety_stop"):
        raise ValueError("Controller safety stop is latched; inspect it on site before proceeding")
    for name, size in (("q", 6), ("force", 3), ("torque", 3), ("pose", 7)):
        vector(state.get(name), size, f"Server {name}")
    pose = state["pose"]
    if not math.isclose(math.hypot(*pose[3:]), 1.0, abs_tol=1e-3):
        raise ValueError("Server TCP orientation is not a unit quaternion")
    bounds = config["workspace_bounds"]
    outside = [axis for axis, value, low, high in zip(
        "XYZ", pose[:3], bounds["position_low_m"], bounds["position_high_m"]
    ) if not low - 1e-9 <= value <= high + 1e-9]
    return {
        "server_reachable": True,
        "server_safety": state.get("safety_guard"),
        "current_tcp_xyz_m": pose[:3],
        "current_tcp_quaternion_xyzw": pose[3:],
        "current_tcp_outside_workspace_axes": outside,
    }


def check_setup(path=None, stage="offline", probe_server=False):
    path = resolve_config_path(path)
    config = load_config(path)
    report = {
        "config_path": str(path),
        "schema_valid": True,
        "action_dim": config["action"]["action_dim"],
        "network_action_dim": config["training"]["network_action_dim"],
        "proprio_dim": 18,
        "gripper_enabled": False,
        "reward_input": "images_only",
        "classifier_path": str(resolve_path(config["reward_classifier"]["checkpoint_path"])),
        "reset_translation_range_m": config["reset"]["translation_range_m"],
        "reset_rotation_range_deg": config["reset"]["rotation_range_deg"],
        "workspace_bounds": config["workspace_bounds"],
        "nominal_inserted_tcp_pose_xyz_rotvec": config["task"]["nominal_inserted_tcp_pose_m_rotvec"],
        "nominal_reset_tcp_pose_xyz_rotvec": config["reset"]["nominal_tcp_pose_m_rotvec"],
        "stage": stage,
    }
    if probe_server:
        report.update(_probe_server(config))

    # 离线检查可查看结构；进入采集/训练前必须明确通过标定与模型检查。
    try:
        require_hardware_ready(config, classifier=(stage == "training"))
        outside = report.get("current_tcp_outside_workspace_axes", [])
        if outside:
            axes = ", ".join(outside)
            raise ValueError(f"Current TCP {report['current_tcp_xyz_m']} is outside workspace_bounds on {axes}")
        report["ready"] = True
    except (ValueError, OSError) as error:
        report["ready"] = False
        report["not_ready_reason"] = str(error)
        if stage != "offline":
            raise ValueError(report["not_ready_reason"]) from error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON 路径；省略时读 UR5_USB_TRAIN_CONFIG 或任务默认文件")
    parser.add_argument("--stage", choices=["offline", "collection", "training"], default="offline")
    parser.add_argument("--probe-server", action="store_true", help="额外只读查询 /getstate")
    args = parser.parse_args()
    try:
        report = check_setup(args.config, args.stage, args.probe_server)
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"ready": False, "error": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
