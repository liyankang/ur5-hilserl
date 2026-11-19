"""只读检查 USB 配置与分类器验收；默认不连接硬件，更不会发送运动命令。"""
import argparse
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

# 从任何工作目录运行，都使用同一仓库的纯标准库配置检查器。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples.experiments.usb_insertion.runtime_config import load_config, require_hardware_ready, resolve_path


def check_setup(path=None, stage="offline", probe_server=False):
    config = load_config(path)
    report = {
        "schema_valid": True,
        "action_dim": config["action"]["action_dim"],
        "network_action_dim": config["training"]["network_action_dim"],
        "proprio_dim": 18,
        "gripper_enabled": False,
        "reward_input": "images_only",
        "classifier_path": str(resolve_path(config["reward_classifier"]["checkpoint_path"])),
        "reset_translation_range_m": config["reset"]["translation_range_m"],
        "reset_rotation_range_deg": config["reset"]["rotation_range_deg"],
        "stage": stage,
    }
    # 离线检查可查看结构；进入采集/训练前必须明确通过标定与模型检查。
    try:
        require_hardware_ready(config, classifier=(stage == "training"))
        report["ready"] = True
    except (ValueError, OSError) as error:
        report["ready"] = False
        report["not_ready_reason"] = str(error)
        if stage != "offline":
            raise ValueError(report["not_ready_reason"]) from error
    if probe_server:
        # getstate 仅查询；这里不调用 configure_safety/reset/pose/jointreset。
        url = config["robot"]["server_url"].rstrip("/") + "/getstate"
        with urlopen(Request(url, data=b"{}", headers={"Content-Type": "application/json"}, method="POST"), timeout=3.0) as response:
            state = json.load(response)
        if state.get("safety_stop"):
            raise ValueError("Controller safety stop is latched; inspect and explicitly acknowledge on site")
        if len(state.get("q", [])) != 6 or len(state.get("force", [])) != 3 or len(state.get("torque", [])) != 3:
            raise ValueError("Server state must contain six joints and three-value force/torque fields (zero placeholders are valid)")
        report["server_reachable"] = True
        report["server_safety"] = state.get("safety_guard", "not_configured")
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
