#!/usr/bin/env python3
"""Check the UR migration path used by record_demos.

This script focuses on the real path:
record_demos.py -> ram_insertion config -> ur_env -> ur_server.py

It highlights common migration issues from Franka to UR5 without requiring a
real robot. The checks are intentionally lightweight so they can run on a dev
machine.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import werkzeug

from serl_robot_ur.robot_servers.ur_server import create_app, ensure_quat_pose

if not hasattr(werkzeug, "__version__"):
    werkzeug.__version__ = "check"


@dataclass
class Finding:
    level: str
    title: str
    detail: str


class FakeBackend:
    """Minimal backend to exercise the Flask compatibility surface."""

    def __init__(self, robot_ip, reset_joint_target):
        self.robot_ip = robot_ip
        self.reset_joint_target = list(reset_joint_target)
        self.pose = [0.45, -0.05, 0.25, 0.0, 0.0, 0.0, 1.0]
        self.vel = [0.0] * 6
        self.force = [0.0] * 3
        self.torque = [0.0] * 3
        self.q = [0.0] * 6
        self.dq = [0.0] * 6
        self.jacobian = [[0.0] * 6 for _ in range(6)]

    def close(self):
        return None

    def set_target_pose(self, pose, speed=None, accel=None):
        self.pose = ensure_quat_pose(pose).tolist()
        return self.pose

    def move_joints(self, q, speed=None, accel=None):
        self.q = list(q)
        return self.q

    def joint_reset(self, target=None, speed=None, accel=None):
        target = self.reset_joint_target if target is None else list(target)
        self.q = list(target)
        return target

    def get_state(self):
        return {
            "pose": list(self.pose),
            "vel": list(self.vel),
            "force": list(self.force),
            "torque": list(self.torque),
            "q": list(self.q),
            "dq": list(self.dq),
            "jacobian": [list(row) for row in self.jacobian],
            "gripper_pos": [0.0],
        }

    def update_param(self, params):
        return {"applied": dict(params), "ignored": []}

    def set_payload(self, payload):
        return {"applied": True, "payload": dict(payload)}

    def clear_error(self):
        return "Cleared"


def parse_ast(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def iter_string_assignments(tree: ast.AST, name: str) -> Iterable[str]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    continue
                if isinstance(value, str):
                    yield value


def check_record_demos(findings: list[Finding]) -> None:
    path = ROOT / "examples/record_demos.py"
    text = load_text(path)
    if 'classifier=True' in text:
        findings.append(
            Finding(
                "WARN",
                "record_demos 仍强制启用 classifier",
                f"{path}: `get_environment(..., classifier=True)` 仍然存在。没有 classifier checkpoint 时，示教录制可能直接失败或永远等不到成功。",
            )
        )
    if "CONFIG_MAPPING" not in text:
        findings.append(
            Finding(
                "FAIL",
                "record_demos 未通过 CONFIG_MAPPING 建环境",
                f"{path}: 未找到 CONFIG_MAPPING，用实验映射切换任务的逻辑可能被改坏。",
            )
        )


def check_ram_config(findings: list[Finding]) -> None:
    path = ROOT / "examples/experiments/ram_insertion/config.py"
    tree = parse_ast(path)
    text = load_text(path)

    if "from ur_env.envs.ur_env import DefaultEnvConfig" not in text:
        findings.append(
            Finding(
                "FAIL",
                "ram_insertion 没走 ur_env",
                f"{path}: 当前任务配置没有从 `ur_env` 导入 DefaultEnvConfig。",
            )
        )

    if "KeyboardIntervention" not in text:
        findings.append(
            Finding(
                "WARN",
                "示教介入不是键盘路径",
                f"{path}: 未检测到 KeyboardIntervention，当前链路可能仍依赖 SpaceMouse。",
            )
        )

    server_urls = list(iter_string_assignments(tree, "SERVER_URL"))
    if server_urls:
        server_url = server_urls[0]
        if "127.0.0.2" in server_url:
            findings.append(
                Finding(
                    "WARN",
                    "SERVER_URL 仍是 127.0.0.2",
                    f"{path}: 当前配置是 `{server_url}`。如果 ur_server 跑在本机默认地址，常见值会是 `127.0.0.1:5000` 或局域网 IP。",
                )
            )
    else:
        findings.append(
            Finding(
                "FAIL",
                "找不到 SERVER_URL",
                f"{path}: 没解析到 SERVER_URL，环境可能无法连接 robot server。",
            )
        )

    if "obs['state'][0, 6]" in text or 'obs["state"][0, 6]' in text:
        findings.append(
            Finding(
                "FAIL",
                "reward_func 仍含魔法索引",
                f"{path}: 仍然直接写死 `obs['state'][0, 6]`，flatten 布局变化时会静默出错。",
            )
        )

    if "tcp_pose_z_index" not in text:
        findings.append(
            Finding(
                "WARN",
                "reward_func 没有命名索引",
                f"{path}: 没找到 `tcp_pose_z_index`，建议继续保持通过推导索引访问 tcp_pose.z。",
            )
        )


def check_ur_env(findings: list[Finding]) -> None:
    path = ROOT / "ur_env/envs/ur_env.py"
    text = load_text(path)

    required_snippets = [
        "self.q = np.array(ps.get(\"q\", np.zeros((6,))), dtype=np.float32).reshape(-1)",
        "self.dq = np.array(ps.get(\"dq\", np.zeros((6,))), dtype=np.float32).reshape(-1)",
        "return jacobian.reshape(6, jacobian.size // 6)",
    ]
    for snippet in required_snippets:
        if snippet not in text:
            findings.append(
                Finding(
                    "FAIL",
                    "ur_env 状态解析不像 UR5 6 轴版本",
                    f"{path}: 缺少关键解析片段 `{snippet}`。",
                )
            )

    if "requests.post(self.url + \"getstate\")" not in text:
        findings.append(
            Finding(
                "FAIL",
                "ur_env 没有读取 getstate",
                f"{path}: 未检测到 `POST /getstate`，当前数据通路可能已经偏离 UR server 协议。",
            )
        )


def check_ur_server_routes(findings: list[Finding]) -> None:
    def backend_factory(**kwargs):
        return FakeBackend(**kwargs)

    app = create_app(
        robot_ip="127.0.0.1",
        gripper_type="None",
        reset_joint_target=[0.0, -1.57, 1.57, 0.0, 1.57, 0.0],
        backend_factory=backend_factory,
    )
    client = app.test_client()

    pose_response = client.post("/pose", json={"arr": [0.5, -0.1, 0.2, 0.0, 0.0, 0.0, 1.0]})
    state_response = client.post("/getstate")
    reset_response = client.post("/jointreset")
    update_response = client.post("/update_param", json={"translational_stiffness": 2000})
    clear_response = client.post("/clearerr")

    route_responses = {
        "/pose": pose_response.status_code,
        "/getstate": state_response.status_code,
        "/jointreset": reset_response.status_code,
        "/update_param": update_response.status_code,
        "/clearerr": clear_response.status_code,
    }
    bad_routes = [route for route, code in route_responses.items() if code != 200]
    if bad_routes:
        findings.append(
            Finding(
                "FAIL",
                "ur_server 基本兼容路由异常",
                f"这些接口没有返回 200: {', '.join(bad_routes)}。",
            )
        )

    if state_response.status_code == 200:
        state = state_response.get_json()
        expected_lengths = {
            "pose": 7,
            "vel": 6,
            "force": 3,
            "torque": 3,
            "q": 6,
            "dq": 6,
            "gripper_pos": 1,
        }
        for key, expected in expected_lengths.items():
            actual = len(state.get(key, []))
            if actual != expected:
                findings.append(
                    Finding(
                        "FAIL",
                        "ur_server getstate 形状不兼容",
                        f"/getstate 返回 `{key}` 长度为 {actual}，预期 {expected}。",
                    )
                )

        jacobian = state.get("jacobian", [])
        if len(jacobian) != 6 or any(len(row) != 6 for row in jacobian):
            findings.append(
                Finding(
                    "FAIL",
                    "ur_server jacobian 不是 6x6",
                    "当前 fake backend 路径下 /getstate 没返回 6x6 jacobian。",
                )
            )


def check_known_migration_risks(findings: list[Finding]) -> None:
    classifier_dir = ROOT / "classifier_ckpt"
    if not classifier_dir.exists():
        findings.append(
            Finding(
                "WARN",
                "classifier_ckpt 不存在",
                f"{classifier_dir}: record_demos 当前默认依赖 classifier，这会导致无 checkpoint 时无法稳定录 demo。",
            )
        )

    wrench_env = ROOT / "ur_env/envs/ur_wrench_env.py"
    wrench_text = load_text(wrench_env)
    if "shape=(7,))" in wrench_text or "reshape((6, 7))" in wrench_text:
        findings.append(
            Finding(
                "WARN",
                "仓库里仍有 7 轴残留文件",
                f"{wrench_env}: 仍包含 `q/dq shape=(7,)` 或 `jacobian reshape((6, 7))`。它不在当前主链里，但后续误用时会出问题。",
            )
        )


def print_findings(findings: list[Finding]) -> None:
    grouped = {"FAIL": [], "WARN": [], "PASS": []}
    for finding in findings:
        grouped.setdefault(finding.level, []).append(finding)

    if not grouped["FAIL"] and not grouped["WARN"]:
        grouped["PASS"].append(
            Finding(
                "PASS",
                "未发现明显迁移阻塞项",
                "record_demos 到 ur_server 的主通路在当前检查范围内没有发现明显的 Franka 残留阻塞。",
            )
        )

    for level in ("FAIL", "WARN", "PASS"):
        items = grouped[level]
        if not items:
            continue
        print(f"\n[{level}]")
        for item in items:
            print(f"- {item.title}")
            print(f"  {item.detail}")


def main() -> int:
    findings: list[Finding] = []
    check_record_demos(findings)
    check_ram_config(findings)
    check_ur_env(findings)
    check_ur_server_routes(findings)
    check_known_migration_risks(findings)
    print_findings(findings)
    return 1 if any(item.level == "FAIL" for item in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
