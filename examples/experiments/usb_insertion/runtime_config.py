"""加载本仓库的 UR5 JSON 配置；离线校验不依赖 Gym/JAX 或机械臂。"""
import copy
import json
import math
import os
from pathlib import Path

# 所有相对文件路径以仓库根目录为基准，避免不同终端工作目录造成串模型。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("train_config.json")


def vector(value, size, label):
    """拒绝维数错误、NaN、无穷大和布尔值，避免配置静默传播到运动命令。"""
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f"{label} must be finite numbers")
    return value


def resolve_path(value):
    # 用户可给绝对路径；相对路径固定从仓库根目录开始。
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_config_path(path=None):
    # 环境变量让采集、分类器训练和在线训练使用同一份配置。
    return Path(path or os.environ.get("UR5_USB_TRAIN_CONFIG", DEFAULT_CONFIG)).expanduser().resolve()


def load_config(path=None):
    path = resolve_config_path(path)
    with path.open(encoding="utf-8-sig") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config):
    """检查单位、动作形状以及整个随机复位范围，而不是只检查中心点。"""
    if config.get("schema_version") != 1 or config.get("exp_name") != "usb_insertion":
        raise ValueError("Expected schema_version=1 and exp_name=usb_insertion")
    robot, action = config["robot"], config["action"]
    if robot["model"] != "UR5" or robot["joint_dof"] != 6 or robot["gripper_enabled"] is not False:
        raise ValueError("This task requires a six-joint UR5 with no gripper")
    if action["action_dim"] != 6 or config["training"]["network_action_dim"] != 6:
        raise ValueError("Fixed-tool pose correction requires action_dim=network_action_dim=6")
    if action["mode"] != "cartesian_delta" or action["rotation_representation"] != "mrp":
        raise ValueError("Expected Cartesian delta actions with MRP rotation representation")
    if action["gripper_processor_enabled"] or action["gripper_penalty_enabled"]:
        raise ValueError("Gripper processors and gripper penalty must be disabled")
    for name, maximum in (("max_translation_step_m", 0.02), ("max_rotation_step_deg", 5.0)):
        value = action[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(f"action.{name} must be in (0, {maximum}]")
    bounds, reset, task = config["workspace_bounds"], config["reset"], config["task"]
    if bounds["frame"] != "robot_base" or reset["rotation_frame"] != "robot_base":
        raise ValueError("Workspace/reset axes must be expressed in robot_base")
    low = vector(bounds["position_low_m"], 3, "workspace low")
    high = vector(bounds["position_high_m"], 3, "workspace high")
    ref = vector(bounds["reference_rotvec_rad"], 3, "orientation reference")
    angle_limits = vector(bounds["relative_rotation_limit_deg"], 3, "rotation limits")
    nominal = vector(reset["nominal_tcp_pose_m_rotvec"], 6, "reset pose")
    inserted = vector(task["nominal_inserted_tcp_pose_m_rotvec"], 6, "inserted pose")
    jitter = vector(reset["translation_range_m"], 3, "reset translation range")
    angle_jitter = vector(reset["rotation_range_deg"], 3, "reset rotation range")
    axis = vector(task["insertion_axis_base"], 3, "insertion axis")
    if not math.isclose(sum(x*x for x in axis), 1.0, abs_tol=1e-6):
        raise ValueError("insertion_axis_base must be a unit vector")
    if any(lo >= hi for lo, hi in zip(low, high)):
        raise ValueError("workspace low must be smaller than high")
    if any(j < 0 or j > 0.05 for j in jitter) or any(j < 0 or j > 5 for j in angle_jitter):
        raise ValueError("Reset ranges must be non-negative and at most 5 cm / 5 degrees")
    if any(limit <= 0 or j > limit for j, limit in zip(angle_jitter, angle_limits)):
        raise ValueError("Reset orientation range exceeds workspace orientation limits")
    # 用同一个标定基准避免把绝对旋转向量分量误当作可直接相加的角度。
    if any(not math.isclose(x, y, abs_tol=1e-8) for x, y in zip(ref, nominal[3:])):
        raise ValueError("Nominal reset orientation must match the calibrated reference")
    if any(not math.isclose(x, y, abs_tol=1e-8) for x, y in zip(ref, inserted[3:])):
        raise ValueError("Nominal inserted orientation must match the calibrated reference")
    for i in range(3):
        if nominal[i] - jitter[i] < low[i] or nominal[i] + jitter[i] > high[i]:
            raise ValueError("The complete reset translation range must fit workspace_bounds")
        if not low[i] <= inserted[i] <= high[i]:
            raise ValueError("Nominal insertion pose lies outside workspace_bounds")
    # 退出动作到达安全平面后才横移/转动；平面与插入轴垂直。
    clearance = reset["clearance_along_axis_m"]
    if not isinstance(clearance, (int, float)) or not math.isfinite(clearance) or clearance <= 0:
        raise ValueError("reset clearance must be finite and positive")
    minimum_clearance = sum((nominal[i]-inserted[i]) * (-axis[i]) for i in range(3)) - sum(abs(axis[i])*jitter[i] for i in range(3))
    if minimum_clearance < clearance:
        raise ValueError("Random reset can enter the insertion/contact clearance region")
    # 此实现的单轴退出安全平面需与标定的基坐标轴平行，否则边界需更完整几何检查。
    if sum(abs(x) > 1e-8 for x in axis) != 1:
        raise ValueError("Current reset implementation requires an axis-aligned insertion axis")
    if reset["terminate_on_success"] is not True or not isinstance(reset["randomize"], bool):
        raise ValueError("Reset requires boolean randomize and terminate_on_success=true")
    if not isinstance(reset["max_episode_steps"], int) or reset["max_episode_steps"] <= 0:
        raise ValueError("max_episode_steps must be a positive integer")
    if not isinstance(reset["motion_duration_s"], (int, float)) or reset["motion_duration_s"] <= 0:
        raise ValueError("reset motion duration must be positive")
    obs, reward, training = config["observation"], config["reward_classifier"], config["training"]
    if "gripper_pose" in obs["proprio_keys"]:
        raise ValueError("Fixed-tool observations must omit gripper_pose")
    if obs["proprio_keys"] != ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque"]:
        raise ValueError("Expected the four supported proprioceptive fields (18 values)")
    if not reward["enabled"] or reward["use_proprio"] is not False or reward["terminate_on_success"] is not True:
        raise ValueError("USB requires image-only classifier rewards and termination on success")
    if not 0 < reward["success_threshold"] < 1 or not 0.95 <= reward["minimum_validation_accuracy"] < 1:
        raise ValueError("Invalid classifier thresholds")
    if len(set(obs["image_keys"])) != len(obs["image_keys"]) or not set(reward["image_keys"]) <= set(obs["image_keys"]):
        raise ValueError("Classifier camera keys must be a subset of distinct policy camera keys")
    if not reward["image_keys"]:
        raise ValueError("At least one classifier camera is required")
    for key in obs["image_keys"]:
        camera = obs["cameras"][key]
        width, height = vector(camera["dim"], 2, f"{key}.dim")
        y1, y2, x1, x2 = vector(camera["crop_yxyx"], 4, f"{key}.crop")
        if not 0 <= y1 < y2 <= height or not 0 <= x1 < x2 <= width:
            raise ValueError(f"Invalid image crop for {key}")
    policy_hz, control_hz = training["policy_hz"], training["control_hz"]
    if not 0 < policy_hz <= control_hz <= 125 or not math.isclose(control_hz/policy_hz, round(control_hz/policy_hz)):
        raise ValueError("Control rate must be an integer multiple of policy rate, at most 125 Hz")
    if training["batch_size"] <= 0 or training["batch_size"] % 2 or training["demo_sampling_ratio"] != 0.5:
        raise ValueError("RLPD requires an even batch_size and a 50/50 demo sampling ratio")
    safety = config["controller_safety"]
    if safety["enabled"] is not True or safety["latch_on_violation"] is not True:
        raise ValueError("Controller-side safety limits and latching must remain enabled")
    if not isinstance(safety.get("wrench_monitor_enabled"), bool):
        raise ValueError("wrench_monitor_enabled must be explicitly true or false")
    vector(safety["wrench_bias"], 6, "wrench bias")
    for field in ("force_norm_limit_n", "torque_norm_limit_nm"):
        value = safety[field]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{field} must be null (uncalibrated) or a finite positive value")
    return config


def require_hardware_ready(config, *, classifier=False):
    """真实控制前拒绝未标定配置；分类器采集本身不要求先有分类器。"""
    # 无力控/无可信力反馈的 UR5 默认只使用位置监督，不强行要求力传感器。
    checks = ["tcp_verified", "workspace_verified", "reset_path_verified", "camera_fov_verified"]
    if config["controller_safety"]["wrench_monitor_enabled"]:
        checks.append("wrench_verified")
    missing = [name for name in checks if config["calibration"].get(name) is not True]
    if config["calibration"].get("status") != "calibrated":
        missing.append("calibration.status=calibrated")
    if config["controller_safety"]["wrench_monitor_enabled"]:
        missing += [name for name in ("force_norm_limit_n", "torque_norm_limit_nm") if config["controller_safety"].get(name) is None]
    if missing:
        raise ValueError("Real UR5 control is blocked until calibration is complete: " + ", ".join(missing))
    if classifier:
        from examples.classifier_data import validate_classifier_artifact
        reward = config["reward_classifier"]
        validate_classifier_artifact(str(resolve_path(reward["checkpoint_path"])), "usb_insertion", reward["image_keys"], False, reward["success_threshold"], reward["minimum_validation_accuracy"])


def controller_safety_payload(config):
    """把同一份边界和经过标定的力阈值发送给服务端独立保护层。"""
    payload = copy.deepcopy(config["controller_safety"])
    payload.update(workspace_low=config["workspace_bounds"]["position_low_m"], workspace_high=config["workspace_bounds"]["position_high_m"], rotation_reference_rotvec=config["workspace_bounds"]["reference_rotvec_rad"], rotation_limits_rad=[math.radians(x) for x in config["workspace_bounds"]["relative_rotation_limit_deg"]], control_hz=config["training"]["control_hz"])
    return payload
