"""Transition conventions shared by data collectors."""
import copy
import json
import os
import pickle
import tempfile
from collections.abc import Mapping
from pathlib import Path


def save_pickle(path, data):
    """Replace a snapshot atomically so interruption cannot corrupt a resume file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(data, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_transition(obs, action, next_obs, reward, terminated, truncated, info):
    """Timeout ends the episode but preserves value bootstrapping."""
    # 优先记录控制链限幅后实际执行的动作，而不是未经执行的策略建议。
    executed_action = info.get("applied_action", info.get("intervene_action", action))
    intervened = bool(info.get("is_intervention", "intervene_action" in info))
    return copy.deepcopy(dict(
        observations=obs, actions=executed_action,
        next_observations=next_obs, rewards=float(reward),
        masks=float(not terminated), dones=bool(terminated or truncated), infos=info,
        is_intervention=intervened,
    ))


def apply_manual_success(reward, terminated, truncated, info, env=None):
    """人工验收同步更新奖励、结束标记与统计；安全停止始终优先。"""
    if info.get("manual_success", False) and not info.get("safety_stop", False):
        # 人工按键验收也立即停止追逐最后一个压入目标，不能只改数据标签。
        hold = getattr(getattr(env, "unwrapped", None), "hold_position", None)
        if callable(hold):
            hold()
        reward, terminated, truncated = 1.0, True, False
        info["succeed"] = True
    return reward, bool(terminated), bool(truncated), info


def validate_action_space(config, env):
    """在网络初始化前检查动作维数，避免无夹爪任务误建夹爪输出头。"""
    expected = getattr(config, "action_dim", None)
    shape = tuple(env.action_space.shape)
    if expected is not None:
        if expected not in (3, 6) or shape != (expected,):
            raise ValueError(
                f"Fixed-tool action_dim={expected} does not match environment {shape}; "
                "use 3 or 6 Cartesian dimensions without a gripper output."
            )
        if getattr(config, "setup_mode", None) != "single-arm-fixed-gripper":
            raise ValueError("Fixed-tool training requires ordinary single-arm SAC, without gripper critic/penalty.")
    return env.action_space.sample()


def _space_shapes(space):
    """把 Gym 字典空间变成可写入 JSON 的形状说明，不导入硬件或学习框架。"""
    if hasattr(space, "spaces"):
        return {name: _space_shapes(value) for name, value in space.spaces.items()}
    return list(space.shape)


def validate_transition_shapes(transition, observation_space, action_space, source="data"):
    """按当前动作和观测空间验证采集数据的形状。"""
    import numpy as np

    def check(value, space, field):
        if hasattr(space, "spaces"):
            if not isinstance(value, Mapping) or set(value) != set(space.spaces):
                received = sorted(value) if isinstance(value, Mapping) else type(value).__name__
                raise ValueError(f"{source}: {field} keys {received} do not match {sorted(space.spaces)}. Recollect demos for the current configuration.")
            for key, subspace in space.spaces.items():
                check(value[key], subspace, f"{field}.{key}")
            return
        array = np.asarray(value)
        if tuple(array.shape) != tuple(space.shape):
            raise ValueError(
                f"{source}: {field} shape {array.shape} does not match {space.shape}. "
                "Do not trim old gripper/action/state fields: recollect demos with this configuration."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{source}: {field} contains non-finite values.")

    for name in ("observations", "next_observations"):
        check(transition[name], observation_space, name)
    check(transition["actions"], action_space, "actions")


def _save_json(path, payload):
    """原子写入训练结构说明，Actor 和 Learner 可在同一目录并发启动。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ensure_training_schema(config, env, checkpoint_path, exp_name, has_checkpoint=False, algorithm="sac"):
    """固定工具任务恢复模型前，检查网络输入输出与采集结构相容。"""
    if getattr(config, "action_dim", None) is None or checkpoint_path is None:
        return
    expected = {
        "schema_version": 1,
        "algorithm": algorithm,
        "exp_name": exp_name,
        "action_dim": config.action_dim,
        "action_shape": list(env.action_space.shape),
        "observation_shapes": _space_shapes(env.observation_space),
        "proprio_keys": list(config.proprio_keys),
        "setup_mode": config.setup_mode,
    }
    path = Path(checkpoint_path) / "training_schema.json"
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Training schema mismatch in {path}. Old gripper/state/checkpoint dimensions cannot be reused; choose a new checkpoint directory and recollect compatible demos.")
    elif has_checkpoint:
        raise ValueError(f"Existing checkpoint lacks {path.name}; its action/state schema is unverified. Start with a new checkpoint directory instead of loading an old gripper model.")
    else:
        _save_json(path, expected)


def validate_parameter_shapes(expected_params, restored_params, source="checkpoint"):
    """额外核对真实网络参数形状，防止目录说明与实际权重不一致。"""
    def shapes(value, prefix=""):
        if isinstance(value, Mapping):
            result = {}
            for key, child in value.items():
                result.update(shapes(child, f"{prefix}/{key}"))
            return result
        if isinstance(value, (tuple, list)):
            result = {}
            for index, child in enumerate(value):
                result.update(shapes(child, f"{prefix}/{index}"))
            return result
        return {prefix: tuple(value.shape) if hasattr(value, "shape") else type(value).__name__}
    if shapes(expected_params) != shapes(restored_params):
        raise ValueError(f"{source}: network parameter shapes differ from the current action/observation schema. Use a new training directory and a compatible model.")


def label_success(transition):
    transition.update(rewards=1.0, masks=0.0, dones=True)
    transition.setdefault("infos", {})["succeed"] = True


def classifier_sample(transition, success):
    """A selected frame is a standalone labeled example for terminal filtering."""
    sample = copy.deepcopy(transition)
    sample.update(labels=int(success), dones=True, masks=0.0, rewards=float(success))
    sample.setdefault("infos", {})["succeed"] = bool(success)
    return sample
