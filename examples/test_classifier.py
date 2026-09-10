"""验证分类器效果"""
import sys
import os
sys.path.insert(0, 'examples')
sys.path.insert(0, '.')

import glob
import pickle as pkl
import numpy as np
import jax
from jax import numpy as jnp
from flax.training import checkpoints

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.networks.reward_classifier import create_classifier
from flax.training import checkpoints as ckpts


def main():
    exp_name = "ram_insertion"
    config = CONFIG_MAPPING[exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)
    
    # 加载分类器
    checkpoint_path = os.path.join(os.getcwd(), "classifier_ckpt/")
    
    # 创建 sample 用于初始化 classifier
    rng = jax.random.PRNGKey(0)
    # 使用 config 中实际的 classifier_keys
    img_keys = config.classifier_keys  # e.g. ["wrist"] or ["wrist_3"]
    sample_obs = {"state": jnp.zeros((1, 19))}
    for k in img_keys:
        sample_obs[k] = jnp.zeros((1, 128, 128, 3), dtype=jnp.uint8)
    
    classifier = create_classifier(rng, sample_obs, config.classifier_keys)
    classifier = ckpts.restore_checkpoint(checkpoint_path, target=classifier)
    print("分类器加载完成!")
    
    # 定义预测函数
    @jax.jit
    def predict(obs):
        logits = classifier.apply_fn({"params": classifier.params}, obs, train=False)
        return jax.nn.sigmoid(logits)
    
    # 加载测试数据
    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))

    # 旧数据相机名 → 新相机名映射
    # wrist_1→global_1 (/dev/video0), wrist_2→wrist (/dev/video2), wrist_3→global_2 (/dev/video4)
    KEY_MAP = {"wrist_1": "global_1", "wrist_2": "wrist", "wrist_3": "global_2"}

    def build_flat_obs(obs):
        flat = {"state": jnp.array(obs["state"])[None, ...]}
        images = obs.get("images", {})
        for k in img_keys:
            if k in images:
                flat[k] = jnp.array(images[k])[None, ...]
            elif k in obs:
                flat[k] = jnp.array(obs[k])[None, ...]
            else:
                # 尝试从旧 key 映射
                for old_k, new_k in KEY_MAP.items():
                    if new_k == k and old_k in images:
                        flat[k] = jnp.array(images[old_k])[None, ...]
                        break
        return flat
    
    # 测试成功样本
    success_correct = 0
    success_total = 0
    for path in success_paths:
        with open(path, "rb") as f:
            data = pkl.load(f)
        for trans in data:
            if "images" not in trans['observations'].keys():
                continue
            # 展平 observation
            obs = trans["observations"]
            flat_obs = build_flat_obs(obs)
            prob = predict(flat_obs)
            pred_label = (prob >= 0.5).item()
            if pred_label == 1:  # 预测为成功
                success_correct += 1
            success_total += 1
    
    # 测试失败样本 (只取一部分，太多会很慢)
    failure_correct = 0
    failure_total = 0
    max_failure = 500  # 只测试 500 个失败样本
    for path in failure_paths:
        with open(path, "rb") as f:
            data = pkl.load(f)
        for trans in data:
            if failure_total >= max_failure:
                break
            if "images" not in trans['observations'].keys():
                continue
            obs = trans["observations"]
            flat_obs = build_flat_obs(obs)
            prob = predict(flat_obs)
            pred_label = (prob >= 0.5).item()
            if pred_label == 0:  # 预测为失败
                failure_correct += 1
            failure_total += 1
        if failure_total >= max_failure:
            break
    
    print("\n" + "="*50)
    print("分类器验证结果:")
    print("="*50)
    print(f"成功样本准确率: {success_correct}/{success_total} = {success_correct/success_total*100:.1f}%")
    print(f"失败样本准确率: {failure_correct}/{failure_total} = {failure_correct/failure_total*100:.1f}%")
    print(f"总体准确率: {(success_correct+failure_correct)}/{success_total+failure_total} = {(success_correct+failure_correct)/(success_total+failure_total)*100:.1f}%")
    print("="*50)


if __name__ == "__main__":
    main()
