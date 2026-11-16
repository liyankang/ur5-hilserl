# 数据采集约定

- `record_demos.py` 保存成功轨迹；F 标记成功，H 标记失败，R 丢弃当前轨迹并复位。G 保留为环境的移动到目标快捷键，`;` 切换键盘接管。
- `record_success_fail.py` 保存分类器样本；手动控制时用空格或 F 标记当前帧成功，其余帧作为负样本。自动模式按环境在该步返回的成功标记分类，不用后续重新读取的位姿给旧图像打标签。
- `--successes_needed` 是包含续采数据在内的目标总数。`--resume` 按修改时间选择最新文件，并原子更新原文件，避免累计快照被训练程序重复加载。
- `--auto_speed` 是每个策略步的最大平移请求，单位毫米；实际运动受动作空间、旋转耦合、插值和工作空间限制。
- 示范的 `observations` / `next_observations` 分别来自动作前后。人工接管时保存接管动作；成功终止的 mask 为 0，超时的 mask 为 1，但两者的 `dones` 都为 True，防止经验回放跨回合拼图像。
- 分类器文件中的每个条目是独立图像样本，设置 `labels` 和 `dones=True`，与默认的终态样本过滤兼容。分类器训练只读取对应 `exp_name` 的文件。

默认保存目录仍相对于运行脚本时的当前目录：`demo_data/` 和 `classifier_data/`。采集与训练应使用同一个工作目录。

旧自动采集文件可能存在观测错位和错误终止标记，不能仅通过修改代码恢复真实时序，建议重新采集。已有多个累计续采副本的目录，也需先排除重复快照再训练。

离线回归验证：`python -m unittest tests.test_collection -v`。这些测试使用假环境，不连接机械臂，不替代实机验证。

## 启用视觉奖励后的启动顺序

RAM、USB 均已启用在线分类器奖励，阈值为 0.95；采集和假环境不加载分类器。在线环境由分类器判定成功，目标位姿仍用于复位及安全边界。

从项目根目录执行，以 RAM 为例：

1. 启动机器人服务。
2. `python examples/record_success_fail.py --exp_name=ram_insertion --successes_needed=100`，手动确认插接结果并标注图像。自动模式的标签来自坐标阈值，应检查真实插接结果。
3. 退出采集，运行 `python examples/train_reward_classifier.py --exp_name=ram_insertion`。检查验证集上的正负类表现，尤其是接近插入但未完成的负样本。
4. 运行 `record_demos.py` 采集示范，然后启动 Learner，最后启动 Actor。

模型分别保存和加载于 `classifier_ckpt/ram_insertion/`、`classifier_ckpt/usb_insertion/`。USB 命令使用 `--exp_name=usb_insertion`。旧版共享目录下的模型不会自动加载；缺少任务模型时 Actor 会明确报错。

泛化评估应预留未参与训练的位置、光照及采集批次。当前训练脚本按样本随机划分验证集，相邻帧可能分属训练和验证，验证准确率不能代表新场景泛化。分类器当前也读取机器人状态，可能依赖位姿线索；固定位置数据不能证明跨位置能力。
