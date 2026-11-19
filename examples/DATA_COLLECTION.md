# 数据采集约定

- `record_demos.py` 保存成功轨迹；F 标记成功，H 标记失败，R 丢弃当前轨迹并复位。G 保留为环境的移动到目标快捷键，`;` 切换键盘接管。
- `record_success_fail.py` 保存显式人工标签：空格/F 标成功，H 标未完全插入的负样本；未标注帧不保存。成功后保持当前位置并复位。禁止自动按目标坐标生成成功标签，`--auto` 会报错。
- `--successes_needed` 是包含续采数据在内的目标总数。`--resume` 按修改时间选择最新文件，并原子更新原文件，避免累计快照被训练程序重复加载。
- `--auto_speed` 是每个策略步的最大平移请求，单位毫米；实际运动受动作空间、旋转耦合、插值和工作空间限制。
- 示范的 `observations` / `next_observations` 分别来自动作前后。人工接管时保存接管动作；成功终止的 mask 为 0，超时的 mask 为 1，但两者的 `dones` 都为 True，防止经验回放跨回合拼图像。
- 分类器文件中的每个条目是独立图像样本，设置 `labels` 和 `dones=True`，与默认的终态样本过滤兼容。分类器训练只读取对应 `exp_name` 的文件。

默认保存目录仍相对于运行脚本时的当前目录：`demo_data/` 和 `classifier_data/`。采集与训练应使用同一个工作目录。

旧自动采集文件可能存在观测错位和错误终止标记，不能仅通过修改代码恢复真实时序，建议重新采集。已有多个累计续采副本的目录，也需先排除重复快照再训练。

离线回归验证：`python -m unittest tests.test_collection -v`。这些测试使用假环境，不连接机械臂，不替代实机验证。

## 启用视觉奖励后的启动顺序

RAM、USB 均已启用在线分类器奖励，阈值默认 0.95。分类器样本采集和假环境不加载分类器；USB 示范采集与在线训练使用视觉成功判定。USB 参数由 `examples/experiments/usb_insertion/train_config.json` 统一读取。

从项目根目录执行，以 RAM 为例：

1. 启动机器人服务。
2. `python examples/record_success_fail.py --exp_name=ram_insertion --successes_needed=100`，使用 H 与空格/F 明确标注未插入/已插入样本。
3. 退出采集，运行 `python examples/train_reward_classifier.py --exp_name=ram_insertion`。检查验证集上的正负类表现，尤其是接近插入但未完成的负样本。
4. 运行 `record_demos.py` 采集示范，然后启动 Learner，最后启动 Actor。

RAM 模型保存在 `classifier_ckpt/ram_insertion/`；USB 由 JSON 指定，默认 `classifier_ckpt/usb_insertion_visual_v1/`。USB 命令使用 `--exp_name=usb_insertion`，并在真实控制前检查标定与模型报告。旧版模型不能直接用于新的纯视觉分类器或六维/18状态策略。

分类器验证按完整回合/采集批次分组，USB 成功检测只读图像。泛化评估仍需预留未参与训练的位置与光照；起点随机化和母口真实位置变化是不同实验。具体步骤见根目录 `CLASSIFIER_ADAPTATION.md` 与 `TRAINING_SAFETY_CHECKLIST.md`。
