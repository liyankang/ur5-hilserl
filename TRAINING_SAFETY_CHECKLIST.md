# 固定 U 盘 USB 插接：训练前检查清单

适用硬件：六关节 UR5，U 盘刚性固定于末端，没有夹爪。本实现使用 RTDE `servoL` 位置伺服，**没有启用力控、阻抗或持续保持力控制**。

## 1. 实机标定与运动范围

- [ ] U 盘、适配器和插座已可靠固定，TCP 偏置与工具负载已按实际装置设置。
- [ ] 已重新标定 JSON 的插入位姿、复位位姿和插入轴；仓库提供的坐标只是旧装置示例。
- [ ] 已手动验证 `workspace_bounds` 的位置边界和参考姿态附近的旋转边界，覆盖完整初始偏差范围。
- [ ] 已验证整条手臂、相机支架、法兰和 U 盘的扫掠空间，不只检查 TCP 点是否在盒内；TCP 包围盒本身不能证明不碰撞。
- [ ] 开始采集/训练前，实际 TCP 已位于标定盒与姿态范围内。程序不会从盒外自动把机械臂拉入盒内。
- [ ] 已验证退出路径：先沿插入轴反方向拔出到安全平面，实际到位后才能横移/转动；退出、复位路径均不碰桌面或适配器底座。
- [ ] 所有随机起点均在腕部/侧视相机有效视野内，图像裁剪后仍能看清插头与母口相对关系。
- [ ] 确认后才把 `calibration.status` 改为 `calibrated`，并将对应 `*_verified` 标记设为 true；它们是现场确认记录，不是程序能替代完成的认证。

默认只随机化起始位置，基坐标系 XYZ 的偏移分别为 ±2.5、±2.5、±2 cm。每次复位使用同一个标称朝向。可从小范围开始逐步扩大，并用训练时未见过的起点验证插入成功率。工作空间必须覆盖完整随机范围。每策略步的平移向量模长上限为 5 mm，旋转角上限为 1°。

## 2. 视觉奖励与模型一致性

- [ ] 正样本确实完全插入；负样本包含“对准但未插入、半插入、贴近接口或顶住外壳”等难例。
- [ ] 训练/验证按完整回合或采集批次隔离，不能把相邻帧拆到两边。
- [ ] `validation.json` 在实际部署阈值下，整体、正类、负类准确率均严格大于 95%，同时检查误报率与难例表现。
- [ ] 验收报告中的相机键、阈值、纯视觉输入、检查点步数与权重 SHA-256 和部署模型一致。
- [ ] 使用新采集的六维动作、18 维机器人状态示范，以及新模型目录；不复用旧七维动作/十九维状态数据或旧模型。
- [ ] 已确认分类器成功后触发 `hold_position()` 和回合终止。此动作停止追踪旧目标，不执行持续力保持。

## 3. 控制与人工接管

- [ ] 策略动作维度、JSON `action_dim`、`network_action_dim` 及环境空间均为 6；无夹爪输出、夹爪惩罚和夹爪状态输入。
- [ ] 实机服务与 JSON 控制周期一致，默认服务 50 Hz、策略 10 Hz；服务会拒绝不匹配的保护配置。
- [ ] 键盘 XYZ/小角度旋转与退出方式已在空旷区域测试；接管数据能以 `is_intervention=true` 写入人类经验池。
- [ ] 首次在线训练有人全程观察，可立即接管或使用实体急停；不要仅依赖进程退出或网络请求停机。
- [ ] 服务端监督已启用：越界或无效反馈锁存停机；普通 `clearerr` 和复位不能自动解除安全锁存。

默认 `wrench_monitor_enabled=false`：不要求力传感器，不读取或控制力，力/力矩观测为明确的零占位。若以后有可信的力估计或外部传感器，先验证量纲、偏置、噪声及方向，再填写阈值并显式开启监测及 `wrench_verified`。此功能只做超限停止，不实现力控，也不能替代机械臂自身保护。

安全停止后先在现场排查、解除接触并检查姿态；随后才可显式请求 `/reset_safety`，JSON 为 `{"acknowledge": true}`。服务端检查通过后仍保持暂停；重新启动采集/Actor，避免旧环境的本地锁存被自动清除。

## 4. 启动前只读检查

在项目根目录执行：

```bash
# 仅检查 JSON 结构和几何范围；默认示例会显示 ready=false，不连接机器人。
python examples/check_usb_training_setup.py

# 标定后检查是否允许采集；未确认标定时退出码为 2。
python examples/check_usb_training_setup.py --stage collection

# 再检查分类器验收报告与真实权重，允许后才进入在线训练。
python examples/check_usb_training_setup.py --stage training
```

可添加 `--probe-server` 只读查询 `/getstate`，不会调用复位或运动接口。指定另一份配置时设置 `UR5_USB_TRAIN_CONFIG`，采集、分类器训练和 Actor/Learner 都读取同一份 JSON。

## 5. 完成标定后的启动顺序

以下命令在已安装项目依赖的 Linux 环境、仓库根目录执行。当前退出路径要求插入轴与某一基坐标轴平行，斜向插入轴会被配置检查拒绝。

1. 在独立终端启动无夹爪位置伺服服务，传入实际 UR5 的 IP：

   ```bash
   bash serl_robot_ur/robot_servers/launch_ur_server_no_gripper_servo.sh <UR5_IP>
   ```

2. 按 `CLASSIFIER_ADAPTATION.md` 采集正负样本、训练分类器并通过验收。
3. 采集六维人工示范：

   ```bash
   python examples/record_demos.py --exp_name=usb_insertion --successes_needed=20
   ```

4. 先启动 Learner，再在另一终端启动 Actor；Actor 会连接机械臂并复位：

   ```bash
   bash examples/experiments/usb_insertion/run_learner.sh
   # 另一终端
   bash examples/experiments/usb_insertion/run_actor.sh
   ```

两个训练脚本启动时均先执行只读训练预检，检查点目录默认读取 JSON。Learner 默认只加载根目录 `demo_data/usb_insertion_*demos*.pkl`；可用 `DEMO_DIR` 指定示范目录。Actor 的评估入口为 `MODE=eval EVAL_CHECKPOINT_STEP=<实际编号> bash examples/experiments/usb_insertion/run_actor.sh`。

训练步数与 30–60 分钟交互时间只能作为预算参考，不保证在某一步开始插入或收敛。以实际成功率、误报率、重试表现和人工接管需求评估；策略明显错误或卡住时及时介入，稳定后逐步减少干预。
