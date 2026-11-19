# 固定 U 盘任务：视觉奖励分类器适配

本仓库采用 HIL-SERL 的 Gym 环境和 Flask/RTDE 后端。USB 任务的模型路径、相机名称、成功阈值及复位参数由 `examples/experiments/usb_insertion/train_config.json` 加载；它不是未接入代码的 LeRobot 配置示例。

## 采集可靠的人工标签

从项目根目录、在已经安装项目依赖的 Linux 环境中执行。先完成 TCP、工作空间、复位路径和相机视野的实机标定，再启动机器人服务。采集入口会复位机械臂。

```bash
python examples/record_success_fail.py --exp_name=usb_insertion --successes_needed=100
```

- WASD / JK 控制平移；ZX / VB / NM 控制小角度旋转。没有夹爪维度。
- **H**：明确记录当前未完全插入的负样本，可长按以记录多个负样本。
- **F 或空格**：明确记录当前完全插入的正样本，结束当前回合，保持当前位置后复位；达到目标数量时保持并退出。
- **R**：丢弃当前未标注步并复位。
- **未按标签键的帧不保存**，不能将“没有标成功”自动解释成失败。G 的自动追坐标功能在此入口禁用。
- `--auto` 明确报错：到达目标坐标不是完全插入的证明。保留 `--auto_speed` 参数只是兼容旧命令，人工采集不使用它。

正样本必须由人确认插接到位。负样本应包含对准但未插入、贴近接口、半插入、横向偏差、倾斜和接触外壳等容易混淆的状态。不要把端口侧向受力变大当作成功证据。

文件保存在运行目录的 `classifier_data/`。新样本包含回合 UUID `episode_id`；正负样本属于同一回合时使用相同编号。增加 `--resume` 会原子更新最新快照，避免保存重复的累计副本。采集更多样本时应覆盖不同回合、起点、光照及可见角度。

## 训练和离线验收

```bash
python examples/train_reward_classifier.py --exp_name=usb_insertion --num_epochs=150
python examples/test_classifier.py --exp_name=usb_insertion
```

USB 默认 `classifier_use_proprio=False`，成功检测只读取配置中的相机图像，避免从固定位置或夹爪占位值“猜成功”。旧版含机器人状态的分类器不能直接复用，应重新训练。通用网络 API 保留 `use_proprio=True` 默认值，以兼容其他旧任务；USB 配置显式选择纯视觉模式。

验证集按**完整回合联合划分正负样本**，不能随机拆分相邻图像。旧数据缺少 `episode_id` 时，将同一次 `images_时间戳` 的正负文件合为一个采集分组；不能从仅有一次旧采集中伪造独立验证集。若训练和验证不能各自包含正负样本，脚本会要求采集更多独立回合。

训练结束后，保存目录生成 `validation.json`：

- 记录实际部署阈值下的整体准确率、正类准确率、负类准确率和误报率。
- 整体、正类、负类准确率均须**严格大于 95%**，报告才会设置 `passed=true`；阈值不是训练时的默认 0.5，而是 USB 实际部署阈值。
- 记录任务名、相机键及顺序、`use_proprio`、检查点步数、权重 SHA-256、训练和验证分组。
- 在线训练预检核对最新真实检查点及其指纹；换相机、改阈值、换模型或更改网络输入后，旧报告不能继续作为验收依据。

`test_classifier.py` 默认只评估报告中预留的验证回合。`--all_data` 仅供诊断拟合情况，不能替代独立验证结果，也不会改写验收报告。阈值可通过 `--threshold` 临时诊断，但要改变部署阈值，仍需重新验证并生成对应报告。

准确率门槛是训练启动检查的一部分，不是机械安全保证。样本量、难负样本覆盖和未见场景测试同样影响实际误报率。若需要插入深度辅助证据，应先标定插入轴、到位深度和测量误差，再实现明确的判据；不要用未经验证的阈值放宽来替代完全插入标签。

## 离线程序验证

```bash
python -m unittest tests.test_visual_reward_pipeline -v
```

这些测试验证数据分组、标签请求、指标及报告与权重的一致性，不连接相机、机械臂，不初始化真实 JAX 网络。
#(hilserl) serl@serl-ThinkCentre-M8600t-N000:~/Desktop/ur_hilserl$ python examples/record_success_fail.py --exp_name=usb_insertion --successes_needed=100
2026-11-19 18:58:28.969465: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1795085908.984051    8232 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1795085908.989148    8232 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1795085909.000302    8232 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1795085909.000323    8232 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1795085909.000326    8232 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1795085909.000328    8232 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
Traceback (most recent call last):
  File "/home/serl/Desktop/ur_hilserl/examples/record_success_fail.py", line 167, in <module>
    app.run(main)
  File "/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/absl/app.py", line 367, in run
    _run_main(main, args)
  File "/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/absl/app.py", line 312, in _run_main
    sys.exit(main(argv))
  File "/home/serl/Desktop/ur_hilserl/examples/record_success_fail.py", line 95, in main
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
  File "/home/serl/Desktop/ur_hilserl/examples/experiments/usb_insertion/config.py", line 163, in get_environment
    require_hardware_ready(self.task_config, classifier=classifier)
  File "/home/serl/Desktop/ur_hilserl/examples/experiments/usb_insertion/runtime_config.py", line 149, in require_hardware_ready
    raise ValueError("Real UR5 control is blocked until calibration is complete: " + ", ".join(missing))
ValueError: Real UR5 control is blocked until calibration is complete: tcp_verified, workspace_verified, reset_path_verified, camera_fov_verified, calibration.status=calibrated
