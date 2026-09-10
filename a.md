# UR5 RAM 插入任务 — RLPD 训练手册

---

## 1. 机器人位姿参数

```
当前关节角 q (rad): [-1.2960, -2.7787, -1.6049, -0.2510, 1.6751, 2.6673]
当前 TCP 位姿 (xyz + rotvec): [0.0815, -0.6289, -0.2672, 2.8276, 1.1477, 0.1392]
```

| 参数 | 值 |
|------|-----|
| TARGET_POSE | `[0.0815, -0.6289, -0.2672, 2.8276, 1.1477, 0.1392]` |
| GRASP_POSE | 同 TARGET_POSE |
| RESET_POSE | TARGET_POSE + `[0, 0, 0.1, 0, 0, 0]`（抬高 10cm） |
| REWARD_THRESHOLD | xyz ±5mm, rot ±0.05rad（≈±3°） |
| MAX_EPISODE_LENGTH | 200 步 |

---

## 2. 环境配置

### 摄像头

| 名称 | 设备 | 分辨率 | 裁剪区域 |
|------|------|--------|---------|
| wrist_1 | /dev/video0 | 640×480 | `[:540, 100:500]` |
| wrist_2 | /dev/video2 | 640×480 | `[120:500, 500:800]` |
| wrist_3 | /dev/video4 | 640×480 | `[200:500, 450:700]` |

### 频率链

```
UR Server (50Hz RTDE servo)
    ↑ 插值平滑
ActionInterpolationWrapper (policy 10Hz → control 50Hz)
    ↑ 策略推理
RLPD Actor (每步调用一次策略，~10Hz)
    ↓ 每50步
RLPD Learner (梯度更新，约每5秒一次)
```

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 128 | 每批训练样本数 |
| steps_per_update | 50 | 每50个策略步做一次梯度更新 |
| training_starts | 100 | 前100步只收集数据，不训练 |
| buffer_period | 1000 | replay buffer 最小容量 |
| checkpoint_period | 5000 | 每5000步保存 checkpoint |
| replay_buffer_capacity | 200000 | buffer 最大容量 |
| discount | 0.97 | 折扣因子 |
| encoder_type | resnet-pretrained | 冻结 ResNet-10 预训练权重 |
| image_keys | wrist_1, wrist_2, wrist_3 | 3路图像输入 |
| classifier_keys | wrist_3 | 分类器只用 wrist_3 |
| reward 阈值 | 0.85 | sigmoid(classifier) > 0.85 → reward=1 |

---

## 3. 完整操作流程

### 3.1 启动 UR Server

```bash
# 终端 1：启动机器人服务端
cd /home/serl/Desktop/ur_hilserl
bash serl_robot_ur/robot_servers/launch_ur_server_no_gripper.sh

# 参数说明：
#   --robot_ip 192.168.25.18（默认）
#   --gripper_type None
#   --controller_mode servo
#   --control_hz 50.0
#   --reset_joint_target -1.3077 -2.4932 -1.6279 -0.6962 1.5537 -2.4239
```

### 3.2 采集演示数据

```bash
cd /home/serl/Desktop/ur_hilserl

# 手动示教（SpaceMouse 或键盘介入）
python examples/record_success_fail.py --exp_name ram_insertion --successes_needed 50

# 自动模式（默认速度 0.5）
python examples/record_success_fail.py --exp_name ram_insertion --auto --successes_needed 50

# 调快速度
python examples/record_success_fail.py --exp_name ram_insertion --auto --successes_needed 50 --auto_speed 0.8

# 自动模式仍可手动干预：
#   按 ; 开启键盘介入 → WASD/JK 微调 → 按 f 标记成功
```

数据保存在 `examples/demo_data/` 目录下（.pkl 文件）。

### 3.3 训练奖励分类器

```bash
python examples/train_reward_classifier.py --exp_name ram_insertion
```

- 输入：demo_data 中的成功/失败轨迹
- 输出：`classifier_ckpt/` 目录（checkpoint 文件）
- 只用 `wrist_3` 图像训练
- 训练完成后测试准确率约 98%

### 3.4 启动 RLPD 训练

```bash
# 1. 杀掉旧进程（防止端口占用）
lsof -ti:5588 | xargs kill -9 2>/dev/null

# 2. 启动 Learner（终端 2）
cd /home/serl/Desktop/ur_hilserl/examples/experiments/ram_insertion
bash run_learner.sh
# 等待出现 "sent initial network to actor"

# 3. 启动 Actor（终端 3）
bash run_actor.sh
```

**环境变量**（已在脚本中配置）：
```bash
export PATH=/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
export LD_LIBRARY_PATH=.../nvidia/cuda_cudnn/lib:.../nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
```

---

## 4. 键盘操作

训练过程中 Actor 终端支持以下按键：

| 按键 | 功能 | 说明 |
|------|------|------|
| `;` | 切换手动介入 | 默认关闭，按一次开启 |
| `W/S` | Y 轴前/后 | 手动介入时有效 |
| `A/D` | X 轴左/右 | 手动介入时有效 |
| `J/K` | Z 轴上/下 | 手动介入时有效 |
| `L` | 切换夹爪 | 仅夹爪启用时有效 |
| `g` | 切换 go-to-target | 自动移向目标位置 |
| 按 WASD 或 `;` | 取消 go-to-target | — |
| 到达目标 | 自动停止 | 位置误差 < 5mm，角度误差 < 0.05rad |

**数据记录**：
- 所有步骤 → 存入在线 replay buffer（RL 探索数据）
- 手动介入/go-to-target 步骤 → 额外存入 demo buffer（演示数据）
- Learner 训练时 50/50 采样（一半在线数据，一半演示数据）

---

## 5. 常见问题

### 端口占用 (ZMQError: Address already in use)
```bash
lsof -ti:5588 | xargs kill -9 2>/dev/null
```

### XLA ptxas 版本错误
```bash
export PATH=/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
export LD_LIBRARY_PATH=/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cuda_cudnn/lib:/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
```

### Actor 数据没传到 Learner
- 确认 learner 先启动，显示 "sent initial network to actor"
- 确认 actor 在运行（有 tqdm 进度条）
- 每 10 步会自动 `client.update()` 发送数据

### 训练速度慢
- 减小 `batch_size`（当前 128，可试 64）
- 增大 `steps_per_update`（当前 50，可试 100）
- 减少 `image_keys`（3路 → 1路可提速 2-3x）

---

## 6. 文件结构

```
ur_hilserl/
├── serl_robot_ur/robot_servers/     # UR 机器人服务端
│   └── launch_ur_server_no_gripper.sh  # 启动脚本
├── ur_env/envs/
│   ├── ur_env.py                    # 基础环境
│   └── wrappers.py                  # 键盘介入、动作插值等 wrapper
├── examples/
│   ├── demo_data/                   # 采集的演示数据 (.pkl)
│   ├── train_reward_classifier.py   # 分类器训练
│   ├── train_rlpd.py                # RLPD 训练（learner + actor）
│   └── experiments/ram_insertion/
│       ├── config.py                # 环境 + 训练配置
│       ├── run_learner.sh           # Learner 启动脚本
│       └── run_actor.sh             # Actor 启动脚本
├── serl_launcher/
│   ├── agents/continuous/           # SAC agent
│   ├── data/                        # Replay buffer
│   ├── networks/                    # 分类器、MLP、Actor-Critic
│   └── vision/                      # 数据增强
└── classifier_ckpt/                 # 训练好的分类器 checkpoint
```



export PATH=/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:/usr/local/cuda-11.5/bin:$PATH && export LD_LIBRARY_PATH=/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda-11.5/lib64:$LD_LIBRARY_PATH

encoder_def = EncodingWrapper(
    encoder=encoders,
    use_proprio=False,    # ← 不使用状态数据
    enable_stacking=True,
    image_keys=image_keys,
)

bash serl_robot_ur/robot_servers/launch_ur_server_no_gripper_servo.sh

bash serl_robot_ur/robot_servers/launch_u   r_server_no_gripper_forcemode.sh

bash serl_robot_ur/robot_servers/launch_ur_server_no_gripper_forcemode.sh 192.168.25.18



python record_demos.py --exp_name ram_insertion --successes_needed 40 --auto



MODE=train bash run_actor.sh

MODE=eval EVAL_CHECKPOINT_STEP=25000 EVAL_N_TRAJS=10 bash run_actor.sh