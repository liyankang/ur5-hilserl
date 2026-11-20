# UR5 HIL-SERL

项目通过 `ram_insertion` 控制 UR5，将刚性固定于末端的 U 盘插入母口。Gym 环境通过 Flask 服务向 UR 控制器发送 RTDE 命令。

从仓库根目录启动服务，选择与现场配置匹配的入口，并将 `<UR5_IP>` 替换为控制柜地址。

```bash
# 无夹爪力控
bash serl_robot_ur/robot_servers/launch_ur_server_no_gripper_forcemode.sh <UR5_IP>

# 无夹爪位置伺服
bash serl_robot_ur/robot_servers/launch_ur_server_no_gripper_servo.sh <UR5_IP>
```

可用 `python test_force_mode.py --help` 查看独立力控测试的参数。服务端设置与接口见 [机器人服务说明](serl_robot_ur/README.md)，采集与样本字段见 [数据采集说明](examples/DATA_COLLECTION.md)，双相机标定与测试见 [代码阅读指南](代码阅读指南.md#8-母口换位置时的双相机对准)。

RAM 的 Python 入口会优先使用当前 Python 环境已经安装的 CUDA 编译器路径。双相机流程通过 `RAM_PORT_VISION_CONFIG` 指向现场生成的标定文件，并使用现有 Actor 启动脚本的 CUDA 设置。脚本默认训练模式，测试已有策略时指定评估模式、实际检查点目录和步数。

```bash
RAM_PORT_VISION_CONFIG="$PWD/port_vision.json" \
MODE=eval EVAL_CHECKPOINT_STEP=25000 EVAL_N_TRAJS=1 \
bash examples/experiments/ram_insertion/run_actor.sh \
  --checkpoint_path="$PWD/examples/experiments/ram_insertion/first_run"
```

`25000` 和检查点目录需对应真机上已有的策略。`port_vision.json` 及两张模板图由标定工具生成，所需像素与 TCP 配对数据见代码阅读指南第 8 节。
