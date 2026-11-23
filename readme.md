# UR5 HIL-SERL

项目通过 `ram_insertion` 控制 UR5，将刚性固定于末端的 U 盘插入母口。Gym 环境通过 Flask 服务向 UR 控制器发送 RTDE 命令。

从仓库根目录启动服务，选择与现场配置匹配的入口，并将 `<UR5_IP>` 替换为控制柜地址。

```bash
# 无夹爪力控
bash serl_robot_ur/robot_servers/launch_ur_server_no_gripper_forcemode.sh <UR5_IP>

# 无夹爪位置伺服
bash serl_robot_ur/robot_servers/launch_ur_server_no_gripper_servo.sh <UR5_IP>
```

可用 `python test_force_mode.py --help` 查看独立力控测试的参数。服务端设置与接口见 [机器人服务说明](serl_robot_ur/README.md)。采集、标注和换位微调见 [数据采集说明](examples/DATA_COLLECTION.md)。

RAM 的 Python 入口会优先使用当前 Python 环境已经安装的 CUDA 编译器路径。母口换位先评估已有策略，保持母口与 U 盘各自的插入朝向，并沿用同一套复位流程。在现场终端运行以下命令。用实际检查点目录、步数和工位名替换示例值。

```bash
MODE=eval EVAL_CHECKPOINT_STEP=25000 EVAL_N_TRAJS=1 \
bash examples/experiments/ram_insertion/run_actor.sh \
  --checkpoint_path="$PWD/examples/experiments/ram_insertion/first_run" \
  --eval_station=station_a
```

每次评估期间母口固定。完全插入按 F，未完全插入按 H。程序分别记录实际插入结果和奖励分类器结果。完成当前工位的评估后，按现场流程退出母口并停稳，再换到下一个可达工位。通过 `EVAL_N_TRAJS` 增加重复次数。 [换位评估与示范](代码阅读指南.md#8-母口换位评估)
