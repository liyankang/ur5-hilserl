# UR5 HIL-SERL

`ram_insertion` 是已接入 UR5 的任务。Gym 环境通过 Flask 服务向 UR 控制器发送 RTDE 命令。`usb_pickup_insertion` 是依赖 Franka 的历史任务。

从仓库根目录启动服务，选择与现场配置匹配的入口，并将 `<UR5_IP>` 替换为控制柜地址。

```bash
# 无夹爪力控
bash serl_robot_ur/robot_servers/launch_ur_server_no_gripper_forcemode.sh <UR5_IP>

# 无夹爪位置伺服
bash serl_robot_ur/robot_servers/launch_ur_server_no_gripper_servo.sh <UR5_IP>
```

可用 `python test_force_mode.py --help` 查看独立力控测试的参数。服务端设置与接口见 [机器人服务说明](serl_robot_ur/README.md)，采集与样本字段见 [数据采集说明](examples/DATA_COLLECTION.md)。
