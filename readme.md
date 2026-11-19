UR5 control uses streamed Cartesian pose servoing. Start the server with
`serl_robot_ur/robot_servers/launch_ur_server_no_gripper_servo.sh`.

固定 U 盘的 USB 插接使用六维末端增量动作，无夹爪控制。实际运行配置位于
`examples/experiments/usb_insertion/train_config.json`，支持用环境变量
`UR5_USB_TRAIN_CONFIG` 指定另一份配置。它是本仓库原生 HIL-SERL/Gym/RTDE
的配置格式，不是直接交给 LeRobot CLI 的 `GymManipulatorConfig`。

- 奖励数据采集、纯视觉训练与验收：[CLASSIFIER_ADAPTATION.md](CLASSIFIER_ADAPTATION.md)
- 实机标定、控制限制与启动门槛：[TRAINING_SAFETY_CHECKLIST.md](TRAINING_SAFETY_CHECKLIST.md)
- 数据字段与采集入口：[examples/DATA_COLLECTION.md](examples/DATA_COLLECTION.md)

默认 JSON 记录当前装置的现场确认结果，插入点 XYZ 沿用已验证的 `ram_insertion`。更换装置后需要重新确认标定。只读检查：

```bash
python examples/check_usb_training_setup.py
```

本仓库用 `RelativeFrame` / `urEnv.step` 完成末端增量到目标位姿的转换，并由 UR
控制器原生处理笛卡尔运动所需的逆运动学；没有额外接入 LeRobot 的关节处理器。
夹爪转换与夹爪惩罚在 USB 任务中都不启用。TCP 包围盒不是整臂碰撞检测，须先
按检查清单验证实际运动范围。默认仅使用位置伺服和位置监督，未实现力控/阻抗。
