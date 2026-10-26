# SERL Robot Infra for UR5

This package keeps the original SERL runtime split:

- Robot policy / training code talks to a Gym environment.
- The Gym environment talks to a Flask server over HTTP.
- The Flask server talks to the real UR5 controller over RTDE.

The runtime targets the UR5 RTDE interface directly and does not depend on the
old Franka ROS stack.

## Runtime Topology

`SERL policy -> ur_env -> Flask/HTTP -> ur_server.py -> UR5 RTDE`

## Installation

1. Install the Python dependencies for this repository.
2. Install Universal Robots RTDE Python bindings on the robot PC.
3. Ensure the machine running `ur_server.py` can reach the UR5 controller IP.

## Start the UR5 Server

```bash
cd serl_robot_ur/robot_servers
python ur_server.py \
    --robot_ip=<ur5_ip> \
    --controller_mode=servo \
    --flask_host=0.0.0.0 \
    --flask_port=5000 \
    --reset_joint_target <q1> <q2> <q3> <q4> <q5> <q6>
```

There is also a helper script:

```bash
bash launch_ur_server.sh <ur5_ip>
```

## HTTP API

The UR5 server intentionally keeps the Franka-era route names so existing
environments continue to work.

| Request | Description |
| --- | --- |
| `pose` | Update the streamed end-effector target pose (`xyz + quat` or `xyz + rotvec`) |
| `movel_async` | Alias of `pose` |
| `movej` | Move joints immediately |
| `getpos` | Current end-effector pose in `xyz + quat` |
| `getpos_euler` | Current end-effector pose in `xyz + euler xyz` |
| `getvel` | Current TCP velocity |
| `getforce` | Read-only current TCP force measurement |
| `gettorque` | Read-only current TCP torque measurement |
| `getq` | Current joint positions |
| `getdq` | Current joint velocities |
| `getjacobian` | Current Jacobian if available, otherwise zeros |
| `getstate` | Full robot state used by `ur_env` |
| `jointreset` | Move to the configured 6-DOF reset joint target |
| `set_load` | Franka-compatible payload route, translated to UR payload where possible |
| `set_payload` | Alias of `set_load` |
| `clearerr` | Best-effort RTDE stop / script reset |
| `activate_gripper` | No-op compatibility endpoint in the current no-gripper setup |
| `reset_gripper` | No-op compatibility endpoint in the current no-gripper setup |
| `open_gripper` | No-op compatibility endpoint in the current no-gripper setup |
| `close_gripper` | No-op compatibility endpoint in the current no-gripper setup |
| `close_gripper_slow` | No-op compatibility endpoint in the current no-gripper setup |
| `move_gripper` | No-op compatibility endpoint in the current no-gripper setup |

## Manual Smoke Checks

```bash
curl -X POST http://127.0.0.1:5000/getstate
curl -X POST http://127.0.0.1:5000/clearerr
curl -X POST http://127.0.0.1:5000/jointreset
curl -X POST http://127.0.0.1:5000/pose \
  -H 'Content-Type: application/json' \
  -d '{"arr": [0.5, -0.1, 0.2, 0.0, 0.0, 0.0, 1.0]}'
```

## Task Scope

`ur_server.py` uses streamed Cartesian pose servoing with `servoL`. The TCP
force/torque values exposed by `getstate` are read-only observations; they are
not used as motion commands.

The active migrated task in this repository is `ram_insertion`.

- `examples/experiments/mappings.py` exposes `ram_insertion`.
- `examples/train_rlpd.py --exp_name=ram_insertion ...` is the intended real-robot path.


ceshi
