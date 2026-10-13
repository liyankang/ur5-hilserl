# SERL Robot Infra for UR5

This package keeps the original SERL runtime split:

- Robot policy / training code talks to a Gym environment.
- The Gym environment talks to a Flask server over HTTP.
- The Flask server talks to the real UR5 controller over RTDE.

The old Franka ROS stack is still present in the repository for reference, but
the active UR5 path no longer depends on ROS.

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
    --controller_mode=forcemode \
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
| `getforce` | Current TCP force |
| `gettorque` | Current TCP torque |
| `getq` | Current joint positions |
| `getdq` | Current joint velocities |
| `getjacobian` | Current Jacobian if available, otherwise zeros |
| `getstate` | Full robot state used by `ur_env` |
| `jointreset` | Move to the configured 6-DOF reset joint target |
| `set_load` | Franka-compatible payload route, translated to UR payload where possible |
| `set_payload` | Alias of `set_load` |
| `update_param` | Compatibility route for compliance parameters; unsupported keys are ignored |
| `force_mode` | Apply a bounded TCP wrench for up to 2 seconds (requires `--controller_mode=forcemode`) |
| `force_mode_stop` | Cancel a direct force command and stop force mode |
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
curl -X POST http://127.0.0.1:5000/force_mode \
  -H 'Content-Type: application/json' \
  -d '{"wrench": [0, 0, 3, 0, 0, 0], "selection_vector": [0, 0, 1, 0, 0, 0], "duration": 0.5}'
curl -X POST http://127.0.0.1:5000/force_mode_stop
```

`wrench` is `[Fx, Fy, Fz, Tx, Ty, Tz]` in N and Nm. The default task frame is
the base frame; `selection_vector` uses one to enable a force-controlled axis.
Limits are capped at 20 N / 5 Nm and 0.1 m/s / 0.5 rad/s, and each command
expires automatically after at most 2 seconds. Test with low forces and a clear
workspace before using a real robot.

## Task Scope

`ur_server.py` now supports two control backends:

- `--controller_mode=forcemode`: impedance-like `forceMode` tracking, closer to the old Franka compliance behavior
- `--controller_mode=servo`: streamed pose servoing with `servoL`

The launch scripts default to `forcemode`.

The active migrated task in this repository is `ram_insertion`.

- `examples/experiments/mappings.py` exposes `ram_insertion`.
- `examples/train_rlpd.py --exp_name=ram_insertion ...` is the intended real-robot path.
- `usb_pickup_insertion` still carries Franka-specific dependencies and is not part of this UR5 migration.
