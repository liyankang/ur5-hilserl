#!/usr/bin/env bash

set -euo pipefail

ROBOT_IP="${1:-192.168.0.10}"
FLASK_HOST="${FLASK_HOST:-0.0.0.0}"
FLASK_PORT="${FLASK_PORT:-5000}"

python ur_server.py \
  --robot_ip="${ROBOT_IP}" \
  --gripper_type=None \
  --controller_mode=forcemode \
  --flask_host="${FLASK_HOST}" \
  --flask_port="${FLASK_PORT}" \
  --reset_joint_target 0.0 -1.57 1.57 0.0 1.57 0.0
