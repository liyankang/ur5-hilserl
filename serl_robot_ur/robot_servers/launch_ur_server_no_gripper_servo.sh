#!/usr/bin/env bash

set -euo pipefail

ROBOT_IP="${1:-192.168.25.18}"
FLASK_HOST="${FLASK_HOST:-0.0.0.0}"
FLASK_PORT="${FLASK_PORT:-5000}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_FILE="${LOG_FILE:-}"
LOG_MAX_BYTES="${LOG_MAX_BYTES:-10485760}"
LOG_BACKUP_COUNT="${LOG_BACKUP_COUNT:-5}"

python3 serl_robot_ur/robot_servers/ur_server.py \
  --robot_ip="${ROBOT_IP}" \
  --gripper_type=None \
  --controller_mode=servo \
  --control_hz=50.0 \
  --default_speed=0.05 \
  --default_accel=0.05 \
  --flask_host="${FLASK_HOST}" \
  --flask_port="${FLASK_PORT}" \
  --log_level="${LOG_LEVEL}" \
  --log_file="${LOG_FILE}" \
  --log_max_bytes="${LOG_MAX_BYTES}" \
  --log_backup_count="${LOG_BACKUP_COUNT}" \
  --reset_joint_target -1.3077 -2.4932 -1.6279 -0.6962 1.5537 -2.4239
