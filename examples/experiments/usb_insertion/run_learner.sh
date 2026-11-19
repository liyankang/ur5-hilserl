#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEMO_DIR="${DEMO_DIR:-${REPO_ROOT}/demo_data}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"

# 与 Actor 使用同一份 JSON/环境变量，防止未验收模型进入在线训练。
"${PYTHON_BIN}" examples/check_usb_training_setup.py --stage training

# 只加载本任务示范；不要把 RAM 或旧任务的七维数据混入 USB 经验池。
shopt -s nullglob
DEMO_FILES=("${DEMO_DIR}"/usb_insertion_*demos*.pkl)
if [[ ${#DEMO_FILES[@]} -eq 0 ]]; then
  echo "No USB demo files matching usb_insertion_*demos*.pkl in ${DEMO_DIR}" >&2
  exit 1
fi

demo_args=()
for f in "${DEMO_FILES[@]}"; do
  demo_args+=(--demo_path="${f}")
done

exec "${PYTHON_BIN}" examples/train_rlpd.py \
  --exp_name=usb_insertion \
  "${demo_args[@]}" \
  --learner \
  --debug "$@"
