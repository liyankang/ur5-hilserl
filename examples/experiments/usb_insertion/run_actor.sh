#!/usr/bin/env bash
set -euo pipefail

# 固定从仓库根目录读取配置和模型，不依赖某一台电脑的安装路径。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"

MODE="${MODE:-train}"
EVAL_CHECKPOINT_STEP="${EVAL_CHECKPOINT_STEP:-5000}"
EVAL_N_TRAJS="${EVAL_N_TRAJS:-10}"

ARGS=(
    --exp_name=usb_insertion
    --actor
)

case "${MODE}" in
    train)
        ;;
    eval)
        ARGS+=(
            --eval_checkpoint_step="${EVAL_CHECKPOINT_STEP}"
            --eval_n_trajs="${EVAL_N_TRAJS}"
        )
        ;;
    *)
        echo "Unknown MODE='${MODE}'. Use MODE=train or MODE=eval." >&2
        exit 1
        ;;
esac

# 在线交互前核对标定、分类器验收和权重；检查本身不发送运动指令。
"${PYTHON_BIN}" examples/check_usb_training_setup.py --stage training
# 不覆盖 checkpoint_path：默认使用 train_config.json，末尾仍允许显式 CLI 参数。
exec "${PYTHON_BIN}" examples/train_rlpd.py "${ARGS[@]}" "$@"
