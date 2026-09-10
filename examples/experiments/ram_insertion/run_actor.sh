#!/usr/bin/env bash

set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"
export PATH=/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
export LD_LIBRARY_PATH=/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}

MODE="${MODE:-train}"
EVAL_CHECKPOINT_STEP="${EVAL_CHECKPOINT_STEP:-25000}"
EVAL_N_TRAJS="${EVAL_N_TRAJS:-10}"

ARGS=(
    --exp_name=ram_insertion \
    --checkpoint_path=first_run \
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

python /home/serl/Desktop/ur_hilserl/examples/train_rlpd.py "${ARGS[@]}" "$@"
