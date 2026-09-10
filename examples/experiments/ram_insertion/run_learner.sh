#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="/home/serl/Desktop/ur_hilserl/examples/experiments/ram_insertion/first_run/buffer"
CHECKPOINT_PATH="first_run"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"
export PATH=/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
export LD_LIBRARY_PATH=/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}

mapfile -t DEMO_FILES < <(find "$DEMO_DIR" -maxdepth 1 -type f -name '*.pkl' | sort)
if [[ ${#DEMO_FILES[@]} -eq 0 ]]; then
  echo "No .pkl demo files found in $DEMO_DIR"
  exit 1
fi

demo_args=()
for f in "${DEMO_FILES[@]}"; do
  demo_args+=(--demo_path="$f")
done

python /home/serl/Desktop/ur_hilserl/examples/train_rlpd.py "$@" \
  --exp_name=ram_insertion \
  --checkpoint_path="/home/serl/Desktop/ur_hilserl/examples/experiments/ram_insertion/first_run/checkpoint_25000" \
  "${demo_args[@]}" \
  --learner \
  --debug
