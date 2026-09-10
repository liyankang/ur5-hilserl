#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_strict_conv_algorithm_picker=false"
export PATH=/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
export LD_LIBRARY_PATH=/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/serl/miniforge3/envs/hilserl/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}

cd /home/serl/Desktop/ur_hilserl

python /home/serl/Desktop/ur_hilserl/examples/train_reward_classifier.py "$@" \
  --exp_name=ram_insertion \
  --num_epochs=150 \
  --batch_size=256 \
  --val_ratio=0.2 \
  --eval_period=10
