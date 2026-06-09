#!/usr/bin/env bash
set -euo pipefail

cd /home/yile/projects/unitree_rl_lab

PYTHON_BIN="${PYTHON:-/home/yile/miniconda3/envs/isaacsim/bin/python}"
CHECKPOINT="/home/yile/projects/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-06-04_12-39-20_g1_29dof_trackstd_lin02_yaw015_yawrew2_seed42/model_25500.pt"

"${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("CUDA GPU is not available. Isaac Sim play needs a CUDA-capable NVIDIA GPU/session.")
PY

"${PYTHON_BIN}" scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT}" \
  --real-time
