#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/yile/projects/unitree_rl_lab"
PYTHON_BIN="${PYTHON:-/home/yile/miniconda3/envs/isaacsim/bin/python}"
CHECKPOINT="${ROOT_DIR}/logs/rsl_rl/unitree_g1_29dof_agile_reward_velocity/2026-06-14_11-01-04_agile_reward_velocity/model_24000.pt"

cd "${ROOT_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/rsl_rl/train.py \
  --task Unitree-G1-29dof-Agile-Reward-Velocity-SmoothClearance \
  --resume \
  --checkpoint "${CHECKPOINT}" \
  --reset_optimizer \
  "$@"
