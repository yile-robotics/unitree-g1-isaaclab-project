#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/yile/projects/unitree_rl_lab"
PYTHON_BIN="${PYTHON:-/home/yile/miniconda3/envs/isaacsim/bin/python}"
CHECKPOINT="${ROOT_DIR}/logs/rsl_rl/unitree_g1_29dof_lower_body_stand/2026-06-10_13-00-41_lower_body_stand_manip_arm70_finetune3k_seed42/model_25999.pt"

cd "${ROOT_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-LowerBody-Stand \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT}" \
  --real-time \
  "$@"
