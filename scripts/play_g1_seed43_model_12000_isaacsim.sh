#!/usr/bin/env bash
set -euo pipefail

cd /home/yile/projects/unitree_rl_lab

CHECKPOINT="/home/yile/projects/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-05-27_18-37-19_official_retry_seed43/model_12000.pt"

python scripts/rsl_rl/play.py \
  --task Unitree-G1-29dof-Velocity \
  --num_envs 1 \
  --checkpoint "${CHECKPOINT}" \
  --real-time
