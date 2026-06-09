#!/usr/bin/env bash
set -euo pipefail

cd /home/yile/projects/unitree_rl_lab

CHECKPOINT="/home/yile/projects/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-05-31_21-22-36_g1_29dof_stand015_xy15_yaw10_seed42---/model_16200.pt"
OUT_DIR="/home/yile/projects/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-05-31_21-22-36_g1_29dof_stand015_xy15_yaw10_seed42---/tracking_eval/model_16200"

mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"

run_case() {
  local name="$1"
  local vx="$2"
  local vy="$3"
  local wz="$4"

  echo "=== ${name}: vx=${vx}, vy=${vy}, wz=${wz} ==="
  python scripts/rsl_rl/eval_tracking.py \
    --task Unitree-G1-29dof-Velocity \
    --num_envs 16 \
    --steps 1500 \
    --warmup_steps 200 \
    --checkpoint "${CHECKPOINT}" \
    --fixed_cmd "${vx}" "${vy}" "${wz}" \
    --output "${OUT_DIR}/${name}" \
    --headless
}

run_case "stand"       0.0   0.0   0.0
run_case "forward_03"  0.3   0.0   0.0
run_case "forward_05"  0.5   0.0   0.0
run_case "forward_08"  0.8   0.0   0.0
run_case "backward_03" -0.3  0.0   0.0
run_case "left_02"     0.0   0.2   0.0
run_case "right_02"    0.0  -0.2   0.0
run_case "yaw_left_02" 0.0   0.0   0.2
run_case "yaw_right_02" 0.0  0.0  -0.2
run_case "diag_turn"   0.3   0.15  0.1

echo "Saved tracking plots and CSV files to: ${OUT_DIR}"
