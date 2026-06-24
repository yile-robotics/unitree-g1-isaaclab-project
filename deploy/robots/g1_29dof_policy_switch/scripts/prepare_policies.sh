#!/usr/bin/env bash
set -euo pipefail

# 这个脚本只复制策略到双策略控制器自己的目录。
# 它不会修改原来的 policy、config.yaml 或训练日志。

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOCOMOTION_SOURCE="${1:-/home/yile/projects/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/model_33000_yawrew25_mujoco}"
STAND_SOURCE="${2:-/home/yile/projects/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/model_26999_lower_body_stand}"

prepare_one_policy() {
  local source_dir="$1"
  local destination_dir="$2"

  if [[ ! -f "${source_dir}/exported/policy.onnx" ]]; then
    echo "缺少 ONNX: ${source_dir}/exported/policy.onnx"
    exit 1
  fi
  if [[ ! -f "${source_dir}/params/deploy.yaml" ]]; then
    echo "缺少 deploy.yaml: ${source_dir}/params/deploy.yaml"
    exit 1
  fi

  mkdir -p "${destination_dir}/exported" "${destination_dir}/params"
  cp "${source_dir}/exported/policy.onnx" "${destination_dir}/exported/policy.onnx"
  cp "${source_dir}/params/deploy.yaml" "${destination_dir}/params/deploy.yaml"

  # 双策略控制器使用独立 observation 名称，避免依赖原 g1_ctrl 的键盘实现。
  sed -i \
    -e 's/^  velocity_commands:/  policy_switch_velocity_commands:/' \
    -e 's/^  keyboard_velocity_commands:/  policy_switch_velocity_commands:/' \
    "${destination_dir}/params/deploy.yaml"
}

prepare_one_policy "${LOCOMOTION_SOURCE}" "${PROJECT_DIR}/policies/locomotion"
prepare_one_policy "${STAND_SOURCE}" "${PROJECT_DIR}/policies/stand"

echo "双策略文件已准备完成："
echo "  locomotion: ${PROJECT_DIR}/policies/locomotion"
echo "  stand:      ${PROJECT_DIR}/policies/stand"
