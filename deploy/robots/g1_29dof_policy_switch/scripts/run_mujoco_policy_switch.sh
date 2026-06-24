#!/usr/bin/env bash
set -euo pipefail

# 独立 MuJoCo 测试使用 DDS domain 10。
# 原来的速度/站立测试继续使用 domain 0，两套测试不会互相连接。

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MUJOCO_DIR="/home/yile/projects/unitree_mujoco/simulate"
MUJOCO_BIN="${MUJOCO_DIR}/build/unitree_mujoco"
CTRL_BIN="${PROJECT_DIR}/build/g1_policy_switch_ctrl"
DOMAIN_ID=10

if [[ ! -x "${MUJOCO_BIN}" ]]; then
  echo "找不到 MuJoCo: ${MUJOCO_BIN}"
  exit 1
fi
if [[ ! -x "${CTRL_BIN}" ]]; then
  echo "找不到双策略控制器: ${CTRL_BIN}"
  echo "请先执行 scripts/build.sh"
  exit 1
fi
if [[ ! -f "${PROJECT_DIR}/policies/locomotion/exported/policy.onnx" ]]; then
  echo "策略尚未准备，请先执行 scripts/prepare_policies.sh"
  exit 1
fi

cleanup() {
  if [[ -n "${MUJOCO_PID:-}" ]]; then
    kill "${MUJOCO_PID}" 2>/dev/null || true
  fi

  # C++ Keyboard 会把当前终端切换为单字符、无回显模式。
  # Ctrl+C 属于强制中断，C++ 对象可能没有机会执行析构函数恢复 termios。
  # 因此脚本退出时始终恢复终端，避免出现提示符回来但无法正常输入的情况。
  stty sane 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "启动独立 MuJoCo，DDS domain=${DOMAIN_ID}..."
(
  cd "${MUJOCO_DIR}"
  "${MUJOCO_BIN}" -i "${DOMAIN_ID}" -n lo -r g1 -s scene_29dof.xml
) &
MUJOCO_PID=$!

sleep 3

echo "启动双策略控制器..."
echo "MuJoCo 窗口中的弹力绳仍使用原按键控制；控制器终端使用 1/2 切换策略。"
cd "${PROJECT_DIR}"
"${CTRL_BIN}" --network lo --auto_start
