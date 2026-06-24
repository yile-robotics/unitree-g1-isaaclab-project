#!/usr/bin/env bash
set -euo pipefail

# 独立 goal tracking 测试使用 DDS domain 11。
# 原来的速度/站立测试和 policy-switch 测试不会互相连接。

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MUJOCO_DIR="/home/yile/projects/unitree_mujoco/simulate"
MUJOCO_BIN="${MUJOCO_DIR}/build/unitree_mujoco"
CTRL_BIN="${PROJECT_DIR}/build/g1_goal_tracking_ctrl"
DOMAIN_ID=11
SCENE_FILE="scene_29dof_goal_tracking.xml"
OUTPUT_ROOT="/home/yile/projects/unitree_rl_lab/outputs/goal_tracking"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
OUTPUT_DIR="${OUTPUT_ROOT}/runs/${RUN_ID}"
CAMERA_CSV="${OUTPUT_DIR}/camera_vibration.csv"
CAMERA_PNG="${OUTPUT_DIR}/camera_vibration.png"
TRAJECTORY_CSV="${OUTPUT_DIR}/trajectory.csv"
TRAJECTORY_PNG="${OUTPUT_DIR}/trajectory.png"
CAMERA_SUMMARY="${OUTPUT_DIR}/camera_vibration_summary.txt"
RUN_INFO="${OUTPUT_DIR}/run_info.txt"
PLOT_GOAL="/home/yile/projects/unitree_rl_lab/scripts/goal_tracking/plot_goal_tracking.py"
PLOT_CAMERA="/home/yile/projects/unitree_rl_lab/scripts/goal_tracking/plot_camera_vibration.py"

if [[ ! -x "${MUJOCO_BIN}" ]]; then
  echo "找不到 MuJoCo: ${MUJOCO_BIN}"
  exit 1
fi
if [[ ! -x "${CTRL_BIN}" ]]; then
  echo "找不到 goal tracking 控制器: ${CTRL_BIN}"
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

mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${OUTPUT_DIR}" > "${OUTPUT_ROOT}/latest_run.txt"
{
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "scene=${SCENE_FILE}"
  echo "output_dir=${OUTPUT_DIR}"
} > "${RUN_INFO}"
echo "本次结果目录: ${OUTPUT_DIR}"

echo "启动独立 MuJoCo，DDS domain=${DOMAIN_ID}, scene=${SCENE_FILE}..."
(
  cd "${MUJOCO_DIR}"
  MUJOCO_CAMERA_VIBRATION_SITE="goal_tracking_head_camera_site" \
  MUJOCO_CAMERA_VIBRATION_CSV="${CAMERA_CSV}" \
    "${MUJOCO_BIN}" -i "${DOMAIN_ID}" -n lo -r g1 -s "${SCENE_FILE}"
) &
MUJOCO_PID=$!

sleep 3

echo "启动 goal tracking 控制器..."
echo "先在 MuJoCo 窗口按 8 放下、按 9 取消弹力绳；站稳后在控制器终端按 g 开始 goal follower。"
cd "${PROJECT_DIR}"
set +e
GOAL_TRACKING_LOG_PATH="${TRAJECTORY_CSV}" \
  "${CTRL_BIN}" --network lo --auto_start
CTRL_STATUS=$?
set -e

kill "${MUJOCO_PID}" 2>/dev/null || true
wait "${MUJOCO_PID}" 2>/dev/null || true
MUJOCO_PID=""

echo "生成 goal tracking 和相机振动 PNG..."
python3 "${PLOT_GOAL}" "${TRAJECTORY_CSV}" --out "${TRAJECTORY_PNG}" || true
python3 "${PLOT_CAMERA}" "${CAMERA_CSV}" \
  --trajectory "${TRAJECTORY_CSV}" \
  --out "${CAMERA_PNG}" 2>&1 | tee "${CAMERA_SUMMARY}" || true
echo "finished_at=$(date --iso-8601=seconds)" >> "${RUN_INFO}"
echo "本次完整结果: ${OUTPUT_DIR}"
echo "相机振动图: ${CAMERA_PNG}"

exit "${CTRL_STATUS}"
