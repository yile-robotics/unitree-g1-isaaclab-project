#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yile/projects"
UNITREE_RL_LAB="${ROOT}/unitree_rl_lab"
UNITREE_MUJOCO="${ROOT}/unitree_mujoco"

RUN_DIR="${UNITREE_RL_LAB}/logs/rsl_rl/unitree_g1_29dof_velocity/2026-06-07_19-30-37_g1_29dof_yawrew25_finetune3k_seed42"
POLICY_SRC="${RUN_DIR}/exported/policy.onnx"
DEPLOY_SRC="${RUN_DIR}/params/deploy.yaml"

MUJOCO_POLICY_DIR="${UNITREE_RL_LAB}/deploy/robots/g1_29dof/config/policy/velocity/model_32500_mujoco"
CTRL_DIR="${UNITREE_RL_LAB}/deploy/robots/g1_29dof"
CTRL_CONFIG="${CTRL_DIR}/config/config.yaml"
CTRL_BIN="${CTRL_DIR}/build/g1_ctrl"

SIM_DIR="${UNITREE_MUJOCO}/simulate"
SIM_BIN="${SIM_DIR}/build/unitree_mujoco"

if [[ ! -f "${POLICY_SRC}" ]]; then
  echo "Missing policy.onnx: ${POLICY_SRC}"
  echo "Run play.py with model_32500.pt first to export the policy."
  exit 1
fi

if [[ ! -f "${DEPLOY_SRC}" ]]; then
  echo "Missing deploy.yaml: ${DEPLOY_SRC}"
  exit 1
fi

if [[ ! -x "${CTRL_BIN}" ]]; then
  echo "Missing controller binary: ${CTRL_BIN}"
  exit 1
fi

if [[ ! -x "${SIM_BIN}" ]]; then
  echo "Missing MuJoCo simulator binary: ${SIM_BIN}"
  exit 1
fi

mkdir -p "${MUJOCO_POLICY_DIR}/exported" "${MUJOCO_POLICY_DIR}/params"
cp "${POLICY_SRC}" "${MUJOCO_POLICY_DIR}/exported/policy.onnx"
cp "${DEPLOY_SRC}" "${MUJOCO_POLICY_DIR}/params/deploy.yaml"

python3 - "${MUJOCO_POLICY_DIR}/params/deploy.yaml" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
text = text.replace("  velocity_commands:\n", "  keyboard_velocity_commands:\n")
path.write_text(text)
PY

python3 - "${CTRL_CONFIG}" "${MUJOCO_POLICY_DIR}" <<'PY'
from pathlib import Path
import sys

config_path = Path(sys.argv[1])
policy_dir = sys.argv[2]
lines = config_path.read_text().splitlines()
out = []
in_velocity = False
replaced = False
for line in lines:
    stripped = line.strip()
    indent = len(line) - len(line.lstrip(" "))
    if indent == 2 and stripped == "Velocity:":
        in_velocity = True
    elif indent == 2 and stripped.endswith(":"):
        in_velocity = False

    if in_velocity and stripped.startswith("policy_dir:") and not replaced:
        out.append(f"    policy_dir: {policy_dir}")
        replaced = True
    else:
        out.append(line)

if not replaced:
    raise SystemExit("Could not find FSM.Velocity.policy_dir in config.yaml")

config_path.write_text("\n".join(out) + "\n")
PY

echo "Prepared MuJoCo policy:"
echo "  ${MUJOCO_POLICY_DIR}"
echo
echo "Terminal 1: start MuJoCo simulator"
echo "  cd ${SIM_DIR}"
echo "  ${SIM_BIN}"
echo
echo "Terminal 2: start G1 controller"
echo "  cd ${CTRL_DIR}"
echo "  ${CTRL_BIN} --network lo --auto_fixstand_then_velocity"
echo
echo "Keyboard commands in controller terminal:"
echo "  w/s: vx +/- 0.05, range [-0.3, 0.6]"
echo "  a/d: vy +/- 0.05, range [-0.5, 0.5]"
echo "  q/e: wz +/- 0.02, range [-0.4, 0.4]"
echo "  x: reset command to 0"
