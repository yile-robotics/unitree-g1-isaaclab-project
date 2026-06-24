#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yile/projects"
UNITREE_RL_LAB="${ROOT}/unitree_rl_lab"
UNITREE_MUJOCO="${ROOT}/unitree_mujoco"

RUN_DIR="${UNITREE_RL_LAB}/logs/rsl_rl/unitree_g1_29dof_agile_reward_velocity_smooth_clearance/2026-06-15_10-38-22_clearance10_actionrate050_actionrr005"
MODEL_ITERATION="${1:-25000}"
CHECKPOINT="${RUN_DIR}/model_${MODEL_ITERATION}.pt"
DEPLOY_SRC="${RUN_DIR}/params/deploy.yaml"

MUJOCO_POLICY_DIR="${UNITREE_RL_LAB}/deploy/robots/g1_29dof/config/policy/velocity/model_${MODEL_ITERATION}_smooth_clearance_mujoco"
POLICY_DST="${MUJOCO_POLICY_DIR}/exported/policy.onnx"
DEPLOY_DST="${MUJOCO_POLICY_DIR}/params/deploy.yaml"

CTRL_DIR="${UNITREE_RL_LAB}/deploy/robots/g1_29dof"
CTRL_CONFIG="${CTRL_DIR}/config/config.yaml"
CTRL_BIN="${CTRL_DIR}/build/g1_ctrl"

SIM_DIR="${UNITREE_MUJOCO}/simulate"
SIM_BIN="${SIM_DIR}/build/unitree_mujoco"
PYTHON="/home/yile/miniconda3/envs/isaacsim/bin/python"

for path in "${CHECKPOINT}" "${DEPLOY_SRC}" "${CTRL_BIN}" "${SIM_BIN}" "${PYTHON}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required file: ${path}"
    exit 1
  fi
done

mkdir -p "${MUJOCO_POLICY_DIR}/exported" "${MUJOCO_POLICY_DIR}/params"

"${PYTHON}" "${UNITREE_RL_LAB}/scripts/export_rsl_rl_actor_onnx.py" \
  "${CHECKPOINT}" \
  "${POLICY_DST}"

cp "${DEPLOY_SRC}" "${DEPLOY_DST}"

# The MuJoCo controller supplies velocity commands from its keyboard observation.
"${PYTHON}" - "${DEPLOY_DST}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
text = text.replace("  velocity_commands:\n", "  keyboard_velocity_commands:\n")
path.write_text(text)
PY

"${PYTHON}" - "${CTRL_CONFIG}" "${MUJOCO_POLICY_DIR}" <<'PY'
from pathlib import Path
import sys

config_path = Path(sys.argv[1])
policy_dir = sys.argv[2]
lines = config_path.read_text().splitlines()
output = []
inside_velocity = False
replaced = False

for line in lines:
    stripped = line.strip()
    indent = len(line) - len(line.lstrip(" "))
    if indent == 2 and stripped == "Velocity:":
        inside_velocity = True
    elif indent == 2 and stripped.endswith(":"):
        inside_velocity = False

    if inside_velocity and stripped.startswith("policy_dir:") and not replaced:
        output.append(f"    policy_dir: {policy_dir}")
        replaced = True
    else:
        output.append(line)

if not replaced:
    raise SystemExit("Could not find FSM.Velocity.policy_dir in config.yaml")

config_path.write_text("\n".join(output) + "\n")
PY

echo "Prepared model_${MODEL_ITERATION} smooth-clearance policy for MuJoCo:"
echo "  policy: ${POLICY_DST}"
echo "  deploy: ${DEPLOY_DST}"
echo
echo "Terminal 1:"
echo "  cd ${SIM_DIR}"
echo "  ${SIM_BIN}"
echo
echo "Terminal 2:"
echo "  cd ${CTRL_DIR}"
echo "  ${CTRL_BIN} --network lo --auto_fixstand_then_velocity"
echo
echo "Commands: w/s=forward, a/d=lateral, q/e=yaw, x=zero command"
