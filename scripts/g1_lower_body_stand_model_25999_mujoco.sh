#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yile/projects"
UNITREE_RL_LAB="${ROOT}/unitree_rl_lab"
UNITREE_MUJOCO="${ROOT}/unitree_mujoco"

RUN_DIR="${UNITREE_RL_LAB}/logs/rsl_rl/unitree_g1_29dof_lower_body_stand/2026-06-10_13-00-41_lower_body_stand_manip_arm70_finetune3k_seed42"
POLICY_SRC="${RUN_DIR}/exported/policy.onnx"
DEPLOY_SRC="${RUN_DIR}/params/deploy.yaml"

MUJOCO_POLICY_DIR="${UNITREE_RL_LAB}/deploy/robots/g1_29dof/config/policy/velocity/model_25999_lower_body_stand"
CTRL_DIR="${UNITREE_RL_LAB}/deploy/robots/g1_29dof"
CTRL_CONFIG="${CTRL_DIR}/config/config.yaml"
CTRL_BIN="${CTRL_DIR}/build/g1_ctrl"

SIM_DIR="${UNITREE_MUJOCO}/simulate"
SIM_BIN="${SIM_DIR}/build/unitree_mujoco"

for file in "${POLICY_SRC}" "${DEPLOY_SRC}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Missing file: ${file}" >&2
    exit 1
  fi
done

for binary in "${CTRL_BIN}" "${SIM_BIN}"; do
  if [[ ! -x "${binary}" ]]; then
    echo "Missing executable: ${binary}" >&2
    exit 1
  fi
done

mkdir -p "${MUJOCO_POLICY_DIR}/exported" "${MUJOCO_POLICY_DIR}/params"
cp "${POLICY_SRC}" "${MUJOCO_POLICY_DIR}/exported/policy.onnx"
cp "${DEPLOY_SRC}" "${MUJOCO_POLICY_DIR}/params/deploy.yaml"

# Keep the stand command exactly zero in MuJoCo.
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
output = []
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
        output.append(f"    policy_dir: {policy_dir}")
        replaced = True
    else:
        output.append(line)

if not replaced:
    raise SystemExit("Could not find FSM.Velocity.policy_dir in config.yaml")

config_path.write_text("\n".join(output) + "\n")
PY

echo "MuJoCo policy prepared:"
echo "  ${MUJOCO_POLICY_DIR}"
echo
echo "Terminal 1:"
echo "  cd ${SIM_DIR}"
echo "  ${SIM_BIN}"
echo
echo "Terminal 2:"
echo "  cd ${CTRL_DIR}"
echo "  ${CTRL_BIN} --network lo --auto_fixstand_then_velocity"
echo
echo "In the MuJoCo window, press 8 to lower the robot and 9 to release the elastic band."
