#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yile/projects"
UNITREE_RL_LAB="${ROOT}/unitree_rl_lab"
UNITREE_MUJOCO="${ROOT}/unitree_mujoco"

RUN_DIR="${UNITREE_RL_LAB}/logs/rsl_rl/unitree_g1_29dof_velocity/2026-06-07_19-30-37_g1_29dof_yawrew25_finetune3k_seed42"
CHECKPOINT="${RUN_DIR}/model_33000.pt"
DEPLOY_SRC="${RUN_DIR}/params/deploy.yaml"

MUJOCO_POLICY_DIR="${UNITREE_RL_LAB}/deploy/robots/g1_29dof/config/policy/velocity/model_33000_yawrew25_mujoco"
POLICY_DST="${MUJOCO_POLICY_DIR}/exported/policy.onnx"
DEPLOY_DST="${MUJOCO_POLICY_DIR}/params/deploy.yaml"
EVAL_DIR="${RUN_DIR}/mujoco_tracking_eval_model_33000_dense005"
CSV_PATH="${EVAL_DIR}/raw.csv"
CMD_PATH="/tmp/g1_mujoco_tracking_cmd_model_33000.txt"
CASE_DURATION_S="${CASE_DURATION_S:-3.0}"
SETTLE_TIME_S="${SETTLE_TIME_S:-0.5}"
NUM_CASES=60
TOTAL_TRACKING_S="$(python3 -c "print(f'{float(\"${CASE_DURATION_S}\") * ${NUM_CASES}:.1f}')")"

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

mkdir -p "${MUJOCO_POLICY_DIR}/exported" "${MUJOCO_POLICY_DIR}/params" "${EVAL_DIR}"
rm -f "${CMD_PATH}" "${CSV_PATH}"

"${PYTHON}" "${UNITREE_RL_LAB}/scripts/export_rsl_rl_actor_onnx.py" \
  "${CHECKPOINT}" \
  "${POLICY_DST}"

cp "${DEPLOY_SRC}" "${DEPLOY_DST}"

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

echo "Prepared model_33000 automatic MuJoCo tracking evaluation."
echo "The ${NUM_CASES} cases take about ${TOTAL_TRACKING_S} seconds after entering Velocity."
echo
echo "Terminal 1:"
echo "  cd ${SIM_DIR}"
echo "  MUJOCO_TRACKING_EVAL=1 \\"
echo "  MUJOCO_TRACKING_CMD_FILE=${CMD_PATH} \\"
echo "  MUJOCO_TRACKING_CSV=${CSV_PATH} \\"
echo "  ${SIM_BIN}"
echo
echo "Terminal 2:"
echo "  cd ${CTRL_DIR}"
echo "  MUJOCO_TRACKING_EVAL=1 \\"
echo "  MUJOCO_TRACKING_CMD_FILE=${CMD_PATH} \\"
echo "  MUJOCO_TRACKING_CASE_DURATION_S=${CASE_DURATION_S} \\"
echo "  MUJOCO_TRACKING_EXIT_ON_DONE=1 \\"
echo "  ${CTRL_BIN} --network lo --auto_fixstand_then_velocity"
echo
echo "After the final case, generate the summary:"
echo "  cd ${UNITREE_RL_LAB}"
echo "  MPLCONFIGDIR=/tmp/matplotlib-cache ${PYTHON} scripts/plot_mujoco_tracking_summary.py \\"
echo "    --csv ${CSV_PATH} \\"
echo "    --output ${EVAL_DIR} \\"
echo "    --warmup_s ${SETTLE_TIME_S} \\"
echo "    --max_case_time_s ${CASE_DURATION_S}"
