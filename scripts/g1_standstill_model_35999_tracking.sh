#!/usr/bin/env bash
set -euo pipefail

cd /home/yile/projects/unitree_rl_lab

MODEL_NAME="model_35999"
RUN_DIR="/home/yile/projects/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-06-09_11-14-36_g1_29dof_standstill_finetune3k_seed42"
CHECKPOINT="${RUN_DIR}/${MODEL_NAME}.pt"
OUT_DIR="${RUN_DIR}/tracking_eval/${MODEL_NAME}"
RAW_DIR="${OUT_DIR}/raw"
PYTHON_BIN="${PYTHON:-/home/yile/miniconda3/envs/isaacsim/bin/python}"

mkdir -p "${RAW_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"

"${PYTHON_BIN}" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("CUDA GPU is not available. Isaac Lab velocity eval needs a CUDA-capable NVIDIA GPU/session.")
PY

run_case() {
  local name="$1"
  local vx="$2"
  local vy="$3"
  local wz="$4"

  echo "=== ${name}: vx=${vx}, vy=${vy}, wz=${wz} ==="
  "${PYTHON_BIN}" scripts/rsl_rl/eval_tracking.py \
    --task Unitree-G1-29dof-Velocity \
    --num_envs 16 \
    --steps 1500 \
    --warmup_steps 200 \
    --checkpoint "${CHECKPOINT}" \
    --fixed_cmd "${vx}" "${vy}" "${wz}" \
    --output "${RAW_DIR}/${name}" \
    --headless
}

# Small, medium, and faster command tracking tests.
run_case "stand"          0.00   0.00   0.00
run_case "forward_005"    0.05   0.00   0.00
run_case "forward_010"    0.10   0.00   0.00
run_case "forward_020"    0.20   0.00   0.00
run_case "forward_030"    0.30   0.00   0.00
run_case "forward_050"    0.50   0.00   0.00
# Out-of-range stress test: the configured training limit is vx=0.6 m/s.
run_case "forward_080"    0.80   0.00   0.00
run_case "backward_010"  -0.10   0.00   0.00
run_case "backward_030"  -0.30   0.00   0.00
run_case "left_010"       0.00   0.10   0.00
run_case "left_020"       0.00   0.20   0.00
run_case "left_030"       0.00   0.30   0.00
run_case "right_010"      0.00  -0.10   0.00
run_case "right_020"      0.00  -0.20   0.00
run_case "right_030"      0.00  -0.30   0.00
run_case "yaw_left_010"   0.00   0.00   0.10
run_case "yaw_left_020"   0.00   0.00   0.20
run_case "yaw_right_010"  0.00   0.00  -0.10
run_case "yaw_right_020"  0.00   0.00  -0.20
run_case "diag_small"     0.10   0.05   0.05
run_case "diag_medium"    0.30   0.15   0.10

"${PYTHON_BIN}" - "${MODEL_NAME}" "${OUT_DIR}" "${RAW_DIR}" <<'PY'
from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

model_name = sys.argv[1]
out_dir = Path(sys.argv[2])
raw_dir = Path(sys.argv[3])
warmup_steps = 200

def read_case(path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if int(row["step"]) >= warmup_steps:
                rows.append({k: float(v) for k, v in row.items()})
    mean = lambda key: float(np.mean([row[key] for row in rows]))
    return {
        "case": path.stem,
        "cmd_vx": mean("cmd_vx"),
        "cmd_vy": mean("cmd_vy"),
        "cmd_wz": mean("cmd_wz"),
        "actual_vx": mean("actual_vx"),
        "actual_vy": mean("actual_vy"),
        "actual_wz": mean("actual_wz"),
        "mae_vx": mean("abs_err_vx"),
        "mae_vy": mean("abs_err_vy"),
        "mae_wz": mean("abs_err_wz"),
        "done": mean("done_fraction"),
    }

cases = [read_case(path) for path in sorted(raw_dir.glob("*.csv"))]

def axis_cases(axis):
    other = {"vx", "vy", "wz"} - {axis}
    return sorted(
        [
            case for case in cases
            if abs(case[f"cmd_{axis}"]) > 1e-6
            and all(abs(case[f"cmd_{other_axis}"]) < 1e-6 for other_axis in other)
        ],
        key=lambda case: case[f"cmd_{axis}"],
    )

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
fig.suptitle(f"{model_name} velocity tracking: command vs actual", fontsize=15, fontweight="bold")
for axis_plot, axis_name, title in zip(
    axes,
    ("vx", "vy", "wz"),
    ("x velocity vx", "y velocity vy", "yaw velocity wz"),
):
    data = axis_cases(axis_name)
    command = np.array([case[f"cmd_{axis_name}"] for case in data])
    actual = np.array([case[f"actual_{axis_name}"] for case in data])
    axis_plot.plot(command, command, "--", color="0.45", label="ideal")
    axis_plot.plot(command, actual, "o-", linewidth=2, label="actual")
    axis_plot.set_title(title)
    axis_plot.set_xlabel("command")
    axis_plot.set_ylabel("actual")
    axis_plot.grid(True, alpha=0.25)
    axis_plot.legend(fontsize=8)
fig.savefig(out_dir / "summary_response.png", dpi=180)
plt.close(fig)

ordered = sorted(cases, key=lambda case: case["case"])
x = np.arange(len(ordered))
width = 0.27
fig, axis_plot = plt.subplots(figsize=(14, 6), constrained_layout=True)
axis_plot.bar(x - width, [case["mae_vx"] for case in ordered], width, label="vx MAE")
axis_plot.bar(x, [case["mae_vy"] for case in ordered], width, label="vy MAE")
axis_plot.bar(x + width, [case["mae_wz"] for case in ordered], width, label="wz MAE")
axis_plot.set_title(f"{model_name} tracking error", fontsize=15, fontweight="bold")
axis_plot.set_ylabel("mean absolute error")
axis_plot.set_xticks(x)
axis_plot.set_xticklabels([case["case"].replace("_", "\n") for case in ordered], fontsize=8)
axis_plot.grid(True, axis="y", alpha=0.25)
axis_plot.legend()
fig.savefig(out_dir / "summary_mae.png", dpi=180)
plt.close(fig)

table_rows = []
for case in ordered:
    table_rows.append([
        case["case"],
        f"{case['cmd_vx']:+.2f}, {case['cmd_vy']:+.2f}, {case['cmd_wz']:+.2f}",
        f"{case['actual_vx']:+.2f}, {case['actual_vy']:+.2f}, {case['actual_wz']:+.2f}",
        f"{case['mae_vx']:.3f}, {case['mae_vy']:.3f}, {case['mae_wz']:.3f}",
        f"{case['done']:.4f}",
    ])
fig, axis_plot = plt.subplots(figsize=(13, 7), constrained_layout=True)
axis_plot.axis("off")
table = axis_plot.table(
    cellText=table_rows,
    colLabels=["case", "cmd vx,vy,wz", "actual vx,vy,wz", "MAE vx,vy,wz", "done"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.25)
axis_plot.set_title(f"{model_name} tracking summary", fontsize=15, fontweight="bold")
fig.savefig(out_dir / "summary_table.png", dpi=180)
plt.close(fig)

print(f"Saved summary plots to: {out_dir}")
print("  summary_response.png")
print("  summary_mae.png")
print("  summary_table.png")
PY

echo "Done. Results are in: ${OUT_DIR}"
