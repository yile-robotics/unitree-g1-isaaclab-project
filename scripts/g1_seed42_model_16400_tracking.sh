#!/usr/bin/env bash
set -euo pipefail

cd /home/yile/projects/unitree_rl_lab

MODEL_NAME="model_16400"
RUN_DIR="/home/yile/projects/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-05-31_21-22-36_g1_29dof_stand015_xy15_yaw10_seed42---"
CHECKPOINT="${RUN_DIR}/${MODEL_NAME}.pt"
OUT_DIR="${RUN_DIR}/tracking_eval/${MODEL_NAME}"
RAW_DIR="${OUT_DIR}/raw"

mkdir -p "${RAW_DIR}"
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
    --output "${RAW_DIR}/${name}" \
    --headless
}

# Same cases as model_12000, for direct comparison.
run_case "stand"          0.00   0.00   0.00
run_case "forward_005"    0.05   0.00   0.00
run_case "forward_010"    0.10   0.00   0.00
run_case "forward_020"    0.20   0.00   0.00
run_case "forward_030"    0.30   0.00   0.00
run_case "forward_050"    0.50   0.00   0.00
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

python - "${MODEL_NAME}" "${OUT_DIR}" "${RAW_DIR}" <<'PY'
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
            c for c in cases
            if abs(c[f"cmd_{axis}"]) > 1e-6
            and all(abs(c[f"cmd_{o}"]) < 1e-6 for o in other)
        ],
        key=lambda c: c[f"cmd_{axis}"],
    )

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
fig.suptitle(f"{model_name} velocity tracking: command vs actual", fontsize=15, fontweight="bold")
for ax, axis, title in zip(
    axes,
    ("vx", "vy", "wz"),
    ("forward/backward vx", "lateral vy", "yaw wz"),
):
    data = axis_cases(axis)
    cmd = np.array([c[f"cmd_{axis}"] for c in data])
    actual = np.array([c[f"actual_{axis}"] for c in data])
    ax.plot(cmd, cmd, "--", color="0.45", label="ideal")
    ax.plot(cmd, actual, "o-", linewidth=2, label="actual")
    ax.set_title(title)
    ax.set_xlabel("command")
    ax.set_ylabel("actual")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
fig.savefig(out_dir / "summary_response.png", dpi=180)
plt.close(fig)

ordered = sorted(cases, key=lambda c: c["case"])
x = np.arange(len(ordered))
width = 0.27
fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
ax.bar(x - width, [c["mae_vx"] for c in ordered], width, label="vx MAE")
ax.bar(x, [c["mae_vy"] for c in ordered], width, label="vy MAE")
ax.bar(x + width, [c["mae_wz"] for c in ordered], width, label="wz MAE")
ax.set_title(f"{model_name} tracking error", fontsize=15, fontweight="bold")
ax.set_ylabel("mean absolute error")
ax.set_xticks(x)
ax.set_xticklabels([c["case"].replace("_", "\n") for c in ordered], fontsize=8)
ax.grid(True, axis="y", alpha=0.25)
ax.legend()
fig.savefig(out_dir / "summary_mae.png", dpi=180)
plt.close(fig)

table_rows = []
for c in ordered:
    table_rows.append([
        c["case"],
        f"{c['cmd_vx']:+.2f}, {c['cmd_vy']:+.2f}, {c['cmd_wz']:+.2f}",
        f"{c['actual_vx']:+.2f}, {c['actual_vy']:+.2f}, {c['actual_wz']:+.2f}",
        f"{c['mae_vx']:.3f}, {c['mae_vy']:.3f}, {c['mae_wz']:.3f}",
        f"{c['done']:.4f}",
    ])
fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
ax.axis("off")
table = ax.table(
    cellText=table_rows,
    colLabels=["case", "cmd vx,vy,wz", "actual vx,vy,wz", "MAE vx,vy,wz", "done"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.35)
ax.set_title(f"{model_name} tracking summary", fontsize=15, fontweight="bold")
fig.savefig(out_dir / "summary_table.png", dpi=180)
plt.close(fig)

print(f"Saved summary plots to: {out_dir}")
print("  summary_response.png")
print("  summary_mae.png")
print("  summary_table.png")
PY

echo "Done. Results are in: ${OUT_DIR}"
