#!/usr/bin/env python3
"""Plot a compact MuJoCo velocity tracking summary from CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path, warmup_s: float, max_case_time_s: float | None) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if None in row.values():
                continue
            if row["case"] == "none":
                continue
            case_time_s = float(row["case_time_s"])
            if case_time_s < warmup_s:
                continue
            if max_case_time_s is not None and case_time_s > max_case_time_s:
                continue
            parsed: dict[str, float | str] = {"case": row["case"]}
            for key, value in row.items():
                if key != "case":
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def summarize(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    case_order: list[str] = []
    grouped: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        case = str(row["case"])
        if case not in grouped:
            case_order.append(case)
            grouped[case] = []
        grouped[case].append(row)

    summary: list[dict[str, float | str]] = []
    mean_keys = [
        "cmd_vx",
        "cmd_vy",
        "cmd_wz",
        "actual_vx",
        "actual_vy",
        "actual_wz",
        "abs_err_vx",
        "abs_err_vy",
        "abs_err_wz",
    ]
    for case in case_order:
        case_rows = grouped[case]
        item: dict[str, float | str] = {"case": case}
        for key in mean_keys:
            item[key] = float(np.mean([float(row[key]) for row in case_rows]))
        summary.append(item)
    return summary


def plot_table(summary: list[dict[str, float | str]], out_path: Path) -> None:
    table_rows = []
    for row in summary:
        table_rows.append(
            [
                str(row["case"]),
                f"{row['cmd_vx']:+.2f}, {row['cmd_vy']:+.2f}, {row['cmd_wz']:+.2f}",
                f"{row['actual_vx']:+.2f}, {row['actual_vy']:+.2f}, {row['actual_wz']:+.2f}",
                f"{row['abs_err_vx']:.3f}, {row['abs_err_vy']:.3f}, {row['abs_err_wz']:.3f}",
            ]
        )

    fig_h = max(4.0, 0.42 * len(table_rows) + 1.4)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["case", "cmd vx,vy,wz", "actual vx,vy,wz", "abs error vx,vy,wz"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title("MuJoCo velocity tracking summary", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_errors(summary: list[dict[str, float | str]], out_path: Path) -> None:
    labels = [str(row["case"]) for row in summary]
    x = np.arange(len(summary))
    width = 0.27

    fig, ax = plt.subplots(figsize=(12.5, 5.2), constrained_layout=True)
    ax.bar(x - width, [float(row["abs_err_vx"]) for row in summary], width, label="vx error")
    ax.bar(x, [float(row["abs_err_vy"]) for row in summary], width, label="vy error")
    ax.bar(x + width, [float(row["abs_err_wz"]) for row in summary], width, label="wz error")
    ax.set_title("MuJoCo velocity tracking error", fontsize=15, fontweight="bold")
    ax.set_ylabel("mean absolute error")
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", "\n") for label in labels], fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("/tmp/g1_mujoco_tracking.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup_s", type=float, default=2.0)
    parser.add_argument("--max_case_time_s", type=float, default=8.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = summarize(read_rows(args.csv, args.warmup_s, args.max_case_time_s))
    if not summary:
        raise SystemExit(f"No rows found in {args.csv} after warmup_s={args.warmup_s}")

    plot_table(summary, args.output / "mujoco_tracking_summary.png")
    plot_errors(summary, args.output / "mujoco_tracking_error.png")
    print(f"Saved: {args.output / 'mujoco_tracking_summary.png'}")
    print(f"Saved: {args.output / 'mujoco_tracking_error.png'}")


if __name__ == "__main__":
    main()
