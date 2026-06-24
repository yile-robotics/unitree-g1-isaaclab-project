#!/usr/bin/env python3
"""Plot a compact MuJoCo velocity tracking summary from CSV."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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
            if row["case"] in {"none", "done"}:
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
            for axis in ("vx", "vy", "wz"):
                err_key = f"err_{axis}"
                if err_key not in parsed:
                    parsed[err_key] = float(parsed[f"actual_{axis}"]) - float(parsed[f"cmd_{axis}"])
                abs_err_key = f"abs_err_{axis}"
                if abs_err_key not in parsed:
                    parsed[abs_err_key] = abs(float(parsed[err_key]))
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
        "err_vx",
        "err_vy",
        "err_wz",
        "abs_err_vx",
        "abs_err_vy",
        "abs_err_wz",
    ]
    for case in case_order:
        case_rows = grouped[case]
        item: dict[str, float | str] = {"case": case}
        for key in mean_keys:
            item[key] = float(np.mean([float(row[key]) for row in case_rows]))
        for key in ["abs_err_vx", "abs_err_vy", "abs_err_wz"]:
            item[f"max_{key}"] = float(np.max([float(row[key]) for row in case_rows]))
        summary.append(item)
    return summary


def summarize_global(rows: list[dict[str, float | str]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for axis in ("vx", "vy", "wz"):
        err = np.asarray([float(row[f"err_{axis}"]) for row in rows], dtype=np.float64)
        abs_err = np.abs(err)
        metrics[f"{axis}_mean_error"] = float(abs_err.mean())
        metrics[f"{axis}_signed_mean_error"] = float(err.mean())
        metrics[f"{axis}_max_error"] = float(abs_err.max())
    return metrics


def summarize_group(rows: list[dict[str, float | str]]) -> dict[str, float]:
    return summarize_global(rows)


def plot_table(summary: list[dict[str, float | str]], out_path: Path) -> None:
    table_rows = []
    for row in summary:
        table_rows.append(
            [
                str(row["case"]),
                f"{row['cmd_vx']:+.2f}, {row['cmd_vy']:+.2f}, {row['cmd_wz']:+.2f}",
                f"{row['actual_vx']:+.2f}, {row['actual_vy']:+.2f}, {row['actual_wz']:+.2f}",
                f"{row['abs_err_vx']:.3f}, {row['abs_err_vy']:.3f}, {row['abs_err_wz']:.3f}",
                f"{row['err_vx']:+.3f}, {row['err_vy']:+.3f}, {row['err_wz']:+.3f}",
            ]
        )

    fig_h = max(4.0, 0.42 * len(table_rows) + 1.4)
    fig, ax = plt.subplots(figsize=(11.5, fig_h), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["case", "cmd vx,vy,wz", "actual vx,vy,wz", "MAE vx,vy,wz", "signed mean vx,vy,wz"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title("MuJoCo velocity tracking summary", fontsize=15, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_cards(summary: list[dict[str, float | str]], out_path: Path) -> None:
    metrics = [
        ("LIN VEL X ERROR", "abs_err_vx", "err_vx", "max_abs_err_vx"),
        ("LIN VEL Y ERROR", "abs_err_vy", "err_vy", "max_abs_err_vy"),
        ("ANG VEL Z ERROR", "abs_err_wz", "err_wz", "max_abs_err_wz"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 2.2), constrained_layout=True)
    fig.patch.set_facecolor("#f5f6f8")
    for ax, (title, mae_key, signed_key, max_key) in zip(axes, metrics):
        mae = float(np.mean([float(row[mae_key]) for row in summary]))
        signed = float(np.mean([float(row[signed_key]) for row in summary]))
        max_err = float(np.max([float(row[max_key]) for row in summary]))
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color("#5b7cfa")
        ax.spines["left"].set_linewidth(3.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.05, 0.76, title, transform=ax.transAxes, fontsize=9, color="#6c6f77", fontweight="bold")
        ax.text(0.05, 0.46, f"{mae:.4f}", transform=ax.transAxes, fontsize=18, color="#2d2f35", fontweight="bold")
        ax.text(0.05, 0.23, f"SIGNED MEAN: {signed:+.4f}", transform=ax.transAxes, fontsize=9, color="#6c6f77")
        ax.text(0.05, 0.07, f"MAX: {max_err:.4f}", transform=ax.transAxes, fontsize=9, color="#6c6f77")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_cards_from_metrics(metrics: dict[str, float], out_path: Path, title: str) -> None:
    card_specs = [
        ("LIN VEL X ERROR", "vx"),
        ("LIN VEL Y ERROR", "vy"),
        ("ANG VEL Z ERROR", "wz"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 2.4), constrained_layout=True)
    fig.patch.set_facecolor("#f5f6f8")
    for ax, (label, axis) in zip(axes, card_specs):
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color("#5b7cfa")
        ax.spines["left"].set_linewidth(3.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.05, 0.76, label, transform=ax.transAxes, fontsize=9, color="#6c6f77", fontweight="bold")
        ax.text(
            0.05,
            0.46,
            f"{metrics[f'{axis}_mean_error']:.4f}",
            transform=ax.transAxes,
            fontsize=18,
            color="#2d2f35",
            fontweight="bold",
        )
        ax.text(
            0.05,
            0.23,
            f"SIGNED MEAN: {metrics[f'{axis}_signed_mean_error']:+.4f}",
            transform=ax.transAxes,
            fontsize=9,
            color="#6c6f77",
        )
        ax.text(
            0.05,
            0.07,
            f"MAX: {metrics[f'{axis}_max_error']:.4f}",
            transform=ax.transAxes,
            fontsize=9,
            color="#6c6f77",
        )
    fig.suptitle(title, fontsize=13, fontweight="bold")
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


def plot_tracking_performance(rows: list[dict[str, float | str]], out_path: Path) -> None:
    t = np.asarray([float(row["time_s"]) for row in rows])
    t = t - t[0]
    series = [
        ("Linear Velocity X", "m/s", "actual_vx", "cmd_vx"),
        ("Linear Velocity Y", "m/s", "actual_vy", "cmd_vy"),
        ("Angular Velocity Z", "rad/s", "actual_wz", "cmd_wz"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14.0, 8.0), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax, (title, ylabel, actual_key, cmd_key) in zip(axes, series):
        actual = np.asarray([float(row[actual_key]) for row in rows])
        cmd = np.asarray([float(row[cmd_key]) for row in rows])
        finite_values = np.concatenate([actual[np.isfinite(actual)], cmd[np.isfinite(cmd)]])
        if finite_values.size:
            limit = float(np.percentile(np.abs(finite_values), 99.0))
            limit = max(limit * 1.15, float(np.max(np.abs(cmd))) * 1.4, 0.25)
            ax.set_ylim(-limit, limit)
        ax.set_facecolor("#eaf0f7")
        ax.plot(t, actual, color="blue", linewidth=1.2, label="Actual")
        ax.step(t, cmd, where="post", color="red", linestyle=(0, (5, 4)), linewidth=1.5, label="Commanded")
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="white", linewidth=0.8)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.09, 1.0), frameon=False)
    fig.suptitle("Tracking Performance", fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_axis_tracking(rows: list[dict[str, float | str]], axis: str, out_path: Path) -> None:
    labels = {
        "vx": ("Linear Velocity X", "m/s"),
        "vy": ("Linear Velocity Y", "m/s"),
        "wz": ("Angular Velocity Z", "rad/s"),
    }
    title, unit = labels[axis]
    t = np.asarray([float(row["time_s"]) for row in rows])
    t = t - t[0]
    actual = np.asarray([float(row[f"actual_{axis}"]) for row in rows])
    cmd = np.asarray([float(row[f"cmd_{axis}"]) for row in rows])
    err = actual - cmd

    fig, axes = plt.subplots(2, 1, figsize=(14.0, 6.2), sharex=True, constrained_layout=True)
    axes[0].set_facecolor("#eaf0f7")
    axes[0].plot(t, actual, color="blue", linewidth=1.2, label="Actual")
    axes[0].step(t, cmd, where="post", color="red", linestyle=(0, (5, 4)), linewidth=1.5, label="Commanded")
    axes[0].set_title(title, fontsize=13)
    axes[0].set_ylabel(unit)
    axes[0].grid(True, color="white", linewidth=0.8)
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].set_facecolor("#f3f5f8")
    axes[1].plot(t, err, color="#2f5f98", linewidth=1.1, label="Actual - Commanded")
    axes[1].axhline(0.0, color="#444", linewidth=0.8, alpha=0.6)
    axes[1].set_ylabel(f"error [{unit}]")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, color="white", linewidth=0.8)
    axes[1].legend(loc="upper right", frameon=False)
    fig.suptitle(f"{title} Tracking", fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def classify_case(row: dict[str, float | str]) -> str:
    eps = 1.0e-6
    vx = abs(float(row["cmd_vx"])) > eps
    vy = abs(float(row["cmd_vy"])) > eps
    wz = abs(float(row["cmd_wz"])) > eps
    active_count = int(vx) + int(vy) + int(wz)
    if active_count == 0:
        return "stand"
    if active_count > 1:
        return "combo"
    if vx:
        return "vx"
    if vy:
        return "vy"
    return "yawz"


def plot_error_group(rows: list[dict[str, float | str]], group: str, title: str, out_path: Path) -> None:
    group_rows = [row for row in rows if classify_case(row) == group]
    if not group_rows:
        return

    t = np.asarray([float(row["time_s"]) for row in group_rows])
    t = t - t[0]
    series = [
        ("X Velocity Error", "m/s", "err_vx"),
        ("Y Velocity Error", "m/s", "err_vy"),
        ("Yaw Rate Error", "rad/s", "err_wz"),
    ]
    case_changes: list[tuple[float, str]] = []
    last_case: str | None = None
    for row, time_s in zip(group_rows, t):
        case = str(row["case"])
        if case != last_case:
            case_changes.append((float(time_s), case))
            last_case = case

    fig, axes = plt.subplots(3, 1, figsize=(14.0, 8.0), sharex=True, constrained_layout=True)
    for ax, (axis_title, ylabel, key) in zip(axes, series):
        err = np.asarray([float(row[key]) for row in group_rows])
        ax.set_facecolor("#f3f5f8")
        ax.plot(t, err, color="#1f5fbf", linewidth=1.15)
        ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.65)
        for change_t, _case in case_changes:
            ax.axvline(change_t, color="#8a8f99", linewidth=0.7, alpha=0.45)
        finite = err[np.isfinite(err)]
        if finite.size:
            limit = max(float(np.percentile(np.abs(finite), 99.0)) * 1.2, 0.05)
            ax.set_ylim(-limit, limit)
        ax.set_title(axis_title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="white", linewidth=0.8)
    axes[-1].set_xlabel("group time (s)")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_grouped_error_reports(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    groups = [
        ("vx", "VX Commands: X/Y/Yaw Errors", "errors_vx_commands.png"),
        ("vy", "VY Commands: X/Y/Yaw Errors", "errors_vy_commands.png"),
        ("yawz", "Yaw Commands: X/Y/Yaw Errors", "errors_yaw_commands.png"),
        ("combo", "Combined Commands: X/Y/Yaw Errors", "errors_combo_commands.png"),
    ]
    for group, title, filename in groups:
        plot_error_group(rows, group, title, output_dir / filename)


def plot_tracking_group(rows: list[dict[str, float | str]], group: str, title: str, out_path: Path) -> None:
    group_rows = [row for row in rows if classify_case(row) == group]
    if not group_rows:
        return

    t = np.asarray([float(row["time_s"]) for row in group_rows])
    t = t - t[0]
    series = [
        ("Linear Velocity X", "m/s", "actual_vx", "cmd_vx"),
        ("Linear Velocity Y", "m/s", "actual_vy", "cmd_vy"),
        ("Angular Velocity Z", "rad/s", "actual_wz", "cmd_wz"),
    ]
    case_changes: list[tuple[float, str]] = []
    last_case: str | None = None
    for row, time_s in zip(group_rows, t):
        case = str(row["case"])
        if case != last_case:
            case_changes.append((float(time_s), case))
            last_case = case

    fig, axes = plt.subplots(3, 1, figsize=(14.0, 8.0), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    for ax, (axis_title, ylabel, actual_key, cmd_key) in zip(axes, series):
        actual = np.asarray([float(row[actual_key]) for row in group_rows])
        cmd = np.asarray([float(row[cmd_key]) for row in group_rows])
        finite_values = np.concatenate([actual[np.isfinite(actual)], cmd[np.isfinite(cmd)]])
        if finite_values.size:
            limit = float(np.percentile(np.abs(finite_values), 99.0))
            limit = max(limit * 1.15, float(np.max(np.abs(cmd))) * 1.4, 0.25)
            ax.set_ylim(-limit, limit)
        ax.set_facecolor("#eaf0f7")
        ax.plot(t, actual, color="blue", linewidth=1.2, label="Actual")
        ax.step(t, cmd, where="post", color="red", linestyle=(0, (5, 4)), linewidth=1.5, label="Commanded")
        for change_t, _case in case_changes:
            ax.axvline(change_t, color="#8a8f99", linewidth=0.7, alpha=0.45)
        ax.set_title(axis_title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="white", linewidth=0.8)
    axes[-1].set_xlabel("group time (s)")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.09, 1.0), frameon=False)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_grouped_tracking_reports(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    groups = [
        ("vx", "VX Commands Tracking Performance", "tracking_vx_commands.png"),
        ("vy", "VY Commands Tracking Performance", "tracking_vy_commands.png"),
        ("yawz", "Yaw Commands Tracking Performance", "tracking_yaw_commands.png"),
        ("combo", "Combined Commands Tracking Performance", "tracking_combo_commands.png"),
    ]
    for group, title, filename in groups:
        plot_tracking_group(rows, group, title, output_dir / filename)


def write_group_metrics(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    groups = [
        ("vx", "VX Commands", "cards_vx_commands.png"),
        ("vy", "VY Commands", "cards_vy_commands.png"),
        ("yawz", "Yaw Commands", "cards_yaw_commands.png"),
        ("combo", "Combined Commands", "cards_combo_commands.png"),
    ]
    grouped_metrics: dict[str, dict[str, float]] = {}
    csv_rows: list[dict[str, float | str]] = []
    for group, title, card_filename in groups:
        group_rows = [row for row in rows if classify_case(row) == group]
        if not group_rows:
            continue
        metrics = summarize_group(group_rows)
        grouped_metrics[group] = metrics
        plot_metric_cards_from_metrics(metrics, output_dir / card_filename, f"{title} Metrics")
        for axis in ("vx", "vy", "wz"):
            csv_rows.append(
                {
                    "group": group,
                    "axis": axis,
                    "mean_error": metrics[f"{axis}_mean_error"],
                    "signed_mean_error": metrics[f"{axis}_signed_mean_error"],
                    "max_error": metrics[f"{axis}_max_error"],
                }
            )

    with (output_dir / "group_metrics.json").open("w") as f:
        json.dump(grouped_metrics, f, indent=2)
        f.write("\n")

    with (output_dir / "group_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "axis", "mean_error", "signed_mean_error", "max_error"])
        writer.writeheader()
        writer.writerows(csv_rows)


def write_summary_csv(summary: list[dict[str, float | str]], out_path: Path) -> None:
    keys = [
        "case",
        "cmd_vx",
        "cmd_vy",
        "cmd_wz",
        "actual_vx",
        "actual_vy",
        "actual_wz",
        "err_vx",
        "err_vy",
        "err_wz",
        "abs_err_vx",
        "abs_err_vy",
        "abs_err_wz",
        "max_abs_err_vx",
        "max_abs_err_vy",
        "max_abs_err_wz",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in summary:
            writer.writerow({key: row[key] for key in keys})


def write_global_metrics(metrics: dict[str, float], out_path: Path) -> None:
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("/tmp/g1_mujoco_tracking.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup_s", type=float, default=0.5)
    parser.add_argument("--max_case_time_s", type=float, default=8.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.csv, args.warmup_s, args.max_case_time_s)
    summary = summarize(rows)
    if not summary:
        raise SystemExit(f"No rows found in {args.csv} after warmup_s={args.warmup_s}")
    global_metrics = summarize_global(rows)

    raw_copy_path = args.output / "raw.csv"
    if args.csv.resolve() != raw_copy_path.resolve():
        shutil.copyfile(args.csv, raw_copy_path)
    write_summary_csv(summary, args.output / "summary.csv")
    write_global_metrics(global_metrics, args.output / "global_metrics.json")
    plot_metric_cards(summary, args.output / "mujoco_tracking_cards.png")
    plot_tracking_performance(rows, args.output / "tracking_performance.png")
    plot_axis_tracking(rows, "vx", args.output / "tracking_vx.png")
    plot_axis_tracking(rows, "vy", args.output / "tracking_vy.png")
    plot_axis_tracking(rows, "wz", args.output / "tracking_yawz.png")
    plot_grouped_tracking_reports(rows, args.output)
    plot_grouped_error_reports(rows, args.output)
    write_group_metrics(rows, args.output)
    plot_table(summary, args.output / "mujoco_tracking_summary.png")
    plot_errors(summary, args.output / "mujoco_tracking_error.png")
    print(f"Saved: {args.output / 'raw.csv'}")
    print(f"Saved: {args.output / 'summary.csv'}")
    print(f"Saved: {args.output / 'global_metrics.json'}")
    print(f"Saved: {args.output / 'mujoco_tracking_cards.png'}")
    print(f"Saved: {args.output / 'tracking_performance.png'}")
    print(f"Saved: {args.output / 'tracking_vx.png'}")
    print(f"Saved: {args.output / 'tracking_vy.png'}")
    print(f"Saved: {args.output / 'tracking_yawz.png'}")
    print(f"Saved: {args.output / 'tracking_vx_commands.png'}")
    print(f"Saved: {args.output / 'tracking_vy_commands.png'}")
    print(f"Saved: {args.output / 'tracking_yaw_commands.png'}")
    print(f"Saved: {args.output / 'tracking_combo_commands.png'}")
    print(f"Saved: {args.output / 'errors_vx_commands.png'}")
    print(f"Saved: {args.output / 'errors_vy_commands.png'}")
    print(f"Saved: {args.output / 'errors_yaw_commands.png'}")
    print(f"Saved: {args.output / 'errors_combo_commands.png'}")
    print(f"Saved: {args.output / 'cards_vx_commands.png'}")
    print(f"Saved: {args.output / 'cards_vy_commands.png'}")
    print(f"Saved: {args.output / 'cards_yaw_commands.png'}")
    print(f"Saved: {args.output / 'cards_combo_commands.png'}")
    print(f"Saved: {args.output / 'group_metrics.csv'}")
    print(f"Saved: {args.output / 'group_metrics.json'}")
    print(f"Saved: {args.output / 'mujoco_tracking_summary.png'}")
    print(f"Saved: {args.output / 'mujoco_tracking_error.png'}")


if __name__ == "__main__":
    main()
