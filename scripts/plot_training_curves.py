"""Export TensorBoard scalar curves from an RSL-RL run as PNG plots and CSV files."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


DEFAULT_TAGS = {
    "overview": [
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Policy/mean_noise_std",
        "Loss/value_function",
        "Loss/surrogate",
        "Loss/entropy",
    ],
    "tracking": [
        "Metrics/base_velocity/error_vel_xy",
        "Metrics/base_velocity/error_vel_yaw",
        "Episode_Reward/track_lin_vel_xy",
        "Episode_Reward/track_ang_vel_z",
    ],
    "stability": [
        "Episode_Termination/time_out",
        "Episode_Termination/base_height",
        "Episode_Termination/bad_orientation",
        "Episode_Reward/flat_orientation_l2",
        "Episode_Reward/base_height",
        "Episode_Reward/feet_slide",
        "Episode_Reward/undesired_contacts",
    ],
    "curriculum": [
        "Curriculum/terrain_levels",
        "Curriculum/lin_vel_cmd_levels",
    ],
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _smooth(values: list[float], weight: float) -> list[float]:
    if not values or weight <= 0.0:
        return values
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(smoothed[-1] * weight + value * (1.0 - weight))
    return smoothed


def _load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    accumulator = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    curves = {}
    for tag in accumulator.Tags().get("scalars", []):
        curves[tag] = [(event.step, event.value) for event in accumulator.Scalars(tag)]
    return curves


def _write_csv(out_dir: Path, curves: dict[str, list[tuple[int, float]]]) -> None:
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for tag, points in curves.items():
        with (csv_dir / f"{_safe_name(tag)}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["step", "value"])
            writer.writerows(points)


def _plot_group(
    out_dir: Path,
    name: str,
    tags: list[str],
    curves: dict[str, list[tuple[int, float]]],
    smooth_weight: float,
) -> bool:
    present_tags = [tag for tag in tags if tag in curves and curves[tag]]
    if not present_tags:
        return False

    fig, axes = plt.subplots(len(present_tags), 1, figsize=(12, max(3, 2.6 * len(present_tags))), sharex=True)
    if len(present_tags) == 1:
        axes = [axes]

    for axis, tag in zip(axes, present_tags):
        steps = [step for step, _ in curves[tag]]
        values = [value for _, value in curves[tag]]
        axis.plot(steps, values, color="#8a8a8a", alpha=0.35, linewidth=1.0, label="raw")
        axis.plot(steps, _smooth(values, smooth_weight), linewidth=1.7, label="smoothed")
        axis.set_title(tag, fontsize=10)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="RSL-RL run directory containing events.out.tfevents.*")
    parser.add_argument("--out_dir", type=Path, default=None, help="Output directory for PNG and CSV files.")
    parser.add_argument("--smooth", type=float, default=0.9, help="EMA smoothing weight from 0.0 to 0.99.")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not list(run_dir.glob("events.out.tfevents.*")):
        raise FileNotFoundError(f"No TensorBoard event file found in: {run_dir}")

    out_dir = args.out_dir or (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = _load_scalars(run_dir)
    _write_csv(out_dir, curves)

    plotted = []
    for name, tags in DEFAULT_TAGS.items():
        if _plot_group(out_dir, name, tags, curves, max(0.0, min(args.smooth, 0.99))):
            plotted.append(name)

    print(f"Loaded {len(curves)} scalar tags from {run_dir}")
    print(f"Wrote CSV files and {len(plotted)} PNG plot groups to {out_dir}")
    if plotted:
        print("Plots: " + ", ".join(f"{name}.png" for name in plotted))


if __name__ == "__main__":
    main()
