"""Export TensorBoard scalar curves from an RSL-RL run as PNG and CSV files."""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def smooth(values: list[float], weight: float) -> list[float]:
    if not values or weight <= 0.0:
        return values
    result = [values[0]]
    for value in values[1:]:
        result.append(result[-1] * weight + value * (1.0 - weight))
    return result


def load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    accumulator = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags().get("scalars", [])
    }


def write_csv(out_dir: Path, curves: dict[str, list[tuple[int, float]]]) -> None:
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for tag, points in curves.items():
        with (csv_dir / f"{safe_name(tag)}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["iteration", "value"])
            writer.writerows(points)


def plot_group(
    out_dir: Path,
    name: str,
    tags: list[str],
    curves: dict[str, list[tuple[int, float]]],
    smooth_weight: float,
) -> bool:
    tags = [tag for tag in tags if curves.get(tag)]
    if not tags:
        return False

    fig, axes = plt.subplots(len(tags), 1, figsize=(12, max(3.0, 2.5 * len(tags))), sharex=True)
    if len(tags) == 1:
        axes = [axes]

    for axis, tag in zip(axes, tags):
        steps, values = zip(*curves[tag])
        axis.plot(steps, values, color="0.6", alpha=0.35, linewidth=1.0, label="raw")
        axis.plot(steps, smooth(list(values), smooth_weight), linewidth=1.7, label="smoothed")
        axis.set_title(tag, fontsize=10)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    axes[-1].set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=160)
    plt.close(fig)
    return True


def export_plots(run_dir: Path, out_dir: Path | None = None, smooth_weight: float = 0.9) -> Path:
    run_dir = run_dir.expanduser().resolve()
    if not list(run_dir.glob("events.out.tfevents.*")):
        raise FileNotFoundError(f"No TensorBoard event file found in: {run_dir}")

    out_dir = out_dir or (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = load_scalars(run_dir)
    write_csv(out_dir, curves)

    groups = {
        "overview": [
            "Train/mean_reward",
            "Train/mean_episode_length",
            "Policy/mean_noise_std",
            "Perf/total_fps",
        ],
        "losses": sorted(tag for tag in curves if tag.startswith("Loss/")),
        "tracking": sorted(
            tag
            for tag in curves
            if tag.startswith("Metrics/base_velocity/") or tag in {
                "Episode_Reward/track_lin_vel_xy",
                "Episode_Reward/track_ang_vel_z",
            }
        ),
        "rewards": sorted(tag for tag in curves if tag.startswith("Episode_Reward/")),
        "terminations": sorted(tag for tag in curves if tag.startswith("Episode_Termination/")),
        "curriculum": sorted(tag for tag in curves if tag.startswith("Curriculum/")),
    }
    for name, tags in groups.items():
        plot_group(out_dir, name, tags, curves, max(0.0, min(smooth_weight, 0.99)))

    print(f"[INFO] Training plots and CSV files saved to: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--smooth", type=float, default=0.9)
    args = parser.parse_args()
    export_plots(args.run_dir, args.out_dir, args.smooth)


if __name__ == "__main__":
    main()
