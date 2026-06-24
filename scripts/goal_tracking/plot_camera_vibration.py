#!/usr/bin/env python3
"""Plot raw head-camera-point motion during goal tracking."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_numeric_csv(path: Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row or any(value in (None, "") for value in row.values()):
                continue
            parsed: dict[str, float] = {}
            try:
                for key, value in row.items():
                    parsed[key] = float(value)
            except (TypeError, ValueError):
                continue
            for key, value in parsed.items():
                columns.setdefault(key, []).append(value)
    return {key: np.asarray(values, dtype=float) for key, values in columns.items()}


def read_policy_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    locomotion: list[bool] = []
    if not path.exists():
        return np.asarray(times), np.asarray(locomotion, dtype=bool)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                times.append(float(row["time"]))
                locomotion.append(row["active_policy"] == "locomotion")
            except (KeyError, TypeError, ValueError):
                continue
    return np.asarray(times), np.asarray(locomotion, dtype=bool)


def estimate_sample_rate(time: np.ndarray) -> float:
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    return 1.0 / float(np.median(dt)) if dt.size else 50.0


def align_policy(
    camera_time: np.ndarray,
    policy_time: np.ndarray,
    policy_locomotion: np.ndarray,
) -> np.ndarray:
    if not policy_time.size:
        return np.zeros(camera_time.size, dtype=bool)
    offset = camera_time[-1] - policy_time[-1]
    aligned_time = camera_time - offset
    indices = np.searchsorted(policy_time, aligned_time, side="right") - 1
    valid = (indices >= 0) & (indices < policy_locomotion.size)
    result = np.zeros(camera_time.size, dtype=bool)
    result[valid] = policy_locomotion[indices[valid]]
    return result


def rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return math.sqrt(float(np.mean(finite * finite))) if finite.size else math.nan


def peak_to_peak(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.ptp(finite)) if finite.size else math.nan


def absolute_peak(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.max(np.abs(finite))) if finite.size else math.nan


def rotate_world_to_heading_frame(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    yaw: np.ndarray,
) -> dict[str, np.ndarray]:
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return {
        "forward": cos_yaw * x + sin_yaw * y,
        "lateral": -sin_yaw * x + cos_yaw * y,
        "vertical": z.copy(),
    }


def vector_magnitude(components: dict[str, np.ndarray]) -> np.ndarray:
    return np.sqrt(sum(values * values for values in components.values()))


def print_vector_metrics(
    label: str,
    mask: np.ndarray,
    linear_velocity: dict[str, np.ndarray],
    angular_velocity: dict[str, np.ndarray],
    linear_acceleration: dict[str, np.ndarray],
) -> None:
    if not np.any(mask):
        print(f"{label}: no samples")
        return
    print(f"{label}: samples={int(np.count_nonzero(mask))}")
    groups = (
        ("linear_velocity", linear_velocity, "m_s"),
        ("angular_velocity", angular_velocity, "rad_s"),
        ("linear_acceleration", linear_acceleration, "m_s2"),
    )
    for group_name, components, unit_name in groups:
        for axis, values in components.items():
            selected = values[mask]
            print(
                f"  {group_name}_{axis}_rms_{unit_name}={rms(selected):.4f} "
                f"{group_name}_{axis}_abs_peak_{unit_name}={absolute_peak(selected):.4f}"
            )
        magnitude = vector_magnitude(components)[mask]
        print(
            f"  {group_name}_magnitude_rms_{unit_name}={rms(magnitude):.4f} "
            f"{group_name}_magnitude_peak_{unit_name}={absolute_peak(magnitude):.4f}"
        )


def shade_locomotion(ax: plt.Axes, time: np.ndarray, locomotion: np.ndarray) -> None:
    if not np.any(locomotion):
        return
    edges = np.flatnonzero(np.diff(np.r_[False, locomotion, False]))
    for start, stop in edges.reshape(-1, 2):
        ax.axvspan(time[start], time[min(stop, time.size - 1)], color="tab:blue", alpha=0.08)


def plot(
    camera: dict[str, np.ndarray],
    trajectory_path: Path,
    out_path: Path,
) -> None:
    required = {"time", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "wx", "wy", "wz", "ax", "ay", "az"}
    missing = sorted(required - camera.keys())
    if missing:
        raise SystemExit(f"camera CSV missing columns: {', '.join(missing)}")

    time = camera["time"]
    if time.size < 3:
        raise SystemExit("camera CSV has too few samples")
    time = time - time[0]
    sample_rate = estimate_sample_rate(time)
    yaw = np.unwrap(camera["yaw"])
    linear_velocity = rotate_world_to_heading_frame(
        camera["vx"], camera["vy"], camera["vz"], yaw
    )
    angular_velocity = rotate_world_to_heading_frame(
        camera["wx"], camera["wy"], camera["wz"], yaw
    )
    linear_acceleration = rotate_world_to_heading_frame(
        camera["ax"], camera["ay"], camera["az"], yaw
    )

    policy_time, policy_locomotion = read_policy_log(trajectory_path)
    locomotion = align_policy(camera["time"], policy_time, policy_locomotion)
    stand = ~locomotion
    print(f"sample_rate_hz={sample_rate:.3f}")
    print("frame=robot_heading_xy_world_vertical_z")
    print("smoothing=none")
    all_samples = np.ones(time.size, dtype=bool)
    print_vector_metrics(
        "all", all_samples, linear_velocity, angular_velocity, linear_acceleration
    )
    print_vector_metrics(
        "stand", stand, linear_velocity, angular_velocity, linear_acceleration
    )
    print_vector_metrics(
        "locomotion", locomotion, linear_velocity, angular_velocity, linear_acceleration
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    ax = axes[0, 0]
    for axis, values in linear_velocity.items():
        ax.plot(time, values, linewidth=1.0, label=axis)
    ax.set_ylabel("linear velocity [m/s]")
    ax.set_title("Camera-Point Linear Velocity (Robot Heading Frame)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    angular_labels = {"forward": "roll rate", "lateral": "pitch rate", "vertical": "yaw rate"}
    for axis, values in angular_velocity.items():
        ax.plot(time, values, linewidth=1.0, label=angular_labels[axis])
    ax.set_ylabel("angular velocity [rad/s]")
    ax.set_title("Camera-Point Angular Velocity (Robot Heading Frame)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    for axis, values in linear_acceleration.items():
        ax.plot(time, values, linewidth=0.9, label=axis)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("linear acceleration [m/s²]")
    ax.set_title("Camera-Point Linear Acceleration (Robot Heading Frame)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(time, vector_magnitude(linear_velocity), linewidth=1.0, label="linear speed [m/s]")
    ax.plot(time, vector_magnitude(angular_velocity), linewidth=1.0, label="angular speed [rad/s]")
    ax.plot(
        time,
        vector_magnitude(linear_acceleration),
        linewidth=0.9,
        label="acceleration [m/s²]",
    )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("raw magnitude (see legend units)")
    ax.set_title("Camera-Point Total Motion Magnitudes")
    ax.grid(True, alpha=0.3)
    ax.legend()

    for ax in axes.flat:
        shade_locomotion(ax, time, locomotion)

    fig.suptitle(
        "G1 Goal Tracking Head-Camera-Point Raw Motion\n"
        "Robot heading frame, no smoothing; blue shading: locomotion policy"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="?",
        default="/home/yile/projects/unitree_rl_lab/outputs/goal_tracking/latest/camera_vibration.csv",
        help="Camera vibration CSV path.",
    )
    parser.add_argument(
        "--trajectory",
        default="/home/yile/projects/unitree_rl_lab/outputs/goal_tracking/latest/trajectory.csv",
        help="Goal-tracking CSV used to identify stand/locomotion periods.",
    )
    parser.add_argument("--out", default=None, help="Output PNG path.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"camera CSV not found: {csv_path}")
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".png")
    plot(read_numeric_csv(csv_path), Path(args.trajectory), out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
