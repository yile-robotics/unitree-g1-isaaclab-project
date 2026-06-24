#!/usr/bin/env python3
"""Plot independent MuJoCo goal-tracking CSV logs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    skipped = 0
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if any(value in (None, "") for value in row.values()):
                skipped += 1
                continue
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if key in {"active_policy"}:
                    parsed[key] = value
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = math.nan
            rows.append(parsed)
    if skipped:
        print(f"skipped {skipped} incomplete CSV row(s)")
    return rows


def series(rows: list[dict[str, float | str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def has_key(rows: list[dict[str, float | str]], key: str) -> bool:
    return key in rows[0]


def target_points(rows: list[dict[str, float | str]]) -> list[tuple[float, float]]:
    x_key = "target_x" if has_key(rows, "target_x") else "goal_x"
    y_key = "target_y" if has_key(rows, "target_y") else "goal_y"
    points: list[tuple[float, float]] = []
    for row in rows:
        point = (float(row[x_key]), float(row[y_key]))
        if not points or point != points[-1]:
            points.append(point)
    return points


def point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    vx = ex - sx
    vy = ey - sy
    wx = px - sx
    wy = py - sy
    segment_len_sq = vx * vx + vy * vy
    if segment_len_sq <= 1.0e-12:
        return math.hypot(px - sx, py - sy)
    projection = max(0.0, min(1.0, (wx * vx + wy * vy) / segment_len_sq))
    closest_x = sx + projection * vx
    closest_y = sy + projection * vy
    return math.hypot(px - closest_x, py - closest_y)


def point_to_path_distance(point: tuple[float, float], path: list[tuple[float, float]]) -> float:
    if not path:
        return math.nan
    if len(path) == 1:
        return math.hypot(point[0] - path[0][0], point[1] - path[0][1])
    return min(
        point_to_segment_distance(point, path[index], path[index + 1])
        for index in range(len(path) - 1)
    )


def cross_track_errors(
    x: list[float],
    y: list[float],
    path: list[tuple[float, float]],
) -> list[float]:
    return [point_to_path_distance((px, py), path) for px, py in zip(x, y)]


def print_cross_track_summary(errors: list[float]) -> None:
    valid_errors = [error for error in errors if math.isfinite(error)]
    if not valid_errors:
        print("mean_cross_track_error=nan")
        print("max_cross_track_error=nan")
        print("rms_cross_track_error=nan")
        return

    mean_error = sum(valid_errors) / len(valid_errors)
    max_error = max(valid_errors)
    rms_error = math.sqrt(sum(error * error for error in valid_errors) / len(valid_errors))
    print(f"mean_cross_track_error={mean_error:.6f}")
    print(f"max_cross_track_error={max_error:.6f}")
    print(f"rms_cross_track_error={rms_error:.6f}")


def plot(rows: list[dict[str, float | str]], out_path: Path) -> None:
    if not rows:
        raise SystemExit("CSV has no rows.")

    x = series(rows, "x")
    y = series(rows, "y")
    t = series(rows, "time")
    dist = series(rows, "dist")
    cmd_vx = series(rows, "cmd_vx")
    cmd_vy = series(rows, "cmd_vy")
    cmd_wz = series(rows, "cmd_wz")
    targets = target_points(rows)
    goal_x, goal_y = targets[-1]
    cte = cross_track_errors(x, y, targets)
    print_cross_track_summary(cte)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    if len(targets) > 1:
        target_x = [point[0] for point in targets]
        target_y = [point[1] for point in targets]
        ax.plot(target_x, target_y, color="tab:cyan", linestyle="--", linewidth=1.6, label="target path")
        ax.scatter(target_x[:-1], target_y[:-1], c="tab:cyan", s=35, label="waypoints")
    ax.plot(x, y, linewidth=2.0, label="robot")
    ax.scatter([x[0]], [y[0]], c="green", s=60, label="start")
    ax.scatter([goal_x], [goal_y], c="red", s=80, marker="*", label="goal")
    ax.scatter([x[-1]], [y[-1]], c="black", s=45, label="final")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("XY Trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t, dist, linewidth=2.0, label="xy distance")
    ax.plot(t, cte, linewidth=1.6, label="cross-track error")
    ax.axhline(0.12, color="red", linestyle="--", linewidth=1.0, label="xy tolerance")
    if has_key(rows, "yaw_error"):
        yaw_error = [abs(value) for value in series(rows, "yaw_error")]
        ax.plot(t, yaw_error, linewidth=1.4, label="abs yaw error")
        ax.axhline(0.30, color="orange", linestyle="--", linewidth=1.0, label="yaw tolerance")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [m or rad]")
    ax.set_title("Tracking Error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(t, cmd_vx, label="vx")
    ax.plot(t, cmd_vy, label="vy")
    ax.plot(t, cmd_wz, label="wz")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("command")
    ax.set_title("Velocity Commands")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    policies = [1.0 if row["active_policy"] == "locomotion" else 0.0 for row in rows]
    ax.step(t, policies, where="post")
    ax.set_yticks([0, 1], labels=["stand", "locomotion"])
    ax.set_xlabel("time [s]")
    ax.set_title("Active Policy")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv",
        nargs="?",
        default="/home/yile/projects/unitree_rl_lab/outputs/goal_tracking/latest/trajectory.csv",
        help="Goal-tracking CSV path.",
    )
    parser.add_argument("--out", default=None, help="Output PNG path.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".png")
    rows = read_rows(csv_path)
    plot(rows, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
