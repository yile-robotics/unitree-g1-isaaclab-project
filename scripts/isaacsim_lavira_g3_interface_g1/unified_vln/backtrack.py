from __future__ import annotations

"""Pure BACKTRACK route construction helpers.

The model chooses a historical waypoint.  The robot keeps the physical path
that was actually traversed for every committed NAVIGATE action.  Returning to
an old observation pose is therefore implemented by reversing and joining
those successful world-frame paths; no velocity command is replayed.
"""

from dataclasses import dataclass
import math
from typing import Protocol, Sequence

import numpy as np

from .odometry import Pose2D


class BacktrackWaypoint(Protocol):
    waypoint_id: int
    decision_pose: Pose2D | None
    arrival_pose: Pose2D | None
    executed_world_path_xy: np.ndarray | None


@dataclass(frozen=True)
class StoredReverseRoute:
    target_waypoint_id: int
    points_world_xy: np.ndarray
    path_length_m: float
    start_drift_m: float

    @property
    def target_world_xy(self) -> np.ndarray:
        return self.points_world_xy[-1].copy()

    def to_dict(self) -> dict:
        return {
            "strategy": "stored_reverse",
            "target_waypoint": int(self.target_waypoint_id),
            "start_drift_m": float(self.start_drift_m),
            "path_length_m": float(self.path_length_m),
            "points_world_xy": self.points_world_xy.tolist(),
        }


def _validated_path(record: BacktrackWaypoint) -> np.ndarray:
    points = record.executed_world_path_xy
    if points is None:
        raise ValueError(
            f"BACKTRACK waypoint {record.waypoint_id} has no executed world path."
        )
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError(
            f"BACKTRACK waypoint {record.waypoint_id} has invalid path shape "
            f"{points.shape}."
        )
    if points.shape[0] == 0:
        raise ValueError(
            f"BACKTRACK waypoint {record.waypoint_id} has an empty world path."
        )
    return np.ascontiguousarray(points)


def _append_without_duplicate(parts: list[np.ndarray], points: np.ndarray) -> None:
    if points.size == 0:
        return
    if parts and np.linalg.norm(parts[-1][-1] - points[0]) <= 1.0e-6:
        points = points[1:]
    if points.size:
        parts.append(points)


def build_stored_reverse_route(
    records: Sequence[BacktrackWaypoint],
    *,
    target_waypoint_id: int,
    current_pose: Pose2D,
    max_start_drift_m: float,
    max_path_length_m: float,
) -> StoredReverseRoute:
    """Reverse successful NAVIGATE paths through the selected observation pose."""

    records = tuple(records)
    current_pose = current_pose.validated()
    if not records:
        raise ValueError("BACKTRACK requires committed waypoint history.")
    if not isinstance(target_waypoint_id, int):
        raise ValueError("BACKTRACK waypoint must be an integer.")
    if not 0 <= target_waypoint_id < len(records):
        raise ValueError(
            f"BACKTRACK waypoint {target_waypoint_id} is outside history "
            f"[0, {len(records) - 1}]."
        )
    if not math.isfinite(max_start_drift_m) or max_start_drift_m < 0.0:
        raise ValueError("BACKTRACK start tolerance must be finite and non-negative.")
    if not math.isfinite(max_path_length_m) or max_path_length_m <= 0.0:
        raise ValueError("BACKTRACK path limit must be finite and positive.")
    for index, record in enumerate(records):
        if record.waypoint_id != index:
            raise ValueError("BACKTRACK history ids must be contiguous and zero-based.")

    target_record = records[target_waypoint_id]
    if target_record.decision_pose is None:
        raise ValueError(
            f"BACKTRACK waypoint {target_waypoint_id} has no decision pose."
        )
    target_pose = target_record.decision_pose.validated()

    parts: list[np.ndarray] = []
    for record in reversed(records[target_waypoint_id:]):
        _append_without_duplicate(parts, _validated_path(record)[::-1].copy())

    current_xy = np.array([current_pose.x, current_pose.y], dtype=np.float64)
    target_xy = np.array([target_pose.x, target_pose.y], dtype=np.float64)
    if parts:
        merged = np.concatenate(parts, axis=0)
    else:  # Defensive; records is non-empty so this should not normally happen.
        merged = current_xy.reshape(1, 2)

    start_drift = float(np.linalg.norm(current_xy - merged[0]))
    if start_drift > max_start_drift_m:
        raise ValueError(
            "BACKTRACK stored path start is stale: "
            f"drift={start_drift:.3f}m > {max_start_drift_m:.3f}m."
        )
    # Always begin at the measured current pose and end at the exact historical
    # observation pose.  Small connectors absorb normal follower tolerance.
    merged = np.concatenate((current_xy.reshape(1, 2), merged), axis=0)
    merged = np.concatenate((merged, target_xy.reshape(1, 2)), axis=0)
    keep = np.ones(merged.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(merged, axis=0), axis=1) > 1.0e-6
    merged = np.ascontiguousarray(merged[keep])
    if merged.shape[0] == 1:
        path_length = 0.0
    else:
        path_length = float(
            np.sum(np.linalg.norm(np.diff(merged, axis=0), axis=1))
        )
    if path_length > max_path_length_m:
        raise ValueError(
            "BACKTRACK path exceeds execution limit: "
            f"{path_length:.3f}m > {max_path_length_m:.3f}m."
        )
    return StoredReverseRoute(
        target_waypoint_id=target_waypoint_id,
        points_world_xy=merged,
        path_length_m=path_length,
        start_drift_m=start_drift,
    )


def next_route_checkpoint_index(
    points_world_xy: np.ndarray,
    *,
    current_index: int,
    segment_length_m: float,
) -> int:
    """Choose a bounded lookahead checkpoint without skipping the final point."""

    points = np.asarray(points_world_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("BACKTRACK route must be a non-empty Nx2 array.")
    if not 0 <= current_index < points.shape[0]:
        raise ValueError("BACKTRACK route cursor is out of range.")
    if not math.isfinite(segment_length_m) or segment_length_m <= 0.0:
        raise ValueError("BACKTRACK segment length must be finite and positive.")
    if current_index == points.shape[0] - 1:
        return current_index
    accumulated = 0.0
    for index in range(current_index + 1, points.shape[0]):
        accumulated += float(np.linalg.norm(points[index] - points[index - 1]))
        if accumulated >= segment_length_m:
            return index
    return points.shape[0] - 1
