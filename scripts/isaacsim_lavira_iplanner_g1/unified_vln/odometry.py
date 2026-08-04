from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float
    timestamp: float

    def validated(self) -> "Pose2D":
        values = np.array([self.x, self.y, self.yaw, self.timestamp], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Odometry pose contains non-finite values.")
        return self


class OdometryProvider(Protocol):
    def get_pose(self) -> Pose2D | None:
        """Return the latest valid odometry pose, or None when unavailable."""


class NullOdometryProvider:
    def get_pose(self) -> None:
        return None


def wrap_to_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def local_points_to_fixed(points_xy: np.ndarray, pose: Pose2D) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    return points @ rotation.T + np.array([pose.x, pose.y], dtype=np.float64)


def fixed_points_to_local(points_xy: np.ndarray, pose: Pose2D) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    shifted = points - np.array([pose.x, pose.y], dtype=np.float64)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    inverse_rotation = np.array([[c, s], [-s, c]], dtype=np.float64)
    return shifted @ inverse_rotation.T
