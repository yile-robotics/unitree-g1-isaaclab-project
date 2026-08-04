from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


# This order is part of the already-deployed model wire contract.
DIRECTION_ORDER = ("forward", "left", "behind", "right")


@dataclass(frozen=True)
class ViewFrame:
    """One RGB-D observation without any world-frame pose."""

    direction: str
    frame_id: int
    sim_step: int
    timestamp: float
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray

    def validated(self) -> "ViewFrame":
        if self.direction not in DIRECTION_ORDER:
            raise ValueError(f"Unknown camera direction {self.direction!r}.")
        rgb = np.asarray(self.rgb)
        depth = np.asarray(self.depth_m)
        K = np.asarray(self.K)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(
                f"{self.direction} RGB must be HxWx3 uint8, got {rgb.shape}/{rgb.dtype}."
            )
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"{self.direction} depth {depth.shape} is not aligned with RGB {rgb.shape[:2]}."
            )
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            raise ValueError(f"{self.direction} K must be a finite 3x3 matrix.")
        if not np.isfinite(float(self.timestamp)):
            raise ValueError("Camera timestamp must be finite.")
        return self


@dataclass(frozen=True)
class PanoramaBundle:
    """Four model views captured without exposing a robot/world transform."""

    bundle_id: int
    sim_step: int
    timestamp: float
    views: Mapping[str, ViewFrame]

    def validated(self) -> "PanoramaBundle":
        if tuple(self.views) != DIRECTION_ORDER:
            raise ValueError(
                f"Panorama order must be exactly {DIRECTION_ORDER}, got {tuple(self.views)}."
            )
        shape = None
        for direction in DIRECTION_ORDER:
            frame = self.views[direction].validated()
            if frame.direction != direction:
                raise ValueError(
                    f"Panorama key {direction!r} contains {frame.direction!r} frame."
                )
            if shape is None:
                shape = frame.rgb.shape[:2]
            elif frame.rgb.shape[:2] != shape:
                raise ValueError("All four model RGB views must share one resolution.")
        return self
