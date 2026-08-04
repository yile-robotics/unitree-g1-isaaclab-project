from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_contract import NavigationDecisionResponse
from .types import ViewFrame


@dataclass(frozen=True)
class LocalTargetProjection:
    """A target expressed directly in the robot frame after the requested turn."""

    direction: str
    bbox_xyxy: tuple[int, int, int, int]
    pixel_uv: tuple[int, int]
    depth_m: float
    depth_window_xyxy_exclusive: tuple[int, int, int, int]
    valid_depth_count: int
    goal_after_turn_xy_m: np.ndarray

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "bbox_xyxy": list(self.bbox_xyxy),
            "pixel_uv": list(self.pixel_uv),
            "depth_m": self.depth_m,
            "depth_window_xyxy_exclusive": list(
                self.depth_window_xyxy_exclusive
            ),
            "valid_depth_count": self.valid_depth_count,
            "goal_after_turn_xy_m": self.goal_after_turn_xy_m.tolist(),
            "coordinate_convention": "robot_after_turn: x=forward,y=left",
        }


def project_selected_view_target(
    frame: ViewFrame,
    response: NavigationDecisionResponse,
    *,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
    window_radius_px: int = 3,
    depth_percentile: float = 30.0,
) -> LocalTargetProjection:
    """Use bbox + depth from the selected pre-turn camera, without world poses.

    The selected camera optical direction is intentionally treated as the robot's
    forward direction after an ideal relative turn.  Only horizontal pinhole
    geometry is needed for iPlanner's (forward, left) point goal.
    """

    frame.validated()
    action = response.action.upper()
    if action == "BACKTRACK":
        raise ValueError("BACKTRACK has no bbox target to project.")
    if response.direction != frame.direction:
        raise ValueError(
            f"Response selected {response.direction!r}, got {frame.direction!r} frame."
        )
    if not 0.0 < min_depth_m < max_depth_m:
        raise ValueError("Depth range must satisfy 0 < min < max.")
    if window_radius_px < 0:
        raise ValueError("Depth window radius must be non-negative.")
    if not 0.0 <= depth_percentile <= 100.0:
        raise ValueError("Depth percentile must lie in [0, 100].")

    height, width = frame.depth_m.shape
    bbox = response.clipped_bbox(width, height)
    x1, _y1, x2, y2 = bbox
    u = int((x1 + x2) / 2.0)
    v = int(y2)

    x_min = max(0, u - window_radius_px)
    x_max = min(width, u + window_radius_px + 1)
    y_min = max(0, v - window_radius_px)
    y_max = min(height, v + window_radius_px + 1)
    patch = np.asarray(
        frame.depth_m[y_min:y_max, x_min:x_max], dtype=np.float64
    )
    valid = patch[
        np.isfinite(patch)
        & (patch >= float(min_depth_m))
        & (patch <= float(max_depth_m))
    ]
    if valid.size == 0:
        raise ValueError(
            f"No valid selected-view depth around pixel {(u, v)}; no forward fallback generated."
        )
    depth_m = float(np.percentile(valid, depth_percentile))

    K = np.asarray(frame.K, dtype=np.float64)
    fx = float(K[0, 0])
    principal_x = float(K[0, 2])
    if not np.isfinite(fx) or fx <= 0.0 or not np.isfinite(principal_x):
        raise ValueError("Selected camera has invalid fx/cx intrinsics.")

    camera_x_right_m = (float(u) - principal_x) * depth_m / fx
    goal = np.array(
        [depth_m, -camera_x_right_m],
        dtype=np.float64,
    )
    return LocalTargetProjection(
        direction=frame.direction,
        bbox_xyxy=bbox,
        pixel_uv=(u, v),
        depth_m=depth_m,
        depth_window_xyxy_exclusive=(x_min, y_min, x_max, y_max),
        valid_depth_count=int(valid.size),
        goal_after_turn_xy_m=goal,
    )
