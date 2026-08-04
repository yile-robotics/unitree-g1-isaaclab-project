from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import requests

from .types import ViewFrame


class IPlannerClient:
    """HTTP client compatible with Uni-LaViRA G1's local iPlanner server."""

    def __init__(self, server_url: str, timeout_s: float = 5.0):
        if not server_url.strip():
            raise ValueError("iPlanner URL must not be empty.")
        if timeout_s <= 0.0:
            raise ValueError("iPlanner timeout must be positive.")
        self.server_url = server_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()
        self.initialized = False

    def reset(self, intrinsic: Sequence[Sequence[float]]) -> None:
        K = np.asarray(intrinsic, dtype=np.float64)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            raise ValueError("iPlanner intrinsic must be a finite 3x3 matrix.")
        response = self.session.post(
            f"{self.server_url}/navigator_reset",
            json={
                "intrinsic": K.tolist(),
                "stop_threshold": 0.1,
                "batch_size": 1,
            },
            timeout=self.timeout_s,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"iPlanner reset failed: {response.status_code} {response.text}"
            )
        self.initialized = True

    @staticmethod
    def encode_depth_protocol(depth_m: np.ndarray) -> np.ndarray:
        """Encode meters as uint16 0.1 mm units without the original wraparound."""
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.ndim != 2:
            raise ValueError(f"iPlanner depth must be HxW, got {depth.shape}.")
        valid = np.isfinite(depth) & (depth > 0.0)
        encoded = np.zeros(depth.shape, dtype=np.uint16)
        # uint16 at 0.1 mm has a hard maximum of 6.5535 m.
        clipped = np.clip(depth[valid], 0.0, 6.5535)
        encoded[valid] = np.round(clipped * 10000.0).astype(np.uint16)
        return encoded

    def get_plan(
        self,
        front_frame: ViewFrame,
        goal_local_xy: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        import cv2

        front_frame.validated()
        if front_frame.direction != "forward":
            raise ValueError("iPlanner must receive the post-rotation forward RGB-D frame.")
        goal = np.asarray(goal_local_xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(goal)):
            raise ValueError("iPlanner local goal must be finite.")
        if not self.initialized:
            self.reset(front_frame.K)

        rgb_bgr = cv2.cvtColor(
            np.ascontiguousarray(front_frame.rgb), cv2.COLOR_RGB2BGR
        )
        ok_rgb, rgb_encoded = cv2.imencode(".png", rgb_bgr)
        ok_depth, depth_encoded = cv2.imencode(
            ".png", self.encode_depth_protocol(front_frame.depth_m)
        )
        if not ok_rgb or not ok_depth:
            raise RuntimeError("OpenCV failed to encode iPlanner RGB-D request.")

        response = self.session.post(
            f"{self.server_url}/pointgoal_step",
            files={
                "image": ("rgb.png", rgb_encoded.tobytes(), "image/png"),
                "depth": ("depth.png", depth_encoded.tobytes(), "image/png"),
            },
            data={
                "goal_data": json.dumps(
                    {"goal_x": [float(goal[0])], "goal_y": [float(goal[1])]}
                )
            },
            timeout=self.timeout_s,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"iPlanner request failed: {response.status_code} {response.text}"
            )
        payload = response.json()
        try:
            trajectory = np.asarray(payload["trajectory"][0], dtype=np.float64)
            fear = float(payload["all_values"][0][0])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("iPlanner returned an invalid trajectory payload.") from exc
        if (
            trajectory.ndim != 2
            or trajectory.shape[0] < 2
            or trajectory.shape[1] < 2
            or not np.all(np.isfinite(trajectory))
        ):
            raise RuntimeError(
                f"iPlanner returned invalid trajectory shape/data {trajectory.shape}."
            )
        return trajectory, fear
