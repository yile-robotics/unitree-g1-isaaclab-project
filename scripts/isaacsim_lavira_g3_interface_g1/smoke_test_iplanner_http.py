#!/usr/bin/env python3
from __future__ import annotations

"""Exercise navigator_reset and pointgoal_step through the production client."""

import argparse
from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from unified_vln.iplanner_client import IPlannerClient  # noqa: E402
from unified_vln.types import ViewFrame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8888")
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Uni-LaViRA's original iPlanner client uses a fixed 5 second timeout.",
    )
    args = parser.parse_args()

    height, width = 480, 640
    frame = ViewFrame(
        direction="forward",
        frame_id=1,
        sim_step=1,
        timestamp=0.02,
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        depth_m=np.full((height, width), 2.0, dtype=np.float32),
        K=np.array(
            [[384.0, 0.0, 320.0], [0.0, 384.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )
    client = IPlannerClient(args.url, timeout_s=args.timeout)
    trajectory, fear = client.get_plan(frame, np.array([2.0, 0.0]))
    if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] < 3:
        raise RuntimeError(f"Unexpected trajectory shape: {trajectory.shape}")
    if not np.all(np.isfinite(trajectory)) or not np.isfinite(fear):
        raise RuntimeError("HTTP iPlanner returned non-finite data.")
    print("HTTP_SMOKE=PASS")
    print(f"TRAJECTORY_SHAPE={trajectory.shape}")
    print(f"FEAR={fear:.6f}")
    print(f"TRAJECTORY_END={trajectory[-1, :3].tolist()}")


if __name__ == "__main__":
    main()
