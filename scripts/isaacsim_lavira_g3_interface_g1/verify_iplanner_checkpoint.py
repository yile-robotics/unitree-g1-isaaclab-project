#!/usr/bin/env python3
from __future__ import annotations

"""Strict-load a converted checkpoint and run one deterministic CPU inference."""

import argparse
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--module-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    module_dir = args.module_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    config = args.config.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config.is_file():
        raise FileNotFoundError(config)
    sys.path.insert(0, str(module_dir))

    from iplanner_agent import IPlannerAgent

    K = np.array(
        [[384.0, 0.0, 320.0], [0.0, 384.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    agent = IPlannerAgent(
        K,
        model_path=str(checkpoint),
        model_config_path=str(config),
        device=args.device,
    )
    depth = np.full((1, 480, 640, 1), 2.0, dtype=np.float32)
    goal = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    keypoints, trajectory, fear = agent.step_pointgoal(depth, goal)

    keypoints_np = keypoints.detach().cpu().numpy()
    trajectory_np = trajectory.detach().cpu().numpy()
    fear_np = fear.detach().cpu().numpy()
    if keypoints_np.shape != (1, 5, 3):
        raise RuntimeError(f"Unexpected keypoint shape: {keypoints_np.shape}")
    if trajectory_np.ndim != 3 or trajectory_np.shape[0] != 1:
        raise RuntimeError(f"Unexpected trajectory shape: {trajectory_np.shape}")
    if trajectory_np.shape[1] < 2 or trajectory_np.shape[2] != 3:
        raise RuntimeError(f"Unexpected trajectory shape: {trajectory_np.shape}")
    if fear_np.shape != (1, 1):
        raise RuntimeError(f"Unexpected fear shape: {fear_np.shape}")
    if not all(
        np.all(np.isfinite(array))
        for array in (keypoints_np, trajectory_np, fear_np)
    ):
        raise RuntimeError("iPlanner inference produced non-finite values.")

    print("STRICT_LOAD=PASS")
    print(f"KEYPOINT_SHAPE={keypoints_np.shape}")
    print(f"TRAJECTORY_SHAPE={trajectory_np.shape}")
    print(f"FEAR={float(fear_np[0, 0]):.6f}")
    print(f"TRAJECTORY_END={trajectory_np[0, -1].tolist()}")


if __name__ == "__main__":
    main()
