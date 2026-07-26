from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from goal_tracking.path import (  # noqa: E402
    RobotPose2D,
    Waypoint,
    WaypointPathFollower,
    prepare_fmm_path_for_execution,
)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        path_lookahead_distance=0.30,
        goal_tolerance=0.12,
        yaw_tolerance=0.25,
        goal_slow_radius=0.60,
        goal_xy_kp=0.70,
        goal_yaw_kp=1.00,
        max_goal_vx=0.45,
        max_goal_vy=0.35,
        max_goal_wz=0.35,
    )


def _plan(points: list[list[float]], path_length: float | None = None):
    array = np.asarray(points, dtype=np.float64)
    if path_length is None:
        path_length = float(np.sum(np.linalg.norm(np.diff(array, axis=0), axis=1)))
    return SimpleNamespace(
        waypoints_world_xy=array,
        start_world_xy=array[0],
        path_length_m=path_length,
        bundle_id=7,
    )


class _FakeCommandController:
    def __init__(self):
        self.zero_called = False
        self.command = None

    def zero(self) -> None:
        self.zero_called = True

    def set_requested(self, vx: float, vy: float, wz: float) -> None:
        self.command = (vx, vy, wz)


class _FakeSwitchState:
    def __init__(self, tilt: float = 0.0):
        self.tilt = tilt
        self.stand_requested = False
        self.active_mode = "locomotion"
        self.transition_mode = "none"

    def current_tilt_angle(self) -> float:
        return self.tilt

    def request_stand(self) -> None:
        self.stand_requested = True

    def request_locomotion(self) -> None:
        return


def _raw_env(x: float, y: float, yaw: float = 0.0):
    quat = torch.tensor(
        [[math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)]],
        dtype=torch.float32,
    )
    data = SimpleNamespace(
        root_pos_w=torch.tensor([[x, y, 0.8]], dtype=torch.float32),
        root_quat_w=quat,
    )
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})


class FMMPathExecutionPreparationTest(unittest.TestCase):
    def test_world_xy_path_gets_tangent_yaws(self) -> None:
        prepared = prepare_fmm_path_for_execution(
            _plan([[1.0, 2.0], [1.5, 2.0], [1.5, 2.5]]),
            RobotPose2D(1.02, 2.0, 0.3),
            max_start_drift_m=0.15,
            max_path_length_m=2.0,
        )

        self.assertEqual(len(prepared.waypoints), 3)
        self.assertAlmostEqual(prepared.waypoints[0].yaw, 0.0)
        self.assertAlmostEqual(prepared.waypoints[1].yaw, math.pi / 2.0)
        self.assertAlmostEqual(prepared.waypoints[2].yaw, math.pi / 2.0)
        self.assertAlmostEqual(prepared.start_drift_m, 0.02)

    def test_stale_start_and_long_path_are_rejected(self) -> None:
        plan = _plan([[0.0, 0.0], [1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "stale"):
            prepare_fmm_path_for_execution(
                plan,
                RobotPose2D(0.3, 0.0, 0.0),
                max_start_drift_m=0.15,
                max_path_length_m=2.0,
            )
        with self.assertRaisesRegex(ValueError, "longer"):
            prepare_fmm_path_for_execution(
                plan,
                RobotPose2D(0.0, 0.0, 0.0),
                max_start_drift_m=0.15,
                max_path_length_m=0.5,
            )

    def test_cross_track_error_aborts_and_requests_stand(self) -> None:
        follower = WaypointPathFollower(
            _raw_env(0.0, 0.6), None, [], _args()
        )
        follower.replace_waypoints(
            [Waypoint(0.0, 0.0, 0.0), Waypoint(1.0, 0.0, 0.0)],
            source="test_fmm",
            cross_track_abort_m=0.40,
            tilt_abort_rad=0.50,
            velocity_limits=(0.20, 0.12, 0.25),
            lookahead_distance_m=0.20,
        )
        follower.start()
        controller = _FakeCommandController()
        switch = _FakeSwitchState()

        follower.update(controller, switch)

        self.assertFalse(follower.enabled)
        self.assertIn("cross-track", follower.abort_reason)
        self.assertTrue(controller.zero_called)
        self.assertTrue(switch.stand_requested)

    def test_tilt_aborts_before_velocity_is_requested(self) -> None:
        follower = WaypointPathFollower(_raw_env(0.0, 0.0), None, [], _args())
        follower.replace_waypoints(
            [Waypoint(0.0, 0.0, 0.0), Waypoint(1.0, 0.0, 0.0)],
            source="test_fmm",
            tilt_abort_rad=0.50,
        )
        follower.start()
        controller = _FakeCommandController()
        switch = _FakeSwitchState(tilt=0.60)

        follower.update(controller, switch)

        self.assertFalse(follower.enabled)
        self.assertIn("tilt", follower.abort_reason)
        self.assertIsNone(controller.command)
        self.assertTrue(switch.stand_requested)


if __name__ == "__main__":
    unittest.main()
