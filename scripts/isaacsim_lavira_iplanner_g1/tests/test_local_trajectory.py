from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.local_trajectory import (  # noqa: E402
    LocalFollowerConfig,
    LocalTrajectoryFollower,
    truncate_trajectory_for_safety,
)
from unified_vln.odometry import Pose2D  # noqa: E402


class MutableOdometry:
    def __init__(self):
        self.pose = Pose2D(0.0, 0.0, 0.0, 0.0)

    def get_pose(self):
        return self.pose


class LocalTrajectoryTest(unittest.TestCase):
    def test_no_odom_updates_goal_from_applied_command(self):
        follower = LocalTrajectoryFollower(
            LocalFollowerConfig(
                goal_tolerance_m=0.1,
                blind_yaw_radius_m=0.0,
                replan_interval_s=1.0,
            )
        )
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        follower.start(path, np.array([2.0, 0.0]))
        output = follower.update(1.0, np.array([0.5, 0.0, 0.0]))
        np.testing.assert_allclose(output.goal_local_xy, [1.5, 0.0])
        self.assertGreater(output.command[0], 0.0)

    def test_optional_odom_keeps_fixed_goal(self):
        odom = MutableOdometry()
        follower = LocalTrajectoryFollower(
            LocalFollowerConfig(
                goal_tolerance_m=0.1,
                blind_yaw_radius_m=0.0,
                replan_interval_s=1.0,
            ),
            odometry=odom,
        )
        path = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        follower.start(path, np.array([2.0, 0.0]))
        odom.pose = Pose2D(1.0, 0.0, 0.0, 1.0)
        output = follower.update(0.1, np.zeros(3))
        np.testing.assert_allclose(output.goal_local_xy, [1.0, 0.0])

    def test_no_odom_passed_goal_guard_reports_full_local_goal(self):
        follower = LocalTrajectoryFollower(
            LocalFollowerConfig(
                goal_tolerance_m=0.1,
                blind_yaw_radius_m=0.0,
                replan_interval_s=1.0,
            )
        )
        path = np.array([[0.0, 0.0, 0.0], [0.5, 0.4, 0.0]])
        follower.start(path, np.array([0.5, 0.4]))
        output = follower.update(1.0, np.array([0.61, 0.0, 0.0]))
        self.assertIsNotNone(output.abort_reason)
        self.assertIn("without odometry", output.abort_reason)
        self.assertIn("y=0.400m", output.abort_reason)
        self.assertIn("distance=", output.abort_reason)

    def test_odom_does_not_abort_only_because_goal_x_is_behind(self):
        odom = MutableOdometry()
        follower = LocalTrajectoryFollower(
            LocalFollowerConfig(
                goal_tolerance_m=0.1,
                blind_yaw_radius_m=0.0,
                replan_interval_s=1.0,
            ),
            odometry=odom,
        )
        path = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
        follower.start(path, np.array([1.0, 0.5]))
        odom.pose = Pose2D(1.2, 0.0, 0.0, 1.0)
        output = follower.update(0.1, np.zeros(3))
        self.assertIsNone(output.abort_reason)
        self.assertFalse(output.reached)
        self.assertLess(output.goal_local_xy[0], -0.1)
        self.assertGreater(output.command[0], 0.0)

    def test_safe_distance_truncates_path(self):
        path = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
        )
        truncated = truncate_trajectory_for_safety(path, 0.5)
        np.testing.assert_allclose(truncated[-1, :2], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
