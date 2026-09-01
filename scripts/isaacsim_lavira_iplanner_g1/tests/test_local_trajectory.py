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
        # Uni 的无 odometry 线速度经验比例为 0.7：2.0 - 0.5*1.0*0.7。
        np.testing.assert_allclose(output.goal_local_xy, [1.65, 0.0])
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

    def test_odom_matches_uni_by_leaving_old_local_path_unchanged(self):
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
        np.testing.assert_allclose(output.target_local_xy, [2.0, 0.0])

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
        output = follower.update(1.0, np.array([0.9, 0.0, 0.0]))
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

    def test_lookahead_index_only_moves_forward(self):
        follower = LocalTrajectoryFollower(
            LocalFollowerConfig(
                lookahead_m=0.5,
                goal_tolerance_m=0.1,
                blind_yaw_radius_m=0.0,
                replan_interval_s=10.0,
            )
        )
        path = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [1.8, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        follower.start(path, np.array([3.0, 0.0]))

        first = follower.update(0.0, np.zeros(3))
        self.assertEqual(follower.current_idx, 1)
        np.testing.assert_allclose(first.target_local_xy, [0.6, 0.0])

        second = follower.update(1.0, np.array([0.2, 0.0, 0.0]))
        self.assertEqual(follower.current_idx, 2)
        np.testing.assert_allclose(second.target_local_xy, [1.06, 0.0])

        # 按 Uni 的 0.7 比例累计估计前进 0.77m 后，path[0] 已在机器人后方。
        # 旧实现会因其
        # 距离再次大于 lookahead 而选回 path[0]；持久索引应继续选 path[3]。
        third = follower.update(1.0, np.array([0.9, 0.0, 0.0]))
        self.assertEqual(follower.current_idx, 3)
        np.testing.assert_allclose(third.target_local_xy, [1.03, 0.0])

    def test_replacing_path_resets_lookahead_index(self):
        follower = LocalTrajectoryFollower(
            LocalFollowerConfig(
                lookahead_m=0.5,
                goal_tolerance_m=0.1,
                blind_yaw_radius_m=0.0,
                replan_interval_s=10.0,
            )
        )
        follower.start(
            np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [1.2, 0.0, 0.0]]),
            np.array([1.2, 0.0]),
        )
        follower.update(0.0, np.zeros(3))
        self.assertEqual(follower.current_idx, 1)

        follower.replace_path(
            np.array([[0.0, 0.0, 0.0], [0.7, 0.2, 0.0], [1.4, 0.2, 0.0]])
        )
        self.assertEqual(follower.current_idx, 0)
        output = follower.update(0.0, np.zeros(3))
        self.assertEqual(follower.current_idx, 1)
        np.testing.assert_allclose(output.target_local_xy, [0.7, 0.2])

    def test_safe_distance_truncates_path(self):
        path = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
        )
        truncated = truncate_trajectory_for_safety(path, 0.5)
        np.testing.assert_allclose(truncated[-1, :2], [1.0, 0.0])

    def test_yaw_bias_is_added_after_uni_lavira_clamp(self):
        follower = LocalTrajectoryFollower(
            LocalFollowerConfig(
                target_speed_m_s=0.3,
                max_yaw_speed_rad_s=0.5,
                goal_tolerance_m=0.1,
                blind_yaw_radius_m=0.0,
                yaw_bias_rad_s=0.1,
                replan_interval_s=10.0,
            )
        )
        follower.start(
            np.array([[0.0, 0.0, 0.0], [0.01, 1.0, 0.0], [0.02, 2.0, 0.0]]),
            np.array([0.02, 2.0]),
        )

        output = follower.update(0.0, np.zeros(3))
        self.assertAlmostEqual(output.command[2], 0.6)


if __name__ == "__main__":
    unittest.main()
