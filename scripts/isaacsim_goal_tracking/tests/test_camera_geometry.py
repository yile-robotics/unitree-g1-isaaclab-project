from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from goal_tracking.camera import (  # noqa: E402
    FOUR_VIEW_DIRECTIONS,
    FOUR_VIEW_YAW_DEG,
    _quat_wxyz_from_yaw_down_tilt_deg,
    get_four_view_local_poses,
)
from goal_tracking.frame_bundle import (  # noqa: E402
    _R_WORLD_CONVENTION_FROM_ROS,
    _pose_matrix,
)


def _rotation_from_wxyz(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    return _pose_matrix(np.zeros(3), np.asarray(quaternion))[:3, :3]


class FourViewCameraGeometryTest(unittest.TestCase):
    def test_four_optical_centers_match_navigation_axes(self) -> None:
        radius = 0.085
        height = 0.56
        poses = get_four_view_local_poses(height, radius, 12.0)
        expected_positions = {
            "forward": np.array([radius, 0.0, height]),
            "left": np.array([0.0, radius, height]),
            "behind": np.array([-radius, 0.0, height]),
            "right": np.array([0.0, -radius, height]),
        }

        self.assertEqual(tuple(poses), FOUR_VIEW_DIRECTIONS)
        for direction in FOUR_VIEW_DIRECTIONS:
            np.testing.assert_allclose(
                poses[direction][0], expected_positions[direction], atol=1.0e-12
            )

    def test_four_optical_axes_have_expected_yaw_and_down_tilt(self) -> None:
        tilt_deg = 12.0
        horizontal = math.cos(math.radians(tilt_deg))
        vertical = -math.sin(math.radians(tilt_deg))
        expected_axes = {
            "forward": np.array([horizontal, 0.0, vertical]),
            "left": np.array([0.0, horizontal, vertical]),
            "behind": np.array([-horizontal, 0.0, vertical]),
            "right": np.array([0.0, -horizontal, vertical]),
        }

        self.assertEqual(
            FOUR_VIEW_DIRECTIONS,
            ("forward", "left", "behind", "right"),
        )
        for direction in FOUR_VIEW_DIRECTIONS:
            quaternion = _quat_wxyz_from_yaw_down_tilt_deg(
                FOUR_VIEW_YAW_DEG[direction], tilt_deg
            )
            self.assertAlmostEqual(np.linalg.norm(quaternion), 1.0, places=12)
            optical_axis = _rotation_from_wxyz(quaternion)[:, 0]
            np.testing.assert_allclose(
                optical_axis,
                expected_axes[direction],
                atol=1.0e-12,
            )

    def test_pose_matrix_uses_wxyz_and_maps_local_points_to_world(self) -> None:
        half_angle = math.pi / 4.0
        T_world_local = _pose_matrix(
            np.array([1.0, 2.0, 3.0]),
            np.array([math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]),
        )

        local_point = np.array([1.0, 0.0, 0.0, 1.0])
        world_point = T_world_local @ local_point
        np.testing.assert_allclose(world_point, [1.0, 3.0, 3.0, 1.0], atol=1.0e-12)
        self.assertAlmostEqual(np.linalg.det(T_world_local[:3, :3]), 1.0, places=12)

    def test_torso_yaw_composes_with_camera_mount_and_ros_optical_frame(self) -> None:
        torso_yaw_deg = 135.0
        torso_quaternion = _quat_wxyz_from_yaw_down_tilt_deg(torso_yaw_deg, 0.0)
        mount_yaw_deg = FOUR_VIEW_YAW_DEG["forward"]
        mount_quaternion = _quat_wxyz_from_yaw_down_tilt_deg(
            mount_yaw_deg, 12.0
        )
        T_world_torso = _pose_matrix(np.zeros(3), np.asarray(torso_quaternion))
        T_torso_camera = _pose_matrix(np.zeros(3), np.asarray(mount_quaternion))
        R_world_camera = (T_world_torso @ T_torso_camera)[:3, :3]
        R_world_camera_ros = R_world_camera @ _R_WORLD_CONVENTION_FROM_ROS

        expected_yaw_deg = torso_yaw_deg + mount_yaw_deg
        expected_forward = np.array(
            [
                math.cos(math.radians(expected_yaw_deg)) * math.cos(math.radians(12.0)),
                math.sin(math.radians(expected_yaw_deg)) * math.cos(math.radians(12.0)),
                -math.sin(math.radians(12.0)),
            ]
        )
        # ROS optical +Z is the camera's forward ray.
        np.testing.assert_allclose(
            R_world_camera_ros[:, 2],
            expected_forward,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
