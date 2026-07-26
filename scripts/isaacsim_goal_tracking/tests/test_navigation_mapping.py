from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from goal_tracking.camera import FOUR_VIEW_DIRECTIONS  # noqa: E402
from goal_tracking.frame_bundle import CameraFrame, FrameBundle  # noqa: E402
from goal_tracking.navigation_mapping import (  # noqa: E402
    NavigationMapConfig,
    build_navigation_grid_map,
    build_navigation_grid_map_for_world_goal,
    grid_cell_to_world_xy,
    save_navigation_map_debug,
    world_xy_to_grid_cell,
)
from goal_tracking.target_projection import TargetProjection  # noqa: E402


def _camera_transform() -> np.ndarray:
    # ROS optical axes expressed in world: +X camera right -> world -Y,
    # +Y camera down -> world -Z, +Z camera forward -> world +X.
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    transform[:3, 3] = (0.0, 0.0, 1.0)
    return transform


def _bundle_and_projection() -> tuple[FrameBundle, TargetProjection]:
    K = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    T_world_camera = _camera_transform()
    T_world_base = np.eye(4, dtype=np.float64)
    T_world_base[:3, 3] = (0.0, 0.0, 0.8)
    T_base_camera = np.linalg.inv(T_world_base) @ T_world_camera
    views = {}
    for frame_id, direction in enumerate(FOUR_VIEW_DIRECTIONS, start=1):
        depth = np.full((5, 5), np.inf, dtype=np.float32)
        depth[2, 2] = 1.5  # endpoint world=(1.5, 0.0, 1.0): obstacle
        depth[3, 2] = 1.0  # endpoint world=(1.0, 0.0, 0.0): floor
        views[direction] = CameraFrame(
            camera_id=f"camera_{direction}",
            direction=direction,
            sensor_frame_id=frame_id,
            sim_step=5,
            timestamp=0.1,
            rgb=np.zeros((5, 5, 3), dtype=np.uint8),
            depth_z_m=depth,
            K=K.copy(),
            T_world_camera_ros=T_world_camera.copy(),
            T_base_camera=T_base_camera.copy(),
        )
    bundle = FrameBundle(
        bundle_id=7,
        env_index=0,
        sim_step=5,
        timestamp=0.1,
        T_world_base=T_world_base,
        views=views,
    )
    projection = TargetProjection(
        observation_id="robot_01_map_test_decision_000",
        bundle_id=7,
        sim_step=5,
        action="NAVIGATE",
        direction="forward",
        target="wall target",
        camera_id="camera_forward",
        sensor_frame_id=1,
        bbox_response=(1.0, 1.0, 3.0, 3.0),
        bbox_clipped=(1, 1, 3, 3),
        selected_pixel_uv=(2, 2),
        depth_window_size=3,
        depth_window_xyxy_exclusive=(1, 1, 4, 4),
        valid_depth_count=1,
        depth_window_pixel_count=9,
        valid_depth_fraction=1.0 / 9.0,
        valid_depth_min_m=1.5,
        selected_depth_median_m=1.5,
        valid_depth_max_m=1.5,
        point_camera_ros_m=np.array([0.0, 0.0, 1.5]),
        point_base_m=np.array([1.5, 0.0, 0.2]),
        point_world_m=np.array([1.5, 0.0, 1.0]),
        horizontal_distance_base_m=1.5,
        bearing_base_rad=0.0,
        K=K.copy(),
        T_base_camera=T_base_camera.copy(),
        T_world_camera_ros=T_world_camera.copy(),
    )
    return bundle, projection


def _config() -> NavigationMapConfig:
    return NavigationMapConfig(
        resolution_m=0.1,
        size_m=6.0,
        depth_stride=1,
        depth_min_m=0.1,
        depth_max_m=5.0,
        nominal_base_height_m=0.8,
        floor_search_half_range_m=0.2,
        obstacle_min_height_m=0.1,
        obstacle_max_height_m=1.6,
        robot_radius_m=0.2,
        start_clearance_m=0.1,
        target_retreat_step_m=0.1,
        target_snap_max_m=1.0,
    )


class NavigationMappingTest(unittest.TestCase):
    def test_world_grid_round_trip_uses_explicit_xy_axes(self) -> None:
        origin = np.array([-3.0, -3.0])
        cell = world_xy_to_grid_cell(
            np.array([1.24, -0.76]), origin, 0.1, (60, 60)
        )
        self.assertEqual(cell, (22, 42))
        center = grid_cell_to_world_xy(cell, origin, 0.1)
        np.testing.assert_allclose(center, [1.25, -0.75])

    def test_depth_map_marks_obstacle_inflates_and_retreats_target(self) -> None:
        bundle, projection = _bundle_and_projection()
        grid_map = build_navigation_grid_map(bundle, projection, _config())

        obstacle_cell = world_xy_to_grid_cell(
            np.array([1.5, 0.0]),
            grid_map.origin_world_xy,
            grid_map.resolution_m,
            grid_map.shape,
        )
        self.assertIsNotNone(obstacle_cell)
        self.assertTrue(grid_map.occupied[obstacle_cell])
        self.assertTrue(grid_map.inflated_obstacles[obstacle_cell])
        self.assertFalse(grid_map.traversable[obstacle_cell])
        self.assertEqual(grid_map.raw_target_cell_rc, obstacle_cell)
        self.assertEqual(grid_map.target_selection_strategy, "lavira_depth_retreat")
        self.assertIsNotNone(grid_map.safe_target_cell_rc)
        self.assertTrue(grid_map.traversable[grid_map.safe_target_cell_rc])
        self.assertGreaterEqual(grid_map.target_retreat_m, 0.2)
        self.assertAlmostEqual(grid_map.floor_z_world_m, 0.0)

        # Unknown corners are deliberately blocked for G1 safety.
        self.assertFalse(grid_map.observed[0, 0])
        self.assertFalse(grid_map.traversable[0, 0])

    def test_bundle_mismatch_is_rejected(self) -> None:
        bundle, projection = _bundle_and_projection()
        object.__setattr__(projection, "bundle_id", 8)
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_navigation_grid_map(bundle, projection, _config())

    def test_historical_world_goal_does_not_use_bbox_retreat(self) -> None:
        bundle, _ = _bundle_and_projection()
        grid_map = build_navigation_grid_map_for_world_goal(
            bundle,
            np.array([0.0, 0.0], dtype=np.float64),
            _config(),
        )

        self.assertEqual(
            grid_map.target_selection_strategy,
            "historical_waypoint_traversable",
        )
        self.assertEqual(grid_map.target_retreat_m, 0.0)
        self.assertEqual(grid_map.safe_target_cell_rc, grid_map.robot_cell_rc)

    def test_debug_output_contains_arrays_metadata_and_visualization(self) -> None:
        bundle, projection = _bundle_and_projection()
        grid_map = build_navigation_grid_map(bundle, projection, _config())
        with tempfile.TemporaryDirectory() as temp_dir:
            files = save_navigation_map_debug(Path(temp_dir), grid_map)
            self.assertEqual(
                set(files), {"metadata_json", "arrays_npz", "visualization_png"}
            )
            for filename in files.values():
                self.assertTrue((Path(temp_dir) / filename).is_file())
            arrays = np.load(Path(temp_dir) / files["arrays_npz"])
            np.testing.assert_array_equal(arrays["traversable"], grid_map.traversable)


if __name__ == "__main__":
    unittest.main()
