from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.map_progress import (  # noqa: E402
    SparseEpisodeExplorationMap,
    SparseMapConfig,
)
from unified_vln.odometry import Pose2D  # noqa: E402
from unified_vln.types import ViewFrame  # noqa: E402


def _frame(
    *,
    frame_id: int = 1,
    depth_m: float = 2.0,
    invalid: bool = False,
) -> ViewFrame:
    depth = np.full((24, 32), depth_m, dtype=np.float32)
    if invalid:
        depth.fill(np.nan)
        depth[0, 0] = np.inf
    return ViewFrame(
        direction="forward",
        frame_id=frame_id,
        sim_step=frame_id,
        timestamp=float(frame_id) * 0.1,
        rgb=np.zeros((24, 32, 3), dtype=np.uint8),
        depth_m=depth,
        K=np.array(
            [[24.0, 0.0, 15.5], [0.0, 24.0, 11.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )


def _config() -> SparseMapConfig:
    return SparseMapConfig(
        resolution_m=0.05,
        depth_stride=2,
        depth_min_m=0.1,
        depth_max_m=5.0,
        camera_offset_x_m=0.0,
        camera_offset_y_m=0.0,
        camera_offset_z_m=0.2,
        camera_yaw_rad=0.0,
        camera_down_tilt_rad=0.0,
        nominal_base_height_m=0.8,
        floor_z_world_m=0.0,
        obstacle_min_height_m=0.1,
        obstacle_max_height_m=1.6,
        robot_radius_m=0.10,
        start_clearance_m=0.05,
    )


class SparseEpisodeExplorationMapTest(unittest.TestCase):
    def test_grid_is_unbounded_and_supports_negative_coordinates(self):
        mapping = SparseEpisodeExplorationMap(
            _config(), pose_frame_id="isaac_world"
        )
        self.assertEqual(mapping.world_xy_to_cell(-0.001, -0.051), (-1, -2))
        self.assertEqual(mapping.world_xy_to_cell(1000.0, -1000.0), (20000, -20000))

    def test_repeated_observation_does_not_double_count_explored_cells(self):
        mapping = SparseEpisodeExplorationMap(
            _config(), pose_frame_id="isaac_world"
        )
        pose = Pose2D(0.0, 0.0, 0.0, 0.0)
        first = mapping.integrate(_frame(frame_id=1), pose)
        first_count = mapping.explored_cells
        second = mapping.integrate(_frame(frame_id=2), pose)

        self.assertGreater(first.new_explored_cells, 0)
        self.assertEqual(second.new_explored_cells, 0)
        self.assertEqual(mapping.explored_cells, first_count)
        self.assertGreater(mapping.snapshot().traversable_cells, 0)

    def test_motion_into_new_world_cells_increases_window_gain(self):
        mapping = SparseEpisodeExplorationMap(
            _config(), pose_frame_id="isaac_world"
        )
        mapping.integrate(_frame(frame_id=1), Pose2D(0.0, 0.0, 0.0, 0.0))
        baseline = mapping.explored_cells
        mapping.integrate(_frame(frame_id=2), Pose2D(0.75, 0.0, 0.0, 1.0))
        snapshot = mapping.snapshot(explored_before=baseline)

        self.assertGreater(snapshot.new_explored_cells, 0)
        self.assertEqual(snapshot.explored_cells, mapping.explored_cells)
        self.assertEqual(snapshot.pose_frame_id, "isaac_world")
        self.assertEqual(snapshot.frame_epoch, 0)
        self.assertAlmostEqual(snapshot.resolution_m, 0.05)

    def test_rotation_observes_a_new_sector_without_changing_map_origin(self):
        mapping = SparseEpisodeExplorationMap(
            _config(), pose_frame_id="isaac_world"
        )
        mapping.integrate(_frame(frame_id=1), Pose2D(0.0, 0.0, 0.0, 0.0))
        baseline = mapping.explored_cells
        mapping.integrate(
            _frame(frame_id=2), Pose2D(0.0, 0.0, math.pi / 2.0, 1.0)
        )
        self.assertGreater(mapping.explored_cells - baseline, 0)

    def test_invalid_depth_is_ignored_without_cast_or_projection_failure(self):
        mapping = SparseEpisodeExplorationMap(
            _config(), pose_frame_id="isaac_world"
        )
        result = mapping.integrate(
            _frame(frame_id=1, invalid=True), Pose2D(0.0, 0.0, 0.0, 0.0)
        )
        self.assertEqual(result.valid_depth_samples, 0)
        self.assertGreater(mapping.explored_cells, 0)
        first_count = mapping.explored_cells
        mapping.integrate(
            _frame(frame_id=2, invalid=True), Pose2D(0.0, 0.0, 0.0, 1.0)
        )
        self.assertEqual(mapping.explored_cells, first_count)

    def test_debug_render_uses_dynamic_extent_without_changing_sparse_counts(self):
        mapping = SparseEpisodeExplorationMap(
            _config(), pose_frame_id="isaac_world"
        )
        mapping.integrate(_frame(frame_id=1), Pose2D(-1.0, 2.0, 0.0, 0.0))
        snapshot = mapping.snapshot(explored_before=0)
        before = mapping.explored_cells
        with tempfile.TemporaryDirectory() as temporary_dir:
            files = mapping.save_debug(
                Path(temporary_dir) / "map_progress_0000", snapshot
            )
            self.assertTrue(Path(files["metadata"]).is_file())
            self.assertTrue(Path(files["visualization"]).is_file())
        self.assertEqual(mapping.explored_cells, before)


if __name__ == "__main__":
    unittest.main()
