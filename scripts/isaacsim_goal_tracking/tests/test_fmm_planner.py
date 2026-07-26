from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from goal_tracking.fmm_planner import (  # noqa: E402
    FMMPlannerConfig,
    FMMPlanningError,
    build_fmm_plan,
    compute_lavira_fmm_distance,
    lavira_local_ring_mask,
    save_fmm_plan_debug,
)
from goal_tracking.navigation_mapping import (  # noqa: E402
    NavigationGridMap,
    NavigationMapConfig,
    grid_cell_to_world_xy,
)


def _grid_map(
    traversable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    resolution_m: float = 0.05,
) -> NavigationGridMap:
    traversable = np.asarray(traversable, dtype=bool)
    origin = np.zeros(2, dtype=np.float64)
    robot_world = grid_cell_to_world_xy(start, origin, resolution_m)
    goal_world = grid_cell_to_world_xy(goal, origin, resolution_m)
    occupied = ~traversable
    return NavigationGridMap(
        bundle_id=4,
        sim_step=5,
        config=NavigationMapConfig(
            resolution_m=resolution_m,
            size_m=traversable.shape[0] * resolution_m,
        ),
        resolution_m=resolution_m,
        size_m=traversable.shape[0] * resolution_m,
        origin_world_xy=origin,
        floor_z_world_m=0.0,
        floor_estimation_method="test",
        floor_candidate_count=1,
        sampled_point_counts={
            "forward": 1,
            "left": 1,
            "behind": 1,
            "right": 1,
        },
        observed=np.ones_like(traversable),
        free=traversable.copy(),
        occupied=occupied.copy(),
        inflated_obstacles=occupied.copy(),
        traversable=traversable.copy(),
        obstacle_hits=occupied.astype(np.uint16),
        robot_world_xy=robot_world,
        robot_cell_rc=start,
        raw_target_world_xy=goal_world,
        raw_target_cell_rc=goal,
        safe_target_world_xy=goal_world,
        safe_target_cell_rc=goal,
        target_selection_strategy="test",
        target_retreat_m=0.0,
        target_snap_distance_m=0.0,
    )


class FMMPlannerTest(unittest.TestCase):
    def test_lavira_distance_and_five_cell_ring(self) -> None:
        traversable = np.ones((31, 31), dtype=bool)
        goal = (15, 25)
        distance = compute_lavira_fmm_distance(traversable, goal)

        self.assertEqual(distance.shape, traversable.shape)
        self.assertAlmostEqual(float(distance[goal]), 0.0, places=6)
        self.assertGreater(float(distance[15, 5]), float(distance[15, 15]))
        ring = lavira_local_ring_mask(5)
        self.assertEqual(ring.shape, (11, 11))
        self.assertTrue(ring[5, 5])
        self.assertGreater(np.count_nonzero(ring), 1)

    def test_full_path_detours_through_gap_without_crossing_obstacles(self) -> None:
        traversable = np.ones((45, 45), dtype=bool)
        traversable[:, 22] = False
        traversable[20:25, 22] = True
        start = (10, 5)
        goal = (10, 39)
        grid_map = _grid_map(traversable, start, goal)

        plan = build_fmm_plan(grid_map, FMMPlannerConfig())

        self.assertEqual(tuple(plan.path_cells_rc[0]), start)
        self.assertEqual(tuple(plan.path_cells_rc[-1]), goal)
        self.assertTrue(
            all(traversable[tuple(cell)] for cell in plan.path_cells_rc)
        )
        wall_crossings = plan.path_cells_rc[plan.path_cells_rc[:, 1] == 22]
        self.assertTrue(len(wall_crossings))
        self.assertTrue(
            all(20 <= int(cell[0]) < 25 for cell in wall_crossings)
        )
        self.assertGreater(plan.path_length_m, 34 * grid_map.resolution_m)
        self.assertGreater(plan.waypoint_cells_rc.shape[0], 2)

    def test_unreachable_goal_is_rejected(self) -> None:
        traversable = np.ones((31, 31), dtype=bool)
        traversable[:, 15] = False
        grid_map = _grid_map(traversable, (15, 5), (15, 25))
        with self.assertRaisesRegex(FMMPlanningError, "unreachable"):
            build_fmm_plan(grid_map, FMMPlannerConfig())

    def test_debug_output_contains_plan_distance_and_visualization(self) -> None:
        traversable = np.ones((31, 31), dtype=bool)
        grid_map = _grid_map(traversable, (15, 5), (15, 25))
        plan = build_fmm_plan(grid_map, FMMPlannerConfig())
        with tempfile.TemporaryDirectory() as temp_dir:
            files = save_fmm_plan_debug(Path(temp_dir), grid_map, plan)
            self.assertEqual(
                set(files), {"plan_json", "distance_npy", "visualization_png"}
            )
            for filename in files.values():
                self.assertTrue((Path(temp_dir) / filename).is_file())
            saved_distance = np.load(Path(temp_dir) / files["distance_npy"])
            self.assertEqual(saved_distance.shape, traversable.shape)


if __name__ == "__main__":
    unittest.main()
