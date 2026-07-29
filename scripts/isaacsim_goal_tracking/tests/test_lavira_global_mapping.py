from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from goal_tracking.lavira_global_mapping import (  # noqa: E402
    LaViRAGlobalMapConfig,
    LaViRAGlobalMapState,
    save_lavira_global_map_debug,
)
from goal_tracking.fmm_planner import FMMPlannerConfig  # noqa: E402
from goal_tracking.lavira_episode import (  # noqa: E402
    RuntimeWaypointRecord,
    build_global_replanned_backtrack_execution_request,
)
from goal_tracking.navigation_mapping import (  # noqa: E402
    NavigationGridMap,
    NavigationMapConfig,
    world_xy_to_grid_cell,
)


def _navigation_config() -> NavigationMapConfig:
    return NavigationMapConfig(
        resolution_m=0.1,
        size_m=4.0,
        depth_stride=1,
        depth_min_m=0.1,
        depth_max_m=5.0,
        nominal_base_height_m=0.8,
        floor_search_half_range_m=0.2,
        obstacle_min_height_m=0.1,
        obstacle_max_height_m=1.6,
        robot_radius_m=0.1,
        start_clearance_m=0.1,
        target_retreat_step_m=0.1,
        target_snap_max_m=1.0,
    )


def _observation(
    robot_world_xy: tuple[float, float],
    *,
    bundle_id: int,
    observed_world_xy: tuple[tuple[float, float], ...],
    occupied_world_xy: tuple[tuple[float, float], ...] = (),
) -> NavigationGridMap:
    config = _navigation_config()
    shape = (40, 40)
    robot = np.asarray(robot_world_xy, dtype=np.float64)
    origin = robot - config.size_m * 0.5
    observed = np.zeros(shape, dtype=bool)
    occupied = np.zeros(shape, dtype=bool)
    obstacle_hits = np.zeros(shape, dtype=np.uint16)
    for point in observed_world_xy:
        cell = world_xy_to_grid_cell(
            np.asarray(point), origin, config.resolution_m, shape
        )
        if cell is not None:
            observed[cell] = True
    for point in occupied_world_xy:
        cell = world_xy_to_grid_cell(
            np.asarray(point), origin, config.resolution_m, shape
        )
        if cell is not None:
            observed[cell] = True
            occupied[cell] = True
            obstacle_hits[cell] = 1
    robot_cell = world_xy_to_grid_cell(
        robot, origin, config.resolution_m, shape
    )
    observed[robot_cell] = True
    return NavigationGridMap(
        bundle_id=bundle_id,
        sim_step=bundle_id * 10,
        config=config,
        resolution_m=config.resolution_m,
        size_m=config.size_m,
        origin_world_xy=origin,
        floor_z_world_m=0.0,
        floor_estimation_method="test",
        floor_candidate_count=1,
        sampled_point_counts={"forward": 1, "left": 0, "behind": 0, "right": 0},
        observed=observed,
        free=observed & ~occupied,
        occupied=occupied,
        inflated_obstacles=occupied.copy(),
        traversable=observed & ~occupied,
        obstacle_hits=obstacle_hits,
        robot_world_xy=robot,
        robot_cell_rc=robot_cell,
        raw_target_world_xy=robot.copy(),
        raw_target_cell_rc=robot_cell,
        safe_target_world_xy=robot.copy(),
        safe_target_cell_rc=robot_cell,
        target_selection_strategy="test",
        target_retreat_m=0.0,
        target_snap_distance_m=0.0,
    )


class LaViRAGlobalMappingTest(unittest.TestCase):
    def test_spawn_origin_is_fixed_and_observations_accumulate(self) -> None:
        state = LaViRAGlobalMapState(
            _navigation_config(),
            LaViRAGlobalMapConfig(
                global_downscaling=2,
                center_reset_steps=25,
            ),
        )
        first = _observation(
            (0.0, 0.0),
            bundle_id=1,
            observed_world_xy=((0.0, 0.0), (0.5, 0.0)),
            occupied_world_xy=((0.5, 0.0),),
        )
        state.integrate_grid_map(first)
        np.testing.assert_allclose(state.origin_world_xy, [-2.0, -2.0])
        first_origin = state.origin_world_xy.copy()

        second = _observation(
            (0.8, 0.0),
            bundle_id=2,
            observed_world_xy=((0.0, 0.0), (0.8, 0.0), (1.0, 0.0)),
        )
        state.integrate_grid_map(second)
        np.testing.assert_allclose(state.origin_world_xy, first_origin)
        old_obstacle = world_xy_to_grid_cell(
            np.array([0.5, 0.0]),
            state.origin_world_xy,
            state.navigation_config.resolution_m,
            state.full_map.shape[1:],
        )
        new_observation = world_xy_to_grid_cell(
            np.array([1.0, 0.0]),
            state.origin_world_xy,
            state.navigation_config.resolution_m,
            state.full_map.shape[1:],
        )
        self.assertEqual(state.full_map[0, old_obstacle[0], old_obstacle[1]], 1)
        self.assertEqual(
            state.full_map[1, new_observation[0], new_observation[1]], 1
        )
        self.assertGreater(np.count_nonzero(state.full_map[3]), 0)
        self.assertEqual(state.update_count, 2)
        self.assertEqual(state.local_map.shape, (4, 20, 20))

    def test_historical_target_uses_accumulated_global_traversability(self) -> None:
        state = LaViRAGlobalMapState(
            _navigation_config(),
            LaViRAGlobalMapConfig(unknown_space_policy="blocked"),
        )
        corridor = tuple((value, 0.0) for value in np.arange(0.0, 1.01, 0.1))
        state.integrate_grid_map(
            _observation(
                (0.0, 0.0),
                bundle_id=1,
                observed_world_xy=corridor,
            )
        )
        state.integrate_grid_map(
            _observation(
                (1.0, 0.0),
                bundle_id=2,
                observed_world_xy=corridor,
            )
        )

        planning_map = state.build_navigation_grid_map(
            historical_target_world_xy=np.array([0.0, 0.0])
        )
        self.assertTrue(planning_map.traversable[planning_map.robot_cell_rc])
        self.assertIsNotNone(planning_map.safe_target_cell_rc)
        self.assertTrue(
            planning_map.target_selection_strategy.startswith(
                "global_historical_waypoint"
            )
        )
        np.testing.assert_allclose(planning_map.origin_world_xy, [-2.0, -2.0])

    def test_stable_world_goal_and_collision_mask_share_global_fmm_grid(self) -> None:
        state = LaViRAGlobalMapState(
            _navigation_config(),
            LaViRAGlobalMapConfig(unknown_space_policy="blocked"),
        )
        corridor = tuple((value, 0.0) for value in np.arange(0.0, 1.01, 0.1))
        state.integrate_grid_map(
            _observation(
                (0.0, 0.0),
                bundle_id=1,
                observed_world_xy=corridor,
            )
        )
        state.integrate_grid_map(
            _observation(
                (0.2, 0.0),
                bundle_id=2,
                observed_world_xy=corridor,
            )
        )
        collision_xy = np.array([0.6, 0.0])
        added = state.mark_collision_world_xy(collision_xy, radius_m=0.0)
        self.assertEqual(added, 1)

        planning_map = state.build_navigation_grid_map(
            stable_target_world_xy=np.array([1.0, 0.0])
        )
        collision_cell = world_xy_to_grid_cell(
            collision_xy,
            state.origin_world_xy,
            state.navigation_config.resolution_m,
            state.collision_map.shape,
        )
        self.assertFalse(planning_map.traversable[collision_cell])
        self.assertTrue(
            planning_map.target_selection_strategy.startswith("global_stable_")
        )
        self.assertEqual(
            planning_map.raw_target_world_xy.tolist(),
            [1.0, 0.0],
        )

        removed = state.clear_collision_world_xy(collision_xy, radius_m=0.0)
        self.assertEqual(removed, 1)
        self.assertEqual(np.count_nonzero(state.collision_map), 0)

    def test_manual_origin_and_debug_artifacts(self) -> None:
        state = LaViRAGlobalMapState(
            _navigation_config(),
            LaViRAGlobalMapConfig(
                origin_mode="manual",
                manual_origin_world_x_m=-1.0,
                manual_origin_world_y_m=-1.5,
            ),
        )
        state.integrate_grid_map(
            _observation(
                (0.0, 0.0),
                bundle_id=1,
                observed_world_xy=((0.0, 0.0),),
            )
        )
        np.testing.assert_allclose(state.origin_world_xy, [-1.0, -1.5])
        planning_map = state.build_navigation_grid_map(
            historical_target_world_xy=np.array([0.0, 0.0])
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            files = save_lavira_global_map_debug(
                Path(temp_dir), state, planning_map
            )
            self.assertEqual(
                set(files),
                {"metadata_json", "arrays_npz", "visualization_png"},
            )
            arrays = np.load(Path(temp_dir) / files["arrays_npz"])
            np.testing.assert_array_equal(arrays["full_map"], state.full_map)
            np.testing.assert_array_equal(
                arrays["collision_map"], state.collision_map
            )

    def test_qwen_waypoint_backtrack_replans_on_same_global_map(self) -> None:
        state = LaViRAGlobalMapState(
            _navigation_config(),
            LaViRAGlobalMapConfig(unknown_space_policy="blocked"),
        )
        corridor = tuple((value, 0.0) for value in np.arange(0.0, 1.01, 0.1))
        state.integrate_grid_map(
            _observation(
                (0.0, 0.0),
                bundle_id=1,
                observed_world_xy=corridor,
            )
        )
        state.integrate_grid_map(
            _observation(
                (1.0, 0.0),
                bundle_id=2,
                observed_world_xy=corridor,
            )
        )
        decision_pose = np.eye(4, dtype=np.float64)
        decision_pose[:3, 3] = (0.0, 0.0, 0.8)
        record = RuntimeWaypointRecord(
            waypoint_id=0,
            decision_index=0,
            decision_step=10,
            decision_world_pose=decision_pose,
            direction="forward",
            target="test",
            bbox_2d=(0.0, 0.0, 1.0, 1.0),
            progress_analysis="",
            reasoning="",
            init_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            dir_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            projected_target_world=None,
            safe_target_world_xy=np.array([1.0, 0.0]),
            fmm_waypoints_world_xy=np.array([[0.0, 0.0], [1.0, 0.0]]),
            execution_status="arrived",
            arrival_step=20,
        )

        request, planning_map, fmm_plan = (
            build_global_replanned_backtrack_execution_request(
                [record],
                state,
                target_waypoint_id=0,
                decision_index=2,
                fmm_planner_config=FMMPlannerConfig(
                    step_size_cells=2,
                    waypoint_spacing_m=0.2,
                ),
            )
        )
        self.assertEqual(request.fmm_plan.target_waypoint_id, 0)
        self.assertEqual(
            request.fmm_plan.execution_source,
            "lavira_global_replanned_backtrack_waypoint_000_decision_002",
        )
        np.testing.assert_allclose(fmm_plan.start_world_xy, [1.0, 0.0])
        np.testing.assert_allclose(
            planning_map.raw_target_world_xy, [0.0, 0.0]
        )
        self.assertGreater(fmm_plan.path_length_m, 0.0)

    def test_robot_outside_fixed_bounds_has_actionable_error(self) -> None:
        state = LaViRAGlobalMapState(
            _navigation_config(),
            LaViRAGlobalMapConfig(),
        )
        state.integrate_grid_map(
            _observation(
                (0.0, 0.0),
                bundle_id=1,
                observed_world_xy=((0.0, 0.0),),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "Increase --nav_map_size_m"):
            state.integrate_grid_map(
                _observation(
                    (3.0, 0.0),
                    bundle_id=2,
                    observed_world_xy=((3.0, 0.0),),
                )
            )


if __name__ == "__main__":
    unittest.main()
