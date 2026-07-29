from __future__ import annotations

"""LaViRA-compatible cumulative global map for Isaac Sim four-view RGB-D.

The existing Isaac mapper produces one robot-centred geometric observation from
the current four-camera ``FrameBundle``.  This module keeps that well-tested
projection code, shifts each observation into one fixed world grid, and fuses it
with ``max`` exactly like LaViRA's full-map update.

Channels intentionally follow the first four LaViRA map channels:

0. obstacle
1. explored
2. current robot location
3. past robot locations

Semantic category channels are not fabricated here: the current Isaac pipeline
does not run LaViRA's Grounded-SAM semantic mapper.  The fixed global geometry,
full/local windows, coordinate transforms, history persistence and FMM input are
nevertheless shared by NAVIGATE, BACKTRACK and STOP.  A separate episode-local
``collision_map`` mirrors LaViRA's online collision mask without corrupting the
camera-derived obstacle channel.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from .frame_bundle import FrameBundle
from .navigation_mapping import (
    NavigationGridMap,
    NavigationMapConfig,
    _binary_morphology,
    _component_connected_to_robot,
    _fill_disk,
    _select_historical_waypoint_target,
    _select_safe_target,
    build_navigation_grid_map_for_world_goal,
    cv2_ellipse_kernel,
    navigation_map_visualization,
    world_xy_to_grid_cell,
)
from .target_projection import TargetProjection


LAVIRA_GEOMETRIC_CHANNELS = (
    "obstacle",
    "explored",
    "current_location",
    "past_locations",
)


@dataclass(frozen=True)
class LaViRAGlobalMapConfig:
    """Runtime choices that are independent from depth/map geometry settings."""

    origin_mode: str = "spawn_center"
    manual_origin_world_x_m: float | None = None
    manual_origin_world_y_m: float | None = None
    global_downscaling: int = 2
    center_reset_steps: int = 25
    unknown_space_policy: str = "blocked"

    def validated(
        self, navigation_config: NavigationMapConfig
    ) -> "LaViRAGlobalMapConfig":
        navigation_config = navigation_config.validated()
        if self.origin_mode not in {"spawn_center", "manual"}:
            raise ValueError(
                "Global map origin mode must be 'spawn_center' or 'manual'."
            )
        manual_values = (
            self.manual_origin_world_x_m,
            self.manual_origin_world_y_m,
        )
        if self.origin_mode == "manual":
            if any(value is None for value in manual_values):
                raise ValueError(
                    "Manual global map origin requires both world X and world Y."
                )
            if not all(np.isfinite(float(value)) for value in manual_values):
                raise ValueError("Manual global map origin must be finite.")
        if self.global_downscaling <= 0:
            raise ValueError("Global map downscaling must be positive.")
        cells = int(
            round(navigation_config.size_m / navigation_config.resolution_m)
        )
        if cells % int(self.global_downscaling) != 0:
            raise ValueError(
                "Global map cell count must be divisible by global_downscaling."
            )
        if self.center_reset_steps <= 0:
            raise ValueError("Global map center reset steps must be positive.")
        if self.unknown_space_policy not in {"blocked", "lavira"}:
            raise ValueError(
                "Global unknown-space policy must be 'blocked' or 'lavira'."
            )
        return self


def lavira_global_map_config_from_args(args_cli) -> LaViRAGlobalMapConfig:
    return LaViRAGlobalMapConfig(
        origin_mode=str(
            getattr(args_cli, "nav_global_origin_mode", "spawn_center")
        ),
        manual_origin_world_x_m=getattr(
            args_cli, "nav_global_origin_world_x_m", None
        ),
        manual_origin_world_y_m=getattr(
            args_cli, "nav_global_origin_world_y_m", None
        ),
        global_downscaling=int(
            getattr(args_cli, "nav_global_downscaling", 2)
        ),
        center_reset_steps=int(
            getattr(args_cli, "nav_global_center_reset_steps", 25)
        ),
        unknown_space_policy=str(
            getattr(args_cli, "nav_global_unknown_space_policy", "blocked")
        ),
    )


class LaViRAGlobalMapState:
    """Episode-persistent full/local map state in one fixed world frame."""

    def __init__(
        self,
        navigation_config: NavigationMapConfig,
        global_config: LaViRAGlobalMapConfig,
    ):
        self.navigation_config = navigation_config.validated()
        self.global_config = global_config.validated(self.navigation_config)
        self.cells = int(
            round(
                self.navigation_config.size_m
                / self.navigation_config.resolution_m
            )
        )
        self.local_cells = self.cells // self.global_config.global_downscaling
        self.origin_world_xy: np.ndarray | None = None
        self.initial_robot_world_xy: np.ndarray | None = None
        self.full_map = np.zeros(
            (len(LAVIRA_GEOMETRIC_CHANNELS), self.cells, self.cells),
            dtype=np.uint8,
        )
        self.one_step_full_map = np.zeros_like(self.full_map)
        self.local_map = np.zeros(
            (
                len(LAVIRA_GEOMETRIC_CHANNELS),
                self.local_cells,
                self.local_cells,
            ),
            dtype=np.uint8,
        )
        self.global_obstacle_hits = np.zeros(
            (self.cells, self.cells), dtype=np.uint16
        )
        self.collision_map = np.zeros(
            (self.cells, self.cells), dtype=bool
        )
        self.local_bounds_rc: tuple[int, int, int, int] | None = None
        self.update_count = 0
        self.last_grid_map: NavigationGridMap | None = None
        self.last_bundle_id: int | None = None
        self.last_sim_step: int | None = None

    @property
    def initialized(self) -> bool:
        return self.origin_world_xy is not None

    def integrate_bundle(self, bundle: FrameBundle) -> NavigationGridMap:
        """Project one FrameBundle and fuse its geometry into the full map."""

        robot_world_xy = np.asarray(
            bundle.T_world_base[:2, 3], dtype=np.float64
        )
        # A historical target equal to the current root is only a way to reuse
        # the projection/map observation builder. Target selection is discarded.
        observation = build_navigation_grid_map_for_world_goal(
            bundle,
            robot_world_xy,
            self.navigation_config,
        )
        self.integrate_grid_map(observation)
        return observation

    def integrate_grid_map(self, observation: NavigationGridMap) -> None:
        """Fuse a robot-centred observation into the fixed episode grid."""

        self._ensure_initialized(observation.robot_world_xy)
        self._validate_observation(observation)
        robot_cell = self._required_global_cell(
            observation.robot_world_xy, "robot"
        )

        # LaViRA moves the previous current-location channel into past locations
        # before writing the new current pose.
        self.full_map[3] = np.maximum(self.full_map[3], self.full_map[2])
        self.full_map[2].fill(0)
        self.one_step_full_map.fill(0)

        self._project_mask_into_full(
            observation.observed,
            observation.origin_world_xy,
            self.one_step_full_map[1],
        )
        self._project_mask_into_full(
            observation.occupied,
            observation.origin_world_xy,
            self.one_step_full_map[0],
        )
        self._project_hits_into_full(
            observation.obstacle_hits,
            observation.origin_world_xy,
        )

        location_radius_cells = max(
            1,
            int(
                math.ceil(
                    self.navigation_config.start_clearance_m
                    / self.navigation_config.resolution_m
                )
            ),
        )
        _fill_disk(
            self.one_step_full_map[2],
            robot_cell,
            location_radius_cells,
            True,
        )
        self.full_map = np.maximum(self.full_map, self.one_step_full_map)
        # Current location must not accumulate; only channel 3 is historical.
        self.full_map[2].fill(0)
        _fill_disk(
            self.full_map[2],
            robot_cell,
            location_radius_cells,
            True,
        )

        self.update_count += 1
        self.last_grid_map = observation
        self.last_bundle_id = int(observation.bundle_id)
        self.last_sim_step = int(observation.sim_step)
        if (
            self.local_bounds_rc is None
            or (self.update_count - 1)
            % self.global_config.center_reset_steps
            == 0
        ):
            self.local_bounds_rc = self._centered_local_bounds(robot_cell)
        self._refresh_local_map()

    def build_navigation_grid_map(
        self,
        *,
        projection: TargetProjection | None = None,
        historical_target_world_xy: np.ndarray | None = None,
        stable_target_world_xy: np.ndarray | None = None,
    ) -> NavigationGridMap:
        """Build the FMM input from accumulated full-map channels."""

        if not self.initialized or self.last_grid_map is None:
            raise RuntimeError("Global map has no integrated FrameBundle.")
        target_count = sum(
            target is not None
            for target in (
                projection,
                historical_target_world_xy,
                stable_target_world_xy,
            )
        )
        if target_count != 1:
            raise ValueError(
                "Provide exactly one projection, historical target, or stable "
                "world target."
            )
        if (
            projection is not None
            and int(projection.bundle_id) != int(self.last_bundle_id)
        ):
            raise ValueError(
                f"Target projection bundle {projection.bundle_id} does not match "
                f"latest global-map bundle {self.last_bundle_id}."
            )

        latest = self.last_grid_map
        robot_world_xy = np.asarray(
            latest.robot_world_xy, dtype=np.float64
        ).copy()
        robot_cell = self._required_global_cell(robot_world_xy, "robot")
        observed = self.full_map[1].astype(bool)
        occupied = self.full_map[0].astype(bool)
        occupied = _binary_morphology(
            occupied,
            "close",
            cv2_ellipse_kernel(1),
        )

        start_radius_cells = int(
            math.ceil(
                self.navigation_config.start_clearance_m
                / self.navigation_config.resolution_m
            )
        )
        _fill_disk(occupied, robot_cell, start_radius_cells, False)
        observed = observed.copy()
        _fill_disk(observed, robot_cell, start_radius_cells, True)

        inflation_cells = int(
            math.ceil(
                self.navigation_config.robot_radius_m
                / self.navigation_config.resolution_m
            )
        )
        inflated = _binary_morphology(
            occupied,
            "dilate",
            cv2_ellipse_kernel(inflation_cells),
        )
        _fill_disk(inflated, robot_cell, start_radius_cells, False)
        collision_blocked = np.logical_or(inflated, self.collision_map)
        # A collision mark is always placed ahead of the robot, but clear the
        # measured current footprint defensively so FMM can never mask its start.
        _fill_disk(collision_blocked, robot_cell, start_radius_cells, False)
        free = observed & ~occupied
        if self.global_config.unknown_space_policy == "lavira":
            traversable = ~collision_blocked
        else:
            traversable = observed & ~collision_blocked
        _fill_disk(traversable, robot_cell, start_radius_cells, True)
        traversable = _component_connected_to_robot(traversable, robot_cell)

        if projection is not None:
            raw_target_world_xy = np.asarray(
                projection.point_world_m[:2], dtype=np.float64
            )
            selection = _select_safe_target(
                raw_target_world_xy,
                np.asarray(
                    projection.T_world_camera_ros[:2, 3],
                    dtype=np.float64,
                ),
                traversable,
                self.origin_world_xy,
                self.navigation_config,
            )
            strategy_prefix = "global_"
        else:
            raw_target_world_xy = np.asarray(
                (
                    historical_target_world_xy
                    if historical_target_world_xy is not None
                    else stable_target_world_xy
                ),
                dtype=np.float64,
            )
            if raw_target_world_xy.shape != (2,) or not np.all(
                np.isfinite(raw_target_world_xy)
            ):
                raise ValueError(
                    "World target must contain two finite values."
                )
            selection = _select_historical_waypoint_target(
                raw_target_world_xy,
                traversable,
                self.origin_world_xy,
                self.navigation_config,
            )
            strategy_prefix = (
                "global_"
                if historical_target_world_xy is not None
                else "global_stable_"
            )

        return NavigationGridMap(
            bundle_id=int(latest.bundle_id),
            sim_step=int(latest.sim_step),
            config=self.navigation_config,
            resolution_m=self.navigation_config.resolution_m,
            size_m=self.navigation_config.size_m,
            origin_world_xy=self.origin_world_xy.copy(),
            floor_z_world_m=float(latest.floor_z_world_m),
            floor_estimation_method=(
                "latest_observation_for_lavira_global_map:"
                f"{latest.floor_estimation_method}"
            ),
            floor_candidate_count=int(latest.floor_candidate_count),
            sampled_point_counts=dict(latest.sampled_point_counts),
            observed=observed,
            free=free,
            occupied=occupied,
            inflated_obstacles=collision_blocked,
            traversable=traversable,
            obstacle_hits=self.global_obstacle_hits.copy(),
            robot_world_xy=robot_world_xy,
            robot_cell_rc=robot_cell,
            raw_target_world_xy=raw_target_world_xy,
            raw_target_cell_rc=selection.raw_cell,
            safe_target_world_xy=selection.safe_world_xy,
            safe_target_cell_rc=selection.safe_cell,
            target_selection_strategy=(
                f"{strategy_prefix}{selection.strategy}"
            ),
            target_retreat_m=selection.retreat_m,
            target_snap_distance_m=selection.snap_distance_m,
        )

    def to_metadata(self) -> dict:
        if not self.initialized:
            raise RuntimeError("Global map has not been initialized.")
        bounds = self.local_bounds_rc
        return {
            "schema_version": 1,
            "map_type": "lavira_compatible_cumulative_global_map",
            "channel_names": list(LAVIRA_GEOMETRIC_CHANNELS),
            "semantic_channels": {
                "enabled": False,
                "reason": (
                    "Isaac pipeline currently supplies geometric RGB-D mapping "
                    "but no Grounded-SAM semantic tensor."
                ),
            },
            "navigation_config": asdict(self.navigation_config),
            "global_config": asdict(self.global_config),
            "origin_world_xy": self.origin_world_xy.tolist(),
            "initial_robot_world_xy": self.initial_robot_world_xy.tolist(),
            "shape_chw": list(self.full_map.shape),
            "local_shape_chw": list(self.local_map.shape),
            "local_bounds_rc_exclusive": list(bounds) if bounds else None,
            "update_count": self.update_count,
            "last_bundle_id": self.last_bundle_id,
            "last_sim_step": self.last_sim_step,
            "cell_counts": {
                "obstacle": int(np.count_nonzero(self.full_map[0])),
                "explored": int(np.count_nonzero(self.full_map[1])),
                "current_location": int(np.count_nonzero(self.full_map[2])),
                "past_locations": int(np.count_nonzero(self.full_map[3])),
                "collision": int(np.count_nonzero(self.collision_map)),
            },
            "fusion": "channel-wise maximum, matching LaViRA full_map update",
            "coordinate_rule": (
                "row increases with world +Y; column increases with world +X"
            ),
        }

    def mark_collision_world_xy(
        self,
        world_xy: np.ndarray,
        *,
        radius_m: float,
    ) -> int:
        """Persist one LaViRA-style collision disk and return newly blocked cells."""

        if not self.initialized:
            raise RuntimeError("Global map has not been initialized.")
        if not np.isfinite(float(radius_m)) or float(radius_m) < 0.0:
            raise ValueError("Collision radius must be finite and non-negative.")
        cell = self._required_global_cell(
            np.asarray(world_xy, dtype=np.float64),
            "collision observation",
        )
        radius_cells = int(
            math.ceil(
                float(radius_m) / self.navigation_config.resolution_m
            )
        )
        before = int(np.count_nonzero(self.collision_map))
        _fill_disk(self.collision_map, cell, radius_cells, True)
        return int(np.count_nonzero(self.collision_map)) - before

    def clear_collision_world_xy(
        self,
        world_xy: np.ndarray,
        *,
        radius_m: float,
    ) -> int:
        """Clear a proven-free collision area and return removed cell count."""

        if not self.initialized:
            raise RuntimeError("Global map has not been initialized.")
        if not np.isfinite(float(radius_m)) or float(radius_m) < 0.0:
            raise ValueError("Collision radius must be finite and non-negative.")
        cell = self._required_global_cell(
            np.asarray(world_xy, dtype=np.float64),
            "collision clear observation",
        )
        radius_cells = int(
            math.ceil(
                float(radius_m) / self.navigation_config.resolution_m
            )
        )
        before = int(np.count_nonzero(self.collision_map))
        _fill_disk(self.collision_map, cell, radius_cells, False)
        return before - int(np.count_nonzero(self.collision_map))

    def _ensure_initialized(self, robot_world_xy: np.ndarray) -> None:
        if self.initialized:
            return
        robot_xy = np.asarray(robot_world_xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(robot_xy)):
            raise ValueError("Initial robot world position must be finite.")
        if self.global_config.origin_mode == "spawn_center":
            origin = robot_xy - self.navigation_config.size_m * 0.5
        else:
            origin = np.array(
                [
                    self.global_config.manual_origin_world_x_m,
                    self.global_config.manual_origin_world_y_m,
                ],
                dtype=np.float64,
            )
        self.origin_world_xy = origin
        self.initial_robot_world_xy = robot_xy.copy()

    def _validate_observation(self, observation: NavigationGridMap) -> None:
        if (
            abs(
                float(observation.resolution_m)
                - self.navigation_config.resolution_m
            )
            > 1.0e-9
        ):
            raise ValueError("Observation/global map resolutions do not match.")
        if observation.observed.shape != observation.occupied.shape:
            raise ValueError("Observation map channel shapes do not match.")

    def _required_global_cell(
        self, world_xy: np.ndarray, label: str
    ) -> tuple[int, int]:
        cell = world_xy_to_grid_cell(
            np.asarray(world_xy, dtype=np.float64),
            self.origin_world_xy,
            self.navigation_config.resolution_m,
            (self.cells, self.cells),
        )
        if cell is None:
            maximum = (
                self.origin_world_xy + self.navigation_config.size_m
            )
            raise RuntimeError(
                f"{label} world position {np.asarray(world_xy).tolist()} is "
                "outside the fixed global map bounds "
                f"[{self.origin_world_xy.tolist()}, {maximum.tolist()}]. "
                "Increase --nav_map_size_m or set a manual global origin."
            )
        return cell

    def _project_mask_into_full(
        self,
        mask: np.ndarray,
        source_origin_world_xy: np.ndarray,
        destination: np.ndarray,
    ) -> None:
        source_cells = np.argwhere(np.asarray(mask, dtype=bool))
        if source_cells.size == 0:
            return
        destination_cells, inside = self._source_cells_to_global(
            source_cells, source_origin_world_xy
        )
        destination_cells = destination_cells[inside]
        destination[
            destination_cells[:, 0], destination_cells[:, 1]
        ] = 1

    def _project_hits_into_full(
        self,
        hits: np.ndarray,
        source_origin_world_xy: np.ndarray,
    ) -> None:
        source_cells = np.argwhere(np.asarray(hits) > 0)
        if source_cells.size == 0:
            return
        destination_cells, inside = self._source_cells_to_global(
            source_cells, source_origin_world_xy
        )
        source_cells = source_cells[inside]
        destination_cells = destination_cells[inside]
        values = np.asarray(
            hits[source_cells[:, 0], source_cells[:, 1]],
            dtype=np.uint32,
        )
        additions = np.zeros_like(self.global_obstacle_hits, dtype=np.uint32)
        np.add.at(
            additions,
            (destination_cells[:, 0], destination_cells[:, 1]),
            values,
        )
        summed = (
            self.global_obstacle_hits.astype(np.uint32) + additions
        )
        self.global_obstacle_hits[:] = np.minimum(
            summed,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)

    def _source_cells_to_global(
        self,
        source_cells_rc: np.ndarray,
        source_origin_world_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        source = np.asarray(source_cells_rc, dtype=np.int64)
        source_origin = np.asarray(
            source_origin_world_xy, dtype=np.float64
        ).reshape(2)
        resolution = self.navigation_config.resolution_m
        world_x = source_origin[0] + (
            source[:, 1].astype(np.float64) + 0.5
        ) * resolution
        world_y = source_origin[1] + (
            source[:, 0].astype(np.float64) + 0.5
        ) * resolution
        global_cols = np.floor(
            (world_x - self.origin_world_xy[0]) / resolution
        ).astype(np.int64)
        global_rows = np.floor(
            (world_y - self.origin_world_xy[1]) / resolution
        ).astype(np.int64)
        inside = (
            (global_rows >= 0)
            & (global_rows < self.cells)
            & (global_cols >= 0)
            & (global_cols < self.cells)
        )
        return np.column_stack((global_rows, global_cols)), inside

    def _centered_local_bounds(
        self, robot_cell_rc: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        row, col = robot_cell_rc
        half = self.local_cells // 2
        row_min = min(max(int(row) - half, 0), self.cells - self.local_cells)
        col_min = min(max(int(col) - half, 0), self.cells - self.local_cells)
        return (
            row_min,
            row_min + self.local_cells,
            col_min,
            col_min + self.local_cells,
        )

    def _refresh_local_map(self) -> None:
        if self.local_bounds_rc is None:
            return
        row_min, row_max, col_min, col_max = self.local_bounds_rc
        self.local_map = self.full_map[
            :, row_min:row_max, col_min:col_max
        ].copy()


def save_lavira_global_map_debug(
    output_dir: Path,
    state: LaViRAGlobalMapState,
    planning_map: NavigationGridMap | None = None,
) -> dict[str, str]:
    """Save cumulative channels and the final global traversability snapshot."""

    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_name = "lavira_global_map.json"
    arrays_name = "lavira_global_map.npz"
    image_name = "lavira_global_map.png"
    (output_dir / metadata_name).write_text(
        json.dumps(state.to_metadata(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / arrays_name,
        full_map=state.full_map,
        one_step_full_map=state.one_step_full_map,
        local_map=state.local_map,
        obstacle_hits=state.global_obstacle_hits,
        collision_map=state.collision_map,
    )
    if planning_map is not None:
        image = navigation_map_visualization(planning_map)
    else:
        height, width = state.full_map.shape[1:]
        image = np.full((height, width, 3), (85, 85, 85), dtype=np.uint8)
        image[state.full_map[1].astype(bool)] = (235, 235, 235)
        image[state.full_map[0].astype(bool)] = (0, 0, 255)
        image[state.collision_map] = (255, 0, 255)
        image[state.full_map[3].astype(bool)] = (255, 160, 0)
        image[state.full_map[2].astype(bool)] = (255, 80, 0)
        image = np.flipud(image).copy()
    if not cv2.imwrite(str(output_dir / image_name), image):
        raise OSError(f"Failed to write {output_dir / image_name}.")
    return {
        "metadata_json": metadata_name,
        "arrays_npz": arrays_name,
        "visualization_png": image_name,
    }
