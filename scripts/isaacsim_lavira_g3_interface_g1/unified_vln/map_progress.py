from __future__ import annotations

"""Episode-local sparse RGB-D exploration map.

The map deliberately has no preallocated metric extent.  Every cell is keyed by
its integer coordinate in one stable episode frame, so negative coordinates and
previously unknown environment sizes are supported without shifting an origin.
The implementation consumes a single forward optical-Z depth frame, a fixed
robot-to-camera transform, and the robot's planar world pose.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .odometry import Pose2D
from .types import ViewFrame


GridCell = tuple[int, int]


@dataclass(frozen=True)
class SparseMapConfig:
    """Geometry and filtering parameters shared by Isaac and the real G1."""

    resolution_m: float = 0.05
    depth_stride: int = 8
    depth_min_m: float = 0.10
    depth_max_m: float = 5.0
    camera_offset_x_m: float = 0.085
    camera_offset_y_m: float = 0.0
    camera_offset_z_m: float = 0.56
    camera_yaw_rad: float = 0.0
    camera_down_tilt_rad: float = math.radians(12.0)
    nominal_base_height_m: float = 0.80
    floor_z_world_m: float = 0.0
    obstacle_min_height_m: float = 0.10
    obstacle_max_height_m: float = 1.60
    robot_radius_m: float = 0.35
    start_clearance_m: float = 0.15
    connectivity: int = 8

    def validated(self) -> "SparseMapConfig":
        finite = (
            self.resolution_m,
            self.depth_min_m,
            self.depth_max_m,
            self.camera_offset_x_m,
            self.camera_offset_y_m,
            self.camera_offset_z_m,
            self.camera_yaw_rad,
            self.camera_down_tilt_rad,
            self.nominal_base_height_m,
            self.floor_z_world_m,
            self.obstacle_min_height_m,
            self.obstacle_max_height_m,
            self.robot_radius_m,
            self.start_clearance_m,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("Sparse map parameters must be finite.")
        if self.resolution_m <= 0.0:
            raise ValueError("Sparse map resolution must be positive.")
        if self.depth_stride <= 0:
            raise ValueError("Sparse map depth stride must be positive.")
        if not 0.0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("Sparse map depth range is invalid.")
        if not 0.0 <= self.obstacle_min_height_m < self.obstacle_max_height_m:
            raise ValueError("Sparse map obstacle height range is invalid.")
        if self.robot_radius_m < 0.0 or self.start_clearance_m < 0.0:
            raise ValueError("Sparse map radii must be non-negative.")
        if self.connectivity not in {4, 8}:
            raise ValueError("Sparse map connectivity must be 4 or 8.")
        return self


@dataclass(frozen=True)
class MapIntegrationResult:
    frame_id: int
    sim_step: int
    valid_depth_samples: int
    unique_ray_endpoints: int
    new_explored_cells: int
    new_occupied_cells: int
    explored_cells: int
    occupied_cells: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MapProgressSnapshot:
    pose_frame_id: str
    frame_epoch: int
    resolution_m: float
    explored_cells: int
    new_explored_cells: int
    traversable_cells: int
    occupied_cells: int
    update_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class SparseEpisodeExplorationMap:
    """Unbounded sparse 5 cm map accumulated for one episode.

    ``observed_cells`` is monotonic.  Occupied and inflated cells are also fused
    monotonically, matching LaViRA's max-fused geometric map.  Traversability is
    recomputed from the current robot cell and therefore may increase or decrease
    as new geometry is observed.
    """

    def __init__(
        self,
        config: SparseMapConfig | None = None,
        *,
        pose_frame_id: str,
        frame_epoch: int = 0,
    ):
        self.config = (config or SparseMapConfig()).validated()
        if not str(pose_frame_id).strip():
            raise ValueError("pose_frame_id must not be empty.")
        if int(frame_epoch) < 0:
            raise ValueError("frame_epoch must be non-negative.")
        self.pose_frame_id = str(pose_frame_id)
        self.frame_epoch = int(frame_epoch)
        self.observed_cells: set[GridCell] = set()
        self.occupied_cells: set[GridCell] = set()
        self.inflated_obstacle_cells: set[GridCell] = set()
        self.obstacle_hits: dict[GridCell, int] = {}
        self.current_robot_cell: GridCell | None = None
        self.update_count = 0
        self.last_frame_id: int | None = None
        self.last_sim_step: int | None = None

        self._inflation_offsets = tuple(
            _disk_offsets(
                int(
                    math.ceil(
                        self.config.robot_radius_m / self.config.resolution_m
                    )
                )
            )
        )
        self._start_offsets = tuple(
            _disk_offsets(
                int(
                    math.ceil(
                        self.config.start_clearance_m / self.config.resolution_m
                    )
                )
            )
        )

    @property
    def explored_cells(self) -> int:
        return len(self.observed_cells)

    def world_xy_to_cell(self, world_x: float, world_y: float) -> GridCell:
        """Map a world point to an unbounded integer cell; negatives are valid."""

        if not math.isfinite(float(world_x)) or not math.isfinite(float(world_y)):
            raise ValueError("World coordinates must be finite.")
        resolution = self.config.resolution_m
        return (
            int(math.floor(float(world_x) / resolution)),
            int(math.floor(float(world_y) / resolution)),
        )

    def integrate(self, frame: ViewFrame, robot_pose: Pose2D) -> MapIntegrationResult:
        """Fuse one physical forward RGB-D observation into the episode map."""

        frame = frame.validated()
        robot_pose = robot_pose.validated()
        config = self.config
        depth = np.asarray(frame.depth_m, dtype=np.float64)
        K = np.asarray(frame.K, dtype=np.float64)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("Sparse map camera fx/fy must be positive.")

        rows = np.arange(0, depth.shape[0], config.depth_stride, dtype=np.int64)
        cols = np.arange(0, depth.shape[1], config.depth_stride, dtype=np.int64)
        uu, vv = np.meshgrid(cols, rows)
        sampled_depth = depth[vv, uu]
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= config.depth_min_m)
            & (sampled_depth <= config.depth_max_m)
        )

        before_explored = len(self.observed_cells)
        before_occupied = len(self.occupied_cells)
        robot_cell = self.world_xy_to_cell(robot_pose.x, robot_pose.y)
        self.current_robot_cell = robot_cell
        self._mark_start_footprint(robot_cell)

        valid_count = int(np.count_nonzero(valid))
        unique_endpoint_count = 0
        if valid_count:
            z_optical = sampled_depth[valid]
            x_optical = (uu[valid].astype(np.float64) - cx) * z_optical / fx
            y_optical = (vv[valid].astype(np.float64) - cy) * z_optical / fy
            points_optical = np.stack(
                (x_optical, y_optical, z_optical), axis=1
            )
            points_world, camera_world = self._optical_points_to_world(
                points_optical, robot_pose
            )
            endpoint_cells = {
                self.world_xy_to_cell(point[0], point[1])
                for point in points_world
            }
            unique_endpoint_count = len(endpoint_cells)
            camera_cell = self.world_xy_to_cell(camera_world[0], camera_world[1])
            for endpoint_cell in endpoint_cells:
                self.observed_cells.update(
                    _bresenham_cells(camera_cell, endpoint_cell)
                )

            relative_height = points_world[:, 2] - config.floor_z_world_m
            obstacle_mask = (
                (relative_height >= config.obstacle_min_height_m)
                & (relative_height <= config.obstacle_max_height_m)
            )
            obstacle_endpoints = {
                self.world_xy_to_cell(point[0], point[1])
                for point in points_world[obstacle_mask]
            }
            for cell in obstacle_endpoints:
                self.obstacle_hits[cell] = self.obstacle_hits.get(cell, 0) + 1
                if cell not in self.occupied_cells:
                    self.occupied_cells.add(cell)
                    self._inflate_new_obstacle(cell)

        # The measured robot footprint is always usable even if conservative
        # obstacle inflation from a nearby endpoint overlaps it.
        self._mark_start_footprint(robot_cell)
        self.update_count += 1
        self.last_frame_id = int(frame.frame_id)
        self.last_sim_step = int(frame.sim_step)
        return MapIntegrationResult(
            frame_id=int(frame.frame_id),
            sim_step=int(frame.sim_step),
            valid_depth_samples=valid_count,
            unique_ray_endpoints=unique_endpoint_count,
            new_explored_cells=len(self.observed_cells) - before_explored,
            new_occupied_cells=len(self.occupied_cells) - before_occupied,
            explored_cells=len(self.observed_cells),
            occupied_cells=len(self.occupied_cells),
        )

    def snapshot(self, *, explored_before: int | None = None) -> MapProgressSnapshot:
        """Return the cumulative counts and optional window-local map gain."""

        current = len(self.observed_cells)
        if explored_before is None:
            new_explored = 0
        else:
            if int(explored_before) < 0 or int(explored_before) > current:
                raise ValueError("explored_before is outside the cumulative map count.")
            new_explored = current - int(explored_before)
        return MapProgressSnapshot(
            pose_frame_id=self.pose_frame_id,
            frame_epoch=self.frame_epoch,
            resolution_m=self.config.resolution_m,
            explored_cells=current,
            new_explored_cells=new_explored,
            traversable_cells=self.traversable_cell_count(),
            occupied_cells=len(self.occupied_cells),
            update_count=self.update_count,
        )

    def traversable_cell_count(self) -> int:
        """Count observed, non-inflated cells connected to the current robot."""

        if self.current_robot_cell is None:
            return 0
        start = self.current_robot_cell
        blocked = self.inflated_obstacle_cells.difference(
            _translated_cells(start, self._start_offsets)
        )
        candidates = self.observed_cells.difference(blocked)
        candidates.add(start)
        if start not in candidates:
            return 0
        neighbours = _neighbour_offsets(self.config.connectivity)
        visited: set[GridCell] = {start}
        pending = [start]
        while pending:
            cell_x, cell_y = pending.pop()
            for offset_x, offset_y in neighbours:
                neighbour = (cell_x + offset_x, cell_y + offset_y)
                if neighbour in candidates and neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)
        return len(visited)

    def save_debug(self, output_prefix: Path, snapshot: MapProgressSnapshot) -> dict:
        """Save metadata plus a dynamically bounded visualization.

        The rendering array is temporary and follows the explored extent.  It
        does not impose a storage boundary on the sparse map itself.
        """

        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = output_prefix.with_suffix(".json")
        image_path = output_prefix.with_suffix(".png")
        metadata = {
            "map_type": "unbounded_sparse_episode_exploration",
            "snapshot": snapshot.to_dict(),
            "config": asdict(self.config),
            "last_frame_id": self.last_frame_id,
            "last_sim_step": self.last_sim_step,
            "visualization_file": image_path.name,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._save_visualization(image_path)
        return {"metadata": str(metadata_path), "visualization": str(image_path)}

    def _optical_points_to_world(
        self, points_optical: np.ndarray, robot_pose: Pose2D
    ) -> tuple[np.ndarray, np.ndarray]:
        config = self.config
        # Optical convention: +x right, +y down, +z forward.
        optical_to_robot_level = np.array(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
        robot_from_camera_rotation = (
            _rotation_z(config.camera_yaw_rad)
            @ _rotation_y(config.camera_down_tilt_rad)
            @ optical_to_robot_level
        )
        camera_in_robot = np.array(
            [
                config.camera_offset_x_m,
                config.camera_offset_y_m,
                config.camera_offset_z_m,
            ],
            dtype=np.float64,
        )
        world_from_robot_rotation = _rotation_z(robot_pose.yaw)
        robot_world = np.array(
            [robot_pose.x, robot_pose.y, config.nominal_base_height_m],
            dtype=np.float64,
        )
        camera_world = robot_world + world_from_robot_rotation @ camera_in_robot
        points_robot = (
            robot_from_camera_rotation @ np.asarray(points_optical).T
        ).T + camera_in_robot
        points_world = (
            world_from_robot_rotation @ points_robot.T
        ).T + robot_world
        return points_world, camera_world

    def _mark_start_footprint(self, robot_cell: GridCell) -> None:
        self.observed_cells.update(_translated_cells(robot_cell, self._start_offsets))

    def _inflate_new_obstacle(self, cell: GridCell) -> None:
        self.inflated_obstacle_cells.update(
            _translated_cells(cell, self._inflation_offsets)
        )

    def _save_visualization(self, image_path: Path) -> None:
        import cv2

        cells = set(self.observed_cells)
        if self.current_robot_cell is not None:
            cells.add(self.current_robot_cell)
        if not cells:
            image = np.full((1, 1, 3), 85, dtype=np.uint8)
            if not cv2.imwrite(str(image_path), image):
                raise OSError(f"Failed to save sparse map visualization: {image_path}")
            return

        xs = [cell[0] for cell in cells]
        ys = [cell[1] for cell in cells]
        margin = max(2, int(math.ceil(0.5 / self.config.resolution_m)))
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        width_cells = max_x - min_x + 1
        height_cells = max_y - min_y + 1
        scale = max(1, int(math.ceil(max(width_cells, height_cells) / 2048.0)))
        width = int(math.ceil(width_cells / scale))
        height = int(math.ceil(height_cells / scale))
        image = np.full((height, width, 3), 85, dtype=np.uint8)

        def pixel(cell: GridCell) -> tuple[int, int]:
            col = (cell[0] - min_x) // scale
            row_from_bottom = (cell[1] - min_y) // scale
            return int(col), int(height - 1 - row_from_bottom)

        blocked = self.inflated_obstacle_cells
        for cell in self.observed_cells:
            col, row = pixel(cell)
            if 0 <= row < height and 0 <= col < width:
                image[row, col] = (245, 245, 245)
        for cell in blocked:
            if cell not in self.observed_cells:
                continue
            col, row = pixel(cell)
            if 0 <= row < height and 0 <= col < width:
                image[row, col] = (0, 0, 120)
        for cell in self.occupied_cells:
            col, row = pixel(cell)
            if 0 <= row < height and 0 <= col < width:
                image[row, col] = (0, 0, 255)
        if self.current_robot_cell is not None:
            cv2.circle(image, pixel(self.current_robot_cell), 4, (255, 80, 0), -1)
        if not cv2.imwrite(str(image_path), image):
            raise OSError(f"Failed to save sparse map visualization: {image_path}")


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(float(angle)), math.sin(float(angle))
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(float(angle)), math.sin(float(angle))
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _disk_offsets(radius_cells: int) -> Iterable[GridCell]:
    radius_cells = max(0, int(radius_cells))
    radius_squared = radius_cells * radius_cells
    for offset_x in range(-radius_cells, radius_cells + 1):
        for offset_y in range(-radius_cells, radius_cells + 1):
            if offset_x * offset_x + offset_y * offset_y <= radius_squared:
                yield offset_x, offset_y


def _translated_cells(
    center: GridCell, offsets: Iterable[GridCell]
) -> set[GridCell]:
    return {
        (center[0] + offset[0], center[1] + offset[1])
        for offset in offsets
    }


def _neighbour_offsets(connectivity: int) -> tuple[GridCell, ...]:
    cardinal = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if connectivity == 4:
        return cardinal
    return cardinal + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _bresenham_cells(start: GridCell, end: GridCell) -> Iterable[GridCell]:
    """Yield every integer cell on an unbounded 2-D line, including endpoints."""

    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += step_x
        if doubled <= dx:
            error += dx
            y0 += step_y
