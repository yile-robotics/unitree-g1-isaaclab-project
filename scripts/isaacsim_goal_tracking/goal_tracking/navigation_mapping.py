from __future__ import annotations

"""LaViRA 风格的四视图 depth 栅格地图与安全目标构建器。

本模块复刻 LaViRA mapping/FMM 前的数据契约：5 cm 栅格、depth 点云、
obstacle/explored/traversable 通道、障碍膨胀、目标每次缩短 0.1 m，以及最近
可通行栅格回退。Isaac/G1 的必要适配是使用四台相机的真实 K/外参、完整世界位姿、
G1 高度包络和机体半径。

这里只生成地图和目标落点，不运行 FMM，不产生路径或速度命令。
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from .camera import FOUR_VIEW_DIRECTIONS
from .frame_bundle import FrameBundle
from .target_projection import TargetProjection


@dataclass(frozen=True)
class NavigationMapConfig:
    resolution_m: float = 0.05
    size_m: float = 24.0
    depth_stride: int = 4
    depth_min_m: float = 0.1
    depth_max_m: float = 5.0
    nominal_base_height_m: float = 0.80
    floor_search_half_range_m: float = 0.30
    floor_histogram_bin_m: float = 0.02
    obstacle_min_height_m: float = 0.10
    obstacle_max_height_m: float = 1.60
    robot_radius_m: float = 0.35
    start_clearance_m: float = 0.15
    target_retreat_step_m: float = 0.10
    target_snap_max_m: float = 1.00

    def validated(self) -> "NavigationMapConfig":
        if self.resolution_m <= 0.0:
            raise ValueError("Navigation map resolution must be positive.")
        if self.size_m <= 0.0:
            raise ValueError("Navigation map size must be positive.")
        cells_float = self.size_m / self.resolution_m
        if abs(cells_float - round(cells_float)) > 1.0e-6:
            raise ValueError("Navigation map size must be divisible by its resolution.")
        if int(round(cells_float)) < 16:
            raise ValueError("Navigation map must contain at least 16 cells per side.")
        if self.depth_stride <= 0:
            raise ValueError("Navigation depth stride must be positive.")
        if not 0.0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("Navigation depth range is invalid.")
        if self.nominal_base_height_m <= 0.0:
            raise ValueError("Nominal G1 base height must be positive.")
        if self.floor_search_half_range_m <= 0.0 or self.floor_histogram_bin_m <= 0.0:
            raise ValueError("Floor estimation ranges must be positive.")
        if not 0.0 <= self.obstacle_min_height_m < self.obstacle_max_height_m:
            raise ValueError("Obstacle height range is invalid.")
        if self.robot_radius_m < 0.0 or self.start_clearance_m < 0.0:
            raise ValueError("Robot radius and start clearance must be non-negative.")
        if self.target_retreat_step_m <= 0.0 or self.target_snap_max_m < 0.0:
            raise ValueError("Target correction distances are invalid.")
        return self


@dataclass(frozen=True)
class NavigationGridMap:
    bundle_id: int
    sim_step: int
    config: NavigationMapConfig
    resolution_m: float
    size_m: float
    origin_world_xy: np.ndarray
    floor_z_world_m: float
    floor_estimation_method: str
    floor_candidate_count: int
    sampled_point_counts: dict[str, int]
    observed: np.ndarray
    free: np.ndarray
    occupied: np.ndarray
    inflated_obstacles: np.ndarray
    traversable: np.ndarray
    obstacle_hits: np.ndarray
    robot_world_xy: np.ndarray
    robot_cell_rc: tuple[int, int]
    raw_target_world_xy: np.ndarray
    raw_target_cell_rc: tuple[int, int] | None
    safe_target_world_xy: np.ndarray | None
    safe_target_cell_rc: tuple[int, int] | None
    target_selection_strategy: str
    target_retreat_m: float | None
    target_snap_distance_m: float | None

    @property
    def shape(self) -> tuple[int, int]:
        return self.traversable.shape

    def to_metadata(self) -> dict:
        return {
            "schema_version": 1,
            "map_type": "lavira_style_local_navigation_grid",
            "motion_enabled": False,
            "bundle_id": self.bundle_id,
            "sim_step": self.sim_step,
            "mapping_config": asdict(self.config),
            "resolution_m": self.resolution_m,
            "size_m": self.size_m,
            "shape_rc": list(self.shape),
            "grid_axes": (
                "row increases with world +Y; column increases with world +X; "
                "PNG is vertically flipped so world +Y appears upward"
            ),
            "origin_world_xy": self.origin_world_xy.tolist(),
            "floor_z_world_m": self.floor_z_world_m,
            "floor_estimation_method": self.floor_estimation_method,
            "floor_candidate_count": self.floor_candidate_count,
            "sampled_point_counts": self.sampled_point_counts,
            "robot_world_xy": self.robot_world_xy.tolist(),
            "robot_cell_rc": list(self.robot_cell_rc),
            "raw_target_world_xy": self.raw_target_world_xy.tolist(),
            "raw_target_cell_rc": (
                list(self.raw_target_cell_rc)
                if self.raw_target_cell_rc is not None
                else None
            ),
            "safe_target_world_xy": (
                self.safe_target_world_xy.tolist()
                if self.safe_target_world_xy is not None
                else None
            ),
            "safe_target_cell_rc": (
                list(self.safe_target_cell_rc)
                if self.safe_target_cell_rc is not None
                else None
            ),
            "target_selection_strategy": self.target_selection_strategy,
            "target_retreat_m": self.target_retreat_m,
            "target_snap_distance_m": self.target_snap_distance_m,
            "cell_counts": {
                "total": int(self.traversable.size),
                "observed": int(np.count_nonzero(self.observed)),
                "free": int(np.count_nonzero(self.free)),
                "occupied": int(np.count_nonzero(self.occupied)),
                "inflated_obstacles": int(np.count_nonzero(self.inflated_obstacles)),
                "traversable_connected_to_robot": int(
                    np.count_nonzero(self.traversable)
                ),
                "unknown": int(np.count_nonzero(~self.observed)),
            },
            "array_file": "navigation_map.npz",
            "visualization_file": "navigation_map.png",
            "note": (
                "Unknown cells are blocked. The green target is map-validated; this "
                "map artifact by itself contains no path or robot command."
            ),
        }


@dataclass(frozen=True)
class _FloorEstimate:
    z_world_m: float
    method: str
    candidate_count: int


@dataclass(frozen=True)
class _TargetSelection:
    raw_cell: tuple[int, int] | None
    safe_cell: tuple[int, int] | None
    safe_world_xy: np.ndarray | None
    strategy: str
    retreat_m: float | None
    snap_distance_m: float | None


def navigation_map_config_from_args(args_cli) -> NavigationMapConfig:
    return NavigationMapConfig(
        resolution_m=float(args_cli.nav_map_resolution_m),
        size_m=float(args_cli.nav_map_size_m),
        depth_stride=int(args_cli.nav_depth_stride),
        depth_min_m=float(args_cli.rgbd_camera_near),
        depth_max_m=float(args_cli.rgbd_camera_far),
        nominal_base_height_m=float(args_cli.nav_nominal_base_height_m),
        floor_search_half_range_m=float(args_cli.nav_floor_search_half_range_m),
        obstacle_min_height_m=float(args_cli.nav_obstacle_min_height_m),
        obstacle_max_height_m=float(args_cli.nav_obstacle_max_height_m),
        robot_radius_m=float(args_cli.nav_robot_radius_m),
        target_retreat_step_m=float(args_cli.nav_target_retreat_step_m),
        target_snap_max_m=float(args_cli.nav_target_snap_max_m),
    ).validated()


def build_navigation_grid_map(
    bundle: FrameBundle,
    projection: TargetProjection | None,
    config: NavigationMapConfig,
    *,
    historical_target_world_xy: np.ndarray | None = None,
) -> NavigationGridMap:
    """从同一 FrameBundle 的四路深度建立地图并修正本轮目标落点。

    ``projection`` 服务于 NAVIGATE/STOP 的 bbox-depth 表面目标。
    ``historical_target_world_xy`` 服务于 LaViRA BACKTRACK：历史 waypoint
    本身就是机器人曾站立的世界坐标，不应套用语义表面目标的相机方向退让。
    两种目标来源必须且只能提供一种。
    """
    config = config.validated()
    has_projection = projection is not None
    has_historical_target = historical_target_world_xy is not None
    if has_projection == has_historical_target:
        raise ValueError(
            "Provide exactly one of projection or historical_target_world_xy."
        )
    if projection is not None and int(projection.bundle_id) != int(bundle.bundle_id):
        raise ValueError(
            f"Target projection bundle {projection.bundle_id} does not match "
            f"FrameBundle {bundle.bundle_id}."
        )

    robot_world_xy = np.asarray(bundle.T_world_base[:2, 3], dtype=np.float64)
    cells = int(round(config.size_m / config.resolution_m))
    origin_world_xy = robot_world_xy - config.size_m * 0.5
    robot_cell = world_xy_to_grid_cell(
        robot_world_xy, origin_world_xy, config.resolution_m, (cells, cells)
    )
    if robot_cell is None:
        raise RuntimeError("Robot is outside its own centered navigation map.")

    sampled_views: list[tuple[np.ndarray, np.ndarray]] = []
    all_world_points: list[np.ndarray] = []
    sampled_point_counts: dict[str, int] = {}
    for direction in FOUR_VIEW_DIRECTIONS:
        frame = bundle.views[direction]
        points_world = _sample_depth_world_points(
            frame.depth_z_m,
            frame.K,
            frame.T_world_camera_ros,
            stride=config.depth_stride,
            min_depth_m=config.depth_min_m,
            max_depth_m=config.depth_max_m,
        )
        sampled_views.append(
            (np.asarray(frame.T_world_camera_ros[:3, 3], dtype=np.float64), points_world)
        )
        sampled_point_counts[direction] = int(points_world.shape[0])
        if points_world.size:
            all_world_points.append(points_world)

    if not all_world_points:
        raise ValueError("Four-view FrameBundle contains no valid depth points for mapping.")
    all_points = np.concatenate(all_world_points, axis=0)
    floor = _estimate_floor_z(all_points[:, 2], bundle, config)

    observed = np.zeros((cells, cells), dtype=np.uint8)
    obstacle_hits = np.zeros((cells, cells), dtype=np.uint16)
    for camera_world_xyz, points_world in sampled_views:
        if points_world.size == 0:
            continue
        camera_cell = world_xy_to_grid_cell(
            camera_world_xyz[:2], origin_world_xy, config.resolution_m, observed.shape
        )
        if camera_cell is None:
            continue
        endpoint_cells, inside_mask = _world_points_to_grid_cells(
            points_world[:, :2], origin_world_xy, config.resolution_m, observed.shape
        )
        points_inside = points_world[inside_mask]
        if endpoint_cells.size == 0:
            continue

        unique_endpoint_cells = np.unique(endpoint_cells, axis=0)
        _raycast_observed(observed, camera_cell, unique_endpoint_cells)

        relative_height = points_inside[:, 2] - floor.z_world_m
        obstacle_mask = (
            (relative_height >= config.obstacle_min_height_m)
            & (relative_height <= config.obstacle_max_height_m)
        )
        obstacle_cells = endpoint_cells[obstacle_mask]
        if obstacle_cells.size:
            np.add.at(
                obstacle_hits,
                (obstacle_cells[:, 0], obstacle_cells[:, 1]),
                1,
            )

    occupied = obstacle_hits > 0
    # LaViRA closes obstacle holes before producing traversability. OpenCV is
    # used here because Isaac's environment does not currently have skimage.
    close_kernel = cv2_ellipse_kernel(1)
    occupied = _binary_morphology(occupied, "close", close_kernel)

    start_radius_cells = int(math.ceil(config.start_clearance_m / config.resolution_m))
    _fill_disk(occupied, robot_cell, start_radius_cells, False)
    observed_bool = observed.astype(bool)
    _fill_disk(observed_bool, robot_cell, start_radius_cells, True)

    inflation_cells = int(math.ceil(config.robot_radius_m / config.resolution_m))
    inflated = _binary_morphology(
        occupied,
        "dilate",
        cv2_ellipse_kernel(inflation_cells),
    )
    # The robot's measured current footprint is known free; retain a small start
    # seed even if a nearby obstacle's conservative dilation touches it.
    _fill_disk(inflated, robot_cell, start_radius_cells, False)

    free = observed_bool & ~occupied
    traversable = observed_bool & ~inflated
    _fill_disk(traversable, robot_cell, start_radius_cells, True)
    traversable = _component_connected_to_robot(traversable, robot_cell)

    if projection is not None:
        raw_target_world_xy = np.asarray(
            projection.point_world_m[:2], dtype=np.float64
        )
        selected_camera_xy = np.asarray(
            projection.T_world_camera_ros[:2, 3], dtype=np.float64
        )
        target_selection = _select_safe_target(
            raw_target_world_xy,
            selected_camera_xy,
            traversable,
            origin_world_xy,
            config,
        )
    else:
        raw_target_world_xy = np.asarray(
            historical_target_world_xy, dtype=np.float64
        )
        if raw_target_world_xy.shape != (2,) or not np.all(
            np.isfinite(raw_target_world_xy)
        ):
            raise ValueError(
                "historical_target_world_xy must contain two finite coordinates."
            )
        target_selection = _select_historical_waypoint_target(
            raw_target_world_xy,
            traversable,
            origin_world_xy,
            config,
        )

    return NavigationGridMap(
        bundle_id=int(bundle.bundle_id),
        sim_step=int(bundle.sim_step),
        config=config,
        resolution_m=config.resolution_m,
        size_m=config.size_m,
        origin_world_xy=origin_world_xy,
        floor_z_world_m=floor.z_world_m,
        floor_estimation_method=floor.method,
        floor_candidate_count=floor.candidate_count,
        sampled_point_counts=sampled_point_counts,
        observed=observed_bool,
        free=free,
        occupied=occupied,
        inflated_obstacles=inflated,
        traversable=traversable,
        obstacle_hits=obstacle_hits,
        robot_world_xy=robot_world_xy,
        robot_cell_rc=robot_cell,
        raw_target_world_xy=raw_target_world_xy,
        raw_target_cell_rc=target_selection.raw_cell,
        safe_target_world_xy=target_selection.safe_world_xy,
        safe_target_cell_rc=target_selection.safe_cell,
        target_selection_strategy=target_selection.strategy,
        target_retreat_m=target_selection.retreat_m,
        target_snap_distance_m=target_selection.snap_distance_m,
    )


def build_navigation_grid_map_for_world_goal(
    bundle: FrameBundle,
    target_world_xy: np.ndarray,
    config: NavigationMapConfig,
) -> NavigationGridMap:
    """为 LaViRA BACKTRACK 在当前深度地图上设置历史世界坐标目标。"""

    return build_navigation_grid_map(
        bundle,
        None,
        config,
        historical_target_world_xy=np.asarray(target_world_xy, dtype=np.float64),
    )


def save_navigation_map_debug(output_dir: Path, grid_map: NavigationGridMap) -> dict[str, str]:
    """保存地图数组、元数据和便于人工核对的俯视图。"""
    import cv2

    output_dir = Path(output_dir)
    json_name = "navigation_map.json"
    arrays_name = "navigation_map.npz"
    image_name = "navigation_map.png"
    (output_dir / json_name).write_text(
        json.dumps(grid_map.to_metadata(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / arrays_name,
        observed=grid_map.observed,
        free=grid_map.free,
        occupied=grid_map.occupied,
        inflated_obstacles=grid_map.inflated_obstacles,
        traversable=grid_map.traversable,
        obstacle_hits=grid_map.obstacle_hits,
    )
    visualization = navigation_map_visualization(grid_map)
    if not cv2.imwrite(str(output_dir / image_name), visualization):
        raise OSError(f"Failed to write {output_dir / image_name}.")
    return {
        "metadata_json": json_name,
        "arrays_npz": arrays_name,
        "visualization_png": image_name,
    }


def navigation_map_visualization(grid_map: NavigationGridMap) -> np.ndarray:
    """生成 BGR 俯视图；世界 +Y 在保存图片中朝上。"""
    import cv2

    height, width = grid_map.shape
    image = np.full((height, width, 3), (85, 85, 85), dtype=np.uint8)
    image[grid_map.observed] = (225, 225, 225)
    image[grid_map.free] = (245, 245, 245)
    image[grid_map.inflated_obstacles] = (0, 0, 110)
    image[grid_map.occupied] = (0, 0, 255)
    image[grid_map.traversable] = (235, 255, 235)
    image = np.flipud(image).copy()

    def display_xy(cell: tuple[int, int]) -> tuple[int, int]:
        row, col = cell
        return int(col), int(height - 1 - row)

    cv2.circle(image, display_xy(grid_map.robot_cell_rc), 5, (255, 80, 0), -1)
    if grid_map.raw_target_cell_rc is not None:
        cv2.drawMarker(
            image,
            display_xy(grid_map.raw_target_cell_rc),
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )
    if grid_map.safe_target_cell_rc is not None:
        safe_xy = display_xy(grid_map.safe_target_cell_rc)
        cv2.circle(image, safe_xy, 6, (0, 180, 0), -1)
        cv2.circle(image, safe_xy, 7, (255, 255, 255), 1)
    cv2.putText(
        image,
        (
            f"res={grid_map.resolution_m:.2f}m "
            f"floor_z={grid_map.floor_z_world_m:+.2f}m "
            f"target={grid_map.target_selection_strategy}"
        ),
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def world_xy_to_grid_cell(
    world_xy: np.ndarray,
    origin_world_xy: np.ndarray,
    resolution_m: float,
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    xy = np.asarray(world_xy, dtype=np.float64).reshape(2)
    origin = np.asarray(origin_world_xy, dtype=np.float64).reshape(2)
    col = int(math.floor((xy[0] - origin[0]) / resolution_m))
    row = int(math.floor((xy[1] - origin[1]) / resolution_m))
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        return None
    return row, col


def grid_cell_to_world_xy(
    cell_rc: tuple[int, int],
    origin_world_xy: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    row, col = cell_rc
    origin = np.asarray(origin_world_xy, dtype=np.float64).reshape(2)
    return origin + np.array(
        [(float(col) + 0.5) * resolution_m, (float(row) + 0.5) * resolution_m]
    )


def _sample_depth_world_points(
    depth_z_m: np.ndarray,
    K: np.ndarray,
    T_world_camera_ros: np.ndarray,
    *,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    depth = np.asarray(depth_z_m)
    if depth.ndim != 2:
        raise ValueError(f"Navigation mapping depth must be HxW, got {depth.shape}.")
    rows = np.arange(0, depth.shape[0], stride, dtype=np.int64)
    cols = np.arange(0, depth.shape[1], stride, dtype=np.int64)
    uu, vv = np.meshgrid(cols, rows)
    sampled_depth = np.asarray(depth[vv, uu], dtype=np.float64)
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= min_depth_m)
        & (sampled_depth <= max_depth_m)
    )
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64)

    pixels = np.stack(
        (uu[valid].astype(np.float64), vv[valid].astype(np.float64), np.ones(np.count_nonzero(valid))),
        axis=0,
    )
    try:
        rays = np.linalg.solve(np.asarray(K, dtype=np.float64), pixels)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Navigation mapping camera intrinsic matrix K is singular.") from exc
    points_camera = rays * sampled_depth[valid][None, :]
    points_camera_h = np.concatenate(
        (points_camera, np.ones((1, points_camera.shape[1]), dtype=np.float64)),
        axis=0,
    )
    points_world_h = np.asarray(T_world_camera_ros, dtype=np.float64) @ points_camera_h
    return np.ascontiguousarray((points_world_h[:3] / points_world_h[3:4]).T)


def _estimate_floor_z(
    point_z_world: np.ndarray,
    bundle: FrameBundle,
    config: NavigationMapConfig,
) -> _FloorEstimate:
    expected = float(bundle.T_world_base[2, 3]) - config.nominal_base_height_m
    z_values = np.asarray(point_z_world, dtype=np.float64)
    candidates = z_values[
        np.isfinite(z_values)
        & (z_values >= expected - config.floor_search_half_range_m)
        & (z_values <= expected + config.floor_search_half_range_m)
    ]
    if candidates.size < 20:
        return _FloorEstimate(expected, "nominal_base_height_fallback", int(candidates.size))

    # The standing G1 root-height prior is reliable in the current flat-floor
    # house. Prefer depth evidence close to that prior so a dense lower wall or
    # furniture band cannot win the broad histogram merely by pixel count.
    tight_half_range = max(config.floor_histogram_bin_m * 3.0, 0.06)
    tight_candidates = candidates[np.abs(candidates - expected) <= tight_half_range]
    if tight_candidates.size >= 20:
        histogram_candidates = tight_candidates
        histogram_method = "depth_height_histogram_near_base_prior"
        histogram_lower = expected - tight_half_range
        histogram_upper = expected + tight_half_range
    else:
        histogram_candidates = candidates
        histogram_method = "depth_height_histogram_broad"
        histogram_lower = expected - config.floor_search_half_range_m
        histogram_upper = expected + config.floor_search_half_range_m

    bin_width = config.floor_histogram_bin_m
    edges = np.arange(
        histogram_lower,
        histogram_upper + bin_width,
        bin_width,
        dtype=np.float64,
    )
    histogram, edges = np.histogram(histogram_candidates, bins=edges)
    peak = int(np.argmax(histogram))
    peak_low, peak_high = edges[peak], edges[peak + 1]
    peak_values = histogram_candidates[
        (histogram_candidates >= peak_low) & (histogram_candidates < peak_high)
    ]
    if peak_values.size == 0:
        return _FloorEstimate(expected, "nominal_base_height_fallback", int(candidates.size))
    return _FloorEstimate(
        float(np.median(peak_values)),
        histogram_method,
        int(candidates.size),
    )


def _world_points_to_grid_cells(
    points_world_xy: np.ndarray,
    origin_world_xy: np.ndarray,
    resolution_m: float,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_world_xy, dtype=np.float64)
    cols = np.floor((points[:, 0] - origin_world_xy[0]) / resolution_m).astype(np.int64)
    rows = np.floor((points[:, 1] - origin_world_xy[1]) / resolution_m).astype(np.int64)
    inside = (
        (rows >= 0)
        & (rows < shape[0])
        & (cols >= 0)
        & (cols < shape[1])
    )
    return np.stack((rows[inside], cols[inside]), axis=1), inside


def _raycast_observed(
    observed: np.ndarray,
    camera_cell_rc: tuple[int, int],
    endpoint_cells_rc: np.ndarray,
) -> None:
    import cv2

    camera_row, camera_col = camera_cell_rc
    for endpoint_row, endpoint_col in endpoint_cells_rc:
        cv2.line(
            observed,
            (int(camera_col), int(camera_row)),
            (int(endpoint_col), int(endpoint_row)),
            color=1,
            thickness=1,
            lineType=cv2.LINE_8,
        )


def _select_safe_target(
    raw_target_world_xy: np.ndarray,
    camera_world_xy: np.ndarray,
    traversable: np.ndarray,
    origin_world_xy: np.ndarray,
    config: NavigationMapConfig,
) -> _TargetSelection:
    raw_cell = world_xy_to_grid_cell(
        raw_target_world_xy, origin_world_xy, config.resolution_m, traversable.shape
    )
    if raw_cell is not None and traversable[raw_cell]:
        return _TargetSelection(
            raw_cell,
            raw_cell,
            grid_cell_to_world_xy(raw_cell, origin_world_xy, config.resolution_m),
            "raw_target_traversable",
            0.0,
            0.0,
        )

    target = np.asarray(raw_target_world_xy, dtype=np.float64)
    camera = np.asarray(camera_world_xy, dtype=np.float64)
    vector = target - camera
    distance = float(np.linalg.norm(vector))
    if distance > 1.0e-9:
        direction = vector / distance
        max_steps = int(math.floor(distance / config.target_retreat_step_m))
        for step in range(1, max_steps + 1):
            retreat = min(step * config.target_retreat_step_m, distance)
            candidate = target - direction * retreat
            cell = world_xy_to_grid_cell(
                candidate, origin_world_xy, config.resolution_m, traversable.shape
            )
            if cell is not None and traversable[cell]:
                return _TargetSelection(
                    raw_cell,
                    cell,
                    grid_cell_to_world_xy(cell, origin_world_xy, config.resolution_m),
                    "lavira_depth_retreat",
                    float(retreat),
                    float(np.linalg.norm(candidate - target)),
                )

    traversable_cells = np.argwhere(traversable)
    if traversable_cells.size:
        world_centers = np.column_stack(
            (
                origin_world_xy[0]
                + (traversable_cells[:, 1].astype(np.float64) + 0.5)
                * config.resolution_m,
                origin_world_xy[1]
                + (traversable_cells[:, 0].astype(np.float64) + 0.5)
                * config.resolution_m,
            )
        )
        distances = np.linalg.norm(world_centers - target[None, :], axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        if nearest_distance <= config.target_snap_max_m:
            row, col = traversable_cells[nearest_index]
            cell = (int(row), int(col))
            return _TargetSelection(
                raw_cell,
                cell,
                world_centers[nearest_index],
                "nearest_traversable",
                None,
                nearest_distance,
            )

    return _TargetSelection(raw_cell, None, None, "no_safe_target", None, None)


def _select_historical_waypoint_target(
    target_world_xy: np.ndarray,
    traversable: np.ndarray,
    origin_world_xy: np.ndarray,
    config: NavigationMapConfig,
) -> _TargetSelection:
    """选择历史机器人位姿，不执行 bbox 表面目标的相机方向退让。"""

    target = np.asarray(target_world_xy, dtype=np.float64)
    raw_cell = world_xy_to_grid_cell(
        target,
        origin_world_xy,
        config.resolution_m,
        traversable.shape,
    )
    if raw_cell is not None and traversable[raw_cell]:
        return _TargetSelection(
            raw_cell,
            raw_cell,
            grid_cell_to_world_xy(raw_cell, origin_world_xy, config.resolution_m),
            "historical_waypoint_traversable",
            0.0,
            float(
                np.linalg.norm(
                    grid_cell_to_world_xy(
                        raw_cell, origin_world_xy, config.resolution_m
                    )
                    - target
                )
            ),
        )

    traversable_cells = np.argwhere(traversable)
    if traversable_cells.size:
        world_centers = np.column_stack(
            (
                origin_world_xy[0]
                + (traversable_cells[:, 1].astype(np.float64) + 0.5)
                * config.resolution_m,
                origin_world_xy[1]
                + (traversable_cells[:, 0].astype(np.float64) + 0.5)
                * config.resolution_m,
            )
        )
        distances = np.linalg.norm(world_centers - target[None, :], axis=1)
        nearest_index = int(np.argmin(distances))
        nearest_distance = float(distances[nearest_index])
        if nearest_distance <= config.target_snap_max_m:
            row, col = traversable_cells[nearest_index]
            return _TargetSelection(
                raw_cell,
                (int(row), int(col)),
                world_centers[nearest_index],
                "historical_waypoint_nearest_traversable",
                None,
                nearest_distance,
            )

    return _TargetSelection(
        raw_cell,
        None,
        None,
        "historical_waypoint_unreachable",
        None,
        None,
    )


def cv2_ellipse_kernel(radius_cells: int) -> np.ndarray:
    import cv2

    radius = max(int(radius_cells), 0)
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _binary_morphology(mask: np.ndarray, operation: str, kernel: np.ndarray) -> np.ndarray:
    import cv2

    source = np.asarray(mask, dtype=np.uint8)
    if operation == "close":
        result = cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel)
    elif operation == "dilate":
        result = cv2.dilate(source, kernel, iterations=1)
    else:
        raise ValueError(f"Unsupported binary morphology operation {operation!r}.")
    return result.astype(bool)


def _fill_disk(
    array: np.ndarray,
    center_rc: tuple[int, int],
    radius_cells: int,
    value: bool,
) -> None:
    row, col = center_rc
    radius = max(int(radius_cells), 0)
    row_min = max(0, int(row) - radius)
    row_max = min(array.shape[0], int(row) + radius + 1)
    col_min = max(0, int(col) - radius)
    col_max = min(array.shape[1], int(col) + radius + 1)
    rows, cols = np.ogrid[row_min:row_max, col_min:col_max]
    disk = (rows - int(row)) ** 2 + (cols - int(col)) ** 2 <= radius**2
    patch = array[row_min:row_max, col_min:col_max]
    patch[disk] = value


def _component_connected_to_robot(
    traversable: np.ndarray, robot_cell_rc: tuple[int, int]
) -> np.ndarray:
    import cv2

    count, labels = cv2.connectedComponents(
        np.asarray(traversable, dtype=np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(traversable, dtype=bool)
    robot_label = int(labels[robot_cell_rc])
    if robot_label == 0:
        return np.zeros_like(traversable, dtype=bool)
    return labels == robot_label
