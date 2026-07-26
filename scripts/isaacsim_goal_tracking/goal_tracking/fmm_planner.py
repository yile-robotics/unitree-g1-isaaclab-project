from __future__ import annotations

"""LaViRA FMM planner adapted to the Isaac/G1 navigation-grid contract.

The distance-field construction and the five-cell short-term-goal ring follow
``lavira_code/vlnce_baselines/models/fmm_planner.py``.  LaViRA converts that
single short-term goal into Habitat discrete actions.  Isaac/G1 instead needs
a collision-checked polyline, so this module additionally descends the FMM
field one traversable cell at a time and exports world-frame waypoints.

This module is planner-only: it never sends velocity commands or changes the
stand/locomotion policy state.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from .navigation_mapping import (
    NavigationGridMap,
    grid_cell_to_world_xy,
    navigation_map_visualization,
)


class FMMPlanningError(RuntimeError):
    """Raised when a safe FMM path cannot be produced from the supplied grid."""


@dataclass(frozen=True)
class FMMPlannerConfig:
    # LaViRA uses a five-cell ring. At 5 cm/cell this is a 25 cm local target.
    step_size_cells: int = 5
    goal_tolerance_cells: int = 1
    waypoint_spacing_m: float = 0.25
    max_path_steps: int = 20_000
    descent_epsilon_cells: float = 1.0e-6

    def validated(self) -> "FMMPlannerConfig":
        if self.step_size_cells <= 0:
            raise ValueError("FMM step size must be positive.")
        if self.goal_tolerance_cells < 0:
            raise ValueError("FMM goal tolerance must be non-negative.")
        if self.waypoint_spacing_m <= 0.0:
            raise ValueError("FMM waypoint spacing must be positive.")
        if self.max_path_steps <= 0:
            raise ValueError("FMM maximum path steps must be positive.")
        if self.descent_epsilon_cells < 0.0:
            raise ValueError("FMM descent epsilon must be non-negative.")
        return self


@dataclass(frozen=True)
class FMMPlan:
    config: FMMPlannerConfig
    bundle_id: int
    sim_step: int
    resolution_m: float
    origin_world_xy: np.ndarray
    start_cell_rc: tuple[int, int]
    goal_cell_rc: tuple[int, int]
    start_world_xy: np.ndarray
    goal_world_xy: np.ndarray
    distance_field_cells: np.ndarray
    start_distance_cells: float
    lavira_short_term_goal_cell_rc: tuple[int, int]
    lavira_short_term_goal_world_xy: np.ndarray
    lavira_short_term_goal_line_safe: bool
    path_cells_rc: np.ndarray
    path_world_xy: np.ndarray
    waypoint_cells_rc: np.ndarray
    waypoints_world_xy: np.ndarray
    path_length_m: float

    def to_metadata(self) -> dict:
        finite = self.distance_field_cells[np.isfinite(self.distance_field_cells)]
        return {
            "schema_version": 1,
            "plan_type": "lavira_fmm_world_waypoint_probe",
            "status": "success",
            "motion_enabled": False,
            "bundle_id": self.bundle_id,
            "sim_step": self.sim_step,
            "source": {
                "distance_field": (
                    "LaViRA FMMPlanner.set_goal: masked traversable grid + "
                    "skfmm.distance(dx=1)"
                ),
                "short_term_goal": (
                    "LaViRA five-cell local-ring minimum with a goal-aware "
                    "terminal fix"
                ),
                "isaac_adaptation": (
                    "8-neighbor monotone FMM descent, no diagonal corner cutting, "
                    "then line-checked world-frame waypoints"
                ),
            },
            "planner_config": asdict(self.config),
            "resolution_m": self.resolution_m,
            "grid_axes": "cell=[row(+world Y), column(+world X)]",
            "origin_world_xy": self.origin_world_xy.tolist(),
            "start_cell_rc": list(self.start_cell_rc),
            "goal_cell_rc": list(self.goal_cell_rc),
            "start_world_xy": self.start_world_xy.tolist(),
            "goal_world_xy": self.goal_world_xy.tolist(),
            "start_distance_cells": self.start_distance_cells,
            "start_distance_m": self.start_distance_cells * self.resolution_m,
            "distance_field_finite_min_cells": (
                float(np.min(finite)) if finite.size else None
            ),
            "distance_field_finite_max_cells": (
                float(np.max(finite)) if finite.size else None
            ),
            "lavira_short_term_goal_cell_rc": list(
                self.lavira_short_term_goal_cell_rc
            ),
            "lavira_short_term_goal_world_xy": (
                self.lavira_short_term_goal_world_xy.tolist()
            ),
            "lavira_short_term_goal_line_safe": (
                self.lavira_short_term_goal_line_safe
            ),
            "path_cell_count": int(self.path_cells_rc.shape[0]),
            "path_length_m": self.path_length_m,
            "path_cells_rc": self.path_cells_rc.tolist(),
            "path_world_xy": self.path_world_xy.tolist(),
            "waypoint_count": int(self.waypoint_cells_rc.shape[0]),
            "waypoint_cells_rc": self.waypoint_cells_rc.tolist(),
            "waypoints_world_xy": self.waypoints_world_xy.tolist(),
            "distance_field_file": "fmm_distance.npy",
            "visualization_file": "fmm_path.png",
            "note": (
                "The path is validated against the current connected traversable "
                "snapshot. It has not been sent to pure-pursuit or the G1 policy."
            ),
        }


def fmm_planner_config_from_args(args_cli) -> FMMPlannerConfig:
    return FMMPlannerConfig(
        step_size_cells=int(args_cli.fmm_step_size_cells),
        goal_tolerance_cells=int(args_cli.fmm_goal_tolerance_cells),
        waypoint_spacing_m=float(args_cli.fmm_waypoint_spacing_m),
        max_path_steps=int(args_cli.fmm_max_path_steps),
    ).validated()


def build_fmm_plan(
    grid_map: NavigationGridMap,
    config: FMMPlannerConfig,
) -> FMMPlan:
    """Compute a LaViRA FMM field and a safe world-frame polyline."""
    config = config.validated()
    if grid_map.safe_target_cell_rc is None or grid_map.safe_target_world_xy is None:
        raise FMMPlanningError("Navigation map has no safe target for FMM.")

    traversable = np.asarray(grid_map.traversable, dtype=bool)
    if traversable.ndim != 2:
        raise ValueError(f"FMM traversable grid must be 2-D, got {traversable.shape}.")
    start = _validated_cell(grid_map.robot_cell_rc, traversable.shape, "start")
    goal = _validated_cell(grid_map.safe_target_cell_rc, traversable.shape, "goal")
    if not traversable[start]:
        raise FMMPlanningError(f"FMM start cell {start} is not traversable.")
    if not traversable[goal]:
        raise FMMPlanningError(f"FMM goal cell {goal} is not traversable.")

    distance_field = compute_lavira_fmm_distance(traversable, goal)
    start_distance = float(distance_field[start])
    if not np.isfinite(start_distance):
        raise FMMPlanningError(
            f"FMM goal {goal} is unreachable from start {start} on this map."
        )

    dense_path = extract_monotone_fmm_path(
        distance_field,
        traversable,
        start,
        goal,
        goal_tolerance_cells=config.goal_tolerance_cells,
        max_steps=config.max_path_steps,
        descent_epsilon_cells=config.descent_epsilon_cells,
    )
    spacing_cells = max(
        1, int(round(config.waypoint_spacing_m / grid_map.resolution_m))
    )
    waypoint_cells = _line_checked_waypoints(
        dense_path, traversable, spacing_cells
    )
    short_term_goal = select_lavira_short_term_goal(
        distance_field,
        traversable,
        start,
        step_size_cells=config.step_size_cells,
        goal_cell_rc=goal,
    )

    path_world = _cells_to_world(
        dense_path, grid_map.origin_world_xy, grid_map.resolution_m
    )
    waypoint_world = _cells_to_world(
        waypoint_cells, grid_map.origin_world_xy, grid_map.resolution_m
    )
    # Preserve the measured root location at the start instead of replacing it
    # with the center of its containing cell.
    path_world[0] = np.asarray(grid_map.robot_world_xy, dtype=np.float64)
    waypoint_world[0] = np.asarray(grid_map.robot_world_xy, dtype=np.float64)
    path_world[-1] = np.asarray(grid_map.safe_target_world_xy, dtype=np.float64)
    waypoint_world[-1] = np.asarray(grid_map.safe_target_world_xy, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(path_world, axis=0), axis=1)

    return FMMPlan(
        config=config,
        bundle_id=int(grid_map.bundle_id),
        sim_step=int(grid_map.sim_step),
        resolution_m=float(grid_map.resolution_m),
        origin_world_xy=np.asarray(grid_map.origin_world_xy, dtype=np.float64),
        start_cell_rc=start,
        goal_cell_rc=goal,
        start_world_xy=np.asarray(grid_map.robot_world_xy, dtype=np.float64),
        goal_world_xy=np.asarray(grid_map.safe_target_world_xy, dtype=np.float64),
        distance_field_cells=distance_field,
        start_distance_cells=start_distance,
        lavira_short_term_goal_cell_rc=short_term_goal,
        lavira_short_term_goal_world_xy=grid_cell_to_world_xy(
            short_term_goal,
            grid_map.origin_world_xy,
            grid_map.resolution_m,
        ),
        lavira_short_term_goal_line_safe=_line_is_traversable(
            start, short_term_goal, traversable
        ),
        path_cells_rc=dense_path,
        path_world_xy=path_world,
        waypoint_cells_rc=waypoint_cells,
        waypoints_world_xy=waypoint_world,
        path_length_m=float(np.sum(segment_lengths)),
    )


def compute_lavira_fmm_distance(
    traversable: np.ndarray,
    goal_cell_rc: tuple[int, int],
) -> np.ndarray:
    """Port of LaViRA ``set_goal`` with masked obstacles and ``dx=1``."""
    try:
        import skfmm
    except ImportError as exc:
        raise RuntimeError(
            "scikit-fmm is required for --lavira_fmm_probe; install it in the "
            "same Isaac Sim environment with `python -m pip install scikit-fmm`."
        ) from exc

    traversable = np.asarray(traversable, dtype=bool)
    goal = _validated_cell(goal_cell_rc, traversable.shape, "goal")
    if not traversable[goal]:
        raise FMMPlanningError(f"FMM goal cell {goal} is not traversable.")

    # This intentionally mirrors LaViRA: free cells are +1, obstacles are
    # masked, and the goal is the zero level set. Distances remain in cells.
    phi = np.ma.MaskedArray(
        np.ones(traversable.shape, dtype=np.float64),
        mask=~traversable,
    )
    phi[goal] = 0.0
    try:
        distance_ma = skfmm.distance(phi, dx=1)
    except (ValueError, RuntimeError) as exc:
        raise FMMPlanningError(f"skfmm.distance failed: {exc}") from exc
    distance = np.asarray(
        np.ma.filled(distance_ma, np.inf), dtype=np.float64
    )
    distance[~traversable] = np.inf
    return distance


def lavira_local_ring_mask(step_size_cells: int = 5) -> np.ndarray:
    """Exact integer-position counterpart of LaViRA ``get_mask``."""
    step_size = int(step_size_cells)
    if step_size <= 0:
        raise ValueError("LaViRA local ring step size must be positive.")
    size = step_size * 2 + 1
    mask = np.zeros((size, size), dtype=bool)
    center = size // 2
    for row in range(size):
        for col in range(size):
            radius_sq = ((row + 0.5) - center) ** 2 + ((col + 0.5) - center) ** 2
            if (step_size - 1) ** 2 < radius_sq <= step_size**2:
                mask[row, col] = True
    mask[center, center] = True
    return mask


def select_lavira_short_term_goal(
    distance_field_cells: np.ndarray,
    traversable: np.ndarray,
    agent_cell_rc: tuple[int, int],
    *,
    step_size_cells: int = 5,
    goal_cell_rc: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Select LaViRA's local-ring minimum, stopping on a nearby safe goal.

    The nearby-goal branch corrects the original implementation's overwritten
    ``stop`` flag: without it, a goal inside the five-cell ring can be skipped
    in favor of a point beyond the destination.
    """
    distance = np.asarray(distance_field_cells, dtype=np.float64)
    traversable = np.asarray(traversable, dtype=bool)
    agent = _validated_cell(agent_cell_rc, distance.shape, "agent")
    if distance.shape != traversable.shape:
        raise ValueError("FMM distance and traversable shapes do not match.")

    if goal_cell_rc is not None:
        goal = _validated_cell(goal_cell_rc, distance.shape, "goal")
        if (
            math.hypot(goal[0] - agent[0], goal[1] - agent[1])
            <= int(step_size_cells)
            and _line_is_traversable(agent, goal, traversable)
        ):
            return goal

    ring = lavira_local_ring_mask(step_size_cells)
    radius = int(step_size_cells)
    best = agent
    best_distance = float(distance[agent])
    for local_row, local_col in np.argwhere(ring):
        row = agent[0] + int(local_row) - radius
        col = agent[1] + int(local_col) - radius
        if not (0 <= row < distance.shape[0] and 0 <= col < distance.shape[1]):
            continue
        candidate = (row, col)
        candidate_distance = float(distance[candidate])
        if traversable[candidate] and candidate_distance < best_distance:
            best = candidate
            best_distance = candidate_distance
    return best


def extract_monotone_fmm_path(
    distance_field_cells: np.ndarray,
    traversable: np.ndarray,
    start_cell_rc: tuple[int, int],
    goal_cell_rc: tuple[int, int],
    *,
    goal_tolerance_cells: int = 1,
    max_steps: int = 20_000,
    descent_epsilon_cells: float = 1.0e-6,
) -> np.ndarray:
    """Follow the FMM field through safe 8-neighbors until the goal is reached."""
    distance = np.asarray(distance_field_cells, dtype=np.float64)
    traversable = np.asarray(traversable, dtype=bool)
    if distance.shape != traversable.shape:
        raise ValueError("FMM distance and traversable shapes do not match.")
    start = _validated_cell(start_cell_rc, distance.shape, "start")
    goal = _validated_cell(goal_cell_rc, distance.shape, "goal")
    if not traversable[start] or not traversable[goal]:
        raise FMMPlanningError("FMM path endpoints must both be traversable.")
    if not np.isfinite(distance[start]):
        raise FMMPlanningError("FMM start has no finite distance to the goal.")

    path: list[tuple[int, int]] = [start]
    current = start
    for _ in range(int(max_steps)):
        if _chebyshev_distance(current, goal) <= int(goal_tolerance_cells):
            connector = _bresenham_cells(current, goal)
            if all(traversable[cell] for cell in connector) and _line_is_traversable(
                current, goal, traversable
            ):
                path.extend(connector[1:])
                return np.asarray(path, dtype=np.int32)

        current_distance = float(distance[current])
        candidates: list[tuple[float, float, tuple[int, int]]] = []
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                candidate = (current[0] + delta_row, current[1] + delta_col)
                if not _cell_inside(candidate, distance.shape):
                    continue
                if not traversable[candidate]:
                    continue
                if delta_row and delta_col:
                    # Prevent a point robot from squeezing diagonally between
                    # two inflated obstacle cells.
                    if not traversable[current[0] + delta_row, current[1]]:
                        continue
                    if not traversable[current[0], current[1] + delta_col]:
                        continue
                candidate_distance = float(distance[candidate])
                if not np.isfinite(candidate_distance):
                    continue
                if candidate_distance >= current_distance - descent_epsilon_cells:
                    continue
                candidates.append(
                    (
                        candidate_distance,
                        math.hypot(candidate[0] - goal[0], candidate[1] - goal[1]),
                        candidate,
                    )
                )
        if not candidates:
            raise FMMPlanningError(
                "FMM path extraction reached a local plateau at "
                f"{current} (distance={current_distance:.6f} cells)."
            )
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        current = candidates[0][2]
        path.append(current)

    raise FMMPlanningError(
        f"FMM path exceeded max_path_steps={max_steps} before reaching {goal}."
    )


def save_fmm_plan_debug(
    output_dir: Path,
    grid_map: NavigationGridMap,
    plan: FMMPlan,
) -> dict[str, str]:
    """Save metadata, the raw distance field and a navigation-map overlay."""
    import cv2

    output_dir = Path(output_dir)
    json_name = "fmm_plan.json"
    distance_name = "fmm_distance.npy"
    image_name = "fmm_path.png"
    (output_dir / json_name).write_text(
        json.dumps(plan.to_metadata(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(output_dir / distance_name, plan.distance_field_cells.astype(np.float32))
    visualization = fmm_plan_visualization(grid_map, plan)
    if not cv2.imwrite(str(output_dir / image_name), visualization):
        raise OSError(f"Failed to write {output_dir / image_name}.")
    return {
        "plan_json": json_name,
        "distance_npy": distance_name,
        "visualization_png": image_name,
    }


def fmm_plan_visualization(
    grid_map: NavigationGridMap,
    plan: FMMPlan,
) -> np.ndarray:
    """Overlay the dense FMM path and safe waypoints on the map debug image."""
    import cv2

    image = navigation_map_visualization(grid_map)
    height = image.shape[0]

    def display_xy(cell_rc: tuple[int, int] | np.ndarray) -> tuple[int, int]:
        row, col = int(cell_rc[0]), int(cell_rc[1])
        return col, height - 1 - row

    path_points = np.asarray(
        [display_xy(cell) for cell in plan.path_cells_rc], dtype=np.int32
    )
    if path_points.shape[0] >= 2:
        cv2.polylines(
            image,
            [path_points.reshape(-1, 1, 2)],
            False,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    for cell in plan.waypoint_cells_rc[1:-1]:
        cv2.circle(image, display_xy(cell), 2, (0, 255, 255), -1)
    cv2.drawMarker(
        image,
        display_xy(plan.lavira_short_term_goal_cell_rc),
        (0, 165, 255),
        markerType=cv2.MARKER_DIAMOND,
        markerSize=8,
        thickness=2,
    )
    cv2.circle(image, display_xy(plan.start_cell_rc), 5, (255, 80, 0), -1)
    cv2.circle(image, display_xy(plan.goal_cell_rc), 6, (0, 180, 0), -1)
    cv2.circle(image, display_xy(plan.goal_cell_rc), 7, (255, 255, 255), 1)
    cv2.putText(
        image,
        (
            f"FMM path={plan.path_length_m:.2f}m "
            f"cells={plan.path_cells_rc.shape[0]} "
            f"waypoints={plan.waypoint_cells_rc.shape[0]}"
        ),
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def _line_checked_waypoints(
    dense_path: np.ndarray,
    traversable: np.ndarray,
    spacing_cells: int,
) -> np.ndarray:
    if dense_path.ndim != 2 or dense_path.shape[1] != 2 or not len(dense_path):
        raise ValueError("Dense FMM path must have shape Nx2 and be non-empty.")
    selected = [0]
    current_index = 0
    final_index = len(dense_path) - 1
    while current_index < final_index:
        candidate_index = min(current_index + int(spacing_cells), final_index)
        while candidate_index > current_index + 1 and not _line_is_traversable(
            tuple(dense_path[current_index]),
            tuple(dense_path[candidate_index]),
            traversable,
        ):
            candidate_index -= 1
        if candidate_index <= current_index:
            raise FMMPlanningError("Could not construct safe FMM waypoint segments.")
        selected.append(candidate_index)
        current_index = candidate_index
    return np.asarray(dense_path[selected], dtype=np.int32)


def _line_is_traversable(
    start: tuple[int, int],
    end: tuple[int, int],
    traversable: np.ndarray,
) -> bool:
    cells = _bresenham_cells(start, end)
    previous = cells[0]
    for cell in cells:
        if not _cell_inside(cell, traversable.shape) or not traversable[cell]:
            return False
        delta_row = cell[0] - previous[0]
        delta_col = cell[1] - previous[1]
        if delta_row and delta_col:
            if not traversable[previous[0] + delta_row, previous[1]]:
                return False
            if not traversable[previous[0], previous[1] + delta_col]:
                return False
        previous = cell
    return True


def _bresenham_cells(
    start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    row0, col0 = int(start[0]), int(start[1])
    row1, col1 = int(end[0]), int(end[1])
    x0, y0, x1, y1 = col0, row0, col1, row1
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += sx
        if twice_error <= dx:
            error += dx
            y0 += sy
    return cells


def _cells_to_world(
    cells_rc: np.ndarray,
    origin_world_xy: np.ndarray,
    resolution_m: float,
) -> np.ndarray:
    return np.asarray(
        [
            grid_cell_to_world_xy(
                (int(cell[0]), int(cell[1])), origin_world_xy, resolution_m
            )
            for cell in cells_rc
        ],
        dtype=np.float64,
    )


def _validated_cell(
    cell_rc: tuple[int, int],
    shape: tuple[int, int],
    name: str,
) -> tuple[int, int]:
    if len(cell_rc) != 2:
        raise ValueError(f"FMM {name} cell must contain row and column.")
    cell = (int(cell_rc[0]), int(cell_rc[1]))
    if not _cell_inside(cell, shape):
        raise FMMPlanningError(
            f"FMM {name} cell {cell} is outside grid shape {shape}."
        )
    return cell


def _cell_inside(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _chebyshev_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
