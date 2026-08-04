from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .odometry import (
    NullOdometryProvider,
    OdometryProvider,
    Pose2D,
    fixed_points_to_local,
    local_points_to_fixed,
)


def _validated_trajectory(trajectory: np.ndarray) -> np.ndarray:
    path = np.asarray(trajectory, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] < 2 or path.shape[0] < 2:
        raise ValueError(f"iPlanner trajectory must be Nx2/Nx3 with N>=2, got {path.shape}.")
    if not np.all(np.isfinite(path)):
        raise ValueError("iPlanner trajectory contains non-finite values.")
    if path.shape[1] == 2:
        path = np.column_stack([path, np.zeros(path.shape[0], dtype=np.float64)])
    return path[:, :3].copy()


def truncate_trajectory_for_safety(
    trajectory: np.ndarray,
    safe_distance_m: float,
) -> np.ndarray | None:
    """Stop before the target using Uni-LaViRA's path-length convention."""
    path = _validated_trajectory(trajectory)
    if safe_distance_m < 0.0:
        raise ValueError("Safe distance must be non-negative.")
    segment_lengths = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    target_length = total_length - float(safe_distance_m)
    if target_length <= 0.0:
        return None
    cumulative = 0.0
    cutoff = path.shape[0] - 1
    for index, length in enumerate(segment_lengths):
        cumulative += float(length)
        if cumulative >= target_length:
            cutoff = index + 1
            break
    result = path[: cutoff + 1].copy()
    return result if result.shape[0] >= 2 else None


@dataclass(frozen=True)
class LocalFollowerConfig:
    target_speed_m_s: float = 0.3
    lookahead_m: float = 0.5
    max_forward_speed_m_s: float = 0.4
    max_yaw_speed_rad_s: float = 0.5
    goal_tolerance_m: float = 1.0
    blind_yaw_radius_m: float = 2.0
    yaw_bias_rad_s: float = 0.0
    replan_interval_s: float = 0.1
    dead_reckoning_linear_scale: float = 1.0
    dead_reckoning_angular_scale: float = 1.0

    def validated(self) -> "LocalFollowerConfig":
        positive = (
            self.target_speed_m_s,
            self.lookahead_m,
            self.max_forward_speed_m_s,
            self.max_yaw_speed_rad_s,
            self.goal_tolerance_m,
            self.replan_interval_s,
            self.dead_reckoning_linear_scale,
            self.dead_reckoning_angular_scale,
        )
        if any(not math.isfinite(v) or v <= 0.0 for v in positive):
            raise ValueError("Local follower positive parameters must be finite and > 0.")
        if self.blind_yaw_radius_m < 0.0:
            raise ValueError("Blind-yaw radius must be non-negative.")
        return self


@dataclass(frozen=True)
class LocalFollowerOutput:
    command: np.ndarray
    reached: bool
    abort_reason: str | None
    goal_local_xy: np.ndarray
    target_local_xy: np.ndarray | None
    distance_to_goal_m: float
    alpha_rad: float


class _LocalReferenceTracker:
    """Keep path/goal local using odometry when available, command integration otherwise."""

    def __init__(
        self,
        path: np.ndarray,
        goal_local_xy: np.ndarray,
        odometry: OdometryProvider | None,
        config: LocalFollowerConfig,
    ):
        self.config = config
        self.odometry = odometry or NullOdometryProvider()
        self.path_local = _validated_trajectory(path)
        self.goal_local = np.asarray(goal_local_xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(self.goal_local)):
            raise ValueError("Local goal must be finite.")

        pose = self.odometry.get_pose()
        self.uses_odometry = pose is not None
        self.path_fixed_xy: np.ndarray | None = None
        self.goal_fixed_xy: np.ndarray | None = None
        if pose is not None:
            pose.validated()
            self.path_fixed_xy = local_points_to_fixed(self.path_local[:, :2], pose)
            self.goal_fixed_xy = local_points_to_fixed(
                self.goal_local.reshape(1, 2), pose
            )[0]

    def advance(self, dt: float, applied_command: np.ndarray) -> str | None:
        if self.uses_odometry:
            pose = self.odometry.get_pose()
            if pose is None:
                return "odometry became unavailable during local trajectory execution"
            try:
                pose.validated()
            except ValueError as exc:
                return str(exc)
            self.path_local[:, :2] = fixed_points_to_local(self.path_fixed_xy, pose)
            self.goal_local = fixed_points_to_local(
                self.goal_fixed_xy.reshape(1, 2), pose
            )[0]
            return None

        command = np.asarray(applied_command, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(command)):
            return "applied velocity command contains non-finite values"
        if dt <= 0.0:
            return None
        dx = command[0] * dt * self.config.dead_reckoning_linear_scale
        dy = command[1] * dt * self.config.dead_reckoning_linear_scale
        d_yaw = command[2] * dt * self.config.dead_reckoning_angular_scale
        c, s = math.cos(-d_yaw), math.sin(-d_yaw)
        inverse_delta_rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        translation = np.array([dx, dy], dtype=np.float64)
        self.path_local[:, :2] = (
            self.path_local[:, :2] - translation
        ) @ inverse_delta_rotation.T
        self.goal_local = (
            self.goal_local - translation
        ) @ inverse_delta_rotation.T
        return None

    def replace_path(self, path: np.ndarray) -> None:
        new_path = _validated_trajectory(path)
        if self.uses_odometry:
            pose = self.odometry.get_pose()
            if pose is None:
                raise RuntimeError("Cannot replace path after odometry became unavailable.")
            pose.validated()
            self.path_fixed_xy = local_points_to_fixed(new_path[:, :2], pose)
        self.path_local = new_path


class LocalTrajectoryFollower:
    """Uni-LaViRA G1 local pure pursuit with optional odometry."""

    def __init__(
        self,
        config: LocalFollowerConfig,
        odometry: OdometryProvider | None = None,
    ):
        self.config = config.validated()
        self.odometry = odometry
        self._reference: _LocalReferenceTracker | None = None
        self.replan_elapsed_s = 0.0

    @property
    def active(self) -> bool:
        return self._reference is not None

    @property
    def current_goal_local_xy(self) -> np.ndarray:
        if self._reference is None:
            raise RuntimeError("Local follower is not active.")
        return self._reference.goal_local.copy()

    def start(self, path: np.ndarray, goal_local_xy: np.ndarray) -> None:
        self._reference = _LocalReferenceTracker(
            path,
            goal_local_xy,
            self.odometry,
            self.config,
        )
        self.replan_elapsed_s = 0.0

    def stop(self) -> None:
        self._reference = None
        self.replan_elapsed_s = 0.0

    def needs_replan(self) -> bool:
        return self.active and self.replan_elapsed_s >= self.config.replan_interval_s

    def replace_path(self, path: np.ndarray) -> None:
        if self._reference is None:
            raise RuntimeError("Cannot replace an inactive local path.")
        self._reference.replace_path(path)
        self.replan_elapsed_s = 0.0

    def defer_replan(self) -> None:
        """Start a new replan interval after a recoverable planner failure."""
        if self.active:
            self.replan_elapsed_s = 0.0

    def update(
        self,
        dt: float,
        applied_command: np.ndarray,
    ) -> LocalFollowerOutput:
        if self._reference is None:
            return LocalFollowerOutput(
                command=np.zeros(3, dtype=np.float64),
                reached=False,
                abort_reason="local follower is inactive",
                goal_local_xy=np.zeros(2, dtype=np.float64),
                target_local_xy=None,
                distance_to_goal_m=math.inf,
                alpha_rad=0.0,
            )
        self.replan_elapsed_s += max(float(dt), 0.0)
        abort_reason = self._reference.advance(dt, applied_command)
        goal = self._reference.goal_local.copy()
        distance_to_goal = float(np.linalg.norm(goal))
        if abort_reason is not None:
            return LocalFollowerOutput(
                np.zeros(3), False, abort_reason, goal, None, distance_to_goal, 0.0
            )
        # Match Uni-LaViRA G1: the passed-goal guard is only a dead-reckoning
        # safeguard.  With odometry, a laterally offset goal can temporarily
        # move behind the robot's x axis while pure pursuit is still reducing
        # the true 2-D distance, so aborting on x alone is incorrect.
        if not self._reference.uses_odometry and goal[0] < -0.1:
            return LocalFollowerOutput(
                np.zeros(3),
                False,
                (
                    "local goal was passed without odometry "
                    f"(x={goal[0]:.3f}m, y={goal[1]:.3f}m, "
                    f"distance={distance_to_goal:.3f}m)"
                ),
                goal,
                None,
                distance_to_goal,
                0.0,
            )
        if distance_to_goal < self.config.goal_tolerance_m:
            return LocalFollowerOutput(
                np.zeros(3), True, None, goal, None, distance_to_goal, 0.0
            )

        path = self._reference.path_local
        target = path[-1, :2]
        for point in path:
            if float(np.linalg.norm(point[:2])) > self.config.lookahead_m:
                target = point[:2]
                break
        target_distance = max(float(np.linalg.norm(target)), 0.1)
        alpha = float(math.atan2(float(target[1]), float(target[0])))

        if distance_to_goal < self.config.blind_yaw_radius_m:
            command_yaw = 0.0
            steering_alpha = 0.0
        else:
            steering_alpha = alpha
            command_yaw = (
                2.0
                * self.config.target_speed_m_s
                * math.sin(alpha)
                / target_distance
            )
        command_yaw = float(
            np.clip(
                command_yaw + self.config.yaw_bias_rad_s,
                -self.config.max_yaw_speed_rad_s,
                self.config.max_yaw_speed_rad_s,
            )
        )
        command_forward = max(
            0.1,
            self.config.target_speed_m_s
            * (1.0 - abs(steering_alpha) / math.pi),
        )
        command_forward = min(
            command_forward, self.config.max_forward_speed_m_s
        )
        command = np.array([command_forward, 0.0, command_yaw], dtype=np.float64)
        return LocalFollowerOutput(
            command=command,
            reached=False,
            abort_reason=None,
            goal_local_xy=goal,
            target_local_xy=np.asarray(target, dtype=np.float64).copy(),
            distance_to_goal_m=distance_to_goal,
            alpha_rad=alpha,
        )
