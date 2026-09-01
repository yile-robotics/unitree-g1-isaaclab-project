from __future__ import annotations

from dataclasses import dataclass
import math
import time

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
    if path.shape[1] == 2:
        path = np.column_stack([path, np.zeros(path.shape[0], dtype=np.float64)])
    return path[:, :3].copy()


def _path_projection_lookahead(
    path_xy: np.ndarray,
    robot_xy: np.ndarray,
    lookahead_m: float,
) -> tuple[np.ndarray, float, float]:
    """Project onto a polyline and interpolate a lookahead target.

    Returns ``(target_xy, tangent_yaw, cross_track_error_m)`` in the same
    coordinate frame as ``path_xy``. This is the geometric core used by the
    older ``isaacsim_goal_tracking`` waypoint follower.
    """

    path = np.asarray(path_xy, dtype=np.float64)
    robot = np.asarray(robot_xy, dtype=np.float64).reshape(2)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 2:
        raise ValueError("Projected Kp path must be Nx2 with N>=2.")

    segment_vectors = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    total_length = float(np.sum(segment_lengths))
    if not math.isfinite(total_length) or total_length <= 1.0e-5:
        raise ValueError("Projected Kp path has no usable length.")

    best_progress = 0.0
    best_distance = math.inf
    accumulated = 0.0
    for index, segment_length in enumerate(segment_lengths):
        length = float(segment_length)
        if length <= 1.0e-5:
            continue
        start = path[index]
        segment = segment_vectors[index]
        projection = float(
            np.clip(
                np.dot(robot - start, segment) / (length * length),
                0.0,
                1.0,
            )
        )
        closest = start + projection * segment
        distance = float(np.linalg.norm(robot - closest))
        if distance < best_distance:
            best_distance = distance
            best_progress = accumulated + projection * length
        accumulated += length

    target_progress = min(
        best_progress + max(float(lookahead_m), 0.0), total_length
    )
    accumulated = 0.0
    last_tangent = 0.0
    for index, segment_length in enumerate(segment_lengths):
        length = float(segment_length)
        if length <= 1.0e-5:
            continue
        segment = segment_vectors[index]
        last_tangent = math.atan2(float(segment[1]), float(segment[0]))
        if (
            target_progress <= accumulated + length
            or index == len(segment_lengths) - 1
        ):
            fraction = float(
                np.clip((target_progress - accumulated) / length, 0.0, 1.0)
            )
            return (
                path[index] + fraction * segment,
                last_tangent,
                best_distance,
            )
        accumulated += length

    return path[-1].copy(), last_tangent, best_distance

#按已有 waypoint 截断，而不是连续精确插值截断，在目标前 safe_distance_m=0.5米处停止
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
    # ``pure_pursuit`` keeps the frozen Uni-LaViRA-style controller unchanged;
    # ``kp`` reuses the older odometry path follower's proportional tracking law.
    tracking_controller: str = "pure_pursuit"
    #期望前进速度，默认 0.3 m/s
    target_speed_m_s: float = 0.3
    #前视距离，默认 0.5 m
    lookahead_m: float = 0.5
    #最大前进速度，防止最终算出的速度超过 0.4 m/s
    max_forward_speed_m_s: float = 0.4
    #最大旋转速度，防止最终算出的旋转速度超过 0.5 rad/s
    max_yaw_speed_rad_s: float = 0.5
    #目标容忍度，默认 1.0 m
    goal_tolerance_m: float = 1.0
    # odometry 模式下，曾进入 tolerance+0.2m 后才启用越过目标保护。
    odom_pass_guard_arm_margin_m: float = 0.2
    # 当前距离比历史最近距离增加 0.2m，才视为开始远离目标。
    odom_pass_guard_distance_margin_m: float = 0.2
    # 持续远离 0.3s 后，按已到达当前局部目标正常完成。
    odom_pass_guard_confirm_s: float = 0.3
    # 当前 G3/scene_200 固定测试值：进入 1.5m 后停止继续 yaw 修正。
    blind_yaw_radius_m: float = 1.5
    # 可选 Kp 控制器参数。Kp 分支复用旧 goal_tracking 的全向路径跟踪律。
    kp_xy: float = 0.7
    kp_yaw: float = 1.0
    kp_slow_radius_m: float = 1.0
    kp_yaw_deadband_rad: float = math.radians(10.0)
    kp_max_lateral_speed_m_s: float = 0.12
    kp_max_yaw_speed_rad_s: float = 0.35
    #旋转偏置，默认 0.0 rad/s
    yaw_bias_rad_s: float = 0.0
    #重新规划间隔，默认 0.1 s
    replan_interval_s: float = 0.1
    # 死算里程线性缩放；Uni-LaViRA G1 真机代码固定使用 0.7。
    dead_reckoning_linear_scale: float = 0.7
    # 死算里程角度缩放；Uni-LaViRA G1 真机代码固定使用 0.8。
    dead_reckoning_angular_scale: float = 0.8

    def validated(self) -> "LocalFollowerConfig":
        positive = (
            self.target_speed_m_s,
            self.lookahead_m,
            self.max_forward_speed_m_s,
            self.max_yaw_speed_rad_s,
            self.goal_tolerance_m,
            self.odom_pass_guard_confirm_s,
            self.kp_xy,
            self.kp_yaw,
            self.kp_slow_radius_m,
            self.kp_max_lateral_speed_m_s,
            self.kp_max_yaw_speed_rad_s,
            self.replan_interval_s,
            self.dead_reckoning_linear_scale,
            self.dead_reckoning_angular_scale,
        )
        if any(not math.isfinite(v) or v <= 0.0 for v in positive):
            raise ValueError("Local follower positive parameters must be finite and > 0.")
        if self.blind_yaw_radius_m < 0.0:
            raise ValueError("Blind-yaw radius must be non-negative.")
        if self.tracking_controller not in {"pure_pursuit", "kp"}:
            raise ValueError(
                "Tracking controller must be 'pure_pursuit' or 'kp'."
            )
        if (
            not math.isfinite(self.kp_yaw_deadband_rad)
            or self.kp_yaw_deadband_rad < 0.0
            or self.kp_yaw_deadband_rad >= math.pi
        ):
            raise ValueError(
                "Kp yaw deadband must be finite and in [0, pi)."
            )
        if (
            not math.isfinite(self.odom_pass_guard_arm_margin_m)
            or self.odom_pass_guard_arm_margin_m < 0.0
            or not math.isfinite(self.odom_pass_guard_distance_margin_m)
            or self.odom_pass_guard_distance_margin_m <= 0.0
        ):
            raise ValueError(
                "Odometry passed-goal guard margins must be finite; "
                "arm margin must be >= 0 and distance margin must be > 0."
            )
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
        goal_tolerance_m: float | None = None,
    ):
        self.config = config
        self.odometry = odometry or NullOdometryProvider()
        self.path_local = _validated_trajectory(path)
        self.goal_local = np.asarray(goal_local_xy, dtype=np.float64).reshape(2)
        self.goal_tolerance_m = (
            float(config.goal_tolerance_m)
            if goal_tolerance_m is None
            else float(goal_tolerance_m)
        )
        if (
            not math.isfinite(self.goal_tolerance_m)
            or self.goal_tolerance_m <= 0.0
        ):
            raise ValueError("Active goal tolerance must be finite and positive.")

        pose = self.odometry.get_pose()
        self.uses_odometry = pose is not None
        self.goal_fixed_xy: np.ndarray | None = None
        self.path_fixed_xy: np.ndarray | None = None
        if pose is not None:
            self.goal_fixed_xy = local_points_to_fixed(
                self.goal_local.reshape(1, 2), pose
            )[0]
            self.path_fixed_xy = local_points_to_fixed(
                self.path_local[:, :2], pose
            )
    # 有 odometry 时，Uni 只更新局部目标，不变换两次重规划之间的旧局部路径；
    # 无 odometry 时，使用上一条速度命令同时更新局部路径和目标。
    def advance(self, dt: float, applied_command: np.ndarray) -> str | None:
        if self.uses_odometry:
            pose = self.odometry.get_pose()
            # Match Uni: keep the old local path untouched between replans and
            # only refresh the fixed global goal when a pose is available.
            if pose is not None:
                self.goal_local = fixed_points_to_local(
                    self.goal_fixed_xy.reshape(1, 2), pose
                )[0]
            return None

        command = np.asarray(applied_command, dtype=np.float64).reshape(3)
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
    # 新规划已经以机器人当前位姿为局部原点，因此直接整体替换并由跟随器重置索引。
    def replace_path(self, path: np.ndarray) -> None:
        self.path_local = _validated_trajectory(path)
        if self.uses_odometry:
            pose = self.odometry.get_pose()
            if pose is not None:
                self.path_fixed_xy = local_points_to_fixed(
                    self.path_local[:, :2], pose
                )

    def kp_path_and_robot(self) -> tuple[np.ndarray, np.ndarray, Pose2D | None]:
        """Return the Kp path and robot point in one consistent frame."""

        if self.uses_odometry:
            pose = self.odometry.get_pose()
            if pose is not None and self.path_fixed_xy is not None:
                return (
                    self.path_fixed_xy.copy(),
                    np.array([pose.x, pose.y], dtype=np.float64),
                    pose,
                )
        return (
            self.path_local[:, :2].copy(),
            np.zeros(2, dtype=np.float64),
            None,
        )


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
        # Pure Pursuit 只允许沿当前路径向前推进。若每帧都从 path[0] 重新查找，
        # 机器人走过的旧点在身后重新变远后，可能再次被误选为 lookahead 目标。
        self.current_idx = 0
        self.last_replan_time_s = time.time()
        self.best_goal_distance_m = math.inf
        self.goal_distance_increasing_s = 0.0
        self.goal_pass_guard_armed = False

    @property
    def active(self) -> bool:
        return self._reference is not None

    @property
    def current_goal_local_xy(self) -> np.ndarray:
        if self._reference is None:
            raise RuntimeError("Local follower is not active.")
        return self._reference.goal_local.copy()

    def start(
        self,
        path: np.ndarray,
        goal_local_xy: np.ndarray,
        *,
        goal_tolerance_m: float | None = None,
    ) -> None:
        self._reference = _LocalReferenceTracker(
            path,
            goal_local_xy,
            self.odometry,
            self.config,
            goal_tolerance_m,
        )
        self.current_idx = 0
        self.last_replan_time_s = time.time()
        self.best_goal_distance_m = math.inf
        self.goal_distance_increasing_s = 0.0
        self.goal_pass_guard_armed = False

    def stop(self) -> None:
        self._reference = None
        self.current_idx = 0
        self.best_goal_distance_m = math.inf
        self.goal_distance_increasing_s = 0.0
        self.goal_pass_guard_armed = False

    def needs_replan(self) -> bool:
        return self.active and (
            time.time() - self.last_replan_time_s > self.config.replan_interval_s
        )

    def replace_path(self, path: np.ndarray) -> None:
        if self._reference is None:
            raise RuntimeError("Cannot replace an inactive local path.")
        self._reference.replace_path(path)
        # iPlanner 的新路径以机器人当前位姿为原点，必须从新路径开头重新搜索。
        self.current_idx = 0

    def mark_replan_attempt(self, started_at_s: float) -> None:
        """Match Uni: remember the wall-clock timestamp from before planning."""

        if self.active:
            self.last_replan_time_s = float(started_at_s)

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
        abort_reason = self._reference.advance(dt, applied_command)
        goal = self._reference.goal_local.copy()
        distance_to_goal = float(np.linalg.norm(goal))
        if abort_reason is not None:
            return LocalFollowerOutput(
                np.zeros(3), False, abort_reason, goal, None, distance_to_goal, 0.0
            )
        if distance_to_goal < self.best_goal_distance_m:
            self.best_goal_distance_m = distance_to_goal
            self.goal_distance_increasing_s = 0.0
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
        if distance_to_goal < self._reference.goal_tolerance_m:
            return LocalFollowerOutput(
                np.zeros(3), True, None, goal, None, distance_to_goal, 0.0
            )

        # Isaac root pose and real SLAM odometry keep the local goal tied to a
        # fixed frame.  Once the robot has genuinely approached that goal,
        # detect a sustained increase in true 2-D distance and finish the
        # current local action instead of continuing forward past the target.
        if self._reference.uses_odometry:
            arm_distance_m = (
                self._reference.goal_tolerance_m
                + self.config.odom_pass_guard_arm_margin_m
            )
            if self.best_goal_distance_m <= arm_distance_m:
                self.goal_pass_guard_armed = True
            distance_increased = (
                distance_to_goal
                >= self.best_goal_distance_m
                + self.config.odom_pass_guard_distance_margin_m
            )
            if self.goal_pass_guard_armed and distance_increased:
                self.goal_distance_increasing_s += max(float(dt), 0.0)
            else:
                self.goal_distance_increasing_s = 0.0
            if (
                self.goal_distance_increasing_s
                >= self.config.odom_pass_guard_confirm_s
            ):
                print(
                    "[LOCAL-VLN] odometry passed-goal guard: "
                    f"best={self.best_goal_distance_m:.3f}m "
                    f"current={distance_to_goal:.3f}m; "
                    "completing the local action."
                )
                return LocalFollowerOutput(
                    np.zeros(3), True, None, goal, None, distance_to_goal, 0.0
                )

        if self.config.tracking_controller == "kp":
            # Port the older isaacsim_goal_tracking follower's geometry: find
            # the closest projection on the complete fixed-frame path, advance
            # by an interpolated lookahead distance, then convert that target
            # into the robot base frame and command vx/vy/wz proportionally.
            path_xy, robot_xy, pose = self._reference.kp_path_and_robot()
            target_in_path_frame, tangent_yaw, _cross_track = (
                _path_projection_lookahead(
                    path_xy,
                    robot_xy,
                    self.config.lookahead_m,
                )
            )
            if pose is not None:
                target = fixed_points_to_local(
                    target_in_path_frame.reshape(1, 2), pose
                )[0]
                alpha = math.atan2(
                    math.sin(tangent_yaw - pose.yaw),
                    math.cos(tangent_yaw - pose.yaw),
                )
            else:
                target = target_in_path_frame
                alpha = tangent_yaw

            command_yaw = self.config.kp_yaw * alpha
            yaw_limit = min(
                self.config.max_yaw_speed_rad_s,
                self.config.kp_max_yaw_speed_rad_s,
            )
            command_yaw = float(
                np.clip(command_yaw, -yaw_limit, yaw_limit)
            )
            command_yaw += self.config.yaw_bias_rad_s

            # Match the old follower's terminal slowdown. The G1 locomotion
            # policy is holonomic, but lateral speed is deliberately bounded
            # below its trained limit for the forward-camera deployment.
            speed_scale = max(
                0.15,
                min(1.0, distance_to_goal / self.config.kp_slow_radius_m),
            )
            command_forward = self.config.kp_xy * float(target[0]) * speed_scale
            command_forward = float(
                np.clip(
                    command_forward,
                    -self.config.max_forward_speed_m_s,
                    self.config.max_forward_speed_m_s,
                )
            )
            command_lateral = self.config.kp_xy * float(target[1]) * speed_scale
            command_lateral = float(
                np.clip(
                    command_lateral,
                    -self.config.kp_max_lateral_speed_m_s,
                    self.config.kp_max_lateral_speed_m_s,
                )
            )
        else:
            path = self._reference.path_local
            found_target = False
            for index in range(self.current_idx, len(path)):
                if float(np.linalg.norm(path[index, :2])) > self.config.lookahead_m:
                    self.current_idx = index
                    found_target = True
                    break
            if not found_target:
                self.current_idx = len(path) - 1
            target = path[self.current_idx, :2]
            target_distance = max(float(np.linalg.norm(target)), 0.1)
            alpha = float(math.atan2(float(target[1]), float(target[0])))
            # Frozen Pure-Pursuit branch.  Keeping this byte-for-byte behavior
            # separate makes the controller selectable and preserves existing
            # runs when ``tracking_controller`` is not explicitly set to kp.
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
            # Match Uni-LaViRA exactly: clamp the pursuit term first, then add
            # the configured mechanical yaw-bias without a second clamp.
            command_yaw = float(
                np.clip(
                    command_yaw,
                    -self.config.max_yaw_speed_rad_s,
                    self.config.max_yaw_speed_rad_s,
                )
            )
            command_yaw += self.config.yaw_bias_rad_s
            command_forward = max(
                0.1,
                self.config.target_speed_m_s
                * (1.0 - abs(steering_alpha) / math.pi),
            )
            command_forward = min(
                command_forward, self.config.max_forward_speed_m_s
            )
            command_lateral = 0.0
        command = np.array(
            [command_forward, command_lateral, command_yaw], dtype=np.float64
        )
        return LocalFollowerOutput(
            command=command,
            reached=False,
            abort_reason=None,
            goal_local_xy=goal,
            target_local_xy=np.asarray(target, dtype=np.float64).copy(),
            distance_to_goal_m=distance_to_goal,
            alpha_rad=alpha,
        )
