from __future__ import annotations

"""Waypoint path follower 和路径可视化。

这个模块做两件事：
1. 把世界坐标系下的 waypoint path 转换成机器人 base frame 下的速度命令
   `[vx, vy, wz]`，再交给 locomotion policy 跟踪。
2. 在 Isaac Sim stage 里画出路径、lookahead target 和一些场景清理辅助。

注意：这里不是规划器，也不直接控制关节。它只是一个轻量 path follower；
路径点可以来自命令行手写路径，也可以由本轮 FMM 世界路径动态装入。
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from .config import DEFAULT_HOUSE_WAYPOINTS
from .control import tensor_first


def wrap_to_pi(angle: float) -> float:
    """把角度规整到 [-pi, pi)，避免 yaw error 绕圈。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_wxyz_to_yaw(quat_wxyz: torch.Tensor) -> float:
    """从 IsaacLab root_quat_w 的 wxyz 四元数中取平面 yaw。"""
    qw, qx, qy, qz = [float(value) for value in quat_wxyz.detach().cpu()]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class Waypoint:
    """路径点：世界坐标 x/y 加目标 yaw。"""
    x: float
    y: float
    yaw: float


@dataclass
class RobotPose2D:
    """机器人当前 2D 位姿。valid=False 表示本帧没读到有效 root pose。"""
    x: float
    y: float
    yaw: float
    valid: bool = True


@dataclass
class PathTarget:
    """path follower 当前追踪的 lookahead target 和调试信息。"""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    cross_track_error: float = 0.0
    progress: float = 0.0
    total_length: float = 0.0
    segment_index: int = 0
    valid: bool = False


@dataclass(frozen=True)
class PreparedFMMPath:
    """经过执行前检查并转换为现有 follower 格式的 FMM 路径。"""

    waypoints: tuple[Waypoint, ...]
    start_drift_m: float
    path_length_m: float


def prepare_fmm_path_for_execution(
    fmm_plan,
    current_pose: RobotPose2D,
    *,
    max_start_drift_m: float,
    max_path_length_m: float,
) -> PreparedFMMPath:
    """校验本轮 FMM 路径，并为每个 world XY 补出切线 yaw。

    这里故意接收 duck-typed FMMPlan，避免 path follower 反向依赖规划模块。
    """
    if not current_pose.valid:
        raise ValueError("Cannot execute FMM path without a valid robot pose.")
    if max_start_drift_m < 0.0 or max_path_length_m <= 0.0:
        raise ValueError("FMM execution safety limits are invalid.")

    points = np.asarray(fmm_plan.waypoints_world_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"FMM world waypoints must have shape Nx2, got {points.shape}.")
    if points.shape[0] < 2 or not np.all(np.isfinite(points)):
        raise ValueError("FMM execution requires at least two finite world waypoints.")

    # Remove only consecutive duplicates. Non-consecutive repetition can be a
    # legitimate detour and must not be reordered or globally deduplicated.
    keep = np.ones(points.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1.0e-6
    points = points[keep]
    if points.shape[0] < 2:
        raise ValueError("FMM path collapses to one unique world point.")

    plan_start = np.asarray(fmm_plan.start_world_xy, dtype=np.float64).reshape(2)
    current_xy = np.array([current_pose.x, current_pose.y], dtype=np.float64)
    start_drift = float(np.linalg.norm(current_xy - plan_start))
    if start_drift > max_start_drift_m:
        raise ValueError(
            "FMM plan start is stale: "
            f"robot drifted {start_drift:.3f}m > {max_start_drift_m:.3f}m."
        )

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    polyline_length = float(np.sum(segment_lengths))
    reported_length = float(fmm_plan.path_length_m)
    if not math.isfinite(reported_length) or reported_length < 0.0:
        raise ValueError("FMM plan reports an invalid path length.")
    checked_length = max(polyline_length, reported_length)
    if checked_length > max_path_length_m:
        raise ValueError(
            "FMM path is longer than the configured execution limit: "
            f"{checked_length:.3f}m > {max_path_length_m:.3f}m."
        )

    segment_yaws = np.arctan2(
        np.diff(points, axis=0)[:, 1],
        np.diff(points, axis=0)[:, 0],
    )
    yaws = np.concatenate((segment_yaws, segment_yaws[-1:]))
    waypoints = tuple(
        Waypoint(float(point[0]), float(point[1]), float(yaw))
        for point, yaw in zip(points, yaws)
    )
    return PreparedFMMPath(waypoints, start_drift, checked_length)


def parse_path_waypoints(value: str | None) -> list[Waypoint]:
    """解析命令行传入的 waypoint 字符串；未传时使用默认房间路径。"""
    if value is None:
        return [Waypoint(*waypoint) for waypoint in DEFAULT_HOUSE_WAYPOINTS]

    waypoints: list[Waypoint] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid waypoint {item!r}; expected x,y,yaw.")
        waypoints.append(Waypoint(float(parts[0]), float(parts[1]), float(parts[2])))
    if not waypoints:
        raise ValueError("--path_waypoints must contain at least one waypoint.")
    if len(waypoints) > 1 and math.hypot(waypoints[-1].x - waypoints[0].x, waypoints[-1].y - waypoints[0].y) < 0.15:
        print(
            "[WARN] The first and final path waypoints are very close. "
            "For this first-pass follower, use an open path to avoid early completion."
        )
    return waypoints


class WaypointPathFollower:
    """把世界系 waypoint path 转换为 locomotion policy 的速度命令。

    算法是典型 pure-pursuit 风格：
    - 找到机器人在 path 上的最近投影点。
    - 沿路径向前取 lookahead_distance 得到目标点。
    - 把世界系目标误差转到机器人机体系。
    - 用比例控制生成 vx/vy/wz，再交给 SwitchCommandController。
    """

    def __init__(self, raw_env, env_cfg, waypoints: list[Waypoint], args_cli):
        self.args_cli = args_cli
        self.raw_env = raw_env
        self.env_cfg = env_cfg
        self.waypoints = waypoints
        self.enabled = False
        self.goal_reached = False
        self.last_pose = RobotPose2D(0.0, 0.0, 0.0, valid=False)
        self.last_target = PathTarget()
        self.path_source = "configured"
        self.abort_reason: str | None = None
        self.cross_track_abort_m: float | None = None
        self.tilt_abort_rad: float | None = None
        self.velocity_limits: tuple[float, float, float] | None = None
        self.lookahead_distance_m: float | None = None

    def replace_waypoints(
        self,
        waypoints: list[Waypoint] | tuple[Waypoint, ...],
        *,
        source: str,
        cross_track_abort_m: float | None = None,
        tilt_abort_rad: float | None = None,
        velocity_limits: tuple[float, float, float] | None = None,
        lookahead_distance_m: float | None = None,
    ) -> None:
        """安全停止旧路径并装入新的动态路径，不会自动启动机器人。"""
        checked = list(waypoints)
        if len(checked) < 2:
            raise ValueError("A dynamic path requires at least two waypoints.")
        if any(
            not all(math.isfinite(value) for value in (point.x, point.y, point.yaw))
            for point in checked
        ):
            raise ValueError("Dynamic path contains a non-finite waypoint.")
        if cross_track_abort_m is not None and cross_track_abort_m <= 0.0:
            raise ValueError("Cross-track abort distance must be positive.")
        if tilt_abort_rad is not None and tilt_abort_rad <= 0.0:
            raise ValueError("Tilt abort angle must be positive.")
        if velocity_limits is not None and any(value <= 0.0 for value in velocity_limits):
            raise ValueError("Dynamic path velocity limits must be positive.")
        if lookahead_distance_m is not None and lookahead_distance_m <= 0.0:
            raise ValueError("Dynamic path lookahead must be positive.")

        self.stop("replace path")
        self.waypoints = checked
        self.path_source = str(source)
        self.goal_reached = False
        self.abort_reason = None
        self.last_target = PathTarget()
        self.cross_track_abort_m = cross_track_abort_m
        self.tilt_abort_rad = tilt_abort_rad
        self.velocity_limits = velocity_limits
        self.lookahead_distance_m = lookahead_distance_m

    def replace_waypoints_while_active(
        self,
        waypoints: list[Waypoint] | tuple[Waypoint, ...],
        *,
        source: str,
        cross_track_abort_m: float | None = None,
        tilt_abort_rad: float | None = None,
        velocity_limits: tuple[float, float, float] | None = None,
        lookahead_distance_m: float | None = None,
    ) -> None:
        """Atomically replace an executing path without a stand/locomotion cycle."""

        if not self.enabled or self.goal_reached:
            raise RuntimeError(
                "Online FMM replan requires an actively executing path."
            )
        checked = list(waypoints)
        if len(checked) < 2:
            raise ValueError("A dynamic path requires at least two waypoints.")
        if any(
            not all(
                math.isfinite(value)
                for value in (point.x, point.y, point.yaw)
            )
            for point in checked
        ):
            raise ValueError("Dynamic path contains a non-finite waypoint.")
        if cross_track_abort_m is not None and cross_track_abort_m <= 0.0:
            raise ValueError("Cross-track abort distance must be positive.")
        if tilt_abort_rad is not None and tilt_abort_rad <= 0.0:
            raise ValueError("Tilt abort angle must be positive.")
        if velocity_limits is not None and any(
            value <= 0.0 for value in velocity_limits
        ):
            raise ValueError("Dynamic path velocity limits must be positive.")
        if lookahead_distance_m is not None and lookahead_distance_m <= 0.0:
            raise ValueError("Dynamic path lookahead must be positive.")

        self.waypoints = checked
        self.path_source = str(source)
        self.goal_reached = False
        self.abort_reason = None
        self.last_target = PathTarget()
        self.cross_track_abort_m = cross_track_abort_m
        self.tilt_abort_rad = tilt_abort_rad
        self.velocity_limits = velocity_limits
        self.lookahead_distance_m = lookahead_distance_m
        print(
            "[INFO] Active path hot-swapped: "
            f"{len(self.waypoints)} waypoints, "
            f"lookahead={self._lookahead_distance():.2f}, "
            f"source={self.path_source}."
        )

    def start(self) -> None:
        """启动路径跟踪，重置完成状态和上一次 target。"""
        if not self.waypoints:
            print("[WARN] No path waypoints configured.")
            return
        self.enabled = True
        self.goal_reached = False
        self.abort_reason = None
        self.last_target = PathTarget()
        first = self.waypoints[0]
        final = self.waypoints[-1]
        print(
            "[INFO] Path follower started: "
            f"{len(self.waypoints)} waypoints, first=({first.x:.2f},{first.y:.2f},{first.yaw:.2f}), "
            f"final=({final.x:.2f},{final.y:.2f},{final.yaw:.2f}), "
            f"lookahead={self._lookahead_distance():.2f}, source={self.path_source}."
        )

    def stop(self, reason: str | None = None) -> None:
        """停止路径跟踪；reason 只用于打印说明。"""
        if self.enabled or self.goal_reached:
            suffix = f" ({reason})" if reason else ""
            print(f"[INFO] Path follower stopped{suffix}.")
        self.enabled = False

    def update(self, command_controller: "SwitchCommandController", switch_state: "PolicySwitchState") -> None:
        """每个控制周期更新一次 path follower。

        如果到达终点，会请求切回 stand；如果还在路径中，则写入新的 requested velocity。
        """
        if not self.enabled or self.goal_reached:
            return

        if (
            self.tilt_abort_rad is not None
            and switch_state.current_tilt_angle() > self.tilt_abort_rad
        ):
            self._abort(
                "body tilt exceeded safety limit",
                command_controller,
                switch_state,
            )
            return

        pose = self.current_robot_pose()
        self.last_pose = pose
        if not pose.valid:
            command_controller.zero()
            return

        target = self.compute_path_target(pose)
        self.last_target = target
        if not target.valid:
            command_controller.zero()
            return
        if (
            self.cross_track_abort_m is not None
            and target.cross_track_error > self.cross_track_abort_m
        ):
            self._abort(
                (
                    f"cross-track error {target.cross_track_error:.3f}m exceeded "
                    f"{self.cross_track_abort_m:.3f}m"
                ),
                command_controller,
                switch_state,
            )
            return

        final = self.waypoints[-1]
        final_dx = final.x - pose.x
        final_dy = final.y - pose.y
        final_dist = math.hypot(final_dx, final_dy)
        final_yaw_error = wrap_to_pi(final.yaw - pose.yaw)
        near_path_end = target.progress >= target.total_length - self.args_cli.goal_tolerance
        if near_path_end and final_dist <= self.args_cli.goal_tolerance and abs(final_yaw_error) <= self.args_cli.yaw_tolerance:
            self.goal_reached = True
            self.enabled = False
            command_controller.zero()
            print(
                "[INFO] Path goal reached: "
                f"dist={final_dist:.3f} yaw_err={final_yaw_error:.3f} "
                f"cross_track={target.cross_track_error:.3f}. Request stand."
            )
            switch_state.request_stand()
            return

        dx_w = target.x - pose.x
        dy_w = target.y - pose.y
        yaw_error = wrap_to_pi(target.yaw - pose.yaw)

        # 世界系误差转到机器人 base frame。locomotion policy 训练时吃的是机体系速度命令，
        # 所以 vx/vy 不能直接用世界坐标差。
        cy = math.cos(pose.yaw)
        sy = math.sin(pose.yaw)
        dx_b = cy * dx_w + sy * dy_w
        dy_b = -sy * dx_w + cy * dy_w

        slow_radius = max(float(self.args_cli.goal_slow_radius), 1.0e-3)
        # 终点附近减速，避免最后一个 waypoint 前冲过头。
        speed_scale = max(0.15, min(1.0, final_dist / slow_radius))
        max_vx, max_vy, max_wz = self._velocity_limits()
        vx = max(-max_vx, min(max_vx, self.args_cli.goal_xy_kp * dx_b)) * speed_scale
        vy = max(-max_vy, min(max_vy, self.args_cli.goal_xy_kp * dy_b)) * speed_scale
        wz = max(-max_wz, min(max_wz, self.args_cli.goal_yaw_kp * yaw_error))
        command_controller.set_requested(vx, vy, wz)
        if switch_state.active_mode == "stand" and switch_state.transition_mode == "none":
            switch_state.request_locomotion()

    def current_robot_pose(self) -> RobotPose2D:
        """从 IsaacLab robot root state 读取当前世界系 x/y/yaw。"""
        robot = self.raw_env.scene["robot"]
        root_pos = tensor_first(robot.data.root_pos_w)
        root_quat = tensor_first(robot.data.root_quat_w)
        if root_pos is None or root_quat is None:
            return RobotPose2D(0.0, 0.0, 0.0, valid=False)
        return RobotPose2D(float(root_pos[0]), float(root_pos[1]), quat_wxyz_to_yaw(root_quat), valid=True)

    def _lookahead_distance(self) -> float:
        if self.lookahead_distance_m is not None:
            return float(self.lookahead_distance_m)
        return float(self.args_cli.path_lookahead_distance)

    def _velocity_limits(self) -> tuple[float, float, float]:
        configured = (
            float(self.args_cli.max_goal_vx),
            float(self.args_cli.max_goal_vy),
            float(self.args_cli.max_goal_wz),
        )
        if self.velocity_limits is None:
            return configured
        return tuple(
            min(configured_value, dynamic_value)
            for configured_value, dynamic_value in zip(configured, self.velocity_limits)
        )

    def _abort(
        self,
        reason: str,
        command_controller: "SwitchCommandController",
        switch_state: "PolicySwitchState",
    ) -> None:
        self.abort_reason = reason
        self.enabled = False
        command_controller.zero()
        print(f"[WARN] Path follower safety abort: {reason}. Request stand.")
        switch_state.request_stand()

    def compute_path_target(self, pose: RobotPose2D) -> PathTarget:
        """计算当前 lookahead target。

        返回值包含目标点、当前路径进度、最近路径段和横向误差，既用于控制也用于打印/可视化。
        """
        result = PathTarget()
        if not pose.valid or not self.waypoints:
            return result
        if len(self.waypoints) == 1:
            waypoint = self.waypoints[0]
            result.x = waypoint.x
            result.y = waypoint.y
            result.yaw = waypoint.yaw
            result.cross_track_error = math.hypot(pose.x - result.x, pose.y - result.y)
            result.valid = True
            return result

        best_progress = 0.0
        best_distance = math.inf
        total_length = 0.0
        best_segment = 0
        for index, (start, end) in enumerate(zip(self.waypoints[:-1], self.waypoints[1:])):
            # 在每条线段上投影机器人当前位置，找出离整条 path 最近的点。
            vx = end.x - start.x
            vy = end.y - start.y
            segment_len = math.hypot(vx, vy)
            if segment_len <= 1.0e-5:
                continue
            wx = pose.x - start.x
            wy = pose.y - start.y
            projection = max(0.0, min(1.0, (wx * vx + wy * vy) / (segment_len * segment_len)))
            closest_x = start.x + projection * vx
            closest_y = start.y + projection * vy
            distance = math.hypot(pose.x - closest_x, pose.y - closest_y)
            if distance < best_distance:
                best_distance = distance
                best_progress = total_length + projection * segment_len
                best_segment = index
            total_length += segment_len

        if not math.isfinite(best_distance) or total_length <= 1.0e-5:
            return result

        lookahead_progress = min(
            best_progress + max(self._lookahead_distance(), 0.0), total_length
        )
        accumulated = 0.0
        for index, (start, end) in enumerate(zip(self.waypoints[:-1], self.waypoints[1:])):
            vx = end.x - start.x
            vy = end.y - start.y
            segment_len = math.hypot(vx, vy)
            if segment_len <= 1.0e-5:
                continue
            if lookahead_progress <= accumulated + segment_len or index + 2 == len(self.waypoints):
                segment_progress = max(0.0, min(1.0, (lookahead_progress - accumulated) / segment_len))
                result.x = start.x + segment_progress * vx
                result.y = start.y + segment_progress * vy
                result.yaw = end.yaw if lookahead_progress >= total_length - 1.0e-4 else math.atan2(vy, vx)
                result.cross_track_error = best_distance
                result.progress = best_progress
                result.total_length = total_length
                result.segment_index = best_segment
                result.valid = True
                return result
            accumulated += segment_len

        final = self.waypoints[-1]
        result.x = final.x
        result.y = final.y
        result.yaw = final.yaw
        result.cross_track_error = best_distance
        result.progress = best_progress
        result.total_length = total_length
        result.segment_index = best_segment
        result.valid = True
        return result


class PathVisualizer:
    """直接在 Isaac Sim stage 中画路径调试图形。

    静态部分：绿色 waypoint、黄色路径线。
    动态部分：蓝色 lookahead target、红色 robot->target 连线。
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.stage = None
        self.root_path = "/World/PathFollowerViz"
        self._warned = False
        if not self.enabled:
            return
        try:
            import omni.usd

            self.stage = omni.usd.get_context().get_stage()
            if self.stage is not None and self.stage.GetPrimAtPath(self.root_path):
                self.stage.RemovePrim(self.root_path)
            self._define_xform(self.root_path)
        except Exception as exc:
            self.enabled = False
            print(f"[WARN] Path visualization disabled: {exc}")

    def draw_static_path(self, waypoints: list[Waypoint]) -> None:
        """画不随机器人变化的路径点和路径线。"""
        if not self.enabled or self.stage is None or not waypoints:
            return
        for index, waypoint in enumerate(waypoints):
            self._sphere(
                f"{self.root_path}/Waypoint_{index:02d}",
                (waypoint.x, waypoint.y, 0.08),
                radius=0.07,
                color=(0.0, 0.9, 0.2),
            )
        if len(waypoints) > 1:
            points = [(waypoint.x, waypoint.y, 0.10) for waypoint in waypoints]
            self._curve(f"{self.root_path}/PathLine", points, width=0.035, color=(1.0, 0.75, 0.0))

    def draw_tracking(self, pose: RobotPose2D, target: PathTarget, active: bool) -> None:
        """画每帧变化的 lookahead target 和机器人到目标的连线。"""
        if not self.enabled or self.stage is None:
            return
        dynamic_path = f"{self.root_path}/Dynamic"
        try:
            if self.stage.GetPrimAtPath(dynamic_path):
                self.stage.RemovePrim(dynamic_path)
            self._define_xform(dynamic_path)
            if not active or not pose.valid or not target.valid:
                return
            self._sphere(
                f"{dynamic_path}/LookaheadTarget",
                (target.x, target.y, 0.18),
                radius=0.09,
                color=(0.0, 0.25, 1.0),
            )
            self._curve(
                f"{dynamic_path}/RobotToTarget",
                [(pose.x, pose.y, 0.16), (target.x, target.y, 0.16)],
                width=0.025,
                color=(1.0, 0.1, 0.1),
            )
        except Exception as exc:
            if not self._warned:
                self._warned = True
                print(f"[WARN] Could not update path visualization: {exc}")

    def _define_xform(self, path: str) -> None:
        """定义一个空 Xform，作为可视化 prim 的父节点。"""
        from pxr import UsdGeom

        UsdGeom.Xform.Define(self.stage, path)

    def _sphere(self, path: str, position: tuple[float, float, float], radius: float, color: tuple[float, float, float]) -> None:
        """创建一个彩色球体 marker。"""
        from pxr import Gf, UsdGeom

        sphere = UsdGeom.Sphere.Define(self.stage, path)
        sphere.CreateRadiusAttr(radius)
        UsdGeom.XformCommonAPI(sphere).SetTranslate(Gf.Vec3d(*position))
        sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    def _curve(self, path: str, points: list[tuple[float, float, float]], width: float, color: tuple[float, float, float]) -> None:
        """创建一条彩色线段/折线。"""
        from pxr import Gf, UsdGeom

        curve = UsdGeom.BasisCurves.Define(self.stage, path)
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        curve.CreateWidthsAttr([width])
        curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])



def remove_bedroom_wardrobe_doors_from_stage(args_cli) -> None:
    """运行时删除已打开衣柜门板，避免挡住路径或视觉观察。

    注意这只是运行时 stage 清理，不会修改原始 USD 文件。之前如果已经直接编辑过 USD，
    这里可能不会找到对应 prim，但保留这个函数可以兼容未编辑场景。
    """
    if not args_cli.remove_bedroom_wardrobe_doors:
        return
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[WARN] Could not remove wardrobe doors: no USD stage.")
            return
        wardrobe_prefixes = (
            "bedroom_wardrobe_",
            "walk_in_closet_wardrobe_",
        )
        door_paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if any(prefix in str(prim.GetPath()) for prefix in wardrobe_prefixes) and "E_door" in str(prim.GetPath())
        ]
        for path in sorted(door_paths, key=len, reverse=True):
            stage.RemovePrim(path)
        print(f"[INFO] Removed wardrobe door prims at runtime: {len(door_paths)}")
    except Exception as exc:
        print(f"[WARN] Could not remove wardrobe doors: {exc}")
