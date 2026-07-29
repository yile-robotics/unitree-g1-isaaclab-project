from __future__ import annotations

"""Isaac Sim + G1 的 LaViRA 风格有限多轮 episode 控制器。

本模块参考 ``lavira_code/vlnce_baselines/lavira_main.py::rollout``，把其中的
多轮导航语义适配到 Isaac Sim、四方向 RGB-D、FMM 和 G1 locomotion：

1. 在稳定 stand 状态抓取同一仿真 step 的四方向 FrameBundle。
2. 用固定 instruction、已提交 history 和当前 panorama 请求一次模型决策。
3. NAVIGATE 使用本轮 bbox/depth 投影、局部地图和 FMM 路径执行短程移动。
4. 只有路径到达并重新稳定 stand 后，候选 waypoint 才提交到下一轮 history。
5. BACKTRACK 默认把 Qwen 选择的历史 waypoint 世界坐标设为新 FMM 目标，
   接受动作时立即截断 history；可选保留旧的历史路径反向策略作对照。
6. STOP 复用 NAVIGATE 的投影与规划链做最终接近，随后切回 stand 并终止请求。

当前实现是有明确请求上限的 bounded episode，而不是无限自主导航：

- ``max_decisions=N`` 时，前 ``N-1`` 个普通 NAVIGATE/BACKTRACK 可以执行。
- 第 ``N`` 个普通响应只校验和保存，机器人保持停止。
- STOP 是终止动作，即使出现在最后一次允许的请求中也会执行最终接近。
- 可选 online navigation 在执行期低频融合新 RGB-D，并对同一个活动世界目标
  周期 FMM 或碰撞触发 FMM；它不会产生额外模型请求或改变 history。
- 初始规划或运动安全失败会进入 FAILED；普通周期重规划失败保留上一个已验证
  路径，碰撞触发重规划失败则安全停止。
"""

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .fmm_planner import (
    FMMPlan,
    FMMPlannerConfig,
    build_fmm_plan,
    fmm_planner_config_from_args,
    save_fmm_plan_debug,
)
from .frame_bundle import FourViewCameraRig, FrameBundle
from .lavira_global_mapping import (
    LaViRAGlobalMapState,
    lavira_global_map_config_from_args,
    save_lavira_global_map_debug,
)
from .lavira_offline import NavigationDecisionOfflineProbe
from .lavira_protocol import (
    LAVIRA_MAX_HISTORY_IMAGE_WAYPOINTS,
    NavigationHistoryEntry,
)
from .navigation_mapping import (
    NavigationGridMap,
    NavigationMapConfig,
    build_navigation_grid_map_for_world_goal,
    navigation_map_config_from_args,
    save_navigation_map_debug,
)


class EpisodeState:
    """字符串状态便于日志、JSON 和测试，不依赖 Isaac/Habitat enum。"""

    WAIT_WARMUP = "WAIT_WARMUP"
    CAPTURE_AND_DECIDE = "CAPTURE_AND_DECIDE"
    EXECUTING = "EXECUTING"
    WAITING_FOR_STAND = "WAITING_FOR_STAND"
    TERMINAL_RESPONSE = "TERMINAL_RESPONSE"
    ACTION_PENDING = "ACTION_PENDING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass
class RuntimeWaypointRecord:
    """Isaac 本地的完整 waypoint；只有精简视图会发给服务器。

    ``decision_world_pose`` 和两张 history 图来自做出决策的同一个 FrameBundle。
    ``arrival_step`` 只说明出边目标已完成，不能替代这个决策点位姿。
    """

    waypoint_id: int
    decision_index: int
    decision_step: int
    decision_world_pose: np.ndarray
    direction: str
    target: str
    bbox_2d: tuple[float, float, float, float]
    progress_analysis: str
    reasoning: str
    init_rgb: np.ndarray
    dir_rgb: np.ndarray
    projected_target_world: np.ndarray | None
    safe_target_world_xy: np.ndarray | None
    fmm_waypoints_world_xy: np.ndarray | None
    execution_status: str = "candidate"
    arrival_step: int | None = None

    @property
    def turn_action(self) -> str:
        return f"turn {self.direction}"

    def to_history_entry(self, *, include_images: bool) -> NavigationHistoryEntry:
        prefix = f"history_{self.waypoint_id}"
        return NavigationHistoryEntry(
            waypoint_id=self.waypoint_id,
            step=self.decision_step,
            turn_action=self.turn_action,
            description=self.target,
            init_image_field=(f"{prefix}_init" if include_images else None),
            dir_image_field=(f"{prefix}_dir" if include_images else None),
        )

    def to_local_dict(self) -> dict:
        return {
            "waypoint_id": self.waypoint_id,
            "decision_index": self.decision_index,
            "decision_step": self.decision_step,
            "decision_world_pose": self.decision_world_pose.tolist(),
            "direction": self.direction,
            "target": self.target,
            "bbox_2d": list(self.bbox_2d),
            "turn_action": self.turn_action,
            "progress_analysis": self.progress_analysis,
            "reasoning": self.reasoning,
            "projected_target_world": (
                self.projected_target_world.tolist()
                if self.projected_target_world is not None
                else None
            ),
            "safe_target_world_xy": (
                self.safe_target_world_xy.tolist()
                if self.safe_target_world_xy is not None
                else None
            ),
            "fmm_waypoints_world_xy": (
                self.fmm_waypoints_world_xy.tolist()
                if self.fmm_waypoints_world_xy is not None
                else None
            ),
            "execution_status": self.execution_status,
            "arrival_step": self.arrival_step,
        }


@dataclass(frozen=True)
class StoredBacktrackPlan:
    """供现有 FMM 执行入口消费的已验证历史反向路径。

    ``prepare_fmm_path_for_execution`` 只依赖以下几个字段，因此不需要伪造新的
    occupancy map 或 FMM distance field。路径的每一段都来自之前真正成功执行并
    提交的 FMM path。
    """

    bundle_id: int
    start_world_xy: np.ndarray
    waypoints_world_xy: np.ndarray
    path_length_m: float
    execution_source: str
    target_waypoint_id: int
    execution_max_path_m: float


@dataclass(frozen=True)
class BacktrackExecutionRequest:
    """让 runner 复用现有 ``start_lavira_fmm_path_execution`` 的最小请求。"""

    fmm_plan: StoredBacktrackPlan


@dataclass
class ActiveNavigationGoal:
    """Stable world goal retained while maps and FMM paths are refreshed."""

    action: str
    decision_index: int
    goal_world_xy: np.ndarray
    execution_max_path_m: float
    target_waypoint_id: int | None = None
    replan_count: int = 0
    last_replan_step: int | None = None
    last_fmm_plan: FMMPlan | None = None

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "decision_index": self.decision_index,
            "goal_world_xy": self.goal_world_xy.tolist(),
            "execution_max_path_m": self.execution_max_path_m,
            "target_waypoint_id": self.target_waypoint_id,
            "replan_count": self.replan_count,
            "last_replan_step": self.last_replan_step,
            "last_plan_bundle_id": (
                int(getattr(self.last_fmm_plan, "bundle_id"))
                if self.last_fmm_plan is not None
                and getattr(self.last_fmm_plan, "bundle_id", None) is not None
                else None
            ),
        }


def build_replanned_backtrack_execution_request(
    records: list[RuntimeWaypointRecord] | tuple[RuntimeWaypointRecord, ...],
    bundle: FrameBundle,
    *,
    target_waypoint_id: int,
    decision_index: int,
    navigation_map_config: NavigationMapConfig,
    fmm_planner_config: FMMPlannerConfig,
    execution_max_path_m: float = 6.0,
) -> tuple[BacktrackExecutionRequest, NavigationGridMap, FMMPlan]:
    """按原版 LaViRA 语义把历史 waypoint 世界坐标设为当前 FMM 目标。"""

    records = tuple(records)
    if not records:
        raise ValueError("BACKTRACK requires at least one committed history waypoint.")
    if not isinstance(target_waypoint_id, int):
        raise ValueError("BACKTRACK waypoint must be an integer.")
    if not 0 <= target_waypoint_id < len(records):
        raise ValueError(
            f"BACKTRACK waypoint {target_waypoint_id} is outside committed "
            f"history [0, {len(records) - 1}]."
        )
    if execution_max_path_m <= 0.0:
        raise ValueError("BACKTRACK maximum path length must be positive.")

    target_record = records[target_waypoint_id]
    if target_record.execution_status != "arrived":
        raise ValueError(
            f"BACKTRACK waypoint {target_waypoint_id} is not an arrived record."
        )
    decision_pose = np.asarray(target_record.decision_world_pose, dtype=np.float64)
    if decision_pose.shape != (4, 4) or not np.all(np.isfinite(decision_pose)):
        raise ValueError(
            f"BACKTRACK waypoint {target_waypoint_id} has an invalid decision pose."
        )
    target_world_xy = decision_pose[:2, 3].copy()
    grid_map = build_navigation_grid_map_for_world_goal(
        bundle,
        target_world_xy,
        navigation_map_config,
    )
    if grid_map.safe_target_world_xy is None:
        raise RuntimeError(
            f"BACKTRACK waypoint {target_waypoint_id} is not reachable on the "
            "current four-view traversability map."
        )
    fmm_plan = build_fmm_plan(grid_map, fmm_planner_config)
    execution_plan = StoredBacktrackPlan(
        bundle_id=int(decision_index),
        start_world_xy=np.asarray(fmm_plan.start_world_xy, dtype=np.float64).copy(),
        waypoints_world_xy=np.asarray(
            fmm_plan.waypoints_world_xy, dtype=np.float64
        ).copy(),
        path_length_m=float(fmm_plan.path_length_m),
        execution_source=(
            f"lavira_replanned_backtrack_waypoint_{target_waypoint_id:03d}_"
            f"decision_{decision_index:03d}"
        ),
        target_waypoint_id=int(target_waypoint_id),
        execution_max_path_m=float(execution_max_path_m),
    )
    return BacktrackExecutionRequest(execution_plan), grid_map, fmm_plan


def build_global_replanned_backtrack_execution_request(
    records: list[RuntimeWaypointRecord] | tuple[RuntimeWaypointRecord, ...],
    global_map_state: LaViRAGlobalMapState,
    *,
    target_waypoint_id: int,
    decision_index: int,
    fmm_planner_config: FMMPlannerConfig,
    execution_max_path_m: float = 6.0,
) -> tuple[BacktrackExecutionRequest, NavigationGridMap, FMMPlan]:
    """Replan to Qwen's selected history waypoint on the cumulative full map."""

    records = tuple(records)
    if not records:
        raise ValueError("BACKTRACK requires at least one committed history waypoint.")
    if not isinstance(target_waypoint_id, int):
        raise ValueError("BACKTRACK waypoint must be an integer.")
    if not 0 <= target_waypoint_id < len(records):
        raise ValueError(
            f"BACKTRACK waypoint {target_waypoint_id} is outside committed "
            f"history [0, {len(records) - 1}]."
        )
    if execution_max_path_m <= 0.0:
        raise ValueError("BACKTRACK maximum path length must be positive.")
    target_record = records[target_waypoint_id]
    if target_record.execution_status != "arrived":
        raise ValueError(
            f"BACKTRACK waypoint {target_waypoint_id} is not an arrived record."
        )
    decision_pose = np.asarray(target_record.decision_world_pose, dtype=np.float64)
    if decision_pose.shape != (4, 4) or not np.all(np.isfinite(decision_pose)):
        raise ValueError(
            f"BACKTRACK waypoint {target_waypoint_id} has an invalid decision pose."
        )

    target_world_xy = decision_pose[:2, 3].copy()
    grid_map = global_map_state.build_navigation_grid_map(
        historical_target_world_xy=target_world_xy
    )
    if grid_map.safe_target_world_xy is None:
        raise RuntimeError(
            f"BACKTRACK waypoint {target_waypoint_id} is not reachable on the "
            "cumulative global traversability map."
        )
    fmm_plan = build_fmm_plan(grid_map, fmm_planner_config)
    execution_plan = StoredBacktrackPlan(
        bundle_id=int(decision_index),
        start_world_xy=np.asarray(fmm_plan.start_world_xy, dtype=np.float64).copy(),
        waypoints_world_xy=np.asarray(
            fmm_plan.waypoints_world_xy, dtype=np.float64
        ).copy(),
        path_length_m=float(fmm_plan.path_length_m),
        execution_source=(
            f"lavira_global_replanned_backtrack_waypoint_"
            f"{target_waypoint_id:03d}_decision_{decision_index:03d}"
        ),
        target_waypoint_id=int(target_waypoint_id),
        execution_max_path_m=float(execution_max_path_m),
    )
    return BacktrackExecutionRequest(execution_plan), grid_map, fmm_plan


def build_backtrack_execution_request(
    records: list[RuntimeWaypointRecord] | tuple[RuntimeWaypointRecord, ...],
    *,
    target_waypoint_id: int,
    decision_index: int,
    execution_max_path_m: float = 6.0,
) -> BacktrackExecutionRequest:
    """把目标 waypoint 之后的成功路径按时间逆序拼成返回路径。

    LaViRA 原版直接把 planner 目标设为历史 ``world_coords``。Isaac 版本保存了
    每一段实际接受的 FMM 世界路径，因此沿这些静态环境中的已知安全路径反向返回
    比当前位置到旧坐标的直线连接更保守。
    """

    records = tuple(records)
    if not records:
        raise ValueError("BACKTRACK requires at least one committed history waypoint.")
    if not isinstance(target_waypoint_id, int):
        raise ValueError("BACKTRACK waypoint must be an integer.")
    if execution_max_path_m <= 0.0:
        raise ValueError("BACKTRACK maximum path length must be positive.")
    if not 0 <= target_waypoint_id < len(records):
        raise ValueError(
            f"BACKTRACK waypoint {target_waypoint_id} is outside committed "
            f"history [0, {len(records) - 1}]."
        )

    reversed_segments: list[np.ndarray] = []
    for record in reversed(records[target_waypoint_id:]):
        if record.execution_status != "arrived":
            raise ValueError(
                f"BACKTRACK waypoint {record.waypoint_id} is not an arrived record."
            )
        if record.fmm_waypoints_world_xy is None:
            raise ValueError(
                f"BACKTRACK waypoint {record.waypoint_id} has no stored FMM path."
            )
        points = np.asarray(record.fmm_waypoints_world_xy, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or points.shape[0] < 2
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(
                f"BACKTRACK waypoint {record.waypoint_id} has an invalid stored "
                f"path shape {points.shape}."
            )
        reversed_segments.append(points[::-1].copy())

    merged = reversed_segments[0]
    for segment in reversed_segments[1:]:
        # Consecutive decisions can differ by the normal goal tolerance because
        # the next bundle records the measured arrival pose. Keep that short
        # connector; only remove an exactly duplicated join.
        if np.linalg.norm(merged[-1] - segment[0]) <= 1.0e-6:
            segment = segment[1:]
        if segment.size:
            merged = np.concatenate((merged, segment), axis=0)

    keep = np.ones(merged.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(merged, axis=0), axis=1) > 1.0e-6
    merged = np.ascontiguousarray(merged[keep])
    if merged.shape[0] < 2:
        raise ValueError("BACKTRACK path collapses to fewer than two unique points.")
    path_length_m = float(np.sum(np.linalg.norm(np.diff(merged, axis=0), axis=1)))
    return BacktrackExecutionRequest(
        fmm_plan=StoredBacktrackPlan(
            bundle_id=int(decision_index),
            start_world_xy=merged[0].copy(),
            waypoints_world_xy=merged,
            path_length_m=path_length_m,
            execution_source=(
                f"lavira_backtrack_waypoint_{target_waypoint_id:03d}_"
                f"decision_{decision_index:03d}"
            ),
            target_waypoint_id=int(target_waypoint_id),
            execution_max_path_m=float(execution_max_path_m),
        )
    )


def build_server_history(
    records: list[RuntimeWaypointRecord] | tuple[RuntimeWaypointRecord, ...],
) -> tuple[tuple[NavigationHistoryEntry, ...], dict[str, np.ndarray]]:
    """按原版 LaViRA 图片预算构造服务器 history 和 multipart 图片。

    所有历史点保留文字，只有最近四个历史点携带 ``init/dir`` 图片。
    """

    records = tuple(records)
    image_start = max(0, len(records) - LAVIRA_MAX_HISTORY_IMAGE_WAYPOINTS)
    entries: list[NavigationHistoryEntry] = []
    images: dict[str, np.ndarray] = {}
    for index, record in enumerate(records):
        if record.waypoint_id != index:
            raise ValueError(
                "Runtime history waypoint ids must be contiguous and zero-based."
            )
        if record.execution_status != "arrived":
            raise ValueError(
                f"Waypoint {record.waypoint_id} is not an arrived history point."
            )
        entry = record.to_history_entry(include_images=index >= image_start)
        entries.append(entry)
        if entry.has_images:
            images[entry.init_image_field] = np.asarray(record.init_rgb).copy()
            images[entry.dir_image_field] = np.asarray(record.dir_rgb).copy()
    return tuple(entries), images


class LaViRABoundedEpisodeController:
    """调度有限多轮请求、NAVIGATE/BACKTRACK 执行和 STOP 最终接近。

    ``--lavira_history_probe`` 是沿用早期测试阶段的命令行名称；当前对象已经是
    真正执行路径并维护 history 的 bounded episode controller。
    """

    def __init__(self, args_cli):
        self.args_cli = args_cli
        self.enabled = bool(getattr(args_cli, "lavira_history_probe", False))
        self.max_decisions = int(
            getattr(args_cli, "lavira_history_max_decisions", 3)
        )
        if self.max_decisions < 2:
            raise ValueError("LaViRA bounded episode requires at least two decisions.")
        self.backtrack_strategy = str(
            getattr(args_cli, "lavira_backtrack_strategy", "replan_world_goal")
        )
        if self.backtrack_strategy not in {
            "replan_world_goal",
            "stored_reverse",
        }:
            raise ValueError(
                "lavira_backtrack_strategy must be 'replan_world_goal' "
                "or 'stored_reverse'."
            )
        self.navigation_map_mode = str(
            getattr(args_cli, "nav_map_mode", "local_current_bundle")
        )
        if self.navigation_map_mode not in {
            "local_current_bundle",
            "lavira_compatible_global",
        }:
            raise ValueError(
                "nav_map_mode must be 'local_current_bundle' or "
                "'lavira_compatible_global'."
            )
        self.global_map_state: LaViRAGlobalMapState | None = None
        if self.navigation_map_mode == "lavira_compatible_global":
            navigation_config = navigation_map_config_from_args(args_cli)
            global_config = lavira_global_map_config_from_args(args_cli)
            self.global_map_state = LaViRAGlobalMapState(
                navigation_config,
                global_config,
            )
        self.online_navigation = bool(
            getattr(args_cli, "lavira_online_navigation", False)
        )
        if self.online_navigation and self.global_map_state is None:
            raise ValueError(
                "Online LaViRA navigation requires the cumulative global map."
            )
        if (
            self.online_navigation
            and self.backtrack_strategy != "replan_world_goal"
        ):
            raise ValueError(
                "Online LaViRA navigation requires replan_world_goal BACKTRACK."
            )
        if self.online_navigation:
            positive_names = (
                "lavira_online_mapping_interval_s",
                "lavira_online_replan_interval_s",
                "lavira_collision_command_speed_m_s",
                "lavira_collision_window_s",
                "lavira_collision_mark_distance_m",
            )
            for name in positive_names:
                if float(getattr(args_cli, name)) <= 0.0:
                    raise ValueError(f"{name} must be positive.")
            if float(args_cli.lavira_collision_min_progress_m) < 0.0:
                raise ValueError(
                    "lavira_collision_min_progress_m must be non-negative."
                )
            if float(args_cli.lavira_collision_mark_radius_m) < 0.0:
                raise ValueError(
                    "lavira_collision_mark_radius_m must be non-negative."
                )
        self.state = EpisodeState.WAIT_WARMUP
        self.history: list[RuntimeWaypointRecord] = []
        self.pending_waypoint: RuntimeWaypointRecord | None = None
        self.current_probe: NavigationDecisionOfflineProbe | None = None
        self.terminal_response: dict | None = None
        self.pending_action: dict | None = None
        self.completed = False
        self.failure_reason: str | None = None
        self.execution_started_step: int | None = None
        self.active_execution_action: str | None = None
        self.active_execution_decision_index: int | None = None
        self.backtrack_target_waypoint: int | None = None
        self.backtrack_request: BacktrackExecutionRequest | None = None
        self.backtrack_event: dict | None = None
        self.stop_goal_world_xy: np.ndarray | None = None
        self.stop_event: dict | None = None
        self.active_goal: ActiveNavigationGoal | None = None
        self.online_events: list[dict] = []
        self.online_map_update_count = 0
        self.online_replan_count = 0
        self.online_collision_count = 0
        self._last_online_map_time_s: float | None = None
        self._last_online_replan_time_s: float | None = None
        self._collision_window_start_time_s: float | None = None
        self._collision_window_start_xy: np.ndarray | None = None
        self._collision_direction_world_xy: np.ndarray | None = None
        self.decisions_completed = 0
        self._stand_stable_elapsed = 0.0
        self._run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def update_after_step(
        self,
        camera_rig: FourViewCameraRig | None,
        *,
        completed_step: int,
        step_dt: float,
        path_follower,
        path_visualizer,
        command_controller,
        switch_state,
        start_path: Callable[[object], bool],
        hot_swap_path: Callable[[FMMPlan, float], bool] | None = None,
        applied_velocity_command: np.ndarray | None = None,
    ) -> None:
        if not self.enabled or self.completed or self.state == EpisodeState.FAILED:
            return

        if camera_rig is None:
            self._fail(
                "FourViewCameraRig is unavailable.",
                path_follower,
                command_controller,
                switch_state,
            )
            return

        if self.state == EpisodeState.WAIT_WARMUP:
            command_controller.zero()
            self._request_stand_if_needed(switch_state)
            if completed_step < int(self.args_cli.lavira_decision_warmup_steps):
                return
            if not self._stand_ready(switch_state):
                return
            self.state = EpisodeState.CAPTURE_AND_DECIDE

        if self.state == EpisodeState.CAPTURE_AND_DECIDE:
            self._capture_and_handle_decision(
                camera_rig,
                completed_step=completed_step,
                step_dt=step_dt,
                path_follower=path_follower,
                path_visualizer=path_visualizer,
                command_controller=command_controller,
                switch_state=switch_state,
                start_path=start_path,
            )
            return

        if self.state == EpisodeState.EXECUTING:
            decision_index = self.current_decision_index
            if self.active_execution_decision_index is not None:
                decision_index = self.active_execution_decision_index
            if self.pending_waypoint is not None:
                decision_index = self.pending_waypoint.decision_index
            if path_follower.abort_reason is not None:
                self._fail(
                    (
                        f"decision {decision_index} path aborted: "
                        f"{path_follower.abort_reason}"
                    ),
                    path_follower,
                    command_controller,
                    switch_state,
                )
                return
            if not path_follower.enabled and not path_follower.goal_reached:
                self._fail(
                    f"decision {decision_index} path stopped before reaching its goal.",
                    path_follower,
                    command_controller,
                    switch_state,
                )
                return
            if self.active_execution_action == "STOP":
                stop_distance = self._current_stop_distance(path_follower)
                if stop_distance is not None:
                    self._update_stop_distance(stop_distance)
                    if stop_distance < self.stop_reached_threshold_m:
                        self._finish_stop_path_approach(
                            completed_step=completed_step,
                            stop_distance=stop_distance,
                            path_follower=path_follower,
                            command_controller=command_controller,
                            switch_state=switch_state,
                        )
                        return
            if (
                not path_follower.goal_reached
                and self.online_navigation
            ):
                self._maybe_update_online_navigation(
                    camera_rig,
                    completed_step=completed_step,
                    step_dt=step_dt,
                    path_follower=path_follower,
                    command_controller=command_controller,
                    switch_state=switch_state,
                    hot_swap_path=hot_swap_path,
                    applied_velocity_command=applied_velocity_command,
                )
                if self.state == EpisodeState.FAILED:
                    return
            timeout_seconds = float(self.args_cli.lavira_history_execution_timeout)
            elapsed = (
                (int(completed_step) - int(self.execution_started_step))
                * float(step_dt)
                if self.execution_started_step is not None
                else 0.0
            )
            if elapsed > timeout_seconds:
                self._fail(
                    (
                        f"decision {decision_index} path timed out after "
                        f"{elapsed:.2f}s."
                    ),
                    path_follower,
                    command_controller,
                    switch_state,
                )
                return
            if not path_follower.goal_reached:
                return
            command_controller.zero()
            self._request_stand_if_needed(switch_state)
            self._stand_stable_elapsed = 0.0
            self.state = EpisodeState.WAITING_FOR_STAND
            action = self.active_execution_action or "NAVIGATE"
            print(
                f"[LAVIRA EPISODE] decision_{decision_index:03d} {action} path "
                "reached; "
                "waiting for stable stand before episode update."
            )
            return

        if self.state == EpisodeState.WAITING_FOR_STAND:
            command_controller.zero()
            self._request_stand_if_needed(switch_state)
            if not self._stand_ready(switch_state):
                self._stand_stable_elapsed = 0.0
                return
            if (
                switch_state.current_tilt_angle()
                > float(self.args_cli.fmm_execute_tilt_abort_rad)
            ):
                self._fail(
                    "robot tilt exceeded the history commit limit.",
                    path_follower,
                    command_controller,
                    switch_state,
                )
                return
            self._stand_stable_elapsed += float(step_dt)
            if (
                self._stand_stable_elapsed
                < float(self.args_cli.lavira_history_settle_seconds)
            ):
                return
            if self.active_execution_action == "BACKTRACK":
                self._complete_backtrack(completed_step)
            elif self.active_execution_action == "NAVIGATE":
                self._commit_pending_waypoint(completed_step)
            elif self.active_execution_action == "STOP":
                self._complete_stop(completed_step)
                self._clear_active_execution()
                self._save_controller_status()
                return
            else:
                self._fail(
                    "stable stand reached without a known active episode action.",
                    path_follower,
                    command_controller,
                    switch_state,
                )
                return
            self._clear_active_execution()
            self.state = EpisodeState.CAPTURE_AND_DECIDE
            return

    def _capture_and_handle_decision(
        self,
        camera_rig: FourViewCameraRig,
        *,
        completed_step: int,
        step_dt: float,
        path_follower,
        path_visualizer,
        command_controller,
        switch_state,
        start_path: Callable[[object], bool],
    ) -> None:
        decision_index = self.current_decision_index
        try:
            probe = self._run_probe(
                camera_rig,
                decision_index=decision_index,
                completed_step=completed_step,
                step_dt=step_dt,
            )
            if probe.response is None:
                raise RuntimeError(
                    f"decision {decision_index} returned no parsed response."
                )
            self.current_probe = probe
            self.decisions_completed += 1
            action = probe.response.action.upper()
            is_terminal_read = decision_index >= self.max_decisions - 1
            self._apply_global_map_to_probe(
                probe,
                action=action,
                plan_target=(not is_terminal_read or action == "STOP"),
            )

            # STOP is a terminal navigation action in LaViRA, not merely another
            # bounded-episode response. Let it finish its bbox/FMM approach even
            # when it happens to be the final allowed model request.
            if is_terminal_read and action != "STOP":
                command_controller.zero()
                self._request_stand_if_needed(switch_state)
                self.terminal_response = probe.response.to_dict()
                self.state = EpisodeState.TERMINAL_RESPONSE
                self.completed = True
                self._save_controller_status()
                print(
                    f"[LAVIRA EPISODE] decision_{decision_index:03d} response "
                    f"valid: action={action} history={len(self.history)}. "
                    f"Bounded {self.max_decisions}-decision episode complete; "
                    "robot remains stopped."
                )
                return

            if action == "STOP":
                self._start_stop_action(
                    probe,
                    decision_index=decision_index,
                    completed_step=completed_step,
                    step_dt=step_dt,
                    path_follower=path_follower,
                    command_controller=command_controller,
                    switch_state=switch_state,
                    start_path=start_path,
                )
                return

            if action == "BACKTRACK":
                target_waypoint = probe.response.waypoint
                if target_waypoint is None:
                    raise RuntimeError("BACKTRACK response lacks waypoint.")
                target_waypoint = int(target_waypoint)
                replanned_grid_map: NavigationGridMap | None = None
                replanned_fmm_plan: FMMPlan | None = None
                if self.backtrack_strategy == "replan_world_goal":
                    if probe.bundle is None:
                        raise RuntimeError(
                            "BACKTRACK world-goal replanning requires the current "
                            "FrameBundle."
                        )
                    if self.global_map_state is not None:
                        (
                            request,
                            replanned_grid_map,
                            replanned_fmm_plan,
                        ) = build_global_replanned_backtrack_execution_request(
                            self.history,
                            self.global_map_state,
                            target_waypoint_id=target_waypoint,
                            decision_index=decision_index,
                            fmm_planner_config=fmm_planner_config_from_args(
                                self.args_cli
                            ),
                            execution_max_path_m=float(
                                self.args_cli.lavira_backtrack_max_path_m
                            ),
                        )
                    else:
                        (
                            request,
                            replanned_grid_map,
                            replanned_fmm_plan,
                        ) = build_replanned_backtrack_execution_request(
                            self.history,
                            probe.bundle,
                            target_waypoint_id=target_waypoint,
                            decision_index=decision_index,
                            navigation_map_config=navigation_map_config_from_args(
                                self.args_cli
                            ),
                            fmm_planner_config=fmm_planner_config_from_args(
                                self.args_cli
                            ),
                            execution_max_path_m=float(
                                self.args_cli.lavira_backtrack_max_path_m
                            ),
                        )
                    if probe.output_dir is not None:
                        if self.global_map_state is not None:
                            global_files = save_lavira_global_map_debug(
                                Path(probe.output_dir),
                                self.global_map_state,
                                replanned_grid_map,
                            )
                            global_planning_dir = (
                                Path(probe.output_dir) / "global_planning"
                            )
                            global_planning_dir.mkdir(
                                parents=True, exist_ok=True
                            )
                            fmm_files = save_fmm_plan_debug(
                                global_planning_dir,
                                replanned_grid_map,
                                replanned_fmm_plan,
                            )
                            self._record_global_planning_interpretation(
                                probe,
                                action="BACKTRACK",
                                planning_map=replanned_grid_map,
                                fmm_plan=replanned_fmm_plan,
                                global_files=global_files,
                                fmm_files=fmm_files,
                            )
                        else:
                            save_navigation_map_debug(
                                Path(probe.output_dir), replanned_grid_map
                            )
                            save_fmm_plan_debug(
                                Path(probe.output_dir),
                                replanned_grid_map,
                                replanned_fmm_plan,
                            )
                else:
                    request = build_backtrack_execution_request(
                        self.history,
                        target_waypoint_id=target_waypoint,
                        decision_index=decision_index,
                        execution_max_path_m=float(
                            self.args_cli.lavira_backtrack_max_path_m
                        ),
                    )
                self.pending_waypoint = None
                self.backtrack_target_waypoint = target_waypoint
                self.backtrack_request = request
                history_count_before = len(self.history)
                # Match lavira_main.py exactly: once BACKTRACK is accepted, discard
                # the abandoned branch before physical execution begins. A later
                # planning/execution failure does not restore that branch.
                self.history = self.history[: target_waypoint + 1]
                self.backtrack_event = {
                    "status": "accepted",
                    "decision_index": int(decision_index),
                    "strategy": self.backtrack_strategy,
                    "target_waypoint": target_waypoint,
                    "target_world_xy": (
                        np.asarray(
                            self.history[target_waypoint].decision_world_pose[:2, 3],
                            dtype=np.float64,
                        ).tolist()
                    ),
                    "history_count_before": history_count_before,
                    "history_count_after": len(self.history),
                    "path_length_m": request.fmm_plan.path_length_m,
                    "waypoints_world_xy": (
                        request.fmm_plan.waypoints_world_xy.tolist()
                    ),
                    "completion_step": None,
                }
                self._save_backtrack_event()
                started = bool(start_path(request))
                if not started:
                    raise RuntimeError(
                        f"decision {decision_index} BACKTRACK path was not accepted "
                        "for locomotion."
                    )
                self._begin_active_goal(
                    action="BACKTRACK",
                    decision_index=decision_index,
                    goal_world_xy=np.asarray(
                        self.history[target_waypoint].decision_world_pose[:2, 3],
                        dtype=np.float64,
                    ),
                    execution_max_path_m=float(
                        self.args_cli.lavira_backtrack_max_path_m
                    ),
                    completed_step=completed_step,
                    step_dt=step_dt,
                    target_waypoint_id=target_waypoint,
                    initial_fmm_plan=replanned_fmm_plan,
                )
                self.execution_started_step = int(completed_step)
                self.active_execution_action = "BACKTRACK"
                self.active_execution_decision_index = int(decision_index)
                self.state = EpisodeState.EXECUTING
                print(
                    f"[LAVIRA EPISODE] decision_{decision_index:03d} BACKTRACK "
                    f"accepted: waypoint={target_waypoint} "
                    f"strategy={self.backtrack_strategy} "
                    f"history={history_count_before}->{len(self.history)} "
                    f"path_length={request.fmm_plan.path_length_m:.3f}m."
                )
                return

            if action != "NAVIGATE":
                command_controller.zero()
                self._request_stand_if_needed(switch_state)
                self.pending_action = probe.response.to_dict()
                self.state = EpisodeState.ACTION_PENDING
                self.completed = True
                self._save_controller_status()
                print(
                    f"[LAVIRA EPISODE] decision_{decision_index:03d} returned "
                    f"{action}; this action is unsupported by the bounded episode "
                    "controller. "
                    "Action saved as pending and robot remains stopped."
                )
                return

            if probe.fmm_plan is None:
                raise RuntimeError(
                    f"decision {decision_index} produced no executable FMM plan."
                )
            self.pending_waypoint = self._candidate_from_probe(probe)
            started = bool(start_path(probe))
            if not started:
                raise RuntimeError(
                    f"decision {decision_index} FMM path was not accepted "
                    "for locomotion."
                )
            self._begin_active_goal(
                action="NAVIGATE",
                decision_index=decision_index,
                goal_world_xy=np.asarray(
                    probe.navigation_map.safe_target_world_xy,
                    dtype=np.float64,
                ),
                execution_max_path_m=float(
                    getattr(
                        self.args_cli,
                        "fmm_execute_max_path_m",
                        6.0,
                    )
                ),
                completed_step=completed_step,
                step_dt=step_dt,
                initial_fmm_plan=probe.fmm_plan,
            )
            self.execution_started_step = int(completed_step)
            self.active_execution_action = "NAVIGATE"
            self.active_execution_decision_index = int(decision_index)
            self.state = EpisodeState.EXECUTING
            print(
                f"[LAVIRA EPISODE] decision_{decision_index:03d} accepted and "
                f"path started: target={probe.response.target!r} "
                f"direction={probe.response.direction}."
            )
        except Exception as exc:
            self._fail(
                f"decision {decision_index} failed: {exc}",
                path_follower,
                command_controller,
                switch_state,
            )

    def _run_probe(
        self,
        camera_rig: FourViewCameraRig,
        *,
        decision_index: int,
        completed_step: int,
        step_dt: float,
    ) -> NavigationDecisionOfflineProbe:
        history, history_images = build_server_history(self.history)
        probe = NavigationDecisionOfflineProbe(
            self.args_cli,
            enabled=True,
            decision_index=decision_index,
            history=history,
            history_images=history_images,
            run_id=self._run_id,
            execution_requested=decision_index < self.max_decisions - 1,
        )
        print(
            "[LAVIRA EPISODE] sending decision request: "
            f"decision={decision_index:03d} history={len(history)} "
            f"images={4 + len(history_images)}."
        )
        probe.maybe_run(
            camera_rig,
            completed_step=completed_step,
            step_dt=step_dt,
        )
        if not probe.completed:
            raise RuntimeError(f"decision {decision_index} probe did not complete.")
        return probe

    def _apply_global_map_to_probe(
        self,
        probe: NavigationDecisionOfflineProbe,
        *,
        action: str,
        plan_target: bool,
    ) -> None:
        """Fuse this decision and replace local planning with the full map."""

        state = self.global_map_state
        if state is None:
            return
        if probe.bundle is None:
            raise RuntimeError("Global mapping requires the current FrameBundle.")

        # Reuse the already projected current observation when available. For a
        # BACKTRACK response the offline probe intentionally skipped target map
        # construction, so build a target-independent observation here.
        if probe.navigation_map is not None:
            state.integrate_grid_map(probe.navigation_map)
        else:
            state.integrate_bundle(probe.bundle)

        planning_map: NavigationGridMap | None = None
        if plan_target and action in {"NAVIGATE", "STOP"}:
            if probe.target_projection is None:
                raise RuntimeError(
                    f"{action} global planning requires a valid target projection."
                )
            planning_map = state.build_navigation_grid_map(
                projection=probe.target_projection
            )
            if planning_map.safe_target_cell_rc is None:
                raise RuntimeError(
                    f"{action} target has no reachable cell on the cumulative "
                    "global map."
                )
            probe.navigation_map = planning_map
            probe.fmm_plan = build_fmm_plan(
                planning_map,
                fmm_planner_config_from_args(self.args_cli),
            )

        if probe.output_dir is not None:
            global_files = save_lavira_global_map_debug(
                Path(probe.output_dir),
                state,
                planning_map,
            )
            fmm_files: dict[str, str] | None = None
            if planning_map is not None and probe.fmm_plan is not None:
                global_planning_dir = (
                    Path(probe.output_dir) / "global_planning"
                )
                global_planning_dir.mkdir(parents=True, exist_ok=True)
                fmm_files = save_fmm_plan_debug(
                    global_planning_dir,
                    planning_map,
                    probe.fmm_plan,
                )
            self._record_global_planning_interpretation(
                probe,
                action=action,
                planning_map=planning_map,
                fmm_plan=probe.fmm_plan if planning_map is not None else None,
                global_files=global_files,
                fmm_files=fmm_files,
            )
        print(
            "[LAVIRA GLOBAL MAP] fused decision observation: "
            f"updates={state.update_count} "
            f"origin={state.origin_world_xy.tolist()} "
            f"explored={np.count_nonzero(state.full_map[1])} cells."
        )

    def _record_global_planning_interpretation(
        self,
        probe: NavigationDecisionOfflineProbe,
        *,
        action: str,
        planning_map: NavigationGridMap | None,
        fmm_plan: FMMPlan | None,
        global_files: dict[str, str],
        fmm_files: dict[str, str] | None,
    ) -> None:
        """Append the effective execution map/plan without rewriting local probe data."""

        if probe.output_dir is None or self.global_map_state is None:
            return
        path = Path(probe.output_dir) / "response_interpretation.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        state = self.global_map_state
        payload["global_map_fusion"] = {
            "status": "ok",
            "map_mode": self.navigation_map_mode,
            "files": global_files,
            "update_count": state.update_count,
            "origin_world_xy": state.origin_world_xy.tolist(),
            "shape_rc": [state.cells, state.cells],
            "channel_names": [
                "obstacle",
                "explored",
                "current_location",
                "past_locations",
            ],
        }
        if planning_map is None or fmm_plan is None:
            payload["global_execution_plan"] = {
                "status": "not_run",
                "action": action,
                "reason": "This bounded response does not execute a target plan.",
            }
        else:
            payload["global_execution_plan"] = {
                "status": "ok",
                "action": action,
                "target_selection_strategy": (
                    planning_map.target_selection_strategy
                ),
                "safe_target_cell_rc": list(
                    planning_map.safe_target_cell_rc
                ),
                "safe_target_world_xy": (
                    planning_map.safe_target_world_xy.tolist()
                ),
                "start_cell_rc": list(fmm_plan.start_cell_rc),
                "goal_cell_rc": list(fmm_plan.goal_cell_rc),
                "path_length_m": fmm_plan.path_length_m,
                "waypoint_count": int(
                    fmm_plan.waypoints_world_xy.shape[0]
                ),
                "lavira_short_term_goal_cell_rc": list(
                    fmm_plan.lavira_short_term_goal_cell_rc
                ),
                "files": {
                    key: f"global_planning/{value}"
                    for key, value in (fmm_files or {}).items()
                },
            }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _candidate_from_probe(
        self, probe: NavigationDecisionOfflineProbe
    ) -> RuntimeWaypointRecord:
        if (
            probe.bundle is None
            or probe.response is None
            or probe.target_projection is None
            or probe.navigation_map is None
            or probe.fmm_plan is None
        ):
            raise RuntimeError("Cannot create history candidate from incomplete decision.")
        direction = probe.response.direction
        if (
            direction is None
            or probe.response.target is None
            or probe.response.bbox_2d is None
        ):
            raise RuntimeError("NAVIGATE response lacks direction/target/bbox.")
        return RuntimeWaypointRecord(
            waypoint_id=len(self.history),
            decision_index=int(probe.decision_index),
            decision_step=int(probe.bundle.sim_step),
            decision_world_pose=np.asarray(probe.bundle.T_world_base).copy(),
            direction=direction,
            target=probe.response.target,
            bbox_2d=tuple(probe.response.bbox_2d),
            progress_analysis=probe.response.progress_analysis,
            reasoning=probe.response.reasoning,
            init_rgb=np.asarray(probe.bundle.views["forward"].rgb).copy(),
            dir_rgb=np.asarray(probe.bundle.views[direction].rgb).copy(),
            projected_target_world=np.asarray(
                probe.target_projection.point_world_m
            ).copy(),
            safe_target_world_xy=(
                np.asarray(probe.navigation_map.safe_target_world_xy).copy()
                if probe.navigation_map.safe_target_world_xy is not None
                else None
            ),
            fmm_waypoints_world_xy=np.asarray(
                probe.fmm_plan.waypoints_world_xy
            ).copy(),
        )

    @property
    def current_decision_index(self) -> int:
        # BACKTRACK truncates history, but request/observation ids must never be
        # reused within one session.
        return self.decisions_completed

    @property
    def stop_reached_threshold_m(self) -> float:
        """LaViRA 默认 15 cells × 0.05 m/cell = 0.75 m。"""
        return float(self.args_cli.lavira_stop_reached_threshold_m)

    def _begin_active_goal(
        self,
        *,
        action: str,
        decision_index: int,
        goal_world_xy: np.ndarray,
        execution_max_path_m: float,
        completed_step: int,
        step_dt: float,
        target_waypoint_id: int | None = None,
        initial_fmm_plan: FMMPlan | None = None,
    ) -> None:
        """Retain one model-selected goal across online map/FMM refreshes."""

        goal = np.asarray(goal_world_xy, dtype=np.float64)
        if goal.shape != (2,) or not np.all(np.isfinite(goal)):
            raise ValueError("Active navigation goal must contain finite world XY.")
        if action not in {"NAVIGATE", "BACKTRACK", "STOP"}:
            raise ValueError(f"Unsupported active navigation action {action!r}.")
        if float(execution_max_path_m) <= 0.0:
            raise ValueError("Active navigation maximum path length must be positive.")
        self.active_goal = ActiveNavigationGoal(
            action=action,
            decision_index=int(decision_index),
            goal_world_xy=goal.copy(),
            execution_max_path_m=float(execution_max_path_m),
            target_waypoint_id=target_waypoint_id,
            last_fmm_plan=initial_fmm_plan,
        )
        current_time_s = float(completed_step) * float(step_dt)
        self._last_online_map_time_s = current_time_s
        self._last_online_replan_time_s = current_time_s
        self._reset_collision_window()

    def _maybe_update_online_navigation(
        self,
        camera_rig: FourViewCameraRig,
        *,
        completed_step: int,
        step_dt: float,
        path_follower,
        command_controller,
        switch_state,
        hot_swap_path: Callable[[FMMPlan, float], bool] | None,
        applied_velocity_command: np.ndarray | None,
    ) -> None:
        """Fuse a low-rate observation and replan to the unchanged model goal."""

        state = self.global_map_state
        active = self.active_goal
        if state is None or active is None:
            self._fail(
                "online navigation has no cumulative map or active goal.",
                path_follower,
                command_controller,
                switch_state,
            )
            return

        current_time_s = float(completed_step) * float(step_dt)
        collision_detected = self._update_online_collision_map(
            completed_step=completed_step,
            current_time_s=current_time_s,
            path_follower=path_follower,
            switch_state=switch_state,
            applied_velocity_command=applied_velocity_command,
        )

        map_interval_s = float(
            self.args_cli.lavira_online_mapping_interval_s
        )
        map_due = (
            self._last_online_map_time_s is None
            or current_time_s - self._last_online_map_time_s
            >= map_interval_s - 1.0e-9
        )
        map_updated = False
        if map_due:
            # Advance the scheduler even on a failed render so a transient
            # camera error cannot trigger an expensive retry every physics step.
            self._last_online_map_time_s = current_time_s
            try:
                bundle = camera_rig.capture(
                    sim_step=int(completed_step),
                    timestamp=current_time_s,
                )
                state.integrate_bundle(bundle)
                self.online_map_update_count += 1
                map_updated = True
                self._record_online_event(
                    "map_update",
                    completed_step,
                    {
                        "bundle_id": int(bundle.bundle_id),
                        "global_update_count": int(state.update_count),
                        "explored_cells": int(
                            np.count_nonzero(state.full_map[1])
                        ),
                        "obstacle_cells": int(
                            np.count_nonzero(state.full_map[0])
                        ),
                        "collision_cells": int(
                            np.count_nonzero(state.collision_map)
                        ),
                    },
                )
                print(
                    "[LAVIRA ONLINE] fused execution observation: "
                    f"bundle={bundle.bundle_id} "
                    f"updates={state.update_count} "
                    f"explored={np.count_nonzero(state.full_map[1])}."
                )
            except Exception as exc:
                self._record_online_event(
                    "map_update_failed",
                    completed_step,
                    {"error": str(exc)},
                )
                print(
                    "[WARN] [LAVIRA ONLINE] map update failed; continuing the "
                    f"last validated path: {exc}"
                )

        replan_interval_s = float(
            self.args_cli.lavira_online_replan_interval_s
        )
        periodic_replan_due = (
            map_updated
            and (
                self._last_online_replan_time_s is None
                or current_time_s - self._last_online_replan_time_s
                >= replan_interval_s - 1.0e-9
            )
        )
        if not collision_detected and not periodic_replan_due:
            return

        self._last_online_replan_time_s = current_time_s
        try:
            if hot_swap_path is None:
                raise RuntimeError(
                    "runner did not provide an online FMM hot-swap callback."
                )
            if active.action == "BACKTRACK":
                planning_map = state.build_navigation_grid_map(
                    historical_target_world_xy=active.goal_world_xy
                )
            else:
                planning_map = state.build_navigation_grid_map(
                    stable_target_world_xy=active.goal_world_xy
                )
            if planning_map.safe_target_cell_rc is None:
                raise RuntimeError(
                    "active world goal has no reachable cell on the latest map."
                )
            fmm_plan = build_fmm_plan(
                planning_map,
                fmm_planner_config_from_args(self.args_cli),
            )
            if not bool(
                hot_swap_path(
                    fmm_plan,
                    active.execution_max_path_m,
                )
            ):
                raise RuntimeError(
                    "online FMM path was rejected by the active follower."
                )

            active.replan_count += 1
            active.last_replan_step = int(completed_step)
            active.last_fmm_plan = fmm_plan
            self.online_replan_count += 1
            if active.action == "NAVIGATE" and self.pending_waypoint is not None:
                self.pending_waypoint.safe_target_world_xy = np.asarray(
                    fmm_plan.goal_world_xy, dtype=np.float64
                ).copy()
                self.pending_waypoint.fmm_waypoints_world_xy = np.asarray(
                    fmm_plan.waypoints_world_xy, dtype=np.float64
                ).copy()
            elif active.action == "STOP":
                self.stop_goal_world_xy = np.asarray(
                    fmm_plan.goal_world_xy, dtype=np.float64
                ).copy()
                if self.stop_event is not None:
                    self.stop_event["stop_goal_world_xy"] = (
                        self.stop_goal_world_xy.tolist()
                    )
                    self.stop_event["online_replan_count"] = active.replan_count
                    self._save_stop_event()
            elif active.action == "BACKTRACK" and self.backtrack_event is not None:
                self.backtrack_event["online_replan_count"] = active.replan_count
                self.backtrack_event["latest_path_length_m"] = (
                    fmm_plan.path_length_m
                )
                self._save_backtrack_event()

            self._record_online_event(
                "fmm_replan",
                completed_step,
                {
                    "trigger": (
                        "collision"
                        if collision_detected
                        else "periodic"
                    ),
                    "action": active.action,
                    "goal_world_xy": active.goal_world_xy.tolist(),
                    "safe_goal_world_xy": (
                        fmm_plan.goal_world_xy.tolist()
                    ),
                    "path_length_m": float(fmm_plan.path_length_m),
                    "waypoint_count": int(
                        fmm_plan.waypoints_world_xy.shape[0]
                    ),
                    "replan_count": int(active.replan_count),
                },
            )
            self._save_online_latest_debug(planning_map, fmm_plan)
            print(
                "[LAVIRA ONLINE] replanned unchanged goal: "
                f"action={active.action} trigger="
                f"{'collision' if collision_detected else 'periodic'} "
                f"count={active.replan_count} "
                f"path={fmm_plan.path_length_m:.3f}m."
            )
        except Exception as exc:
            self._record_online_event(
                "fmm_replan_failed",
                completed_step,
                {
                    "trigger": (
                        "collision"
                        if collision_detected
                        else "periodic"
                    ),
                    "error": str(exc),
                },
            )
            if collision_detected:
                self._fail(
                    (
                        "collision-triggered online FMM replan failed: "
                        f"{exc}"
                    ),
                    path_follower,
                    command_controller,
                    switch_state,
                )
                return
            print(
                "[WARN] [LAVIRA ONLINE] periodic FMM replan failed; "
                f"continuing the last validated path: {exc}"
            )

    def _update_online_collision_map(
        self,
        *,
        completed_step: int,
        current_time_s: float,
        path_follower,
        switch_state,
        applied_velocity_command: np.ndarray | None,
    ) -> bool:
        """Infer a persistent collision only from sustained commanded non-progress."""

        state = self.global_map_state
        pose = path_follower.current_robot_pose()
        command = (
            None
            if applied_velocity_command is None
            else np.asarray(applied_velocity_command, dtype=np.float64).reshape(-1)
        )
        valid_motion_state = (
            state is not None
            and getattr(pose, "valid", False)
            and getattr(switch_state, "active_mode", None) == "locomotion"
            and getattr(switch_state, "transition_mode", None) == "none"
            and command is not None
            and command.size >= 2
            and np.all(np.isfinite(command[:2]))
        )
        if not valid_motion_state:
            self._reset_collision_window()
            return False

        planar_command = command[:2]
        command_speed = float(np.linalg.norm(planar_command))
        if command_speed < float(
            self.args_cli.lavira_collision_command_speed_m_s
        ):
            self._reset_collision_window()
            return False

        direction_base = planar_command / command_speed
        cy = math.cos(float(pose.yaw))
        sy = math.sin(float(pose.yaw))
        direction_world = np.array(
            [
                cy * direction_base[0] - sy * direction_base[1],
                sy * direction_base[0] + cy * direction_base[1],
            ],
            dtype=np.float64,
        )
        current_xy = np.array([float(pose.x), float(pose.y)], dtype=np.float64)
        if (
            self._collision_window_start_time_s is None
            or self._collision_window_start_xy is None
        ):
            self._collision_window_start_time_s = float(current_time_s)
            self._collision_window_start_xy = current_xy
            self._collision_direction_world_xy = direction_world
            return False

        previous_direction = self._collision_direction_world_xy
        if (
            previous_direction is not None
            and float(np.dot(previous_direction, direction_world))
            < math.cos(math.radians(45.0))
        ):
            # A strong commanded-direction change is a turn/reorientation, not
            # one continuous failed translation window.
            self._collision_window_start_time_s = float(current_time_s)
            self._collision_window_start_xy = current_xy
            self._collision_direction_world_xy = direction_world
            return False
        self._collision_direction_world_xy = direction_world
        elapsed = (
            float(current_time_s) - self._collision_window_start_time_s
        )
        if elapsed < float(self.args_cli.lavira_collision_window_s):
            return False

        progress = float(
            np.linalg.norm(current_xy - self._collision_window_start_xy)
        )
        if progress >= float(
            self.args_cli.lavira_collision_min_progress_m
        ):
            self._collision_window_start_time_s = float(current_time_s)
            self._collision_window_start_xy = current_xy
            return False

        direction = self._collision_direction_world_xy
        collision_world_xy = current_xy + direction * float(
            self.args_cli.lavira_collision_mark_distance_m
        )
        added_cells = state.mark_collision_world_xy(
            collision_world_xy,
            radius_m=float(
                self.args_cli.lavira_collision_mark_radius_m
            ),
        )
        self.online_collision_count += 1
        self._record_online_event(
            "collision",
            completed_step,
            {
                "robot_world_xy": current_xy.tolist(),
                "collision_world_xy": collision_world_xy.tolist(),
                "command_speed_m_s": command_speed,
                "window_s": elapsed,
                "progress_m": progress,
                "new_collision_cells": int(added_cells),
                "collision_count": int(self.online_collision_count),
            },
        )
        print(
            "[WARN] [LAVIRA ONLINE] commanded non-progress marked collision: "
            f"progress={progress:.3f}m/{elapsed:.2f}s "
            f"point={collision_world_xy.tolist()} "
            f"new_cells={added_cells}."
        )
        self._collision_window_start_time_s = float(current_time_s)
        self._collision_window_start_xy = current_xy
        return True

    def _reset_collision_window(self) -> None:
        self._collision_window_start_time_s = None
        self._collision_window_start_xy = None
        self._collision_direction_world_xy = None

    def _record_online_event(
        self,
        event_type: str,
        completed_step: int,
        payload: dict,
    ) -> None:
        event = {
            "event_index": len(self.online_events),
            "event_type": str(event_type),
            "completed_step": int(completed_step),
            **payload,
        }
        self.online_events.append(event)
        if self.current_probe is None or self.current_probe.output_dir is None:
            return
        path = Path(self.current_probe.output_dir) / "online_navigation.json"
        path.write_text(
            json.dumps(
                {
                    "enabled": self.online_navigation,
                    "map_update_count": self.online_map_update_count,
                    "replan_count": self.online_replan_count,
                    "collision_count": self.online_collision_count,
                    "active_goal": (
                        self.active_goal.to_dict()
                        if self.active_goal is not None
                        else None
                    ),
                    "events": self.online_events,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _save_online_latest_debug(
        self,
        planning_map: NavigationGridMap,
        fmm_plan: FMMPlan,
    ) -> None:
        if (
            self.current_probe is None
            or self.current_probe.output_dir is None
            or self.global_map_state is None
        ):
            return
        output_dir = Path(self.current_probe.output_dir) / "online_latest"
        save_lavira_global_map_debug(
            output_dir,
            self.global_map_state,
            planning_map,
        )
        save_fmm_plan_debug(output_dir, planning_map, fmm_plan)

    def _start_stop_action(
        self,
        probe: NavigationDecisionOfflineProbe,
        *,
        decision_index: int,
        completed_step: int,
        step_dt: float,
        path_follower,
        command_controller,
        switch_state,
        start_path: Callable[[object], bool],
    ) -> None:
        """用 NAVIGATE 的同一条 FMM 链执行 LaViRA 最终接近。"""
        if probe.fmm_plan is None:
            raise RuntimeError(
                f"decision {decision_index} STOP produced no executable FMM plan."
            )
        waypoints = np.asarray(
            probe.fmm_plan.waypoints_world_xy, dtype=np.float64
        )
        if (
            waypoints.ndim != 2
            or waypoints.shape[0] < 1
            or waypoints.shape[1] != 2
            or not np.all(np.isfinite(waypoints))
        ):
            raise RuntimeError(
                f"decision {decision_index} STOP has invalid FMM waypoints "
                f"shape {waypoints.shape}."
            )

        self.pending_waypoint = None
        self.stop_goal_world_xy = waypoints[-1].copy()
        initial_distance = self._current_stop_distance(path_follower)
        self.stop_event = {
            "status": "accepted",
            "decision_index": int(decision_index),
            "action": "STOP",
            "direction": probe.response.direction,
            "target": probe.response.target,
            "bbox_2d": list(probe.response.bbox_2d),
            "stop_goal_world_xy": self.stop_goal_world_xy.tolist(),
            "reached_threshold_m": self.stop_reached_threshold_m,
            "initial_distance_m": initial_distance,
            "last_distance_m": initial_distance,
            "actual_distance_m": None,
            "threshold_reached_step": None,
            "completion_step": None,
            "robot_standing": False,
            "model_stop_completed": False,
            # A model STOP is not simulator ground-truth task success.
            "task_success": None,
        }
        self._save_stop_event()
        self.active_execution_action = "STOP"
        self.active_execution_decision_index = int(decision_index)
        self._begin_active_goal(
            action="STOP",
            decision_index=decision_index,
            goal_world_xy=self.stop_goal_world_xy,
            execution_max_path_m=float(
                getattr(
                    self.args_cli,
                    "fmm_execute_max_path_m",
                    6.0,
                )
            ),
            completed_step=completed_step,
            step_dt=step_dt,
            initial_fmm_plan=probe.fmm_plan,
        )

        if (
            initial_distance is not None
            and initial_distance < self.stop_reached_threshold_m
        ):
            self._finish_stop_path_approach(
                completed_step=completed_step,
                stop_distance=initial_distance,
                path_follower=path_follower,
                command_controller=command_controller,
                switch_state=switch_state,
            )
            print(
                f"[LAVIRA EPISODE] decision_{decision_index:03d} STOP target "
                f"already within LaViRA threshold: distance={initial_distance:.3f}m "
                f"threshold={self.stop_reached_threshold_m:.3f}m."
            )
            return

        started = bool(start_path(probe))
        if not started:
            raise RuntimeError(
                f"decision {decision_index} STOP FMM path was not accepted "
                "for locomotion."
            )
        self.execution_started_step = int(completed_step)
        self.state = EpisodeState.EXECUTING
        distance_text = (
            "unknown" if initial_distance is None else f"{initial_distance:.3f}m"
        )
        print(
            f"[LAVIRA EPISODE] decision_{decision_index:03d} STOP final approach "
            f"started with the NAVIGATE FMM path: target={probe.response.target!r} "
            f"direction={probe.response.direction} distance={distance_text} "
            f"threshold={self.stop_reached_threshold_m:.3f}m."
        )

    def _current_stop_distance(self, path_follower) -> float | None:
        if self.stop_goal_world_xy is None:
            return None
        pose = path_follower.current_robot_pose()
        if not getattr(pose, "valid", False):
            return None
        return float(
            np.hypot(
                float(self.stop_goal_world_xy[0]) - float(pose.x),
                float(self.stop_goal_world_xy[1]) - float(pose.y),
            )
        )

    def _update_stop_distance(self, stop_distance: float) -> None:
        if self.stop_event is None:
            return
        self.stop_event["last_distance_m"] = float(stop_distance)

    def _finish_stop_path_approach(
        self,
        *,
        completed_step: int,
        stop_distance: float,
        path_follower,
        command_controller,
        switch_state,
    ) -> None:
        if self.stop_event is None:
            raise RuntimeError("STOP threshold reached without an active STOP event.")
        path_follower.stop("LaViRA STOP target threshold reached")
        command_controller.zero()
        self._request_stand_if_needed(switch_state)
        self.stop_event.update(
            {
                "status": "threshold_reached",
                "last_distance_m": float(stop_distance),
                "actual_distance_m": float(stop_distance),
                "threshold_reached_step": int(completed_step),
            }
        )
        self._save_stop_event()
        self._stand_stable_elapsed = 0.0
        self.state = EpisodeState.WAITING_FOR_STAND
        print(
            "[LAVIRA EPISODE] STOP final-approach threshold reached: "
            f"distance={stop_distance:.3f}m "
            f"threshold={self.stop_reached_threshold_m:.3f}m; "
            "waiting for stable stand."
        )

    def _complete_stop(self, completed_step: int) -> None:
        if self.stop_event is None:
            raise RuntimeError("Stable stand reached without an active STOP event.")
        self.stop_event.update(
            {
                "status": "completed",
                "completion_step": int(completed_step),
                "robot_standing": True,
                "model_stop_completed": True,
            }
        )
        self.terminal_response = (
            self.current_probe.response.to_dict()
            if self.current_probe is not None
            and self.current_probe.response is not None
            else None
        )
        self.state = EpisodeState.STOPPED
        self.completed = True
        self._save_stop_event()
        print(
            "[LAVIRA EPISODE] STOP completed: robot is standing and no further "
            "decision requests will be sent. This confirms model STOP execution, "
            "not simulator ground-truth task success."
        )

    def _commit_pending_waypoint(self, completed_step: int) -> None:
        if self.pending_waypoint is None:
            raise RuntimeError("No pending waypoint is available for history commit.")
        record = self.pending_waypoint
        record.execution_status = "arrived"
        record.arrival_step = int(completed_step)
        self.history.append(record)
        self.pending_waypoint = None
        self._save_history_commit(record)
        print(
            "[LAVIRA EPISODE] committed history waypoint: "
            f"waypoint={record.waypoint_id} target={record.target!r} "
            f"decision_step={record.decision_step} arrival_step={record.arrival_step}."
        )

    def _save_history_commit(self, record: RuntimeWaypointRecord) -> None:
        if self.current_probe is None or self.current_probe.output_dir is None:
            return
        path = Path(self.current_probe.output_dir) / "history_commit.json"
        path.write_text(
            json.dumps(record.to_local_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _complete_backtrack(self, completed_step: int) -> None:
        if self.backtrack_target_waypoint is None or self.backtrack_event is None:
            raise RuntimeError("BACKTRACK completion has no active target waypoint.")
        target = self.backtrack_target_waypoint
        if not 0 <= target < len(self.history):
            raise RuntimeError(
                f"BACKTRACK target {target} disappeared from committed history."
            )
        if len(self.history) != target + 1:
            raise RuntimeError(
                "BACKTRACK history was not truncated when the action was accepted."
            )
        self.backtrack_event.update(
            {
                "status": "arrived",
                "history_count_after": len(self.history),
                "completion_step": int(completed_step),
            }
        )
        self._save_backtrack_event()
        print(
            "[LAVIRA EPISODE] BACKTRACK completed: "
            f"waypoint={target} history={self.backtrack_event['history_count_before']}"
            f"->{len(self.history)} "
            f"arrival_step={completed_step}."
        )

    def _save_backtrack_event(self) -> None:
        if (
            self.backtrack_event is None
            or self.current_probe is None
            or self.current_probe.output_dir is None
        ):
            return
        path = Path(self.current_probe.output_dir) / "backtrack_execution.json"
        path.write_text(
            json.dumps(self.backtrack_event, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _save_stop_event(self) -> None:
        if (
            self.stop_event is None
            or self.current_probe is None
            or self.current_probe.output_dir is None
        ):
            return
        path = Path(self.current_probe.output_dir) / "stop_execution.json"
        path.write_text(
            json.dumps(self.stop_event, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _clear_active_execution(self) -> None:
        self.execution_started_step = None
        self.active_execution_action = None
        self.active_execution_decision_index = None
        self.backtrack_target_waypoint = None
        self.backtrack_request = None
        self.stop_goal_world_xy = None
        self.active_goal = None
        self._last_online_map_time_s = None
        self._last_online_replan_time_s = None
        self._reset_collision_window()

    def _save_controller_status(self) -> None:
        if self.current_probe is None or self.current_probe.output_dir is None:
            return
        payload = {
            "state": self.state,
            "navigation_map_mode": self.navigation_map_mode,
            "global_map": (
                self.global_map_state.to_metadata()
                if self.global_map_state is not None
                and self.global_map_state.initialized
                else None
            ),
            "max_decisions": self.max_decisions,
            "decisions_completed": self.decisions_completed,
            "history_count": len(self.history),
            "terminal_response": self.terminal_response,
            "pending_action": self.pending_action,
            "active_execution_action": self.active_execution_action,
            "online_navigation": {
                "enabled": self.online_navigation,
                "map_update_count": self.online_map_update_count,
                "replan_count": self.online_replan_count,
                "collision_count": self.online_collision_count,
                "active_goal": (
                    self.active_goal.to_dict()
                    if self.active_goal is not None
                    else None
                ),
            },
            "backtrack_event": self.backtrack_event,
            "stop_event": self.stop_event,
            "failure_reason": self.failure_reason,
        }
        path = Path(self.current_probe.output_dir) / "bounded_episode_status.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _stand_ready(switch_state) -> bool:
        return (
            switch_state.active_mode == "stand"
            and switch_state.transition_mode == "none"
        )

    @classmethod
    def _request_stand_if_needed(cls, switch_state) -> None:
        if not cls._stand_ready(switch_state):
            switch_state.request_stand()

    def _fail(
        self,
        reason: str,
        path_follower,
        command_controller,
        switch_state,
    ) -> None:
        self.failure_reason = str(reason)
        self.state = EpisodeState.FAILED
        if (
            self.backtrack_event is not None
            and self.backtrack_event.get("status") == "accepted"
        ):
            self.backtrack_event.update(
                {
                    "status": "failed",
                    "failure_reason": self.failure_reason,
                }
            )
            self._save_backtrack_event()
        if (
            self.stop_event is not None
            and self.stop_event.get("status") in {"accepted", "threshold_reached"}
        ):
            self.stop_event.update(
                {
                    "status": "failed",
                    "failure_reason": self.failure_reason,
                    "model_stop_completed": False,
                }
            )
            self._save_stop_event()
        path_follower.stop("LaViRA bounded episode failed")
        command_controller.zero()
        self._request_stand_if_needed(switch_state)
        self._save_controller_status()
        print(f"[WARN] [LAVIRA EPISODE] {self.failure_reason}")

    def report_status(self) -> None:
        if not self.enabled or self.completed:
            return
        if self.state == EpisodeState.FAILED:
            print(
                "[WARN] LaViRA bounded episode failed: "
                f"{self.failure_reason or 'unknown failure'}"
            )
            return
        print(
            "[WARN] LaViRA bounded episode ended before its terminal response: "
            f"state={self.state} decisions={self.decisions_completed}/"
            f"{self.max_decisions} history={len(self.history)}."
        )


# Backward-compatible import name for the runner and any existing local scripts.
LaViRAHistoryProbeController = LaViRABoundedEpisodeController
