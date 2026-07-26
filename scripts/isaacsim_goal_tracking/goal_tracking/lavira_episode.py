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
- 规划或执行失败会进入 FAILED；当前不会自动重新抓图、重规划或恢复 episode。
"""

from dataclasses import dataclass
from datetime import datetime
import json
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

    def _start_stop_action(
        self,
        probe: NavigationDecisionOfflineProbe,
        *,
        decision_index: int,
        completed_step: int,
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

    def _save_controller_status(self) -> None:
        if self.current_probe is None or self.current_probe.output_dir is None:
            return
        payload = {
            "state": self.state,
            "max_decisions": self.max_decisions,
            "decisions_completed": self.decisions_completed,
            "history_count": len(self.history),
            "terminal_response": self.terminal_response,
            "pending_action": self.pending_action,
            "active_execution_action": self.active_execution_action,
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
