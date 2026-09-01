from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import threading
import time
from typing import Protocol

import numpy as np

from .backtrack import (
    StoredReverseRoute,
    build_stored_reverse_route,
    next_route_checkpoint_index,
)
from .iplanner_client import IPlannerClient
from .local_projection import LocalTargetProjection, project_selected_view_target
from .local_trajectory import (
    LocalFollowerConfig,
    LocalTrajectoryFollower,
    truncate_trajectory_for_safety,
)
from .map_progress import SparseEpisodeExplorationMap
from .model_client import (
    CombinedModelClient,
    CompletedWaypoint,
    build_model_history,
    response_debug_dict,
    select_model_history_records,
)
from .model_contract import NavigationDecisionResponse
from .odometry import (
    NullOdometryProvider,
    OdometryProvider,
    Pose2D,
    fixed_points_to_local,
)
from .rotation import TimedFixedSpeedRotation, YawProvider
from .session_client import (
    G3DecisionSupervision,
    G3ExecutionControl,
    G3SessionClient,
)
from .types import DIRECTION_ORDER, PanoramaBundle, ViewFrame


def _create_unique_run_output_dir(base_dir: Path, session_id: str) -> Path:
    """为每次启动创建独立目录，避免同名 session 的实验文件互相覆盖。"""

    session_dir = Path(base_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidate = session_dir / f"run_{timestamp}"
    suffix = 1
    while True:
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = session_dir / f"run_{timestamp}_{suffix:02d}"
            suffix += 1


class CameraBackend(Protocol):
    """状态机所需的最小相机接口，仿真或真实相机都可以实现。"""

    def capture_panorama(self, sim_step: int, timestamp: float) -> PanoramaBundle:
        ...

    def capture_forward(self, sim_step: int, timestamp: float) -> ViewFrame:
        ...


class EpisodeState:
    """导航回合状态常量。

    一次典型循环为：预热 → 拍全景并决策 → 等待运动模式 → 原地转向 → 等待稳定
    → 拍前视图并规划 → 等待运动模式 → 执行轨迹 → 恢复站立 → 下一次决策。
    ``STOPPED`` 和 ``FAILED`` 是两个终止状态。
    """

    WARMUP = "warmup"
    CAPTURE_AND_DECIDE = "capture_and_decide"
    WAIT_PANORAMA_LOCOMOTION = "wait_panorama_locomotion"
    PANORAMA_ROTATING = "panorama_rotating"
    PANORAMA_DECIDE = "panorama_decide"
    WAITING_DECISION = "waiting_decision"
    WAIT_ROTATION_LOCOMOTION = "wait_rotation_locomotion"
    ROTATING = "rotating"
    ROTATION_SETTLE = "rotation_settle"
    PLAN_AFTER_ROTATION = "plan_after_rotation"
    WAIT_EXECUTION_LOCOMOTION = "wait_execution_locomotion"
    EXECUTING = "executing"
    WAIT_BACKTRACK_LOCOMOTION = "wait_backtrack_locomotion"
    BACKTRACK_ROTATING = "backtrack_rotating"
    BACKTRACK_PLANNING = "backtrack_planning"
    BACKTRACK_EXECUTING = "backtrack_executing"
    WAIT_ACTION_STAND = "wait_action_stand"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class EpisodeConfig:
    """一次端到端导航回合的配置。

    包含模型指令、独立的历史/决策次数上限、转向/站立等待时间、深度范围、安全
    距离、动作超时和日志目录。``history_max_waypoints=None`` 保留全部文字历史，
    ``max_decisions=None`` 表示持续运行到模型 STOP、失败或外层程序退出。所有
    时间单位为秒、距离单位为米、角速度单位为 rad/s。
    """

    session_id: str
    instruction: str
    warmup_steps: int = 5
    history_max_waypoints: int | None = None
    max_decisions: int | None = None
    rotation_speed_rad_s: float = 0.4
    rotation_duration_scale: float = 1.0
    rotation_settle_s: float = 0.5
    post_action_stand_s: float = 0.8
    safe_distance_m: float = 0.5
    min_depth_m: float = 0.1
    max_depth_m: float = 5.0
    action_timeout_s: float = 60.0
    motion_window_s: float = 1.0
    pose_frame_id: str = "local_odom"
    frame_epoch: int = 0
    single_forward_panorama: bool = False
    enable_backtrack: bool = False
    backtrack_max_path_m: float = 6.0
    backtrack_start_tolerance_m: float = 1.0
    backtrack_segment_length_m: float = 1.0
    backtrack_goal_tolerance_m: float = 0.35
    backtrack_heading_tolerance_rad: float = 0.20
    backtrack_breadcrumb_spacing_m: float = 0.15
    output_dir: Path | None = None

    def validated(self) -> "EpisodeConfig":
        """集中验证配置，避免非法参数在运行中途才导致机器人行为异常。"""

        if not self.session_id.strip() or not self.instruction.strip():
            raise ValueError("Session id and instruction must not be empty.")
        if not self.pose_frame_id.strip():
            raise ValueError("Pose frame id must not be empty.")
        if (
            isinstance(self.frame_epoch, bool)
            or not isinstance(self.frame_epoch, int)
            or self.frame_epoch < 0
        ):
            raise ValueError("Frame epoch must be a non-negative integer.")
        if not isinstance(self.enable_backtrack, bool):
            raise ValueError("enable_backtrack must be a boolean.")
        if self.warmup_steps < 0:
            raise ValueError("Warmup must be >=0.")
        if self.history_max_waypoints is not None and self.history_max_waypoints <= 0:
            raise ValueError(
                "History max waypoints must be >0 when a limit is configured."
            )
        if self.max_decisions is not None and self.max_decisions <= 0:
            raise ValueError("Max decisions must be >0 when a limit is configured.")
        positive = (
            self.rotation_speed_rad_s,
            self.rotation_duration_scale,
            self.rotation_settle_s,
            self.post_action_stand_s,
            self.max_depth_m,
            self.action_timeout_s,
            self.motion_window_s,
            self.backtrack_max_path_m,
            self.backtrack_segment_length_m,
            self.backtrack_goal_tolerance_m,
            self.backtrack_heading_tolerance_rad,
            self.backtrack_breadcrumb_spacing_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Episode timing/range values must be finite and positive.")
        if not 0.0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("Episode depth range must satisfy 0 < min < max.")
        if self.safe_distance_m < 0.0:
            raise ValueError("Safe distance must be non-negative.")
        if (
            not math.isfinite(self.backtrack_start_tolerance_m)
            or self.backtrack_start_tolerance_m < 0.0
        ):
            raise ValueError(
                "BACKTRACK start tolerance must be finite and non-negative."
            )
        return self


@dataclass(frozen=True)
class EpisodeUpdate:
    """状态机每个仿真/控制周期返回给外层控制程序的结果。

    ``command`` 是三维速度命令，``desired_mode`` 告诉外层应切换到站立还是行走，
    ``completed`` 表示整个回合已经停止或失败。
    """

    command: np.ndarray
    desired_mode: str
    state: str
    completed: bool
    failure_reason: str | None


@dataclass
class _PendingAction:
    """模型已决定、但尚未完成执行的一次动作所需的临时上下文。"""

    response: NavigationDecisionResponse
    panorama: PanoramaBundle
    projection: LocalTargetProjection
    initial_forward_frame_id: int
    decision_pose: Pose2D | None
    action_source: str = "NAVIGATOR"
    post_rotation_forward: ViewFrame | None = None


@dataclass
class _BacktrackAction:
    """One accepted model BACKTRACK and its physical stored-reverse route."""

    response: NavigationDecisionResponse
    route: StoredReverseRoute
    wire_waypoint_id: int
    target_waypoint_id: int
    history_count_before: int
    decision_pose: Pose2D
    action_source: str = "NAVIGATOR"
    route_cursor: int = 0
    checkpoint_index: int = 0
    segments_completed: int = 0

    @property
    def checkpoint_world_xy(self) -> np.ndarray:
        return self.route.points_world_xy[self.checkpoint_index].copy()


@dataclass
class _PanoramaSweep:
    """One four-view panorama captured by rotating a single forward RGB-D."""

    views: dict[str, ViewFrame]
    decision_pose: Pose2D | None
    capture_poses: dict[str, Pose2D | None]
    quarter_turns_completed: int = 0


@dataclass
class _DecisionTask:
    """One in-flight model request for the physical single-camera path."""

    decision_index: int
    panorama: PanoramaBundle
    decision_pose: Pose2D | None
    done: threading.Event
    response: NavigationDecisionResponse | None = None
    raw_response: dict | None = None
    error: Exception | None = None


class LocalEndToEndEpisode:
    """串联视觉模型、目标投影、转向、iPlanner 和跟随器的导航状态机。

    外部控制循环每步调用一次 ``update``，状态机只返回期望模式和速度，不直接
    操纵机器人。这样同一流程既可接 Isaac Sim，也可接真实 G1 控制后端。
    """

    def __init__(
        self,
        config: EpisodeConfig,
        follower_config: LocalFollowerConfig,
        *,
        camera: CameraBackend,
        model: CombinedModelClient,
        planner: IPlannerClient,
        session_client: G3SessionClient | None = None,
        odometry: OdometryProvider | None = None,
        yaw_provider: YawProvider | None = None,
        exploration_map: SparseEpisodeExplorationMap | None = None,
    ):
        """装配各组件、初始化所有计时器，并按需创建本回合日志目录。"""

        self.config = config.validated()
        self.camera = camera
        self.model = model
        self.planner = planner
        self.session_client = session_client
        self.exploration_map = exploration_map
        self.rotation = TimedFixedSpeedRotation(
            config.rotation_speed_rad_s,
            config.rotation_duration_scale,
            yaw_provider=yaw_provider,
        )
        self.odometry = odometry or NullOdometryProvider()
        self.follower = LocalTrajectoryFollower(follower_config, self.odometry)
        self.follower_config = follower_config.validated()
        self.state = EpisodeState.WARMUP
        self.history: list[CompletedWaypoint] = []
        self._wire_history_records: tuple[CompletedWaypoint, ...] = ()
        self.decision_index = 0
        self.pending: _PendingAction | None = None
        self.backtrack: _BacktrackAction | None = None
        self.panorama_sweep: _PanoramaSweep | None = None
        self._decision_task: _DecisionTask | None = None
        self.next_panorama_bundle_id = 0
        self._active_world_trace: list[np.ndarray] = []
        self.failure_reason: str | None = None
        self.session_success_reason: str | None = None
        self.rotation_settle_elapsed_s = 0.0
        self.action_stand_elapsed_s = 0.0
        self.action_elapsed_s = 0.0
        self.motion_window_index = 0
        self.next_motion_window_elapsed_s = self.config.motion_window_s
        self.pose_frame_id = self.config.pose_frame_id
        self.frame_epoch = int(self.config.frame_epoch)
        self._motion_window_start_pose: Pose2D | None = None
        self._motion_window_start_goal_distance_m: float | None = None
        self._map_window_explored_before: int | None = None
        self._map_update_failures = 0
        self.replan_failures = 0
        self.last_fear: float | None = None
        self.iplanner_history: list[str] = []
        self._commit_pending_after_stand = True
        self._action_completion_status = "COMPLETED"
        self._action_completion_reason = "local_action_completed"
        self._action_planner_result = "REACHED"
        self._action_reached_local_goal = True
        self._action_final_pose: Pose2D | None = None
        self._reported_action_complete_indices: set[int] = set()
        self._server_waypoint_to_local_index: dict[int, int] = {}
        self._recovery_expected = False
        self._safe_stop_requested = False
        self.session_failure_reason: str | None = None
        self._remote_session_active = False
        self._remote_session_ended = False
        self._last_logged_state: str | None = None
        self.output_dir = (
            None
            if config.output_dir is None
            else _create_unique_run_output_dir(config.output_dir, config.session_id)
        )
        if self.output_dir is not None:
            print(f"[LOCAL-VLN] output directory: {self.output_dir}")

    @property
    def remote_session_active(self) -> bool:
        """Whether this episode currently owns an ACTIVE server-side session."""

        return self._remote_session_active and not self._remote_session_ended

    def start_remote_session(self) -> None:
        """Run health/start_session once before the first panorama decision."""

        if self.session_client is None:
            return
        if self.remote_session_active:
            raise RuntimeError("The remote G3 session is already active.")
        if self.exploration_map is None:
            raise RuntimeError(
                "Phase-three execution reports require the RGB-D exploration map."
            )
        pose = self.odometry.get_pose()
        if pose is None:
            raise RuntimeError(
                "Phase-three execution reports require Isaac/SLAM odometry."
            )
        pose.validated()
        health = self.session_client.health_check()
        self._save_json("g3_health.json", health)
        started, raw_started = self.session_client.start_session(
            session_id=self.config.session_id,
            instruction=self.config.instruction,
        )
        self._remote_session_active = True
        self._remote_session_ended = False
        self._save_json("g3_session_started.json", raw_started)
        print(
            "[LOCAL-VLN G3] session ACTIVE: "
            f"stage_plan_id={started.stage_plan_id} "
            f"stages={started.stage_total}"
        )

    def end_remote_session(self, *, status: str, reason: str) -> None:
        """End the owned server-side session once; safe to call during cleanup."""

        if self.session_client is None or not self.remote_session_active:
            return
        ended, raw_ended = self.session_client.end_session(
            status=status,
            reason=reason,
        )
        self._remote_session_ended = True
        self._remote_session_active = False
        self._save_json("g3_session_ended.json", raw_ended)
        print(
            "[LOCAL-VLN G3] session ENDED: "
            f"final_status={ended.final_status} reason={ended.reason!r}"
        )

    @property
    def completed(self) -> bool:
        """回合是否已进入成功停止或失败这两种终态之一。"""

        return self.state in {EpisodeState.STOPPED, EpisodeState.FAILED}

    def update(
        self,
        *,
        completed_step: int,
        step_dt: float,
        timestamp: float,
        applied_command: np.ndarray,
        stand_ready: bool,
        locomotion_ready: bool,
    ) -> EpisodeUpdate:
        """推进状态机一个控制周期。

        参数中的 ``stand_ready``/``locomotion_ready`` 由外层机器人模式控制器给出；
        状态机只有在对应模式准备好后才发运动命令。``applied_command`` 是上一周期
        实际执行速度，供无里程计的航位推算使用。
        """

        if step_dt <= 0.0:
            raise ValueError("Episode step_dt must be positive.")
        command = np.zeros(3, dtype=np.float64)

        if self.state == EpisodeState.WAITING_DECISION:
            try:
                self._poll_decision_request()
            except Exception as exc:
                self._fail(str(exc))
            self._log_state_transition()
            return self._result(command)

        if self.state in {
            EpisodeState.WAIT_BACKTRACK_LOCOMOTION,
            EpisodeState.BACKTRACK_ROTATING,
            EpisodeState.BACKTRACK_PLANNING,
            EpisodeState.BACKTRACK_EXECUTING,
        }:
            try:
                command = self._update_backtrack(
                    completed_step=completed_step,
                    step_dt=step_dt,
                    timestamp=timestamp,
                    applied_command=applied_command,
                    locomotion_ready=locomotion_ready,
                )
            except Exception as exc:
                self._fail(str(exc))
                command.fill(0.0)
            self._log_state_transition()
            return self._result(command)

        if self.state in {
            EpisodeState.WAIT_PANORAMA_LOCOMOTION,
            EpisodeState.PANORAMA_ROTATING,
            EpisodeState.PANORAMA_DECIDE,
        }:
            try:
                command = self._update_single_camera_panorama(
                    completed_step=completed_step,
                    step_dt=step_dt,
                    timestamp=timestamp,
                    locomotion_ready=locomotion_ready,
                )
            except Exception as exc:
                self._fail(str(exc))
                command.fill(0.0)
            self._log_state_transition()
            return self._result(command)

        # 默认命令始终为零；只有 ROTATING 或 EXECUTING 阶段会明确写入运动速度。
        try:
            if self.state == EpisodeState.WARMUP:
                # 等待若干帧让传感器稳定，并确认机器人处于站立模式。
                if completed_step >= self.config.warmup_steps and stand_ready:
                    self.state = EpisodeState.CAPTURE_AND_DECIDE

            if self.state == EpisodeState.CAPTURE_AND_DECIDE:
                # 单前向相机模式先拍 forward，再切到 locomotion 连续转一圈；
                # 旧的四相机同时采集只保留为显式关闭该模式时的诊断回退。
                if not stand_ready:
                    return self._result(command)
                if self.config.single_forward_panorama:
                    self._start_single_camera_panorama(completed_step, timestamp)
                else:
                    self._capture_and_decide(completed_step, timestamp)

            if self.state == EpisodeState.WAIT_ROTATION_LOCOMOTION:
                # 模式切换可能需要多个控制周期，确认完成后才开始累计旋转时间。
                if locomotion_ready:
                    self.state = EpisodeState.ROTATING

            if self.state == EpisodeState.ROTATING:
                if not locomotion_ready:
                    return self._result(command)
                rotation = self.rotation.update(step_dt)
                command[:] = (rotation.vx, rotation.vy, rotation.wz)
                if rotation.done:
                    command.fill(0.0)
                    self.rotation_settle_elapsed_s = 0.0
                    self.state = EpisodeState.ROTATION_SETTLE

            elif self.state == EpisodeState.ROTATION_SETTLE:
                # 转向结束后短暂停顿，让机身与相机画面稳定下来。
                self.rotation_settle_elapsed_s += step_dt
                if self.rotation_settle_elapsed_s >= self.config.rotation_settle_s:
                    self.state = EpisodeState.PLAN_AFTER_ROTATION

            if self.state == EpisodeState.PLAN_AFTER_ROTATION:
                self._capture_forward_and_plan(completed_step, timestamp)

            if self.state == EpisodeState.WAIT_EXECUTION_LOCOMOTION:
                if locomotion_ready:
                    self.state = EpisodeState.EXECUTING
                    self.action_elapsed_s = 0.0
                    self.motion_window_index = 0
                    self.next_motion_window_elapsed_s = self.config.motion_window_s
                    self._reset_motion_window_baseline()

            if self.state == EpisodeState.EXECUTING:
                # 执行阶段同时负责超时保护、跟随控制以及周期性视觉重规划。
                if not locomotion_ready:
                    return self._result(command)
                self.action_elapsed_s += step_dt
                self._record_world_trace()
                if self.action_elapsed_s > self.config.action_timeout_s:
                    print(
                        "[LOCAL-VLN WARN] local trajectory execution timed out; "
                        "ending this segment like Uni-LaViRA"
                    )
                    self._begin_action_finish(
                        status="FAILED",
                        reason="local_action_timeout",
                        planner_result="TIMEOUT",
                    )
                    return self._result(command)

                follower_output = self.follower.update(step_dt, applied_command)
                if follower_output.abort_reason is not None:
                    print(
                        "[LOCAL-VLN WARN] local trajectory ended: "
                        f"{follower_output.abort_reason}"
                    )
                    self._begin_action_finish(
                        status="FAILED",
                        reason=follower_output.abort_reason,
                        planner_result="EXECUTION_FAILED",
                    )
                    return self._result(command)
                if follower_output.reached:
                    self._begin_action_finish()
                    return self._result(command)
                command[:] = follower_output.command

                control = self._report_motion_window_if_due()
                if control is not None and control.control == "PREEMPT":
                    command.fill(0.0)
                    self._atomic_preempt_active_action(
                        "physical_failure_verifier_confirmed"
                    )
                    return self._result(command)

                if self.follower.needs_replan():
                    replan_started_at = time.time()
                    # 使用最新前视 RGB-D 和更新后的局部目标重新规划，修正环境变化。
                    try:
                        fresh_front, _fresh_pose = self._capture_forward_observation(
                            completed_step, timestamp
                        )
                        new_path, fear = self.planner.get_plan(
                            fresh_front,
                            self.follower.current_goal_local_xy,
                        )
                        if new_path is not None and len(new_path) > 1:
                            self.follower.replace_path(new_path)
                            self.last_fear = fear
                            self._save_iplanner_trajectory_image(
                                new_path,
                                fresh_front,
                                save_name=f"replan_{time.strftime('%H%M%S')}.jpg",
                                target_xy=self.follower.current_goal_local_xy,
                            )
                    except Exception as exc:
                        print(f"[LOCAL-VLN WARN] iPlanner replan failed: {exc}")
                    # Uni 记录的是规划开始前的 now，而不是 HTTP 返回后的时间。
                    self.follower.mark_replan_attempt(replan_started_at)

            elif self.state == EpisodeState.WAIT_ACTION_STAND:
                # 必须连续保持站立达到指定时间；中途失去 ready 就重新计时。
                if stand_ready:
                    self.action_stand_elapsed_s += step_dt
                    if self.action_stand_elapsed_s >= self.config.post_action_stand_s:
                        self._commit_or_stop()
                else:
                    self.action_stand_elapsed_s = 0.0

        # 任意未预期异常统一转成 FAILED 状态和零速度，防止机器人继续执行旧命令。
        except Exception as exc:
            self._fail(str(exc))
            command.fill(0.0)

        self._log_state_transition()
        return self._result(command)

    def _capture_forward_observation(
        self, completed_step: int, timestamp: float
    ) -> tuple[ViewFrame, Pose2D | None]:
        """Capture one physical forward frame and fuse it into the sparse map.

        Mapping is diagnostic evidence and must not stop an otherwise safe
        navigation action.  A bad frame is therefore logged and skipped while
        the RGB-D frame remains available to the model/iPlanner path.
        """

        frame = self.camera.capture_forward(completed_step, timestamp).validated()
        pose = self.odometry.get_pose()
        if self.exploration_map is not None:
            if pose is None:
                self._map_update_failures += 1
                print(
                    "[LOCAL-VLN MAP WARN] sparse map skipped a frame because "
                    "no Isaac/SLAM pose is available"
                )
            else:
                try:
                    integration = self.exploration_map.integrate(frame, pose)
                    print(
                        "[LOCAL-VLN MAP] integrated forward RGB-D: "
                        f"frame={integration.frame_id} "
                        f"new={integration.new_explored_cells} "
                        f"explored={integration.explored_cells}"
                    )
                except Exception as exc:
                    self._map_update_failures += 1
                    print(f"[LOCAL-VLN MAP WARN] sparse map update failed: {exc}")
        return frame, pose

    @staticmethod
    def _relabel_forward_frame(frame: ViewFrame, direction: str) -> ViewFrame:
        """Copy one physical forward-camera frame under a panorama direction."""

        if direction not in DIRECTION_ORDER:
            raise ValueError(f"Unsupported panorama direction {direction!r}.")
        checked = frame.validated()
        return ViewFrame(
            direction=direction,
            frame_id=checked.frame_id,
            sim_step=checked.sim_step,
            timestamp=checked.timestamp,
            rgb=np.asarray(checked.rgb).copy(),
            depth_m=np.asarray(checked.depth_m).copy(),
            K=np.asarray(checked.K).copy(),
        ).validated()

    def _start_single_camera_panorama(
        self, completed_step: int, timestamp: float
    ) -> None:
        """Capture forward, then request four consecutive left quarter-turns."""

        if self.panorama_sweep is not None:
            raise RuntimeError("A single-camera panorama sweep is already active.")
        physical_forward, pose = self._capture_forward_observation(
            completed_step, timestamp
        )
        forward = self._relabel_forward_frame(
            physical_forward, "forward"
        )
        self.panorama_sweep = _PanoramaSweep(
            views={"forward": forward},
            decision_pose=pose,
            capture_poses={"forward": pose},
        )
        self.rotation.start("left")
        self.state = EpisodeState.WAIT_PANORAMA_LOCOMOTION
        print(
            "[LOCAL-VLN] single-camera panorama started: "
            "captured=forward; rotating left through 360deg"
        )

    def _update_single_camera_panorama(
        self,
        *,
        completed_step: int,
        step_dt: float,
        timestamp: float,
        locomotion_ready: bool,
    ) -> np.ndarray:
        """Rotate continuously and capture at each completed 90-degree segment."""

        if self.panorama_sweep is None:
            raise RuntimeError("Single-camera panorama state has no active sweep.")
        command = np.zeros(3, dtype=np.float64)
        if self.state == EpisodeState.WAIT_PANORAMA_LOCOMOTION:
            if locomotion_ready:
                self.state = EpisodeState.PANORAMA_ROTATING
            else:
                return command

        if self.state == EpisodeState.PANORAMA_ROTATING:
            if not locomotion_ready:
                return command
            rotation = self.rotation.update(step_dt)
            command[:] = (rotation.vx, rotation.vy, rotation.wz)
            if not rotation.done:
                return command

            # The completed control tick returns zero yaw without switching to
            # the stand policy.  Capture immediately, then start the next
            # quarter-turn on the following tick.
            command.fill(0.0)
            self.panorama_sweep.quarter_turns_completed += 1
            completed_quarters = self.panorama_sweep.quarter_turns_completed
            if completed_quarters <= 3:
                direction = DIRECTION_ORDER[completed_quarters]
                physical_forward, capture_pose = self._capture_forward_observation(
                    completed_step, timestamp
                )
                frame = self._relabel_forward_frame(
                    physical_forward,
                    direction,
                )
                self.panorama_sweep.views[direction] = frame
                self.panorama_sweep.capture_poses[direction] = capture_pose
                print(
                    "[LOCAL-VLN] single-camera panorama captured: "
                    f"direction={direction} quarter={completed_quarters}/4"
                )

            if completed_quarters < 4:
                self.rotation.start("left")
            else:
                self.state = EpisodeState.PANORAMA_DECIDE
                print(
                    "[LOCAL-VLN] single-camera panorama complete: "
                    "returned to the reference heading"
                )
            return command

        if self.state == EpisodeState.PANORAMA_DECIDE:
            # PANORAMA_DECIDE is deliberately reached one outer tick after the
            # final yaw command became zero.  The slow HTTP/model call runs in
            # one daemon worker so Isaac/G1 control keeps advancing at zero.
            sweep = self.panorama_sweep
            if tuple(sweep.views) != DIRECTION_ORDER:
                raise RuntimeError(
                    "Single-camera panorama did not capture forward/left/behind/right."
                )
            panorama = PanoramaBundle(
                bundle_id=self.next_panorama_bundle_id,
                sim_step=int(completed_step),
                timestamp=float(timestamp),
                views=dict(sweep.views),
            ).validated()
            self.next_panorama_bundle_id += 1
            self._save_single_camera_panorama_trace(sweep)
            decision_pose = sweep.decision_pose
            self.panorama_sweep = None
            self._start_decision_request(
                panorama,
                decision_pose=decision_pose,
            )
        return command

    def _save_single_camera_panorama_trace(self, sweep: _PanoramaSweep) -> None:
        def pose_dict(pose: Pose2D | None) -> dict | None:
            if pose is None:
                return None
            return {
                "x": pose.x,
                "y": pose.y,
                "yaw": pose.yaw,
                "timestamp": pose.timestamp,
            }

        self._save_json(
            f"decision_{self.decision_index:03d}_panorama_capture.json",
            {
                "camera_mode": "single_forward_rgbd_continuous_rotation",
                "rotation_direction": "left",
                "quarter_turns_completed": sweep.quarter_turns_completed,
                "captures": {
                    direction: {
                        "frame_id": sweep.views[direction].frame_id,
                        "sim_step": sweep.views[direction].sim_step,
                        "timestamp": sweep.views[direction].timestamp,
                        "pose": pose_dict(sweep.capture_poses.get(direction)),
                    }
                    for direction in DIRECTION_ORDER
                },
            },
        )

    def _capture_and_decide(self, completed_step: int, timestamp: float) -> None:
        """Legacy simultaneous-camera diagnostic path."""

        capture_panorama = getattr(self.camera, "capture_panorama", None)
        if not callable(capture_panorama):
            raise RuntimeError(
                "Camera backend has no capture_panorama(); enable the default "
                "single-forward panorama mode."
            )
        panorama = capture_panorama(completed_step, timestamp)
        self._decide_from_panorama(panorama, decision_pose=self.odometry.get_pose())

    def _decide_from_panorama(
        self,
        panorama: PanoramaBundle,
        *,
        decision_pose: Pose2D | None,
    ) -> None:
        """Synchronous legacy four-camera diagnostic decision path."""

        request, images = self._build_decision_request(panorama)
        self._save_json(
            f"decision_{self.decision_index:03d}_request.json",
            request.to_metadata(),
        )
        response, raw_response = self.model.decide(request, images)
        self._save_json(
            f"decision_{self.decision_index:03d}_response.json",
            raw_response,
        )
        self._apply_decision_response(
            panorama,
            decision_pose=decision_pose,
            response=response,
            raw_response=raw_response,
        )

    def _build_decision_request(self, panorama: PanoramaBundle):
        """Freeze one request and all image bytes on the control thread."""

        self._wire_history_records = select_model_history_records(
            self.history,
            max_waypoints=self.config.history_max_waypoints,
        )
        history, history_images = build_model_history(
            self.history,
            max_waypoints=self.config.history_max_waypoints,
        )
        request = self.model.make_request(
            panorama,
            session_id=self.config.session_id,
            instruction=self.config.instruction,
            decision_index=self.decision_index,
            history=history,
        )
        images = self.model.image_fields(panorama, request, history_images)
        return request, images

    def _start_decision_request(
        self,
        panorama: PanoramaBundle,
        *,
        decision_pose: Pose2D | None,
    ) -> None:
        """Start one background model request for an immutable panorama."""

        if self._decision_task is not None:
            raise RuntimeError("A model decision request is already in flight.")
        request, images = self._build_decision_request(panorama)
        decision_index = int(self.decision_index)
        self._save_json(
            f"decision_{decision_index:03d}_request.json",
            request.to_metadata(),
        )
        task = _DecisionTask(
            decision_index=decision_index,
            panorama=panorama,
            decision_pose=decision_pose,
            done=threading.Event(),
        )
        self._decision_task = task
        self.state = EpisodeState.WAITING_DECISION

        def worker() -> None:
            try:
                task.response, task.raw_response = self.model.decide(request, images)
            except Exception as exc:
                task.error = exc
            finally:
                task.done.set()

        try:
            threading.Thread(
                target=worker,
                name=f"lavira-decision-{decision_index}",
                daemon=True,
            ).start()
        except Exception:
            self._decision_task = None
            raise
        print(
            "[LOCAL-VLN] decision request started in background: "
            f"index={decision_index}; holding locomotion zero velocity"
        )

    def _poll_decision_request(self) -> None:
        """Apply a completed response without blocking a control tick."""

        task = self._decision_task
        if task is None:
            raise RuntimeError("WAITING_DECISION has no active model request.")
        if not task.done.is_set():
            return
        self._decision_task = None
        if task.error is not None:
            raise RuntimeError(
                f"Background model decision failed: {task.error}"
            ) from task.error
        if task.raw_response is None:
            raise RuntimeError("Background model decision returned no response.")
        if task.decision_index != self.decision_index:
            raise RuntimeError(
                "Background model decision index changed while the request was in flight."
            )
        self._save_json(
            f"decision_{task.decision_index:03d}_response.json",
            task.raw_response,
        )
        self._apply_decision_response(
            task.panorama,
            decision_pose=task.decision_pose,
            response=task.response,
            raw_response=task.raw_response,
        )

    def _apply_decision_response(
        self,
        panorama: PanoramaBundle,
        *,
        decision_pose: Pose2D | None,
        response: NavigationDecisionResponse | None,
        raw_response: dict,
    ) -> None:
        """Validate supervision and apply the existing action transition."""

        supervision: G3DecisionSupervision | None = None
        if self.session_client is not None and self.remote_session_active:
            supervision = self.session_client.validate_decision_context(
                raw_response,
                decision_index=self.decision_index,
            )

        if supervision is not None and supervision.control == "SAFE_STOP":
            self._enter_safe_stop()
            return

        if response is None:
            raise RuntimeError(
                "Model returned no Navigator decision outside Recovery SAFE_STOP."
            )

        if supervision is not None and supervision.stage_progress is not None:
            progress = supervision.stage_progress
            if progress.parse_success:
                print(
                    "[LOCAL-VLN G3] stage_progress: "
                    f"decision={self.decision_index} "
                    f"completed={progress.stage_completed}/{progress.stage_total} "
                    f"stage={progress.current_stage!r}"
                )
            else:
                print(
                    "[LOCAL-VLN G3 WARN] stage_progress parse failed: "
                    f"decision={self.decision_index} error={progress.parse_error!r}; "
                    "Navigator action remains valid"
                )

        if supervision is not None and supervision.control == "PREEMPT":
            self._accept_decision_preempt(
                panorama,
                decision_pose=decision_pose,
                response=response,
                supervision=supervision,
            )
            return

        if supervision is not None:
            is_recovery = bool(getattr(supervision, "recovery", False))
            if self._recovery_expected and not is_recovery:
                raise RuntimeError(
                    "Server returned a Navigator decision while Recovery was required."
                )
            if is_recovery and not self._recovery_expected:
                raise RuntimeError(
                    "Server returned an unexpected Recovery decision."
                )

        if response.action.upper() == "STOP" and supervision is not None:
            self._handle_phase4_stop(supervision)
            return

        if response.action.upper() == "BACKTRACK":
            if not self.config.enable_backtrack:
                raise RuntimeError(
                    "BACKTRACK is disabled while the phase-three execution "
                    "report protocol is being validated."
                )
            self._start_backtrack(
                response,
                decision_pose=self.odometry.get_pose() or decision_pose,
                stable_registry_id=bool(
                    supervision is not None
                    and getattr(supervision, "recovery", False)
                ),
                action_source=(
                    getattr(supervision, "action_source", "NAVIGATOR")
                    if supervision is not None
                    else "NAVIGATOR"
                ),
            )
            return
        # A decision pose belongs to the accepted high-level action, not to the
        # beginning of a single-camera panorama sweep that may have happened
        # several seconds earlier.
        action_decision_pose = self.odometry.get_pose() or decision_pose
        # 只用模型选中方向的检测框和深度，计算完成理想转向后的目标点。
        selected_frame = panorama.views[response.direction]
        projection = project_selected_view_target(
            selected_frame,
            response,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
        )
        self._save_json(
            f"decision_{self.decision_index:03d}_projection.json",
            projection.to_dict(),
        )
        #把这次的状态保存起来 用来给以后时候的决策做参考
        self.pending = _PendingAction(
            response=response,
            panorama=panorama,
            projection=projection,
            initial_forward_frame_id=panorama.views["forward"].frame_id,
            decision_pose=action_decision_pose,
            action_source=(
                getattr(supervision, "action_source", "NAVIGATOR")
                if supervision is not None
                else "NAVIGATOR"
            ),
        )
        self._active_world_trace = []
        self._record_world_trace(force=True, fallback_pose=action_decision_pose)
        # 若目标方向为前方，则无需旋转，可直接进入规划阶段；
        # 否则先等待机器人进入 locomotion 模式，再执行原地转向。
        self.rotation.start(response.direction)
        if self.rotation.active:
            self.state = EpisodeState.WAIT_ROTATION_LOCOMOTION
        else:
            self.rotation_settle_elapsed_s = self.config.rotation_settle_s
            self.state = EpisodeState.PLAN_AFTER_ROTATION
        print(
            "[LOCAL-VLN] decision accepted: "
            f"index={self.decision_index} source={self.pending.action_source} "
            f"action={response.action} "
            f"direction={response.direction} target={response.target!r} "
            f"goal_after_turn={projection.goal_after_turn_xy_m.tolist()}"
        )

    def _accept_decision_preempt(
        self,
        panorama: PanoramaBundle,
        *,
        decision_pose: Pose2D | None,
        response: NavigationDecisionResponse,
        supervision: G3DecisionSupervision,
    ) -> None:
        """Acknowledge a verified decision-level PREEMPT without locomotion."""

        preempt_source = getattr(supervision, "preempt_source", None)
        expected_action = {
            "semantic_audit": "NAVIGATE",
            "premature_stop": "STOP",
        }.get(preempt_source)
        if expected_action is None or response.action.upper() != expected_action:
            raise RuntimeError(
                "Decision PREEMPT action does not match its verified candidate source."
            )
        action_decision_pose = self.odometry.get_pose() or decision_pose
        if action_decision_pose is None:
            raise RuntimeError("Decision PREEMPT requires a valid robot pose.")
        projection = project_selected_view_target(
            panorama.views[response.direction],
            response,
            min_depth_m=self.config.min_depth_m,
            max_depth_m=self.config.max_depth_m,
        )
        self.pending = _PendingAction(
            response=response,
            panorama=panorama,
            projection=projection,
            initial_forward_frame_id=panorama.views["forward"].frame_id,
            decision_pose=action_decision_pose,
            action_source="NAVIGATOR",
        )
        self._active_world_trace = []
        self._record_world_trace(force=True, fallback_pose=action_decision_pose)
        self._begin_action_finish(
            commit_pending=False,
            status="PREEMPTED",
            reason=f"{preempt_source}_failure_verifier_confirmed",
            planner_result="PREEMPTED",
        )
        print(
            "[LOCAL-VLN G3] decision PREEMPT: "
            f"source={preempt_source} action={expected_action}; "
            "no locomotion, holding zero velocity before Recovery acknowledgement"
        )

    def _handle_phase4_stop(self, supervision: G3DecisionSupervision) -> None:
        """Apply the deployed phase-four STOP response without executing motion."""

        stop_phase = supervision.stop_phase
        if supervision.stop_gate is None or stop_phase is None:
            raise RuntimeError("Phase-four STOP decision lacks STOP Gate supervision.")
        self.pending = None
        self.backtrack = None
        self._active_world_trace = []
        if stop_phase == "STOP_CONFIRMED":
            self.session_success_reason = "stop_confirmed"
            self.state = EpisodeState.STOPPED
            print(
                "[LOCAL-VLN G3] STOP_CONFIRMED: holding stand; "
                "next action is end_session(SUCCESS)"
            )
            return
        if stop_phase not in {"PREMATURE_STOP", "STOP_PENDING"}:
            raise RuntimeError(f"Unsupported phase-four STOP phase {stop_phase!r}.")
        self.decision_index += 1
        if (
            self.config.max_decisions is not None
            and self.decision_index >= self.config.max_decisions
        ):
            self._fail("Configured decision limit reached before STOP confirmation.")
            return
        self.state = EpisodeState.CAPTURE_AND_DECIDE
        print(
            f"[LOCAL-VLN G3] {stop_phase}: no motion/action_complete; "
            f"requesting decision={self.decision_index}"
        )

    def _start_backtrack(
        self,
        response: NavigationDecisionResponse,
        *,
        decision_pose: Pose2D | None,
        stable_registry_id: bool = False,
        action_source: str = "NAVIGATOR",
    ) -> None:
        """Resolve a wire waypoint id and accept a stored-reverse physical return."""

        wire_waypoint = response.waypoint
        if wire_waypoint is None:
            raise ValueError("BACKTRACK response lacks waypoint.")
        wire_waypoint = int(wire_waypoint)
        if stable_registry_id:
            local_index = self._server_waypoint_to_local_index.get(wire_waypoint)
            if local_index is None or not 0 <= local_index < len(self.history):
                raise ValueError(
                    f"Recovery BACKTRACK registry waypoint {wire_waypoint} "
                    "is not available in local measured history."
                )
            target_record = self.history[local_index]
        else:
            if not 0 <= wire_waypoint < len(self._wire_history_records):
                raise ValueError(
                    f"BACKTRACK wire waypoint {wire_waypoint} is outside the current "
                    f"request history [0, {len(self._wire_history_records) - 1}]."
                )
            target_record = self._wire_history_records[wire_waypoint]
        current_pose = decision_pose or self.odometry.get_pose()
        if current_pose is None:
            raise RuntimeError(
                "BACKTRACK requires a valid Isaac/SLAM world pose; robot remains stopped."
            )
        route = build_stored_reverse_route(
            self.history,
            target_waypoint_id=int(target_record.waypoint_id),
            current_pose=current_pose,
            max_start_drift_m=self.config.backtrack_start_tolerance_m,
            max_path_length_m=self.config.backtrack_max_path_m,
        )
        current_pose = current_pose.validated()
        history_count_before = len(self.history)
        # Match the old, already verified LaViRA controller: accepting BACKTRACK
        # abandons all waypoints after the selected branch immediately.
        self.history = self.history[: target_record.waypoint_id + 1]
        self._server_waypoint_to_local_index = {
            server_id: local_id
            for server_id, local_id in self._server_waypoint_to_local_index.items()
            if local_id <= target_record.waypoint_id
        }
        checkpoint_index = next_route_checkpoint_index(
            route.points_world_xy,
            current_index=0,
            segment_length_m=self.config.backtrack_segment_length_m,
        )
        self.pending = None
        self.backtrack = _BacktrackAction(
            response=response,
            route=route,
            wire_waypoint_id=wire_waypoint,
            target_waypoint_id=int(target_record.waypoint_id),
            history_count_before=history_count_before,
            decision_pose=current_pose,
            action_source=str(action_source).upper(),
            route_cursor=0,
            checkpoint_index=checkpoint_index,
        )
        self.action_elapsed_s = 0.0
        self.motion_window_index = 0
        self.next_motion_window_elapsed_s = self.config.motion_window_s
        self._reset_motion_window_baseline()
        self._action_completion_status = "COMPLETED"
        self._action_completion_reason = "backtrack_arrived"
        self._save_backtrack_event("accepted")
        remaining = float(
            np.linalg.norm(route.target_world_xy - np.array([current_pose.x, current_pose.y]))
        )
        print(
            "[LOCAL-VLN] BACKTRACK accepted: "
            f"source={self.backtrack.action_source} "
            f"wire_waypoint={wire_waypoint} "
            f"local_waypoint={target_record.waypoint_id} "
            f"history={history_count_before}->{len(self.history)} "
            f"path_length={route.path_length_m:.3f}m"
        )
        if remaining <= self.config.backtrack_goal_tolerance_m:
            print(
                "[LOCAL-VLN] BACKTRACK target is already within tolerance; "
                "no locomotion command will be issued."
            )
            self._begin_action_finish(
                commit_pending=False,
                reason="backtrack_already_at_target",
            )
        else:
            self.state = EpisodeState.WAIT_BACKTRACK_LOCOMOTION

    def _update_backtrack(
        self,
        *,
        completed_step: int,
        step_dt: float,
        timestamp: float,
        applied_command: np.ndarray,
        locomotion_ready: bool,
    ) -> np.ndarray:
        """Advance one physical BACKTRACK control tick."""

        if self.backtrack is None:
            raise RuntimeError("BACKTRACK state has no active context.")
        command = np.zeros(3, dtype=np.float64)
        self.action_elapsed_s += step_dt
        if self.action_elapsed_s > self.config.action_timeout_s:
            self._begin_action_finish(
                commit_pending=False,
                status="FAILED",
                reason="backtrack_timeout",
            )
            return command

        if self.state == EpisodeState.WAIT_BACKTRACK_LOCOMOTION:
            if locomotion_ready:
                self.state = EpisodeState.BACKTRACK_ROTATING

        elif self.state == EpisodeState.BACKTRACK_ROTATING:
            if not locomotion_ready:
                return command
            pose = self._require_backtrack_pose()
            goal_local = fixed_points_to_local(
                self.backtrack.checkpoint_world_xy.reshape(1, 2), pose
            )[0]
            distance = float(np.linalg.norm(goal_local))
            if distance <= self.config.backtrack_goal_tolerance_m:
                self._advance_backtrack_checkpoint()
            else:
                heading_error = float(math.atan2(goal_local[1], goal_local[0]))
                if abs(heading_error) <= self.config.backtrack_heading_tolerance_rad:
                    self.state = EpisodeState.BACKTRACK_PLANNING
                else:
                    command[2] = float(
                        np.clip(
                            1.5 * heading_error,
                            -self.config.rotation_speed_rad_s,
                            self.config.rotation_speed_rad_s,
                        )
                    )

        elif self.state == EpisodeState.BACKTRACK_PLANNING:
            if not locomotion_ready:
                return command
            self._plan_backtrack_checkpoint(completed_step, timestamp)

        elif self.state == EpisodeState.BACKTRACK_EXECUTING:
            if not locomotion_ready:
                return command
            follower_output = self.follower.update(step_dt, applied_command)
            if follower_output.abort_reason is not None:
                self._begin_action_finish(
                    commit_pending=False,
                    status="FAILED",
                    reason=f"backtrack_follower: {follower_output.abort_reason}",
                )
                return command
            if follower_output.reached:
                self.follower.stop()
                self._advance_backtrack_checkpoint()
            else:
                command[:] = follower_output.command
                if self.follower.needs_replan():
                    self._replan_backtrack_checkpoint(completed_step, timestamp)

        control = self._report_motion_window_if_due()
        if control is not None and control.control == "PREEMPT":
            command.fill(0.0)
            self._atomic_preempt_active_action(
                "physical_failure_verifier_confirmed_during_backtrack"
            )
        return command

    def _atomic_preempt_active_action(self, reason: str) -> None:
        """Cancel the current local path and prepare one PREEMPTED report."""

        if self.pending is None and self.backtrack is None:
            raise RuntimeError("PREEMPT has no active local action to cancel.")
        self.follower.stop()
        self._begin_action_finish(
            commit_pending=False,
            status="PREEMPTED",
            reason=reason,
            planner_result="PREEMPTED",
        )
        print(
            "[LOCAL-VLN G3] PREEMPT accepted: iPlanner path cleared; "
            "holding zero velocity before action_complete=PREEMPTED"
        )

    def _require_backtrack_pose(self) -> Pose2D:
        pose = self.odometry.get_pose()
        if pose is None:
            raise RuntimeError(
                "BACKTRACK lost its Isaac/SLAM world pose; robot remains stopped."
            )
        return pose.validated()

    def _plan_backtrack_checkpoint(
        self, completed_step: int, timestamp: float
    ) -> None:
        if self.backtrack is None:
            raise RuntimeError("No BACKTRACK checkpoint is active.")
        pose = self._require_backtrack_pose()
        goal_local = fixed_points_to_local(
            self.backtrack.checkpoint_world_xy.reshape(1, 2), pose
        )[0]
        if float(np.linalg.norm(goal_local)) <= self.config.backtrack_goal_tolerance_m:
            self._advance_backtrack_checkpoint()
            return
        front, _front_pose = self._capture_forward_observation(
            completed_step, timestamp
        )
        path, fear = self.planner.get_plan(front, goal_local)
        if path is None or len(path) < 2:
            self._begin_action_finish(
                commit_pending=False,
                status="FAILED",
                reason="backtrack_iplanner_no_path",
            )
            return
        self.last_fear = fear
        self.follower.start(
            path,
            goal_local,
            goal_tolerance_m=self.config.backtrack_goal_tolerance_m,
        )
        self._save_iplanner_trajectory_image(
            path,
            front,
            save_name=(
                f"backtrack_d{self.decision_index:03d}_"
                f"segment{self.backtrack.segments_completed:03d}.jpg"
            ),
            target_xy=goal_local,
        )
        self._save_json(
            (
                f"decision_{self.decision_index:03d}_backtrack_segment_"
                f"{self.backtrack.segments_completed:03d}.json"
            ),
            {
                "checkpoint_index": self.backtrack.checkpoint_index,
                "checkpoint_world_xy": self.backtrack.checkpoint_world_xy.tolist(),
                "goal_local_xy": goal_local.tolist(),
                "fear": fear,
                "trajectory": np.asarray(path).tolist(),
            },
        )
        self.state = EpisodeState.BACKTRACK_EXECUTING
        # BACKTRACK rotation/planning does not count as translational map
        # progress.  Start a fresh approximately-one-second window only when
        # iPlanner path following actually begins.
        self.next_motion_window_elapsed_s = (
            self.action_elapsed_s + self.config.motion_window_s
        )
        self._reset_motion_window_baseline()

    def _replan_backtrack_checkpoint(
        self, completed_step: int, timestamp: float
    ) -> None:
        if self.backtrack is None or not self.follower.active:
            return
        replan_started_at = time.time()
        try:
            pose = self._require_backtrack_pose()
            goal_local = fixed_points_to_local(
                self.backtrack.checkpoint_world_xy.reshape(1, 2), pose
            )[0]
            front, _front_pose = self._capture_forward_observation(
                completed_step, timestamp
            )
            path, fear = self.planner.get_plan(front, goal_local)
            if path is not None and len(path) > 1:
                self.follower.replace_path(path)
                self.last_fear = fear
        except Exception as exc:
            print(f"[LOCAL-VLN WARN] BACKTRACK iPlanner replan failed: {exc}")
        self.follower.mark_replan_attempt(replan_started_at)

    def _advance_backtrack_checkpoint(self) -> None:
        if self.backtrack is None:
            raise RuntimeError("No BACKTRACK route is active.")
        self.backtrack.route_cursor = self.backtrack.checkpoint_index
        self.backtrack.segments_completed += 1
        if self.backtrack.route_cursor >= self.backtrack.route.points_world_xy.shape[0] - 1:
            self._begin_action_finish(
                commit_pending=False,
                reason="backtrack_arrived",
            )
            return
        self.backtrack.checkpoint_index = next_route_checkpoint_index(
            self.backtrack.route.points_world_xy,
            current_index=self.backtrack.route_cursor,
            segment_length_m=self.config.backtrack_segment_length_m,
        )
        self.state = EpisodeState.BACKTRACK_ROTATING

    def _capture_forward_and_plan(
        self, completed_step: int, timestamp: float
    ) -> None:
        """转向完成后获取新前视 RGB-D，请求 iPlanner 并启动局部跟随。"""

        if self.pending is None:
            raise RuntimeError("No pending action exists after rotation.")
        try:
            front, _front_pose = self._capture_forward_observation(
                completed_step, timestamp
            )
        except Exception as exc:
            print(f"[LOCAL-VLN WARN] forward camera update failed: {exc}")
            self._begin_action_finish(
                commit_pending=False,
                status="FAILED",
                reason="forward_camera_update_failed",
                planner_result="PLANNING_FAILED",
            )
            return
        path, fear = self.planner.get_plan(
            front,
            self.pending.projection.goal_after_turn_xy_m,
        )
        self.pending.post_rotation_forward = front
        self.last_fear = fear
        if path is None or len(path) < 2:
            print("[LOCAL-VLN WARN] iPlanner failed to find a path; skipping this action.")
            self._begin_action_finish(
                commit_pending=False,
                status="FAILED",
                reason="iplanner_no_path",
                planner_result="PLANNING_FAILED",
            )
            return
        self._save_iplanner_trajectory_image(
            path,
            front,
            save_name=(
                f"step{self.decision_index}_plan_{time.strftime('%H%M%S')}.jpg"
            ),
            target_xy=self.pending.projection.goal_after_turn_xy_m,
        )
        # 不走到模型目标的几何中心，在路径尾部保留配置的安全距离。
        try:
            safe_path = truncate_trajectory_for_safety(
                path, self.config.safe_distance_m
            )
        except Exception as exc:
            # Uni-LaViRA 的原始兜底：截短失败时退回执行 iPlanner 原轨迹。
            print(
                "[LOCAL-VLN WARN] trajectory truncation failed; "
                f"using the original trajectory: {exc}"
            )
            safe_path = np.asarray(path)
        if safe_path is None:
            print(
                "[LOCAL-VLN] iPlanner target is inside the configured safe distance; "
                "no locomotion command will be issued."
            )
            self._begin_action_finish()
            return
        # 截短后的最后一个点才是跟随器实际需要到达的安全目标。
        self.follower.start(safe_path, safe_path[-1, :2])
        self.replan_failures = 0
        self.state = EpisodeState.WAIT_EXECUTION_LOCOMOTION
        self._save_json(
            f"decision_{self.decision_index:03d}_plan.json",
            {
                "fear": fear,
                "trajectory": np.asarray(path).tolist(),
                "safe_trajectory": safe_path.tolist(),
                "safe_goal_local_xy": safe_path[-1, :2].tolist(),
            },
        )

    def _begin_action_finish(
        self,
        *,
        commit_pending: bool = True,
        status: str = "COMPLETED",
        reason: str = "local_action_completed",
        planner_result: str | None = None,
    ) -> None:
        """结束当前路径跟随，进入等待机器人稳定站立的收尾阶段。"""

        self._record_world_trace(force=True)
        self._action_final_pose = self.odometry.get_pose()
        self.follower.stop()
        self._action_completion_status = str(status).upper()
        self._commit_pending_after_stand = (
            commit_pending and self._action_completion_status == "COMPLETED"
        )
        self._action_completion_reason = reason
        if planner_result is None:
            if self._action_completion_status == "COMPLETED":
                planner_result = "REACHED"
            elif self._action_completion_status == "PREEMPTED":
                planner_result = "PREEMPTED"
            else:
                planner_result = "EXECUTION_FAILED"
        self._action_planner_result = str(planner_result).upper()
        self._action_reached_local_goal = (
            self._action_completion_status == "COMPLETED"
            and self._action_planner_result == "REACHED"
        )
        self.action_stand_elapsed_s = 0.0
        self.state = EpisodeState.WAIT_ACTION_STAND

    def _commit_or_stop(self) -> None:
        """动作稳定完成后保存结果；STOP 则终止，否则提交历史并继续决策。"""

        if self.backtrack is not None:
            self._commit_backtrack()
            return
        if self.pending is None:
            self._fail("pending action disappeared before history commit")
            return
        response = self.pending.response
        action_source = self.pending.action_source
        control = self._report_action_complete(response.action)
        if control is not None and control.control == "PREEMPT":
            self._fail(
                "G3 requested PREEMPT after the terminal action_complete event; "
                "the frozen protocol requires PREEMPT before that event"
            )
            return
        if control is not None and control.control == "SAFE_STOP":
            self.pending = None
            self._enter_safe_stop()
            return
        next_action = None if control is None else getattr(control, "next_action", None)
        if self._action_completion_status == "PREEMPTED":
            if next_action != "REQUEST_RECOVERY_DECISION":
                self._fail(
                    "PREEMPTED acknowledgement did not request a Recovery decision."
                )
                return
            self.pending = None
            self._recovery_expected = True
            self._advance_decision("Recovery requested after PREEMPT acknowledgement")
            return
        if not self._commit_pending_after_stand:
            self.pending = None
            self._recovery_expected = next_action == "REQUEST_RECOVERY_DECISION"
            self._advance_decision("local action ended without a committed waypoint")
            return
        arrival_pose = self.odometry.get_pose()
        completed_payload = response_debug_dict(
            response,
            projected_goal_xy=self.pending.projection.goal_after_turn_xy_m,
        )
        completed_payload.update(
            {
                "decision_pose": self._pose_dict(self.pending.decision_pose),
                "arrival_pose": self._pose_dict(arrival_pose),
                "executed_world_path_xy": (
                    [point.tolist() for point in self._active_world_trace]
                    if self._active_world_trace
                    else None
                ),
            }
        )
        self._save_json(
            f"decision_{self.decision_index:03d}_completed.json",
            completed_payload,
        )
        # STOP 仍会先完成最后一段靠近目标的路径，然后才结束整个回合。
        if response.action.upper() == "STOP":
            self.pending = None
            self.state = EpisodeState.STOPPED
            print("[LOCAL-VLN] STOP final approach completed; episode stopped.")
            return

        # 只有物理执行完的动作才写入模型历史，尚未完成的决策不会污染上下文。
        record = CompletedWaypoint(
            waypoint_id=len(self.history),
            decision_step=int(self.pending.panorama.sim_step),
            direction=str(response.direction),
            target=str(response.target),
            init_rgb=self.pending.panorama.views["forward"].rgb.copy(),
            direction_rgb=self.pending.panorama.views[response.direction].rgb.copy(),
            decision_pose=self.pending.decision_pose,
            arrival_pose=arrival_pose,
            executed_world_path_xy=(
                np.asarray(self._active_world_trace, dtype=np.float64)
                if self._active_world_trace
                else None
            ),
        )
        self.history.append(record)
        self._server_waypoint_to_local_index[self.decision_index] = record.waypoint_id
        self._active_world_trace = []
        self.pending = None
        if action_source == "RECOVERY":
            if next_action not in {"REQUEST_RECOVERY_DECISION", "REQUEST_DECISION"}:
                self._fail(
                    "Recovery action_complete lacks a valid Recovery/Handback transition."
                )
                return
            self._recovery_expected = next_action == "REQUEST_RECOVERY_DECISION"
            if next_action == "REQUEST_DECISION":
                print(
                    "[LOCAL-VLN G3] NAVIGATOR_HANDBACK: one measured Escape "
                    "success returned control to Navigator"
                )
        else:
            self._recovery_expected = False
        self._advance_decision("action completion")

    def _advance_decision(self, reason: str) -> None:
        """Advance exactly one high-level index and request the next decision."""

        self.decision_index += 1
        if (
            self.config.max_decisions is not None
            and self.decision_index >= self.config.max_decisions
        ):
            self.state = EpisodeState.STOPPED
            print(
                f"[LOCAL-VLN] Configured decision limit reached after {reason}; "
                "episode stopped."
            )
            return
        self.state = EpisodeState.CAPTURE_AND_DECIDE

    def _enter_safe_stop(self) -> None:
        """Fail closed after the server exhausts the Recovery budget."""

        self.follower.stop()
        self.backtrack = None
        self.pending = None
        self._active_world_trace = []
        self._recovery_expected = False
        self._safe_stop_requested = True
        self.session_failure_reason = "recovery_safe_stop"
        self._fail("recovery_safe_stop")
        print(
            "[LOCAL-VLN G3] SAFE_STOP: Recovery budget exhausted; "
            "holding stand and ending Session as FAILURE"
        )

    def _commit_backtrack(self) -> None:
        """Report BACKTRACK and follow the server Escape/Handback transition."""

        if self.backtrack is None:
            raise RuntimeError("No BACKTRACK action exists at completion.")
        active = self.backtrack
        control = self._report_action_complete("BACKTRACK")
        if control is not None and control.control == "PREEMPT":
            self._fail(
                "G3 requested PREEMPT after BACKTRACK action_complete; "
                "this transition is not in the frozen Recovery protocol."
            )
            return
        if control is not None and control.control == "SAFE_STOP":
            self._save_backtrack_event(
                "safe_stop", failure_reason="recovery_safe_stop"
            )
            self._enter_safe_stop()
            return
        next_action = None if control is None else getattr(control, "next_action", None)
        if self._action_completion_status != "COMPLETED":
            reason = self._action_completion_reason
            self._save_backtrack_event("failed", failure_reason=reason)
            self.backtrack = None
            if active.action_source == "RECOVERY" and next_action == "REQUEST_RECOVERY_DECISION":
                self._recovery_expected = True
                self._advance_decision("failed Recovery BACKTRACK")
            else:
                self._fail(reason)
            return
        self._save_backtrack_event("arrived")
        completed = active
        print(
            "[LOCAL-VLN] BACKTRACK completed: "
            f"waypoint={completed.target_waypoint_id} "
            f"history={completed.history_count_before}->{len(self.history)} "
            f"segments={completed.segments_completed}"
        )
        self.backtrack = None
        if active.action_source == "RECOVERY":
            if next_action == "REQUEST_DECISION":
                self._recovery_expected = False
                print(
                    "[LOCAL-VLN G3] NAVIGATOR_HANDBACK: Recovery BACKTRACK "
                    "proved one valid Escape"
                )
            elif next_action == "REQUEST_RECOVERY_DECISION":
                self._recovery_expected = True
            else:
                self._fail(
                    "Recovery BACKTRACK completion lacks a valid next_action."
                )
                return
        else:
            self._recovery_expected = False
        self._advance_decision("BACKTRACK completion")

    def _report_motion_window_if_due(self) -> G3ExecutionControl | None:
        """Record and report one phase-three translational execution window."""

        active_action = self._active_action_name()
        if (
            self.state not in {
                EpisodeState.EXECUTING,
                EpisodeState.BACKTRACK_EXECUTING,
            }
            or active_action is None
            or self.action_elapsed_s + 1e-9 < self.next_motion_window_elapsed_s
        ):
            return None

        snapshot = None
        if self.exploration_map is not None:
            before = self._map_window_explored_before
            if before is None:
                before = self.exploration_map.explored_cells
            snapshot = self.exploration_map.snapshot(explored_before=before)
            map_payload = snapshot.to_dict()
            map_payload.update(
                {
                    "decision_index": int(self.decision_index),
                    "window_index": int(self.motion_window_index),
                    "action": active_action,
                    "window_end_action_elapsed_s": float(self.action_elapsed_s),
                    "map_update_failures": int(self._map_update_failures),
                }
            )
            self._save_json(
                (
                    f"map_progress_decision_{self.decision_index:03d}_window_"
                    f"{self.motion_window_index:04d}.json"
                ),
                map_payload,
            )
            if self.output_dir is not None:
                self.exploration_map.save_debug(
                    self.output_dir
                    / "map_progress"
                    / (
                        f"decision_{self.decision_index:03d}_window_"
                        f"{self.motion_window_index:04d}"
                    ),
                    snapshot,
                )
            print(
                "[LOCAL-VLN MAP] motion window: "
                f"decision={self.decision_index} window={self.motion_window_index} "
                f"new={snapshot.new_explored_cells} "
                f"explored={snapshot.explored_cells} "
                f"traversable={snapshot.traversable_cells}"
            )

        control: G3ExecutionControl | None = None
        pose_end = self.odometry.get_pose()
        goal_distance_end = self._current_local_goal_distance_m()
        if self.session_client is not None and self.remote_session_active:
            if snapshot is None:
                raise RuntimeError(
                    "Phase-three motion_window requires a map_progress snapshot."
                )
            pose_start = self._motion_window_start_pose
            goal_distance_start = self._motion_window_start_goal_distance_m
            if pose_start is None or pose_end is None:
                raise RuntimeError(
                    "Phase-three motion_window lost its Isaac/SLAM pose."
                )
            if goal_distance_start is None or goal_distance_end is None:
                raise RuntimeError(
                    "Phase-three motion_window lost its local-goal distance."
                )
            pose_start = pose_start.validated()
            pose_end = pose_end.validated()
            displacement_m = float(
                math.hypot(pose_end.x - pose_start.x, pose_end.y - pose_start.y)
            )
            wire_map_progress = {
                "resolution_m": float(snapshot.resolution_m),
                "explored_cells": int(snapshot.explored_cells),
                "new_explored_cells": int(snapshot.new_explored_cells),
                "traversable_cells": int(snapshot.traversable_cells),
            }
            request_log = {
                "schema_version": 2,
                "request_type": "report_execution",
                "event_type": "motion_window",
                "session_id": self.config.session_id,
                "decision_index": int(self.decision_index),
                "event_id": (
                    f"{self.config.session_id}:d{self.decision_index}:"
                    f"w{self.motion_window_index}"
                ),
                "window_index": int(self.motion_window_index),
                "action": active_action,
                "timestamp_start": float(pose_start.timestamp),
                "timestamp_end": float(pose_end.timestamp),
                "pose_frame_id": self.pose_frame_id,
                "frame_epoch": int(self.frame_epoch),
                "pose_start": self._pose_array(pose_start),
                "pose_end": self._pose_array(pose_end),
                "displacement_m": displacement_m,
                "local_planner_status": "RUNNING",
                "distance_to_local_goal_start": float(goal_distance_start),
                "distance_to_local_goal_end": float(goal_distance_end),
                "map_progress": wire_map_progress,
            }
            control, raw_control = self.session_client.report_motion_window(
                decision_index=self.decision_index,
                window_index=self.motion_window_index,
                action=active_action,
                timestamp_start=pose_start.timestamp,
                timestamp_end=pose_end.timestamp,
                pose_frame_id=self.pose_frame_id,
                frame_epoch=self.frame_epoch,
                pose_start=self._pose_array(pose_start),
                pose_end=self._pose_array(pose_end),
                displacement_m=displacement_m,
                local_planner_status="RUNNING",
                distance_to_local_goal_start=goal_distance_start,
                distance_to_local_goal_end=goal_distance_end,
                map_progress=wire_map_progress,
            )
            self._save_json(
                (
                    f"g3_decision_{self.decision_index:03d}_motion_"
                    f"{self.motion_window_index:04d}.json"
                ),
                {"request": request_log, "response": raw_control},
            )
            print(
                "[LOCAL-VLN G3] motion_window: "
                f"decision={self.decision_index} window={self.motion_window_index} "
                f"control={control.control}"
            )
        if snapshot is not None:
            self._map_window_explored_before = snapshot.explored_cells
        self._motion_window_start_pose = pose_end
        self._motion_window_start_goal_distance_m = goal_distance_end
        self.motion_window_index += 1
        # Do not burst-send several stale windows after a blocking model/planner call.
        self.next_motion_window_elapsed_s = (
            self.action_elapsed_s + self.config.motion_window_s
        )
        return control

    def _reset_motion_window_baseline(self) -> None:
        """Start a fresh evidence window when true translation begins."""

        self._map_window_explored_before = (
            None
            if self.exploration_map is None
            else self.exploration_map.explored_cells
        )
        self._motion_window_start_pose = self.odometry.get_pose()
        self._motion_window_start_goal_distance_m = (
            self._current_local_goal_distance_m()
        )

    def _current_local_goal_distance_m(self) -> float | None:
        if not self.follower.active:
            return None
        return float(np.linalg.norm(self.follower.current_goal_local_xy))

    def _active_action_name(self) -> str | None:
        if self.backtrack is not None:
            return "BACKTRACK"
        if self.pending is not None:
            return str(self.pending.response.action).upper()
        return None

    def _record_world_trace(
        self,
        *,
        force: bool = False,
        fallback_pose: Pose2D | None = None,
    ) -> None:
        """Record measured NAVIGATE breadcrumbs without changing the wire protocol."""

        if self.pending is None:
            return
        pose = self.odometry.get_pose() or fallback_pose
        if pose is None:
            return
        pose = pose.validated()
        point = np.array([pose.x, pose.y], dtype=np.float64)
        if not self._active_world_trace:
            self._active_world_trace.append(point)
            return
        spacing = float(np.linalg.norm(point - self._active_world_trace[-1]))
        if force or spacing >= self.config.backtrack_breadcrumb_spacing_m:
            if spacing > 1.0e-6:
                self._active_world_trace.append(point)

    @staticmethod
    def _pose_dict(pose: Pose2D | None) -> dict | None:
        if pose is None:
            return None
        return {
            "x": float(pose.x),
            "y": float(pose.y),
            "yaw": float(pose.yaw),
            "timestamp": float(pose.timestamp),
        }

    @staticmethod
    def _pose_array(pose: Pose2D) -> list[float]:
        checked = pose.validated()
        return [float(checked.x), float(checked.y), float(checked.yaw)]

    def _save_backtrack_event(
        self, status: str, *, failure_reason: str | None = None
    ) -> None:
        if self.backtrack is None:
            return
        pose = self.odometry.get_pose()
        payload = self.backtrack.route.to_dict()
        payload.update(
            {
                "status": status,
                "decision_index": int(self.decision_index),
                "wire_waypoint": int(self.backtrack.wire_waypoint_id),
                "target_waypoint": int(self.backtrack.target_waypoint_id),
                "history_count_before": int(self.backtrack.history_count_before),
                "history_count_after": len(self.history),
                "segments_completed": int(self.backtrack.segments_completed),
                "completion_pose": self._pose_dict(pose),
                "failure_reason": failure_reason,
            }
        )
        self._save_json(
            f"decision_{self.decision_index:03d}_backtrack_execution.json",
            payload,
        )

    def _report_action_complete(self, action: str) -> G3ExecutionControl | None:
        """Report exactly once immediately before committing a high-level action."""

        if self.session_client is None or not self.remote_session_active:
            return None
        if self.decision_index in self._reported_action_complete_indices:
            return None
        if self.pending is not None:
            decision_pose = self.pending.decision_pose
        elif self.backtrack is not None:
            decision_pose = self.backtrack.decision_pose
        else:
            raise RuntimeError("action_complete has no active local action.")
        final_pose = self._action_final_pose or self.odometry.get_pose()
        if decision_pose is None or final_pose is None:
            raise RuntimeError(
                "Phase-three action_complete requires decision and final poses."
            )
        decision_pose = decision_pose.validated()
        final_pose = final_pose.validated()
        displacement_m = float(
            math.hypot(
                final_pose.x - decision_pose.x,
                final_pose.y - decision_pose.y,
            )
        )
        request_log = {
            "schema_version": 2,
            "request_type": "report_execution",
            "event_type": "action_complete",
            "session_id": self.config.session_id,
            "decision_index": int(self.decision_index),
            "event_id": (
                f"{self.config.session_id}:d{self.decision_index}:complete"
            ),
            "action": str(action).upper(),
            "status": self._action_completion_status,
            "reached_local_goal": self._action_reached_local_goal,
            "timestamp": float(final_pose.timestamp),
            "pose_frame_id": self.pose_frame_id,
            "frame_epoch": int(self.frame_epoch),
            "decision_pose": self._pose_array(decision_pose),
            "final_pose": self._pose_array(final_pose),
            "displacement_m": displacement_m,
            "planner_result": self._action_planner_result,
            "waypoint_id": int(self.decision_index),
        }
        control, raw_control = self.session_client.report_action_complete(
            decision_index=self.decision_index,
            action=action,
            status=self._action_completion_status,
            reached_local_goal=self._action_reached_local_goal,
            timestamp=final_pose.timestamp,
            pose_frame_id=self.pose_frame_id,
            frame_epoch=self.frame_epoch,
            decision_pose=self._pose_array(decision_pose),
            final_pose=self._pose_array(final_pose),
            displacement_m=displacement_m,
            planner_result=self._action_planner_result,
            waypoint_id=self.decision_index,
        )
        self._reported_action_complete_indices.add(self.decision_index)
        self._save_json(
            f"g3_decision_{self.decision_index:03d}_action_complete.json",
            {
                "local_reason": self._action_completion_reason,
                "request": request_log,
                "response": raw_control,
            },
        )
        print(
            "[LOCAL-VLN G3] action_complete: "
            f"decision={self.decision_index} status={self._action_completion_status} "
            f"control={control.control}"
        )
        return control

    def _fail(self, reason: str) -> None:
        """统一执行失败收尾：停止跟随、记录原因、写日志并进入终态。"""

        self.follower.stop()
        if self.backtrack is not None:
            self._save_backtrack_event("failed", failure_reason=reason)
        self.failure_reason = reason
        self.state = EpisodeState.FAILED
        self._save_json("failure.json", {"reason": reason})
        print(f"[LOCAL-VLN ERROR] {reason}")

    def _desired_mode(self) -> str:
        """根据当前状态判断外层控制器应切换到行走还是站立模式。"""

        if self.state in {
            EpisodeState.WAIT_PANORAMA_LOCOMOTION,
            EpisodeState.PANORAMA_ROTATING,
            EpisodeState.PANORAMA_DECIDE,
            EpisodeState.WAITING_DECISION,
            EpisodeState.WAIT_ROTATION_LOCOMOTION,
            EpisodeState.ROTATING,
            EpisodeState.ROTATION_SETTLE,
            EpisodeState.PLAN_AFTER_ROTATION,
            EpisodeState.WAIT_EXECUTION_LOCOMOTION,
            EpisodeState.EXECUTING,
            EpisodeState.WAIT_BACKTRACK_LOCOMOTION,
            EpisodeState.BACKTRACK_ROTATING,
            EpisodeState.BACKTRACK_PLANNING,
            EpisodeState.BACKTRACK_EXECUTING,
        }:
            return "locomotion"
        return "stand"

    def _result(self, command: np.ndarray) -> EpisodeUpdate:
        """构造对外返回值，并复制速度数组避免外部修改内部数据。"""

        return EpisodeUpdate(
            command=np.asarray(command, dtype=np.float64).reshape(3).copy(),
            desired_mode=self._desired_mode(),
            state=self.state,
            completed=self.completed,
            failure_reason=self.failure_reason,
        )

    def _save_json(self, name: str, payload: dict) -> None:
        """若配置了输出目录，则以易读且保留中文的格式写入调试 JSON。"""

        if self.output_dir is None:
            return
        path = self.output_dir / name
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _save_iplanner_trajectory_image(
        self,
        trajectory: np.ndarray,
        frame: ViewFrame,
        *,
        save_name: str,
        target_xy: np.ndarray,
    ) -> None:
        """Save the same ground-plane trajectory overlay used by Uni-LaViRA."""

        if self.output_dir is None or trajectory is None or len(trajectory) == 0:
            return
        import cv2

        image_bgr = cv2.cvtColor(
            np.ascontiguousarray(frame.rgb), cv2.COLOR_RGB2BGR
        )
        height, width = image_bgr.shape[:2]
        K = np.asarray(frame.K)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        points_2d: list[tuple[int, int]] = []
        for point in trajectory:
            camera_z = float(point[0])
            camera_x = -float(point[1])
            # Uni G1 Config defaults: CAMERA_HEIGHT=1.0, roll correction=0.0.
            camera_y = 1.0
            if camera_z <= 0.01:
                continue
            u = int(camera_x * fx / camera_z + cx)
            v = int(camera_y * fy / camera_z + cy)
            if 0 <= u < width and 0 <= v < height:
                points_2d.append((u, v))
        if len(points_2d) > 1:
            for index in range(len(points_2d) - 1):
                cv2.line(
                    image_bgr,
                    points_2d[index],
                    points_2d[index + 1],
                    (0, 255, 0),
                    2,
                )
            cv2.circle(image_bgr, points_2d[-1], 6, (0, 0, 255), -1)
        target_x = float(target_xy[0])
        target_y = float(target_xy[1])
        target_camera_z = target_x
        target_camera_x = -target_y
        if target_camera_z > 0.01:
            target_u = int(target_camera_x * fx / target_camera_z + cx)
            target_v = int(1.0 * fy / target_camera_z + cy)
            cv2.drawMarker(
                image_bgr,
                (target_u, target_v),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )
        cv2.putText(
            image_bgr,
            f"Goal: ({target_x:.2f}, {target_y:.2f})m",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image_bgr,
            "G1 Humanoid",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        image_dir = self.output_dir / "images" / "iplanner"
        image_dir.mkdir(parents=True, exist_ok=True)
        save_path = image_dir / save_name
        cv2.imwrite(str(save_path), image_bgr)
        self.iplanner_history.append(str(save_path))

    def _log_state_transition(self) -> None:
        """只在状态真正变化时打印一次，避免每个控制周期重复刷屏。"""

        if self.state != self._last_logged_state:
            print(f"[LOCAL-VLN] state={self.state}")
            self._last_logged_state = self.state
