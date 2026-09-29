# 中文导读：
# 阅读提示：本文件是状态机，不是按文件从上到下执行一次。update 根据 self.state 选择分支。
# 一轮高层 decision 可跨越许多控制周期、多次 iPlanner 重规划和多个 motion_window。
# 普通完成、被抢占、STOP_CONFIRMED 是不同事件；沿函数调用阅读，不要把所有 STOP 分支混为一谈。

"""LaViRA G3 机器人端单个 Episode 的主状态机。

这个文件不实现 Navigator、Stage Planner 或 Recovery Planner 模型；这些角色在
远端 LaViRA G3 服务器内运行。本文件负责把服务器的高层决策变成 Isaac Sim/
真机 G1 可以执行的本地流程：

    采集四方向 RGB-D
        → 后台请求服务器决策
        → bbox + depth 投影为机器人局部目标
        → iPlanner 生成局部轨迹
        → 轨迹跟随器输出 [vx, vy, wz]
        → 约1秒一次上报 Motion Window
        → 动作结束上报 action_complete
        → 根据 CONTINUE/PREEMPT/SAFE_STOP 转移状态

设计边界：

* 外层 Isaac/G1 控制循环每帧调用 :meth:`LocalEndToEndEpisode.update`。
* 本类只返回期望速度和期望模式，不直接操作机器人。
* 慢速 HTTP/模型请求放在后台线程，等待期间主控制循环继续发送零速度。
* 里程计、RGB-D 稀疏地图和 iPlanner 都在机器人端；服务器只接收冻结协议字段。
* PREEMPT 始终先清除本地轨迹、保持零速度，再上报 PREEMPTED 确认。
* ``COMPLETED/REACHED`` 只表示到达本地局部轨迹终点，不等于语义任务或 Recovery
  已成功；后者由服务器 STOP/Escape 链路决定。
"""

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

    普通 NAVIGATE 主链路：

    ``WARMUP → CAPTURE_AND_DECIDE → PANORAMA_* → WAITING_DECISION
    → ROTATING → PLAN_AFTER_ROTATION → EXECUTING
    → WAIT_ACTION_STAND → CAPTURE_AND_DECIDE``。

    BACKTRACK 使用独立的 ``BACKTRACK_*`` 子状态，把世界坐标面包屑路径分段
    交给 iPlanner。``STOPPED`` 表示任务协议成功或达到本地测试上限；
    ``FAILED`` 表示安全失败终态（包括 Recovery SAFE_STOP）。
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

    # 远端 Session 标识和整个 Episode 只提交一次的自然语言任务。
    session_id: str
    instruction: str
    # 启动后等待的传感器帧数，以及发给模型的历史/总决策数上限。
    warmup_steps: int = 5
    history_max_waypoints: int | None = None
    max_decisions: int | None = None
    # 单相机全景和指定方向转身共用的旋转参数。
    rotation_speed_rad_s: float = 0.4
    rotation_duration_scale: float = 1.0
    rotation_settle_s: float = 0.5
    # 动作结束后必须稳定站立的时间，避免运动模式立即切换。
    post_action_stand_s: float = 0.8
    # iPlanner 原轨迹尾部预留的安全距离，不是 follower 的 goal tolerance。
    safe_distance_m: float = 0.5
    # bbox 投影时允许使用的深度范围。
    min_depth_m: float = 0.1
    max_depth_m: float = 5.0
    # 一次高层局部动作的超时上限，以及执行证据上报的目标周期。
    action_timeout_s: float = 60.0
    motion_window_s: float = 1.0
    # 上报位姿的坐标系和重定位世代；SLAM重置时应增加 frame_epoch。
    pose_frame_id: str = "local_odom"
    frame_epoch: int = 0
    # True：用唯一前向 RGB-D 连续旋转采集四个方向。
    single_forward_panorama: bool = False
    # BACKTRACK 必须显式开启；以下参数限制面包屑回退的可执行范围。
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
    """模型已决定、但尚未完成执行的一次 NAVIGATE 临时上下文。

    ``projection`` 是 bbox/depth 得到的转向后局部目标；``decision_pose`` 是高层动作
    被接受时的世界位姿，后续 action_complete 使用它计算动作首尾位移。
    ``action_source`` 区分普通 ``NAVIGATOR`` 与 ``RECOVERY``。
    """

    response: NavigationDecisionResponse
    panorama: PanoramaBundle
    projection: LocalTargetProjection
    initial_forward_frame_id: int
    decision_pose: Pose2D | None
    action_source: str = "NAVIGATOR"
    post_rotation_forward: ViewFrame | None = None


@dataclass
class _BacktrackAction:
    """一次已接受 BACKTRACK 的物理执行上下文。

    ``wire_waypoint_id`` 是服务器协议中的 ID，``target_waypoint_id`` 是本地历史记录 ID。
    ``route`` 是从已测量世界路径反向构造的面包屑路线；下标和计数器用于将长路径
    分成多个 iPlanner 局部段。
    """

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
    """单个前向 RGB-D 通过连续旋转一圈生成的四视图临时结果。

    ``views`` 保存 forward/left/behind/right 图像，``capture_poses`` 保存每张图对应的
    实测位姿，便于检查旋转采集是否到达预期方向。
    """

    views: dict[str, ViewFrame]
    decision_pose: Pose2D | None
    capture_poses: dict[str, Pose2D | None]
    quarter_turns_completed: int = 0


@dataclass
class _DecisionTask:
    """一个正在后台线程中执行的模型决策请求。

    图像和 decision index 在启动线程前已冻结。工作线程只写入 ``response``/
    ``raw_response``/「``error``」并设置 ``done``；真正的状态转移仍由控制线程完成。
    """

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

    类中有三类不同层级的“完成”：

    * follower ``reached``：到达本地安全轨迹终点；
    * ``action_complete``：一次高层 NAVIGATE/BACKTRACK 物理执行结束；
    * Episode ``STOPPED``：服务器 STOP Gate 确认整个语义任务完成。

    Recovery 动作即使本地 ``COMPLETED``，也要继续等待 Escape Evaluator 返回
    ``REQUEST_DECISION``（成功 Handback）或 ``REQUEST_RECOVERY_DECISION``（继续恢复）。
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

        # 固定依赖和经过集中校验的配置。
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
        # -------------------- 高层状态与模型历史 --------------------
        self.state = EpisodeState.WARMUP
        self.history: list[CompletedWaypoint] = []
        # 本次请求真正发给服务器的历史子集；旧式 BACKTRACK 索引以它为准。
        self._wire_history_records: tuple[CompletedWaypoint, ...] = ()
        self.decision_index = 0
        # 三类互斥的活动上下文：普通动作、BACKTRACK、单相机全景。
        self.pending: _PendingAction | None = None
        self.backtrack: _BacktrackAction | None = None
        self.panorama_sweep: _PanoramaSweep | None = None
        # 后台模型请求只能同时存在一个。
        self._decision_task: _DecisionTask | None = None
        self.next_panorama_bundle_id = 0
        self._active_world_trace: list[np.ndarray] = []
        # -------------------- 动作计时、失败与日志 --------------------
        self.failure_reason: str | None = None
        self.session_success_reason: str | None = None
        self.rotation_settle_elapsed_s = 0.0
        self.action_stand_elapsed_s = 0.0
        self.action_elapsed_s = 0.0
        # -------------------- Motion Window 窗口基线 --------------------
        # window_index 在每个 decision 中从0递增；窗口位移是首尾直线距离。
        self.motion_window_index = 0
        self.next_motion_window_elapsed_s = self.config.motion_window_s
        self.pose_frame_id = self.config.pose_frame_id
        self.frame_epoch = int(self.config.frame_epoch)
        self._motion_window_start_pose: Pose2D | None = None
        self._motion_window_start_goal_distance_m: float | None = None
        self._map_window_explored_before: int | None = None
        self._map_update_failures = 0
        # -------------------- iPlanner/动作收尾上下文 --------------------
        self.replan_failures = 0
        self.last_fear: float | None = None
        self.iplanner_history: list[str] = []
        self._commit_pending_after_stand = True
        self._action_completion_status = "COMPLETED"
        self._action_completion_reason = "local_action_completed"
        self._action_planner_result = "REACHED"
        self._action_reached_local_goal = True
        self._action_final_pose: Pose2D | None = None
        # 用 decision index 防止同一高层动作重复上报 action_complete。
        self._reported_action_complete_indices: set[int] = set()
        # 服务器稳定 Waypoint Registry ID 到本地 history 下标的映射。
        self._server_waypoint_to_local_index: dict[int, int] = {}
        # True 表示下一次 /decision 必须来自 Recovery Planner，不允许普通 Navigator 插入。
        self._recovery_expected = False
        # SAFE_STOP 是协议失败终态，不是普通 STOP 任务成功。
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
        """当前对象是否拥有一个尚未结束的远端 G3 Session。

        只有返回 True 时才可发送 decision/execution report/end_session，防止本地重复结束
        或向已结束 Session 继续上报。
        """

        return self._remote_session_active and not self._remote_session_ended

    # 地图对象和位姿是本地执行上报的准备条件，不是 start_session JSON 的上传字段。
    # 此时地图可以尚未融合任何相机帧；实际探索数据在后续采集时累积。
    def start_remote_session(self) -> None:
        """在第一次全景决策前执行一次健康检查和 ``start_session``。

        启动前强制检查稀疏地图和里程计，因为阶段3协议的 Motion Window 必须包含
        真实位姿和 ``map_progress``。instruction 只在这里提交，服务器在 Session 中保存
        Frozen Stage Plan；后续 decision 通过 session_id 取回任务。
        """

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
        # 服务响应已被 client 解析验证，至此 Episode 才拥有活动远端会话。
        self._remote_session_active = True
        self._remote_session_ended = False
        self._save_json("g3_session_started.json", raw_started)
        print(
            "[LOCAL-VLN G3] session ACTIVE: "
            f"stage_plan_id={started.stage_plan_id} "
            f"stages={started.stage_total}"
        )

    def end_remote_session(self, *, status: str, reason: str) -> None:
        """最多结束一次当前远端 Session，可在正常结束或异常清理中调用。

        ``status=SUCCESS`` 只应用于 STOP_CONFIRMED；Recovery SAFE_STOP、本地异常或人工中断
        应以 ``FAILURE`` 结束并保留明确 reason。
        """

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

        处理优先级是：

        1. 等待模型时只轮询后台任务，继续返回零速度；
        2. BACKTRACK 和单相机全景使用各自子状态；
        3. 普通旋转/规划/跟随在主分支执行；
        4. 任何未捕获异常都转成 ``FAILED`` 并强制零速度。
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

                # 每轮计算一个速度目标，不需要先走到前瞻点才能进行下一轮计算。
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
                    # 这里只结束局部动作，任务是否成功仍由 G3 STOP Gate 确认。
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

                # 此处同步等待 iPlanner；远端模型 decision 则使用后台请求线程。
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
        """采集一帧物理前向 RGB-D，并尝试融合到 Episode 稀疏地图。

        返回 ``(frame, pose)``。地图是 Physical Monitor 的证据和本地日志，不参与 iPlanner
        避障；地图单帧融合失败不应中断一个原本安全的导航动作。因此映射异常只记录
        ``_map_update_failures``，RGB-D 仍可继续供模型或 iPlanner 使用。
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
        """深拷贝一张物理前向相机帧，并将它标记为当前全景方向。

        机器人转到 left/behind/right 时，硬件上仍是同一个 forward 相机；这个函数只改
        协议语义中的 direction，同时复制 RGB/depth/K，防止后续相机缓冲区复用数组。
        """

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
        """启动一次单前向相机的四方向全景采集。

        先在当前朝向采集 ``forward``，再请求进入 locomotion 模式。后续连续完成四次
        90°左转，前三次结束时分别采集 left/behind/right，第四次只用来回到参考
        朝向。这一过程不切换到 stand，减少真机频繁模式切换。
        """

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
        """推进单相机全景的一个控制周期，返回当前 ``[vx, vy, wz]``。

        * ``WAIT_PANORAMA_LOCOMOTION``：等待行走策略接管，期间零速度。
        * ``PANORAMA_ROTATING``：输出纯 yaw 命令，每90°分段结束立即采图。
        * ``PANORAMA_DECIDE``：固定四张图和位姿日志，启动后台模型请求。

        HTTP 请求不在这个控制调用中同步等待，因此 Isaac/真机主循环不会被冻结。
        """

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
        """保存四张图的帧号、时间戳和采集位姿，不修改决策数据。"""

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
        """旧的多相机同时采集诊断路径。

        当 ``single_forward_panorama=False`` 时才使用。它依赖 camera backend 直接提供
        ``capture_panorama``，不是当前真机计划使用的默认路径。
        """

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
        """在旧多相机诊断路径中同步请求一次模型决策。

        此方法会阻塞调用线程，所以不应用于实际单相机 G1 主流程；主流程使用
        ``_start_decision_request`` + ``_poll_decision_request``。
        """

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
        """在控制线程中冻结一次 decision 的 metadata、历史和所有图像字节。

        先根据 ``history_max_waypoints`` 选出协议允许的历史，再构建最近 waypoint 的文字与
        图像字段。冻结后后台线程不再读取正在变化的相机或 history，避免请求内部数据
        不一致。
        """

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
        """为已冻结全景启动唯一的后台模型请求。

        方法立即把状态转为 ``WAITING_DECISION``。daemon worker 仅执行 HTTP/模型调用并
        写入 ``_DecisionTask``；等待期间 ``update`` 持续返回 locomotion 模式下的零速度，
        符合真机速度指令需要持续刷新的要求。
        """

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
        """非阻塞轮询后台 decision，完成后在控制线程应用响应。

        未完成时直接返回；完成后先检查 worker 异常、raw JSON 和 decision index，再落盘并
        进入统一的 ``_apply_decision_response``。这样所有状态变更只发生在主控制线程。
        """

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
        """校验 G3 监督字段，并按 ``control`` 优先级应用高层动作。

        关键优先级：

        1. ``control=SAFE_STOP``：不再解释普通 Navigator 动作，直接安全失败收尾；
        2. ``control=PREEMPT``：验证候选来源，不执行该决策的运动，转入 PREEMPTED ack；
        3. STOP：交给阶段4 STOP Gate 状态处理；
        4. BACKTRACK：将服务器 waypoint 解析成本地面包屑回退路线；
        5. NAVIGATE：用选中视图 bbox/depth 投影局部目标，再转向和调用 iPlanner。

        ``action_source`` 同时作为 Recovery 状态机守卫：服务器要求 Recovery 时不允许返回
        普通 Navigator，反之亦然。
        """

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
        # 保存这次已接受动作的决策图像、投影目标和位姿。
        # 动作物理完成后才会转成 CompletedWaypoint 加入模型历史。
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
        """接受已经独立 Verifier 确认的“决策阶段 PREEMPT”，不执行运动。

        当前冻结协议只允许两种来源：Semantic Audit 抢占 NAVIGATE，以及
        PREMATURE_STOP 抢占 STOP。本方法创建一个仅用于回报的 pending 上下文，立即进入
        零速度站稳流程，随后发送 ``action_complete=PREEMPTED`` 作为原子抢占确认。
        """

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

    # 这是有活动 G3 Session 的 STOP 路径；与旧无 Session 模式的“最后靠近后停止”不同。
    def _handle_phase4_stop(self, supervision: G3DecisionSupervision) -> None:
        """处理阶段4 STOP Gate 响应，STOP 本身不产生运动或 action_complete。

        * ``STOP_CONFIRMED``：任务成功，保持站立并等外层 ``end_session(SUCCESS)``；
        * ``STOP_PENDING``：有效 Stage Progress 证据还不足，重新拍全景请求下一次决策；
        * ``PREMATURE_STOP``：STOP Gate 不认可任务完成。未被P0 Verifier升级为PREEMPT时，
          同样重新请求决策；已升级的情况在更早的 ``control=PREEMPT`` 分支处理。
        """

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
        """解析服务器 waypoint ID，构造并接受一次 stored-reverse 物理回退。

        有两种 ID 语义：

        * Recovery BACKTRACK：``waypoint`` 是服务器 Waypoint Registry 的稳定 ID，必须通过
          ``_server_waypoint_to_local_index`` 查到本地实测历史；
        * 旧式 Navigator BACKTRACK：``waypoint`` 是当前请求中被截断的 wire history 下标。

        解析后从 CompletedWaypoint 中保存的世界路径反向构造 route，立即剪掉目标分支
        之后的历史，再按 ``backtrack_segment_length_m`` 分段执行。若已在目标容差内，
        不发送任何运动速度，直接进入完成上报。
        """

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
        """推进物理 BACKTRACK 的一个控制周期，返回 ``[vx, vy, wz]``。

        每个 checkpoint 都按“世界目标转到机器人局部坐标 → 原地对准 → iPlanner
        规划 → follower执行”的顺序处理。只有 ``BACKTRACK_EXECUTING`` 的真实平移期间才会
        定期上报 Motion Window。若服务器在窗口响应中返回 PREEMPT，立即清除当前轨迹。
        """

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
        """原子取消当前本地动作，并准备唯一的 PREEMPTED 完成上报。

        “原子”在机器人端意味着：先 ``follower.stop()`` 清除旧 iPlanner 路径，当前周期
        立即返回零速度，等机器人稳定站立后才向服务器确认
        ``action_complete=PREEMPTED``。确认前不会请求 Recovery。
        """

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
        """取得并校验 BACKTRACK 必需的 Isaac/SLAM 世界位姿，缺失时安全失败。"""

        pose = self.odometry.get_pose()
        if pose is None:
            raise RuntimeError(
                "BACKTRACK lost its Isaac/SLAM world pose; robot remains stopped."
            )
        return pose.validated()

    def _plan_backtrack_checkpoint(
        self, completed_step: int, timestamp: float
    ) -> None:
        """将当前世界坐标 checkpoint 转换为局部目标，并用最新前视 RGB-D 规划。

        BACKTRACK 使用独立的较小 goal tolerance，不受普通 NAVIGATE 的 1.0m tolerance
        影响。旋转与规划时的地图变化不算平移进展，因此只在 follower 真正开始后
        重置 Motion Window 基线。
        """

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
        """对正在执行的 BACKTRACK checkpoint 进行周期性 iPlanner 重规划。

        规划失败只记录警告，保留原轨迹；无论成功与否都更新重规划时间，避免在
        每个控制周期对失效服务连续重试。
        """

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
        """标记当前 BACKTRACK 分段完成，选择下一 checkpoint 或进入动作收尾。"""

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

    # 目标来自此前选中视图的投影，规划输入图像则是转向后新拍的前视帧。
    # fear 在本地用于记录，不会自动把风险分数换成速度或截短距离。
    def _capture_forward_and_plan(
        self, completed_step: int, timestamp: float
    ) -> None:
        """转向完成后获取新前视 RGB-D，请求 iPlanner 并启动局部跟随。

        模型 bbox 投影得到的目标是「理想局部目标」；iPlanner 返回从机器人到该目标的轨迹。
        然后 ``truncate_trajectory_for_safety`` 从轨迹末尾留出 ``safe_distance_m``，截短后的
        ``safe_path[-1, :2]`` 才是 follower 真正追踪的安全目标。``fear`` 在这里只记录，
        不参与客户端截断长度计算。

        若截短后目标本身落在 follower goal tolerance 内，本地动作可以很快返回
        ``COMPLETED/REACHED``；这仍不代表 Recovery Escape 或整个语义任务完成。
        """

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
        """结束当前路径跟随，进入等待机器人稳定站立的收尾阶段。

        该方法不立即发 action_complete：先记录最终位姿、清除 follower 并把结果规范为
        ``COMPLETED/FAILED/PREEMPTED`` 及 ``REACHED/TIMEOUT/...``，再转到
        ``WAIT_ACTION_STAND``。只有站稳持续时间达标后，``_commit_or_stop`` 才对服务器上报。
        """

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

    # 只把成功动作登记到 CompletedWaypoint；服务器 next_action 可以要求继续恢复或返回普通导航。
    def _commit_or_stop(self) -> None:
        """机器人稳定站立后上报 action_complete，并执行服务器的下一步控制。

        处理要点：

        * ``SAFE_STOP`` 优先级最高，直接进入失败终态；
        * PREEMPTED ack 必须换来 ``REQUEST_RECOVERY_DECISION``，否则视为协议错误；
        * FAILED/PREEMPTED 动作不写入 CompletedWaypoint，避免污染 Navigator 历史；
        * COMPLETED NAVIGATE 才保存图像、decision/arrival pose 和实测面包屑；
        * 普通 NAVIGATE 完成后若服务器才确认失败，直接等待 Recovery decision，不补发
          ``action_complete=PREEMPTED``；
        * Recovery COMPLETED 还要根据 Escape Evaluator 的 ``next_action`` 决定继续 Recovery 或
          Handback 到 Navigator。
        """

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
            # Late-completion failure：本地动作已经以 COMPLETED 结束并登记 waypoint 后，
            # 服务器才由 Physical Monitor/Verifier 确认需要恢复。此时动作已经终止，不能
            # 伪造 PREEMPTED ack；下一次 decision 应直接接受 Recovery Planner 的结果。
            self._recovery_expected = next_action == "REQUEST_RECOVERY_DECISION"
            if self._recovery_expected:
                print(
                    "[LOCAL-VLN G3] LATE_COMPLETION_RECOVERY: terminal action "
                    "was recorded; requesting Recovery without PREEMPTED ack"
                )
        self._advance_decision("action completion")

    def _advance_decision(self, reason: str) -> None:
        """高层 decision index 严格增加一次，然后重新进入全景采集。

        decision index 是 Session 内的高层因果顺序，不是仿真帧号；一个 decision 可以包含
        多条 Motion Window，但只能有一条 action_complete。
        """

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
        """服务器判定 Recovery 预算耗尽后执行 fail-closed 安全停止。

        立即停止 follower，丢弃 pending/BACKTRACK 轨迹，清空 Recovery 期待并进入 ``FAILED``。
        外层随后以 ``end_session(FAILURE, recovery_safe_stop)`` 结束会话。这个 FAILED 是
        “任务未完成但已安全收尾”，不是 Python 崩溃。
        """

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
        """上报 BACKTRACK 结果，并根据 Escape/Handback 状态转移收尾。

        Recovery BACKTRACK 成功到达 waypoint 也不自动等于脱困：服务器返回
        ``REQUEST_DECISION`` 才表示一次有效 Escape 并 Handback；返回
        ``REQUEST_RECOVERY_DECISION`` 则继续 Recovery。失败的 Recovery BACKTRACK 也可在服务器
        允许时继续规划，而不是直接结束整个 Episode。
        """

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

    # 窗口是高层执行证据，非每条 DDS 指令；默认约一秒一次，且受同步请求耗时影响。
    # 首尾位移是直线距离，不是机器人走过的路径弧长，转圈或绕行时二者差别很大。
    def _report_motion_window_if_due(self) -> G3ExecutionControl | None:
        """在到达约定时间点时，生成并上报一条阶段3平移 Motion Window。

        只有 ``EXECUTING`` 或 ``BACKTRACK_EXECUTING`` 的真实轨迹跟随才记录窗口；
        拍全景、原地转向、等模型、等 iPlanner 和站立都不会生成伪平移证据。

        一条窗口的关键定义：

        * ``timestamp_start/end``：窗口首尾里程计时间，不强假设恰好1.000秒；
        * ``pose_start/end``：同一 ``pose_frame_id/frame_epoch`` 下的 ``[x,y,yaw]``；
        * ``displacement_m``：首尾 ``(x,y)`` 直线距离，不是轨迹弧长；
        * ``distance_to_local_goal_*``：follower 安全局部目标在窗口首尾的真实距离；
        * ``new_explored_cells``：该窗口期间新增的唯一 5cm 观测格数。

        服务器使用这些证据运行 Physical Monitor。若返回 ``PREEMPT``，调用者必须在
        当前控制周期清零速度并进入原子抢占。为避免模型/规划阻塞后连续补发过时窗口，
        下一截止时间始终从“当前 action elapsed”重新计算。
        """

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
        """在真实平移刚开始或 BACKTRACK 切换分段时重置 Motion Window 基线。

        同时冻结当前 explored计数、里程计位姿和局部目标距离，下一窗口结束时才能
        得到正确的增量和首尾差值。
        """

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
        """返回 follower 当前固定安全目标的二维欧氏距离；未执行时返回 None。"""

        if not self.follower.active:
            return None
        return float(np.linalg.norm(self.follower.current_goal_local_xy))

    def _active_action_name(self) -> str | None:
        """返回当前执行上下文的协议 action 名称，BACKTRACK 优先于 pending。"""

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
        """记录实测 NAVIGATE 世界坐标面包屑，但不改变对外 HTTP 协议。

        相邻记录点至少间隔 ``backtrack_breadcrumb_spacing_m``，防止把50Hz里程计每帧都
        写入路径。``force=True`` 用于强制尝试保存决策起点/动作终点，但仍会过滤数值上
        完全重合的点。
        """

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
        """把位姿转成用于本地可读日志的字典；允许 None。"""

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
        """把已校验位姿转成冻结 HTTP Schema 要求的 ``[x, y, yaw]`` 数组。"""

        checked = pose.validated()
        return [float(checked.x), float(checked.y), float(checked.yaw)]

    def _save_backtrack_event(
        self, status: str, *, failure_reason: str | None = None
    ) -> None:
        """落盘 BACKTRACK 路线、ID映射、执行段数、最终位姿和失败原因。"""

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
        """在提交高层动作历史前，幂等且仅上报一次 ``action_complete``。

        ``decision_pose`` 和 ``final_pose`` 计算的 ``displacement_m`` 是整个高层动作首尾
        直线位移，和每秒 Motion Window 的位移不是同一累计口径。``event_id`` 由
        ``session_id + decision_index + complete`` 确定，配合服务器幂等重试。

        ``status`` 表示机器人端执行结果：

        * ``COMPLETED/REACHED``：到达本地轨迹终点；
        * ``FAILED/TIMEOUT|PLANNING_FAILED|EXECUTION_FAILED``：本地动作失败；
        * ``PREEMPTED/PREEMPTED``：已完成服务器要求的原子抢占。

        服务器响应的 ``control/next_action`` 才决定后续是普通决策、Recovery、Handback
        还是 SAFE_STOP。
        """

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
        """根据当前状态判断外层控制器应切换到行走还是站立模式。

        等待模型也保持 ``locomotion``，但 ``update`` 返回的命令是 ``[0,0,0]``。这样真机
        可继续以零速度刷新高层行走控制，而不会因每次模型请求都频繁切换 stand。
        只有预热、动作收尾和终态需要 ``stand``。
        """

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
        """将 iPlanner 地面轨迹投影回 RGB 图像，保存与 Uni-LaViRA 一致的诊断叠加图。

        轨迹坐标约定为 ``x=前、y=左``；投影到相机时使用 ``camera_z=x``、
        ``camera_x=-y`` 和 Uni G1 默认1m相机高度。绿线是轨迹，红点是轨迹终点，红色十字
        是请求给 iPlanner 的原局部目标。该图仅用于实验分析，不反馈给模型或控制器。
        """

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
