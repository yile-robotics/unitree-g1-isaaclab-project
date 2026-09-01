from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Protocol

import numpy as np

from .iplanner_client import IPlannerClient
from .local_projection import LocalTargetProjection, project_selected_view_target
from .local_trajectory import (
    LocalFollowerConfig,
    LocalTrajectoryFollower,
    truncate_trajectory_for_safety,
)
from .model_client import (
    CombinedModelClient,
    CompletedWaypoint,
    build_model_history,
    response_debug_dict,
)
from .model_contract import NavigationDecisionResponse
from .odometry import OdometryProvider
from .rotation import TimedFixedSpeedRotation, YawProvider
from .types import PanoramaBundle, ViewFrame


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
    WAIT_ROTATION_LOCOMOTION = "wait_rotation_locomotion"
    ROTATING = "rotating"
    ROTATION_SETTLE = "rotation_settle"
    PLAN_AFTER_ROTATION = "plan_after_rotation"
    WAIT_EXECUTION_LOCOMOTION = "wait_execution_locomotion"
    EXECUTING = "executing"
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
    output_dir: Path | None = None

    def validated(self) -> "EpisodeConfig":
        """集中验证配置，避免非法参数在运行中途才导致机器人行为异常。"""

        if not self.session_id.strip() or not self.instruction.strip():
            raise ValueError("Session id and instruction must not be empty.")
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
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Episode timing/range values must be finite and positive.")
        if not 0.0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("Episode depth range must satisfy 0 < min < max.")
        if self.safe_distance_m < 0.0:
            raise ValueError("Safe distance must be non-negative.")
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
    post_rotation_forward: ViewFrame | None = None


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
        odometry: OdometryProvider | None = None,
        yaw_provider: YawProvider | None = None,
    ):
        """装配各组件、初始化所有计时器，并按需创建本回合日志目录。"""

        self.config = config.validated()
        self.camera = camera
        self.model = model
        self.planner = planner
        self.rotation = TimedFixedSpeedRotation(
            config.rotation_speed_rad_s,
            config.rotation_duration_scale,
            yaw_provider=yaw_provider,
        )
        self.follower = LocalTrajectoryFollower(follower_config, odometry)
        self.state = EpisodeState.WARMUP
        self.history: list[CompletedWaypoint] = []
        self.decision_index = 0
        self.pending: _PendingAction | None = None
        self.failure_reason: str | None = None
        self.rotation_settle_elapsed_s = 0.0
        self.action_stand_elapsed_s = 0.0
        self.action_elapsed_s = 0.0
        self.replan_failures = 0
        self.last_fear: float | None = None
        self.iplanner_history: list[str] = []
        self._commit_pending_after_stand = True
        self._last_logged_state: str | None = None
        self.output_dir = (
            None
            if config.output_dir is None
            else _create_unique_run_output_dir(config.output_dir, config.session_id)
        )
        if self.output_dir is not None:
            print(f"[LOCAL-VLN] output directory: {self.output_dir}")

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

        # 默认命令始终为零；只有 ROTATING 或 EXECUTING 阶段会明确写入运动速度。
        try:
            if self.state == EpisodeState.WARMUP:
                # 等待若干帧让传感器稳定，并确认机器人处于站立模式。
                if completed_step >= self.config.warmup_steps and stand_ready:
                    self.state = EpisodeState.CAPTURE_AND_DECIDE

            if self.state == EpisodeState.CAPTURE_AND_DECIDE:
                # 拍摄和模型推理期间保持站立，避免四张图对应不同机器人姿态。
                if not stand_ready:
                    return self._result(command)
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

            if self.state == EpisodeState.EXECUTING:
                # 执行阶段同时负责超时保护、跟随控制以及周期性视觉重规划。
                if not locomotion_ready:
                    return self._result(command)
                self.action_elapsed_s += step_dt
                if self.action_elapsed_s > self.config.action_timeout_s:
                    print(
                        "[LOCAL-VLN WARN] local trajectory execution timed out; "
                        "ending this segment like Uni-LaViRA"
                    )
                    self._begin_action_finish()
                    return self._result(command)

                follower_output = self.follower.update(step_dt, applied_command)
                if follower_output.abort_reason is not None:
                    print(
                        "[LOCAL-VLN WARN] local trajectory ended: "
                        f"{follower_output.abort_reason}"
                    )
                    self._begin_action_finish()
                    return self._result(command)
                if follower_output.reached:
                    self._begin_action_finish()
                    return self._result(command)
                command[:] = follower_output.command

                if self.follower.needs_replan():
                    replan_started_at = time.time()
                    # 使用最新前视 RGB-D 和更新后的局部目标重新规划，修正环境变化。
                    try:
                        fresh_front = self.camera.capture_forward(
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

    def _capture_and_decide(self, completed_step: int, timestamp: float) -> None:
        """拍摄四视图，携带历史请求模型，并把视觉目标投影到局部坐标。"""

        panorama = self.camera.capture_panorama(completed_step, timestamp)
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
        response, raw_response = self.model.decide(request, images)
        self._save_json(
            f"decision_{self.decision_index:03d}_request.json",
            request.to_metadata(),
        )
        self._save_json(
            f"decision_{self.decision_index:03d}_response.json",
            raw_response,
        )

        # 协议仍认识 BACKTRACK，但这套纯局部执行器没有 FMM 回退路径实现。
        if response.action.upper() == "BACKTRACK":
            raise RuntimeError(
                "BACKTRACK is preserved by the model schema but intentionally "
                "unsupported by the local no-FMM executor; robot remains stopped"
            )
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
        )
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
            f"index={self.decision_index} action={response.action} "
            f"direction={response.direction} target={response.target!r} "
            f"goal_after_turn={projection.goal_after_turn_xy_m.tolist()}"
        )

    def _capture_forward_and_plan(
        self, completed_step: int, timestamp: float
    ) -> None:
        """转向完成后获取新前视 RGB-D，请求 iPlanner 并启动局部跟随。"""

        if self.pending is None:
            raise RuntimeError("No pending action exists after rotation.")
        try:
            front = self.camera.capture_forward(completed_step, timestamp)
        except Exception as exc:
            print(f"[LOCAL-VLN WARN] forward camera update failed: {exc}")
            self._begin_action_finish(commit_pending=False)
            return
        path, fear = self.planner.get_plan(
            front,
            self.pending.projection.goal_after_turn_xy_m,
        )
        self.pending.post_rotation_forward = front
        self.last_fear = fear
        if path is None or len(path) < 2:
            print("[LOCAL-VLN WARN] iPlanner failed to find a path; skipping this action.")
            self._begin_action_finish(commit_pending=False)
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

    def _begin_action_finish(self, *, commit_pending: bool = True) -> None:
        """结束当前路径跟随，进入等待机器人稳定站立的收尾阶段。"""

        self.follower.stop()
        self._commit_pending_after_stand = commit_pending
        self.action_stand_elapsed_s = 0.0
        self.state = EpisodeState.WAIT_ACTION_STAND

    def _commit_or_stop(self) -> None:
        """动作稳定完成后保存结果；STOP 则终止，否则提交历史并继续决策。"""

        if self.pending is None:
            self._fail("pending action disappeared before history commit")
            return
        response = self.pending.response
        if not self._commit_pending_after_stand:
            self.pending = None
            self.decision_index += 1
            if response.action.upper() == "STOP" or (
                self.config.max_decisions is not None
                and self.decision_index >= self.config.max_decisions
            ):
                self.state = EpisodeState.STOPPED
            else:
                self.state = EpisodeState.CAPTURE_AND_DECIDE
            return
        self._save_json(
            f"decision_{self.decision_index:03d}_completed.json",
            response_debug_dict(
                response,
                projected_goal_xy=self.pending.projection.goal_after_turn_xy_m,
            ),
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
        )
        self.history.append(record)
        self.pending = None
        self.decision_index += 1
        if (
            self.config.max_decisions is not None
            and self.decision_index >= self.config.max_decisions
        ):
            self.state = EpisodeState.STOPPED
            print("[LOCAL-VLN] Configured decision limit reached; episode stopped.")
        else:
            self.state = EpisodeState.CAPTURE_AND_DECIDE

    def _fail(self, reason: str) -> None:
        """统一执行失败收尾：停止跟随、记录原因、写日志并进入终态。"""

        self.follower.stop()
        self.failure_reason = reason
        self.state = EpisodeState.FAILED
        self._save_json("failure.json", {"reason": reason})
        print(f"[LOCAL-VLN ERROR] {reason}")

    def _desired_mode(self) -> str:
        """根据当前状态判断外层控制器应切换到行走还是站立模式。"""

        if self.state in {
            EpisodeState.WAIT_ROTATION_LOCOMOTION,
            EpisodeState.ROTATING,
            EpisodeState.ROTATION_SETTLE,
            EpisodeState.WAIT_EXECUTION_LOCOMOTION,
            EpisodeState.EXECUTING,
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
