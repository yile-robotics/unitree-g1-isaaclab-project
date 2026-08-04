from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
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
from .rotation import TimedFixedSpeedRotation
from .types import PanoramaBundle, ViewFrame


class CameraBackend(Protocol):
    def capture_panorama(self, sim_step: int, timestamp: float) -> PanoramaBundle:
        ...

    def capture_forward(self, sim_step: int, timestamp: float) -> ViewFrame:
        ...


class EpisodeState:
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
    session_id: str
    instruction: str
    warmup_steps: int = 5
    max_decisions: int = 20
    rotation_speed_rad_s: float = 0.4
    rotation_duration_scale: float = 1.0
    rotation_settle_s: float = 0.5
    post_action_stand_s: float = 0.8
    safe_distance_m: float = 0.5
    min_depth_m: float = 0.1
    max_depth_m: float = 5.0
    action_timeout_s: float = 60.0
    max_replan_failures: int = 3
    output_dir: Path | None = None

    def validated(self) -> "EpisodeConfig":
        if not self.session_id.strip() or not self.instruction.strip():
            raise ValueError("Session id and instruction must not be empty.")
        if self.warmup_steps < 0 or self.max_decisions <= 0:
            raise ValueError("Warmup must be >=0 and max decisions must be >0.")
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
        if self.safe_distance_m < 0.0 or self.max_replan_failures < 0:
            raise ValueError("Safe distance/replan failure limit is invalid.")
        return self


@dataclass(frozen=True)
class EpisodeUpdate:
    command: np.ndarray
    desired_mode: str
    state: str
    completed: bool
    failure_reason: str | None


@dataclass
class _PendingAction:
    response: NavigationDecisionResponse
    panorama: PanoramaBundle
    projection: LocalTargetProjection
    initial_forward_frame_id: int
    post_rotation_forward: ViewFrame | None = None


class LocalEndToEndEpisode:
    """Single-model, selected-depth, local-iPlanner navigation state machine."""

    def __init__(
        self,
        config: EpisodeConfig,
        follower_config: LocalFollowerConfig,
        *,
        camera: CameraBackend,
        model: CombinedModelClient,
        planner: IPlannerClient,
        odometry: OdometryProvider | None = None,
    ):
        self.config = config.validated()
        self.camera = camera
        self.model = model
        self.planner = planner
        self.rotation = TimedFixedSpeedRotation(
            config.rotation_speed_rad_s,
            config.rotation_duration_scale,
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
        self._last_logged_state: str | None = None
        self.output_dir = (
            None if config.output_dir is None else Path(config.output_dir) / config.session_id
        )
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def completed(self) -> bool:
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
        if step_dt <= 0.0:
            raise ValueError("Episode step_dt must be positive.")
        command = np.zeros(3, dtype=np.float64)

        try:
            if self.state == EpisodeState.WARMUP:
                if completed_step >= self.config.warmup_steps and stand_ready:
                    self.state = EpisodeState.CAPTURE_AND_DECIDE

            if self.state == EpisodeState.CAPTURE_AND_DECIDE:
                if not stand_ready:
                    return self._result(command)
                self._capture_and_decide(completed_step, timestamp)

            if self.state == EpisodeState.WAIT_ROTATION_LOCOMOTION:
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
                if not locomotion_ready:
                    return self._result(command)
                self.action_elapsed_s += step_dt
                if self.action_elapsed_s > self.config.action_timeout_s:
                    self._fail("local trajectory execution timed out")
                    return self._result(command)

                follower_output = self.follower.update(step_dt, applied_command)
                if follower_output.abort_reason is not None:
                    self._fail(follower_output.abort_reason)
                    return self._result(command)
                if follower_output.reached:
                    self._begin_action_finish()
                    return self._result(command)
                command[:] = follower_output.command

                if self.follower.needs_replan():
                    try:
                        fresh_front = self.camera.capture_forward(
                            completed_step, timestamp
                        )
                        new_path, fear = self.planner.get_plan(
                            fresh_front,
                            self.follower.current_goal_local_xy,
                        )
                        self.follower.replace_path(new_path)
                        self.last_fear = fear
                        self.replan_failures = 0
                    except Exception as exc:
                        self.replan_failures += 1
                        self.follower.defer_replan()
                        print(
                            "[LOCAL-VLN WARN] iPlanner replan failed "
                            f"({self.replan_failures}/{self.config.max_replan_failures}): {exc}"
                        )
                        if self.replan_failures > self.config.max_replan_failures:
                            self._fail("iPlanner exceeded the replan failure limit")
                            command.fill(0.0)

            elif self.state == EpisodeState.WAIT_ACTION_STAND:
                if stand_ready:
                    self.action_stand_elapsed_s += step_dt
                    if self.action_stand_elapsed_s >= self.config.post_action_stand_s:
                        self._commit_or_stop()
                else:
                    self.action_stand_elapsed_s = 0.0

        except Exception as exc:
            self._fail(str(exc))
            command.fill(0.0)

        self._log_state_transition()
        return self._result(command)

    def _capture_and_decide(self, completed_step: int, timestamp: float) -> None:
        panorama = self.camera.capture_panorama(completed_step, timestamp)
        history, history_images = build_model_history(self.history)
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

        if response.action.upper() == "BACKTRACK":
            raise RuntimeError(
                "BACKTRACK is preserved by the model schema but intentionally "
                "unsupported by the local no-FMM executor; robot remains stopped"
            )
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
        self.pending = _PendingAction(
            response=response,
            panorama=panorama,
            projection=projection,
            initial_forward_frame_id=panorama.views["forward"].frame_id,
        )
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
        if self.pending is None:
            raise RuntimeError("No pending action exists after rotation.")
        front = self.camera.capture_forward(completed_step, timestamp)
        if front.frame_id == self.pending.initial_forward_frame_id:
            raise RuntimeError(
                "Post-rotation forward RGB-D did not advance to a fresh sensor frame."
            )
        path, fear = self.planner.get_plan(
            front,
            self.pending.projection.goal_after_turn_xy_m,
        )
        self.pending.post_rotation_forward = front
        self.last_fear = fear
        safe_path = truncate_trajectory_for_safety(
            path, self.config.safe_distance_m
        )
        if safe_path is None:
            print(
                "[LOCAL-VLN] iPlanner target is inside the configured safe distance; "
                "no locomotion command will be issued."
            )
            self._begin_action_finish()
            return
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

    def _begin_action_finish(self) -> None:
        self.follower.stop()
        self.action_stand_elapsed_s = 0.0
        self.state = EpisodeState.WAIT_ACTION_STAND

    def _commit_or_stop(self) -> None:
        if self.pending is None:
            self._fail("pending action disappeared before history commit")
            return
        response = self.pending.response
        self._save_json(
            f"decision_{self.decision_index:03d}_completed.json",
            response_debug_dict(
                response,
                projected_goal_xy=self.pending.projection.goal_after_turn_xy_m,
            ),
        )
        if response.action.upper() == "STOP":
            self.pending = None
            self.state = EpisodeState.STOPPED
            print("[LOCAL-VLN] STOP final approach completed; episode stopped.")
            return

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
        if self.decision_index >= self.config.max_decisions:
            self.state = EpisodeState.STOPPED
            print("[LOCAL-VLN] Configured decision limit reached; episode stopped.")
        else:
            self.state = EpisodeState.CAPTURE_AND_DECIDE

    def _fail(self, reason: str) -> None:
        self.follower.stop()
        self.failure_reason = reason
        self.state = EpisodeState.FAILED
        self._save_json("failure.json", {"reason": reason})
        print(f"[LOCAL-VLN ERROR] {reason}")

    def _desired_mode(self) -> str:
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
        return EpisodeUpdate(
            command=np.asarray(command, dtype=np.float64).reshape(3).copy(),
            desired_mode=self._desired_mode(),
            state=self.state,
            completed=self.completed,
            failure_reason=self.failure_reason,
        )

    def _save_json(self, name: str, payload: dict) -> None:
        if self.output_dir is None:
            return
        path = self.output_dir / name
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _log_state_transition(self) -> None:
        if self.state != self._last_logged_state:
            print(f"[LOCAL-VLN] state={self.state}")
            self._last_logged_state = self.state
