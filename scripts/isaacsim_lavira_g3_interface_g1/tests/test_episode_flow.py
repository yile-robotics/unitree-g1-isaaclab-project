from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.episode import (  # noqa: E402
    EpisodeConfig,
    EpisodeState,
    LocalEndToEndEpisode,
    _create_unique_run_output_dir,
)
from unified_vln.local_trajectory import LocalFollowerConfig  # noqa: E402
from unified_vln.map_progress import (  # noqa: E402
    SparseEpisodeExplorationMap,
    SparseMapConfig,
)
from unified_vln.model_client import (  # noqa: E402
    CombinedModelClient,
    CompletedWaypoint,
)
from unified_vln.model_contract import NavigationDecisionResponse  # noqa: E402
from unified_vln.odometry import Pose2D  # noqa: E402
from unified_vln.types import DIRECTION_ORDER, PanoramaBundle, ViewFrame  # noqa: E402


def _frame(
    direction: str,
    *,
    frame_id: int,
    depth_m: float,
    sim_step: int,
    timestamp: float,
) -> ViewFrame:
    return ViewFrame(
        direction=direction,
        frame_id=frame_id,
        sim_step=sim_step,
        timestamp=timestamp,
        rgb=np.zeros((20, 20, 3), dtype=np.uint8),
        depth_m=np.full((20, 20), depth_m, dtype=np.float32),
        K=np.array(
            [[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )


class _Camera:
    def __init__(self):
        self.panorama_calls = 0
        self.forward_calls = 0

    def capture_panorama(self, sim_step: int, timestamp: float) -> PanoramaBundle:
        self.panorama_calls += 1
        depth_by_direction = {
            "forward": 4.0,
            "left": 2.0,
            "behind": 5.0,
            "right": 6.0,
        }
        views = {
            direction: _frame(
                direction,
                frame_id=10 + index,
                depth_m=depth_by_direction[direction],
                sim_step=sim_step,
                timestamp=timestamp,
            )
            for index, direction in enumerate(DIRECTION_ORDER)
        }
        return PanoramaBundle(1, sim_step, timestamp, views)

    def capture_forward(self, sim_step: int, timestamp: float) -> ViewFrame:
        self.forward_calls += 1
        return _frame(
            "forward",
            frame_id=100 + self.forward_calls,
            depth_m=7.0,
            sim_step=sim_step,
            timestamp=timestamp,
        )


class _SingleForwardSweepCamera(_Camera):
    def __init__(self):
        super().__init__()
        self.sweep_depths = [4.0, 2.0, 5.0, 6.0]

    def capture_forward(self, sim_step: int, timestamp: float) -> ViewFrame:
        self.forward_calls += 1
        index = min(self.forward_calls - 1, len(self.sweep_depths) - 1)
        return _frame(
            "forward",
            frame_id=200 + self.forward_calls,
            depth_m=self.sweep_depths[index],
            sim_step=sim_step,
            timestamp=timestamp,
        )


class _MissingSelectedDepthCamera(_Camera):
    def capture_panorama(self, sim_step: int, timestamp: float) -> PanoramaBundle:
        panorama = super().capture_panorama(sim_step, timestamp)
        panorama.views["left"].depth_m.fill(0.0)
        return panorama


class _Model:
    make_request = staticmethod(CombinedModelClient.make_request)

    @staticmethod
    def image_fields(bundle, request, history_images):
        del bundle, history_images
        fake_png = b"\x89PNG\r\n\x1a\nplaceholder"
        return {name: fake_png for name in request.required_image_fields}

    @staticmethod
    def decide(request, images):
        del images
        response = NavigationDecisionResponse(
            session_id=request.session_id,
            observation_id=request.observation_id,
            action="NAVIGATE",
            direction="left",
            target="door",
            bbox_2d=(8.0, 8.0, 12.0, 12.0),
            waypoint=None,
            progress_analysis="",
            reasoning="test",
        )
        return response, response.to_dict()


class _BlockingModel(_Model):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def decide(self, request, images):
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise RuntimeError("test model was not released")
        return super().decide(request, images)


class _Planner:
    def __init__(self):
        self.calls = []

    def get_plan(self, frame, goal_local_xy):
        self.calls.append((frame, np.asarray(goal_local_xy).copy()))
        return np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float64,
        ), 0.1


class _NoPathPlanner:
    def get_plan(self, frame, goal_local_xy):
        del frame, goal_local_xy
        return None, None


class _StaticOdometry:
    def __init__(self, pose: Pose2D):
        self.pose = pose

    def get_pose(self):
        return self.pose


class _BacktrackModel(_Model):
    @staticmethod
    def decide(request, images):
        del images
        response = NavigationDecisionResponse(
            session_id=request.session_id,
            observation_id=request.observation_id,
            action="BACKTRACK",
            direction=None,
            target=None,
            bbox_2d=None,
            waypoint=0,
            progress_analysis="return",
            reasoning="test backtrack",
        )
        return response, response.to_dict()


class _StopModel(_Model):
    @staticmethod
    def decide(request, images):
        del images
        response = NavigationDecisionResponse(
            session_id=request.session_id,
            observation_id=request.observation_id,
            action="STOP",
            direction="forward",
            target="open door",
            bbox_2d=(8.0, 8.0, 12.0, 12.0),
            waypoint=None,
            progress_analysis="progress",
            reasoning="reason",
        )
        return response, response.to_dict()


class _SessionClient:
    def __init__(
        self,
        *,
        motion_control="CONTINUE",
        complete_control="CONTINUE",
        complete_next_action=None,
        decision_supervision=None,
    ):
        self.motion_control = motion_control
        self.complete_control = complete_control
        self.complete_next_action = complete_next_action
        self.decision_supervision = decision_supervision
        self.calls = []

    def health_check(self):
        self.calls.append(("health",))
        return {"status": "ok"}

    def start_session(self, *, session_id, instruction):
        self.calls.append(("start", session_id, instruction))
        parsed = SimpleNamespace(stage_plan_id="sha256:test", stage_total=2)
        raw = {
            "response_type": "session_started",
            "session_id": session_id,
            "stage_plan_id": "sha256:test",
        }
        return parsed, raw

    def report_motion_window(self, **kwargs):
        self.calls.append(("motion_window", kwargs))
        parsed = SimpleNamespace(
            control=self.motion_control,
            next_action=(
                "ACTION_COMPLETE_PREEMPTED"
                if self.motion_control == "PREEMPT"
                else None
            ),
        )
        return parsed, {
            "response_type": "execution_control",
            "event_type": "motion_window",
            "control": self.motion_control,
        }

    def validate_decision_context(self, payload, *, decision_index):
        self.calls.append(("validate_decision", payload["session_id"]))
        del decision_index
        return self.decision_supervision

    def report_action_complete(self, **kwargs):
        self.calls.append(("action_complete", kwargs))
        next_action = self.complete_next_action
        if kwargs.get("status") == "PREEMPTED":
            next_action = "REQUEST_RECOVERY_DECISION"
        if self.complete_control == "SAFE_STOP":
            next_action = "SAFE_STOP"
        parsed = SimpleNamespace(
            control=self.complete_control,
            next_action=next_action,
        )
        return parsed, {
            "response_type": "execution_control",
            "event_type": "action_complete",
            "control": self.complete_control,
        }

    def end_session(self, *, status, reason):
        self.calls.append(("end", status, reason))
        parsed = SimpleNamespace(final_status=status, reason=reason)
        return parsed, {
            "response_type": "session_ended",
            "status": "ENDED",
            "final_status": status,
            "reason": reason,
        }


class _Phase4StopSessionClient(_SessionClient):
    def __init__(self, stop_phase):
        super().__init__()
        self.stop_phase = stop_phase

    def validate_decision_context(self, payload, *, decision_index):
        self.calls.append(("validate_decision", payload["session_id"]))
        progress = SimpleNamespace(
            parse_success=True,
            stage_completed=(2 if self.stop_phase != "PREMATURE_STOP" else 0),
            stage_total=2,
            current_stage=(
                "COMPLETED"
                if self.stop_phase != "PREMATURE_STOP"
                else "Go through the door."
            ),
            parse_error=None,
        )
        control = (
            "STOP_CONFIRMED"
            if self.stop_phase == "STOP_CONFIRMED"
            else "CONTINUE"
        )
        next_action = (
            "END_SESSION_SUCCESS"
            if self.stop_phase == "STOP_CONFIRMED"
            else "REQUEST_DECISION"
        )
        return SimpleNamespace(
            stage_progress=progress,
            stop_gate=SimpleNamespace(stop_phase=self.stop_phase),
            stop_phase=self.stop_phase,
            control=control,
            next_action=next_action,
        )

class EpisodeFlowTest(unittest.TestCase):
    def _poll_background_decision(
        self,
        episode,
        *,
        completed_step,
        timestamp,
        expected_state,
    ):
        update = None
        for offset in range(100):
            update = episode.update(
                completed_step=completed_step + offset,
                step_dt=0.02,
                timestamp=timestamp + offset * 0.02,
                applied_command=np.zeros(3),
                stand_ready=False,
                locomotion_ready=True,
            )
            if update.state != EpisodeState.WAITING_DECISION:
                break
            time.sleep(0.001)
        self.assertIsNotNone(update)
        self.assertEqual(update.state, expected_state)
        return update

    def _phase4_stop_episode(self, stop_phase):
        session = _Phase4StopSessionClient(stop_phase)
        planner = _Planner()
        pose = Pose2D(0.0, 0.0, 0.0, 0.0)
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id=f"phase4_{stop_phase.lower()}",
                instruction="Go through the door.",
                warmup_steps=0,
                max_decisions=None,
                min_depth_m=0.1,
                max_depth_m=10.0,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=_Camera(),
            model=_StopModel(),
            planner=planner,
            session_client=session,
            odometry=_StaticOdometry(pose),
            exploration_map=SparseEpisodeExplorationMap(
                SparseMapConfig(depth_stride=4),
                pose_frame_id="isaac_world",
            ),
        )
        episode.start_remote_session()
        update = episode.update(
            completed_step=0,
            step_dt=0.1,
            timestamp=0.0,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        return episode, update, session, planner

    def test_phase4_stop_confirmed_ends_without_execution_report(self):
        episode, update, session, planner = self._phase4_stop_episode(
            "STOP_CONFIRMED"
        )

        self.assertEqual(update.state, EpisodeState.STOPPED)
        self.assertTrue(update.completed)
        self.assertEqual(update.desired_mode, "stand")
        self.assertEqual(episode.session_success_reason, "stop_confirmed")
        self.assertIsNone(episode.pending)
        self.assertEqual(planner.calls, [])
        self.assertFalse(
            any(call[0] in {"motion_window", "action_complete"} for call in session.calls)
        )
        episode.end_remote_session(status="SUCCESS", reason="stop_confirmed")
        self.assertIn(("end", "SUCCESS", "stop_confirmed"), session.calls)

    def test_phase4_premature_and_pending_request_next_decision_without_motion(self):
        for stop_phase in ("PREMATURE_STOP", "STOP_PENDING"):
            with self.subTest(stop_phase=stop_phase):
                episode, update, session, planner = self._phase4_stop_episode(stop_phase)

                self.assertEqual(update.state, EpisodeState.CAPTURE_AND_DECIDE)
                self.assertFalse(update.completed)
                self.assertEqual(update.desired_mode, "stand")
                self.assertEqual(episode.decision_index, 1)
                self.assertIsNone(episode.pending)
                self.assertEqual(planner.calls, [])
                self.assertFalse(
                    any(
                        call[0] in {"motion_window", "action_complete"}
                        for call in session.calls
                    )
                )

    def test_single_forward_camera_rotates_four_quarters_and_relabels_depth(self):
        camera = _SingleForwardSweepCamera()
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="single_camera_panorama_test",
                instruction="go left",
                warmup_steps=0,
                single_forward_panorama=True,
                rotation_speed_rad_s=math.pi / 2.0,
                rotation_duration_scale=1.0,
                min_depth_m=0.1,
                max_depth_m=10.0,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=camera,
            model=_Model(),
            planner=_Planner(),
        )

        started = episode.update(
            completed_step=0,
            step_dt=0.1,
            timestamp=0.0,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        self.assertEqual(started.state, EpisodeState.WAIT_PANORAMA_LOCOMOTION)
        self.assertEqual(camera.forward_calls, 1)
        self.assertEqual(camera.panorama_calls, 0)

        for quarter in range(4):
            rotated = episode.update(
                completed_step=quarter + 1,
                step_dt=1.0,
                timestamp=float(quarter + 1),
                applied_command=np.array([0.0, 0.0, math.pi / 2.0]),
                stand_ready=False,
                locomotion_ready=True,
            )
            np.testing.assert_allclose(rotated.command, np.zeros(3))

        self.assertEqual(rotated.state, EpisodeState.PANORAMA_DECIDE)
        self.assertEqual(camera.forward_calls, 4)
        self.assertEqual(camera.panorama_calls, 0)

        decided = episode.update(
            completed_step=5,
            step_dt=0.1,
            timestamp=5.0,
            applied_command=np.zeros(3),
            stand_ready=False,
            locomotion_ready=True,
        )
        self.assertEqual(decided.state, EpisodeState.WAITING_DECISION)
        np.testing.assert_allclose(decided.command, np.zeros(3))
        self.assertEqual(decided.desired_mode, "locomotion")
        decided = self._poll_background_decision(
            episode,
            completed_step=6,
            timestamp=5.1,
            expected_state=EpisodeState.WAIT_ROTATION_LOCOMOTION,
        )
        self.assertIsNotNone(episode.pending)
        self.assertEqual(
            tuple(episode.pending.panorama.views), DIRECTION_ORDER
        )
        self.assertEqual(
            [
                episode.pending.panorama.views[direction].frame_id
                for direction in DIRECTION_ORDER
            ],
            [201, 202, 203, 204],
        )
        self.assertTrue(
            np.all(episode.pending.panorama.views["left"].depth_m == 2.0)
        )
        np.testing.assert_allclose(
            episode.pending.projection.goal_after_turn_xy_m,
            [2.0, 0.0],
            atol=1e-6,
        )

    def test_single_camera_model_wait_keeps_control_ticks_at_zero_velocity(self):
        camera = _SingleForwardSweepCamera()
        model = _BlockingModel()
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="async_decision_wait_test",
                instruction="go left",
                warmup_steps=0,
                single_forward_panorama=True,
                rotation_speed_rad_s=math.pi / 2.0,
                rotation_duration_scale=1.0,
                min_depth_m=0.1,
                max_depth_m=10.0,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=camera,
            model=model,
            planner=_Planner(),
        )

        episode.update(
            completed_step=0,
            step_dt=0.1,
            timestamp=0.0,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        for quarter in range(4):
            episode.update(
                completed_step=quarter + 1,
                step_dt=1.0,
                timestamp=float(quarter + 1),
                applied_command=np.array([0.0, 0.0, math.pi / 2.0]),
                stand_ready=False,
                locomotion_ready=True,
            )

        started = episode.update(
            completed_step=5,
            step_dt=0.02,
            timestamp=5.0,
            applied_command=np.zeros(3),
            stand_ready=False,
            locomotion_ready=True,
        )
        self.assertTrue(model.started.wait(timeout=1.0))
        self.assertEqual(started.state, EpisodeState.WAITING_DECISION)
        self.assertEqual(started.desired_mode, "locomotion")
        np.testing.assert_allclose(started.command, np.zeros(3))

        for step in range(6, 11):
            waiting = episode.update(
                completed_step=step,
                step_dt=0.02,
                timestamp=step * 0.02,
                applied_command=np.zeros(3),
                stand_ready=False,
                locomotion_ready=True,
            )
            self.assertEqual(waiting.state, EpisodeState.WAITING_DECISION)
            self.assertEqual(waiting.desired_mode, "locomotion")
            np.testing.assert_allclose(waiting.command, np.zeros(3))
        self.assertIsNone(episode.pending)

        model.release.set()
        finished = self._poll_background_decision(
            episode,
            completed_step=11,
            timestamp=0.22,
            expected_state=EpisodeState.WAIT_ROTATION_LOCOMOTION,
        )
        np.testing.assert_allclose(finished.command, np.zeros(3))
        self.assertIsNotNone(episode.pending)

    def test_single_forward_sweep_updates_sparse_map_and_window_gain_locally(self):
        camera = _SingleForwardSweepCamera()
        odometry = _StaticOdometry(Pose2D(0.0, 0.0, 0.0, 0.0))
        exploration_map = SparseEpisodeExplorationMap(
            SparseMapConfig(
                depth_stride=4,
                depth_max_m=10.0,
                camera_down_tilt_rad=0.0,
                robot_radius_m=0.10,
            ),
            pose_frame_id="isaac_world",
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            episode = LocalEndToEndEpisode(
                EpisodeConfig(
                    session_id="single_camera_map_test",
                    instruction="go left",
                    warmup_steps=0,
                    single_forward_panorama=True,
                    rotation_speed_rad_s=math.pi / 2.0,
                    rotation_duration_scale=1.0,
                    min_depth_m=0.1,
                    max_depth_m=10.0,
                    output_dir=Path(temporary_dir),
                ),
                LocalFollowerConfig(replan_interval_s=10.0),
                camera=camera,
                model=_Model(),
                planner=_Planner(),
                odometry=odometry,
                exploration_map=exploration_map,
            )
            episode.update(
                completed_step=0,
                step_dt=0.1,
                timestamp=0.0,
                applied_command=np.zeros(3),
                stand_ready=True,
                locomotion_ready=False,
            )
            for quarter in range(4):
                episode.update(
                    completed_step=quarter + 1,
                    step_dt=1.0,
                    timestamp=float(quarter + 1),
                    applied_command=np.array([0.0, 0.0, math.pi / 2.0]),
                    stand_ready=False,
                    locomotion_ready=True,
                )
            episode.update(
                completed_step=5,
                step_dt=0.1,
                timestamp=5.0,
                applied_command=np.zeros(3),
                stand_ready=False,
                locomotion_ready=True,
            )
            self._poll_background_decision(
                episode,
                completed_step=6,
                timestamp=5.1,
                expected_state=EpisodeState.WAIT_ROTATION_LOCOMOTION,
            )

            self.assertEqual(exploration_map.update_count, 4)
            self.assertGreater(exploration_map.explored_cells, 0)
            self.assertIsNotNone(episode.pending)
            baseline = exploration_map.explored_cells
            exploration_map.integrate(
                _frame(
                    "forward",
                    frame_id=999,
                    depth_m=3.0,
                    sim_step=6,
                    timestamp=6.0,
                ),
                Pose2D(0.75, 0.0, 0.0, 6.0),
            )
            episode.state = EpisodeState.EXECUTING
            episode.action_elapsed_s = episode.config.motion_window_s
            episode.next_motion_window_elapsed_s = episode.config.motion_window_s
            episode._map_window_explored_before = baseline
            episode._report_motion_window_if_due()

            progress_file = (
                episode.output_dir
                / "map_progress_decision_000_window_0000.json"
            )
            self.assertTrue(progress_file.is_file())
            payload = json.loads(progress_file.read_text())
            self.assertGreater(payload["new_explored_cells"], 0)
            self.assertEqual(payload["pose_frame_id"], "isaac_world")

    def test_backtrack_at_current_waypoint_reports_completion_and_continues(self):
        session = _SessionClient()
        pose = Pose2D(1.0, 0.0, 0.0, 1.0)
        exploration_map = SparseEpisodeExplorationMap(
            SparseMapConfig(depth_stride=4),
            pose_frame_id="isaac_world",
        )
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="backtrack_episode_test",
                instruction="return",
                warmup_steps=0,
                post_action_stand_s=0.1,
                backtrack_goal_tolerance_m=0.35,
                enable_backtrack=True,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=_Camera(),
            model=_BacktrackModel(),
            planner=_Planner(),
            session_client=session,
            odometry=_StaticOdometry(pose),
            exploration_map=exploration_map,
        )
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        episode.history.append(
            CompletedWaypoint(
                waypoint_id=0,
                decision_step=5,
                direction="forward",
                target="painting",
                init_rgb=image,
                direction_rgb=image,
                decision_pose=pose,
                arrival_pose=pose,
                executed_world_path_xy=np.array([[1.0, 0.0]]),
            )
        )
        episode.decision_index = 1
        episode.start_remote_session()

        first = episode.update(
            completed_step=10,
            step_dt=0.1,
            timestamp=1.0,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        self.assertEqual(first.state, EpisodeState.CAPTURE_AND_DECIDE)
        self.assertEqual(episode.decision_index, 2)
        complete_calls = [
            call[1] for call in session.calls if call[0] == "action_complete"
        ]
        self.assertEqual(len(complete_calls), 1)
        self.assertEqual(complete_calls[0]["action"], "BACKTRACK")
        self.assertEqual(complete_calls[0]["status"], "COMPLETED")

    def test_recovery_backtrack_uses_stable_registry_waypoint(self):
        supervision = SimpleNamespace(
            stage_progress=None,
            stop_gate=None,
            stop_phase=None,
            control="CONTINUE",
            next_action="EXECUTE_RECOVERY",
            action_source="RECOVERY",
            recovery=True,
        )
        session = _SessionClient(
            decision_supervision=supervision,
            complete_next_action="REQUEST_DECISION",
        )
        pose = Pose2D(1.0, 0.0, 0.0, 1.0)
        camera = _Camera()
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="recovery_backtrack_test",
                instruction="return",
                warmup_steps=0,
                post_action_stand_s=0.1,
                enable_backtrack=True,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=camera,
            model=_BacktrackModel(),
            planner=_Planner(),
            session_client=session,
            odometry=_StaticOdometry(pose),
            exploration_map=SparseEpisodeExplorationMap(
                SparseMapConfig(depth_stride=4), pose_frame_id="isaac_world"
            ),
        )
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        episode.history.append(
            CompletedWaypoint(
                waypoint_id=0,
                decision_step=5,
                direction="forward",
                target="painting",
                init_rgb=image,
                direction_rgb=image,
                decision_pose=pose,
                arrival_pose=pose,
                executed_world_path_xy=np.array([[1.0, 0.0]]),
            )
        )
        episode._server_waypoint_to_local_index[7] = 0
        episode._recovery_expected = True
        episode.decision_index = 8
        episode.start_remote_session()
        panorama = camera.capture_panorama(8, 1.0)
        response = NavigationDecisionResponse(
            session_id="recovery_backtrack_test",
            observation_id="recovery_backtrack_test_decision_008",
            action="BACKTRACK",
            direction=None,
            target=None,
            bbox_2d=None,
            waypoint=7,
            progress_analysis="recover",
            reasoning="return to stable registry waypoint",
        )

        episode._apply_decision_response(
            panorama,
            decision_pose=pose,
            response=response,
            raw_response={"session_id": "recovery_backtrack_test"},
        )

        self.assertEqual(episode.state, EpisodeState.WAIT_ACTION_STAND)
        self.assertEqual(episode.backtrack.wire_waypoint_id, 7)
        update = episode.update(
            completed_step=9,
            step_dt=0.1,
            timestamp=1.1,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        self.assertEqual(update.state, EpisodeState.CAPTURE_AND_DECIDE)
        self.assertFalse(episode._recovery_expected)
        complete = [call for call in session.calls if call[0] == "action_complete"]
        self.assertEqual(complete[0][1]["action"], "BACKTRACK")

    def test_g3_session_reports_motion_and_action_completion(self):
        session = _SessionClient()
        odometry = _StaticOdometry(Pose2D(0.0, 0.0, 0.0, 0.0))
        exploration_map = SparseEpisodeExplorationMap(
            SparseMapConfig(depth_stride=4),
            pose_frame_id="isaac_world",
        )
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="g3_episode_test",
                instruction="go to the door",
                warmup_steps=0,
                post_action_stand_s=0.1,
                min_depth_m=0.1,
                max_depth_m=10.0,
                motion_window_s=1.0,
            ),
            LocalFollowerConfig(
                goal_tolerance_m=0.1,
                blind_yaw_radius_m=0.0,
                replan_interval_s=10.0,
            ),
            camera=_Camera(),
            model=_Model(),
            planner=_Planner(),
            session_client=session,
            odometry=odometry,
            exploration_map=exploration_map,
        )
        episode.start_remote_session()
        self.assertTrue(episode.remote_session_active)

        # Create decision 0 and then isolate the continuous execution portion.
        episode.update(
            completed_step=0,
            step_dt=0.1,
            timestamp=0.0,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        episode.follower.start(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([2.0, 0.0]),
        )
        episode.state = EpisodeState.EXECUTING
        episode.action_elapsed_s = 0.0
        episode.next_motion_window_elapsed_s = 1.0
        episode._reset_motion_window_baseline()
        for step in range(2):
            odometry.pose = Pose2D(
                0.1 * (step + 1), 0.0, 0.0, (step + 1) * 0.5
            )
            episode.update(
                completed_step=step + 1,
                step_dt=0.5,
                timestamp=(step + 1) * 0.5,
                applied_command=np.zeros(3),
                stand_ready=False,
                locomotion_ready=True,
            )
        motion_calls = [call[1] for call in session.calls if call[0] == "motion_window"]
        self.assertEqual(len(motion_calls), 1)
        self.assertEqual(motion_calls[0]["decision_index"], 0)
        self.assertEqual(motion_calls[0]["window_index"], 0)
        self.assertEqual(motion_calls[0]["pose_start"], [0.0, 0.0, 0.0])
        self.assertEqual(motion_calls[0]["pose_end"], [0.2, 0.0, 0.0])
        self.assertAlmostEqual(motion_calls[0]["displacement_m"], 0.2)
        self.assertEqual(motion_calls[0]["timestamp_start"], 0.0)
        self.assertEqual(motion_calls[0]["timestamp_end"], 1.0)
        self.assertEqual(
            set(motion_calls[0]["map_progress"]),
            {
                "resolution_m",
                "explored_cells",
                "new_explored_cells",
                "traversable_cells",
            },
        )
        self.assertEqual(motion_calls[0]["map_progress"]["resolution_m"], 0.05)
        self.assertGreaterEqual(
            motion_calls[0]["map_progress"]["explored_cells"], 0
        )

        episode.state = EpisodeState.WAIT_ACTION_STAND
        episode.update(
            completed_step=3,
            step_dt=0.1,
            timestamp=1.1,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        complete_calls = [
            call[1] for call in session.calls if call[0] == "action_complete"
        ]
        self.assertEqual(len(complete_calls), 1)
        self.assertEqual(complete_calls[0]["status"], "COMPLETED")
        self.assertTrue(complete_calls[0]["reached_local_goal"])
        self.assertEqual(complete_calls[0]["planner_result"], "REACHED")
        self.assertEqual(complete_calls[0]["waypoint_id"], 0)
        episode.end_remote_session(status="SUCCESS", reason="test_completed")
        self.assertFalse(episode.remote_session_active)
        self.assertIn(("end", "SUCCESS", "test_completed"), session.calls)

    def test_g3_motion_preempt_atomically_stops_and_requests_recovery(self):
        session = _SessionClient(motion_control="PREEMPT")
        exploration_map = SparseEpisodeExplorationMap(
            SparseMapConfig(depth_stride=4),
            pose_frame_id="isaac_world",
        )
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="g3_preempt_test",
                instruction="go",
                warmup_steps=0,
                motion_window_s=0.1,
            ),
            LocalFollowerConfig(
                goal_tolerance_m=0.1,
                blind_yaw_radius_m=0.0,
                replan_interval_s=10.0,
            ),
            camera=_Camera(),
            model=_Model(),
            planner=_Planner(),
            session_client=session,
            odometry=_StaticOdometry(Pose2D(0.0, 0.0, 0.0, 0.0)),
            exploration_map=exploration_map,
        )
        episode.start_remote_session()
        episode.update(
            completed_step=0,
            step_dt=0.1,
            timestamp=0.0,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        episode.follower.start(
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([2.0, 0.0]),
        )
        episode.state = EpisodeState.EXECUTING
        episode.next_motion_window_elapsed_s = 0.1
        episode._reset_motion_window_baseline()
        result = episode.update(
            completed_step=1,
            step_dt=0.1,
            timestamp=0.1,
            applied_command=np.zeros(3),
            stand_ready=False,
            locomotion_ready=True,
        )
        self.assertEqual(result.state, EpisodeState.WAIT_ACTION_STAND)
        self.assertIsNone(result.failure_reason)
        np.testing.assert_allclose(result.command, np.zeros(3))
        self.assertFalse(episode.follower.active)

        result = episode.update(
            completed_step=2,
            step_dt=1.0,
            timestamp=1.1,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        self.assertEqual(result.state, EpisodeState.CAPTURE_AND_DECIDE)
        self.assertTrue(episode._recovery_expected)
        complete_calls = [
            call[1] for call in session.calls if call[0] == "action_complete"
        ]
        self.assertEqual(len(complete_calls), 1)
        self.assertEqual(complete_calls[0]["status"], "PREEMPTED")
        self.assertEqual(complete_calls[0]["planner_result"], "PREEMPTED")

    def test_semantic_preempt_ignores_navigate_and_requests_recovery(self):
        supervision = SimpleNamespace(
            stage_progress=None,
            stop_gate=None,
            stop_phase=None,
            control="PREEMPT",
            next_action="ACTION_COMPLETE_PREEMPTED",
            action_source="NAVIGATOR",
            recovery=False,
            preempt_source="semantic_audit",
        )
        session = _SessionClient(decision_supervision=supervision)
        pose = Pose2D(0.0, 0.0, 0.0, 0.0)
        camera = _Camera()
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="semantic_preempt_test",
                instruction="go",
                warmup_steps=0,
                post_action_stand_s=0.1,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=camera,
            model=_Model(),
            planner=_Planner(),
            session_client=session,
            odometry=_StaticOdometry(pose),
            exploration_map=SparseEpisodeExplorationMap(
                SparseMapConfig(depth_stride=4), pose_frame_id="isaac_world"
            ),
        )
        episode.start_remote_session()
        panorama = camera.capture_panorama(0, 0.0)
        response, raw = _Model.decide(
            _Model.make_request(
                panorama,
                session_id="semantic_preempt_test",
                instruction="go",
                decision_index=0,
            ),
            {},
        )
        raw["session_id"] = "semantic_preempt_test"

        episode._apply_decision_response(
            panorama, decision_pose=pose, response=response, raw_response=raw
        )

        self.assertEqual(episode.state, EpisodeState.WAIT_ACTION_STAND)
        self.assertFalse(episode.follower.active)
        update = episode.update(
            completed_step=1,
            step_dt=0.1,
            timestamp=0.1,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        self.assertEqual(update.state, EpisodeState.CAPTURE_AND_DECIDE)
        self.assertTrue(episode._recovery_expected)

    def test_premature_stop_preempt_acks_stop_then_requests_recovery(self):
        supervision = SimpleNamespace(
            stage_progress=None,
            stop_gate=SimpleNamespace(verdict="PREMATURE"),
            stop_phase="PREMATURE_STOP",
            control="PREEMPT",
            next_action="ACTION_COMPLETE_PREEMPTED",
            action_source="NAVIGATOR",
            recovery=False,
            preempt_source="premature_stop",
        )
        session = _SessionClient(decision_supervision=supervision)
        pose = Pose2D(0.0, 0.0, 0.0, 0.0)
        camera = _Camera()
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="premature_stop_preempt_test",
                instruction="go",
                warmup_steps=0,
                post_action_stand_s=0.1,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=camera,
            model=_StopModel(),
            planner=_Planner(),
            session_client=session,
            odometry=_StaticOdometry(pose),
            exploration_map=SparseEpisodeExplorationMap(
                SparseMapConfig(depth_stride=4), pose_frame_id="isaac_world"
            ),
        )
        episode.start_remote_session()
        panorama = camera.capture_panorama(0, 0.0)
        response, raw = _StopModel.decide(
            _StopModel.make_request(
                panorama,
                session_id="premature_stop_preempt_test",
                instruction="go",
                decision_index=0,
            ),
            {},
        )
        raw["session_id"] = "premature_stop_preempt_test"

        episode._apply_decision_response(
            panorama, decision_pose=pose, response=response, raw_response=raw
        )

        self.assertEqual(episode.state, EpisodeState.WAIT_ACTION_STAND)
        self.assertFalse(episode.follower.active)
        update = episode.update(
            completed_step=1,
            step_dt=0.1,
            timestamp=0.1,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        self.assertEqual(update.state, EpisodeState.CAPTURE_AND_DECIDE)
        self.assertTrue(episode._recovery_expected)
        complete_calls = [
            call[1] for call in session.calls if call[0] == "action_complete"
        ]
        self.assertEqual(len(complete_calls), 1)
        self.assertEqual(complete_calls[0]["action"], "STOP")
        self.assertEqual(complete_calls[0]["status"], "PREEMPTED")
        self.assertFalse(complete_calls[0]["reached_local_goal"])
        self.assertEqual(complete_calls[0]["planner_result"], "PREEMPTED")

    def test_recovery_navigate_one_escape_success_hands_back(self):
        supervision = SimpleNamespace(
            stage_progress=None,
            stop_gate=None,
            stop_phase=None,
            control="CONTINUE",
            next_action="EXECUTE_RECOVERY",
            action_source="RECOVERY",
            recovery=True,
        )
        session = _SessionClient(
            decision_supervision=supervision,
            complete_next_action="REQUEST_DECISION",
        )
        pose = Pose2D(0.0, 0.0, 0.0, 0.0)
        camera = _Camera()
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="recovery_navigate_test",
                instruction="go",
                warmup_steps=0,
                post_action_stand_s=0.1,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=camera,
            model=_Model(),
            planner=_Planner(),
            session_client=session,
            odometry=_StaticOdometry(pose),
            exploration_map=SparseEpisodeExplorationMap(
                SparseMapConfig(depth_stride=4), pose_frame_id="isaac_world"
            ),
        )
        episode.start_remote_session()
        episode._recovery_expected = True
        episode.decision_index = 3
        panorama = camera.capture_panorama(0, 0.0)
        response, raw = _Model.decide(
            _Model.make_request(
                panorama,
                session_id="recovery_navigate_test",
                instruction="go",
                decision_index=3,
            ),
            {},
        )
        raw["session_id"] = "recovery_navigate_test"
        episode._apply_decision_response(
            panorama, decision_pose=pose, response=response, raw_response=raw
        )
        episode._begin_action_finish()

        update = episode.update(
            completed_step=1,
            step_dt=0.1,
            timestamp=0.1,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )

        self.assertEqual(update.state, EpisodeState.CAPTURE_AND_DECIDE)
        self.assertFalse(episode._recovery_expected)
        self.assertEqual(episode._server_waypoint_to_local_index, {3: 0})
        self.assertEqual(episode.decision_index, 4)

    def test_recovery_safe_stop_fails_closed_with_session_reason(self):
        session = _SessionClient(
            complete_control="SAFE_STOP",
            complete_next_action="SAFE_STOP",
        )
        pose = Pose2D(0.0, 0.0, 0.0, 0.0)
        camera = _Camera()
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="recovery_safe_stop_test",
                instruction="go",
                warmup_steps=0,
                post_action_stand_s=0.1,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=camera,
            model=_Model(),
            planner=_Planner(),
            session_client=session,
            odometry=_StaticOdometry(pose),
            exploration_map=SparseEpisodeExplorationMap(
                SparseMapConfig(depth_stride=4), pose_frame_id="isaac_world"
            ),
        )
        episode.start_remote_session()
        panorama = camera.capture_panorama(0, 0.0)
        response, _ = _Model.decide(
            _Model.make_request(
                panorama,
                session_id="recovery_safe_stop_test",
                instruction="go",
                decision_index=0,
            ),
            {},
        )
        episode.pending = SimpleNamespace(
            response=response,
            panorama=panorama,
            projection=SimpleNamespace(goal_after_turn_xy_m=np.array([1.0, 0.0])),
            decision_pose=pose,
            action_source="RECOVERY",
        )
        episode._begin_action_finish()

        update = episode.update(
            completed_step=1,
            step_dt=0.1,
            timestamp=0.1,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )

        self.assertEqual(update.state, EpisodeState.FAILED)
        self.assertEqual(update.failure_reason, "recovery_safe_stop")
        self.assertEqual(episode.session_failure_reason, "recovery_safe_stop")

    def test_recovery_safe_stop_decision_bypasses_navigator_response(self):
        supervision = SimpleNamespace(
            stage_progress=None,
            stop_gate=None,
            stop_phase="SAFE_STOP",
            control="SAFE_STOP",
            next_action="SAFE_STOP",
            action_source="RECOVERY",
            recovery=True,
        )
        session = _SessionClient(decision_supervision=supervision)
        pose = Pose2D(0.0, 0.0, 0.0, 0.0)
        camera = _Camera()
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="recovery_safe_stop_decision_test",
                instruction="go",
                warmup_steps=0,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=camera,
            model=_Model(),
            planner=_Planner(),
            session_client=session,
            odometry=_StaticOdometry(pose),
            exploration_map=SparseEpisodeExplorationMap(
                SparseMapConfig(depth_stride=4), pose_frame_id="isaac_world"
            ),
        )
        episode.start_remote_session()
        panorama = camera.capture_panorama(0, 0.0)

        episode._apply_decision_response(
            panorama,
            decision_pose=pose,
            response=None,
            raw_response={
                "session_id": "recovery_safe_stop_decision_test",
                "control": "SAFE_STOP",
            },
        )

        self.assertEqual(episode.state, EpisodeState.FAILED)
        self.assertEqual(episode.session_failure_reason, "recovery_safe_stop")
        self.assertTrue(episode._safe_stop_requested)

    def test_failed_recovery_requests_another_plan_without_waypoint_commit(self):
        session = _SessionClient(
            complete_next_action="REQUEST_RECOVERY_DECISION"
        )
        pose = Pose2D(0.0, 0.0, 0.0, 0.0)
        camera = _Camera()
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="recovery_retry_test",
                instruction="go",
                warmup_steps=0,
                post_action_stand_s=0.1,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=camera,
            model=_Model(),
            planner=_Planner(),
            session_client=session,
            odometry=_StaticOdometry(pose),
            exploration_map=SparseEpisodeExplorationMap(
                SparseMapConfig(depth_stride=4), pose_frame_id="isaac_world"
            ),
        )
        episode.start_remote_session()
        panorama = camera.capture_panorama(0, 0.0)
        response, _ = _Model.decide(
            _Model.make_request(
                panorama,
                session_id="recovery_retry_test",
                instruction="go",
                decision_index=0,
            ),
            {},
        )
        episode.pending = SimpleNamespace(
            response=response,
            panorama=panorama,
            projection=SimpleNamespace(goal_after_turn_xy_m=np.array([1.0, 0.0])),
            decision_pose=pose,
            action_source="RECOVERY",
        )
        episode._begin_action_finish(
            status="FAILED",
            reason="recovery_escape_not_proven",
            planner_result="EXECUTION_FAILED",
        )

        update = episode.update(
            completed_step=1,
            step_dt=0.1,
            timestamp=0.1,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )

        self.assertEqual(update.state, EpisodeState.CAPTURE_AND_DECIDE)
        self.assertTrue(episode._recovery_expected)
        self.assertEqual(episode.history, [])

    def test_missing_selected_depth_continues_with_uni_forward_fallback(self):
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="depth_fallback_test",
                instruction="go to the door",
                warmup_steps=0,
                min_depth_m=0.1,
                max_depth_m=10.0,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=_MissingSelectedDepthCamera(),
            model=_Model(),
            planner=_Planner(),
        )

        update = episode.update(
            completed_step=0,
            step_dt=0.1,
            timestamp=0.0,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )

        self.assertEqual(update.state, EpisodeState.WAIT_ROTATION_LOCOMOTION)
        self.assertIsNone(update.failure_reason)
        self.assertIsNotNone(episode.pending)
        self.assertTrue(episode.pending.projection.used_forward_fallback)
        np.testing.assert_allclose(
            episode.pending.projection.goal_after_turn_xy_m,
            [1.5, 0.0],
        )

    def test_repeated_session_uses_a_distinct_local_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            base_dir = Path(temporary_dir)
            first = _create_unique_run_output_dir(base_dir, "repeated_session")
            second = _create_unique_run_output_dir(base_dir, "repeated_session")

            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, base_dir / "repeated_session")
            self.assertEqual(second.parent, base_dir / "repeated_session")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_default_decision_limit_is_unbounded(self):
        self.assertIsNone(
            EpisodeConfig(session_id="test", instruction="go").validated().max_decisions
        )

    def test_unbounded_episode_continues_after_completed_navigate(self):
        camera = _Camera()
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="unbounded_test",
                instruction="keep navigating",
                warmup_steps=0,
                max_decisions=None,
                post_action_stand_s=0.1,
                min_depth_m=0.1,
                max_depth_m=10.0,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=camera,
            model=_Model(),
            planner=_Planner(),
        )
        episode.update(
            completed_step=0,
            step_dt=0.1,
            timestamp=0.0,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        self.assertIsNotNone(episode.pending)

        # 本测试只验证回合计数，不重复验证运动过程，因此直接模拟动作已完成。
        episode.state = EpisodeState.WAIT_ACTION_STAND
        result = episode.update(
            completed_step=1,
            step_dt=0.1,
            timestamp=0.1,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        self.assertEqual(result.state, EpisodeState.CAPTURE_AND_DECIDE)
        self.assertFalse(result.completed)
        self.assertEqual(episode.decision_index, 1)
        self.assertEqual(len(episode.history), 1)

    def test_selected_pre_turn_depth_then_fresh_forward_iplanner_depth(self):
        camera = _Camera()
        planner = _Planner()
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="episode_test",
                instruction="go to the door",
                warmup_steps=0,
                rotation_speed_rad_s=math.pi / 2.0,
                rotation_duration_scale=1.0,
                rotation_settle_s=0.1,
                post_action_stand_s=0.1,
                safe_distance_m=0.5,
                min_depth_m=0.1,
                max_depth_m=10.0,
                output_dir=None,
            ),
            LocalFollowerConfig(
                goal_tolerance_m=0.1,
                replan_interval_s=10.0,
            ),
            camera=camera,
            model=_Model(),
            planner=planner,
        )

        first = episode.update(
            completed_step=0,
            step_dt=0.25,
            timestamp=0.0,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        self.assertEqual(first.state, EpisodeState.WAIT_ROTATION_LOCOMOTION)
        self.assertIsNotNone(episode.pending)
        np.testing.assert_allclose(
            episode.pending.projection.goal_after_turn_xy_m,
            [2.0, 0.0],
            atol=1e-6,
        )

        saw_positive_left_rotation = False
        update = first
        for step in range(1, 8):
            update = episode.update(
                completed_step=step,
                step_dt=0.25,
                timestamp=step * 0.25,
                applied_command=update.command,
                stand_ready=False,
                locomotion_ready=True,
            )
            saw_positive_left_rotation |= update.command[2] > 0.0
            if planner.calls:
                break

        self.assertTrue(saw_positive_left_rotation)
        self.assertEqual(camera.panorama_calls, 1)
        self.assertEqual(camera.forward_calls, 1)
        self.assertEqual(len(planner.calls), 1)
        planned_frame, planned_goal = planner.calls[0]
        self.assertEqual(planned_frame.direction, "forward")
        self.assertGreater(planned_frame.frame_id, 10)
        np.testing.assert_allclose(planned_frame.depth_m, 7.0)
        np.testing.assert_allclose(planned_goal, [2.0, 0.0], atol=1e-6)
        self.assertIn(
            episode.state,
            {EpisodeState.WAIT_EXECUTION_LOCOMOTION, EpisodeState.EXECUTING},
        )

    def test_missing_initial_plan_skips_action_instead_of_failing_episode(self):
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="no_path_test",
                instruction="go to the door",
                warmup_steps=0,
                rotation_speed_rad_s=math.pi / 2.0,
                rotation_settle_s=0.1,
                post_action_stand_s=0.1,
                min_depth_m=0.1,
                max_depth_m=10.0,
            ),
            LocalFollowerConfig(replan_interval_s=10.0),
            camera=_Camera(),
            model=_Model(),
            planner=_NoPathPlanner(),
        )

        update = episode.update(
            completed_step=0,
            step_dt=0.25,
            timestamp=0.0,
            applied_command=np.zeros(3),
            stand_ready=True,
            locomotion_ready=False,
        )
        for step in range(1, 10):
            update = episode.update(
                completed_step=step,
                step_dt=0.25,
                timestamp=step * 0.25,
                applied_command=update.command,
                stand_ready=episode.state == EpisodeState.WAIT_ACTION_STAND,
                locomotion_ready=episode.state != EpisodeState.WAIT_ACTION_STAND,
            )
            if episode.state == EpisodeState.CAPTURE_AND_DECIDE:
                break

        self.assertNotEqual(episode.state, EpisodeState.FAILED)
        self.assertEqual(episode.state, EpisodeState.CAPTURE_AND_DECIDE)
        self.assertEqual(episode.decision_index, 1)
        self.assertEqual(episode.history, [])

    def test_replan_failure_keeps_old_path_without_failure_limit(self):
        camera = _Camera()
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id="replan_failure_test",
                instruction="go",
                warmup_steps=0,
            ),
            LocalFollowerConfig(
                goal_tolerance_m=0.1,
                blind_yaw_radius_m=0.0,
                replan_interval_s=0.1,
            ),
            camera=camera,
            model=_Model(),
            planner=_NoPathPlanner(),
        )
        episode.follower.start(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([2.0, 0.0]),
        )
        episode.state = EpisodeState.EXECUTING

        for step in range(5):
            # Uni 使用墙钟而不是传入的 step_dt 判断重规划时机。
            episode.follower.last_replan_time_s = 0.0
            update = episode.update(
                completed_step=step,
                step_dt=0.1,
                timestamp=step * 0.1,
                applied_command=np.zeros(3),
                stand_ready=False,
                locomotion_ready=True,
            )
            self.assertEqual(update.state, EpisodeState.EXECUTING)
            self.assertIsNone(update.failure_reason)

        self.assertTrue(episode.follower.active)
        self.assertEqual(camera.forward_calls, 5)


if __name__ == "__main__":
    unittest.main()
