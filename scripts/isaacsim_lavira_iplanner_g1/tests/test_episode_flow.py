from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
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
from unified_vln.model_client import CombinedModelClient  # noqa: E402
from unified_vln.model_contract import NavigationDecisionResponse  # noqa: E402
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


class EpisodeFlowTest(unittest.TestCase):
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
