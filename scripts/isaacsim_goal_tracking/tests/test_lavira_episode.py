from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from goal_tracking.frame_bundle import CameraFrame, FrameBundle  # noqa: E402
from goal_tracking.lavira_episode import (  # noqa: E402
    EpisodeState,
    LaViRABoundedEpisodeController,
    RuntimeWaypointRecord,
    build_backtrack_execution_request,
    build_replanned_backtrack_execution_request,
    build_server_history,
)
from goal_tracking.lavira_protocol import NavigationDecisionResponse  # noqa: E402


def make_bundle(decision_index: int = 0) -> FrameBundle:
    views = {}
    for frame_id, direction in enumerate(
        ("forward", "left", "behind", "right"), start=1
    ):
        rgb = np.full(
            (6, 8, 3), decision_index * 50 + frame_id * 10, dtype=np.uint8
        )
        views[direction] = CameraFrame(
            camera_id=f"camera_{direction}",
            direction=direction,
            sensor_frame_id=frame_id,
            sim_step=5 + decision_index * 100,
            timestamp=0.1 + decision_index * 2.0,
            rgb=rgb,
            depth_z_m=np.ones((6, 8), dtype=np.float32),
            K=np.eye(3),
            T_world_camera_ros=np.eye(4),
            T_base_camera=np.eye(4),
        )
    return FrameBundle(
        bundle_id=decision_index,
        env_index=0,
        sim_step=5 + decision_index * 100,
        timestamp=0.1 + decision_index * 2.0,
        T_world_base=np.eye(4),
        views=views,
    )


def make_runtime_record(index: int, *, arrived: bool = True) -> RuntimeWaypointRecord:
    path = np.array(
        [[float(index), 0.0], [float(index + 1), 0.0]], dtype=np.float64
    )
    decision_pose = np.eye(4)
    decision_pose[0, 3] = float(index)
    return RuntimeWaypointRecord(
        waypoint_id=index,
        decision_index=index,
        decision_step=index * 10,
        decision_world_pose=decision_pose,
        direction="left",
        target=f"target_{index}",
        bbox_2d=(0, 0, 8, 6),
        progress_analysis="progress",
        reasoning="reasoning",
        init_rgb=np.full((6, 8, 3), index, dtype=np.uint8),
        dir_rgb=np.full((6, 8, 3), index + 10, dtype=np.uint8),
        projected_target_world=np.array([1.0, 2.0, 0.0]),
        safe_target_world_xy=path[-1].copy(),
        fmm_waypoints_world_xy=path,
        execution_status=("arrived" if arrived else "candidate"),
        arrival_step=(index * 10 + 5 if arrived else None),
    )


def make_probe(
    decision_index: int = 0,
    *,
    action: str = "NAVIGATE",
    direction: str = "left",
    waypoint: int = 0,
):
    bundle = make_bundle(decision_index)
    if action == "BACKTRACK":
        response = NavigationDecisionResponse(
            session_id="robot_01_history_test",
            observation_id=f"robot_01_history_test_decision_{decision_index:03d}",
            action=action,
            direction=None,
            target=None,
            bbox_2d=None,
            waypoint=waypoint,
            progress_analysis="The route is unproductive.",
            reasoning="Return to waypoint zero.",
        )
    else:
        response = NavigationDecisionResponse(
            session_id="robot_01_history_test",
            observation_id=f"robot_01_history_test_decision_{decision_index:03d}",
            action=action,
            direction=direction,
            target=f"target_{decision_index}",
            bbox_2d=(0, 0, 8, 6),
            waypoint=None,
            progress_analysis=f"Progress at decision {decision_index}.",
            reasoning=f"Reasoning at decision {decision_index}.",
        )
    return SimpleNamespace(
        completed=True,
        decision_index=decision_index,
        bundle=bundle,
        response=response,
        target_projection=SimpleNamespace(
            point_world_m=np.array([1.0, 2.0, 0.5])
        ),
        navigation_map=SimpleNamespace(
            safe_target_world_xy=np.array([0.8, 1.8])
        ),
        fmm_plan=SimpleNamespace(
            waypoints_world_xy=np.array([[0.0, 0.0], [0.8, 1.8]])
        ),
        output_dir=None,
    )


class FakePathFollower:
    def __init__(self):
        self.enabled = True
        self.goal_reached = False
        self.abort_reason = None
        self.stop_reason = None
        self.pose = SimpleNamespace(x=0.0, y=0.0, yaw=0.0, valid=True)

    def stop(self, reason=None):
        self.enabled = False
        self.stop_reason = reason

    def current_robot_pose(self):
        return self.pose


class FakeCommandController:
    def __init__(self):
        self.zero_count = 0

    def zero(self):
        self.zero_count += 1


class FakeSwitchState:
    def __init__(self):
        self.active_mode = "stand"
        self.transition_mode = "none"
        self.stand_requests = 0

    def request_stand(self):
        self.stand_requests += 1

    def current_tilt_angle(self):
        return 0.0


def make_args():
    return SimpleNamespace(
        lavira_history_probe=True,
        lavira_history_max_decisions=3,
        lavira_decision_warmup_steps=0,
        lavira_history_execution_timeout=30.0,
        lavira_online_navigation=False,
        lavira_online_mapping_interval_s=1.0,
        lavira_online_replan_interval_s=1.0,
        lavira_collision_command_speed_m_s=0.12,
        lavira_collision_window_s=0.75,
        lavira_collision_min_progress_m=0.04,
        lavira_collision_mark_distance_m=0.45,
        lavira_collision_mark_radius_m=0.15,
        lavira_stop_reached_threshold_m=0.75,
        lavira_backtrack_max_path_m=6.0,
        lavira_backtrack_strategy="stored_reverse",
        lavira_history_settle_seconds=0.2,
        fmm_execute_tilt_abort_rad=0.5,
        fmm_execute_max_path_m=6.0,
    )


class LaViRAEpisodeHistoryTest(unittest.TestCase):
    def test_bounded_controller_requires_at_least_two_decisions(self) -> None:
        args = make_args()
        args.lavira_history_max_decisions = 1
        with self.assertRaisesRegex(ValueError, "at least two"):
            LaViRABoundedEpisodeController(args)

    def test_only_latest_four_history_points_carry_images(self) -> None:
        records = [make_runtime_record(index) for index in range(6)]
        history, images = build_server_history(records)

        self.assertEqual(len(history), 6)
        self.assertFalse(history[0].has_images)
        self.assertFalse(history[1].has_images)
        self.assertTrue(history[2].has_images)
        self.assertEqual(len(images), 8)
        self.assertIn("history_5_init", images)
        self.assertIn("history_5_dir", images)

    def test_candidate_waypoint_cannot_be_sent_as_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an arrived"):
            build_server_history([make_runtime_record(0, arrived=False)])

    def test_backtrack_request_reverses_all_paths_through_target(self) -> None:
        request = build_backtrack_execution_request(
            [make_runtime_record(0), make_runtime_record(1)],
            target_waypoint_id=0,
            decision_index=2,
        )

        np.testing.assert_allclose(
            request.fmm_plan.waypoints_world_xy,
            np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        )
        np.testing.assert_allclose(
            request.fmm_plan.start_world_xy, np.array([2.0, 0.0])
        )
        self.assertAlmostEqual(request.fmm_plan.path_length_m, 2.0)
        self.assertEqual(request.fmm_plan.target_waypoint_id, 0)
        self.assertIn("waypoint_000", request.fmm_plan.execution_source)

    @patch("goal_tracking.lavira_episode.build_fmm_plan")
    @patch("goal_tracking.lavira_episode.build_navigation_grid_map_for_world_goal")
    def test_replanned_backtrack_uses_selected_waypoint_decision_pose(
        self,
        build_map,
        build_plan,
    ) -> None:
        grid_map = SimpleNamespace(
            safe_target_world_xy=np.array([1.0, 0.0], dtype=np.float64)
        )
        fmm_plan = SimpleNamespace(
            start_world_xy=np.array([2.0, 0.0], dtype=np.float64),
            waypoints_world_xy=np.array(
                [[2.0, 0.0], [1.0, 0.0]], dtype=np.float64
            ),
            path_length_m=1.0,
        )
        build_map.return_value = grid_map
        build_plan.return_value = fmm_plan

        request, returned_map, returned_plan = (
            build_replanned_backtrack_execution_request(
                [make_runtime_record(0), make_runtime_record(1)],
                make_bundle(2),
                target_waypoint_id=1,
                decision_index=2,
                navigation_map_config=SimpleNamespace(),
                fmm_planner_config=SimpleNamespace(),
            )
        )

        np.testing.assert_allclose(
            build_map.call_args.args[1],
            np.array([1.0, 0.0]),
        )
        self.assertIs(returned_map, grid_map)
        self.assertIs(returned_plan, fmm_plan)
        np.testing.assert_allclose(
            request.fmm_plan.waypoints_world_xy,
            fmm_plan.waypoints_world_xy,
        )
        self.assertIn("replanned_backtrack", request.fmm_plan.execution_source)

    def test_three_decisions_execute_two_paths_and_commit_two_waypoints(self) -> None:
        controller = LaViRABoundedEpisodeController(make_args())
        decision_0 = make_probe(0, direction="left")
        decision_1 = make_probe(1, direction="forward")
        decision_2 = make_probe(2, direction="right")
        controller._run_probe = Mock(
            side_effect=[decision_0, decision_1, decision_2]
        )
        path = FakePathFollower()
        command = FakeCommandController()
        switch = FakeSwitchState()

        def start_path_side_effect(_probe):
            path.enabled = True
            path.goal_reached = False
            path.abort_reason = None
            return True

        start_path = Mock(side_effect=start_path_side_effect)

        controller.update_after_step(
            object(),
            completed_step=5,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )
        self.assertEqual(controller.state, EpisodeState.EXECUTING)
        self.assertEqual(controller.history, [])
        self.assertIsNotNone(controller.pending_waypoint)
        np.testing.assert_array_equal(
            controller.pending_waypoint.init_rgb,
            decision_0.bundle.views["forward"].rgb,
        )
        np.testing.assert_array_equal(
            controller.pending_waypoint.dir_rgb,
            decision_0.bundle.views["left"].rgb,
        )

        path.goal_reached = True
        controller.update_after_step(
            object(),
            completed_step=6,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )
        self.assertEqual(controller.state, EpisodeState.WAITING_FOR_STAND)

        for completed_step in (7, 8):
            controller.update_after_step(
                object(),
                completed_step=completed_step,
                step_dt=0.1,
                path_follower=path,
                path_visualizer=None,
                command_controller=command,
                switch_state=switch,
                start_path=start_path,
            )

        self.assertFalse(controller.completed)
        self.assertEqual(len(controller.history), 1)
        self.assertEqual(controller.history[0].execution_status, "arrived")
        self.assertIsNone(controller.pending_waypoint)

        controller.update_after_step(
            object(),
            completed_step=9,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )
        self.assertEqual(controller.state, EpisodeState.EXECUTING)
        self.assertEqual(controller.pending_waypoint.decision_index, 1)
        np.testing.assert_array_equal(
            controller.pending_waypoint.init_rgb,
            decision_1.bundle.views["forward"].rgb,
        )
        np.testing.assert_array_equal(
            controller.pending_waypoint.dir_rgb,
            decision_1.bundle.views["forward"].rgb,
        )

        path.goal_reached = True
        controller.update_after_step(
            object(),
            completed_step=10,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )
        for completed_step in (11, 12):
            controller.update_after_step(
                object(),
                completed_step=completed_step,
                step_dt=0.1,
                path_follower=path,
                path_visualizer=None,
                command_controller=command,
                switch_state=switch,
                start_path=start_path,
            )

        self.assertEqual(len(controller.history), 2)
        self.assertFalse(controller.completed)

        controller.update_after_step(
            object(),
            completed_step=13,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )

        self.assertTrue(controller.completed)
        self.assertEqual(controller.state, EpisodeState.TERMINAL_RESPONSE)
        self.assertEqual(controller.decisions_completed, 3)
        self.assertEqual(controller.terminal_response["action"], "NAVIGATE")
        self.assertEqual(controller._run_probe.call_count, 3)
        self.assertEqual(start_path.call_count, 2)
        start_path.assert_any_call(decision_0)
        start_path.assert_any_call(decision_1)
        called_indices = [
            call.kwargs["decision_index"]
            for call in controller._run_probe.call_args_list
        ]
        self.assertEqual(called_indices, [0, 1, 2])

        history, images = build_server_history(controller.history)
        self.assertEqual(len(history), 2)
        self.assertEqual(len(images), 4)
        self.assertEqual(history[1].turn_action, "turn forward")
        self.assertEqual(4 + len(images), 8)

    def test_path_abort_discards_candidate_history(self) -> None:
        controller = LaViRABoundedEpisodeController(make_args())
        controller._run_probe = Mock(return_value=make_probe(0))
        path = FakePathFollower()
        command = FakeCommandController()
        switch = FakeSwitchState()

        controller.update_after_step(
            object(),
            completed_step=5,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=Mock(return_value=True),
        )
        path.abort_reason = "cross-track error"
        controller.update_after_step(
            object(),
            completed_step=6,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=Mock(return_value=True),
        )

        self.assertEqual(controller.state, EpisodeState.FAILED)
        self.assertEqual(controller.history, [])
        self.assertIsNotNone(controller.pending_waypoint)
        self.assertIn("cross-track", controller.failure_reason)

    def test_stop_reuses_fmm_path_then_stands_at_lavira_threshold(self) -> None:
        controller = LaViRABoundedEpisodeController(make_args())
        stop_probe = make_probe(0, action="STOP", direction="forward")
        controller._run_probe = Mock(return_value=stop_probe)
        path = FakePathFollower()
        command = FakeCommandController()
        switch = FakeSwitchState()
        start_path = Mock(return_value=True)

        controller.update_after_step(
            object(),
            completed_step=5,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )

        self.assertFalse(controller.completed)
        self.assertEqual(controller.state, EpisodeState.EXECUTING)
        self.assertEqual(controller.active_execution_action, "STOP")
        self.assertEqual(controller.stop_event["status"], "accepted")
        self.assertEqual(controller.history, [])
        start_path.assert_called_once_with(stop_probe)

        path.pose = SimpleNamespace(x=0.4, y=1.4, yaw=0.0, valid=True)
        controller.update_after_step(
            object(),
            completed_step=6,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )

        self.assertEqual(controller.state, EpisodeState.WAITING_FOR_STAND)
        self.assertEqual(controller.stop_event["status"], "threshold_reached")
        self.assertLess(controller.stop_event["actual_distance_m"], 0.75)
        self.assertIn("STOP target threshold", path.stop_reason)

        for completed_step in (7, 8):
            controller.update_after_step(
                object(),
                completed_step=completed_step,
                step_dt=0.1,
                path_follower=path,
                path_visualizer=None,
                command_controller=command,
                switch_state=switch,
                start_path=start_path,
            )

        self.assertTrue(controller.completed)
        self.assertEqual(controller.state, EpisodeState.STOPPED)
        self.assertEqual(controller.stop_event["status"], "completed")
        self.assertTrue(controller.stop_event["robot_standing"])
        self.assertTrue(controller.stop_event["model_stop_completed"])
        self.assertIsNone(controller.stop_event["task_success"])
        self.assertEqual(controller.history, [])

    def test_stop_already_inside_threshold_does_not_start_locomotion(self) -> None:
        controller = LaViRABoundedEpisodeController(make_args())
        stop_probe = make_probe(0, action="STOP", direction="forward")
        controller._run_probe = Mock(return_value=stop_probe)
        path = FakePathFollower()
        path.pose = SimpleNamespace(x=0.8, y=1.7, yaw=0.0, valid=True)
        command = FakeCommandController()
        switch = FakeSwitchState()
        start_path = Mock(return_value=True)

        controller.update_after_step(
            object(),
            completed_step=5,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )

        self.assertEqual(controller.state, EpisodeState.WAITING_FOR_STAND)
        self.assertEqual(controller.stop_event["threshold_reached_step"], 5)
        start_path.assert_not_called()

    def test_stop_executes_even_on_final_bounded_decision(self) -> None:
        controller = LaViRABoundedEpisodeController(make_args())
        controller.history.extend([make_runtime_record(0), make_runtime_record(1)])
        controller.decisions_completed = 2
        controller.state = EpisodeState.CAPTURE_AND_DECIDE
        stop_probe = make_probe(2, action="STOP", direction="forward")
        controller._run_probe = Mock(return_value=stop_probe)
        path = FakePathFollower()
        command = FakeCommandController()
        switch = FakeSwitchState()
        start_path = Mock(return_value=True)

        controller.update_after_step(
            object(),
            completed_step=205,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )

        self.assertFalse(controller.completed)
        self.assertEqual(controller.state, EpisodeState.EXECUTING)
        self.assertEqual(controller.active_execution_action, "STOP")
        self.assertIsNone(controller.terminal_response)
        start_path.assert_called_once_with(stop_probe)

    def test_stop_safety_abort_is_failure_not_completion(self) -> None:
        controller = LaViRABoundedEpisodeController(make_args())
        controller._run_probe = Mock(
            return_value=make_probe(0, action="STOP", direction="forward")
        )
        path = FakePathFollower()
        command = FakeCommandController()
        switch = FakeSwitchState()
        start_path = Mock(return_value=True)

        controller.update_after_step(
            object(),
            completed_step=5,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )
        path.pose = SimpleNamespace(x=0.8, y=1.7, yaw=0.0, valid=True)
        path.abort_reason = "body tilt exceeded safety limit"
        controller.update_after_step(
            object(),
            completed_step=6,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )

        self.assertEqual(controller.state, EpisodeState.FAILED)
        self.assertEqual(controller.stop_event["status"], "failed")
        self.assertFalse(controller.stop_event["model_stop_completed"])
        self.assertIn("body tilt", controller.failure_reason)

    @patch("goal_tracking.lavira_episode.fmm_planner_config_from_args")
    @patch("goal_tracking.lavira_episode.navigation_map_config_from_args")
    @patch(
        "goal_tracking.lavira_episode.build_replanned_backtrack_execution_request"
    )
    def test_default_backtrack_replans_world_goal_and_truncates_on_accept(
        self,
        build_replanned,
        map_config_from_args,
        planner_config_from_args,
    ) -> None:
        args = make_args()
        args.lavira_history_max_decisions = 4
        args.lavira_backtrack_strategy = "replan_world_goal"
        controller = LaViRABoundedEpisodeController(args)
        records = [make_runtime_record(0), make_runtime_record(1)]
        controller.history.extend(records)
        controller.decisions_completed = 2
        controller.state = EpisodeState.CAPTURE_AND_DECIDE
        backtrack_probe = make_probe(2, action="BACKTRACK", waypoint=0)
        controller._run_probe = Mock(return_value=backtrack_probe)
        request = build_backtrack_execution_request(
            records,
            target_waypoint_id=0,
            decision_index=2,
        )
        build_replanned.return_value = (
            request,
            SimpleNamespace(),
            SimpleNamespace(),
        )
        path = FakePathFollower()
        command = FakeCommandController()
        switch = FakeSwitchState()
        start_path = Mock(return_value=True)

        controller.update_after_step(
            object(),
            completed_step=205,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )

        self.assertEqual(controller.state, EpisodeState.EXECUTING)
        self.assertEqual(len(controller.history), 1)
        self.assertEqual(controller.backtrack_event["strategy"], "replan_world_goal")
        self.assertEqual(controller.backtrack_event["history_count_after"], 1)
        build_replanned.assert_called_once()
        self.assertIs(build_replanned.call_args.args[1], backtrack_probe.bundle)
        self.assertEqual(
            build_replanned.call_args.kwargs["target_waypoint_id"],
            0,
        )
        map_config_from_args.assert_called_once_with(args)
        planner_config_from_args.assert_called_once_with(args)
        start_path.assert_called_once_with(request)

    def test_intermediate_backtrack_truncates_on_accept_then_executes(self) -> None:
        args = make_args()
        args.lavira_history_max_decisions = 4
        controller = LaViRABoundedEpisodeController(args)
        controller.history.extend([make_runtime_record(0), make_runtime_record(1)])
        controller.decisions_completed = 2
        controller.state = EpisodeState.CAPTURE_AND_DECIDE
        backtrack_probe = make_probe(2, action="BACKTRACK", waypoint=0)
        controller._run_probe = Mock(return_value=backtrack_probe)
        path = FakePathFollower()
        command = FakeCommandController()
        switch = FakeSwitchState()
        start_path = Mock(return_value=True)

        controller.update_after_step(
            object(),
            completed_step=205,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )

        self.assertFalse(controller.completed)
        self.assertEqual(controller.state, EpisodeState.EXECUTING)
        self.assertEqual(controller.active_execution_action, "BACKTRACK")
        self.assertEqual(controller.backtrack_target_waypoint, 0)
        self.assertEqual(len(controller.history), 1)
        self.assertEqual(controller.history[0].waypoint_id, 0)
        self.assertEqual(controller.backtrack_event["history_count_before"], 2)
        self.assertEqual(controller.backtrack_event["history_count_after"], 1)
        start_path.assert_called_once()
        request = start_path.call_args.args[0]
        np.testing.assert_allclose(
            request.fmm_plan.waypoints_world_xy,
            np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        )

        path.goal_reached = True
        controller.update_after_step(
            object(),
            completed_step=206,
            step_dt=0.1,
            path_follower=path,
            path_visualizer=None,
            command_controller=command,
            switch_state=switch,
            start_path=start_path,
        )
        self.assertEqual(controller.state, EpisodeState.WAITING_FOR_STAND)

        for completed_step in (207, 208):
            controller.update_after_step(
                object(),
                completed_step=completed_step,
                step_dt=0.1,
                path_follower=path,
                path_visualizer=None,
                command_controller=command,
                switch_state=switch,
                start_path=start_path,
            )

        self.assertEqual(controller.state, EpisodeState.CAPTURE_AND_DECIDE)
        self.assertEqual(len(controller.history), 1)
        self.assertEqual(controller.history[0].waypoint_id, 0)
        self.assertEqual(controller.current_decision_index, 3)
        self.assertEqual(controller.backtrack_event["status"], "arrived")
        self.assertEqual(controller.backtrack_event["history_count_before"], 2)
        self.assertEqual(controller.backtrack_event["history_count_after"], 1)

    def test_online_collision_requires_sustained_commanded_non_progress(self) -> None:
        controller = LaViRABoundedEpisodeController(make_args())
        controller.global_map_state = SimpleNamespace(
            mark_collision_world_xy=Mock(return_value=7)
        )
        path = FakePathFollower()
        switch = FakeSwitchState()
        switch.active_mode = "locomotion"

        first = controller._update_online_collision_map(
            completed_step=10,
            current_time_s=1.0,
            path_follower=path,
            switch_state=switch,
            applied_velocity_command=np.array([0.2, 0.0, 0.0]),
        )
        second = controller._update_online_collision_map(
            completed_step=18,
            current_time_s=1.8,
            path_follower=path,
            switch_state=switch,
            applied_velocity_command=np.array([0.2, 0.0, 0.0]),
        )

        self.assertFalse(first)
        self.assertTrue(second)
        controller.global_map_state.mark_collision_world_xy.assert_called_once()
        np.testing.assert_allclose(
            controller.global_map_state.mark_collision_world_xy.call_args.args[0],
            [0.45, 0.0],
        )
        self.assertEqual(controller.online_collision_count, 1)

    @patch("goal_tracking.lavira_episode.fmm_planner_config_from_args")
    @patch("goal_tracking.lavira_episode.build_fmm_plan")
    def test_online_map_update_replans_same_stable_world_goal(
        self,
        build_plan,
        planner_config,
    ) -> None:
        controller = LaViRABoundedEpisodeController(make_args())
        controller.online_navigation = True
        planning_map = SimpleNamespace(
            safe_target_cell_rc=(2, 3),
        )
        state = SimpleNamespace(
            integrate_bundle=Mock(),
            build_navigation_grid_map=Mock(return_value=planning_map),
            update_count=4,
            full_map=np.zeros((4, 8, 8), dtype=np.uint8),
            collision_map=np.zeros((8, 8), dtype=bool),
        )
        controller.global_map_state = state
        initial_plan = SimpleNamespace(bundle_id=1)
        controller._begin_active_goal(
            action="NAVIGATE",
            decision_index=0,
            goal_world_xy=np.array([1.25, -0.5]),
            execution_max_path_m=6.0,
            completed_step=0,
            step_dt=0.1,
            initial_fmm_plan=initial_plan,
        )
        replanned = SimpleNamespace(
            bundle_id=9,
            goal_world_xy=np.array([1.25, -0.5]),
            waypoints_world_xy=np.array(
                [[0.0, 0.0], [1.25, -0.5]], dtype=np.float64
            ),
            path_length_m=1.35,
        )
        build_plan.return_value = replanned
        camera = SimpleNamespace(
            capture=Mock(return_value=SimpleNamespace(bundle_id=9))
        )
        hot_swap = Mock(return_value=True)
        path = FakePathFollower()
        command = FakeCommandController()
        switch = FakeSwitchState()

        controller._maybe_update_online_navigation(
            camera,
            completed_step=10,
            step_dt=0.1,
            path_follower=path,
            command_controller=command,
            switch_state=switch,
            hot_swap_path=hot_swap,
            applied_velocity_command=np.zeros(3),
        )

        camera.capture.assert_called_once()
        state.integrate_bundle.assert_called_once()
        np.testing.assert_allclose(
            state.build_navigation_grid_map.call_args.kwargs[
                "stable_target_world_xy"
            ],
            [1.25, -0.5],
        )
        build_plan.assert_called_once_with(
            planning_map,
            planner_config.return_value,
        )
        hot_swap.assert_called_once_with(replanned, 6.0)
        self.assertEqual(controller.online_map_update_count, 1)
        self.assertEqual(controller.online_replan_count, 1)
        self.assertEqual(controller.active_goal.replan_count, 1)


if __name__ == "__main__":
    unittest.main()
