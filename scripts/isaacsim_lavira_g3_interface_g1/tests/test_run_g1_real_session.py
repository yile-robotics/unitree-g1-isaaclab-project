from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_g1_real as runner
from unified_vln.odometry import Pose2D


class RealSessionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.map_path = Path(self.temp.name) / "map.json"
        self.map_path.write_text(json.dumps({
            "camera_offset_x_m": 0.1, "camera_offset_y_m": 0.0,
            "camera_offset_z_m": 0.5, "camera_yaw_rad": 0.0,
            "camera_down_tilt_rad": 0.2, "nominal_base_height_m": 0.8,
            "floor_z_world_m": 0.0,
        }))

    def tearDown(self):
        runner._active_episode = None
        runner._active_dds = None
        runner._active_camera = None
        runner._active_odometry = None

    def args(self, *extra):
        return runner.build_parser().parse_args([
            "--instruction", "Go to the door", "--model-url",
            "http://localhost:18765/v1/lavira/decision",
            "--iplanner-url", "http://localhost:8888",
            "--network-interface", "test0", "--camera-factory", "test:create",
            "--camera-config", "unused.json", "--rotation-duration-scale", "1.4",
            *extra,
        ])

    def test_g3_missing_evidence_rejected_before_hardware(self):
        with patch.object(runner, "UnitreeG1DDSBackend") as dds:
            with self.assertRaisesRegex(ValueError, "--odometry-topic"):
                runner.run(self.args())
            dds.assert_not_called()

    def test_map_requires_explicit_real_geometry(self):
        self.map_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "camera_offset"):
            runner._load_map_config(self.map_path)

    def test_odometry_startup_timeout(self):
        odom = SimpleNamespace(get_pose=lambda: None, frame_id="")
        with patch.object(runner.time, "monotonic", side_effect=[0.0, 2.0]):
            with self.assertRaisesRegex(RuntimeError, "fresh SLAM"):
                runner._wait_for_odometry(odom, 1.0)

    def exercise_run(self, *, legacy=False, start_error=False, update_error=False,
                     success=True):
        events = []
        dds, camera, odom, episode = (MagicMock() for _ in range(4))
        odom.frame_id = "odom"
        dds.stop.side_effect = lambda: events.append("stop")
        dds.close.side_effect = lambda: events.append("close")
        dds.high_stand.side_effect = lambda: events.append("high_stand")
        episode.completed = False
        episode.remote_session_active = False
        episode.failure_reason = None
        episode.session_failure_reason = None
        episode.session_success_reason = "stop_confirmed" if success else None
        episode.history = []
        episode.state = "stopped"

        def start():
            events.append("start")
            if start_error:
                raise RuntimeError("start rejected")
            episode.remote_session_active = not legacy

        def update(**kwargs):
            events.append("update")
            if update_error:
                raise RuntimeError("control failed")
            episode.completed = True
            return SimpleNamespace(desired_mode="stand", command=[0, 0, 0])

        episode.start_remote_session.side_effect = start
        episode.update.side_effect = update
        episode.end_remote_session.side_effect = lambda **kw: events.append("end")
        extra = (["--no-g3-session"] if legacy else [
            "--odometry-topic", "/Odometry", "--map-config", str(self.map_path)])
        with ExitStack() as stack:
            for name, value in [("UnitreeG1DDSBackend", dds),
                                ("_load_camera_backend", camera),
                                ("Ros2OdometryProvider", odom),
                                ("IPlannerClient", MagicMock())]:
                stack.enter_context(patch.object(runner, name, return_value=value))
            make_episode = stack.enter_context(patch.object(
                runner, "LocalEndToEndEpisode", return_value=episode))
            stack.enter_context(patch.object(runner.time, "sleep"))
            if start_error or update_error:
                with self.assertRaises(RuntimeError):
                    runner.run(self.args(*extra))
            else:
                self.assertEqual(runner.run(self.args(*extra)), 0)

        kwargs = make_episode.call_args.kwargs
        self.assertEqual(kwargs["model"].send_instruction, legacy)
        if legacy:
            self.assertIsNone(kwargs["session_client"])
            self.assertIsNone(kwargs["exploration_map"])
        else:
            self.assertIsNotNone(kwargs["session_client"])
            self.assertEqual(kwargs["exploration_map"].pose_frame_id, "odom")
            self.assertIs(kwargs["yaw_provider"], kwargs["odometry"])
            self.assertIs(kwargs["odometry"].provider, odom)
        self.assertIsNone(runner._active_episode)
        self.assertIn("close", events)
        if start_error or legacy:
            episode.end_remote_session.assert_not_called()
        else:
            self.assertLess(events.index("close"), events.index("end"))
            self.assertLess(events.index("start"), events.index("high_stand"))
            expected_success = success and not update_error
            # The mock's STOP reason must match the scenario, like the real state machine.
            self.assertEqual(episode.end_remote_session.call_args.kwargs["status"],
                             "SUCCESS" if expected_success else "FAILURE")

    def test_confirmed_stop_lifecycle(self):
        self.exercise_run()

    def test_decision_cap_is_not_semantic_success(self):
        self.exercise_run(success=False)

    def test_start_failure_cleans_hardware_without_navigation(self):
        self.exercise_run(start_error=True, success=False)

    def test_control_exception_ends_session_as_failure(self):
        self.exercise_run(update_error=True, success=False)

    def test_legacy_mode_does_not_create_session(self):
        self.exercise_run(legacy=True)

    def test_slam_source_shares_world_pose_and_yaw_and_rejects_loss(self):
        pose = Pose2D(1234.0, -56.0, 1.2, 0.0)
        provider = MagicMock(frame_id="map")
        provider.get_pose.return_value = pose
        source = runner._SlamPoseSource(provider, "map")
        self.assertEqual(source.get_pose(), pose)
        self.assertEqual(source.get_yaw(), 1.2)
        provider.get_pose.return_value = None
        with self.assertRaisesRegex(RuntimeError, "stale"):
            source.get_pose()
        with self.assertRaisesRegex(RuntimeError, "stale"):
            source.get_yaw()
        provider.get_pose.return_value = pose
        provider.frame_id = "new_map"
        with self.assertRaisesRegex(RuntimeError, "frame_id"):
            source.get_pose()

    def test_sigint_stops_before_ending_session_even_if_http_fails(self):
        events = []
        runner._active_dds = MagicMock()
        runner._active_dds.close.side_effect = lambda: events.append("close")
        episode = MagicMock(remote_session_active=True, completed=False,
                            failure_reason=None, session_failure_reason=None,
                            session_success_reason=None)
        runner._active_episode = episode
        def end(**kwargs):
            events.append("end")
            raise RuntimeError("network unavailable")
        episode.end_remote_session.side_effect = end
        with patch.object(runner.os, "_exit") as exit_mock:
            runner._signal_handler(None, None)
        self.assertEqual(events, ["close", "end"])
        episode.end_remote_session.assert_called_once_with(
            status="FAILURE", reason="runner_terminated")
        exit_mock.assert_called_once_with(0)
        runner._shutdown_active_resources()
        self.assertEqual(episode.end_remote_session.call_count, 1)


if __name__ == "__main__":
    unittest.main()
