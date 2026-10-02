"""离线验证速度对照的坐标投影和异常停车；不初始化 DDS。"""

import contextlib
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g1_slam_speed_probe as probe


class SpeedProbeTests(unittest.TestCase):
    def test_speed_specific_time_limits(self):
        for vx, allowed_seconds, rejected_seconds in (("0.3", "3", "3.1"),
                                                      ("0.5", "4", "4.1")):
            with self.subTest(vx=vx):
                allowed_output = io.StringIO()
                with patch.object(sys, "argv", ["probe", "--network-interface", "fake",
                                                "--vx", vx, "--seconds", allowed_seconds]), \
                        contextlib.redirect_stderr(allowed_output):
                    with self.assertRaises(SystemExit) as allowed:
                        probe.main()
                    self.assertEqual(allowed.exception.code, 2)
                    self.assertIn("--execute is required", allowed_output.getvalue())
                rejected_output = io.StringIO()
                with patch.object(sys, "argv", ["probe", "--network-interface", "fake",
                                                "--vx", vx, "--seconds", rejected_seconds]), \
                        contextlib.redirect_stderr(rejected_output):
                    with self.assertRaises(SystemExit) as rejected:
                        probe.main()
                    self.assertEqual(rejected.exception.code, 2)
                    self.assertIn("--seconds must be", rejected_output.getvalue())

    def test_0_4_allows_two_seconds_only(self):
        for seconds, expected_error in (("2", "--execute is required"),
                                        ("2.1", "--seconds must be")):
            output = io.StringIO()
            with patch.object(sys, "argv", ["probe", "--network-interface", "fake",
                                            "--vx", "0.4", "--seconds", seconds]), \
                    contextlib.redirect_stderr(output):
                with self.assertRaises(SystemExit):
                    probe.main()
            self.assertIn(expected_error, output.getvalue())

    def test_forward_projection_uses_initial_heading(self):
        start = probe.MapPose(1.0, 2.0, math.pi / 2, 0.0)
        end = probe.MapPose(0.9, 2.4, math.pi / 2, 1.0)
        forward, left, straight, yaw = probe._displacement(start, end)
        self.assertAlmostEqual(forward, 0.4)
        self.assertAlmostEqual(left, 0.1)
        self.assertAlmostEqual(straight, math.hypot(0.1, 0.4))
        self.assertAlmostEqual(yaw, 0.0)

    def test_measures_command_and_stops_before_settle(self):
        state = SimpleNamespace(now=0.0, x=0.0, moving=False, stops=0)

        class Backend:
            def set_velocity(self, vx, vy, wz):
                self.assert_command = (vx, vy, wz)
                state.moving = True

            def stop(self):
                state.moving = False
                state.stops += 1

        class LowState:
            def require_fresh(self): return 0.0

        class Slam:
            def snapshot(self):
                return ({"x": state.x, "y": 0.0, "q_x": 0.0, "q_y": 0.0,
                         "q_z": 0.0, "q_w": 1.0}, state.now, "")

        def sleep(duration):
            if state.moving:
                state.x += 0.4 * duration
            state.now += duration

        fake_time = SimpleNamespace(monotonic=lambda: state.now, sleep=sleep)
        output = io.StringIO()
        with patch.object(probe, "time", fake_time), contextlib.redirect_stdout(output):
            probe._run_motion(Backend(), LowState(), Slam(), 1.0)
        self.assertEqual(state.stops, 1)
        self.assertAlmostEqual(state.x, 0.4)
        self.assertIn("0.5 m/s × duration=0.500 m", output.getvalue())
        self.assertIn("average forward=+0.400 m/s", output.getvalue())
        self.assertIn("final/command distance ratio=0.80", output.getvalue())

    def test_stale_slam_stops_motion(self):
        state = SimpleNamespace(now=0.0, moving=False, stops=0)

        class Backend:
            def set_velocity(self, *args): state.moving = True
            def stop(self):
                state.moving = False
                state.stops += 1

        class LowState:
            def require_fresh(self): return 0.0

        class Slam:
            def snapshot(self):
                received = 0.0 if state.now > 0.0 else state.now
                return ({"x": 0.0, "y": 0.0, "q_x": 0.0, "q_y": 0.0,
                         "q_z": 0.0, "q_w": 1.0}, received, "")

        def sleep(duration): state.now += duration

        fake_time = SimpleNamespace(monotonic=lambda: state.now, sleep=sleep)
        with patch.object(probe, "time", fake_time), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "SLAM currentPose missing or stale"):
                probe._run_motion(Backend(), LowState(), Slam(), 2.0)
        self.assertEqual(state.stops, 1)
        self.assertFalse(state.moving)

    def test_ctrl_c_during_motion_requests_stop(self):
        state = SimpleNamespace(now=0.0, moving=False, stops=0)

        class Backend:
            def set_velocity(self, *args): state.moving = True
            def stop(self):
                state.moving = False
                state.stops += 1

        class LowState:
            def require_fresh(self): return 0.0

        class Slam:
            def snapshot(self):
                return ({"x": 0.0, "y": 0.0, "q_x": 0.0, "q_y": 0.0,
                         "q_z": 0.0, "q_w": 1.0}, state.now, "")

        def sleep(duration):
            state.now += duration
            if state.moving and state.now >= 0.35:
                raise KeyboardInterrupt

        fake_time = SimpleNamespace(monotonic=lambda: state.now, sleep=sleep)
        with patch.object(probe, "time", fake_time), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                probe._run_motion(Backend(), LowState(), Slam(), 4.0)
        self.assertEqual(state.stops, 1)
        self.assertFalse(state.moving)


if __name__ == "__main__":
    unittest.main()
