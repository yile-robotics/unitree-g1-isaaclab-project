"""离线验证四段全景旋转的停车和分段行为；不连接真机。"""

import contextlib
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g1_vln_rotation_probe as probe


class PanoramaRotationTests(unittest.TestCase):
    def run_panorama(self, map_scale: float, capture_failure=None, real_capture=False):
        state = SimpleNamespace(now=0.0, yaw=0.0, moving=False, starts=0, stops=0)
        state.captures = []

        class Backend:
            def set_velocity(self, vx, vy, wz):
                assert (vx, vy, wz) == (0.0, 0.0, 0.8)
                state.moving = True
                state.starts += 1

            def stop(self):
                state.moving = False
                state.stops += 1

        class LowState:
            def require_fresh(self):
                return state.yaw

        class Slam:
            def snapshot(self):
                yaw = state.yaw * map_scale
                return ({"q_x": 0.0, "q_y": 0.0,
                         "q_z": math.sin(yaw / 2), "q_w": math.cos(yaw / 2)},
                        state.now, "")

        def sleep(duration):
            if state.moving:
                state.yaw += math.radians(30.0) * duration
            state.now += duration

        fake_time = SimpleNamespace(monotonic=lambda: state.now, sleep=sleep)
        output = io.StringIO()
        def capture(direction, quarter):
            self.assertFalse(state.moving, "必须停车后才能取图")
            state.captures.append((direction, quarter))
            if direction == capture_failure:
                raise RuntimeError("camera disconnected")

        with patch.object(probe, "time", fake_time), contextlib.redirect_stdout(output):
            try:
                probe._run_imu75_panorama(Backend(), LowState(), Slam(), quarters=4,
                                         capture=capture if real_capture else None)
                error = None
            except RuntimeError as exc:
                error = str(exc)
        return state, output.getvalue(), error

    def test_four_quarters_stop_and_capture_in_order(self):
        state, output, error = self.run_panorama(1.2)
        self.assertIsNone(error)
        self.assertEqual((state.starts, state.stops), (4, 4))
        for direction in ("left", "behind", "right", "forward"):
            self.assertIn(f"simulated capture: {direction}", output)
        self.assertIn("sum of SLAM quarter turns=+360.0°", output)

    def test_bad_map_turn_stops_before_second_quarter(self):
        state, output, error = self.run_panorama(0.5)
        self.assertIn("outside 70°–110°", error)
        self.assertEqual((state.starts, state.stops), (1, 1))
        self.assertNotIn("simulated capture: left", output)

    def test_real_capture_runs_only_after_stop_in_five_stage_order(self):
        state, output, error = self.run_panorama(1.2, real_capture=True)
        self.assertIsNone(error)
        self.assertEqual(state.captures, [("forward", 0), ("left", 1), ("behind", 2),
                                          ("right", 3), ("forward_return", 4)])
        self.assertEqual((state.starts, state.stops), (4, 5))
        self.assertNotIn("simulated capture", output)

    def test_camera_failure_after_first_turn_prevents_next_motion(self):
        state, _, error = self.run_panorama(1.2, capture_failure="left", real_capture=True)
        self.assertEqual(error, "camera disconnected")
        self.assertFalse(state.moving)
        self.assertEqual(state.starts, 1)

    def test_initial_camera_failure_prevents_all_rotation(self):
        state, _, error = self.run_panorama(1.2, capture_failure="forward", real_capture=True)
        self.assertEqual(error, "camera disconnected")
        self.assertEqual(state.starts, 0)

    def test_ctrl_c_stops_dds_before_closing_capture(self):
        calls = []
        backend = SimpleNamespace(stop=lambda: calls.append("stop"),
                                  close=lambda: calls.append("dds_close"))
        recorder = SimpleNamespace(close=lambda: calls.append("camera_close"))
        with patch.multiple(probe, _active_dds=backend, _active_capture=recorder,
                            _active_subscriber=None, _active_slam_subscriber=None,
                            _active_odom=None, _cleaned_up=False), \
                patch.object(probe.os, "_exit", side_effect=SystemExit), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                probe._on_sigint(None, None)
        self.assertEqual(calls, ["stop", "dds_close", "camera_close"])


if __name__ == "__main__":
    unittest.main()
