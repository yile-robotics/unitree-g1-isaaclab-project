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
    def run_panorama(self, map_scale: float):
        state = SimpleNamespace(now=0.0, yaw=0.0, moving=False, starts=0, stops=0)

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
        with patch.object(probe, "time", fake_time), contextlib.redirect_stdout(output):
            try:
                probe._run_imu75_panorama(Backend(), LowState(), Slam(), quarters=4)
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


if __name__ == "__main__":
    unittest.main()
