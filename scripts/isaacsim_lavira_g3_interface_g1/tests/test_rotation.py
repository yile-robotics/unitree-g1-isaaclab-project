from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.rotation import TimedFixedSpeedRotation  # noqa: E402


class TimedRotationTest(unittest.TestCase):
    def test_left_uses_positive_fixed_speed_and_expected_duration(self):
        rotation = TimedFixedSpeedRotation(0.4, 1.0)
        rotation.start("left")
        self.assertAlmostEqual(rotation.duration_s, (math.pi / 2.0) / 0.4)
        command = rotation.update(0.1)
        self.assertFalse(command.done)
        self.assertEqual(command.wz, 0.4)

    def test_right_and_behind_use_negative_yaw(self):
        for direction in ("right", "behind"):
            rotation = TimedFixedSpeedRotation(0.4, 1.0)
            rotation.start(direction)
            self.assertLess(rotation.update(0.01).wz, 0.0)

    def test_forward_finishes_without_command(self):
        rotation = TimedFixedSpeedRotation(0.4, 1.0)
        rotation.start("forward")
        command = rotation.update(0.1)
        self.assertTrue(command.done)
        self.assertEqual(command.wz, 0.0)

    def test_fresh_imu_yaw_closes_left_rotation_across_pi_wrap(self):
        class MutableYaw:
            value = 3.10

            def get_yaw(self):
                return self.value

        yaw = MutableYaw()
        rotation = TimedFixedSpeedRotation(0.4, 1.4, yaw_provider=yaw)
        rotation.start("left")
        rotation.update(0.0)
        self.assertTrue(rotation.feedback_active)

        yaw.value = -3.10
        command = rotation.update(0.1)
        self.assertFalse(command.done)
        self.assertGreater(rotation.accumulated_yaw_rad, 0.0)

        yaw.value = -1.40
        command = rotation.update(0.1)
        self.assertTrue(command.done)
        self.assertEqual(command.wz, 0.0)

    def test_fresh_imu_prevents_open_loop_time_from_finishing_early(self):
        class FixedYaw:
            def get_yaw(self):
                return 0.0

        rotation = TimedFixedSpeedRotation(1.0, 1.0, yaw_provider=FixedYaw())
        rotation.start("left")
        command = rotation.update(2.0)
        self.assertFalse(command.done)
        self.assertTrue(rotation.feedback_active)

    def test_stale_imu_falls_back_to_remaining_open_loop_rotation(self):
        class OneShotYaw:
            def __init__(self):
                self.calls = 0

            def get_yaw(self):
                self.calls += 1
                return 0.0 if self.calls == 1 else None

        rotation = TimedFixedSpeedRotation(1.0, 1.4, yaw_provider=OneShotYaw())
        rotation.start("left")
        command = rotation.update(0.1)
        self.assertFalse(command.done)
        self.assertFalse(rotation.feedback_active)
        self.assertGreater(rotation.duration_s, 0.1)


if __name__ == "__main__":
    unittest.main()
