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


if __name__ == "__main__":
    unittest.main()
