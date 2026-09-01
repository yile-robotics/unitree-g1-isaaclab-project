from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.control_schedule import FixedRateUpdateSchedule  # noqa: E402


class FixedRateUpdateScheduleTest(unittest.TestCase):
    def test_twenty_hz_on_fifty_hz_policy_steps(self):
        schedule = FixedRateUpdateSchedule(20.0, first_dt_s=0.02)
        due = []
        for step in range(11):
            timestamp = step * 0.02
            elapsed = schedule.due_dt(timestamp)
            if elapsed is not None:
                due.append((timestamp, elapsed))

        self.assertEqual(
            [round(timestamp, 2) for timestamp, _ in due],
            [0.00, 0.06, 0.10, 0.16, 0.20],
        )
        self.assertEqual(
            [round(elapsed, 2) for _, elapsed in due],
            [0.02, 0.06, 0.04, 0.06, 0.04],
        )

    def test_rejects_non_monotonic_time(self):
        schedule = FixedRateUpdateSchedule(20.0, first_dt_s=0.02)
        schedule.due_dt(0.1)
        with self.assertRaises(ValueError):
            schedule.due_dt(0.09)


if __name__ == "__main__":
    unittest.main()
