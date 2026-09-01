from __future__ import annotations

"""Small timing helpers shared by the Isaac runner and unit tests."""

import math


class FixedRateUpdateSchedule:
    """Schedule a slower controller on top of a fixed-step policy loop.

    Isaac's locomotion policy must remain at its trained 50 Hz rate.  Uni-LaViRA's
    trajectory controller, however, updates at roughly 20 Hz.  Since 50/20 is not
    an integer, deadlines land on alternating 40/60 ms policy boundaries.  This
    class preserves that phase instead of resetting an accumulator and accidentally
    producing 16.7 Hz (one update every three 20 ms policy steps).
    """

    def __init__(self, rate_hz: float, *, first_dt_s: float):
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("Update rate must be finite and positive.")
        if not math.isfinite(first_dt_s) or first_dt_s <= 0.0:
            raise ValueError("First update dt must be finite and positive.")
        self.period_s = 1.0 / float(rate_hz)
        self.first_dt_s = float(first_dt_s)
        self.next_deadline_s = 0.0
        self.last_update_time_s: float | None = None

    def due_dt(self, timestamp_s: float) -> float | None:
        """Return elapsed controller time when an update is due, otherwise None."""

        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s) or timestamp_s < 0.0:
            raise ValueError("Schedule timestamp must be finite and non-negative.")
        if (
            self.last_update_time_s is not None
            and timestamp_s + 1e-12 < self.last_update_time_s
        ):
            raise ValueError("Schedule timestamps must be monotonic.")
        if timestamp_s + 1e-12 < self.next_deadline_s:
            return None

        if self.last_update_time_s is None:
            elapsed_s = self.first_dt_s
        else:
            elapsed_s = timestamp_s - self.last_update_time_s
            if elapsed_s <= 0.0:
                return None
        self.last_update_time_s = timestamp_s

        # Preserve the ideal 20 Hz phase even when a deadline falls between two
        # 50 Hz policy frames.  This gives 60/40 ms intervals with a 50 ms mean.
        while self.next_deadline_s <= timestamp_s + 1e-12:
            self.next_deadline_s += self.period_s
        return elapsed_s
