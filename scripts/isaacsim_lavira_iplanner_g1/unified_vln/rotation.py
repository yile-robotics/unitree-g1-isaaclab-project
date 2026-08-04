from __future__ import annotations

from dataclasses import dataclass
import math


# Positive yaw is left/CCW.  Behind follows Uni-LaViRA G1's right-turn choice.
DIRECTION_YAW_RAD = {
    "forward": 0.0,
    "left": math.pi / 2.0,
    "behind": -math.pi,
    "right": -math.pi / 2.0,
}


@dataclass(frozen=True)
class RotationCommand:
    vx: float
    vy: float
    wz: float
    done: bool
    elapsed_s: float
    duration_s: float


class TimedFixedSpeedRotation:
    """Open-loop relative rotation using only fixed speed and elapsed control time."""

    def __init__(self, speed_rad_s: float, duration_scale: float):
        if speed_rad_s <= 0.0 or not math.isfinite(speed_rad_s):
            raise ValueError("Rotation speed must be finite and positive.")
        if duration_scale <= 0.0 or not math.isfinite(duration_scale):
            raise ValueError("Rotation duration scale must be finite and positive.")
        self.speed_rad_s = float(speed_rad_s)
        self.duration_scale = float(duration_scale)
        self.direction = "forward"
        self.target_yaw_rad = 0.0
        self.duration_s = 0.0
        self.elapsed_s = 0.0
        self.active = False

    def start(self, direction: str) -> None:
        if direction not in DIRECTION_YAW_RAD:
            raise ValueError(f"Unsupported rotation direction {direction!r}.")
        self.direction = direction
        self.target_yaw_rad = float(DIRECTION_YAW_RAD[direction])
        self.duration_s = (
            abs(self.target_yaw_rad) / self.speed_rad_s * self.duration_scale
        )
        self.elapsed_s = 0.0
        self.active = self.duration_s > 0.0

    def update(self, dt: float) -> RotationCommand:
        if dt < 0.0 or not math.isfinite(dt):
            raise ValueError("Rotation dt must be finite and non-negative.")
        if not self.active:
            return RotationCommand(
                0.0, 0.0, 0.0, True, self.elapsed_s, self.duration_s
            )
        self.elapsed_s += float(dt)
        if self.elapsed_s >= self.duration_s:
            self.active = False
            return RotationCommand(
                0.0, 0.0, 0.0, True, self.elapsed_s, self.duration_s
            )
        sign = 1.0 if self.target_yaw_rad > 0.0 else -1.0
        return RotationCommand(
            0.0,
            0.0,
            sign * self.speed_rad_s,
            False,
            self.elapsed_s,
            self.duration_s,
        )
