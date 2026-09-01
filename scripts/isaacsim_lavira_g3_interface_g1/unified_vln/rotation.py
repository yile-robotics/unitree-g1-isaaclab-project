from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


# yaw 为正表示向左/逆时针旋转；为负表示向右/顺时针旋转。
# 对于正后方，左右转 180° 都能到达，这里沿用 Uni-LaViRA G1 的右转约定。
DIRECTION_YAW_RAD = {
    "forward": 0.0,
    "left": math.pi / 2.0,
    "behind": -math.pi,
    "right": -math.pi / 2.0,
}


class YawProvider(Protocol):
    """转向控制需要的最小 IMU 接口；数据不可用或过期时返回 ``None``。"""

    def get_yaw(self) -> float | None:
        ...


@dataclass(frozen=True)
class RotationCommand:
    """一次控制周期输出的旋转速度和进度信息。"""

    vx: float
    vy: float
    wz: float
    done: bool
    elapsed_s: float
    duration_s: float


class TimedFixedSpeedRotation:
    """固定角速度相对转向：优先使用 IMU yaw，缺失时按时间开环。

    这与 Uni-LaViRA G1 的策略一致：若 ``yaw_provider`` 有新鲜数据，则逐帧累计
    实际 yaw 变化；否则根据 ``速度 × 时间 × duration_scale`` 估计旋转时长。
    Isaac Sim 不传 provider，因此继续使用原有开环行为。
    """

    def __init__(
        self,
        speed_rad_s: float,
        duration_scale: float,
        yaw_provider: YawProvider | None = None,
    ):
        """保存转速与时长补偿系数，并初始化为空闲状态。"""

        if speed_rad_s <= 0.0 or not math.isfinite(speed_rad_s):
            raise ValueError("Rotation speed must be finite and positive.")
        if duration_scale <= 0.0 or not math.isfinite(duration_scale):
            raise ValueError("Rotation duration scale must be finite and positive.")
        self.speed_rad_s = float(speed_rad_s)
        self.duration_scale = float(duration_scale)
        self.yaw_provider = yaw_provider
        self.direction = "forward"
        self.target_yaw_rad = 0.0
        self.duration_s = 0.0
        self.elapsed_s = 0.0
        self.active = False
        self.feedback_initialized = False
        self.feedback_active = False
        self.previous_yaw_rad: float | None = None
        self.accumulated_yaw_rad = 0.0
        self.feedback_timeout_s = 0.0
        self.feedback_timed_out = False

    def start(self, direction: str) -> None:
        """根据观察方向开始一次新的相对转向。前方方向的时长为零。"""

        if direction not in DIRECTION_YAW_RAD:
            raise ValueError(f"Unsupported rotation direction {direction!r}.")
        self.direction = direction
        self.target_yaw_rad = float(DIRECTION_YAW_RAD[direction])
        # 理论时长 = 目标角度 / 角速度，再乘经验补偿系数。
        self.duration_s = (
            abs(self.target_yaw_rad) / self.speed_rad_s * self.duration_scale
        )
        self.elapsed_s = 0.0
        self.active = self.duration_s > 0.0
        self.accumulated_yaw_rad = 0.0
        self.feedback_timeout_s = (
            abs(self.target_yaw_rad) / self.speed_rad_s * 2.0 + 5.0
        )
        self.feedback_timed_out = False
        self.feedback_initialized = False
        self.feedback_active = False
        self.previous_yaw_rad = None

    def update(self, dt: float) -> RotationCommand:
        """推进一个控制周期，返回当前应发送的速度以及是否已经完成。"""

        if dt < 0.0 or not math.isfinite(dt):
            raise ValueError("Rotation dt must be finite and non-negative.")
        if not self.active:
            return RotationCommand(
                0.0, 0.0, 0.0, True, self.elapsed_s, self.duration_s
            )
        if not self.feedback_initialized:
            # 到真正进入 ROTATING、第一次计算命令时才记录起始 yaw，避免把等待
            # locomotion 模式切换期间的身体变化误算成已完成旋转角度。
            self.previous_yaw_rad = self._read_yaw()
            self.feedback_active = self.previous_yaw_rad is not None
            self.feedback_initialized = True
        self.elapsed_s += float(dt)

        sign = 1.0 if self.target_yaw_rad > 0.0 else -1.0
        if self.feedback_active:
            current_yaw = self._read_yaw()
            if current_yaw is None:
                # IMU 中途过期时，从已累计角度开始对“剩余角度”做开环计时。
                self.feedback_active = False
                remaining = max(
                    0.0,
                    abs(self.target_yaw_rad) - max(self.accumulated_yaw_rad, 0.0),
                )
                self.duration_s = (
                    self.elapsed_s
                    + remaining / self.speed_rad_s * self.duration_scale
                )
            else:
                delta = self._wrap_to_pi(current_yaw - self.previous_yaw_rad)
                self.accumulated_yaw_rad += delta * sign
                self.previous_yaw_rad = current_yaw
                if self.accumulated_yaw_rad >= abs(self.target_yaw_rad):
                    self.active = False
                    return RotationCommand(
                        0.0, 0.0, 0.0, True, self.elapsed_s, self.duration_s
                    )
                if self.elapsed_s >= self.feedback_timeout_s:
                    self.active = False
                    self.feedback_timed_out = True
                    return RotationCommand(
                        0.0, 0.0, 0.0, True, self.elapsed_s, self.duration_s
                    )

        if not self.feedback_active and self.elapsed_s >= self.duration_s:
            self.active = False
            return RotationCommand(
                0.0, 0.0, 0.0, True, self.elapsed_s, self.duration_s
            )
        # 目标角为正则左转，为负则右转；线速度始终为零。
        return RotationCommand(
            0.0,
            0.0,
            sign * self.speed_rad_s,
            False,
            self.elapsed_s,
            self.duration_s,
        )

    def _read_yaw(self) -> float | None:
        """读取并验证 yaw；provider 自己负责判断消息是否已经过期。"""

        if self.yaw_provider is None:
            return None
        try:
            yaw = self.yaw_provider.get_yaw()
        except Exception:
            return None
        if yaw is None:
            return None
        yaw = float(yaw)
        return yaw if math.isfinite(yaw) else None

    @staticmethod
    def _wrap_to_pi(value: float) -> float:
        """处理 IMU yaw 从 +π 跳到 -π（或反向）的角度环绕。"""

        return (float(value) + math.pi) % (2.0 * math.pi) - math.pi
