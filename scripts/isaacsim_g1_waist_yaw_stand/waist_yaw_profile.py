from __future__ import annotations

"""WaistYaw 实验中与仿真器无关的纯控制数学。

这个模块有意不导入 Isaac Sim，使目标混合、限速和动作序列逻辑可以直接使用普通
pytest 测试，不需要启动耗时的仿真程序。
"""

from dataclasses import dataclass
import math


VALID_MODES = ("policy", "override", "blend")


def rate_limited_step(
    current: float,
    target: float,
    max_velocity_rad_s: float,
    dt: float,
) -> float:
    """向 ``target`` 前进一步，但步长不超过 ``最大速度 × dt``。

    这对应 Unitree arm_sdk 示例的关节位置更新方法：发送的仍是位置目标，
    只是每个 20ms 控制周期内不允许位置目标变化得过快。
    """

    if not all(math.isfinite(value) for value in (current, target, max_velocity_rad_s, dt)):
        raise ValueError("Rate-limited WaistYaw inputs must be finite.")
    if max_velocity_rad_s <= 0.0 or dt <= 0.0:
        raise ValueError("WaistYaw maximum velocity and dt must be positive.")
    # 例如 0.5rad/s、dt=0.02s 时，本周期最多移动 0.01rad。
    max_delta = max_velocity_rad_s * dt
    error = target - current
    # 把误差夹到 [-max_delta, +max_delta]，因此正转和反转使用同一套公式；
    # 若剩余误差小于 max_delta，则会直接精确落在目标上而不会越过目标。
    return current + min(max(error, -max_delta), max_delta)


def blended_goal(
    mode: str,
    baseline_q_des_rad: float,
    user_q_des_rad: float,
    weight: float,
) -> float:
    """根据控制模式计算 WaistYaw 最终希望到达的稳态目标。"""

    mode = str(mode).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown WaistYaw mode {mode!r}; expected {VALID_MODES}.")
    if not all(
        math.isfinite(value)
        for value in (baseline_q_des_rad, user_q_des_rad, weight)
    ):
        raise ValueError("WaistYaw targets and weight must be finite.")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("WaistYaw blend weight must lie in [0, 1].")
    if mode == "policy":
        # 当前 stand policy 不输出腰部动作，所以这里实际代表“保持默认位置”。
        return float(baseline_q_des_rad)
    if mode == "override":
        # 完全使用用户给出的目标角度。
        return float(user_q_des_rad)
    # blend 模式：weight=0 完全使用 baseline，weight=1 完全使用用户目标。
    return float(
        (1.0 - weight) * baseline_q_des_rad + weight * user_q_des_rad
    )


def parse_degree_sequence(value: str) -> tuple[float, ...]:
    """解析逗号分隔的角度序列，并把 degree 转成内部使用的 radian。"""

    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("WaistYaw sequence must contain at least one angle.")
    result = tuple(math.radians(float(piece)) for piece in pieces)
    if not all(math.isfinite(angle) for angle in result):
        raise ValueError("WaistYaw sequence angles must be finite.")
    return result


@dataclass(frozen=True)
class ScheduleStage:
    """一个腰部目标阶段：预留运动时间，然后继续保持一段时间。"""

    target_rad: float
    transition_s: float
    hold_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_rad):
            raise ValueError("Schedule target must be finite.")
        if not math.isfinite(self.transition_s) or self.transition_s <= 0.0:
            raise ValueError("Schedule transition time must be finite and positive.")
        if not math.isfinite(self.hold_s) or self.hold_s < 0.0:
            raise ValueError("Schedule hold time must be finite and non-negative.")


class ExperimentSchedule:
    """根据累计的仿真时间，在一组固定目标阶段之间依次前进。"""

    def __init__(self, stages: tuple[ScheduleStage, ...]):
        if not stages:
            raise ValueError("Experiment schedule must contain at least one stage.")
        self.stages = stages
        self.index = -1
        self.stage_elapsed_s = 0.0

    @property
    def active(self) -> bool:
        return 0 <= self.index < len(self.stages)

    @property
    def completed(self) -> bool:
        return self.index >= len(self.stages)

    @property
    def current(self) -> ScheduleStage | None:
        return self.stages[self.index] if self.active else None

    def start(self) -> ScheduleStage:
        # 序列只能启动一次；返回第一阶段，让调用方立即下发第一个目标。
        if self.index != -1:
            raise RuntimeError("Experiment schedule has already started.")
        self.index = 0
        self.stage_elapsed_s = 0.0
        return self.stages[0]

    def update(self, dt: float) -> ScheduleStage | None:
        """累计一个控制周期；切换时返回新阶段，否则返回 ``None``。

        注意：transition_s 只是阶段计时，不直接生成插值轨迹。真正的 WaistYaw
        运动速度由 CommandableWaistAction 的逐周期速度限制决定。
        """

        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("Schedule dt must be finite and positive.")
        if not self.active:
            return None
        self.stage_elapsed_s += dt
        stage = self.stages[self.index]
        # 当前阶段总停留时间 = 为运动预留的时间 + 到达后的保持时间。
        if self.stage_elapsed_s < stage.transition_s + stage.hold_s:
            return None
        self.index += 1
        self.stage_elapsed_s = 0.0
        if self.completed:
            return None
        return self.stages[self.index]
