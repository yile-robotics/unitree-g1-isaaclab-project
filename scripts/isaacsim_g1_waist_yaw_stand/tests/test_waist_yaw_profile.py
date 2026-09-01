from __future__ import annotations

"""不启动 Isaac Sim，直接验证 WaistYaw 的纯数学和日程逻辑。"""

import math
from pathlib import Path
import sys

import pytest

# 被测模块位于 tests 的上一级目录，将它加入搜索路径以支持直接运行 pytest。
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from waist_yaw_profile import (  # noqa: E402
    ExperimentSchedule,
    ScheduleStage,
    blended_goal,
    parse_degree_sequence,
    rate_limited_step,
)


def test_control_modes_use_expected_steady_state_formula() -> None:
    """三种模式应分别得到 baseline、用户目标或二者的线性混合。"""

    assert blended_goal("policy", 0.1, 0.5, 0.75) == pytest.approx(0.1)
    assert blended_goal("override", 0.1, 0.5, 0.25) == pytest.approx(0.5)
    assert blended_goal("blend", 0.1, 0.5, 0.0) == pytest.approx(0.1)
    assert blended_goal("blend", 0.1, 0.5, 0.5) == pytest.approx(0.3)
    assert blended_goal("blend", 0.1, 0.5, 1.0) == pytest.approx(0.5)


def test_arm_sdk_rate_limit_matches_half_rad_per_second_at_50hz() -> None:
    """0.5rad/s 在 50Hz 下每步应为0.01rad，且最后一步不能越过目标。"""

    assert rate_limited_step(0.0, 1.0, 0.5, 0.02) == pytest.approx(0.01)
    assert rate_limited_step(0.0, -1.0, 0.5, 0.02) == pytest.approx(-0.01)
    assert rate_limited_step(0.995, 1.0, 0.5, 0.02) == pytest.approx(1.0)


def test_invalid_weight_is_rejected() -> None:
    """blend 权重只能位于闭区间 [0,1]。"""

    with pytest.raises(ValueError):
        blended_goal("blend", 0.0, 0.1, -0.1)
    with pytest.raises(ValueError):
        blended_goal("blend", 0.0, 0.1, 1.1)


def test_degree_sequence_converts_to_radians() -> None:
    """命令行使用角度制，但控制内部必须转换成弧度制。"""

    result = parse_degree_sequence("0, 5, -10")
    assert result == pytest.approx((0.0, math.radians(5), math.radians(-10)))


def test_schedule_advances_after_transition_and_hold() -> None:
    """每个阶段经过 transition+hold 后才切换到下一个目标。"""

    schedule = ExperimentSchedule(
        (
            ScheduleStage(0.1, transition_s=1.0, hold_s=0.5),
            ScheduleStage(-0.1, transition_s=2.0, hold_s=0.0),
        )
    )
    assert schedule.start().target_rad == pytest.approx(0.1)
    assert schedule.update(1.0) is None
    second = schedule.update(0.5)
    assert second is not None
    assert second.target_rad == pytest.approx(-0.1)
    assert schedule.update(2.0) is None
    assert schedule.completed
