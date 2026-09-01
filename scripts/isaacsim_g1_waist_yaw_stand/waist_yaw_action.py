from __future__ import annotations

"""在 IsaacLab 中近似复现 Unitree 高层 ``arm_sdk`` 的腰部控制方式。

这个文件只负责三个腰部关节的目标生成和下发。双腿仍由原来的 12 维站立
policy 控制，因此这里不会修改 policy 网络，也不会改动 checkpoint。
"""

import math

import torch
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

try:
    from .waist_yaw_profile import VALID_MODES, blended_goal
except ImportError:  # Direct script execution puts this directory on sys.path.
    from waist_yaw_profile import VALID_MODES, blended_goal


class CommandableWaistAction(JointAction):
    """让腰部 roll/pitch 保持默认姿态，并以 arm_sdk 风格控制 WaistYaw。

    原来的 lower-body stand policy 继续拥有 12 个腿部动作。本 Action Term 的
    ``action_dim`` 为 0，所以不会从 policy 输出张量中占用任何维度；它只是替换
    训练环境里原本自动生成的 ``random_waist`` 动作项。

    WaistYaw 的位置参考值会受到逐周期速度限制，并通过一个显式的接管权重模拟
    arm_sdk 的启用和释放。Isaac Sim 没有 Unitree 真机固件内部的控制权仲裁，
    因此这个权重只能近似真机的外部行为，不能理解为真机固件的内部实现。
    """

    cfg: "CommandableWaistActionCfg"

    def __init__(self, cfg: "CommandableWaistActionCfg", env):
        super().__init__(cfg, env)

        # 这个实验必须且只能找到三个腰部关节。这里主动检查名称，避免正则表达式
        # 意外匹配到腿或手臂关节后，把用户的腰部命令下发到错误的关节。
        expected = {
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        }
        if set(self._joint_names) != expected:
            raise RuntimeError(
                "CommandableWaistAction must resolve exactly the three waist "
                f"joints, got {self._joint_names}."
            )
        self._yaw_local_id = self._joint_names.index("waist_yaw_joint")

        # baseline 是三个腰部关节在资产配置中的默认位置。通常 WaistYaw 的
        # baseline 为 0rad，WaistRoll/WaistPitch 也保持各自默认值。
        self._baseline_targets = self._asset.data.default_joint_pos[
            :, self._joint_ids
        ].clone()

        # _processed_actions 是最终交给 IsaacLab 执行器的三个关节位置目标。
        self._processed_actions = self._baseline_targets.clone()

        # 用户请求值、经过模式混合后的稳态目标，以及逐周期限速后的当前参考值
        # 分开保存，便于日志清楚地区分“想去哪里”和“本周期实际命令到哪里”。
        self._requested_user_target = self._baseline_targets[
            :, self._yaw_local_id
        ].clone()
        self._arm_sdk_goal_target = self._requested_user_target.clone()
        self._arm_sdk_reference = self._requested_user_target.clone()

        # 初始状态不接管腰部：使用 baseline，接管权重为 0。
        self._mode = "policy"
        self._blend_weight = 0.0
        self._takeover_weight = 0.0
        self._control_state = "disabled"

    @property
    def action_dim(self) -> int:
        # 返回 0 表示这个 Action Term 不读取 policy 输出。12 维 policy 张量全部
        # 留给腿部 Action Term，本项会在 Action Manager 内部独立执行。
        return 0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def weight(self) -> float:
        """模拟 arm_sdk 接管权重：0 表示未接管，1 表示完全接管。"""

        return self._takeover_weight

    @property
    def blend_weight(self) -> float:
        """仅供 ``mode=blend`` 使用的“用户目标/baseline”混合比例。"""

        return self._blend_weight

    @property
    def control_state(self) -> str:
        return self._control_state

    @property
    def at_target(self) -> bool:
        # 这里比较的是“限速参考值”和“稳态目标”，不是实际关节角。它用来判断
        # 指令轨迹是否已回到 baseline，从而进入释放控制权阶段。
        error = torch.max(torch.abs(self._arm_sdk_reference - self._arm_sdk_goal_target))
        return bool(error <= self.cfg.target_tolerance_rad)

    @property
    def release_complete(self) -> bool:
        return self._control_state == "disabled" and self._takeover_weight <= 0.0

    @property
    def yaw_joint_id(self) -> int:
        # _yaw_local_id 是 WaistYaw 在本 Action Term 三个关节中的局部索引；
        # 这里把它转换成机器人完整关节数组中的索引，供 joint_pos 等数据使用。
        joint_ids = self._joint_ids
        if isinstance(joint_ids, slice):
            return self._yaw_local_id
        return int(joint_ids[self._yaw_local_id])

    @property
    def baseline_yaw_target(self) -> torch.Tensor:
        return self._baseline_targets[:, self._yaw_local_id]

    @property
    def requested_user_yaw_target(self) -> torch.Tensor:
        return self._requested_user_target

    @property
    def smoothed_user_yaw_target(self) -> torch.Tensor:
        """当前经过逐周期速度限制的 arm_sdk 参考值（保留旧属性名兼容日志）。"""

        return self._arm_sdk_reference

    @property
    def final_yaw_target(self) -> torch.Tensor:
        return self._processed_actions[:, self._yaw_local_id]

    def set_command(
        self,
        *,
        mode: str,
        target_rad: float,
        weight: float,
    ) -> float:
        """启用模拟 arm_sdk，并返回经过关节范围限制后的用户目标。"""

        mode = str(mode).lower()
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown WaistYaw mode {mode!r}; expected {VALID_MODES}.")
        if not math.isfinite(target_rad):
            raise ValueError("WaistYaw target must be finite.")
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("WaistYaw blend weight must lie in [0, 1].")

        # 优先使用 IsaacLab 的软限位；如果当前资产没有软限位，才退回机械关节
        # 限位。之后还会叠加实验参数 max_abs_yaw_rad，取二者更保守的交集。
        limits = getattr(self._asset.data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = self._asset.data.joint_pos_limits
        yaw_limits = limits[0, self.yaw_joint_id]
        safety_limit = abs(float(self.cfg.max_abs_yaw_rad))
        lower = max(float(yaw_limits[0]), -safety_limit)
        upper = min(float(yaw_limits[1]), safety_limit)
        clamped_target = min(max(float(target_rad), lower), upper)

        # mode 决定稳态目标：policy 使用 baseline，override 使用用户目标，
        # blend 则按 weight 在二者之间插值。这里的 weight 不是接管权重。
        baseline = float(self.baseline_yaw_target[0])
        goal_final = blended_goal(mode, baseline, clamped_target, weight)

        # Unitree arm_sdk 示例在接管时从 LowState 里的当前关节角开始生成命令。
        # 仿真里同样从当前实际 WaistYaw 角度开始，避免第一次接管时 q_des 跳变。
        if self._control_state == "disabled":
            current_q = self._asset.data.joint_pos[:, self.yaw_joint_id]
            self._arm_sdk_reference.copy_(current_q)

        self._mode = mode
        self._blend_weight = float(weight)
        # 接收到有效命令后立刻进入 active，并由辅助控制层完全接管 WaistYaw。
        self._takeover_weight = 1.0
        self._control_state = "active"
        self._requested_user_target.fill_(clamped_target)
        self._arm_sdk_goal_target.fill_(goal_final)
        return clamped_target

    def begin_release(self) -> None:
        """开始安全释放：先限速回到 baseline，再把接管权重逐渐降到 0。"""

        if self.release_complete:
            return
        # 此时不能直接把 weight 设为 0，否则若腰部仍在大角度，最终 q_des 会瞬间
        # 跳回 baseline。先保持完全接管并把运动目标改成 baseline。
        self._requested_user_target.copy_(self.baseline_yaw_target)
        self._arm_sdk_goal_target.copy_(self.baseline_yaw_target)
        self._control_state = "returning"

    def process_actions(self, actions: torch.Tensor) -> None:
        # action_dim=0，所以 Action Manager 传进来的 actions 没有需要消费的数据。
        del actions

        # 每个仿真控制周期允许的最大 q_des 变化量：
        # 默认 0.5rad/s × 0.02s = 0.01rad。torch.clamp 同时处理正转和反转。
        max_delta = self.cfg.max_joint_velocity_rad_s * self._env.step_dt
        delta = torch.clamp(
            self._arm_sdk_goal_target - self._arm_sdk_reference,
            min=-max_delta,
            max=max_delta,
        )
        self._arm_sdk_reference += delta

        # returning 阶段只有参考值真正到达 baseline 后才允许释放接管权重。
        if self._control_state == "returning" and self.at_target:
            self._arm_sdk_reference.copy_(self.baseline_yaw_target)
            self._control_state = "releasing"

        if self._control_state == "releasing":
            # 默认每秒下降 0.2，所以从 1 降到 0 大约需要 5 秒。
            self._takeover_weight = max(
                0.0,
                self._takeover_weight
                - self.cfg.release_weight_rate_per_s * self._env.step_dt,
            )
            if self._takeover_weight <= 0.0:
                self._takeover_weight = 0.0
                self._control_state = "disabled"
        elif self._control_state in ("active", "returning"):
            # 正常动作和回正期间都保持完全接管，防止目标位置发生额外混合。
            self._takeover_weight = 1.0
        else:
            self._takeover_weight = 0.0

        # 用接管权重把 baseline 和限速参考值组合成最终 q_des：
        # weight=1 时 final=reference；weight=0 时 final=baseline。
        baseline_yaw = self.baseline_yaw_target
        final_yaw = baseline_yaw + self._takeover_weight * (
            self._arm_sdk_reference - baseline_yaw
        )
        # 先让三个腰部关节全部保持默认位置，再只覆盖 WaistYaw。这样实验不会
        # 意外改变 WaistRoll 或 WaistPitch。
        self._processed_actions[:] = self._baseline_targets
        self._processed_actions[:, self._yaw_local_id] = final_yaw

    def apply_actions(self) -> None:
        # 向 IsaacLab 下发的是绝对关节位置目标 q_des，不是 normalized action、
        # 速度或力矩。仿真执行器再使用 kp/kd 跟踪这些位置目标。
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids=None) -> None:
        # 环境 reset 后清除旧任务的目标和接管状态，避免新一轮仿真继承上一次角度。
        if env_ids is None:
            env_ids = slice(None)
        self._baseline_targets[env_ids] = self._asset.data.default_joint_pos[
            env_ids
        ][:, self._joint_ids]
        self._processed_actions[env_ids] = self._baseline_targets[env_ids]
        baseline_yaw = self._baseline_targets[env_ids, self._yaw_local_id]
        self._requested_user_target[env_ids] = baseline_yaw
        self._arm_sdk_goal_target[env_ids] = baseline_yaw
        self._arm_sdk_reference[env_ids] = baseline_yaw
        self._mode = "policy"
        self._blend_weight = 0.0
        self._takeover_weight = 0.0
        self._control_state = "disabled"


@configclass
class CommandableWaistActionCfg(mdp.JointActionCfg):
    """``CommandableWaistAction`` 的 IsaacLab 配置。"""

    class_type: type[ActionTerm] = CommandableWaistAction
    asset_name: str = "robot"
    joint_names: list[str] = [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ]
    preserve_order: bool = True
    scale: float = 1.0
    offset: float = 0.0
    # WaistYaw 位置参考的最大变化速度；0.5rad/s 对齐参考 arm_sdk 示例。
    max_joint_velocity_rad_s: float = 0.5
    # 释放阶段接管权重每秒减少多少；0.2/s 对应约 5 秒完全释放。
    release_weight_rate_per_s: float = 0.2
    # 判断限速参考已到达目标所用的数值容差。
    target_tolerance_rad: float = 1.0e-5
    # 实验自身的安全角度上限，还会与机器人关节限位取交集。
    max_abs_yaw_rad: float = math.radians(30.0)
