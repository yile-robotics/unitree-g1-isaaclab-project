from __future__ import annotations

"""控制辅助逻辑。

这个模块不直接跑仿真循环，而是提供 runner 需要的基础工具：
- 写入 IsaacLab 原生命令项 base_velocity。
- 对键盘/path follower 给出的速度命令做范围限制和平滑。
- 在 velocity env 里手写构造 lower-body stand policy 需要的历史观测。
- 管理 stand policy 和 locomotion policy 之间的平滑切换。

最重要的原则：尽量复用 IsaacLab 原生 action manager/command manager，
只在必要处手写 policy 输入输出拼接，避免再造一套动力学控制栈。
"""

import numpy as np
import torch


def tensor_first(value):
    """取第 0 个 env 的数据，并搬到 CPU，主要用于打印/诊断。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "ndim") and value.ndim > 1:
        value = value[0]
    return value


def print_native_stack_diagnostics(env) -> None:
    """打印原生 IsaacLab 关节顺序和 action term 映射。

    迁移 policy 时最容易错的是 joint order。这个诊断会把 IsaacLab 当前机器人
    的 joint_names、default_joint_pos、action term 控制的 joint_ids 全部打出来，
    用来对齐 unitree_rl_lab/MuJoCo deploy 中的顺序。
    """
    raw_env = env.unwrapped
    robot = raw_env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    default_joint_pos = tensor_first(robot.data.default_joint_pos)

    print("[INFO] Native IsaacLab robot joint order:")
    for joint_id, name in enumerate(joint_names):
        print(f"  {joint_id:02d}: {name:28s} default={float(default_joint_pos[joint_id]): .4f}")

    print("[INFO] Native IsaacLab action terms:")
    for term_name in raw_env.action_manager.active_terms:
        term = raw_env.action_manager.get_term(term_name)
        joint_ids = getattr(term, "_joint_ids", None)
        if hasattr(joint_ids, "detach"):
            joint_ids = joint_ids.detach().cpu().tolist()
        elif isinstance(joint_ids, slice):
            joint_ids = list(range(len(joint_names)))
        elif joint_ids is not None:
            joint_ids = list(joint_ids)
        print(f"  {term_name}: joint_ids={joint_ids}")
        if joint_ids is not None:
            for joint_id in joint_ids:
                print(f"    {joint_id:02d}: {joint_names[joint_id]}")




def set_velocity_command(raw_env, command: torch.Tensor) -> None:
    """把外部 SE(2) 速度命令写入 IsaacLab 原生 base_velocity command term。

    command 形状为 [num_envs, 3]，分别是 vx、vy、wz，坐标系和训练环境保持一致。
    """
    command_term = raw_env.command_manager.get_term("base_velocity")
    command_term.vel_command_b[:, :] = command.to(device=raw_env.device, dtype=torch.float32)
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False


def clamp_command(command: torch.Tensor, env_cfg) -> torch.Tensor:
    """把速度命令限制在 policy 训练过的范围内。"""
    ranges = env_cfg.commands.base_velocity.ranges
    command[:, 0].clamp_(float(ranges.lin_vel_x[0]), float(ranges.lin_vel_x[1]))
    command[:, 1].clamp_(float(ranges.lin_vel_y[0]), float(ranges.lin_vel_y[1]))
    command[:, 2].clamp_(float(ranges.ang_vel_z[0]), float(ranges.ang_vel_z[1]))
    return command


def clamp_step_tensor(current: torch.Tensor, target: torch.Tensor, max_delta: torch.Tensor) -> torch.Tensor:
    """对张量做逐元素限速更新，用于速度命令和动作切换的平滑。"""
    return current + torch.clamp(target - current, -max_delta, max_delta)



def get_policy_obs(obs) -> torch.Tensor:
    """兼容 IsaacLab 不同 wrapper 返回格式，取出真正喂给 policy 的 obs tensor。"""
    if isinstance(obs, tuple):
        obs = obs[0]
    if isinstance(obs, dict) or hasattr(obs, "keys"):
        return obs["policy"]
    return obs


def smoothstep(value: float) -> float:
    """0..1 平滑插值曲线，比线性插值少一点切换突变。"""
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def resolve_joint_ids(robot, joint_names: list[str]) -> list[int]:
    """把关节名列表转换成 IsaacLab 当前 robot.data.joint_names 下的整数 id。"""
    name_to_id = {name: index for index, name in enumerate(robot.data.joint_names)}
    return [name_to_id[name] for name in joint_names]


class SwitchCommandController:
    """管理“请求速度”和“实际写入 policy 的滤波速度”。

    requested 是键盘/path follower 想要的目标速度；filtered 是按 ramp_duration
    限速后的速度。这样用户按键或 path target 突变时，policy 看到的命令不会跳变太猛。
    """

    def __init__(self, raw_env, env_cfg, args_cli):
        self.raw_env = raw_env
        self.env_cfg = env_cfg
        self.args_cli = args_cli
        self.requested = torch.zeros((raw_env.num_envs, 3), device=raw_env.device, dtype=torch.float32)
        self.filtered = torch.zeros_like(self.requested)

    def set_initial(self, vx: float, vy: float, wz: float) -> None:
        """设置初始命令，同时让 filtered 直接等于 requested，避免启动时慢慢爬升。"""
        self.set_requested(vx, vy, wz)
        self.filtered[:] = self.requested

    def set_requested(self, vx: float, vy: float, wz: float) -> None:
        """设置目标命令，并立刻 clamp 到训练范围。"""
        self.requested[:, 0] = float(vx)
        self.requested[:, 1] = float(vy)
        self.requested[:, 2] = float(wz)
        clamp_command(self.requested, self.env_cfg)

    def zero(self) -> None:
        """请求速度归零；filtered 会在 update_filtered 中按限速逐渐回零。"""
        self.requested.zero_()

    def apply_key(self, key: str) -> None:
        """处理 switch 模式的离散按键增量。

        每按一次 W/S/A/D/Q/E 只改变一个小步长，和 g1_29dof_goal_tracking 的
        速度命令调节习惯一致。
        """
        if key == "w":
            self.requested[:, 0] += float(self.args_cli.linear_command_step)
        elif key == "s":
            self.requested[:, 0] -= float(self.args_cli.linear_command_step)
        elif key == "a":
            self.requested[:, 1] += float(self.args_cli.linear_command_step)
        elif key == "d":
            self.requested[:, 1] -= float(self.args_cli.linear_command_step)
        elif key == "q":
            self.requested[:, 2] += float(self.args_cli.yaw_command_step)
        elif key == "e":
            self.requested[:, 2] -= float(self.args_cli.yaw_command_step)
        clamp_command(self.requested, self.env_cfg)
        print(
            "[INFO] Requested command: "
            f"vx={float(self.requested[0, 0]):+.2f}, "
            f"vy={float(self.requested[0, 1]):+.2f}, "
            f"wz={float(self.requested[0, 2]):+.2f}"
        )

    def update_filtered(self, dt: float, force_zero: bool = False) -> torch.Tensor:
        """根据 dt 推进一步滤波命令，并返回应该写入 base_velocity 的值。"""
        target = torch.zeros_like(self.requested) if force_zero else self.requested
        max_delta = self._max_delta(dt)
        self.filtered = clamp_step_tensor(self.filtered, target, max_delta)
        clamp_command(self.filtered, self.env_cfg)
        return self.filtered

    def _max_delta(self, dt: float) -> torch.Tensor:
        """根据命令范围和 ramp_duration 计算每个控制周期允许变化的最大量。"""
        ranges = self.env_cfg.commands.base_velocity.ranges
        ramp_duration = max(float(self.args_cli.command_ramp_duration), dt)
        max_delta = torch.tensor(
            [
                float(ranges.lin_vel_x[1] - ranges.lin_vel_x[0]) * dt / ramp_duration,
                float(ranges.lin_vel_y[1] - ranges.lin_vel_y[0]) * dt / ramp_duration,
                float(ranges.ang_vel_z[1] - ranges.ang_vel_z[0]) * dt / ramp_duration,
            ],
            device=self.raw_env.device,
            dtype=torch.float32,
        )
        return max_delta.reshape(1, 3)


class StandObservationHistory:
    """在 velocity env 里构造 lower-body stand policy 的历史观测。

    stand ONNX 是从 stand 任务导出的，但 switch 模式运行在 velocity env 中。
    因此不能直接拿 velocity env 的 policy obs 给 stand policy；这里按 stand 训练时
    的观测项顺序手工拼出历史序列。
    """

    def __init__(self, raw_env, obs_joint_ids: list[int], history_length: int = 5):
        self.raw_env = raw_env
        self.obs_joint_ids = obs_joint_ids
        self.history_length = history_length
        self.num_envs = raw_env.num_envs
        self.device = raw_env.device
        self._history: dict[str, list[torch.Tensor]] = {}

    def reset(self, last_action: torch.Tensor) -> torch.Tensor:
        """初始化历史缓存：第一帧观测复制 history_length 次，避免空历史。"""
        terms = self._make_terms(last_action)
        self._history = {
            name: [value.clone() for _ in range(self.history_length)] for name, value in terms.items()
        }
        return self.as_tensor()

    def append(self, last_action: torch.Tensor) -> torch.Tensor:
        """追加一帧观测，丢掉最旧一帧，保持固定历史长度。"""
        terms = self._make_terms(last_action)
        if not self._history:
            self._history = {
                name: [value.clone() for _ in range(self.history_length)] for name, value in terms.items()
            }
        else:
            for name, value in terms.items():
                self._history[name].pop(0)
                self._history[name].append(value)
        return self.as_tensor()

    def as_tensor(self) -> torch.Tensor:
        """按 stand policy 训练时的 term 顺序拼接成最终 observation tensor。"""
        blocks = []
        for name in ("base_ang_vel", "projected_gravity", "velocity_commands", "leg_joint_pos", "leg_joint_vel", "last_action"):
            blocks.extend(self._history[name])
        return torch.cat(blocks, dim=-1)

    def _make_terms(self, last_action: torch.Tensor) -> dict[str, torch.Tensor]:
        """生成单帧 stand 观测项，并套用训练时的缩放系数。"""
        robot = self.raw_env.scene["robot"]
        data = robot.data
        zero_command = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        return {
            "base_ang_vel": data.root_ang_vel_b * 0.2,
            "projected_gravity": data.projected_gravity_b,
            "velocity_commands": zero_command,
            "leg_joint_pos": data.joint_pos[:, self.obs_joint_ids] - data.default_joint_pos[:, self.obs_joint_ids],
            "leg_joint_vel": data.joint_vel[:, self.obs_joint_ids] * 0.05,
            "last_action": last_action,
        }


class PolicySwitchState:
    """管理 stand/locomotion 两个 policy 的状态机和动作混合。

    locomotion policy 输出 29 维全身动作；stand policy 只输出 12 维腿部动作。
    切到 stand 时，先等待速度命令归零且身体姿态比较安全，再把当前动作平滑混到
    stand 动作，避免正在行走时突然硬切导致摔倒。
    """
    def __init__(self, raw_env, stand_action_joint_ids: list[int], args_cli):
        self.args_cli = args_cli
        self.raw_env = raw_env
        self.stand_action_joint_ids = stand_action_joint_ids
        self.leg_velocity_joint_ids = stand_action_joint_ids
        self.active_mode = "stand"
        self.destination_mode = "stand"
        self.transition_mode = "none"
        self.blend_elapsed = 0.0
        self.settle_elapsed = 0.0
        self.stand_wait_elapsed = 0.0
        self.stand_wait_log_elapsed = 0.0
        self.hold_before_stand_elapsed = 0.0
        self.output_action = torch.zeros((raw_env.num_envs, 29), device=raw_env.device, dtype=torch.float32)
        self.blend_start_action = self.output_action.clone()

    def request_locomotion(self) -> None:
        """请求切到行走 policy；如果已经在切换或已经是行走，则不重复触发。"""
        if self.destination_mode == "locomotion" and self.transition_mode == "blending":
            return
        if self.active_mode == "locomotion" and self.transition_mode == "none":
            print("[INFO] Already in locomotion policy.")
            return
        self._begin_blend("locomotion")

    def request_stand(self) -> None:
        """请求切到站立 policy。

        和切到 locomotion 不同，站立接管前需要等待一个安全窗口：
        命令接近 0、yaw rate 小、身体倾角不大。否则 stand policy 可能在动态姿态下接管失败。
        """
        if self.destination_mode == "stand" and self.transition_mode in ("waiting_for_stand", "holding_before_stand", "blending"):
            return
        if self.active_mode == "stand" and self.transition_mode == "none":
            print("[INFO] Already in stand policy.")
            return
        self.destination_mode = "stand"
        self.transition_mode = "waiting_for_stand"
        self.settle_elapsed = 0.0
        self.stand_wait_elapsed = 0.0
        self.stand_wait_log_elapsed = 0.0
        print("[INFO] Stand requested: keep locomotion with zero command and wait for a takeover window.")

    def force_stand(self) -> None:
        """跳过等待窗口，直接开始平滑切到 stand。只在肉眼确认姿态安全时使用。"""
        self._begin_blend("stand")

    def update_waiting_for_stand(self, command: torch.Tensor, dt: float) -> None:
        """检查 locomotion->stand 的接管窗口。

        这里不是让机器人马上站住，而是先让 locomotion policy 继续执行零速度命令，
        等机器人自然减速、姿态稳定，再冻结当前动作并混到 stand policy。
        """
        if self.transition_mode != "waiting_for_stand":
            return

        robot = self.raw_env.scene["robot"]
        command_norm = float(torch.linalg.norm(command[0]).item())
        leg_joint_velocity = float(torch.max(torch.abs(robot.data.joint_vel[:, self.leg_velocity_joint_ids])).item())
        yaw_rate = abs(float(robot.data.root_ang_vel_b[0, 2].item()))
        tilt_angle = self.current_tilt_angle()
        waited_long_enough = self.stand_wait_elapsed >= 2.0
        command_is_zero = command_norm < 0.03
        yaw_is_slow = yaw_rate < 0.25
        tilt_is_safe = tilt_angle < 0.50

        if waited_long_enough and command_is_zero and yaw_is_slow and tilt_is_safe:
            self.settle_elapsed += dt
            if self.settle_elapsed >= 0.10:
                print("[INFO] Stand takeover window found: freeze current target then blend to stand.")
                self._begin_hold_before_stand()
                return
        else:
            self.settle_elapsed = 0.0

        self.stand_wait_elapsed += dt
        self.stand_wait_log_elapsed += dt
        if self.stand_wait_log_elapsed >= 0.5:
            self.stand_wait_log_elapsed = 0.0
            print(
                "[INFO] Waiting stand window: "
                f"cmd={command_norm:.3f} leg_vel={leg_joint_velocity:.3f} "
                f"yaw={yaw_rate:.3f} tilt={tilt_angle:.3f} wait={self.stand_wait_elapsed:.2f}/2.00"
            )
        if self.stand_wait_elapsed >= 6.0:
            print("[WARN] Still waiting for stand window. Press 3 to force smooth stand if posture looks safe.")
            self.stand_wait_elapsed = 2.0

    def action(self, stand_leg_action: torch.Tensor, locomotion_action: torch.Tensor, dt: float) -> torch.Tensor:
        """根据当前状态机输出最终 29 维动作。

        stand_leg_action 是 12 维腿部动作，需要填回 29 维动作对应关节；
        locomotion_action 已经是 29 维动作。切换时用 smoothstep 做插值。
        """
        stand_action = torch.zeros_like(locomotion_action)
        stand_action[:, self.stand_action_joint_ids] = stand_leg_action

        if self.transition_mode == "blending":
            duration = self.args_cli.stand_blend_duration if self.destination_mode == "stand" else self.args_cli.blend_duration
            self.blend_elapsed += dt
            alpha = smoothstep(self.blend_elapsed / max(duration, dt))
            target = stand_action if self.destination_mode == "stand" else locomotion_action
            self.output_action = (1.0 - alpha) * self.blend_start_action + alpha * target
            if self.blend_elapsed >= duration:
                self.active_mode = self.destination_mode
                self.transition_mode = "none"
                self.output_action = target
                print(f"[INFO] Switch complete: {self.active_mode}.")
        elif self.transition_mode == "holding_before_stand":
            self.hold_before_stand_elapsed += dt
            self.output_action = self.blend_start_action
            if self.hold_before_stand_elapsed >= 0.0:
                self._begin_blend("stand")
        elif self.transition_mode == "waiting_for_stand":
            self.output_action = locomotion_action
        elif self.active_mode == "stand":
            self.output_action = stand_action
        else:
            self.output_action = locomotion_action
        return self.output_action

    def should_zero_command(self) -> bool:
        """判断当前是否应该给 locomotion policy 写零速度命令。"""
        return self.transition_mode in ("waiting_for_stand", "holding_before_stand") or (
            self.transition_mode == "blending" and self.destination_mode == "stand"
        ) or (self.active_mode == "stand" and self.transition_mode == "none")

    def current_tilt_angle(self) -> float:
        """根据 projected_gravity 估计机身偏离竖直方向的角度。"""
        robot = self.raw_env.scene["robot"]
        upright_cosine = max(-1.0, min(1.0, -float(robot.data.projected_gravity_b[0, 2].item())))
        return float(np.arccos(upright_cosine))

    def _begin_hold_before_stand(self) -> None:
        """进入 stand 前冻结当前动作一拍，作为平滑混合的起点。"""
        self.destination_mode = "stand"
        self.transition_mode = "holding_before_stand"
        self.hold_before_stand_elapsed = 0.0
        self.blend_start_action = self.output_action.clone()

    def _begin_blend(self, destination: str) -> None:
        """开始从当前输出动作平滑混合到目标 policy 动作。"""
        self.destination_mode = destination
        self.transition_mode = "blending"
        self.blend_elapsed = 0.0
        self.blend_start_action = self.output_action.clone()
        print(f"[INFO] Begin smooth switch to {destination}.")
