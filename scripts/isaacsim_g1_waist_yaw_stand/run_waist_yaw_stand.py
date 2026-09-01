#!/usr/bin/env python3
from __future__ import annotations

"""运行“12维下半身站立 policy + arm_sdk 风格 WaistYaw”的隔离实验。

这个入口脚本负责组装仿真环境、加载站立 policy、替换腰部 Action Term、按时间切换
用户目标并保存指标。真正的 WaistYaw 限速和位置目标下发在 waist_yaw_action.py 中。
"""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time


# 从当前脚本位置推导各项目目录，避免运行命令必须依赖某个特定工作目录。
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROJECTS_DIR = PROJECT_DIR.parent
UNITREE_RL_LAB_DIR = PROJECTS_DIR / "unitree_rl_lab"
GOAL_TRACKING_DIR = PROJECT_DIR / "scripts" / "isaacsim_goal_tracking"
# 机器人模型是 29DoF，但这个任务使用的 stand policy 只输出 12 个腿部动作。
DEFAULT_TASK = "Unitree-G1-29dof-LowerBody-Stand"
DEFAULT_CHECKPOINT = (
    UNITREE_RL_LAB_DIR
    / "logs/rsl_rl/unitree_g1_29dof_lower_body_stand"
    / "2026-06-13_20-46-55_lower_body_stand_agile_reward_disturbance80_from22000_finetune5k_seed42"
    / "model_26999.pt"
)
DEFAULT_HOUSE_USD = Path(
    "/home/yile/scene/House/scene_047/mujoco/usd/scene_scene_047.usda"
)
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs/isaacsim_g1_waist_yaw_stand"

# 这两个工程目前以源码目录方式使用，还没有作为统一 Python 包安装，因此在启动
# Isaac Sim 前把它们加入模块搜索路径。
sys.path.insert(0, str(UNITREE_RL_LAB_DIR / "source" / "unitree_rl_lab"))
sys.path.insert(0, str(GOAL_TRACKING_DIR))

from isaaclab.app import AppLauncher  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """定义实验命令行参数，并追加 IsaacLab AppLauncher 的通用参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Test a commandable absolute WaistYaw target while the trained "
            "12-DoF lower-body stand policy keeps G1 balanced."
        )
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scene-usd", type=Path, default=DEFAULT_HOUSE_USD)
    # WaistYaw 的稳态目标模式和自动测试角度序列。
    parser.add_argument(
        "--plane",
        action="store_true",
        help="Use the native plane instead of the default room USD.",
    )
    parser.add_argument("--spawn", type=float, nargs=3, default=(2.45, 1.15, 0.8))
    parser.add_argument("--yaw", type=float, default=math.pi)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--disable-fabric", action="store_true", default=False)
    parser.add_argument(
        "--disable-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--disable-auto-reset",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--waist-mode", choices=("policy", "override", "blend"), default="override"
    )
    parser.add_argument("--waist-yaw-target-rad", type=float, default=0.0)
    parser.add_argument(
        "--waist-yaw-sequence-deg",
        default="0,2,0,-2,0,3,0,-3,0,5,0,-5,0",
        help=(
            "Comma-separated automatic target sequence in degrees. Pass an "
            "empty string to use --waist-yaw-target-rad once."
        ),
    )
    parser.add_argument(
        "--waist-yaw-transition-time",
        type=float,
        default=3.0,
        help=(
            "Time reserved for each stage before its hold interval. Actual q_des "
            "motion is bounded by --waist-yaw-max-velocity-rad-s."
        ),
    )
    # transition_time 现在只是每个 stage 的计时窗口；真正的角度变化速度由
    # waist-yaw-max-velocity-rad-s 决定。
    parser.add_argument("--waist-yaw-hold-time", type=float, default=2.0)
    parser.add_argument("--waist-yaw-weight", type=float, default=1.0)
    parser.add_argument(
        "--waist-yaw-max-velocity-rad-s",
        type=float,
        default=0.5,
        help="Unitree arm_sdk-style q_des rate limit; official example uses 0.5rad/s.",
    )
    parser.add_argument(
        "--arm-sdk-release-weight-rate",
        type=float,
        default=0.2,
        help="Simulated arm_sdk release rate; 0.2/s takes about 5s from 1 to 0.",
    )
    parser.add_argument("--waist-yaw-kp", type=float, default=60.0)
    parser.add_argument("--waist-yaw-kd", type=float, default=1.5)
    parser.add_argument(
        "--waist-yaw-max-abs-rad",
        type=float,
        default=math.radians(30.0),
        help="Experiment safety cap applied in addition to the model joint limits.",
    )
    # 稳定性判断和实验运行控制参数。
    parser.add_argument("--warmup-time", type=float, default=3.0)
    parser.add_argument("--fall-height-m", type=float, default=0.45)
    parser.add_argument("--fall-tilt-rad", type=float, default=0.65)
    parser.add_argument("--contact-force-threshold-n", type=float, default=10.0)
    parser.add_argument(
        "--abort-on-fall", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--exit-after-sequence", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument(
        "--real-time", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    AppLauncher.add_app_launcher_args(parser)
    return parser


# AppLauncher 必须在导入大部分 Isaac/Omniverse 模块之前创建，因此参数解析和
# simulation_app 初始化特意放在文件上半部分。
parser = build_parser()
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.num_envs != 1:
    parser.error("This metrics probe currently requires --num-envs 1.")
if not args_cli.checkpoint.exists():
    parser.error(f"Stand checkpoint does not exist: {args_cli.checkpoint}")
if not args_cli.plane and not args_cli.scene_usd.exists():
    parser.error(f"Scene USD does not exist: {args_cli.scene_usd}")
# 对会直接进入控制公式的参数做有限性和正值检查，避免 NaN、Inf 或负数目标进入
# 仿真执行器。
for name in (
    "waist_yaw_transition_time",
    "waist_yaw_max_velocity_rad_s",
    "arm_sdk_release_weight_rate",
    "waist_yaw_kp",
    "waist_yaw_kd",
    "waist_yaw_max_abs_rad",
    "fall_height_m",
    "fall_tilt_rad",
    "contact_force_threshold_n",
):
    if not math.isfinite(getattr(args_cli, name)) or getattr(args_cli, name) <= 0.0:
        parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
if not math.isfinite(args_cli.waist_yaw_hold_time) or args_cli.waist_yaw_hold_time < 0.0:
    parser.error("--waist-yaw-hold-time must be finite and non-negative.")
if not math.isfinite(args_cli.warmup_time) or args_cli.warmup_time < 0.0:
    parser.error("--warmup-time must be finite and non-negative.")
if not 0.0 <= args_cli.waist_yaw_weight <= 1.0:
    parser.error("--waist-yaw-weight must lie in [0, 1].")
if args_cli.print_every <= 0:
    parser.error("--print-every must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# IsaacLab 要求先创建 SimulationApp，再导入下面这些依赖 Omniverse 的模块。
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import unitree_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

from goal_tracking.config import (  # noqa: E402
    configure_scene_probe,
    disable_auto_reset_terms,
    disable_observation_corruption,
)
from goal_tracking.control import print_native_stack_diagnostics  # noqa: E402

from metrics import ExperimentMetrics  # noqa: E402
from waist_yaw_action import (  # noqa: E402
    CommandableWaistAction,
    CommandableWaistActionCfg,
)
from waist_yaw_profile import (  # noqa: E402
    ExperimentSchedule,
    ScheduleStage,
    parse_degree_sequence,
)


def _disable_randomization_everywhere(env_cfg) -> None:
    """按命令行设置关闭训练期随机扰动，使重复实验更容易比较。"""

    if not args_cli.disable_randomization:
        return
    for name in (
        "physics_material",
        "add_base_mass",
        "push_robot",
        "base_external_force_torque",
    ):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)


def _configure_arm_sdk_like_waist_gains(env_cfg) -> None:
    """只在当前内存配置中把 WaistYaw 执行器增益改成 arm_sdk 测试值。"""

    # 遍历资产的 actuator 分组，找到确切负责 waist_yaw_joint 的唯一分组。
    matched = 0
    for actuator_cfg in env_cfg.scene.robot.actuators.values():
        joint_patterns = actuator_cfg.joint_names_expr
        if isinstance(joint_patterns, str):
            joint_patterns = (joint_patterns,)
        if not any("waist_yaw_joint" in pattern for pattern in joint_patterns):
            continue

        # IsaacLab 的 stiffness/damping 既可能是单个数，也可能是按关节名保存的
        # 字典，因此这里分别处理两种配置形式。
        if isinstance(actuator_cfg.stiffness, dict):
            if "waist_yaw_joint" not in actuator_cfg.stiffness:
                raise RuntimeError(
                    "WaistYaw actuator has dictionary gains but no exact "
                    "waist_yaw_joint stiffness entry."
                )
            actuator_cfg.stiffness["waist_yaw_joint"] = float(args_cli.waist_yaw_kp)
        else:
            actuator_cfg.stiffness = float(args_cli.waist_yaw_kp)

        if isinstance(actuator_cfg.damping, dict):
            if "waist_yaw_joint" not in actuator_cfg.damping:
                raise RuntimeError(
                    "WaistYaw actuator has dictionary gains but no exact "
                    "waist_yaw_joint damping entry."
                )
            actuator_cfg.damping["waist_yaw_joint"] = float(args_cli.waist_yaw_kd)
        else:
            actuator_cfg.damping = float(args_cli.waist_yaw_kd)
        matched += 1

    # 没找到意味着配置不兼容；找到多个则可能重复控制。两种情况都应停止实验。
    if matched != 1:
        raise RuntimeError(
            f"Expected exactly one WaistYaw actuator group, found {matched}."
        )


def _jsonable_args() -> dict:
    """把 argparse 参数转换成可写入 config.json 的简单数据类型。"""

    result = {}
    for key, value in vars(args_cli).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None), list, tuple)):
            result[key] = value
        else:
            result[key] = str(value)
    # 额外写入两个容易混淆的事实，方便以后只看输出目录也能理解实验结构。
    result["stand_policy_output_dim"] = 12
    result["stand_policy_controls_waist_yaw"] = False
    return result


def _make_schedule() -> ExperimentSchedule:
    """把命令行中的角度序列构造成按仿真时间推进的实验日程。"""

    if args_cli.waist_yaw_sequence_deg.strip():
        targets = parse_degree_sequence(args_cli.waist_yaw_sequence_deg)
    else:
        targets = (float(args_cli.waist_yaw_target_rad),)
    stages = tuple(
        ScheduleStage(
            target_rad=target,
            transition_s=float(args_cli.waist_yaw_transition_time),
            hold_s=float(args_cli.waist_yaw_hold_time),
        )
        for target in targets
    )
    return ExperimentSchedule(stages)


def run() -> None:
    """创建环境并运行一次完整的 WaistYaw 站立实验。"""

    # 每次运行使用独立目录，防止新实验覆盖旧的 config、CSV 和 summary。
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir = args_cli.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(
        json.dumps(_jsonable_args(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[WAIST-YAW] Output directory: {output_dir}")

    # 读取原 lower-body stand 任务配置。下面所有改动都只作用于这个 Python
    # 对象，不会写回 unitree_rl_lab 的源配置。
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    disable_observation_corruption(env_cfg)
    scene_usd = None if args_cli.plane else args_cli.scene_usd
    configure_scene_probe(env_cfg, scene_usd, args_cli)
    _disable_randomization_everywhere(env_cfg)
    disable_auto_reset_terms(env_cfg, args_cli)
    _configure_arm_sdk_like_waist_gains(env_cfg)

    # 仅替换本次运行内存中的 random_waist 动作项。训练配置和 checkpoint 不变。
    env_cfg.actions.random_waist = CommandableWaistActionCfg(
        max_abs_yaw_rad=float(args_cli.waist_yaw_max_abs_rad),
        max_joint_velocity_rad_s=float(args_cli.waist_yaw_max_velocity_rad_s),
        release_weight_rate_per_s=float(args_cli.arm_sdk_release_weight_rate),
    )

    # policy 的网络结构配置仍来自原任务注册表，权重来自指定 checkpoint。
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    metrics = None
    try:
        # RSL-RL 使用向量环境接口；即使这里只测一个机器人，也需套这一层 wrapper。
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        print_native_stack_diagnostics(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # 这里只进行推理，不训练或更新 checkpoint。
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
        runner.load(str(args_cli.checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        raw_env = env.unwrapped
        step_dt = float(raw_env.step_dt)
        # arm_sdk 参考控制以 50Hz 更新。若环境不是 20ms 一步，直接停止，避免
        # 用户以为正在复现同样的速度限制，实际控制周期却不同。
        if not math.isclose(step_dt, 0.02, rel_tol=0.0, abs_tol=1.0e-6):
            raise RuntimeError(
                "arm_sdk-like experiment requires a 0.02s (50Hz) policy step, "
                f"but this environment reports {step_dt:.6f}s."
            )
        # 从 Action Manager 取回刚替换的腰部项，后续通过它设置目标和读取状态。
        waist_term = raw_env.action_manager.get_term("random_waist")
        if not isinstance(waist_term, CommandableWaistAction):
            raise RuntimeError(
                "Expected random_waist to be CommandableWaistAction, got "
                f"{type(waist_term).__name__}."
            )

        print(
            "[WAIST-YAW] Control mapping: "
            f"Isaac joint id={waist_term.yaw_joint_id}, "
            "stand policy output=12 legs, "
            f"arm_sdk-like rate={args_cli.waist_yaw_max_velocity_rad_s:.3f}rad/s, "
            f"kp={args_cli.waist_yaw_kp:.1f}, kd={args_cli.waist_yaw_kd:.1f}."
        )
        schedule = _make_schedule()
        # 指标记录器每个 policy step 写一行 CSV，同时累计最大倾角、最低高度等摘要。
        metrics = ExperimentMetrics(
            raw_env,
            waist_term,
            output_dir / "metrics.csv",
            contact_force_threshold_n=args_cli.contact_force_threshold_n,
            fall_height_m=args_cli.fall_height_m,
            fall_tilt_rad=args_cli.fall_tilt_rad,
        )

        # 在第一帧取得 policy 观测。某些 IsaacLab wrapper 返回 (obs, extras)，
        # stand policy 只需要第一个元素。
        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        step = 0
        time_s = 0.0
        schedule_started = False
        release_started = False
        print(
            "[WAIST-YAW] Baseline warmup: "
            f"{args_cli.warmup_time:.2f}s, then mode={args_cli.waist_mode}."
        )

        while simulation_app.is_running() and (
            args_cli.max_steps < 0 or step < args_cli.max_steps
        ):
            wall_start = time.time()
            # warmup 期间腰部不接管，只让 stand policy 先把机器人稳定在站立姿态。
            if not schedule_started and time_s >= args_cli.warmup_time:
                stage = schedule.start()
                actual_target = waist_term.set_command(
                    mode=args_cli.waist_mode,
                    target_rad=stage.target_rad,
                    weight=args_cli.waist_yaw_weight,
                )
                schedule_started = True
                print(
                    f"[WAIST-YAW] Stage 0 target={actual_target:+.4f}rad "
                    f"({math.degrees(actual_target):+.1f}deg)."
                )

            # 每个 20ms 周期都重新运行一次 stand policy。腰部 Action Term 会在
            # env.step 内由 IsaacLab Action Manager 同时执行，不需要拼接到这12维动作中。
            with torch.inference_mode():
                actions = policy(obs)
                if actions.shape[-1] != 12:
                    raise RuntimeError(
                        f"Stand policy must output 12 leg actions, got {actions.shape}."
                    )
                obs, _, _, _ = env.step(actions)

            step += 1
            time_s += step_dt
            # env.step 完成后记录实际关节角、最终 q_des、基座姿态和足部状态。
            row = metrics.record(
                step=step,
                time_s=time_s,
                stage_index=schedule.index if schedule_started else -1,
            )

            if step % args_cli.print_every == 0:
                print(
                    "[WAIST-YAW] "
                    f"t={time_s:7.2f}s stage={row['stage_index']:2d} "
                    f"mode={row['mode']} state={row['control_state']} "
                    f"arm_weight={row['arm_sdk_takeover_weight']:.3f} "
                    f"q={row['waist_actual_rad']:+.4f} "
                    f"policy/base={row['waist_policy_baseline_q_des_rad']:+.4f} "
                    f"arm_ref={row['waist_arm_sdk_reference_rad']:+.4f} "
                    f"final={row['waist_final_q_des_rad']:+.4f} "
                    f"err={row['waist_tracking_error_rad']:+.4f} "
                    f"roll={row['base_roll_rad']:+.3f} "
                    f"pitch={row['base_pitch_rad']:+.3f} "
                    f"height={row['base_height_m']:.3f} "
                    f"fallen={row['fallen']}"
                )

            # 达到摔倒阈值时默认立即结束，避免继续向已失稳机器人发送动作。
            if metrics.fallen and args_cli.abort_on_fall:
                print(
                    "[WAIST-YAW ERROR] Fall threshold reached; stopping: "
                    f"{metrics.fall_reason}"
                )
                break

            # 日程按仿真时间推进。切换阶段只更新稳态目标；平滑运动仍由腰部
            # Action Term 的 0.5rad/s 逐周期限速负责。
            if schedule_started and not release_started:
                next_stage = schedule.update(step_dt)
                if next_stage is not None:
                    actual_target = waist_term.set_command(
                        mode=args_cli.waist_mode,
                        target_rad=next_stage.target_rad,
                        weight=args_cli.waist_yaw_weight,
                    )
                    print(
                        f"[WAIST-YAW] Stage {schedule.index} "
                        f"target={actual_target:+.4f}rad "
                        f"({math.degrees(actual_target):+.1f}deg)."
                    )
                elif schedule.completed and args_cli.exit_after_sequence:
                    # 所有目标完成后不直接退出：先回正，再逐渐释放模拟接管权重。
                    waist_term.begin_release()
                    release_started = True
                    print(
                        "[WAIST-YAW] Sequence complete: returning WaistYaw to "
                        "baseline, then releasing simulated arm_sdk weight."
                    )

            # 只有状态机确认权重已经降到 0，才认为本次实验自然完成。
            if release_started and waist_term.release_complete:
                print("[WAIST-YAW] Simulated arm_sdk release complete.")
                break

            # --real-time 让墙钟速度尽量贴近 50Hz；控制公式使用的仍是确定的
            # step_dt，而不是易受系统负载影响的墙钟间隔。
            if args_cli.real_time:
                remaining = step_dt - (time.time() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        # 无论自然结束、用户 Ctrl+C 还是发生异常，都尽量写摘要并关闭文件/环境。
        summary = metrics.summary() if metrics is not None else {
            "rows": 0,
            "fallen": None,
            "fall_reason": "experiment did not reach metrics initialization",
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if metrics is not None:
            metrics.close()
        env.close()
        print(f"[WAIST-YAW] Summary: {summary}")


if __name__ == "__main__":
    try:
        run()
    finally:
        simulation_app.close()
