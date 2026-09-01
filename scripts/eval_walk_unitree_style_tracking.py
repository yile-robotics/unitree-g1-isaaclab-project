"""Evaluate velocity tracking errors for a Unitree-style walking checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECTS_DIR = PROJECT_DIR.parent

sys.path.insert(0, str(PROJECT_DIR / "source" / "unitree_g1_isaaclab"))
sys.path.insert(0, str(PROJECTS_DIR / "unitree_rl_lab" / "source" / "unitree_rl_lab"))

import unitree_g1_isaaclab.tasks.walk_flat_unitree_style  # noqa: E402,F401

ISAACLAB_DIR = PROJECTS_DIR / "IsaacLab"
RSL_RL_SCRIPT_DIR = ISAACLAB_DIR / "scripts" / "reinforcement_learning" / "rsl_rl"

sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-G1-LockWaist-UnitreeStyle-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--steps", type=int, default=1000, help="Number of policy steps to evaluate.")
parser.add_argument("--warmup_steps", type=int, default=100, help="Initial steps excluded from summary statistics.")
parser.add_argument("--cmd_x", type=float, default=0.6, help="Fixed forward command in m/s.")
parser.add_argument("--cmd_y", type=float, default=0.0, help="Fixed lateral command in m/s.")
parser.add_argument("--cmd_yaw", type=float, default=0.0, help="Fixed yaw command in rad/s.")
parser.add_argument(
    "--stochastic_actions",
    action="store_true",
    default=False,
    help="Sample actions from the policy distribution instead of using the deterministic actor mean.",
)
parser.add_argument(
    "--print_initial_axes_only",
    action="store_true",
    default=False,
    help="Print root-frame axes immediately after environment setup, then exit without running the policy.",
)
parser.add_argument(
    "--debug_reward_alignment",
    action="store_true",
    default=False,
    help="Print velocity quantities using the same yaw-frame formula as track_lin_vel_xy_yaw_frame_exp.",
)
parser.add_argument(
    "--debug_command_obs",
    action="store_true",
    default=False,
    help="Print the velocity command slice from the policy observation history.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import os  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    command_ranges = env_cfg.commands.base_velocity.ranges
    command_ranges.lin_vel_x = (args_cli.cmd_x, args_cli.cmd_x)
    command_ranges.lin_vel_y = (args_cli.cmd_y, args_cli.cmd_y)
    command_ranges.ang_vel_z = (args_cli.cmd_yaw, args_cli.cmd_yaw)
    if hasattr(env_cfg.commands.base_velocity, "limit_ranges"):
        env_cfg.commands.base_velocity.limit_ranges = command_ranges
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.resampling_time_range = (1000000.0, 1000000.0)
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum.terrain_levels = None
        env_cfg.curriculum.lin_vel_cmd_levels = None

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint and (Path(args_cli.checkpoint).is_absolute() or Path(args_cli.checkpoint).exists()):
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    command_name = "base_velocity"
    command_term = base_env.command_manager.get_term(command_name)
    fixed_command = torch.tensor(
        [args_cli.cmd_x, args_cli.cmd_y, args_cli.cmd_yaw],
        device=base_env.device,
        dtype=command_term.vel_command_b.dtype,
    )

    def set_fixed_command():
        command_term.vel_command_b[:] = fixed_command

    x_axis_local = torch.tensor([1.0, 0.0, 0.0], device=base_env.device).repeat(args_cli.num_envs, 1)
    y_axis_local = torch.tensor([0.0, 1.0, 0.0], device=base_env.device).repeat(args_cli.num_envs, 1)

    def policy_obs_tensor(obs_data):
        if isinstance(obs_data, torch.Tensor):
            return obs_data
        if hasattr(obs_data, "get"):
            for key in ("policy", "obs"):
                value = obs_data.get(key, None)
                if isinstance(value, torch.Tensor):
                    return value
        raise TypeError(f"Unsupported observation container for debug slicing: {type(obs_data)}")

    set_fixed_command()
    obs = env.get_observations()
    initial_x_axis_w = quat_apply(robot.data.root_quat_w, x_axis_local)
    initial_y_axis_w = quat_apply(robot.data.root_quat_w, y_axis_local)
    if args_cli.print_initial_axes_only:
        print("\nInitial root-frame axes")
        print(f"Checkpoint: {resume_path}")
        print(f"Command buffer: x={args_cli.cmd_x:.3f}, y={args_cli.cmd_y:.3f}, yaw={args_cli.cmd_yaw:.3f}")
        print(
            "Env 0 root quat wxyz: "
            f"{robot.data.root_quat_w[0, 0].item():.6f}, {robot.data.root_quat_w[0, 1].item():.6f}, "
            f"{robot.data.root_quat_w[0, 2].item():.6f}, {robot.data.root_quat_w[0, 3].item():.6f}"
        )
        print(
            "Env 0 root +X axis in world: "
            f"x={initial_x_axis_w[0, 0].item():.6f}, y={initial_x_axis_w[0, 1].item():.6f}, "
            f"z={initial_x_axis_w[0, 2].item():.6f}"
        )
        print(
            "Env 0 root +Y axis in world: "
            f"x={initial_y_axis_w[0, 0].item():.6f}, y={initial_y_axis_w[0, 1].item():.6f}, "
            f"z={initial_y_axis_w[0, 2].item():.6f}"
        )
        print(
            "Mean root +X axis in world: "
            f"x={initial_x_axis_w[:, 0].mean().item():.6f}, y={initial_x_axis_w[:, 1].mean().item():.6f}, "
            f"z={initial_x_axis_w[:, 2].mean().item():.6f}"
        )
        print(
            "Mean root +Y axis in world: "
            f"x={initial_y_axis_w[:, 0].mean().item():.6f}, y={initial_y_axis_w[:, 1].mean().item():.6f}, "
            f"z={initial_y_axis_w[:, 2].mean().item():.6f}"
        )
        env.close()
        return

    body_errors = []
    yaw_errors = []
    yaw_rate_errors = []
    train_track_rewards = []
    yaw_sq_errors = []
    actual_yaw_rates = []
    actual_body_vels = []
    actual_yaw_vels = []
    actual_world_vels = []
    root_x_axes_w = []
    root_y_axes_w = []
    commands = []
    command_obs_first = None
    command_obs_last = None
    done_count = 0

    with torch.inference_mode():
        for step in range(args_cli.steps):
            set_fixed_command()
            obs = env.get_observations()
            if args_cli.debug_command_obs and command_obs_first is None:
                command_obs_first = policy_obs_tensor(obs)[:, 30:45].detach().clone()
            if args_cli.stochastic_actions:
                actions = runner.alg.policy.act(obs)
            else:
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            set_fixed_command()
            if args_cli.debug_command_obs and step >= args_cli.warmup_steps:
                command_obs_last = policy_obs_tensor(obs)[:, 30:45].detach().clone()
            done_count += int(dones.sum().item())

            command = base_env.command_manager.get_command(command_name)
            world_vel_xy = robot.data.root_lin_vel_w[:, :2]
            body_vel_xy = robot.data.root_lin_vel_b[:, :2]
            yaw_vel_xy = quat_apply_inverse(yaw_quat(robot.data.root_quat_w), robot.data.root_lin_vel_w[:, :3])[:, :2]
            root_x_axis_w = quat_apply(robot.data.root_quat_w, x_axis_local)
            root_y_axis_w = quat_apply(robot.data.root_quat_w, y_axis_local)
            body_error = torch.linalg.norm(command[:, :2] - body_vel_xy, dim=-1)
            yaw_error = torch.linalg.norm(command[:, :2] - yaw_vel_xy, dim=-1)
            yaw_sq_error = torch.sum(torch.square(command[:, :2] - yaw_vel_xy), dim=-1)
            train_track_reward = torch.exp(-yaw_sq_error / 0.25)
            actual_yaw_rate = robot.data.root_ang_vel_b[:, 2]
            yaw_rate_error = torch.abs(command[:, 2] - actual_yaw_rate)

            if step >= args_cli.warmup_steps:
                body_errors.append(body_error.mean())
                yaw_errors.append(yaw_error.mean())
                yaw_rate_errors.append(yaw_rate_error.mean())
                train_track_rewards.append(train_track_reward.mean())
                yaw_sq_errors.append(yaw_sq_error.mean())
                actual_yaw_rates.append(actual_yaw_rate.mean())
                actual_body_vels.append(body_vel_xy.mean(dim=0))
                actual_yaw_vels.append(yaw_vel_xy.mean(dim=0))
                actual_world_vels.append(world_vel_xy.mean(dim=0))
                root_x_axes_w.append(root_x_axis_w.mean(dim=0))
                root_y_axes_w.append(root_y_axis_w.mean(dim=0))
                commands.append(command.mean(dim=0))

    body_error_mean = torch.stack(body_errors).mean().item()
    yaw_error_mean = torch.stack(yaw_errors).mean().item()
    yaw_rate_error_mean = torch.stack(yaw_rate_errors).mean().item()
    actual_yaw_rate_mean = torch.stack(actual_yaw_rates).mean().item()
    train_track_reward_mean = torch.stack(train_track_rewards).mean().item()
    yaw_sq_error_mean = torch.stack(yaw_sq_errors).mean().item()
    actual_body_vel_mean = torch.stack(actual_body_vels).mean(dim=0)
    actual_yaw_vel_mean = torch.stack(actual_yaw_vels).mean(dim=0)
    actual_world_vel_mean = torch.stack(actual_world_vels).mean(dim=0)
    root_x_axis_w_mean = torch.stack(root_x_axes_w).mean(dim=0)
    root_y_axis_w_mean = torch.stack(root_y_axes_w).mean(dim=0)
    command_mean = torch.stack(commands).mean(dim=0)

    print("\nVelocity tracking evaluation")
    print(f"Checkpoint: {resume_path}")
    print(f"Command: x={args_cli.cmd_x:.3f} m/s, y={args_cli.cmd_y:.3f} m/s, yaw={args_cli.cmd_yaw:.3f} rad/s")
    print(f"Action mode: {'stochastic sample' if args_cli.stochastic_actions else 'deterministic mean'}")
    print(f"Steps: {args_cli.steps}, warmup excluded: {args_cli.warmup_steps}, envs: {args_cli.num_envs}")
    print(f"Mean body-frame xy error: {body_error_mean:.4f} m/s")
    print(f"Mean yaw-frame  xy error: {yaw_error_mean:.4f} m/s")
    if args_cli.debug_reward_alignment:
        print(f"Mean yaw-frame squared error: {yaw_sq_error_mean:.4f} (m/s)^2")
        print(f"Mean train track_lin_vel_xy reward estimate: {train_track_reward_mean:.4f}")
    print(f"Mean yaw-rate error:       {yaw_rate_error_mean:.4f} rad/s")
    print(f"Mean actual yaw rate:      {actual_yaw_rate_mean:.4f} rad/s")
    print(
        "Mean command actually used: "
        f"x={command_mean[0].item():.4f}, y={command_mean[1].item():.4f} m/s, yaw={command_mean[2].item():.4f} rad/s"
    )
    if args_cli.debug_command_obs:
        if command_obs_first is not None:
            print(
                "Initial policy velocity command obs mean: "
                f"{command_obs_first.mean(dim=0).detach().cpu().tolist()}"
            )
        if command_obs_last is not None:
            print(
                "Warm policy velocity command obs mean: "
                f"{command_obs_last.mean(dim=0).detach().cpu().tolist()}"
            )
    print(f"Mean actual body velocity: x={actual_body_vel_mean[0].item():.4f}, y={actual_body_vel_mean[1].item():.4f} m/s")
    print(f"Mean actual yaw velocity:  x={actual_yaw_vel_mean[0].item():.4f}, y={actual_yaw_vel_mean[1].item():.4f} m/s")
    print(f"Mean actual world velocity: x={actual_world_vel_mean[0].item():.4f}, y={actual_world_vel_mean[1].item():.4f} m/s")
    print(
        "Mean root +X axis in world: "
        f"x={root_x_axis_w_mean[0].item():.4f}, y={root_x_axis_w_mean[1].item():.4f}, z={root_x_axis_w_mean[2].item():.4f}"
    )
    print(
        "Mean root +Y axis in world: "
        f"x={root_y_axis_w_mean[0].item():.4f}, y={root_y_axis_w_mean[1].item():.4f}, z={root_y_axis_w_mean[2].item():.4f}"
    )
    print(f"Reset/done count during evaluation: {done_count}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
