# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate velocity-command tracking for an RSL-RL checkpoint."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate velocity tracking for an RSL-RL checkpoint.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity", help="Name of the task.")
parser.add_argument("--steps", type=int, default=2000, help="Number of policy steps to evaluate.")
parser.add_argument("--warmup_steps", type=int, default=100, help="Initial steps excluded from reported metrics.")
parser.add_argument("--output", type=str, default=None, help="Output prefix for .csv and .png files.")
parser.add_argument("--command_name", type=str, default="base_velocity", help="Velocity command term name.")
parser.add_argument(
    "--fixed_cmd",
    type=float,
    nargs=3,
    metavar=("VX", "VY", "WZ"),
    default=None,
    help="Override velocity command with fixed base-frame [vx vy wz].",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    if not args_cli.checkpoint:
        raise ValueError("Please pass --checkpoint /path/to/model_xxx.pt")

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    output_prefix = args_cli.output
    if output_prefix is None:
        checkpoint_name = os.path.splitext(os.path.basename(resume_path))[0]
        output_prefix = os.path.join(log_dir, f"tracking_eval_{checkpoint_name}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    rows = []
    reward_rows = []
    dt = env.unwrapped.step_dt

    with torch.inference_mode():
        for step in range(args_cli.steps):
            if args_cli.fixed_cmd is not None:
                command_term = env.unwrapped.command_manager.get_term(args_cli.command_name)
                command_term.vel_command_b[:] = torch.tensor(
                    args_cli.fixed_cmd, device=env.unwrapped.device, dtype=command_term.vel_command_b.dtype
                )
                if hasattr(command_term, "is_standing_env"):
                    command_term.is_standing_env[:] = False

            actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)

            if args_cli.fixed_cmd is not None:
                command_term = env.unwrapped.command_manager.get_term(args_cli.command_name)
                command_term.vel_command_b[:] = torch.tensor(
                    args_cli.fixed_cmd, device=env.unwrapped.device, dtype=command_term.vel_command_b.dtype
                )
                if hasattr(command_term, "is_standing_env"):
                    command_term.is_standing_env[:] = False

            command = env.unwrapped.command_manager.get_command(args_cli.command_name)
            robot = env.unwrapped.scene["robot"]
            actual = torch.stack(
                (
                    robot.data.root_lin_vel_b[:, 0],
                    robot.data.root_lin_vel_b[:, 1],
                    robot.data.root_ang_vel_b[:, 2],
                ),
                dim=1,
            )
            error = actual - command

            rows.append(
                {
                    "step": step,
                    "time_s": step * dt,
                    "cmd_vx": float(command[:, 0].mean()),
                    "cmd_vy": float(command[:, 1].mean()),
                    "cmd_wz": float(command[:, 2].mean()),
                    "actual_vx": float(actual[:, 0].mean()),
                    "actual_vy": float(actual[:, 1].mean()),
                    "actual_wz": float(actual[:, 2].mean()),
                    "abs_err_vx": float(error[:, 0].abs().mean()),
                    "abs_err_vy": float(error[:, 1].abs().mean()),
                    "abs_err_wz": float(error[:, 2].abs().mean()),
                    "rmse_vx": float(torch.sqrt(torch.mean(error[:, 0] ** 2))),
                    "rmse_vy": float(torch.sqrt(torch.mean(error[:, 1] ** 2))),
                    "rmse_wz": float(torch.sqrt(torch.mean(error[:, 2] ** 2))),
                    "mean_reward": float(rewards.mean()),
                    "done_fraction": float(dones.float().mean()),
                }
            )
            reward_rows.append(_as_numpy(rewards))

    env.close()

    csv_path = f"{output_prefix}.csv"
    png_path = f"{output_prefix}.png"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    data = {key: np.array([row[key] for row in rows], dtype=np.float64) for key in rows[0].keys()}
    mask = data["step"] >= args_cli.warmup_steps

    print("\nTracking metrics after warmup:")
    for name in ("vx", "vy", "wz"):
        mae = data[f"abs_err_{name}"][mask].mean()
        rmse = data[f"rmse_{name}"][mask].mean()
        print(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    print(f"  mean_reward={data['mean_reward'][mask].mean():.4f}")
    print(f"  done_fraction={data['done_fraction'][mask].mean():.4f}")
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")

    fig, axs = plt.subplots(4, 1, figsize=(13, 10), sharex=True, constrained_layout=True)
    fig.suptitle(f"Velocity tracking: {os.path.basename(resume_path)}")

    for ax, name, ylabel in zip(
        axs[:3],
        ("vx", "vy", "wz"),
        ("linear x [m/s]", "linear y [m/s]", "yaw rate [rad/s]"),
    ):
        ax.plot(data["time_s"], data[f"cmd_{name}"], label=f"cmd_{name}", linewidth=1.4)
        ax.plot(data["time_s"], data[f"actual_{name}"], label=f"actual_{name}", linewidth=1.2)
        ax.plot(data["time_s"], data[f"abs_err_{name}"], label=f"abs_err_{name}", linewidth=0.9, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncols=3, fontsize=8)

    axs[3].plot(data["time_s"], data["mean_reward"], label="mean_reward", linewidth=1.2)
    axs[3].plot(data["time_s"], data["done_fraction"], label="done_fraction", linewidth=1.0)
    axs[3].set_xlabel("time [s]")
    axs[3].set_ylabel("reward / done")
    axs[3].grid(True, alpha=0.25)
    axs[3].legend(loc="upper right", fontsize=8)

    fig.savefig(png_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
    simulation_app.close()
