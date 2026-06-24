"""Reward helpers for smooth-clearance fine-tuning."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.locomotion import mdp


def commanded_foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Apply foot-clearance reward only while a locomotion command is active."""
    clearance = mdp.foot_clearance_reward(
        env,
        asset_cfg=asset_cfg,
        target_height=target_height,
        std=std,
        tanh_mult=tanh_mult,
    )
    command_norm = torch.linalg.vector_norm(
        env.command_manager.get_command(command_name), dim=1
    )
    return clearance * (command_norm > command_threshold)
