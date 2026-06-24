"""Reward helpers for stand-and-smooth fine-tuning."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def stand_still_joint_deviation(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected joint deviation only for near-zero commands."""
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    command_norm = torch.linalg.vector_norm(
        env.command_manager.get_command(command_name), dim=1
    )
    return deviation * (command_norm < command_threshold)
