"""Reward helpers adapted from WBC-AGILE for the independent velocity task."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor


class action_rate_rate_l2(ManagerTermBase):
    """Penalize the second finite difference of policy actions."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_rate = torch.zeros_like(env.action_manager.action)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action_rate = env.action_manager.action - env.action_manager.prev_action
        penalty = torch.sum(torch.square(action_rate - self._previous_rate), dim=1)
        self._previous_rate.copy_(action_rate)
        return penalty

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._previous_rate.zero_()
        else:
            self._previous_rate[env_ids] = 0.0


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-error / std**2)


def flat_body_orientation_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    gravity = asset.data.GRAVITY_VEC_W.unsqueeze(1).expand_as(body_quat[..., :3])
    projected_gravity = math_utils.quat_apply_inverse(body_quat, gravity)
    error = torch.sum(torch.square(projected_gravity[..., :2]), dim=(1, 2))
    return torch.exp(-error / std**2)


def feet_slip(
    env: ManagerBasedRLEnv,
    contact_threshold: float,
    sensor_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = torch.linalg.vector_norm(
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1
    )
    in_contact = torch.max(force_history, dim=1).values > contact_threshold
    horizontal_speed = torch.linalg.vector_norm(
        robot.data.body_lin_vel_w[:, robot_cfg.body_ids, :2], dim=-1
    )
    return torch.sum(horizontal_speed * in_contact, dim=1)


def feet_roll_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    quaternions = asset.data.body_quat_w[:, asset_cfg.body_ids].reshape(-1, 4)
    roll, _, _ = math_utils.euler_xyz_from_quat(quaternions)
    return torch.sum(torch.square(roll.reshape(env.num_envs, -1)), dim=1)


def feet_yaw_diff_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    quaternions = asset.data.body_quat_w[:, asset_cfg.body_ids].reshape(-1, 4)
    _, _, yaw = math_utils.euler_xyz_from_quat(quaternions)
    yaw = yaw.reshape(env.num_envs, -1)
    yaw_difference = math_utils.wrap_to_pi(yaw[:, 1] - yaw[:, 0])
    return torch.square(yaw_difference)


def feet_yaw_mean_vs_base(
    env: ManagerBasedRLEnv,
    feet_asset_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[feet_asset_cfg.name]
    feet_quat = asset.data.body_quat_w[:, feet_asset_cfg.body_ids]
    base_quat = asset.data.body_quat_w[:, base_body_cfg.body_ids[0]]
    relative_quat = math_utils.quat_mul(
        math_utils.quat_inv(base_quat).unsqueeze(1).expand_as(feet_quat), feet_quat
    )
    _, _, relative_yaw = math_utils.euler_xyz_from_quat(relative_quat.reshape(-1, 4))
    return torch.sum(torch.square(relative_yaw.reshape(env.num_envs, -1)), dim=1)


def feet_distance_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ref_distance: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    relative_pos_w = feet_pos_w - asset.data.root_pos_w.unsqueeze(1)
    root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, relative_pos_w.shape[1], -1)
    feet_pos_b = math_utils.quat_apply_inverse(root_quat, relative_pos_w)
    lateral_distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    return torch.abs(lateral_distance - ref_distance)


def jumping(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_forces = torch.linalg.vector_norm(
        sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1
    )
    return torch.all(feet_forces < threshold, dim=1).float()


def body_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    acceleration = asset.data.body_acc_w[:, asset_cfg.body_ids]
    return torch.clamp(torch.sum(torch.square(acceleration), dim=(1, 2)), max=1.0e6)
