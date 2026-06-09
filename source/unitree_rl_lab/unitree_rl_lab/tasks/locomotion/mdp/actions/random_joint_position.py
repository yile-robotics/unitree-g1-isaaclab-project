from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class SmoothRandomJointPositionAction(JointAction):
    """Drive selected joints through smooth random position targets without policy inputs."""

    cfg: SmoothRandomJointPositionActionCfg

    def __init__(self, cfg: SmoothRandomJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._export_IO_descriptor = False
        self._default_targets = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._processed_actions = self._default_targets.clone()
        self._target_actions = self._default_targets.clone()
        self._initial_actions = self._default_targets.clone()
        self._elapsed = torch.zeros(self.num_envs, device=self.device)
        self._resample_time = torch.empty(self.num_envs, device=self.device)
        self._trajectory_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._trajectory_duration = torch.ones(self.num_envs, device=self.device)
        self._accel_duration = torch.zeros(self.num_envs, device=self.device)
        self._cruise_duration = torch.zeros(self.num_envs, device=self.device)
        self._decel_duration = torch.zeros(self.num_envs, device=self.device)
        self._peak_progress_velocity = torch.zeros(self.num_envs, device=self.device)
        self._sample_resample_time()

    @property
    def action_dim(self) -> int:
        return 0

    def _sample_resample_time(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        low, high = self.cfg.resampling_time_range
        self._resample_time[env_ids] = low + (high - low) * torch.rand(len(env_ids), device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        del actions
        self._elapsed += self._env.step_dt
        resample_mask = self._elapsed >= self._resample_time

        if resample_mask.any():
            env_ids = resample_mask.nonzero(as_tuple=False).squeeze(-1)
            low, high = self.cfg.position_delta_range
            delta = low + (high - low) * torch.rand(
                len(env_ids), self._num_joints, device=self.device
            )
            targets = self._default_targets[env_ids] + delta
            limits = self._asset.data.joint_pos_limits[env_ids][:, self._joint_ids]
            self._target_actions[env_ids] = torch.clamp(
                targets,
                min=limits[:, :, 0],
                max=limits[:, :, 1],
            )
            self._plan_synchronized_trajectory(env_ids)
            self._elapsed[env_ids] = 0.0
            self._sample_resample_time(env_ids)

        self._advance_trajectory()

    def _plan_synchronized_trajectory(self, env_ids: torch.Tensor) -> None:
        """Plan one synchronized trapezoidal trajectory per environment."""
        self._initial_actions[env_ids] = self._processed_actions[env_ids]
        distances = torch.abs(self._target_actions[env_ids] - self._initial_actions[env_ids])

        num_envs = len(env_ids)
        acceleration = self._sample_range(
            self.cfg.acceleration_range, (num_envs, self._num_joints)
        )
        deceleration = self._sample_range(
            self.cfg.deceleration_range or self.cfg.acceleration_range,
            (num_envs, self._num_joints),
        )
        max_velocity = self._sample_range(
            self.cfg.max_velocity_range, (num_envs, self._num_joints)
        )
        velocity_limits = self._asset.data.joint_vel_limits[env_ids][:, self._joint_ids]
        max_velocity = torch.minimum(max_velocity, velocity_limits)

        accel_distance = 0.5 * max_velocity.square() / acceleration
        decel_distance = 0.5 * max_velocity.square() / deceleration
        triangular = accel_distance + decel_distance > distances

        peak_velocity = torch.sqrt(
            2.0 * distances / (1.0 / acceleration + 1.0 / deceleration + 1e-8)
        )
        triangular_time = peak_velocity / acceleration + peak_velocity / deceleration

        cruise_distance = torch.clamp(distances - accel_distance - decel_distance, min=0.0)
        trapezoidal_time = (
            max_velocity / acceleration
            + cruise_distance / (max_velocity + 1e-8)
            + max_velocity / deceleration
        )
        joint_duration = torch.where(triangular, triangular_time, trapezoidal_time)
        joint_duration = torch.where(distances > 1e-6, joint_duration, torch.zeros_like(joint_duration))

        duration = joint_duration.max(dim=1).values.clamp(min=self._env.step_dt)
        accel_fraction = self.cfg.accel_fraction
        decel_fraction = self.cfg.decel_fraction
        cruise_fraction = 1.0 - accel_fraction - decel_fraction

        self._trajectory_duration[env_ids] = duration
        self._accel_duration[env_ids] = duration * accel_fraction
        self._cruise_duration[env_ids] = duration * cruise_fraction
        self._decel_duration[env_ids] = duration * decel_fraction
        self._peak_progress_velocity[env_ids] = 1.0 / (
            self._cruise_duration[env_ids]
            + 0.5 * self._accel_duration[env_ids]
            + 0.5 * self._decel_duration[env_ids]
        )
        self._trajectory_elapsed[env_ids] = 0.0

    def _sample_range(self, value_range: tuple[float, float], shape: tuple[int, int]) -> torch.Tensor:
        low, high = value_range
        return low + (high - low) * torch.rand(*shape, device=self.device)

    def _advance_trajectory(self) -> None:
        """Advance synchronized acceleration, cruise, and deceleration phases."""
        self._trajectory_elapsed = torch.minimum(
            self._trajectory_elapsed + self._env.step_dt,
            self._trajectory_duration,
        )

        time = self._trajectory_elapsed
        accel_time = self._accel_duration
        cruise_time = self._cruise_duration
        decel_time = self._decel_duration
        peak_velocity = self._peak_progress_velocity

        progress = torch.zeros_like(time)
        in_accel = time <= accel_time
        in_cruise = (time > accel_time) & (time <= accel_time + cruise_time)
        in_decel = time > accel_time + cruise_time

        progress[in_accel] = (
            0.5
            * peak_velocity[in_accel]
            / accel_time[in_accel].clamp(min=1e-8)
            * time[in_accel].square()
        )

        accel_progress = 0.5 * peak_velocity * accel_time
        cruise_elapsed = time - accel_time
        progress[in_cruise] = (
            accel_progress[in_cruise]
            + peak_velocity[in_cruise] * cruise_elapsed[in_cruise]
        )

        decel_elapsed = time - accel_time - cruise_time
        decel_acceleration = peak_velocity / decel_time.clamp(min=1e-8)
        progress[in_decel] = (
            accel_progress[in_decel]
            + peak_velocity[in_decel] * cruise_time[in_decel]
            + peak_velocity[in_decel] * decel_elapsed[in_decel]
            - 0.5 * decel_acceleration[in_decel] * decel_elapsed[in_decel].square()
        )

        progress = progress.clamp(0.0, 1.0).unsqueeze(-1)
        self._processed_actions = self._initial_actions + progress * (
            self._target_actions - self._initial_actions
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._default_targets[env_ids_tensor] = self._asset.data.default_joint_pos[env_ids_tensor][
            :, self._joint_ids
        ]
        self._processed_actions[env_ids_tensor] = self._default_targets[env_ids_tensor]
        self._target_actions[env_ids_tensor] = self._default_targets[env_ids_tensor]
        self._initial_actions[env_ids_tensor] = self._default_targets[env_ids_tensor]
        self._elapsed[env_ids_tensor] = 0.0
        self._trajectory_elapsed[env_ids_tensor] = 0.0
        self._trajectory_duration[env_ids_tensor] = self._env.step_dt
        self._accel_duration[env_ids_tensor] = 0.0
        self._cruise_duration[env_ids_tensor] = self._env.step_dt
        self._decel_duration[env_ids_tensor] = 0.0
        self._peak_progress_velocity[env_ids_tensor] = 0.0
        self._sample_resample_time(env_ids_tensor)


@configclass
class SmoothRandomJointPositionActionCfg(mdp.JointActionCfg):
    """Configuration for smooth environment-generated random joint targets."""

    class_type: type[ActionTerm] = SmoothRandomJointPositionAction
    resampling_time_range: tuple[float, float] = (1.0, 3.0)
    position_delta_range: tuple[float, float] = (-0.1, 0.1)
    acceleration_range: tuple[float, float] = (0.5, 1.5)
    max_velocity_range: tuple[float, float] = (0.2, 0.5)
    deceleration_range: tuple[float, float] | None = None
    accel_fraction: float = 0.25
    decel_fraction: float = 0.25

    def __post_init__(self):
        if self.accel_fraction <= 0.0 or self.decel_fraction <= 0.0:
            raise ValueError("Acceleration and deceleration fractions must be positive.")
        if self.accel_fraction + self.decel_fraction >= 1.0:
            raise ValueError("Acceleration and deceleration fractions must sum to less than one.")
