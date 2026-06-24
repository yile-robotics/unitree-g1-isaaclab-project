"""Smooth, higher-clearance fine-tuning task for the AGILE-reward policy."""

from importlib import import_module

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import rewards as smooth_rewards


_base_cfg = import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.agile_reward_velocity.velocity_env_cfg"
)


@configclass
class SmoothClearanceRewardsCfg(_base_cfg.AgileRewardsCfg):
    """AGILE rewards with commanded foot clearance and stronger action smoothing."""

    feet_clearance = RewTerm(
        func=smooth_rewards.commanded_foot_clearance,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*ankle_roll_link"]
            ),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.5)
    action_rate_rate = RewTerm(
        func=_base_cfg.agile_rewards.action_rate_rate_l2,
        weight=-0.05,
    )


@configclass
class SmoothClearanceVelocityEnvCfg(_base_cfg.AgileRewardVelocityEnvCfg):
    """Reward-only fine-tuning environment based on the trained AGILE policy."""

    rewards: SmoothClearanceRewardsCfg = SmoothClearanceRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class SmoothClearanceVelocityPlayEnvCfg(SmoothClearanceVelocityEnvCfg):
    """Visualization configuration for the smooth-clearance task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
