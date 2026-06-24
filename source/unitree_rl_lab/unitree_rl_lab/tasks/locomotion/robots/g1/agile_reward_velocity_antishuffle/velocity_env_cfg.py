"""Independent anti-shuffle fine-tuning task for the AGILE-reward policy."""

from importlib import import_module

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp


_base_cfg = import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.agile_reward_velocity.velocity_env_cfg"
)


@configclass
class AntiShuffleRewardsCfg(_base_cfg.AgileRewardsCfg):
    """AGILE reward set plus gated stance-time and swing-clearance rewards."""

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.3,
        params={
            "command_name": "base_velocity",
            "threshold": 0.25,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*ankle_roll_link"]
            ),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*ankle_roll_link"]
            ),
        },
    )


@configclass
class AntiShuffleVelocityEnvCfg(_base_cfg.AgileRewardVelocityEnvCfg):
    """Fine-tuning environment that leaves actions, actuators and events unchanged."""

    rewards: AntiShuffleRewardsCfg = AntiShuffleRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # The source checkpoint already learned the velocity curriculum. Use
        # its full command envelope while fine-tuning the gait shape.
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class AntiShuffleVelocityPlayEnvCfg(AntiShuffleVelocityEnvCfg):
    """Visualization configuration for the anti-shuffle task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
