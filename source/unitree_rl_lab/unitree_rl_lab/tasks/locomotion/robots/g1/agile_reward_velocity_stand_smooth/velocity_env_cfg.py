"""Standing and joint-speed fine-tuning after smooth-clearance training."""

from importlib import import_module

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import rewards as stand_rewards


_base_cfg = import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.agile_reward_velocity_smooth_clearance.velocity_env_cfg"
)

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


@configclass
class StandSmoothRewardsCfg(_base_cfg.SmoothClearanceRewardsCfg):
    """Separate zero-command standing from moving gait shaping."""

    feet_clearance = RewTerm(
        func=_base_cfg.smooth_rewards.commanded_foot_clearance,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.12,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*ankle_roll_link"]
            ),
        },
    )
    stand_still = RewTerm(
        func=stand_rewards.stand_still_joint_deviation,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*ankle_roll_link"]
            ),
        },
    )
    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class StandSmoothVelocityEnvCfg(_base_cfg.SmoothClearanceVelocityEnvCfg):
    """Second-stage fine-tuning from the smooth-clearance policy."""

    rewards: StandSmoothRewardsCfg = StandSmoothRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class StandSmoothVelocityPlayEnvCfg(StandSmoothVelocityEnvCfg):
    """Visualization configuration for the stand-smooth task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
