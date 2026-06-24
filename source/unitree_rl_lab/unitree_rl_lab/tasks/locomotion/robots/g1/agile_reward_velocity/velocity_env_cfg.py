"""Independent 29-DoF velocity task using the WBC-AGILE reward design."""

import math
from importlib import import_module

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from . import rewards as agile_rewards


_base_cfg = import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
)

ALL_JOINTS = SceneEntityCfg("robot", joint_names=[".*"])
FEET = SceneEntityCfg("robot", body_names=[".*ankle_roll_link"])


@configclass
class AgileRewardsCfg:
    """WBC-AGILE velocity rewards adapted to whole-body policy control."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    base_height = RewTerm(
        func=agile_rewards.base_height_exp,
        weight=2.5,
        params={"target_height": 0.78, "std": 0.1},
    )
    orientation = RewTerm(
        func=agile_rewards.flat_body_orientation_exp,
        weight=5.0,
        params={
            "std": math.radians(10.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=["pelvis", "torso_link"]),
        },
    )

    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
        params={"asset_cfg": ALL_JOINTS},
    )
    ankle_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"])},
    )
    ankle_roll_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle_roll.*"])},
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.25)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.25)
    dof_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": ALL_JOINTS},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.25)
    action_rate_rate = RewTerm(func=agile_rewards.action_rate_rate_l2, weight=-0.025)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": ALL_JOINTS},
    )
    dof_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.5,
        params={"asset_cfg": ALL_JOINTS, "soft_ratio": 0.9},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.005,
        params={"asset_cfg": ALL_JOINTS},
    )

    feet_slip = RewTerm(
        func=agile_rewards.feet_slip,
        weight=-0.05,
        params={
            "contact_threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*ankle_roll_link"]),
            "robot_cfg": FEET,
        },
    )
    feet_roll = RewTerm(
        func=agile_rewards.feet_roll_l2,
        weight=-0.05,
        params={"asset_cfg": FEET},
    )
    feet_yaw_diff = RewTerm(
        func=agile_rewards.feet_yaw_diff_l2,
        weight=-0.1,
        params={"asset_cfg": FEET},
    )
    feet_yaw_mean = RewTerm(
        func=agile_rewards.feet_yaw_mean_vs_base,
        weight=-2.0,
        params={
            "feet_asset_cfg": FEET,
            "base_body_cfg": SceneEntityCfg("robot", body_names=["pelvis"]),
        },
    )
    root_acc = RewTerm(
        func=agile_rewards.body_acc_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"])},
    )
    feet_distance = RewTerm(
        func=agile_rewards.feet_distance_from_ref,
        weight=-0.1,
        params={"asset_cfg": FEET, "ref_distance": 0.2},
    )
    jumping = RewTerm(
        func=agile_rewards.jumping,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*ankle_roll_link"]),
            "threshold": 10.0,
        },
    )

    # AGILE drives the arms separately. These terms keep policy-controlled upper-body
    # joints near their nominal pose without changing the original 29-DoF action space.
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            )
        },
    )
    joint_deviation_waist_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
    )
    joint_deviation_waist_roll_pitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["waist_roll_joint", "waist_pitch_joint"]
            )
        },
    )


@configclass
class AgileRewardVelocityEnvCfg(_base_cfg.RobotEnvCfg):
    """Training configuration; all non-reward settings remain from the 29-DoF task."""

    rewards: AgileRewardsCfg = AgileRewardsCfg()


@configclass
class AgileRewardVelocityPlayEnvCfg(AgileRewardVelocityEnvCfg):
    """Visualization configuration for the independent task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
