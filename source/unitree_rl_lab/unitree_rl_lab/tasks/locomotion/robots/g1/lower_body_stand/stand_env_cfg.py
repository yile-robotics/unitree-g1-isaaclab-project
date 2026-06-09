from importlib import import_module

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp

_velocity_env_cfg = import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
)
EventCfg = _velocity_env_cfg.EventCfg
RobotSceneCfg = _velocity_env_cfg.RobotSceneCfg
TerminationsCfg = _velocity_env_cfg.TerminationsCfg


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

ARM_JOINT_NAMES = [
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]

WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


@configclass
class StandCommandsCfg:
    """Always-zero velocity command used to preserve the standard deploy interface."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class LowerBodyActionsCfg:
    """Policy controls 12 leg joints; upper-body targets are environment generated."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )

    random_arms = mdp.SmoothRandomJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        preserve_order=True,
        resampling_time_range=(1.5, 4.0),
        position_delta_range=(-0.25, 0.25),
        acceleration_range=(0.5, 1.5),
        max_velocity_range=(0.2, 0.4),
        deceleration_range=(0.5, 1.5),
        accel_fraction=0.25,
        decel_fraction=0.25,
    )

    random_waist = mdp.SmoothRandomJointPositionActionCfg(
        asset_name="robot",
        joint_names=WAIST_JOINT_NAMES,
        preserve_order=True,
        resampling_time_range=(2.0, 5.0),
        position_delta_range=(-0.08, 0.08),
        acceleration_range=(0.2, 0.6),
        max_velocity_range=(0.08, 0.18),
        deceleration_range=(0.2, 0.6),
        accel_fraction=0.3,
        decel_fraction=0.3,
    )


@configclass
class LowerBodyObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        leg_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )
        leg_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class LowerBodyStandRewardsCfg:
    """Rewards for resisting upper-body motion while maintaining a quiet stance."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    track_zero_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_zero_yaw_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    base_lin_vel_xy = RewTerm(func=mdp.base_lin_vel_xy_l2, weight=-2.0)
    base_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78})

    leg_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    leg_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    leg_joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    leg_energy = RewTerm(
        func=mdp.energy,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES)},
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.5,
        params={
            "soft_ratio": 0.9,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
        },
    )

    feet_contact = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    both_feet_airborne = RewTerm(
        func=mdp.both_feet_airborne,
        weight=-10.0,
        params={
            "force_threshold": 10.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class LowerBodyStandEventCfg(EventCfg):
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.95, 1.05), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.1, 0.1),
            }
        },
    )


@configclass
class EmptyCurriculumCfg:
    pass


@configclass
class LowerBodyStandEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: LowerBodyObservationsCfg = LowerBodyObservationsCfg()
    actions: LowerBodyActionsCfg = LowerBodyActionsCfg()
    commands: StandCommandsCfg = StandCommandsCfg()
    rewards: LowerBodyStandRewardsCfg = LowerBodyStandRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: LowerBodyStandEventCfg = LowerBodyStandEventCfg()
    curriculum: EmptyCurriculumCfg = EmptyCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class LowerBodyStandPlayEnvCfg(LowerBodyStandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
