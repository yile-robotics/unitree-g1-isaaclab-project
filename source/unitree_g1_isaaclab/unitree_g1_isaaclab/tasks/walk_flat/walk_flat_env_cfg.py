"""Flat-ground velocity walking task for the local Unitree G1 lock-waist model."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg, G1FlatEnvCfg_PLAY

from unitree_g1_isaaclab.assets import G1_LOCK_WAIST_CFG


@configclass
class G1LockWaistWalkFlatEnvCfg(G1FlatEnvCfg):
    """Velocity tracking task that reuses the local lock-waist G1 model."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_LOCK_WAIST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"

        # Start with conservative command ranges. Once this walks reliably, widen
        # these ranges before training a faster omnidirectional policy.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.base_velocity.ranges.heading = None
        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)

        self.actions.joint_pos.joint_names = [
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
            "waist_yaw_joint",
        ]
        self.actions.joint_pos.scale = 0.25

        # Re-enable velocity tracking while keeping the stabilizing penalties from
        # the standing phase strong enough for a first walking curriculum.
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.feet_air_time.weight = 0.35
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*_ankle_roll_link"
        )
        self.rewards.feet_air_time.params["threshold"] = 0.35

        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -2.0e-7
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_arms.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
        )
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_yaw_joint"]
        )

        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)


class G1LockWaistWalkFlatEnvCfg_PLAY(G1FlatEnvCfg_PLAY, G1LockWaistWalkFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
