"""Flat-ground standing task for the local Unitree G1 lock-waist model."""
#这个用来告诉 reward / termination / sensor：我要操作 scene 里面的哪个对象，以及这个对象的哪些 link / joint。
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass
#这里导入 IsaacLab 官方已有的 G1 平地速度跟踪任务配置 然后修改成你自己的站立任务
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg, G1FlatEnvCfg_PLAY
#这个任务不用官方 G1，而是用你自己本地的 lock-waist G1。
from unitree_g1_isaaclab.assets import G1_LOCK_WAIST_CFG

#这个是站立任务配置。它做的事情是：

#使用你的 G1_LOCK_WAIST_CFG
#地形设成平地
#速度命令全部设为 0
#关闭走路相关 reward，比如 feet_air_time
#强化站立相关 reward，比如身体保持竖直、动作别抖、关节别乱动
#注册接触传感器和摔倒终止条件
@configclass
class G1LockWaistStandFlatEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        #这里先复制官方 G1 平地行走任务  然后修改成你的 lock-waist 站立任务
        super().__post_init__()
        #这句是把环境里的机器人换成我的usd机器人模型
        self.scene.robot = G1_LOCK_WAIST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #contact force sensor 监听 robot 下面所有 body/link 的接触力
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"
        #速度命令全部设为 0   这个阶段不训练走路，只训练站住
        
        # Standing phase: train zero commanded velocity first.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = None
        #也就是说所有并行环境里的机器人都被要求站立
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False

        #站立阶段先只控制腿和腰，手臂由 PD 保持默认姿态，避免 policy 学会甩手来维持平衡。
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

        #关闭走路奖励
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        #奖励机器人达到目标转向速度
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        #这个是关闭“脚离地时间奖励”。
        self.rewards.feet_air_time = None

        # Keep the body quiet and upright.
        #强化站立相关 reward
        #身体越不竖直，惩罚越大
        self.rewards.flat_orientation_l2.weight = -3.0
        #惩罚 base 在 z 方向的速度。也就是不希望机器人上下跳
        self.rewards.lin_vel_z_l2.weight = -2.0
        #惩罚动作变化太快
        self.rewards.action_rate_l2.weight = -0.01
        #惩罚关节加速度过大
        self.rewards.dof_acc_l2.weight = -2.0e-7
        #惩罚关节力矩过大
        self.rewards.dof_torques_l2.weight = -2.0e-6
        #torque penalty 只作用在腿部点。
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        #关闭 finger reward
        # The local lock-waist URDF does not have the official G1 finger joints,
        # and its elbow joints are named left/right_elbow_joint.
        self.rewards.joint_deviation_fingers = None
        #修改手臂关节 deviation reward
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
        #惩罚 waist_yaw_joint 偏离默认位置
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_yaw_joint"]
        )
        #修改摔倒终止条件 如果 torso_link 接触到地面，就认为机器人摔倒，结束这一轮 episode
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        #关闭外力扰动和推机器人事件(等训练好了再增加鲁棒性)
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        #reset 时关节不要随机扰动 也可以以后加扰动
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)


class G1LockWaistStandFlatEnvCfg_PLAY(G1FlatEnvCfg_PLAY, G1LockWaistStandFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        #play时候可视化机器人的数量
        self.scene.num_envs = 16
        #环境间距
        self.scene.env_spacing = 2.5
        #关闭 observation corruption 以后加上去让训练更鲁棒
        self.observations.policy.enable_corruption = False
