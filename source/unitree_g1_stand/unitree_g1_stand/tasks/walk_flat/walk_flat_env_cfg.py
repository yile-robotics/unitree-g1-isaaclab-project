"""Flat-ground velocity walking task for the local Unitree G1 lock-waist model."""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg, G1FlatEnvCfg_PLAY

from unitree_g1_stand.assets import G1_LOCK_WAIST_CFG


@configclass
class G1LockWaistWalkFlatEnvCfg(G1FlatEnvCfg):
    """Velocity tracking task that reuses the local lock-waist G1 model."""

    def __post_init__(self):
        super().__post_init__()
        #原本 G1FlatEnvCfg 里面用的是 IsaacLab 自带的 G1 模型USD 和默认配置，这里替换成我们自己锁腰的 G1 模型 USD 和配置
        self.scene.robot = G1_LOCK_WAIST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #contact_forces 是接触力传感器，监听机器人所有 link的接触力信息也就是脚、腿、躯干等 link 的碰撞接触都可以被检测到
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*"

        # Start with conservative command ranges. Once this walks reliably, widen
        # these ranges before training a faster omnidirectional policy.
        #这里定义训练时给机器人的随机速度命令范围
        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        #heading_command = False说明你不用“朝向目标角度”模式，而是直接用：vx, vy, wz
        self.commands.base_velocity.ranges.heading = None
        #意思是大约 15% 的环境会采样到站立命令，剩下的 85% 会采样到行走命令，这样可以让机器人在训练初期更多地练习站立，逐渐适应行走。
        self.commands.base_velocity.rel_standing_envs = 0.15

        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        #意思是每隔 4 到 8 秒重新采样一次速度命令
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        #这里定义强化学习 policy 输出的 action 控制哪些关节，动作空间是机器人所有腿部关节加上腰部旋转关节，一共 25 个自由度。动作值会被缩放 0.25 倍，限制在较小范围内，帮助训练初期更稳定地学习。
        self.actions.joint_pos.joint_names = [
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
            "waist_yaw_joint",
        ]
        #也就是说，policy 每一步不能突然给很大的关节目标，有利于稳定训练，尤其是在训练初期。随着训练的进行，可以考虑逐渐增加这个缩放因子，让机器人学会更大范围的动作。
        self.actions.joint_pos.scale = 0.25

        # Re-enable velocity tracking while keeping the stabilizing penalties from
        # the standing phase strong enough for a first walking curriculum.
        #重新打开速度跟踪奖励
        #奖励机器人跟踪目标平面速度
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        #奖励机器人跟踪目标转向速度
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        #这个奖励鼓励脚有合适的离地时间
        self.rewards.feet_air_time.weight = 0.35
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*_ankle_roll_link"
        )
        self.rewards.feet_air_time.params["threshold"] = 0.35
        #惩罚身体倾斜
        self.rewards.flat_orientation_l2.weight = -2.0
        #惩罚竖直方向速度
        self.rewards.lin_vel_z_l2.weight = -1.0
        #惩罚 action 变化太快，鼓励平滑动作，这有助于训练初期的稳定性，尤其是在学习行走这样的动态任务时。
        self.rewards.action_rate_l2.weight = -0.01
        #惩罚关节加速度太大
        self.rewards.dof_acc_l2.weight = -2.0e-7
        #惩罚力矩太大。防止机器人用特别大的力硬撑
        self.rewards.dof_torques_l2.weight = -2.0e-6
        #只对腿部关节计算力矩惩罚
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        #关闭手指偏移奖励
        self.rewards.joint_deviation_fingers = None
        #手臂保持默认姿态
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
        #腰部只检查 waist_yaw
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_yaw_joint"]
        )
        #torso_link 接触地面就认为 episode 失败，重置机器人
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        #关闭外力干扰和推人事件，训练初期先让机器人学会在没有干扰的情况下行走，等行走比较稳定了再打开这些事件让机器人学会应对外部干扰。
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        #这个表示 reset 机器人关节时，不做随机扰动，基本回到默认初始姿态
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

#是测试/可视化用的环境配置
class G1LockWaistWalkFlatEnvCfg_PLAY(G1FlatEnvCfg_PLAY, G1LockWaistWalkFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
