"""Unitree G1 29DoF lock-waist asset configuration."""
#G1 的腿和脚踝用 DCMotorCfg，因为它们负责站立/行走，需要更真实的电机限制；

#腰和手臂用 ImplicitActuatorCfg，因为现在主要是让它们保持默认姿态，不作为主要学习对象；

#腿部关节力矩、速度、Kp/Kd 按不同关节分别设置；

#脚踝更软，膝盖更硬；

#腰和手臂用很大的 stiffness 锁住，防止乱动；

#机器人会生成在每个并行环境的 /World/envs/env_x/Robot 路径下。
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from unitree_g1_isaaclab import UNITREE_G1_ISAACLAB_PROJECT_DIR
#这个文件定义机器人配置：G1_LOCK_WAIST_CFG它告诉 IsaacLab：USD 文件在哪里
#初始站姿是什么
#腿部、脚踝、腰、手臂关节怎么驱动
#关节刚度、阻尼、力矩限制是多少

#USD 文件在哪里
G1_LOCK_WAIST_USD = (
    Path(UNITREE_G1_ISAACLAB_PROJECT_DIR)
    / "assets"
    / "g1_29dof_lock_waist_rev_1_0"
    / "usd"
    / "g1_29dof_lock_waist_rev_1_0.usd"
)

#初始站姿是什么
G1_LOCK_WAIST_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(G1_LOCK_WAIST_USD),
        activate_contact_sensors=True,
        #刚体物理属性
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            #不关闭重力
            disable_gravity=False,
            #它控制物理引擎是否保留上一帧的加速度信息。这里设为 False 是常见配置
            retain_accelerations=False,
            #线速度阻尼和角速度阻尼都设为 0，允许机器人自由运动而不受额外阻力影响。
            linear_damping=0.0,
            #机器人 link 转动时，不额外加旋转阻力
            angular_damping=0.0,
            #最大线速度限制，非常大，基本等于不限制
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            #如果两个物体发生穿透，物理引擎把它们分开的最大速度是 1.0 m/s。
            max_depenetration_velocity=1.0,
        ),
        #关节物理属性
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            #机器人自己的各个 link 之间，不启用自碰撞检测，避免不必要的碰撞计算和潜在的物理不稳定。
            #！！！如果以后机器人乱穿模就把它改成 True 吧
            enabled_self_collisions=False,
            #不要把机器人的 root link 固定在世界中也就是说机器人是自由的，会受重力影响，可以倒，可以走，可以跳。。
            fix_root_link=False,
            #每个仿真 step 里，PhysX 为了修正位置误差，会算几轮
            solver_position_iteration_count=8,
            #这是速度约束的迭代次数
            solver_velocity_iteration_count=4,
        ),
    ),
    #定义 机器人一开始生成到仿真世界时的初始状态
    init_state=ArticulationCfg.InitialStateCfg(
        #机器人生成时的初始位置，单位是米。这里设置在世界坐标系的 (0.0, 0.0, 0.75)，也就是地面上方 0.75 米处。
        pos=(0.0, 0.0, 0.75),
        #这个是机器人初始关节角度 单位弧度
        joint_pos={
            ".*_hip_pitch_joint": -0.10,
            ".*_knee_joint": 0.30,
            ".*_ankle_pitch_joint": -0.20,
            ".*_shoulder_pitch_joint": 0.20,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.30,
            "waist_yaw_joint": 0.0,
        },
        #所有关节初始速度都是 0
        joint_vel={".*": 0.0},
    ),
    #训练/控制时不要用满关节硬限位，只用 90% 的安全范围
    soft_joint_pos_limit_factor=0.9,
    #也就是把机器人关节分成四组：
    #legs  = 髋关节 + 膝盖
    #feet  = 脚踝 pitch / roll
    #waist = 腰 yaw
    #arms  = 肩膀、肘、手腕
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                #髋 yaw 最大力矩：88髋 roll 最大力矩：88髋 pitch 最大力矩：88膝盖最大力矩：139
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit={
                # 这是关节最大速度 rad/s
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                #这个就是 PD 里的 Kp，数值越大，关节越不容易被外力扰动，保持在目标位置。腿部关节需要比较高的刚度来支撑身体重量和抵抗外部扰动。
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                #这个是 PD 里的 Kd，数值越大，关节越不容易振荡。适当的阻尼可以帮助机器人更平稳地运动，减少过度振荡。
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
            },
            #armature 可以理解成电机侧的附加转动惯量加上 armature 后，关节运动会更像真实电机/减速器，有一点惯性
            armature={".*_hip_.*": 0.03, ".*_knee_joint": 0.03},
            #这是电机饱和力矩，也就是电机能输出的最大力矩，超过这个力矩电机会过热或者损坏。设置合理的饱和力矩可以保护电机，同时也能让训练更稳定。
            saturation_effort=180.0,
        ),
        #脚踝也是 DCMotorCfg，因为脚踝也参与真实站立和平衡  脚踝软一点，可以让脚和地面接触更自然
        "feet": DCMotorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={".*_ankle_pitch_joint": 28.5, ".*_ankle_roll_joint": 28.5},
            damping={".*_ankle_pitch_joint": 1.8, ".*_ankle_roll_joint": 1.8},
            effort_limit={".*_ankle_pitch_joint": 50.0, ".*_ankle_roll_joint": 50.0},
            velocity_limit={".*_ankle_pitch_joint": 37.0, ".*_ankle_roll_joint": 37.0},
            armature=0.03,
            saturation_effort=80.0,
        ),
        # 腰和手臂不作为 PPO action，用中高刚度 PD 保持默认姿态，贴近真机上半身位置锁定的用法。
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=300.0,
            damping=8.0,
            armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness=200.0,
            damping=8.0,
            armature={".*_shoulder_.*": 0.001, ".*_elbow_.*": 0.001, ".*_wrist_.*_joint": 0.001},
        ),
    },
    prim_path="/World/envs/env_.*/Robot",
)
