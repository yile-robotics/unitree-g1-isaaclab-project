"""Unitree-style velocity walking task for the local Unitree G1 lock-waist model."""

import math
from copy import deepcopy

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_g1_isaaclab.assets import G1_LOCK_WAIST_CFG
from unitree_rl_lab.tasks.locomotion import mdp


G1_LOCK_WAIST_UNITREE_STYLE_CFG = deepcopy(G1_LOCK_WAIST_CFG)
G1_LOCK_WAIST_UNITREE_STYLE_CFG.init_state.joint_pos = {
    ".*_hip_pitch_joint": -0.10,
    ".*_knee_joint": 0.30,
    ".*_ankle_pitch_joint": -0.20,
    "waist_yaw_joint": -0.0129,
    "left_shoulder_pitch_joint": 0.0316,
    "left_shoulder_roll_joint": 0.0189,
    "left_shoulder_yaw_joint": 0.0187,
    "left_elbow_joint": 1.1029,
    "left_wrist_roll_joint": 0.0155,
    "left_wrist_pitch_joint": 0.1294,
    "left_wrist_yaw_joint": 0.0120,
    "right_shoulder_pitch_joint": 0.1726,
    "right_shoulder_roll_joint": 0.0046,
    "right_shoulder_yaw_joint": -0.0119,
    "right_elbow_joint": 0.8084,
    "right_wrist_roll_joint": -0.0049,
    "right_wrist_pitch_joint": 0.1278,
    "right_wrist_yaw_joint": -0.0026,
}


FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


@configclass
class UnitreeStyleSceneCfg(InteractiveSceneCfg):
    """Flat terrain scene with the local lock-waist G1 robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_TERRAIN_CFG,
        max_init_terrain_level=FLAT_TERRAIN_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAAC_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot = G1_LOCK_WAIST_UNITREE_STYLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
#这个是整个环境的配置类，定义了环境里用到的场景、观察、动作、奖励、终止条件、事件和课程设置。初始化和reset的参数

class EventsCfg:
    """Domain randomization and reset events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
#对你的 G1 walking 任务来说，command 就是：base_velocity = [lin_vel_x, lin_vel_y, ang_vel_z]
class CommandsCfg:
    """Velocity command curriculum from unitree_rl_lab."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        #意思是大约 2% 的环境会被分配站立命令
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        #这是训练一开始的速度命令范围 最后那个从-0.1-0.1改成了-0.2-0.2
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.2, 0.2)
        ),
        #最大命令范围 limit_ranges
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )


@configclass
#PPO policy 输出的 action 到底控制机器人哪些关节，以及 action 数值怎么转换成关节目标角度
class ActionsCfg:
    """Action space matched to the current lock-waist walking task."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
        ],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
#PPO 的神经网络每一步能“看到”机器人哪些信息
class ObservationsCfg:
    """Unitree-style policy and critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        #scale表示把这个观测缩小 0.2 倍输入网络，避免数值太大 noise=... 表示训练时加噪声，让 policy 更鲁棒
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            #表示 policy observation 会加噪声
            self.enable_corruption = True
            #表示把所有 observation term 拼成一个大向量
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    #critic 是 value function，用来估计：当前状态未来大概能拿多少 reward
    class CriticCfg(ObsGroup):
        #Critic 比 policy 多了 base_lin_vel 这个观测项，帮助 critic 更准确地评估当前状态的价值。critic 的观测通常比 policy 更全面，因为它需要评估状态的长期价值，而不仅仅是做出决策。
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        #这些和 policy 基本一样，只是 critic 没有 noise
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Unitree-style rewards adapted to the local lock-waist G1 names."""

    # reward_tracking_yaw_v3: stronger yaw tracking, looser action/hip penalties.
    #速度跟踪奖励
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    #角速度跟踪奖励
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    #活着奖励
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    #这个惩罚 base 的 z 方向速度，鼓励机器人不要上下跳动，保持平稳的行走姿态。
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    #惩罚关节速度太大 防止关节疯狂摆动。
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    #惩罚关节速度太大。防止关节疯狂摆动。
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
   #惩罚关节加速度太大。防止动作抖动、突然改变
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    #惩罚 action 变化太快
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    #惩罚关节位置超出限制
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    #惩罚关节力矩过大。防止机器人用特别大的力硬撑
    energy = RewTerm(func=mdp.energy, weight=-2.0e-5)
    #这个奖励鼓励脚有合适的离地时间。过长或过短都不好，过长可能是机器人跳起来了，过短可能是拖着脚走了。
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
    #这个奖励惩罚腰部关节偏离默认位置，鼓励机器人保持腰部稳定，不要过度扭转。
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
    )
    #这个奖励惩罚腿部关节偏离默认位置，鼓励机器人保持腿部姿态稳定，不要过度扭转或伸展。
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    # 这个奖励惩罚身体倾斜，鼓励机器人保持身体平坦，避免过度倾斜导致失去平衡。
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    #这个奖励惩罚 base 在 z 方向的速度，鼓励机器人不要上下跳，保持平稳的行走姿态。
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.75})
    #这个奖励鼓励机器人跟踪目标平面速度，帮助机器人学会按照命令行走。
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.2,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    #这个奖励惩罚脚滑动，鼓励机器人在行走时保持脚部稳定，避免过度滑动导致能量浪费和不稳定。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )
    #这个奖励鼓励脚有合适的离地时间，过长或过短都不好，过长可能是机器人跳起来了，过短可能是拖着脚走了。
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=0.4,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # 这个奖励惩罚机器人身体与地面接触，鼓励机器人保持身体离地，避免摔倒或拖地。
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

#什么时候认为这一轮训练失败/结束，然后 reset 机器人，开始下一轮 episode
@configclass
class TerminationsCfg:
    """Episode termination conditions."""
    #这个条件是如果这一轮 episode 的时间超过了预设的 episode_length_s，就认为这一轮训练结束，重置机器人，开始下一轮 episode。这有助于训练过程中及时重置机器人，避免长时间处于失败状态。
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    #这个条件是如果机器人 base 的高度低于 0.2 米，就认为机器人摔倒了，结束这一轮 episode。这有助于训练过程中及时重置机器人，避免长时间处于失败状态。
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    #这个条件是如果机器人身体倾斜过大，就认为机器人摔倒了，结束这一轮 episode。这有助于训练过程中及时重置机器人，避免长时间处于失败状态。
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

#训练时根据机器人表现，自动调整任务难度
@configclass
class CurriculumCfg:
    """Terrain and command curriculum."""
    #如果机器人在当前地形走得好，就把它分配到更难的地形等级；如果走得差，就降低地形等级
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    #如果机器人线速度跟踪 reward 足够好，就扩大 lin_vel_x / lin_vel_y 的命令范围。
    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)

#完整环境总配置类
@configclass
#它继承：ManagerBasedRLEnvCfg  这个是 IsaacLab 里一个比较通用的强化学习环境配置类，里面定义了很多默认的设置，比如 episode 长度、simulation 步长、渲染设置等等。你在这个基础上重写了 __post_init__ 方法，覆盖了一些默认设置，适配你的 G1 lock-waist 行走任务。
class G1LockWaistUnitreeStyleEnvCfg(ManagerBasedRLEnvCfg):
    """Unitree-style velocity tracking environment for the local lock-waist G1."""
    #场景配置就是把前面搞的一大堆乱七八糟的环境拼在一起
    scene: UnitreeStyleSceneCfg = UnitreeStyleSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        #policy 不是每一个物理仿真步都输出 action，而是每 4 个 simulation step 输出一次 action
        self.decimation = 4
        #每个 episode 最长 20 秒。如果机器人 20 秒内没有摔倒，就触发：time_out然后 reset。
        self.episode_length_s = 20.0
        #simulation 的步长是 0.005 秒，也就是每秒 200 个 simulation step。因为 policy 每 4 个 simulation step 输出一次 action，所以 policy 的频率是 50 Hz。
        self.sim.dt = 0.005
        #渲染设置，训练时每 4 个 simulation step 渲染一次，这样可以实时看到训练过程，但又不会因为渲染过于频繁而影响性能。
        self.sim.render_interval = self.decimation
        #物理材质设置，直接把地面材质参数应用到机器人上，这样可以让训练过程中机器人和地面之间的摩擦力等物理交互更真实，帮助机器人学会更稳定的行走。
        self.sim.physics_material = self.scene.terrain.physics_material
        #这个设置是针对使用 NVIDIA PhysX 物理引擎的仿真环境，调整 GPU 上的最大刚体补丁数量，以支持更多的环境实例同时进行物理仿真。这个数值需要根据你的 GPU 内存和性能进行调整，过高可能导致性能下降，过低可能限制了环境实例的数量。
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        #这个设置是针对使用 NVIDIA PhysX 物理引擎的仿真环境，调整 GPU 上的最大接触点数量，以支持更多的环境实例同时进行物理仿真。这个数值需要根据你的 GPU 内存和性能进行调整，过高可能导致性能下降，过低可能限制了环境实例的数量。
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        #如果环境里有地形生成器，就根据课程设置启用地形课程。也就是说，训练初期地形比较简单，随着训练进行，地形逐渐变得复杂，帮助机器人逐步适应更具挑战性的地形。
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None


class G1LockWaistUnitreeStyleRobustEnvCfg(G1LockWaistUnitreeStyleEnvCfg):
    """Fine-tuning variant that keeps Unitree RL Lab's original randomization."""

    def __post_init__(self):
        super().__post_init__()

        # No extra randomization here. The base class already matches Unitree RL
        # Lab's G1 locomotion settings for friction, mass, reset, push, and
        # observation noise.


class G1LockWaistUnitreeStyleTrackingTuneEnvCfg(G1LockWaistUnitreeStyleEnvCfg):
    """Fine-tuning variant using Unitree RL Lab's original command mix."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.rel_standing_envs = 0.02


class G1LockWaistUnitreeStyleEnvCfg_PLAY(G1LockWaistUnitreeStyleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        #大部分配置和训练环境一样，但专门改成适合看效果、测试 checkpoint 的版本
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
