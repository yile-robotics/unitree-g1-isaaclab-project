from __future__ import annotations

"""goal_tracking 脚本组的配置入口。

这个模块集中管理三类东西：
1. 默认资源路径：IsaacLab task 名、stand/locomotion policy 路径、房间 USD 路径。
2. 命令行参数：运行模式、键盘控制、路径跟踪、四方向 RGB-D、LaViRA episode 和调试功能。
3. env_cfg 修补：在创建 IsaacLab 环境前，把训练环境改成适合单机器人部署测试的形态。

主入口 `isaacsim_path_follwing.py` 会先调用这里构建 parser，再调用下面的
configure_* 函数改 IsaacLab 配置。这样 policy runner 本身不用关心路径字符串、
随机化开关、命令重采样等细节。
"""

import argparse
from pathlib import Path


# 通过当前文件位置反推项目路径，避免在代码里到处写绝对路径。
PROJECT_DIR = Path(__file__).resolve().parents[3]
PROJECTS_DIR = PROJECT_DIR.parent
UNITREE_RL_LAB_DIR = PROJECTS_DIR / "unitree_rl_lab"

# 默认 task 使用 unitree_rl_lab 中已经训练/验证过的配置。
DEFAULT_STAND_TASK = "Unitree-G1-29dof-LowerBody-Stand"
DEFAULT_LOCOMOTION_TASK = "Unitree-G1-29dof-Velocity"
DEFAULT_STAND_CHECKPOINT = (
    UNITREE_RL_LAB_DIR
    / "logs/rsl_rl/unitree_g1_29dof_lower_body_stand"
    / "2026-06-13_20-46-55_lower_body_stand_agile_reward_disturbance80_from22000_finetune5k_seed42"
    / "model_26999.pt"
)
DEFAULT_LOCOMOTION_ONNX = (
    UNITREE_RL_LAB_DIR
    / "deploy/robots/g1_29dof_goal_tracking/policies/locomotion/exported/policy.onnx"
)
DEFAULT_STAND_ONNX = (
    UNITREE_RL_LAB_DIR
    / "deploy/robots/g1_29dof_goal_tracking/policies/stand/exported/policy.onnx"
)
DEFAULT_CAMERA_OUTPUT_DIR = PROJECT_DIR / "outputs/isaacsim_goal_tracking/camera_probe"
DEFAULT_LAVIRA_OUTPUT_DIR = PROJECT_DIR / "outputs/isaacsim_goal_tracking/lavira_offline"

# 当前用于室内 path following 测试的房间 USD。
DEFAULT_HOUSE_USD = Path("/home/yile/scene/House/scene_047/mujoco/usd/scene_scene_047.usda")
DEFAULT_HOUSE_WAYPOINTS = [
    # From the smaller walk-in-closet room into the larger bedroom.
    # The shared doorway is around x=0.32..1.22, y=2.40 in the USD.
    (2.45, 1.15, 3.14),
    (1.75, 1.15, 3.14),
    (1.05, 1.55, 2.30),
    (0.78, 2.20, 1.57),
    (0.78, 2.85, 1.57),
    (0.95, 3.55, 1.20),
    (1.25, 4.35, 1.10),
    (1.15, 5.10, 1.70),
]

# stand policy 只控制 12 个下半身关节。这里的顺序必须和 stand policy 导出时一致。
STAND_ACTION_JOINT_NAMES = [
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

# stand policy 的 observation 关节顺序和 action 输出顺序不完全一样，不能复用上面的列表。
STAND_OBS_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]



def build_parser(app_launcher_cls) -> argparse.ArgumentParser:
    """创建命令行参数解析器。

    app_launcher_cls 是 IsaacLab 的 AppLauncher 类；它会追加 Isaac Sim 自己的
    参数，比如 --device、--headless 等。本函数只追加当前 goal_tracking 脚本
    关心的参数。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run G1 stand/locomotion policies, waypoint following, four-view "
            "RGB-D capture, and bounded LaViRA navigation in IsaacLab."
        )
    )
    parser.add_argument("--mode", choices=("stand", "locomotion", "switch"), default="stand", help="Policy runner mode.")
    parser.add_argument("--task", type=str, default=None, help="IsaacLab task name. Defaults depend on --mode.")
    parser.add_argument(
    "--checkpoint",
    type=Path,
    default=DEFAULT_STAND_CHECKPOINT,
    help="RSL-RL checkpoint to load for --mode stand.",
    )
    parser.add_argument(
    "--locomotion_onnx",
    type=Path,
    default=DEFAULT_LOCOMOTION_ONNX,
    help="ONNX policy to load for --mode locomotion/switch.",
    )
    parser.add_argument(
    "--stand_onnx",
    type=Path,
    default=DEFAULT_STAND_ONNX,
    help="Lower-body stand ONNX policy to load for --mode switch.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of IsaacLab environments.")
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many RL steps; -1 runs until Ctrl-C.")
    parser.add_argument("--print_every", type=int, default=100, help="Print robot state every N RL steps.")
    parser.add_argument("--real-time", action="store_true", default=False, help="Throttle to environment step_dt.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Use USD I/O instead of Fabric.")
    parser.add_argument(
    "--scene_usd",
    type=Path,
    default=None,
    help="Optional USD terrain/scene to use instead of the native plane.",
    )
    parser.add_argument(
    "--house",
    action="store_true",
    default=False,
    help=f"Shortcut for --scene_usd {DEFAULT_HOUSE_USD}.",
    )
    parser.add_argument(
    "--spawn",
    type=float,
    nargs=3,
    default=(2.45, 1.15, 0.8),
    metavar=("X", "Y", "Z"),
    help="Robot spawn position used when --scene_usd/--house is enabled.",
    )
    parser.add_argument("--yaw", type=float, default=3.141592653589793, help="Robot spawn yaw in radians for scene USD tests.")
    parser.add_argument(
    "--disable_randomization",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable startup friction/mass randomization and interval pushes for clean deployment probing.",
    )
    parser.add_argument(
    "--disable_auto_reset",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable time-out/fall terminations in this runner so IsaacLab does not reset the robot to spawn.",
    )
    parser.add_argument("--vx", type=float, default=0.0, help="Initial commanded forward velocity for locomotion mode.")
    parser.add_argument("--vy", type=float, default=0.0, help="Initial commanded lateral velocity for locomotion mode.")
    parser.add_argument("--wz", type=float, default=0.0, help="Initial commanded yaw velocity for locomotion mode.")
    parser.add_argument("--keyboard", action=argparse.BooleanOptionalAction, default=True, help="Use Omniverse keyboard.")
    parser.add_argument("--keyboard_vx", type=float, default=0.3, help="Keyboard forward/backward speed magnitude for locomotion-only mode.")
    parser.add_argument("--keyboard_vy", type=float, default=0.25, help="Keyboard lateral speed magnitude for locomotion-only mode.")
    parser.add_argument("--keyboard_wz", type=float, default=0.3, help="Keyboard yaw speed magnitude for locomotion-only mode.")
    parser.add_argument("--linear_command_step", type=float, default=0.05, help="Switch mode keyboard linear velocity increment.")
    parser.add_argument("--yaw_command_step", type=float, default=0.02, help="Switch mode keyboard yaw velocity increment.")
    parser.add_argument("--command_ramp_duration", type=float, default=0.5, help="Seconds for filtered command to cross the full command range.")

    # switch 模式不是硬切 policy，而是对输出动作做平滑过渡；这两个参数控制过渡时间。
    parser.add_argument("--blend_duration", type=float, default=0.6, help="Stand->locomotion blend duration in seconds.")
    parser.add_argument("--stand_blend_duration", type=float, default=0.3, help="Locomotion->stand blend duration in seconds.")

    # path follower 只生成 base velocity command，不直接改关节动作。
    # 关节控制仍然全部交给 locomotion policy 和 IsaacLab action manager。
    parser.add_argument(
    "--path_waypoints",
    type=str,
    default=None,
    help='Semicolon-separated world-frame waypoints "x,y,yaw;x,y,yaw". Defaults to a conservative right-to-left house path.',
    )
    parser.add_argument("--start_path_on_enter", action="store_true", default=False, help="Start the waypoint path automatically in switch mode.")
    parser.add_argument("--path_lookahead_distance", type=float, default=0.30, help="Path follower lookahead distance in meters.")
    parser.add_argument("--goal_tolerance", type=float, default=0.12, help="Final xy tolerance in meters for waypoint path completion.")
    parser.add_argument("--yaw_tolerance", type=float, default=0.25, help="Final yaw tolerance in radians for waypoint path completion.")
    parser.add_argument("--goal_slow_radius", type=float, default=0.60, help="Distance from final waypoint where path follower slows down.")
    parser.add_argument("--goal_xy_kp", type=float, default=0.70, help="Path follower xy proportional gain.")
    parser.add_argument("--goal_yaw_kp", type=float, default=1.00, help="Path follower yaw proportional gain.")
    parser.add_argument("--max_goal_vx", type=float, default=0.45, help="Path follower forward speed limit.")
    parser.add_argument("--max_goal_vy", type=float, default=0.35, help="Path follower lateral speed limit.")
    parser.add_argument("--max_goal_wz", type=float, default=0.35, help="Path follower yaw speed limit.")

    # 黄色路径线、绿色 waypoint、蓝色 lookahead target 都由我们自己的 PathVisualizer 生成。
    parser.add_argument(
    "--show_path",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Draw waypoint path markers and the current lookahead target in the Isaac Sim stage.",
    )

    # IsaacLab velocity command 自带蓝/绿箭头；它们会进入头部相机画面，所以默认关闭。
    parser.add_argument(
    "--command_debug_vis",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Draw IsaacLab velocity command debug arrows. Disabled by default so the head camera view is clean.",
    )

    # LaViRA 四方向 RGB-D 相机架默认关闭，避免给纯 stand/locomotion 测试增加
    # 渲染开销；显式启用后，主入口会在 AppLauncher 创建前自动打开 RTX cameras。
    parser.add_argument(
        "--four_rgbd_cameras",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Attach four IsaacLab RGB-D sensors (forward/left/behind/right) on the G1 head-top rig.",
    )
    parser.add_argument("--rgbd_camera_width", type=int, default=640, help="Four-view RGB-D image width.")
    parser.add_argument("--rgbd_camera_height", type=int, default=480, help="Four-view RGB-D image height.")
    parser.add_argument(
        "--rgbd_camera_hfov_deg",
        type=float,
        default=79.0,
        help="Horizontal field of view in degrees; 79 matches LaViRA's Habitat RGB-D sensors.",
    )
    parser.add_argument(
        "--rgbd_camera_update_period",
        type=float,
        default=0.0,
        help="IsaacLab camera update period in seconds; the synchronized first version requires 0.",
    )
    parser.add_argument("--rgbd_camera_near", type=float, default=0.1, help="RGB-D near clipping plane in meters.")
    parser.add_argument(
        "--rgbd_camera_far",
        type=float,
        default=5.0,
        help="RGB-D far clipping plane in meters; 5.0 matches LaViRA's Habitat depth sensor.",
    )
    parser.add_argument(
        "--camera_rig_height",
        type=float,
        default=0.56,
        help="Head-top basket camera optical-center height in torso_link coordinates, in meters.",
    )
    parser.add_argument(
        "--camera_rig_radius",
        type=float,
        default=0.085,
        help="Horizontal distance from basket center to each camera optical center, in meters.",
    )
    parser.add_argument(
        "--camera_down_tilt_deg",
        type=float,
        default=12.0,
        help="Common downward optical-axis tilt for all four cameras, in degrees.",
    )
    parser.add_argument(
        "--four_rgbd_set_viewport",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the forward RGB-D sensor as the GUI viewport camera.",
    )
    parser.add_argument(
        "--four_rgbd_debug_points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Temporarily draw colored spheres at the four camera optical centers in env_0.",
    )
    parser.add_argument(
        "--camera_debug_save_once",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After camera warm-up, save one four-view RGB-D FrameBundle for validation.",
    )
    parser.add_argument(
        "--camera_debug_warmup_steps",
        type=int,
        default=5,
        help="Completed RL steps to wait before the one-shot debug FrameBundle capture.",
    )
    parser.add_argument(
        "--camera_output_dir",
        type=Path,
        default=DEFAULT_CAMERA_OUTPUT_DIR,
        help="Root directory for one-shot RGB-D FrameBundle debug output.",
    )

    # 一次性只读决策模式：生成真实四视图统一导航输入，保存离线副本并向
    # mock/真实 adapter URL 做一次 HTTP multipart 往返。响应只校验和打印，不控制机器人。
    parser.add_argument(
        "--lavira_decision_probe",
        "--lavira_la_probe",
        dest="lavira_decision_probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run one FrameBundle -> unified navigation request -> HTTP response validation probe.",
    )
    parser.add_argument(
        "--lavira_history_probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the guarded bounded LaViRA history loop with NAVIGATE execution "
            "plus world-goal-replanned BACKTRACK and final-approach STOP support."
        ),
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="VLN instruction used by the one-shot or bounded LaViRA navigation mode.",
    )
    parser.add_argument(
        "--lavira_session_id",
        type=str,
        default="robot_01_task_001",
        help="Stable navigation session id echoed by the schema v2 server response.",
    )
    parser.add_argument(
        "--lavira_server_url",
        type=str,
        default="http://127.0.0.1:8765/v1/lavira/decision",
        help="Qwen end-to-end decision HTTP endpoint used by LaViRA decision modes.",
    )
    parser.add_argument(
        "--lavira_timeout",
        type=float,
        default=90.0,
        help="HTTP timeout in seconds for each LaViRA navigation request.",
    )
    parser.add_argument(
        "--lavira_decision_warmup_steps",
        "--lavira_la_warmup_steps",
        dest="lavira_decision_warmup_steps",
        type=int,
        default=5,
        help="Completed RL steps to wait before capturing the navigation panorama.",
    )
    parser.add_argument(
        "--lavira_history_settle_seconds",
        type=float,
        default=0.8,
        help=(
            "Stable stand time required after an executed action before history "
            "or terminal episode state is committed."
        ),
    )
    parser.add_argument(
        "--lavira_history_max_decisions",
        type=int,
        default=3,
        help=(
            "Bounded episode request count. The first N-1 ordinary NAVIGATE/"
            "BACKTRACK decisions may execute; decision N is normally read-only, "
            "but a terminal STOP still completes its final approach."
        ),
    )
    parser.add_argument(
        "--lavira_history_execution_timeout",
        type=float,
        default=30.0,
        help="Maximum locomotion time for each bounded NAVIGATE, BACKTRACK, or STOP action.",
    )
    parser.add_argument(
        "--lavira_stop_reached_threshold_m",
        type=float,
        default=0.75,
        help=(
            "Final STOP target distance in meters. The 0.75 m default mirrors "
            "LaViRA's TARGET_REACHED_THRESHOLD=15 cells at 0.05 m/cell."
        ),
    )
    parser.add_argument(
        "--lavira_backtrack_max_path_m",
        type=float,
        default=6.0,
        help=(
            "Maximum accepted length of a BACKTRACK path."
        ),
    )
    parser.add_argument(
        "--lavira_backtrack_strategy",
        choices=("replan_world_goal", "stored_reverse"),
        default="replan_world_goal",
        help=(
            "BACKTRACK planning strategy. replan_world_goal matches "
            "lavira_code/qwen_end2end by replanning to the selected waypoint's "
            "stored world pose; stored_reverse preserves the previous behavior "
            "of reversing accepted historical FMM segments."
        ),
    )
    parser.add_argument(
        "--lavira_output_dir",
        type=Path,
        default=DEFAULT_LAVIRA_OUTPUT_DIR,
        help="Root directory for offline navigation request/response bundles.",
    )
    parser.add_argument(
        "--lavira_projection_debug_marker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After a successful NAVIGATE/STOP projection, draw its semantic surface "
            "point as a magenta GUI marker. The marker is created after model capture."
        ),
    )
    parser.add_argument(
        "--lavira_local_map_probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After a successful bbox-depth projection, build and save a LaViRA-style "
            "four-view traversability map and safe target; never moves the robot."
        ),
    )
    parser.add_argument(
        "--nav_map_resolution_m",
        type=float,
        default=0.05,
        help="Navigation grid resolution in meters; 0.05 matches LaViRA.",
    )
    parser.add_argument(
        "--nav_map_size_m",
        type=float,
        default=24.0,
        help="Square navigation map side length in meters; 24 matches LaViRA.",
    )
    parser.add_argument(
        "--nav_depth_stride",
        type=int,
        default=4,
        help="Depth sampling stride; 4 maps 640x480 input to LaViRA's 160x120 density.",
    )
    parser.add_argument(
        "--nav_nominal_base_height_m",
        type=float,
        default=0.80,
        help="Expected standing G1 root height used only as floor-estimation prior.",
    )
    parser.add_argument(
        "--nav_floor_search_half_range_m",
        type=float,
        default=0.30,
        help="Half range around the nominal floor prior used by depth height histogram.",
    )
    parser.add_argument(
        "--nav_obstacle_min_height_m",
        type=float,
        default=0.10,
        help="Minimum point height above estimated floor classified as a G1 obstacle.",
    )
    parser.add_argument(
        "--nav_obstacle_max_height_m",
        type=float,
        default=1.60,
        help="Maximum point height above estimated floor included in the G1 obstacle band.",
    )
    parser.add_argument(
        "--nav_robot_radius_m",
        type=float,
        default=0.35,
        help="Conservative horizontal G1 radius used to inflate mapped obstacles.",
    )
    parser.add_argument(
        "--nav_target_retreat_step_m",
        type=float,
        default=0.10,
        help="LaViRA-style distance decrement when a projected target is not traversable.",
    )
    parser.add_argument(
        "--nav_target_snap_max_m",
        type=float,
        default=1.00,
        help="Maximum nearest-traversable fallback distance after ray retreat fails.",
    )
    parser.add_argument(
        "--lavira_fmm_probe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run LaViRA-style skfmm planning on the saved local map and export a "
            "world-frame waypoint path; never moves the robot."
        ),
    )
    parser.add_argument(
        "--fmm_step_size_cells",
        type=int,
        default=5,
        help="LaViRA short-term-goal ring radius in grid cells; original value is 5.",
    )
    parser.add_argument(
        "--fmm_goal_tolerance_cells",
        type=int,
        default=1,
        help="Grid-cell tolerance used only when terminating full FMM path extraction.",
    )
    parser.add_argument(
        "--fmm_waypoint_spacing_m",
        type=float,
        default=0.25,
        help="Approximate spacing of line-checked world-frame FMM waypoints.",
    )
    parser.add_argument(
        "--fmm_max_path_steps",
        type=int,
        default=20_000,
        help="Safety limit for monotone FMM path extraction iterations.",
    )
    parser.add_argument(
        "--lavira_execute_fmm_path",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly execute one successful FMM path through the existing "
            "switch-mode waypoint follower and locomotion ONNX policy."
        ),
    )
    parser.add_argument(
        "--fmm_execute_start_tolerance_m",
        type=float,
        default=0.15,
        help="Reject an FMM path if the robot has drifted this far from its captured start.",
    )
    parser.add_argument(
        "--fmm_execute_max_path_m",
        type=float,
        default=2.0,
        help=(
            "Maximum accepted NAVIGATE/STOP FMM path length; BACKTRACK uses "
            "--lavira_backtrack_max_path_m."
        ),
    )
    parser.add_argument(
        "--fmm_execute_cross_track_abort_m",
        type=float,
        default=0.40,
        help="Stop and request stand if the robot leaves the FMM path by this distance.",
    )
    parser.add_argument(
        "--fmm_execute_tilt_abort_rad",
        type=float,
        default=0.50,
        help="Stop and request stand if body tilt exceeds this angle during FMM execution.",
    )
    parser.add_argument(
        "--fmm_execute_lookahead_m",
        type=float,
        default=0.20,
        help="Pure-pursuit lookahead used only for dynamically loaded FMM paths.",
    )
    parser.add_argument(
        "--fmm_execute_max_vx",
        type=float,
        default=0.20,
        help="Conservative forward speed cap for FMM-driven locomotion.",
    )
    parser.add_argument(
        "--fmm_execute_max_vy",
        type=float,
        default=0.12,
        help="Conservative lateral speed cap for FMM-driven locomotion.",
    )
    parser.add_argument(
        "--fmm_execute_max_wz",
        type=float,
        default=0.25,
        help="Conservative yaw-rate cap for FMM-driven locomotion.",
    )

    parser.add_argument(
        "--remove_bedroom_wardrobe_doors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Runtime-remove the open bedroom wardrobe door prims from the loaded house stage.",
    )
    app_launcher_cls.add_app_launcher_args(parser)
    return parser


def configure_scene_probe(env_cfg, scene_usd: Path | None, args_cli) -> None:
    """把原生 IsaacLab 环境改成适合房间场景部署测试的形态。

    如果 scene_usd 为空，就保持原来的训练/测试环境；如果传入房间 USD，
    则把 terrain 换成这个 USD，同时设置机器人 spawn、关闭 terrain curriculum
    和会干扰复现实验的随机化。
    """
    if scene_usd is None:
        return
    if not scene_usd.exists():
        raise FileNotFoundError(scene_usd)

    spawn_x, spawn_y, spawn_z = (float(v) for v in args_cli.spawn)

    # 只跑一个或少量环境时，env_spacing 不再重要；terrain 改成 usd 后也不应该再用
    # terrain_generator，否则 curriculum/reset 逻辑会访问不存在的 generator。
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain.terrain_type = "usd"
    env_cfg.scene.terrain.usd_path = str(scene_usd)
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.visual_material = None
    env_cfg.scene.robot.init_state.pos = (spawn_x, spawn_y, spawn_z)
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None

    # reset_base 仍然会在 env.reset() 时执行，所以这里把 pose/velocity range 都固定住，
    # 保证机器人不会被 IsaacLab 随机重置到其它位置或带初速度出生。
    reset_base = env_cfg.events.reset_base
    reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (float(args_cli.yaw), float(args_cli.yaw)),
    }
    reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

    if args_cli.disable_randomization:
        # 排查 policy/动力学时先关掉随机摩擦、质量扰动、推搡和外力。
        env_cfg.events.physics_material = None
        env_cfg.events.add_base_mass = None
        env_cfg.events.push_robot = None
        if hasattr(env_cfg.events, "base_external_force_torque"):
            env_cfg.events.base_external_force_torque = None

    print(f"[INFO] Native probe scene USD: {scene_usd}")
    print(f"[INFO] Native probe spawn: pos=({spawn_x:.3f}, {spawn_y:.3f}, {spawn_z:.3f}), yaw={args_cli.yaw:.3f}")
    print(f"[INFO] Native probe randomization disabled: {args_cli.disable_randomization}")


def configure_locomotion_commands(env_cfg, args_cli=None) -> None:
    """让速度命令由键盘/path follower 外部控制，而不是由 IsaacLab 随机采样。"""
    command_cfg = env_cfg.commands.base_velocity

    # 原训练环境会定期重采样命令；部署时命令来自用户或路径跟踪器，所以这里等价关闭重采样。
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.heading_command = False
    if hasattr(command_cfg, "debug_vis"):
        command_cfg.debug_vis = bool(getattr(args_cli, "command_debug_vis", False))

    if hasattr(command_cfg, "limit_ranges") and command_cfg.limit_ranges is not None:
        # 用训练时定义的 limit_ranges 覆盖 ranges，避免给 policy 输入超出训练分布的速度命令。
        command_cfg.ranges = command_cfg.limit_ranges

    # reset 时关节速度固定为 0，避免 policy 接管前已有初始抖动。
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    if hasattr(env_cfg, "curriculum"):
        # 房间 USD 没有 terrain generator，继续跑 terrain curriculum 会报 NoneType.size。
        for term_name in ("terrain_levels", "lin_vel_cmd_levels", "ang_vel_cmd_levels"):
            if hasattr(env_cfg.curriculum, term_name):
                setattr(env_cfg.curriculum, term_name, None)


def disable_auto_reset_terms(env_cfg, args_cli) -> None:
    """Keep deployment probes running until Ctrl-C instead of IsaacLab episode resets."""
    if not args_cli.disable_auto_reset:
        return

    if hasattr(env_cfg, "episode_length_s"):
        # episode_length_s 是 timeout 的常见来源之一；设大后基本只会由 Ctrl-C 停止。
        env_cfg.episode_length_s = 1.0e9

    disabled_terms: list[str] = []
    if hasattr(env_cfg, "terminations"):
        # fall/contact/time_out 等终止项在训练中有用，但部署观察时会让机器人突然回到初始点。
        for term_name, term_cfg in vars(env_cfg.terminations).items():
            if term_name.startswith("_") or term_cfg is None:
                continue
            setattr(env_cfg.terminations, term_name, None)
            disabled_terms.append(term_name)

    print(
        "[INFO] Auto reset disabled in this runner: "
        f"episode_length_s={getattr(env_cfg, 'episode_length_s', 'n/a')}, "
        f"disabled_terminations={disabled_terms or 'none'}."
    )


def disable_observation_corruption(env_cfg) -> None:
    """关闭 policy 观测噪声，保证部署 probe 尽量可复现。"""
    if hasattr(env_cfg.observations, "policy") and hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False
