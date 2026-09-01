#!/usr/bin/env python3
from __future__ import annotations

"""Run the new local-frame combined-model + iPlanner flow in Isaac Sim."""

import argparse
import math
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
PROJECTS_DIR = PROJECT_DIR.parent
LEGACY_DIR = PROJECT_DIR / "scripts" / "isaacsim_goal_tracking"
UNITREE_RL_LAB_DIR = PROJECTS_DIR / "unitree_rl_lab"

sys.path.insert(0, str(UNITREE_RL_LAB_DIR / "source" / "unitree_rl_lab"))
sys.path.insert(0, str(LEGACY_DIR))

from isaaclab.app import AppLauncher  # noqa: E402
from goal_tracking.config import build_parser  # noqa: E402


parser = build_parser(AppLauncher)
parser.set_defaults(
    mode="switch",
    # Match the open-room start used by the existing goal-tracking README's
    # real-model tests.  This override is local to the new runner; the legacy
    # parser and isaacsim_goal_tracking defaults remain unchanged.
    spawn=(1.15, 5.25, 0.8),
    four_rgbd_cameras=True,
    keyboard=False,
    show_path=False,
    lavira_decision_probe=False,
    lavira_history_probe=False,
)
parser.add_argument(
    "--iplanner_url",
    type=str,
    default="http://127.0.0.1:8888",
    help="Uni-LaViRA-compatible local iPlanner HTTP server.",
)
parser.add_argument("--iplanner_timeout_s", type=float, default=5.0)
parser.add_argument(
    "--g3_session",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable the b92 health/start/report/end session lifecycle.",
)
parser.add_argument("--g3_session_timeout_s", type=float, default=180.0)
parser.add_argument(
    "--g3_motion_window_s",
    type=float,
    default=1.0,
    help="Execution-time interval for phase-three motion_window reports.",
)
parser.add_argument(
    "--local_map_progress",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Enable the unbounded sparse 5cm episode map from the physical forward "
        "RGB-D. Requires --local_use_isaac_odometry."
    ),
)
parser.add_argument("--local_map_resolution_m", type=float, default=0.05)
parser.add_argument("--local_map_depth_stride", type=int, default=8)
parser.add_argument("--local_map_nominal_base_height_m", type=float, default=0.80)
parser.add_argument("--local_map_floor_z_world_m", type=float, default=0.0)
parser.add_argument("--local_map_obstacle_min_height_m", type=float, default=0.10)
parser.add_argument("--local_map_obstacle_max_height_m", type=float, default=1.60)
parser.add_argument("--local_map_robot_radius_m", type=float, default=0.35)
parser.add_argument("--local_rotation_speed_rad_s", type=float, default=0.4)
parser.add_argument(
    "--local_single_camera_panorama",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Capture forward/left/behind/right with only the forward RGB-D while "
        "the robot performs four consecutive left quarter-turns. The other "
        "configured Isaac cameras remain attached but are not read."
    ),
)
parser.add_argument(
    "--local_rotation_duration_scale_sim",
    type=float,
    default=1.0,
    help="Isaac-specific timed-rotation scale; calibrate independently from G1.",
)
parser.add_argument("--local_rotation_settle_s", type=float, default=0.5)
parser.add_argument("--local_safe_distance_m", type=float, default=0.5)
parser.add_argument("--local_walk_speed_m_s", type=float, default=0.3)
parser.add_argument("--local_max_forward_speed_m_s", type=float, default=0.4)
parser.add_argument("--local_max_yaw_speed_rad_s", type=float, default=0.5)
parser.add_argument("--local_lookahead_m", type=float, default=0.5)
parser.add_argument("--local_goal_tolerance_m", type=float, default=1.0)
parser.add_argument(
    "--local_tracking_controller",
    choices=("pure_pursuit", "kp"),
    default="pure_pursuit",
    help=(
        "Select the isolated local trajectory controller. pure_pursuit keeps "
        "the frozen existing behavior; kp enables the older fixed-frame "
        "path-projection vx/vy/wz tracking law."
    ),
)
parser.add_argument(
    "--local_blind_yaw_radius_m",
    type=float,
    default=1.5,
    help=(
        "Pure Pursuit suppresses yaw inside this distance. The fixed-frame "
        "holonomic Kp controller does not use this setting."
    ),
)
parser.add_argument("--local_kp_xy", type=float, default=0.7)
parser.add_argument("--local_kp_yaw", type=float, default=1.0)
parser.add_argument("--local_kp_slow_radius_m", type=float, default=1.0)
parser.add_argument("--local_kp_yaw_deadband_deg", type=float, default=10.0)
parser.add_argument("--local_kp_max_lateral_speed_m_s", type=float, default=0.12)
parser.add_argument("--local_kp_max_yaw_speed_rad_s", type=float, default=0.35)
parser.add_argument("--local_yaw_bias_rad_s", type=float, default=0.0)
parser.add_argument("--local_replan_interval_s", type=float, default=0.1)
parser.add_argument(
    "--local_navigation_rate_hz",
    type=float,
    default=20.0,
    help=(
        "High-level rotation/Pure-Pursuit update rate. The locomotion policy "
        "remains at its trained 50 Hz rate and holds the latest command."
    ),
)
parser.add_argument(
    "--local_dead_reckoning_linear_scale_sim", type=float, default=0.7
)
parser.add_argument(
    "--local_dead_reckoning_angular_scale_sim", type=float, default=0.8
)
parser.add_argument("--local_action_timeout_s", type=float, default=60.0)
parser.add_argument("--local_post_action_stand_s", type=float, default=0.8)
parser.add_argument("--local_backtrack_max_path_m", type=float, default=6.0)
parser.add_argument(
    "--local_backtrack_start_tolerance_m", type=float, default=1.0
)
parser.add_argument("--local_backtrack_segment_length_m", type=float, default=1.0)
parser.add_argument("--local_backtrack_goal_tolerance_m", type=float, default=0.35)
parser.add_argument(
    "--local_backtrack_heading_tolerance_rad", type=float, default=0.20
)
parser.add_argument(
    "--local_backtrack_breadcrumb_spacing_m", type=float, default=0.15
)
parser.add_argument(
    "--local_history_max_waypoints",
    type=int,
    default=0,
    help=(
        "Maximum completed waypoints sent as model history. 0 (default) keeps "
        "all text history; schema-v2 still attaches images only to the latest four."
    ),
)
parser.add_argument(
    "--local_max_decisions",
    type=int,
    default=0,
    help=(
        "Maximum combined-model decisions for this local episode. "
        "0 (default) runs until model STOP, failure, Ctrl-C, or --max_steps. "
        "This is independent of model history retention."
    ),
)
parser.add_argument(
    "--local_use_isaac_odometry",
    action="store_true",
    default=False,
    help="Explicitly opt into Isaac root-pose odometry. Default navigation is no-odom.",
)
parser.add_argument(
    "--local_output_dir",
    type=Path,
    default=PROJECT_DIR / "outputs" / "isaacsim_lavira_g3_interface_g1",
)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if not args_cli.instruction.strip():
    parser.error("The local VLN runner requires a non-empty --instruction.")
if args_cli.num_envs != 1:
    parser.error("The local VLN runner currently requires --num_envs 1.")
if args_cli.local_max_decisions < 0:
    parser.error("--local_max_decisions must be >= 0; use 0 for unlimited.")
if args_cli.local_history_max_waypoints < 0:
    parser.error("--local_history_max_waypoints must be >= 0; use 0 for all history.")
if (
    not math.isfinite(args_cli.local_navigation_rate_hz)
    or args_cli.local_navigation_rate_hz <= 0.0
):
    parser.error("--local_navigation_rate_hz must be finite and > 0.")
if (
    not math.isfinite(args_cli.g3_session_timeout_s)
    or args_cli.g3_session_timeout_s <= 0.0
):
    parser.error("--g3_session_timeout_s must be finite and > 0.")
if (
    not math.isfinite(args_cli.g3_motion_window_s)
    or args_cli.g3_motion_window_s <= 0.0
):
    parser.error("--g3_motion_window_s must be finite and > 0.")
if args_cli.local_map_progress and not args_cli.local_use_isaac_odometry:
    parser.error("--local_map_progress requires --local_use_isaac_odometry.")
if args_cli.g3_session and not args_cli.local_map_progress:
    parser.error(
        "The phase-three G3 protocol requires --local_map_progress."
    )
if (
    not math.isfinite(args_cli.local_map_resolution_m)
    or args_cli.local_map_resolution_m <= 0.0
):
    parser.error("--local_map_resolution_m must be finite and > 0.")
if args_cli.local_map_depth_stride <= 0:
    parser.error("--local_map_depth_stride must be > 0.")
if not args_cli.locomotion_onnx.exists():
    parser.error(f"Locomotion ONNX does not exist: {args_cli.locomotion_onnx}")
if not args_cli.stand_onnx.exists():
    parser.error(f"Stand ONNX does not exist: {args_cli.stand_onnx}")

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import unitree_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

from goal_tracking.camera import (  # noqa: E402
    configure_four_rgbd_cameras,
    set_forward_rgbd_camera_viewport,
)
from goal_tracking.config import (  # noqa: E402
    DEFAULT_HOUSE_USD,
    DEFAULT_LOCOMOTION_TASK,
    STAND_ACTION_JOINT_NAMES,
    STAND_OBS_JOINT_NAMES,
    configure_locomotion_commands,
    configure_scene_probe,
    disable_auto_reset_terms,
    disable_observation_corruption,
)
from goal_tracking.control import (  # noqa: E402
    PolicySwitchState,
    StandObservationHistory,
    SwitchCommandController,
    get_policy_obs,
    print_native_stack_diagnostics,
    resolve_joint_ids,
    set_velocity_command,
)
from goal_tracking.path import remove_bedroom_wardrobe_doors_from_stage  # noqa: E402

from unified_vln.episode import EpisodeConfig, LocalEndToEndEpisode  # noqa: E402
from unified_vln.control_schedule import FixedRateUpdateSchedule  # noqa: E402
from unified_vln.iplanner_client import IPlannerClient  # noqa: E402
from unified_vln.isaac_backend import (  # noqa: E402
    IsaacLocalFourViewCamera,
    IsaacRootOdometryProvider,
)
from unified_vln.local_trajectory import LocalFollowerConfig  # noqa: E402
from unified_vln.map_progress import (  # noqa: E402
    SparseEpisodeExplorationMap,
    SparseMapConfig,
)
from unified_vln.model_client import CombinedModelClient  # noqa: E402
from unified_vln.session_client import G3SessionClient  # noqa: E402


def _stand_ready(switch_state: PolicySwitchState) -> bool:
    return (
        switch_state.active_mode == "stand"
        and switch_state.transition_mode == "none"
    )


def _locomotion_ready(switch_state: PolicySwitchState) -> bool:
    return (
        switch_state.active_mode == "locomotion"
        and switch_state.transition_mode == "none"
    )


def _request_mode(switch_state: PolicySwitchState, desired_mode: str) -> None:
    if desired_mode == "locomotion":
        already_requested = (
            switch_state.destination_mode == "locomotion"
            or _locomotion_ready(switch_state)
        )
        if not already_requested:
            switch_state.request_locomotion()
        return
    already_requested = (
        switch_state.destination_mode == "stand"
        or _stand_ready(switch_state)
    )
    if not already_requested:
        switch_state.request_stand()


def run_local_switch(env, env_cfg, agent_cfg) -> None:
    providers = ["CPUExecutionProvider"]
    locomotion_session = ort.InferenceSession(
        str(args_cli.locomotion_onnx), providers=providers
    )
    stand_session = ort.InferenceSession(
        str(args_cli.stand_onnx), providers=providers
    )
    locomotion_input = locomotion_session.get_inputs()[0].name
    locomotion_output = locomotion_session.get_outputs()[0].name
    stand_input = stand_session.get_inputs()[0].name
    stand_output = stand_session.get_outputs()[0].name
    locomotion_input_dim = int(locomotion_session.get_inputs()[0].shape[-1])
    stand_input_dim = int(stand_session.get_inputs()[0].shape[-1])

    raw_env = env.unwrapped
    step_dt = float(raw_env.step_dt)
    robot = raw_env.scene["robot"]
    stand_action_joint_ids = resolve_joint_ids(robot, STAND_ACTION_JOINT_NAMES)
    stand_obs_joint_ids = resolve_joint_ids(robot, STAND_OBS_JOINT_NAMES)
    switch_state = PolicySwitchState(raw_env, stand_action_joint_ids, args_cli)
    stand_history = StandObservationHistory(raw_env, stand_obs_joint_ids)
    command_controller = SwitchCommandController(raw_env, env_cfg, args_cli)
    command_controller.set_initial(0.0, 0.0, 0.0)

    last_stand_action = torch.zeros(
        (raw_env.num_envs, len(stand_action_joint_ids)),
        device=raw_env.device,
        dtype=torch.float32,
    )
    set_velocity_command(raw_env, torch.zeros_like(command_controller.filtered))
    obs = env.get_observations()
    obs_tensor = get_policy_obs(obs)
    if obs_tensor.shape[-1] != locomotion_input_dim:
        raise RuntimeError(
            f"Locomotion ONNX expects {locomotion_input_dim}, got {obs_tensor.shape[-1]}."
        )
    stand_obs = stand_history.reset(last_stand_action)
    if stand_obs.shape[-1] != stand_input_dim:
        raise RuntimeError(
            f"Stand ONNX expects {stand_input_dim}, got {stand_obs.shape[-1]}."
        )

    camera = IsaacLocalFourViewCamera(raw_env)
    odometry = (
        IsaacRootOdometryProvider(raw_env)
        if args_cli.local_use_isaac_odometry
        else None
    )
    exploration_map = None
    if args_cli.local_map_progress:
        exploration_map = SparseEpisodeExplorationMap(
            SparseMapConfig(
                resolution_m=args_cli.local_map_resolution_m,
                depth_stride=args_cli.local_map_depth_stride,
                depth_min_m=args_cli.rgbd_camera_near,
                depth_max_m=args_cli.rgbd_camera_far,
                camera_offset_x_m=args_cli.camera_rig_radius,
                camera_offset_y_m=0.0,
                camera_offset_z_m=args_cli.camera_rig_height,
                camera_yaw_rad=0.0,
                camera_down_tilt_rad=math.radians(args_cli.camera_down_tilt_deg),
                nominal_base_height_m=args_cli.local_map_nominal_base_height_m,
                floor_z_world_m=args_cli.local_map_floor_z_world_m,
                obstacle_min_height_m=args_cli.local_map_obstacle_min_height_m,
                obstacle_max_height_m=args_cli.local_map_obstacle_max_height_m,
                robot_radius_m=args_cli.local_map_robot_radius_m,
            ),
            pose_frame_id="isaac_world",
            frame_epoch=0,
        )
    print(
        "[LOCAL-VLN] odometry backend: "
        + ("Isaac root pose (explicit opt-in)" if odometry else "disabled")
    )
    print(
        "[LOCAL-VLN] map progress: "
        + (
            f"unbounded sparse {args_cli.local_map_resolution_m:.3f}m grid"
            if exploration_map is not None
            else "disabled"
        )
    )
    episode = LocalEndToEndEpisode(
        EpisodeConfig(
            session_id=args_cli.lavira_session_id,
            instruction=args_cli.instruction,
            warmup_steps=args_cli.lavira_decision_warmup_steps,
            history_max_waypoints=(
                None
                if args_cli.local_history_max_waypoints == 0
                else args_cli.local_history_max_waypoints
            ),
            max_decisions=(
                None
                if args_cli.local_max_decisions == 0
                else args_cli.local_max_decisions
            ),
            rotation_speed_rad_s=args_cli.local_rotation_speed_rad_s,
            rotation_duration_scale=args_cli.local_rotation_duration_scale_sim,
            rotation_settle_s=args_cli.local_rotation_settle_s,
            post_action_stand_s=args_cli.local_post_action_stand_s,
            safe_distance_m=args_cli.local_safe_distance_m,
            min_depth_m=args_cli.rgbd_camera_near,
            max_depth_m=args_cli.rgbd_camera_far,
            action_timeout_s=args_cli.local_action_timeout_s,
            motion_window_s=args_cli.g3_motion_window_s,
            pose_frame_id="isaac_world",
            frame_epoch=0,
            single_forward_panorama=args_cli.local_single_camera_panorama,
            enable_backtrack=True,
            backtrack_max_path_m=args_cli.local_backtrack_max_path_m,
            backtrack_start_tolerance_m=(
                args_cli.local_backtrack_start_tolerance_m
            ),
            backtrack_segment_length_m=(
                args_cli.local_backtrack_segment_length_m
            ),
            backtrack_goal_tolerance_m=(
                args_cli.local_backtrack_goal_tolerance_m
            ),
            backtrack_heading_tolerance_rad=(
                args_cli.local_backtrack_heading_tolerance_rad
            ),
            backtrack_breadcrumb_spacing_m=(
                args_cli.local_backtrack_breadcrumb_spacing_m
            ),
            output_dir=args_cli.local_output_dir,
        ),
        LocalFollowerConfig(
            tracking_controller=args_cli.local_tracking_controller,
            target_speed_m_s=args_cli.local_walk_speed_m_s,
            lookahead_m=args_cli.local_lookahead_m,
            max_forward_speed_m_s=args_cli.local_max_forward_speed_m_s,
            max_yaw_speed_rad_s=args_cli.local_max_yaw_speed_rad_s,
            goal_tolerance_m=args_cli.local_goal_tolerance_m,
            blind_yaw_radius_m=args_cli.local_blind_yaw_radius_m,
            kp_xy=args_cli.local_kp_xy,
            kp_yaw=args_cli.local_kp_yaw,
            kp_slow_radius_m=args_cli.local_kp_slow_radius_m,
            kp_yaw_deadband_rad=math.radians(
                args_cli.local_kp_yaw_deadband_deg
            ),
            kp_max_lateral_speed_m_s=(
                args_cli.local_kp_max_lateral_speed_m_s
            ),
            kp_max_yaw_speed_rad_s=(
                args_cli.local_kp_max_yaw_speed_rad_s
            ),
            yaw_bias_rad_s=args_cli.local_yaw_bias_rad_s,
            replan_interval_s=args_cli.local_replan_interval_s,
            dead_reckoning_linear_scale=(
                args_cli.local_dead_reckoning_linear_scale_sim
            ),
            dead_reckoning_angular_scale=(
                args_cli.local_dead_reckoning_angular_scale_sim
            ),
        ),
        camera=camera,
        model=CombinedModelClient(
            args_cli.lavira_server_url,
            args_cli.lavira_timeout,
            send_instruction=not args_cli.g3_session,
        ),
        planner=IPlannerClient(
            args_cli.iplanner_url, args_cli.iplanner_timeout_s
        ),
        session_client=(
            G3SessionClient.from_decision_url(
                args_cli.lavira_server_url,
                args_cli.g3_session_timeout_s,
            )
            if args_cli.g3_session
            else None
        ),
        odometry=odometry,
        yaw_provider=odometry,
        exploration_map=exploration_map,
    )

    print(
        "[LOCAL-VLN] runner ready: "
        f"step_dt={step_dt:.4f}s rotation_scale_sim="
        f"{args_cli.local_rotation_duration_scale_sim:.3f} "
        f"dead_reckoning=({args_cli.local_dead_reckoning_linear_scale_sim:.3f},"
        f"{args_cli.local_dead_reckoning_angular_scale_sim:.3f}) "
        f"tracking_controller={args_cli.local_tracking_controller} "
        f"goal_tolerance={args_cli.local_goal_tolerance_m:.3f}m "
        f"blind_yaw_radius={args_cli.local_blind_yaw_radius_m:.3f}m "
        f"panorama={'single-forward-sweep' if args_cli.local_single_camera_panorama else 'simultaneous-four-camera'} "
        f"navigation_rate={args_cli.local_navigation_rate_hz:.1f}Hz "
        f"policy_rate={1.0 / step_dt:.1f}Hz command_filter=direct-hold "
        "history_limit="
        + (
            "all"
            if args_cli.local_history_max_waypoints == 0
            else str(args_cli.local_history_max_waypoints)
        )
        + " decision_limit="
        + (
            "unlimited"
            if args_cli.local_max_decisions == 0
            else str(args_cli.local_max_decisions)
        )
    )
    navigation_schedule = FixedRateUpdateSchedule(
        args_cli.local_navigation_rate_hz,
        first_dt_s=step_dt,
    )
    # Pure Pursuit updates ``held_command`` at 20 Hz.  The locomotion policy
    # consumes that same command at 50 Hz, matching Uni's 20 Hz controller plus
    # 50 Hz DDS resend thread.
    held_command = np.zeros(3, dtype=np.float64)
    held_desired_mode = "stand"
    last_applied_command = np.zeros(3, dtype=np.float64)
    step = 0
    try:
        episode.start_remote_session()
        while simulation_app.is_running() and (
            args_cli.max_steps < 0 or step < args_cli.max_steps
        ):
            wall_start = time.time()
            timestamp = float(step) * step_dt
            navigation_dt = navigation_schedule.due_dt(timestamp)
            if navigation_dt is not None:
                update = episode.update(
                    completed_step=step,
                    step_dt=navigation_dt,
                    timestamp=timestamp,
                    applied_command=last_applied_command,
                    stand_ready=_stand_ready(switch_state),
                    locomotion_ready=_locomotion_ready(switch_state),
                )
                held_command = update.command.copy()
                held_desired_mode = update.desired_mode

            command_controller.set_requested(*held_command.tolist())
            if held_desired_mode == "stand":
                command_controller.zero()
            _request_mode(switch_state, held_desired_mode)

            # Uni has no extra Python-side command ramp: publish the new target
            # directly, then hold it until the next 20 Hz navigation update.
            command_for_env = command_controller.update_direct(
                switch_state.should_zero_command()
            )
            switch_state.update_waiting_for_stand(command_for_env, step_dt)
            set_velocity_command(raw_env, command_for_env)

            with torch.inference_mode():
                locomotion_obs = (
                    get_policy_obs(obs).detach().cpu().numpy().astype(np.float32)
                )
                locomotion_action_np = locomotion_session.run(
                    [locomotion_output], {locomotion_input: locomotion_obs}
                )[0].astype(np.float32)
                locomotion_action = torch.from_numpy(locomotion_action_np).to(
                    device=raw_env.device
                )

                stand_obs = stand_history.append(last_stand_action)
                stand_action_np = stand_session.run(
                    [stand_output],
                    {stand_input: stand_obs.detach().cpu().numpy().astype(np.float32)},
                )[0].astype(np.float32)
                stand_action = torch.from_numpy(stand_action_np).to(
                    device=raw_env.device
                )
                last_stand_action = stand_action
                actions = switch_state.action(
                    stand_action, locomotion_action, step_dt
                )
                obs, _, _, _ = env.step(actions)

            last_applied_command = command_for_env[0].detach().cpu().numpy().copy()
            step += 1
            if episode.completed and _stand_ready(switch_state):
                break
            if args_cli.real_time:
                remaining = step_dt - (time.time() - wall_start)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        command_controller.zero()
        set_velocity_command(raw_env, torch.zeros_like(command_controller.filtered))
        if episode.remote_session_active:
            if episode.failure_reason is not None:
                final_status = "FAILURE"
                final_reason = (
                    episode.session_failure_reason or "episode_failed"
                )
            elif episode.completed:
                final_status = "SUCCESS"
                final_reason = episode.session_success_reason or "episode_stopped"
            else:
                final_status = "FAILURE"
                final_reason = "runner_terminated"
            try:
                episode.end_remote_session(
                    status=final_status,
                    reason=final_reason,
                )
            except Exception as exc:
                print(f"[LOCAL-VLN G3 WARN] end_session failed: {exc}")
    print(
        f"[LOCAL-VLN] finished: state={episode.state} "
        f"history={len(episode.history)} failure={episode.failure_reason!r}"
    )


def main() -> None:
    task = args_cli.task or DEFAULT_LOCOMOTION_TASK
    scene_usd = DEFAULT_HOUSE_USD if args_cli.house else args_cli.scene_usd
    env_cfg = parse_env_cfg(
        task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    disable_observation_corruption(env_cfg)
    configure_scene_probe(env_cfg, scene_usd, args_cli)
    configure_locomotion_commands(env_cfg, args_cli)
    disable_auto_reset_terms(env_cfg, args_cli)
    configure_four_rgbd_cameras(env_cfg, args_cli)

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    env = gym.make(task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        remove_bedroom_wardrobe_doors_from_stage(args_cli)
        if args_cli.four_rgbd_set_viewport and not args_cli.headless:
            set_forward_rgbd_camera_viewport(args_cli)
        print_native_stack_diagnostics(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        run_local_switch(env, env_cfg, agent_cfg)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
