#!/usr/bin/env python3
"""Run G1 policies, path following, four-view RGB-D, and bounded LaViRA navigation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROJECTS_DIR = PROJECT_DIR.parent
UNITREE_RL_LAB_DIR = PROJECTS_DIR / "unitree_rl_lab"

sys.path.insert(0, str(UNITREE_RL_LAB_DIR / "source" / "unitree_rl_lab"))

from isaaclab.app import AppLauncher  # noqa: E402

from goal_tracking.config import (  # noqa: E402
    DEFAULT_HOUSE_USD,
    DEFAULT_LOCOMOTION_TASK,
    DEFAULT_STAND_TASK,
    build_parser,
)


parser = build_parser(AppLauncher)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if args_cli.camera_debug_save_once and not args_cli.four_rgbd_cameras:
    parser.error("--camera_debug_save_once requires --four_rgbd_cameras.")
if args_cli.camera_debug_warmup_steps < 0:
    parser.error("--camera_debug_warmup_steps must be non-negative.")
if args_cli.lavira_history_probe and args_cli.lavira_decision_probe:
    parser.error(
        "--lavira_history_probe and --lavira_decision_probe are separate modes; "
        "choose exactly one."
    )
if args_cli.lavira_history_probe:
    # The bounded episode is one explicit end-to-end mode. It reuses the same
    # projection/map/FMM implementation and safety limits as the one-shot mode
    # without requiring four redundant feature flags.
    args_cli.lavira_local_map_probe = True
    args_cli.lavira_fmm_probe = True
    args_cli.lavira_execute_fmm_path = True
    # A marker drawn after decision 0 would become visible in decision 1 RGB and
    # contaminate the model input. Keep all episode captures scene-authentic.
    args_cli.lavira_projection_debug_marker = False
lavira_decision_enabled = bool(
    args_cli.lavira_decision_probe or args_cli.lavira_history_probe
)
if lavira_decision_enabled:
    if args_cli.mode != "switch":
        parser.error("LaViRA decision modes currently require --mode switch.")
    if not args_cli.four_rgbd_cameras:
        parser.error("LaViRA decision modes require --four_rgbd_cameras.")
    if not args_cli.instruction.strip():
        parser.error("LaViRA decision modes require a non-empty --instruction.")
    if args_cli.lavira_decision_warmup_steps < 0:
        parser.error("--lavira_decision_warmup_steps must be non-negative.")
    if args_cli.lavira_timeout <= 0.0:
        parser.error("--lavira_timeout must be positive.")
    if args_cli.four_rgbd_debug_points:
        parser.error("Disable --four_rgbd_debug_points for clean LaViRA RGB input.")
    if args_cli.show_path:
        parser.error("LaViRA decision modes require --no-show_path for clean RGB input.")
    if args_cli.start_path_on_enter:
        parser.error("LaViRA decision modes cannot use --start_path_on_enter.")
if args_cli.lavira_history_probe:
    if args_cli.lavira_history_settle_seconds < 0.0:
        parser.error("--lavira_history_settle_seconds must be non-negative.")
    if args_cli.lavira_history_max_decisions < 2:
        parser.error("--lavira_history_max_decisions must be at least 2.")
    if args_cli.lavira_history_execution_timeout <= 0.0:
        parser.error("--lavira_history_execution_timeout must be positive.")
    if args_cli.lavira_stop_reached_threshold_m <= 0.0:
        parser.error("--lavira_stop_reached_threshold_m must be positive.")
    if args_cli.lavira_backtrack_max_path_m <= 0.0:
        parser.error("--lavira_backtrack_max_path_m must be positive.")
if args_cli.lavira_local_map_probe and not lavira_decision_enabled:
    parser.error("--lavira_local_map_probe requires a LaViRA decision mode.")
if args_cli.lavira_local_map_probe:
    if args_cli.nav_map_resolution_m <= 0.0 or args_cli.nav_map_size_m <= 0.0:
        parser.error("Navigation map resolution and size must be positive.")
    if args_cli.nav_depth_stride <= 0:
        parser.error("--nav_depth_stride must be positive.")
    if not 0.0 <= args_cli.nav_obstacle_min_height_m < args_cli.nav_obstacle_max_height_m:
        parser.error("Navigation obstacle height range is invalid.")
    if args_cli.nav_robot_radius_m < 0.0:
        parser.error("--nav_robot_radius_m must be non-negative.")
    if args_cli.nav_target_retreat_step_m <= 0.0:
        parser.error("--nav_target_retreat_step_m must be positive.")
    if args_cli.nav_target_snap_max_m < 0.0:
        parser.error("--nav_target_snap_max_m must be non-negative.")
if args_cli.lavira_fmm_probe and not args_cli.lavira_local_map_probe:
    parser.error("--lavira_fmm_probe requires --lavira_local_map_probe.")
if args_cli.lavira_fmm_probe:
    if args_cli.fmm_step_size_cells <= 0:
        parser.error("--fmm_step_size_cells must be positive.")
    if args_cli.fmm_goal_tolerance_cells < 0:
        parser.error("--fmm_goal_tolerance_cells must be non-negative.")
    if args_cli.fmm_waypoint_spacing_m <= 0.0:
        parser.error("--fmm_waypoint_spacing_m must be positive.")
    if args_cli.fmm_max_path_steps <= 0:
        parser.error("--fmm_max_path_steps must be positive.")
if args_cli.lavira_execute_fmm_path and not args_cli.lavira_fmm_probe:
    parser.error("--lavira_execute_fmm_path requires --lavira_fmm_probe.")
if args_cli.lavira_execute_fmm_path:
    if args_cli.path_waypoints is not None:
        parser.error("--lavira_execute_fmm_path cannot use --path_waypoints.")
    if args_cli.start_path_on_enter:
        parser.error("--lavira_execute_fmm_path cannot use --start_path_on_enter.")
    if max(abs(args_cli.vx), abs(args_cli.vy), abs(args_cli.wz)) > 1.0e-6:
        parser.error("FMM execution requires zero initial --vx/--vy/--wz.")
    positive_execution_values = (
        args_cli.fmm_execute_start_tolerance_m,
        args_cli.fmm_execute_max_path_m,
        args_cli.fmm_execute_cross_track_abort_m,
        args_cli.fmm_execute_tilt_abort_rad,
        args_cli.fmm_execute_lookahead_m,
        args_cli.fmm_execute_max_vx,
        args_cli.fmm_execute_max_vy,
        args_cli.fmm_execute_max_wz,
    )
    if any(value <= 0.0 for value in positive_execution_values):
        parser.error("All FMM execution safety and velocity parameters must be positive.")

# RTX camera rendering must be enabled before AppLauncher creates Isaac Sim.
# Set it automatically for the four-view sensor mode so GUI/headless commands do
# not need a second, easy-to-forget --enable_cameras flag.
if args_cli.four_rgbd_cameras:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import unitree_rl_lab.tasks  # noqa: F401,E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

from goal_tracking.config import (  # noqa: E402
    configure_locomotion_commands,
    configure_scene_probe,
    disable_auto_reset_terms,
    disable_observation_corruption,
)
from goal_tracking.camera import (  # noqa: E402
    configure_four_rgbd_cameras,
    draw_four_rgbd_camera_debug_points,
    set_forward_rgbd_camera_viewport,
)
from goal_tracking.control import print_native_stack_diagnostics  # noqa: E402
from goal_tracking.path import remove_bedroom_wardrobe_doors_from_stage  # noqa: E402
from goal_tracking.runners import run_locomotion, run_stand, run_switch  # noqa: E402


def main() -> None:
    task = args_cli.task
    if task is None:
        task = DEFAULT_STAND_TASK if args_cli.mode == "stand" else DEFAULT_LOCOMOTION_TASK
    if args_cli.mode == "stand" and not args_cli.checkpoint.exists():
        raise FileNotFoundError(args_cli.checkpoint)
    if args_cli.mode in ("locomotion", "switch") and not args_cli.locomotion_onnx.exists():
        raise FileNotFoundError(args_cli.locomotion_onnx)
    if args_cli.mode == "switch" and not args_cli.stand_onnx.exists():
        raise FileNotFoundError(args_cli.stand_onnx)

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
    if args_cli.mode in ("locomotion", "switch"):
        configure_locomotion_commands(env_cfg, args_cli)
    disable_auto_reset_terms(env_cfg, args_cli)
    configure_four_rgbd_cameras(env_cfg, args_cli)
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    print(f"[INFO] Mode: {args_cli.mode}")
    print(f"[INFO] Task: {task}")
    if args_cli.mode == "stand":
        print(f"[INFO] Checkpoint: {args_cli.checkpoint}")
    else:
        print(f"[INFO] Locomotion ONNX: {args_cli.locomotion_onnx}")
        if args_cli.mode == "switch":
            print(f"[INFO] Stand ONNX: {args_cli.stand_onnx}")
    print("[INFO] Creating native IsaacLab environment.")
    env = gym.make(task, cfg=env_cfg)
    try:
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        remove_bedroom_wardrobe_doors_from_stage(args_cli)
        if args_cli.four_rgbd_cameras:
            draw_four_rgbd_camera_debug_points(args_cli)
            if not args_cli.four_rgbd_debug_points:
                set_forward_rgbd_camera_viewport(args_cli)

        print_native_stack_diagnostics(env)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if args_cli.mode == "stand":
            run_stand(env, agent_cfg, args_cli, simulation_app)
        elif args_cli.mode == "locomotion":
            run_locomotion(env, env_cfg, args_cli, simulation_app)
        else:
            run_switch(env, env_cfg, args_cli, simulation_app)
    finally:
        # The entry point owns the environment lifecycle. This also covers
        # policy/session/camera initialization errors before a runner loop starts.
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
