#!/usr/bin/env python3
from __future__ import annotations

"""真实 G1 的统一 VLN runner 框架。

本文件只负责装配已经存在的导航状态机、DDS 和 ROS 2 odometry。真实四方向
RGB-D 相机因设备型号、序列号和标定尚未确定，通过 ``module:function`` 工厂注入，
不会在仓库里写死任何相机或网卡信息。
"""

import argparse
import importlib
import math
import os
from pathlib import Path
import signal
import time

import numpy as np

from unified_vln.episode import CameraBackend, EpisodeConfig, LocalEndToEndEpisode
from unified_vln.g1_dds_backend import UnitreeG1DDSBackend
from unified_vln.iplanner_client import IPlannerClient
from unified_vln.local_trajectory import LocalFollowerConfig
from unified_vln.model_client import CombinedModelClient
from unified_vln.ros2_odometry import Ros2OdometryProvider


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]


# 与 Uni-LaViRA G1 的 main.py 一样，SIGINT 处理器需要能够访问当前已经创建好的
# 真机资源。资源会在各自构造成功后立即登记，避免相机或 ROS 初始化期间按下
# Ctrl+C 时遗漏已经启动的 DDS 控制线程。
_active_dds: UnitreeG1DDSBackend | None = None
_active_camera: CameraBackend | None = None
_active_odometry: Ros2OdometryProvider | None = None


def _shutdown_active_resources() -> None:
    """按 Uni-LaViRA 的顺序尽力停止运动，再关闭所有已创建资源。"""

    global _active_dds, _active_camera, _active_odometry

    dds = _active_dds
    camera = _active_camera
    odometry = _active_odometry

    # 先清零并直接调用 StopMove；dds.close() 会在停止发送线程前再次停车，
    # 对应 Uni-LaViRA 的 ``stop_robot()`` 后再进入 ``shutdown()``。
    if dds is not None:
        try:
            dds.stop()
        except Exception:
            pass
        try:
            dds.close()
        except Exception:
            pass

    # 每项独立清理，某个相机/ROS close 失败不能阻止其余资源继续关闭。
    if camera is not None:
        try:
            _close_optional(camera)
        except Exception:
            pass
    if odometry is not None:
        try:
            odometry.close()
        except Exception:
            pass

    _active_dds = None
    _active_camera = None
    _active_odometry = None


def _signal_handler(_sig, _frame) -> None:
    """处理 Ctrl+C：尽力停车和关闭资源，随后像 Uni-LaViRA 一样强制退出。"""

    print("\n[LOCAL-VLN G1] Signal received, shutting down...", flush=True)
    _shutdown_active_resources()
    os._exit(0)


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and > 0")
    return result


def _non_negative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and >= 0")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the combined-model local VLN state machine on a real Unitree G1. "
            "Hardware identity/calibration values must be supplied explicitly."
        )
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model-timeout-s", type=_positive_float, default=90.0)
    parser.add_argument("--iplanner-url", required=True)
    parser.add_argument("--iplanner-timeout-s", type=_positive_float, default=5.0)

    parser.add_argument(
        "--network-interface",
        required=True,
        help="Host network interface connected to the G1, for example enp4s0.",
    )
    parser.add_argument(
        "--camera-factory",
        required=True,
        metavar="MODULE:FUNCTION",
        help=(
            "Import path of a factory that accepts the --camera-config Path and "
            "returns a CameraBackend."
        ),
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        required=True,
        help="Device-specific camera serial numbers, intrinsics and extrinsics.",
    )
    parser.add_argument(
        "--odometry-topic",
        default=None,
        help=(
            "ROS 2 nav_msgs/msg/Odometry topic. Omit only to use command-based "
            "dead reckoning."
        ),
    )
    parser.add_argument("--odometry-timeout-s", type=_positive_float, default=0.5)

    # 默认值直接采用 Uni-LaViRA G1 的真机经验比例，仍允许显式覆盖。
    parser.add_argument("--rotation-duration-scale", type=_positive_float, required=True)
    parser.add_argument(
        "--dead-reckoning-linear-scale", type=_positive_float, default=0.7
    )
    parser.add_argument(
        "--dead-reckoning-angular-scale", type=_positive_float, default=0.8
    )
    parser.add_argument("--rotation-speed-rad-s", type=_positive_float, default=0.4)
    parser.add_argument("--rotation-settle-s", type=_positive_float, default=0.5)
    parser.add_argument(
        "--use-imu-rotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use fresh DDS IMU yaw for closed-loop relative rotation. Disabled "
            "by default to match Uni-LaViRA G1; pass --use-imu-rotation to opt in."
        ),
    )
    parser.add_argument("--imu-timeout-s", type=_positive_float, default=1.0)
    parser.add_argument("--dds-command-rate-hz", type=_positive_float, default=50.0)
    parser.add_argument("--control-rate-hz", type=_positive_float, default=20.0)

    parser.add_argument("--walk-speed-m-s", type=_positive_float, default=0.3)
    parser.add_argument("--lookahead-m", type=_positive_float, default=0.5)
    parser.add_argument("--max-forward-speed-m-s", type=_positive_float, default=0.4)
    parser.add_argument("--max-yaw-speed-rad-s", type=_positive_float, default=0.5)
    parser.add_argument("--goal-tolerance-m", type=_positive_float, default=1.0)
    parser.add_argument("--blind-yaw-radius-m", type=_non_negative_float, default=2.0)
    parser.add_argument("--yaw-bias-rad-s", type=float, default=0.0)
    parser.add_argument("--replan-interval-s", type=_positive_float, default=0.1)
    parser.add_argument("--safe-distance-m", type=_non_negative_float, default=0.5)
    parser.add_argument("--min-depth-m", type=_positive_float, default=0.1)
    parser.add_argument("--max-depth-m", type=_positive_float, default=5.0)
    parser.add_argument("--action-timeout-s", type=_positive_float, default=60.0)
    parser.add_argument("--post-action-stand-s", type=_positive_float, default=0.8)
    parser.add_argument("--warmup-seconds", type=_non_negative_float, default=2.0)
    parser.add_argument(
        "--history-max-waypoints",
        type=int,
        default=0,
        help=(
            "0 (default) keeps all text history; positive values send only the "
            "most recent N completed waypoints."
        ),
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=0,
        help="0 (default) runs until model STOP; positive values are a safety cap.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "g1_real_unified_vln",
    )
    return parser


def _load_camera_backend(factory_spec: str, config_path: Path) -> CameraBackend:
    module_name, separator, factory_name = factory_spec.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("--camera-factory must use MODULE:FUNCTION syntax.")
    if not config_path.is_file():
        raise FileNotFoundError(f"Camera config does not exist: {config_path}")
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise TypeError(f"Camera factory is not callable: {factory_spec}")
    camera = factory(config_path)
    for method_name in ("capture_panorama", "capture_forward"):
        if not callable(getattr(camera, method_name, None)):
            raise TypeError(
                f"Camera backend from {factory_spec} lacks {method_name}()."
            )
    return camera


def _close_optional(resource) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def _apply_episode_command(
    dds: UnitreeG1DDSBackend,
    update,
    previous_mode: str,
) -> tuple[np.ndarray, str]:
    """下发一帧速度，并在 locomotion→stand 时立即执行一次 StopMove。"""

    desired_mode = str(update.desired_mode)
    if desired_mode == "locomotion":
        command = np.asarray(update.command, dtype=np.float64).reshape(3).copy()
        dds.set_velocity(*command.tolist())
    else:
        command = np.zeros(3, dtype=np.float64)
        if previous_mode == "locomotion":
            # Uni-LaViRA 的每段 execute_trajectory() 退出时都会调用
            # stop_robot()；这里只在模式边沿调用一次，避免站立等待期间反复阻塞。
            dds.stop()
        else:
            dds.set_velocity(0.0, 0.0, 0.0)
    return command, desired_mode


def run(args: argparse.Namespace) -> int:
    global _active_dds, _active_camera, _active_odometry

    if args.max_decisions < 0:
        raise ValueError("--max-decisions must be >= 0; use 0 for unlimited.")
    if args.history_max_waypoints < 0:
        raise ValueError("--history-max-waypoints must be >= 0; use 0 for all history.")
    if not args.network_interface.strip():
        raise ValueError("--network-interface must not be empty.")
    if args.min_depth_m >= args.max_depth_m:
        raise ValueError("Depth range must satisfy min < max.")

    session_id = args.session_id or time.strftime("g1_%Y%m%d_%H%M%S")
    camera = None
    odometry = None
    dds = None
    try:
        # 先建立 DDS 并保持零速度，再初始化可能耗时较长的相机和 ROS。
        dds = UnitreeG1DDSBackend(
            args.network_interface,
            imu_timeout_s=args.imu_timeout_s,
            command_rate_hz=args.dds_command_rate_hz,
        )
        _active_dds = dds
        dds.stop()
        camera = _load_camera_backend(args.camera_factory, args.camera_config)
        _active_camera = camera
        if args.odometry_topic is not None:
            odometry = Ros2OdometryProvider(
                topic=args.odometry_topic,
                pose_timeout_s=args.odometry_timeout_s,
            )
            _active_odometry = odometry

        control_period_s = 1.0 / args.control_rate_hz
        episode = LocalEndToEndEpisode(
            EpisodeConfig(
                session_id=session_id,
                instruction=args.instruction,
                warmup_steps=round(args.warmup_seconds * args.control_rate_hz),
                history_max_waypoints=(
                    None
                    if args.history_max_waypoints == 0
                    else args.history_max_waypoints
                ),
                max_decisions=(None if args.max_decisions == 0 else args.max_decisions),
                rotation_speed_rad_s=args.rotation_speed_rad_s,
                rotation_duration_scale=args.rotation_duration_scale,
                rotation_settle_s=args.rotation_settle_s,
                post_action_stand_s=args.post_action_stand_s,
                safe_distance_m=args.safe_distance_m,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                action_timeout_s=args.action_timeout_s,
                output_dir=args.output_dir,
            ),
            LocalFollowerConfig(
                target_speed_m_s=args.walk_speed_m_s,
                lookahead_m=args.lookahead_m,
                max_forward_speed_m_s=args.max_forward_speed_m_s,
                max_yaw_speed_rad_s=args.max_yaw_speed_rad_s,
                goal_tolerance_m=args.goal_tolerance_m,
                blind_yaw_radius_m=args.blind_yaw_radius_m,
                yaw_bias_rad_s=args.yaw_bias_rad_s,
                replan_interval_s=args.replan_interval_s,
                dead_reckoning_linear_scale=args.dead_reckoning_linear_scale,
                dead_reckoning_angular_scale=args.dead_reckoning_angular_scale,
            ),
            camera=camera,
            model=CombinedModelClient(args.model_url, args.model_timeout_s),
            planner=IPlannerClient(args.iplanner_url, args.iplanner_timeout_s),
            odometry=odometry,
            yaw_provider=dds if args.use_imu_rotation else None,
        )

        # 对齐 Uni-LaViRA 真机入口：所有后端完成初始化后、任务开始前，只调用
        # 一次 HighStand。后续导航仍始终使用同一个 LocoClient 高层速度控制器。
        dds.high_stand()
        print("[LOCAL-VLN G1] G1 high-stand request completed; navigation may start.")

        print(
            "[LOCAL-VLN G1] runner framework ready: "
            f"session={session_id!r} odometry={args.odometry_topic or 'disabled'} "
            f"history_limit={'all' if args.history_max_waypoints == 0 else args.history_max_waypoints} "
            f"decision_limit={'unlimited' if args.max_decisions == 0 else args.max_decisions}"
        )
        if odometry is None:
            print(
                "[LOCAL-VLN G1 WARN] Using Uni-LaViRA-style dead reckoning; "
                "planner/model blocking time is covered by the repeated last command."
            )

        step = 0
        last_tick = time.monotonic()
        started_at = last_tick
        last_applied_command = np.zeros(3, dtype=np.float64)
        previous_mode = "stand"
        while not episode.completed:
            loop_started = time.monotonic()
            step_dt = max(loop_started - last_tick, 1e-6)
            last_tick = loop_started
            update = episode.update(
                completed_step=step,
                step_dt=step_dt,
                timestamp=loop_started - started_at,
                applied_command=last_applied_command,
                # G1 LocoClient 本身就是高层速度接口；真实的额外 FSM/模式确认
                # 若后续需要，可在这里替换成专用 mode adapter。
                stand_ready=True,
                locomotion_ready=True,
            )
            command, previous_mode = _apply_episode_command(
                dds,
                update,
                previous_mode,
            )
            last_applied_command = command.copy()
            step += 1

            elapsed = time.monotonic() - loop_started
            if elapsed < control_period_s:
                time.sleep(control_period_s - elapsed)

        print(
            f"[LOCAL-VLN G1] finished: state={episode.state} "
            f"history={len(episode.history)} failure={episode.failure_reason!r}"
        )
        return 0 if episode.failure_reason is None else 1
    finally:
        _shutdown_active_resources()


def main() -> int:
    # 与 Uni-LaViRA 一致：在解析参数和加载真机资源之前尽早注册 Ctrl+C。
    signal.signal(signal.SIGINT, _signal_handler)
    args = build_parser().parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        # 显式 SIGINT handler 通常会直接 os._exit(0)；保留该分支作为无法安装
        # handler 或测试直接抛出 KeyboardInterrupt 时的兜底。
        print("\n[LOCAL-VLN G1] Interrupted by user.")
        _shutdown_active_resources()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
