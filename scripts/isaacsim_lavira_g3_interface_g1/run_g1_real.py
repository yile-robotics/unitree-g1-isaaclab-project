#!/usr/bin/env python3
# 中文导读：
# 真机阅读主线：main → run → 初始化 DDS/相机/SLAM → start_remote_session → update 循环 → 清理。
# 本文件负责装配组件和发送速度，模型角色在远端服务运行，关节控制由 G1 运动服务完成。
# 默认 G3 模式需要有效 SLAM 位姿和探索地图标定；参数由命令行读取，不自动加载 config.yaml。

from __future__ import annotations

"""真实 G1 的统一 VLN runner 框架。

本文件只负责装配已经存在的导航状态机、DDS 和 ROS 2 odometry。真实前向
RGB-D 相机因设备型号、序列号和标定尚未确定，通过 ``module:function`` 工厂注入；
四方向全景由机器人连续旋转并重复读取同一个前向相机得到。
"""

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np

from unified_vln.episode import CameraBackend, EpisodeConfig
from unified_vln.real_episode import RealG1Episode, RealPanoramaConfig
from unified_vln.real_panorama_imu import LowStateYaw
from unified_vln.g1_dds_backend import UnitreeG1DDSBackend
from unified_vln.iplanner_client import IPlannerClient
from unified_vln.local_trajectory import LocalFollowerConfig
from unified_vln.model_client import CombinedModelClient
from unified_vln.map_progress import SparseEpisodeExplorationMap, SparseMapConfig
from unified_vln.ros2_odometry import Ros2OdometryProvider
from unified_vln.session_client import G3SessionClient


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]


# 与 Uni-LaViRA G1 的 main.py 一样，SIGINT 处理器需要能够访问当前已经创建好的
# 真机资源。资源会在各自构造成功后立即登记，避免相机或 ROS 初始化期间按下
# Ctrl+C 时遗漏已经启动的 DDS 控制线程。
_active_dds: UnitreeG1DDSBackend | None = None
_active_camera: CameraBackend | None = None
_active_odometry: Ros2OdometryProvider | None = None
_active_episode: RealG1Episode | None = None
_active_panorama_imu: LowStateYaw | None = None


# 资源清理顺序很重要：先停止机器人再尝试远端 HTTP，避免等待网络时仍在运动。
# 只有 stop_confirmed 才算语义任务成功；达到决策上限只是停止实验。
def _shutdown_active_resources() -> None:
    """按 Uni-LaViRA 的顺序尽力停止运动，再关闭所有已创建资源。"""

    global _active_dds, _active_camera, _active_odometry, _active_episode, _active_panorama_imu

    dds = _active_dds
    camera = _active_camera
    odometry = _active_odometry
    episode = _active_episode
    _active_episode = None

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
    if _active_panorama_imu is not None:
        try:
            _active_panorama_imu.close()
        except Exception:
            pass
        _active_panorama_imu = None
    if odometry is not None:
        try:
            odometry.close()
        except Exception:
            pass

    _active_dds = None
    _active_camera = None
    _active_odometry = None

    # HTTP 清理可能耗时，必须在机器人和传感器已停止后执行。
    if episode is not None and episode.remote_session_active:
        success = (
            episode.failure_reason is None
            and episode.session_success_reason == "stop_confirmed"
        )
        reason = (
            "stop_confirmed" if success else
            episode.session_failure_reason or episode.failure_reason or
            ("decision_limit_reached" if episode.completed else "runner_terminated")
        )
        try:
            episode.end_remote_session(
                status="SUCCESS" if success else "FAILURE", reason=reason
            )
        except Exception as exc:
            print(f"[LOCAL-VLN G3 WARN] end_session failed: {exc}", flush=True)


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
    parser.add_argument("--g3-session", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--g3-session-timeout-s", type=_positive_float, default=180.0)
    parser.add_argument("--g3-motion-window-s", type=_positive_float, default=1.0)
    parser.add_argument(
        "--map-config", type=Path,
        help="JSON SparseMapConfig with explicit real-camera extrinsics and base/floor heights; required for G3.",
    )
    parser.add_argument("--odometry-startup-timeout-s", type=_positive_float, default=10.0)
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
            "SLAM world-frame base pose (nav_msgs/msg/Odometry); provides both "
            "position and rotation yaw. Omit only for legacy dead-reckoning diagnostics."
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
    parser.add_argument("--panorama-speed-rad-s", type=_positive_float, default=0.8,
                        help="Real four-quarter capture speed; separate from navigation turns.")
    parser.add_argument("--panorama-imu-stop-deg", type=_positive_float, default=75.0,
                        help="IMU early-stop threshold per panorama quarter; default 75 degrees.")
    parser.add_argument("--panorama-quarter-timeout-s", type=_positive_float, default=10.0)
    parser.add_argument(
        "--use-imu-rotation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compatibility flag for navigation turns; with SLAM configured they use SLAM. "
            "Real panorama always uses its dedicated rt/lowstate IMU."
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
    parser.add_argument(
        "--tracking-controller",
        choices=("pure_pursuit", "kp"),
        default="pure_pursuit",
        help=(
            "Select the isolated local trajectory controller. The default "
            "preserves the existing Pure-Pursuit behavior; kp uses fixed-frame "
            "path projection with bounded vx/vy/wz commands."
        ),
    )
    parser.add_argument("--blind-yaw-radius-m", type=_non_negative_float, default=1.5)
    parser.add_argument("--kp-xy", type=_positive_float, default=0.7)
    parser.add_argument("--kp-yaw", type=_positive_float, default=1.0)
    parser.add_argument("--kp-slow-radius-m", type=_positive_float, default=1.0)
    parser.add_argument(
        "--kp-yaw-deadband-deg", type=_non_negative_float, default=10.0
    )
    parser.add_argument(
        "--kp-max-lateral-speed-m-s", type=_positive_float, default=0.12
    )
    parser.add_argument(
        "--kp-max-yaw-speed-rad-s", type=_positive_float, default=0.35
    )
    parser.add_argument("--yaw-bias-rad-s", type=float, default=0.0)
    parser.add_argument("--replan-interval-s", type=_positive_float, default=0.1)
    parser.add_argument("--safe-distance-m", type=_non_negative_float, default=0.5)
    parser.add_argument("--min-depth-m", type=_positive_float, default=0.1)
    parser.add_argument("--max-depth-m", type=_positive_float, default=5.0)
    parser.add_argument("--action-timeout-s", type=_positive_float, default=60.0)
    parser.add_argument("--post-action-stand-s", type=_positive_float, default=0.8)
    parser.add_argument("--backtrack-max-path-m", type=_positive_float, default=6.0)
    parser.add_argument(
        "--backtrack-start-tolerance-m", type=_non_negative_float, default=1.0
    )
    parser.add_argument(
        "--backtrack-segment-length-m", type=_positive_float, default=1.0
    )
    parser.add_argument(
        "--backtrack-goal-tolerance-m", type=_positive_float, default=0.35
    )
    parser.add_argument(
        "--backtrack-heading-tolerance-rad", type=_positive_float, default=0.20
    )
    parser.add_argument(
        "--backtrack-breadcrumb-spacing-m", type=_positive_float, default=0.15
    )
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


# MODULE:FUNCTION 是用户相机工厂入口；这里只验证接口存在，尚未证明能读取有效第一帧。
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
    for method_name in ("capture_forward",):
        if not callable(getattr(camera, method_name, None)):
            raise TypeError(
                f"Camera backend from {factory_spec} lacks {method_name}()."
            )
    return camera


def _close_optional(resource) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


# 必须显式给出真机安装几何，避免默默使用仿真尺寸；相机内参 K 则从 ViewFrame 获取。
def _load_map_config(path: Path) -> SparseMapConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "camera_offset_x_m", "camera_offset_y_m", "camera_offset_z_m",
        "camera_yaw_rad", "camera_down_tilt_rad", "nominal_base_height_m",
        "floor_z_world_m",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("--map-config must explicitly provide: " + ", ".join(sorted(required)))
    return SparseMapConfig(**payload).validated()


# 先等到有效位姿且 frame_id 非空；默认等待上限与位姿过期时间是两个不同参数。
def _wait_for_odometry(odometry, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    while True:
        if odometry.get_pose() is not None and odometry.frame_id.strip():
            return odometry.frame_id
        if time.monotonic() >= deadline:
            raise RuntimeError("G3 requires fresh SLAM odometry with a non-empty frame_id before navigation.")
        time.sleep(0.05)


class _SlamPoseSource:
    """Use the same measured world pose for translation and rotation, without fallback."""

    def __init__(self, provider, frame_id: str):
        self.provider = provider
        self.frame_id = frame_id

    # 包装器把缺失位姿提升为异常，阻止跟随器把 None 当成可使用航位推算的情况。
    # frame_id 只检查名称变化，不能检测同名坐标系内部的重定位跳变。
    def get_pose(self):
        pose = self.provider.get_pose()
        if pose is None:
            raise RuntimeError("SLAM world pose unavailable or stale; stopping navigation.")
        if self.provider.frame_id != self.frame_id:
            raise RuntimeError("SLAM frame_id changed; restart the episode in the new frame.")
        return pose.validated()

    # yaw 直接取同一份 SLAM 位姿，避免把 SLAM 位置与 DDS IMU 朝向混用。
    def get_yaw(self) -> float:
        return self.get_pose().yaw


# locomotion 表示允许发送高层速度，不代表每轮切换机器人内部策略。
# 从运动转入 stand 时主动调用 StopMove；等待期间仍保持零速度。
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


# 入口分为参数校验、硬件装配、会话启动和控制循环四段；finally 覆盖初始化中途失败。
# 本地模块的 import 也可能依赖相邻工程，应先确认 SDK、ROS 与 iPlanner 客户端可导入。
def run(args: argparse.Namespace) -> int:
    global _active_dds, _active_camera, _active_odometry, _active_episode, _active_panorama_imu

    if args.max_decisions < 0:
        raise ValueError("--max-decisions must be >= 0; use 0 for unlimited.")
    if args.history_max_waypoints < 0:
        raise ValueError("--history-max-waypoints must be >= 0; use 0 for all history.")
    if not args.network_interface.strip():
        raise ValueError("--network-interface must not be empty.")
    if args.min_depth_m >= args.max_depth_m:
        raise ValueError("Depth range must satisfy min < max.")
    map_config = None
    session_client = None
    # G3 默认开启：先验证本地证据来源，再构造远端客户端，此时尚未发送 HTTP。
    if args.g3_session:
        if not args.odometry_topic or not args.odometry_topic.strip() or args.map_config is None:
            raise ValueError("G3 requires --odometry-topic and --map-config; use --no-g3-session only for legacy diagnostics.")
        map_config = _load_map_config(args.map_config)
        session_client = G3SessionClient.from_decision_url(
            args.model_url, args.g3_session_timeout_s
        )
    if not args.odometry_topic or not args.odometry_topic.strip():
        raise ValueError("Real panorama requires --odometry-topic for SLAM checks, including legacy model mode.")
    panorama_config = RealPanoramaConfig(
        speed_rad_s=args.panorama_speed_rad_s, imu_stop_deg=args.panorama_imu_stop_deg,
        settle_s=args.rotation_settle_s, quarter_timeout_s=args.panorama_quarter_timeout_s,
    ).validated()

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
        _active_panorama_imu = LowStateYaw(timeout_s=0.5)
        camera = _load_camera_backend(args.camera_factory, args.camera_config)
        _active_camera = camera
        if args.odometry_topic is not None:
            odometry = Ros2OdometryProvider(
                topic=args.odometry_topic,
                pose_timeout_s=args.odometry_timeout_s,
                preserve_world_coordinates=True,
            )
            _active_odometry = odometry

        exploration_map = None
        pose_frame_id = "local_odom"
        pose_source = None
        # 世界位置与导航转向使用SLAM，全景提前停车另用rt/lowstate IMU。
        if odometry is not None:
            pose_frame_id = _wait_for_odometry(odometry, args.odometry_startup_timeout_s)
            pose_source = _SlamPoseSource(odometry, pose_frame_id)
        if args.g3_session:
            exploration_map = SparseEpisodeExplorationMap(
                map_config, pose_frame_id=pose_frame_id, frame_epoch=0
            )

        control_period_s = 1.0 / args.control_rate_hz
        _active_panorama_imu.wait_ready()
        episode = RealG1Episode(
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
                motion_window_s=args.g3_motion_window_s,
                pose_frame_id=pose_frame_id,
                single_forward_panorama=True,
                enable_backtrack=True,
                backtrack_max_path_m=args.backtrack_max_path_m,
                backtrack_start_tolerance_m=args.backtrack_start_tolerance_m,
                backtrack_segment_length_m=args.backtrack_segment_length_m,
                backtrack_goal_tolerance_m=args.backtrack_goal_tolerance_m,
                backtrack_heading_tolerance_rad=(
                    args.backtrack_heading_tolerance_rad
                ),
                backtrack_breadcrumb_spacing_m=(
                    args.backtrack_breadcrumb_spacing_m
                ),
                output_dir=args.output_dir,
            ),
            LocalFollowerConfig(
                tracking_controller=args.tracking_controller,
                target_speed_m_s=args.walk_speed_m_s,
                lookahead_m=args.lookahead_m,
                max_forward_speed_m_s=args.max_forward_speed_m_s,
                max_yaw_speed_rad_s=args.max_yaw_speed_rad_s,
                goal_tolerance_m=args.goal_tolerance_m,
                blind_yaw_radius_m=args.blind_yaw_radius_m,
                kp_xy=args.kp_xy,
                kp_yaw=args.kp_yaw,
                kp_slow_radius_m=args.kp_slow_radius_m,
                kp_yaw_deadband_rad=math.radians(
                    args.kp_yaw_deadband_deg
                ),
                kp_max_lateral_speed_m_s=args.kp_max_lateral_speed_m_s,
                kp_max_yaw_speed_rad_s=args.kp_max_yaw_speed_rad_s,
                yaw_bias_rad_s=args.yaw_bias_rad_s,
                replan_interval_s=args.replan_interval_s,
                dead_reckoning_linear_scale=args.dead_reckoning_linear_scale,
                dead_reckoning_angular_scale=args.dead_reckoning_angular_scale,
            ),
            camera=camera,
            model=CombinedModelClient(
                args.model_url, args.model_timeout_s,
                send_instruction=not args.g3_session,
            ),
            planner=IPlannerClient(args.iplanner_url, args.iplanner_timeout_s),
            session_client=session_client,
            exploration_map=exploration_map,
            odometry=pose_source,
            yaw_provider=(
                pose_source if pose_source is not None
                else dds if args.use_imu_rotation else None
            ),
            panorama_imu=_active_panorama_imu,
            stop_robot=dds.stop,
            real_panorama_config=panorama_config,
        )
        # 在任何会话 HTTP 前登记对象，异常或 Ctrl+C 才能找到它并尝试结束会话。
        _active_episode = episode
        # health 与启动响应校验通过后才继续 HighStand；不把 HTTP 可达误认为协议匹配。
        episode.start_remote_session()

        # 对齐 Uni-LaViRA 真机入口：所有后端完成初始化后、任务开始前，只调用
        # 一次 HighStand。后续导航仍始终使用同一个 LocoClient 高层速度控制器。
        dds.high_stand()
        print("[LOCAL-VLN G1] G1 high-stand request completed; navigation may start.")

        print(
            "[LOCAL-VLN G1] runner framework ready: "
            f"session={session_id!r} odometry={args.odometry_topic or 'disabled'} "
            f"tracking_controller={args.tracking_controller} "
            f"history_limit={'all' if args.history_max_waypoints == 0 else args.history_max_waypoints} "
            f"decision_limit={'unlimited' if args.max_decisions == 0 else args.max_decisions}"
        )
        if odometry is None:
            print(
                "[LOCAL-VLN G1 WARN] Using Uni-LaViRA-style dead reckoning; "
                "planner/model blocking time is covered by the repeated last command. "
                "BACKTRACK will fail closed until a SLAM odometry topic is configured."
            )

        step = 0
        last_tick = time.monotonic()
        started_at = last_tick
        last_applied_command = np.zeros(3, dtype=np.float64)
        previous_mode = "stand"
        # 导航目标约 20Hz，DDS 线程目标约 50Hz：这里计算命令，DDS 重复发送最近命令。
        while not episode.completed:
            if pose_source is not None:
                pose_source.get_pose()
            loop_started = time.monotonic()
            # 用真实循环间隔覆盖上次相机/HTTP 耗时，不固定假设每次都恰好过了 0.05 秒。
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
            # 相机/HTTP 调用可能耗时，下发本轮速度前再次确认位姿仍有效。
            if pose_source is not None:
                pose_source.get_pose()
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
