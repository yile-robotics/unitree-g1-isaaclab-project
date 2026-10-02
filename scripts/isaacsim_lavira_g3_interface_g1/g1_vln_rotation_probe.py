#!/usr/bin/env python3
"""用VLN DDS后端测试旋转，不采集图像。

imu45：0.6rad/s，IMU达到45°停车；timed/slam：复用定时/地图反馈旋转器。
imu75_panorama：0.8rad/s，每段IMU达到75°停车，停稳后检查SLAM转角并进入下一段。
各模式检查状态新鲜度和时限，Ctrl+C或异常时执行stop()/close()。
全景探针参数尚未接入正式VLN旋转器；SLAM需提前启动。
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import time

from g1_real_smoke import StateMonitor
from g1_rotation_rpc_probe import _pose_yaw
from monitor_g1_slam_info import PoseMonitor
from unified_vln.g1_dds_backend import UnitreeG1DDSBackend
from unified_vln.ros2_odometry import Ros2OdometryProvider
from unified_vln.rotation import TimedFixedSpeedRotation


DEFAULT_ROTATION_SPEED_RAD_S = 0.4  # 与 run_g1_real.py 的默认值相同。
IMU45_SPEED_RAD_S = 0.6  # 直接 SDK 诊断中已观测到持续左转。
IMU45_TARGET_DEG = 45.0
IMU45_WRONG_DIRECTION_DEG = -20.0
IMU45_HARD_LIMIT_S = 6.0
CONTROL_PERIOD_S = 0.05
DEFAULT_QUARTER_HARD_LIMIT_S = 10.0
SETTLE_S = 0.5
MAX_SLAM_POSE_JUMP_M = 0.5
MAX_SLAM_YAW_JUMP_RAD = math.radians(45.0)
MAX_IMU_TURN_DEG = 120.0
MIN_IMU_TURN_DEG = -20.0
MIN_QUARTER_DEG = 70.0
MAX_QUARTER_DEG = 110.0

_active_dds: UnitreeG1DDSBackend | None = None
_active_odom: Ros2OdometryProvider | None = None
_active_subscriber = None
_active_slam_subscriber = None
_cleaned_up = False


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _cleanup() -> None:
    """先请求停车并关闭 DDS 发送线程，再关闭只读订阅器。"""

    global _active_dds, _active_odom, _active_subscriber, _active_slam_subscriber, _cleaned_up
    if _cleaned_up:
        return
    _cleaned_up = True
    if _active_dds is not None:
        try:
            _active_dds.stop()
        except Exception as exc:
            print(f"[ROTATION PROBE] stop error: {exc}", flush=True)
        try:
            _active_dds.close()
        except Exception as exc:
            print(f"[ROTATION PROBE] close error: {exc}", flush=True)
        _active_dds = None
    if _active_odom is not None:
        try:
            _active_odom.close()
        except Exception as exc:
            print(f"[ROTATION PROBE] odometry close error: {exc}", flush=True)
        _active_odom = None
    if _active_subscriber is not None:
        try:
            _active_subscriber.Close()
        except Exception as exc:
            print(f"[ROTATION PROBE] subscriber close error: {exc}", flush=True)
        _active_subscriber = None
    if _active_slam_subscriber is not None:
        try:
            _active_slam_subscriber.Close()
        except Exception as exc:
            print(f"[ROTATION PROBE] SLAM subscriber close error: {exc}", flush=True)
        _active_slam_subscriber = None


def _on_sigint(_signum, _frame) -> None:
    print("\n[ROTATION PROBE] Ctrl+C received; stopping...", flush=True)
    _cleanup()
    print("[ROTATION PROBE] cleanup returned; confirm robot stopped visually", flush=True)
    os._exit(0)


class _SlamYaw:
    """地图坐标和偏航角共用同一条 ROS 2 SLAM 位姿消息。"""

    def __init__(self, provider: Ros2OdometryProvider):
        self.provider = provider

    def get_pose(self):
        pose = self.provider.get_pose()
        if pose is None:
            raise RuntimeError("SLAM pose missing or stale; stopping rotation")
        if self.provider.frame_id != "map" or self.provider.child_frame_id != "base_link":
            raise RuntimeError(
                f"unexpected SLAM frames: {self.provider.frame_id} -> "
                f"{self.provider.child_frame_id}; expected map -> base_link"
            )
        return pose

    def get_yaw(self) -> float:
        return self.get_pose().yaw


def _wait_for_lowstate(monitor: StateMonitor) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            monitor.require_fresh()
            return
        except RuntimeError:
            time.sleep(0.05)
    raise RuntimeError("no fresh G1 lowstate within 3 seconds; refusing rotation")


def _wait_for_slam(slam: _SlamYaw) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            slam.get_pose()
            return
        except RuntimeError:
            time.sleep(0.05)
    raise RuntimeError("no fresh map -> base_link pose within 5 seconds; refusing rotation")


def _run_imu45(dds: UnitreeG1DDSBackend, monitor: StateMonitor) -> None:
    """验证正式 VLN DDS 后端能否维持 0.6 rad/s 旋转，按 IMU 角度停车。"""

    initial_yaw = monitor.require_fresh()
    previous_yaw = initial_yaw
    accumulated_yaw = 0.0
    started = time.monotonic()
    last_report = started
    stop_reason = "6 s time limit"
    try:
        print(
            "[ROTATION PROBE] VLN DDS backend: wz=+0.6 rad/s, "
            "IMU +45° limit, at most 6 s; Ctrl+C stops",
            flush=True,
        )
        dds.set_velocity(0.0, 0.0, IMU45_SPEED_RAD_S)
        while time.monotonic() - started < IMU45_HARD_LIMIT_S:
            current_yaw = monitor.require_fresh()
            accumulated_yaw += _wrap_to_pi(current_yaw - previous_yaw)
            previous_yaw = current_yaw
            turn_deg = math.degrees(accumulated_yaw)
            if turn_deg >= IMU45_TARGET_DEG:
                stop_reason = "IMU reached +45° limit"
                break
            if turn_deg <= IMU45_WRONG_DIRECTION_DEG:
                stop_reason = "IMU reached -20° wrong-direction limit"
                break
            now = time.monotonic()
            if now - last_report >= 0.5:
                print(
                    f"[ROTATION PROBE] elapsed={now - started:.1f}s; "
                    f"IMU Δyaw={turn_deg:+.1f}°",
                    flush=True,
                )
                last_report = now
            time.sleep(CONTROL_PERIOD_S)
    finally:
        # 立即使用正式 VLN DDS 后端的停车路径，而非只更新线程目标速度。
        dds.stop()

    time.sleep(SETTLE_S)
    final_yaw = monitor.require_fresh()
    final_deg = math.degrees(_wrap_to_pi(final_yaw - initial_yaw))
    print(
        f"[ROTATION PROBE] stop reason: {stop_reason}; "
        f"IMU Δyaw after stop={final_deg:+.1f}°; confirm physical turn visually",
        flush=True,
    )
    if stop_reason != "IMU reached +45° limit":
        raise RuntimeError("VLN DDS IMU 45° rotation did not complete")


def _fresh_slam_yaw(monitor: PoseMonitor) -> float:
    pose, received_at, error = monitor.snapshot()
    if pose is None or time.monotonic() - received_at > 1.0 or error:
        raise RuntimeError(f"SLAM currentPose missing or stale: {error}")
    return _pose_yaw(pose)


def _run_imu75_panorama(
    dds: UnitreeG1DDSBackend,
    monitor: StateMonitor,
    slam_monitor: PoseMonitor,
    *,
    quarters: int,
) -> None:
    """用正式 VLN DDS 后端连续转向，每段停车稳定后模拟一次相机采集。"""

    completed: list[float] = []
    directions = ("left", "behind", "right", "forward")
    print(
        f"[ROTATION PROBE] panorama: {quarters} left quarter(s), "
        "wz=+0.8 rad/s, IMU +75° stop per quarter; "
        "SLAM is observation only; Ctrl+C stops",
        flush=True,
    )
    print("[ROTATION PROBE] simulated capture: forward (initial); no camera frame saved", flush=True)
    for quarter in range(1, quarters + 1):
        initial_imu = monitor.require_fresh()
        initial_map = _fresh_slam_yaw(slam_monitor)
        previous_imu = initial_imu
        accumulated_imu = 0.0
        started = time.monotonic()
        last_report = started
        stop_reason = "10 s time limit"
        print(f"[ROTATION PROBE] starting quarter {quarter}/{quarters}", flush=True)
        try:
            dds.set_velocity(0.0, 0.0, 0.8)
            while time.monotonic() - started < 10.0:
                current_imu = monitor.require_fresh()
                accumulated_imu += _wrap_to_pi(current_imu - previous_imu)
                previous_imu = current_imu
                imu_deg = math.degrees(accumulated_imu)
                current_map = _fresh_slam_yaw(slam_monitor)
                if imu_deg >= 75.0:
                    stop_reason = "IMU reached +75° limit"
                    break
                if imu_deg <= -20.0:
                    stop_reason = "wrong-direction IMU limit"
                    break
                now = time.monotonic()
                if now - last_report >= 0.5:
                    map_deg = math.degrees(_wrap_to_pi(current_map - initial_map))
                    print(
                        f"[ROTATION PROBE] {quarter}/{quarters} elapsed={now-started:.1f}s "
                        f"IMU Δyaw={imu_deg:+.1f}° SLAM Δyaw={map_deg:+.1f}°",
                        flush=True,
                    )
                    last_report = now
                time.sleep(CONTROL_PERIOD_S)
        finally:
            dds.stop()

        # VLN 在每个 90° 分段结束后保持零速度，并采集下一个朝向。
        time.sleep(SETTLE_S)
        final_imu = monitor.require_fresh()
        final_map = _fresh_slam_yaw(slam_monitor)
        imu_deg = math.degrees(_wrap_to_pi(final_imu - initial_imu))
        map_deg = math.degrees(_wrap_to_pi(final_map - initial_map))
        print(
            f"[ROTATION PROBE] {quarter}/{quarters} stop reason: {stop_reason}; "
            f"IMU Δyaw after stop={imu_deg:+.1f}°; "
            f"SLAM yaw={math.degrees(final_map):+.1f}° "
            f"Δyaw after stop={map_deg:+.1f}°",
            flush=True,
        )
        if stop_reason != "IMU reached +75° limit":
            raise RuntimeError(f"quarter {quarter} did not reach the IMU target")
        if not MIN_QUARTER_DEG <= map_deg <= MAX_QUARTER_DEG:
            raise RuntimeError(
                f"quarter {quarter}: SLAM turn {map_deg:+.1f}° outside "
                f"{MIN_QUARTER_DEG:.0f}°–{MAX_QUARTER_DEG:.0f}°; "
                "stopping before the next quarter"
            )
        completed.append(map_deg)
        print(
            f"[ROTATION PROBE] simulated capture: {directions[quarter-1]} "
            f"({quarter}/{quarters}); no camera frame saved",
            flush=True,
        )
    print(
        f"[ROTATION PROBE] panorama complete; sum of SLAM quarter turns="
        f"{sum(completed):+.1f}°",
        flush=True,
    )


def _run_quarter(
    quarter: int,
    total_quarters: int,
    dds: UnitreeG1DDSBackend,
    monitor: StateMonitor,
    rotation: TimedFixedSpeedRotation,
    slam: _SlamYaw | None,
    hard_limit_s: float,
) -> tuple[float, float]:
    """复用 VLN 旋转器完成一个 90°左转，返回耗时和停稳后的角度变化。"""

    imu_start = monitor.require_fresh()
    previous_imu_yaw = imu_start
    accumulated_imu_yaw = 0.0
    pose_start = slam.get_pose() if slam is not None else None
    previous_pose = pose_start
    rotation.start("left")
    started = time.monotonic()
    last_tick = started
    last_report = started
    try:
        while True:
            imu_yaw = monitor.require_fresh()
            accumulated_imu_yaw += _wrap_to_pi(imu_yaw - previous_imu_yaw)
            previous_imu_yaw = imu_yaw
            imu_turn_deg = math.degrees(accumulated_imu_yaw)
            if imu_turn_deg >= MAX_IMU_TURN_DEG or imu_turn_deg <= MIN_IMU_TURN_DEG:
                raise RuntimeError(
                    f"quarter {quarter}: IMU turn {imu_turn_deg:+.1f}° "
                    "outside safety range; stopping"
                )
            if slam is not None:
                pose = slam.get_pose()
                assert previous_pose is not None
                position_jump = math.hypot(pose.x - previous_pose.x, pose.y - previous_pose.y)
                yaw_jump = abs(_wrap_to_pi(pose.yaw - previous_pose.yaw))
                if position_jump > MAX_SLAM_POSE_JUMP_M or yaw_jump > MAX_SLAM_YAW_JUMP_RAD:
                    raise RuntimeError("SLAM pose jumped during rotation; stopping")
                previous_pose = pose

            now = time.monotonic()
            if now - started >= hard_limit_s:
                raise RuntimeError(f"quarter {quarter}: hard time limit reached before 90°")
            if slam is not None and now - last_report >= 0.5:
                print(
                    f"[ROTATION PROBE] {quarter}/{total_quarters} live map "
                    f"x={pose.x:+.3f}m y={pose.y:+.3f}m "
                    f"yaw={math.degrees(pose.yaw):+.1f}° "
                    f"IMU Δyaw={imu_turn_deg:+.1f}°",
                    flush=True,
                )
                last_report = now
            command = rotation.update(now - last_tick)
            last_tick = now
            if slam is not None and not rotation.feedback_active:
                raise RuntimeError("SLAM yaw feedback lost; refusing timed fallback")
            dds.set_velocity(0.0, 0.0, command.wz)
            if command.done:
                if rotation.feedback_timed_out:
                    raise RuntimeError(f"quarter {quarter}: SLAM yaw feedback timed out")
                break
            time.sleep(CONTROL_PERIOD_S)
    finally:
        # 每个 90°边界都调用正式后端停车，再记录停稳后角度。
        dds.stop()

    # 给机体短暂时间停稳后再记录实际角度；在此期间仍监控状态和 SLAM。
    settle_deadline = time.monotonic() + SETTLE_S
    while time.monotonic() < settle_deadline:
        monitor.require_fresh()
        if slam is not None:
            slam.get_pose()
        time.sleep(min(0.05, max(0.0, settle_deadline - time.monotonic())))

    imu_end = monitor.require_fresh()
    measured_imu_deg = math.degrees(_wrap_to_pi(imu_end - imu_start))
    if slam is None:
        print(
            f"[ROTATION PROBE] {quarter}/{total_quarters}: elapsed={time.monotonic() - started:.2f}s "
            f"IMU Δyaw={measured_imu_deg:+.1f}° (仅供观察；未使用地图坐标)",
            flush=True,
        )
        return time.monotonic() - started, measured_imu_deg

    pose_end = slam.get_pose()
    assert pose_start is not None
    measured_map_deg = math.degrees(_wrap_to_pi(pose_end.yaw - pose_start.yaw))
    drift_m = math.hypot(pose_end.x - pose_start.x, pose_end.y - pose_start.y)
    print(
        f"[ROTATION PROBE] {quarter}/{total_quarters}: elapsed={time.monotonic() - started:.2f}s "
        f"map start=({pose_start.x:+.3f},{pose_start.y:+.3f},{math.degrees(pose_start.yaw):+.1f}°) "
        f"end=({pose_end.x:+.3f},{pose_end.y:+.3f},{math.degrees(pose_end.yaw):+.1f}°) "
        f"Δyaw={measured_map_deg:+.1f}° drift={drift_m:.3f}m "
        f"IMU Δyaw={measured_imu_deg:+.1f}°",
        flush=True,
    )
    return time.monotonic() - started, measured_map_deg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--mode", choices=("imu45", "imu75_panorama", "timed", "slam"), required=True)
    parser.add_argument("--quarters", type=int, choices=(1, 4), required=True)
    parser.add_argument(
        "--rotation-speed", type=float, choices=(0.4, 0.6),
        default=DEFAULT_ROTATION_SPEED_RAD_S,
        help="timed/slam mode yaw command; 0.6 needs one-quarter validation first",
    )
    parser.add_argument(
        "--quarter-hard-limit-s", type=float, default=DEFAULT_QUARTER_HARD_LIMIT_S,
        help="maximum continuous command time per 90° turn; default 10 s, maximum 30 s",
    )
    parser.add_argument("--rotation-duration-scale", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true", help="required to send real motion")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required for real robot rotation")
    if not math.isfinite(args.rotation_duration_scale) or not 0 < args.rotation_duration_scale <= 1.5:
        parser.error("--rotation-duration-scale must be in (0, 1.5]")
    if not math.isfinite(args.quarter_hard_limit_s) or not 0 < args.quarter_hard_limit_s <= 30.0:
        parser.error("--quarter-hard-limit-s must be in (0, 30.0]")
    if args.mode == "imu45" and args.quarters != 1:
        parser.error("--mode imu45 requires --quarters 1")
    if args.mode == "imu75_panorama" and args.quarters != 4:
        parser.error("--mode imu75_panorama requires --quarters 4")
    if args.mode == "timed" and args.quarters == 4:
        parser.error("four quarters require --mode slam and fresh map yaw feedback")
    signal.signal(signal.SIGINT, _on_sigint)

    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    global _active_dds, _active_odom, _active_subscriber, _active_slam_subscriber
    try:
        _active_dds = UnitreeG1DDSBackend(args.network_interface)
        monitor = StateMonitor()
        _active_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        _active_subscriber.Init(monitor.on_state)
        _wait_for_lowstate(monitor)

        if args.mode == "imu45":
            _run_imu45(_active_dds, monitor)
            return 0

        if args.mode == "imu75_panorama":
            from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

            slam_monitor = PoseMonitor()
            _active_slam_subscriber = ChannelSubscriber("rt/slam_info", String_)
            _active_slam_subscriber.Init(slam_monitor.on_message, 1)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                try:
                    _fresh_slam_yaw(slam_monitor)
                    break
                except RuntimeError:
                    time.sleep(0.05)
            else:
                raise RuntimeError("no fresh SLAM currentPose within 3 seconds")
            _run_imu75_panorama(
                _active_dds, monitor, slam_monitor, quarters=args.quarters
            )
            return 0

        slam = None
        if args.mode == "slam":
            _active_odom = Ros2OdometryProvider(
                topic="/unitree/slam_relocation/odom",
                pose_timeout_s=0.5,
                preserve_world_coordinates=True,
                node_name="g1_vln_rotation_probe",
            )
            slam = _SlamYaw(_active_odom)
            _wait_for_slam(slam)

        rotation = TimedFixedSpeedRotation(
            args.rotation_speed,
            args.rotation_duration_scale,
            yaw_provider=slam,
        )
        print(
            f"[ROTATION PROBE] mode={args.mode} quarters={args.quarters} "
            f"left-turn wz={args.rotation_speed:.1f} rad/s; Ctrl+C stops",
            flush=True,
        )
        changes = []
        for quarter in range(1, args.quarters + 1):
            if quarter > 1:
                print(
                    "[ROTATION PROBE] robot stopped; inspect feet, heading and space. "
                    "Type NEXT then Enter for the next 90°, or Ctrl+C to end:",
                    flush=True,
                )
                if input().strip() != "NEXT":
                    print("[ROTATION PROBE] remaining quarters cancelled", flush=True)
                    return 1
            print(f"[ROTATION PROBE] starting left quarter {quarter}/{args.quarters}", flush=True)
            _, delta_deg = _run_quarter(
                quarter, args.quarters, _active_dds, monitor, rotation, slam,
                args.quarter_hard_limit_s,
            )
            changes.append(delta_deg)
            if not MIN_QUARTER_DEG <= delta_deg <= MAX_QUARTER_DEG:
                raise RuntimeError(
                    f"quarter {quarter}: observed {delta_deg:+.1f}°, outside expected left-turn "
                    f"range {MIN_QUARTER_DEG:.0f}°–{MAX_QUARTER_DEG:.0f}°; "
                    "stopping before the next quarter"
                )
        print(
            f"[ROTATION PROBE] completed {args.quarters} quarter(s); "
            f"sum of measured Δyaw={sum(changes):+.1f}°; stopping",
            flush=True,
        )
        return 0
    finally:
        _cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
