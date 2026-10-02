#!/usr/bin/env python3
"""检查 SetVelocity 返回码和实测转角。

默认按 IMU 45°停车，可选75°/90°；地图90°模式固定使用0.6rad/s。
--observe-slam 只增加地图观测，停车条件仍为IMU。到时、Ctrl+C或异常时停车。
"""

from __future__ import annotations

import argparse
import math
import time

from g1_real_smoke import StateMonitor, stop_move

WRONG_DIRECTION_DEG = -20.0


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _pose_yaw(pose: dict[str, float]) -> float:
    qx, qy, qz, qw = (pose[key] for key in ("q_x", "q_y", "q_z", "q_w"))
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--wz", type=float, choices=(0.4, 0.5, 0.6, 0.8), required=True)
    parser.add_argument("--seconds", type=float, default=1.5)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--target-map-deg", type=int, choices=(90,))
    targets.add_argument("--target-imu-deg", type=int, choices=(45, 75, 90), default=45)
    parser.add_argument("--observe-slam", action="store_true",
                        help="print rt/slam_info map yaw; IMU remains the stop condition")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required for real robot rotation")
    max_seconds = 30.0 if args.target_map_deg == 90 or args.target_imu_deg >= 75 else 5.0
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= max_seconds:
        parser.error(f"--seconds must be in (0, {max_seconds}]")
    if args.target_map_deg == 90 and args.wz != 0.6:
        parser.error("--target-map-deg 90 requires --wz 0.6")
    if args.observe_slam and args.target_map_deg is not None:
        parser.error("--observe-slam is for IMU-controlled runs only")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    ChannelFactoryInitialize(0, args.network_interface)
    monitor = StateMonitor()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(monitor.on_state)
    client = LocoClient()
    client.SetTimeout(2.0)
    client.Init()
    odom = None
    slam_subscriber = None
    slam_monitor = None
    try:
        wait_deadline = time.monotonic() + 3.0
        while time.monotonic() < wait_deadline:
            try:
                initial_yaw = monitor.require_fresh()
                break
            except RuntimeError:
                time.sleep(0.05)
        else:
            raise RuntimeError("no fresh lowstate within 3 seconds; refusing rotation")

        if args.observe_slam:
            from monitor_g1_slam_info import PoseMonitor
            from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

            slam_monitor = PoseMonitor()
            slam_subscriber = ChannelSubscriber("rt/slam_info", String_)
            slam_subscriber.Init(slam_monitor.on_message, 1)
            slam_deadline = time.monotonic() + 3.0
            while time.monotonic() < slam_deadline:
                pose, received_at, _ = slam_monitor.snapshot()
                if pose is not None and time.monotonic() - received_at <= 1.0:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("no fresh rt/slam_info currentPose within 3 seconds")

        initial_map_yaw = None
        if args.target_map_deg == 90:
            from unified_vln.ros2_odometry import Ros2OdometryProvider

            odom = Ros2OdometryProvider(
                topic="/unitree/slam_relocation/odom",
                pose_timeout_s=0.5,
                preserve_world_coordinates=True,
                node_name="g1_rotation_rpc_probe",
            )
            map_deadline = time.monotonic() + 5.0
            while time.monotonic() < map_deadline:
                pose = odom.get_pose()
                if (
                    pose is not None and odom.frame_id == "map"
                    and odom.child_frame_id == "base_link"
                ):
                    initial_map_yaw = pose.yaw
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("no fresh map -> base_link pose within 5 seconds")

        fsm_code, fsm_id = client.GetFsmId()
        print(f"GetFsmId code={fsm_code}, id={fsm_id}; this does not prove motion mode", flush=True)
        target_text = (
            "map Δyaw target +90°" if args.target_map_deg == 90
            else f"IMU limit +{args.target_imu_deg:.0f}°"
        )
        print(
            f"rotation RPC probe: wz=+{args.wz:.1f} rad/s continuously "
            f"for at most {args.seconds:.1f} s; {target_text}; Ctrl+C requests stop",
            flush=True,
        )
        # 初始化地图订阅可能耗时；真正发送前重新取 IMU 基准。
        initial_yaw = monitor.require_fresh()
        initial_slam_yaw = None
        if slam_monitor is not None:
            slam_pose, received_at, _ = slam_monitor.snapshot()
            if slam_pose is None or time.monotonic() - received_at > 1.0:
                raise RuntimeError("SLAM currentPose stale before rotation")
            initial_slam_yaw = _pose_yaw(slam_pose)
            print(
                f"SLAM currentPose initial yaw={math.degrees(initial_slam_yaw):+.1f}° "
                "(observation only; IMU controls stop)", flush=True,
            )
        started = time.monotonic()
        deadline = started + args.seconds
        requests_sent = 0
        previous_yaw = initial_yaw
        accumulated_yaw = 0.0
        last_report = started
        stop_reason = "time limit"
        latest_map_deg = None
        try:
            while time.monotonic() < deadline:
                current_yaw = monitor.require_fresh()
                # 按相邻 IMU 读数累计偏航，避免跨越 ±π 时直接相减出错。
                accumulated_yaw += _wrap_to_pi(current_yaw - previous_yaw)
                previous_yaw = current_yaw
                turn_deg = math.degrees(accumulated_yaw)
                if odom is not None:
                    pose = odom.get_pose()
                    if (
                        pose is None or odom.frame_id != "map"
                        or odom.child_frame_id != "base_link"
                    ):
                        raise RuntimeError("SLAM map pose missing or stale; stopping")
                    assert initial_map_yaw is not None
                    latest_map_deg = math.degrees(_wrap_to_pi(pose.yaw - initial_map_yaw))
                    if latest_map_deg >= 90.0:
                        stop_reason = "map yaw reached +90°"
                        break
                elif turn_deg >= args.target_imu_deg:
                    stop_reason = f"IMU reached +{args.target_imu_deg:.0f}° limit"
                    break
                if odom is None and turn_deg <= WRONG_DIRECTION_DEG:
                    stop_reason = f"IMU reached {WRONG_DIRECTION_DEG:.0f}° wrong-direction limit"
                    break
                now = time.monotonic()
                if now - last_report >= 0.5:
                    map_text = (
                        f" map Δyaw={latest_map_deg:+.1f}°" if latest_map_deg is not None
                        else ""
                    )
                    if slam_monitor is not None:
                        slam_pose, received_at, error = slam_monitor.snapshot()
                        if slam_pose is None or now - received_at > 1.0 or error:
                            map_text = " SLAM yaw unavailable"
                        else:
                            assert initial_slam_yaw is not None
                            observed_yaw = _pose_yaw(slam_pose)
                            observed_delta = math.degrees(
                                _wrap_to_pi(observed_yaw - initial_slam_yaw)
                            )
                            map_text = (
                                f" SLAM yaw={math.degrees(observed_yaw):+.1f}°"
                                f" Δyaw={observed_delta:+.1f}°"
                            )
                    print(
                        f"elapsed={now - started:.1f}s; IMU Δyaw={turn_deg:+.1f}°{map_text}",
                        flush=True,
                    )
                    last_report = now
                code = client.SetVelocity(0.0, 0.0, args.wz, 0.3)
                requests_sent += 1
                if code != 0:
                    raise RuntimeError(f"rotation SetVelocity RPC failed: {code}")
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        finally:
            stop_move(client)

        time.sleep(0.5)
        final_yaw = monitor.require_fresh()
        delta_deg = math.degrees(_wrap_to_pi(final_yaw - initial_yaw))
        final_map_text = ""
        if odom is not None:
            final_pose = odom.get_pose()
            if final_pose is not None:
                assert initial_map_yaw is not None
                final_map_deg = math.degrees(_wrap_to_pi(final_pose.yaw - initial_map_yaw))
                final_map_text = f" map Δyaw after stop={final_map_deg:+.1f}°;"
        if slam_monitor is not None:
            slam_pose, received_at, error = slam_monitor.snapshot()
            if slam_pose is None or time.monotonic() - received_at > 1.0 or error:
                final_map_text = " SLAM yaw after stop unavailable;"
            else:
                assert initial_slam_yaw is not None
                observed_yaw = _pose_yaw(slam_pose)
                observed_delta = math.degrees(_wrap_to_pi(observed_yaw - initial_slam_yaw))
                final_map_text = (
                    f" SLAM yaw after stop={math.degrees(observed_yaw):+.1f}°"
                    f" Δyaw={observed_delta:+.1f}°;"
                )
        print(
            f"stop reason: {stop_reason}; rotation RPCs accepted: {requests_sent}; "
            f"IMU Δyaw after stop={delta_deg:+.1f}°;{final_map_text} "
            "confirm physical turn visually",
            flush=True,
        )
        reached_target = (
            stop_reason == "map yaw reached +90°" if args.target_map_deg is not None
            else stop_reason == f"IMU reached +{args.target_imu_deg:.0f}° limit"
        )
        return 0 if reached_target else 1
    finally:
        if slam_subscriber is not None:
            slam_subscriber.Close()
        if odom is not None:
            odom.close()
        subscriber.Close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("rotation RPC probe interrupted; stop requested", flush=True)
        raise SystemExit(1)
