#!/usr/bin/env python3
"""用 VLN DDS 后端前进，并对照 SLAM 地图位移和平均速度。

只发送 vx=0.3、0.4 或 0.5 m/s、vy=wz=0 的短时命令；不启动相机、模型或规划器。
按 Ctrl+C 或状态/位姿失效时会先请求停车。SLAM 位移是定位估计，不是独立真值。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time

from g1_real_smoke import StateMonitor
from g1_rotation_rpc_probe import _pose_yaw, _wrap_to_pi
from monitor_g1_slam_info import PoseMonitor
from unified_vln.g1_dds_backend import UnitreeG1DDSBackend


DEFAULT_FORWARD_M_S = 0.5
MAX_SECONDS_BY_SPEED = {0.3: 3.0, 0.4: 2.0, 0.5: 4.0}
POSE_MAX_AGE_S = 1.0
SETTLE_S = 0.8


@dataclass(frozen=True)
class MapPose:
    x: float
    y: float
    yaw: float
    received_at: float


def _fresh_pose(monitor: PoseMonitor) -> MapPose:
    pose, received_at, error = monitor.snapshot()
    if error or pose is None or time.monotonic() - received_at > POSE_MAX_AGE_S:
        raise RuntimeError(f"SLAM currentPose missing or stale: {error}")
    return MapPose(pose["x"], pose["y"], _pose_yaw(pose), received_at)


def _displacement(start: MapPose, end: MapPose) -> tuple[float, float, float, float]:
    """把地图位移投影到起步时的机器人前方/左方，避免误读 map x。"""

    dx, dy = end.x - start.x, end.y - start.y
    forward = dx * math.cos(start.yaw) + dy * math.sin(start.yaw)
    left = -dx * math.sin(start.yaw) + dy * math.cos(start.yaw)
    return forward, left, math.hypot(dx, dy), math.degrees(_wrap_to_pi(end.yaw - start.yaw))


def _wait_for_data(lowstate: StateMonitor, slam: PoseMonitor) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            lowstate.require_fresh()
            _fresh_pose(slam)
            return
        except RuntimeError:
            time.sleep(0.05)
    raise RuntimeError("no fresh lowstate and SLAM currentPose within 3 seconds")


def _run_motion(
    dds, lowstate: StateMonitor, slam: PoseMonitor, seconds: float,
    *, vx: float = DEFAULT_FORWARD_M_S,
) -> None:
    start_pose = _fresh_pose(slam)
    lowstate.require_fresh()
    print(
        f"[SPEED PROBE] start map x={start_pose.x:+.3f} m y={start_pose.y:+.3f} m "
        f"yaw={math.degrees(start_pose.yaw):+.1f}°; "
        f"command vx={vx:.2f} m/s for {seconds:.1f} s",
        flush=True,
    )
    started = time.monotonic()
    deadline = started + seconds
    last_report = started
    last_active_pose = start_pose
    try:
        dds.set_velocity(vx, 0.0, 0.0)
        while time.monotonic() < deadline:
            lowstate.require_fresh()
            pose = _fresh_pose(slam)
            # 重定位跳变时测量无效；先停车，不把跳变计作机器人移动。
            if (
                math.hypot(pose.x - last_active_pose.x, pose.y - last_active_pose.y) > 0.5
                or abs(_wrap_to_pi(pose.yaw - last_active_pose.yaw)) > math.radians(45.0)
            ):
                raise RuntimeError("SLAM pose jumped between samples")
            last_active_pose = pose
            now = time.monotonic()
            if now - last_report >= 0.25:
                forward, left, _, _ = _displacement(start_pose, pose)
                print(
                    f"[SPEED PROBE] t={now-started:.2f} s "
                    f"map forward={forward:+.3f} m left={left:+.3f} m",
                    flush=True,
                )
                last_report = now
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        # 此帧在请求停车之前取得，用于计算发命令期间的平均速度。
        active_pose = _fresh_pose(slam)
    finally:
        active_seconds = time.monotonic() - started
        dds.stop()

    stop_return_pose = _fresh_pose(slam)
    settle_deadline = time.monotonic() + SETTLE_S
    while time.monotonic() < settle_deadline:
        lowstate.require_fresh()
        _fresh_pose(slam)
        time.sleep(min(0.05, max(0.0, settle_deadline - time.monotonic())))
    final_pose = _fresh_pose(slam)

    forward_active, left_active, _, _ = _displacement(start_pose, active_pose)
    forward_stop, _, _, _ = _displacement(start_pose, stop_return_pose)
    forward_final, left_final, straight_final, yaw_final = _displacement(start_pose, final_pose)
    expected = vx * active_seconds
    average_forward = forward_active / active_seconds
    print(
        f"[SPEED PROBE] command duration={active_seconds:.2f} s; "
        f"{vx:.1f} m/s × duration={expected:.3f} m (command integral)", flush=True,
    )
    print(
        f"[SPEED PROBE] before stop request: map forward={forward_active:+.3f} m "
        f"left={left_active:+.3f} m; average forward={average_forward:+.3f} m/s",
        flush=True,
    )
    print(
        f"[SPEED PROBE] after stop() returned: map forward={forward_stop:+.3f} m; "
        f"after {SETTLE_S:.1f} s settle: forward={forward_final:+.3f} m "
        f"left={left_final:+.3f} m straight-line={straight_final:.3f} m "
        f"yaw change={yaw_final:+.1f}°; "
        f"final map x={final_pose.x:+.3f} m y={final_pose.y:+.3f} m",
        flush=True,
    )
    print(
        f"[SPEED PROBE] final/command distance ratio="
        f"{forward_final/expected:.2f}; "
        f"extra forward after stop request={forward_final-forward_active:+.3f} m",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--vx", type=float, choices=tuple(MAX_SECONDS_BY_SPEED),
                        default=DEFAULT_FORWARD_M_S)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    max_seconds = MAX_SECONDS_BY_SPEED[args.vx]
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= max_seconds:
        parser.error(f"--seconds must be in (0, {max_seconds}] for --vx {args.vx}")
    if not args.execute:
        parser.error("--execute is required for real robot motion")

    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    dds = None
    low_subscriber = None
    slam_subscriber = None
    try:
        dds = UnitreeG1DDSBackend(args.network_interface)
        lowstate = StateMonitor()
        slam = PoseMonitor()
        low_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        low_subscriber.Init(lowstate.on_state)
        slam_subscriber = ChannelSubscriber("rt/slam_info", String_)
        slam_subscriber.Init(slam.on_message, 1)
        _wait_for_data(lowstate, slam)
        _run_motion(dds, lowstate, slam, args.seconds, vx=args.vx)
        return 0
    finally:
        # 运动中异常或 Ctrl+C 都先停车，再释放订阅和 DDS 线程。
        try:
            if dds is not None:
                dds.close()
        finally:
            if slam_subscriber is not None:
                slam_subscriber.Close()
            if low_subscriber is not None:
                low_subscriber.Close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[SPEED PROBE] Ctrl+C received; cleanup returned; confirm stop visually", flush=True)
        raise SystemExit(130)
