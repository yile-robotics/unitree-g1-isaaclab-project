#!/usr/bin/env python3
"""只读对照 Unitree DDS currentPose 与 ROS 2 SLAM 地图位姿。"""

from __future__ import annotations

import argparse
import math
import time

from monitor_g1_slam_info import PoseMonitor
from unified_vln.ros2_odometry import Ros2OdometryProvider


def quaternion_yaw_deg(pose: dict[str, float]) -> float:
    qx, qy, qz, qw = (pose[key] for key in ("q_x", "q_y", "q_z", "q_w"))
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    return math.degrees(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-interface", required=True, help="连接机器人的网卡")
    parser.add_argument("--interval-s", type=float, default=0.5, help="输出间隔，默认 0.5 秒")
    parser.add_argument("--pose-timeout-s", type=float, default=1.0, help="两路位姿过期时间，默认 1 秒")
    args = parser.parse_args()
    for name in ("interval_s", "pose_timeout_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须是正的有限数")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

    ChannelFactoryInitialize(0, args.network_interface)
    dds = PoseMonitor()
    subscriber = ChannelSubscriber("rt/slam_info", String_)
    subscriber.Init(dds.on_message, 1)
    try:
        ros = Ros2OdometryProvider(
            topic="/unitree/slam_relocation/odom",
            pose_timeout_s=args.pose_timeout_s,
            preserve_world_coordinates=True,
            node_name="g1_slam_pose_compare",
        )
        try:
            print("只读比较 DDS currentPose 与 ROS 2 map -> base_link；按 Ctrl+C 结束。", flush=True)
            last_status = ""
            while True:
                now = time.monotonic()
                dds_pose, dds_time, dds_error = dds.snapshot()
                ros_pose = ros.get_pose()

                if dds_error:
                    status = f"DDS SLAM 消息错误：{dds_error}"
                elif dds_pose is None or now - dds_time > args.pose_timeout_s:
                    status = "等待新鲜的 DDS pos_info.currentPose。"
                elif ros_pose is None:
                    status = "等待新鲜的 ROS 2 /unitree/slam_relocation/odom。"
                elif ros.frame_id != "map" or ros.child_frame_id != "base_link":
                    status = f"ROS 2 坐标系不符：{ros.frame_id} -> {ros.child_frame_id}。"
                else:
                    status = ""

                if status:
                    if status != last_status:
                        print(status, flush=True)
                    last_status = status
                else:
                    last_status = ""
                    dds_yaw = quaternion_yaw_deg(dds_pose)
                    ros_yaw = math.degrees(ros_pose.yaw)
                    delta_xy = math.hypot(ros_pose.x - dds_pose["x"], ros_pose.y - dds_pose["y"])
                    delta_yaw = (ros_yaw - dds_yaw + 180.0) % 360.0 - 180.0
                    receive_gap = abs(ros_pose.timestamp - dds_time)
                    print(
                        f"{time.strftime('%H:%M:%S')}  "
                        f"DDS ({dds_pose['x']:+.3f}, {dds_pose['y']:+.3f}, {dds_yaw:+.1f}°)  "
                        f"ROS ({ros_pose.x:+.3f}, {ros_pose.y:+.3f}, {ros_yaw:+.1f}°)  "
                        f"差: xy={delta_xy:.3f} m, yaw={delta_yaw:+.1f}°  "
                        f"接收时间差={receive_gap:.3f} s",
                        flush=True,
                    )
                time.sleep(args.interval_s)
        except KeyboardInterrupt:
            print("\n已停止双路位姿对照。", flush=True)
        finally:
            ros.close()
    finally:
        subscriber.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
