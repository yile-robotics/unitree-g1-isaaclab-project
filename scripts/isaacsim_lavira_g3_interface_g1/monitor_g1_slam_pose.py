#!/usr/bin/env python3
"""只读监控 G1 在 SLAM 地图坐标系中的平面位姿。"""

from __future__ import annotations

import argparse
import math
import time

from unified_vln.ros2_odometry import Ros2OdometryProvider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topic",
        default="/unitree/slam_relocation/odom",
        help="已加载地图后的机器人基座里程计话题",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.5,
        help="终端输出间隔，单位秒；默认 0.5 秒",
    )
    parser.add_argument(
        "--pose-timeout-s",
        type=float,
        default=1.0,
        help="超过此时间未收到位姿则显示等待状态；默认 1 秒",
    )
    args = parser.parse_args()
    if not math.isfinite(args.interval_s) or args.interval_s <= 0:
        parser.error("--interval-s 必须是正的有限数")

    # 与真机 VLN 入口使用同一个适配器，保留 SLAM 原始 map 坐标，不重新归零。
    provider = Ros2OdometryProvider(
        topic=args.topic,
        pose_timeout_s=args.pose_timeout_s,
        preserve_world_coordinates=True,
        node_name="g1_slam_pose_monitor",
    )
    start_xy: tuple[float, float] | None = None
    last_status = ""
    print(f"仅读取 {args.topic}；按 Ctrl+C 结束，不发送运动命令。", flush=True)

    try:
        while True:
            pose = provider.get_pose()
            frame = provider.frame_id
            child = provider.child_frame_id

            # 不把错误坐标系或过期消息误标成可用于导航的世界坐标。
            if pose is None:
                status = "等待新鲜 SLAM 位姿；检查重定位是否运行。"
            elif frame != "map" or child != "base_link":
                status = f"坐标系不符：{frame or '<空>'} -> {child or '<空>'}，期望 map -> base_link。"
            else:
                status = ""

            if status:
                if status != last_status:
                    print(status, flush=True)
                last_status = status
            else:
                last_status = ""
                if start_xy is None:
                    start_xy = (pose.x, pose.y)
                displacement = math.hypot(pose.x - start_xy[0], pose.y - start_xy[1])
                print(
                    f"{time.strftime('%H:%M:%S')}  map -> base_link  "
                    f"x={pose.x:+.3f} m  y={pose.y:+.3f} m  "
                    f"yaw={math.degrees(pose.yaw):+.1f} deg  "
                    f"距监控起点={displacement:.3f} m",
                    flush=True,
                )
            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("\n已停止位姿监控。", flush=True)
    finally:
        provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
