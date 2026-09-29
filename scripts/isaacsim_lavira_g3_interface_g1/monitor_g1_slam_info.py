#!/usr/bin/env python3
"""只读订阅 keyDemo 使用的 rt/slam_info，持续打印 currentPose。"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time


class PoseMonitor:
    """保存最近一条有效的 pos_info；DDS 回调和终端输出在不同线程。"""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pose: dict[str, float] | None = None
        self.received_at = 0.0
        self.message_error = ""

    def on_message(self, message) -> None:
        try:
            payload = json.loads(message.data)
            if payload.get("errorCode") != 0:
                with self.lock:
                    self.message_error = str(payload.get("info", "SLAM 返回错误"))
                return
            if payload.get("type") != "pos_info":
                return
            raw = payload["data"]["currentPose"]
            keys = ("x", "y", "z", "q_x", "q_y", "q_z", "q_w")
            pose = {key: float(raw[key]) for key in keys}
            if not all(math.isfinite(value) for value in pose.values()):
                return
            q_norm = math.sqrt(sum(pose[key] ** 2 for key in keys[3:]))
            if q_norm <= 1e-12:
                return
        except (AttributeError, TypeError, ValueError, KeyError):
            return
        with self.lock:
            self.pose = pose
            self.received_at = time.monotonic()
            self.message_error = ""

    def snapshot(self) -> tuple[dict[str, float] | None, float, str]:
        with self.lock:
            return self.pose, self.received_at, self.message_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-interface",
        required=True,
        help="连接机器人的网卡，例如 enxf8e43bbea486",
    )
    parser.add_argument("--interval-s", type=float, default=0.5, help="输出间隔，默认 0.5 秒")
    parser.add_argument("--pose-timeout-s", type=float, default=2.0, help="位姿过期时间，默认 2 秒")
    args = parser.parse_args()
    if not math.isfinite(args.interval_s) or args.interval_s <= 0:
        parser.error("--interval-s 必须是正的有限数")
    if not math.isfinite(args.pose_timeout_s) or args.pose_timeout_s <= 0:
        parser.error("--pose-timeout-s 必须是正的有限数")

    # 仅初始化 DDS 订阅器，不创建 SLAM 操作或机器人运动客户端。
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

    ChannelFactoryInitialize(0, args.network_interface)
    monitor = PoseMonitor()
    subscriber = ChannelSubscriber("rt/slam_info", String_)
    subscriber.Init(monitor.on_message, 1)
    print("仅读取 rt/slam_info 的 pos_info.currentPose；按 Ctrl+C 结束。", flush=True)
    last_status = ""
    try:
        while True:
            pose, received_at, error = monitor.snapshot()
            if error:
                status = f"SLAM 消息错误：{error}"
            elif pose is None or time.monotonic() - received_at > args.pose_timeout_s:
                status = "等待新鲜的 pos_info.currentPose；检查 SLAM 重定位是否运行。"
            else:
                status = ""

            if status:
                if status != last_status:
                    print(status, flush=True)
                last_status = status
            else:
                last_status = ""
                q_norm = math.sqrt(sum(pose[key] ** 2 for key in ("q_x", "q_y", "q_z", "q_w")))
                qx, qy, qz, qw = (pose[key] / q_norm for key in ("q_x", "q_y", "q_z", "q_w"))
                yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                print(
                    f"{time.strftime('%H:%M:%S')}  currentPose  "
                    f"x={pose['x']:+.3f} m  y={pose['y']:+.3f} m  z={pose['z']:+.3f} m  "
                    f"yaw={math.degrees(yaw):+.1f} deg  "
                    f"q=({pose['q_x']:+.3f}, {pose['q_y']:+.3f}, "
                    f"{pose['q_z']:+.3f}, {pose['q_w']:+.3f})",
                    flush=True,
                )
            time.sleep(args.interval_s)
    except KeyboardInterrupt:
        print("\n已停止 SLAM 信息监控。", flush=True)
    finally:
        subscriber.Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
