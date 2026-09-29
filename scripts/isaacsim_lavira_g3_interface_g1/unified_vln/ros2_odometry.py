# 中文导读：
# 输入是 ROS 2 nav_msgs/msg/Odometry，而非 Unitree SDK 的 rt/slam_info 字符串 JSON。
# 这里只读取外部 SLAM 输出，不启动 SLAM、不读取 PCD、不自动做雷达到基座的 TF 变换。
# 输出 Pose2D 保留平面位置和 yaw；调用方必须确保输入代表所需的机器人基座。

from __future__ import annotations

"""ROS 2 ``/Odometry`` 到统一 ``Pose2D`` 接口的适配器。

Uni-LaViRA 真机代码用 ROS 1 ``rospy`` 订阅 LiDAR/SLAM 里程计。这里保持相同的
数据职责，但换成当前机器已安装的 ROS 2 Humble ``rclpy``。ROS 依赖采用延迟
导入，因此未加载 ROS 环境时仍可导入其余导航模块。
"""

import math
import threading
import time

import numpy as np

from .odometry import Pose2D


class Ros2OdometryProvider:
    """在后台订阅 ROS 2 ``nav_msgs/msg/Odometry`` 并提供最新平面位姿。

    默认主题沿用 Uni-LaViRA 的 ``/Odometry``。收到消息时提取 ``x``、``y`` 和
    四元数对应的 yaw；消息过期、SLAM 发散或尚未收到数据时，``get_pose`` 返回
    ``None``，上层局部跟随器便会选择航位推算。

    ``start_node=False`` 只用于无 ROS/无真机单元测试：此时可调用
    ``ingest_odometry`` 注入假的 Odometry 消息，但不会创建 ROS 节点。
    """

    def __init__(
        self,
        topic: str = "/Odometry",
        *,
        pose_timeout_s: float = 0.5,
        node_name: str = "unified_vln_odometry",
        start_node: bool = True,
        preserve_world_coordinates: bool = False,
    ):
        if not topic.strip() or not node_name.strip():
            raise ValueError("ROS 2 odometry topic/node name must not be empty.")
        if not math.isfinite(pose_timeout_s) or pose_timeout_s <= 0.0:
            raise ValueError("ROS 2 odometry timeout must be finite and positive.")

        self.topic = topic
        self.pose_timeout_s = float(pose_timeout_s)
        self.preserve_world_coordinates = bool(preserve_world_coordinates)
        self.node_name = node_name
        self.lock = threading.Lock()
        self.latest_pose: Pose2D | None = None
        self.odom_offset_xy: np.ndarray | None = None
        self.disabled_reason: str | None = None
        self.frame_id = ""
        self.child_frame_id = ""

        self.running = False
        self.node = None
        self.subscription = None
        self.executor = None
        self.spin_thread: threading.Thread | None = None
        self._rclpy = None
        self._owns_rclpy = False
        self.last_spin_error: str | None = None

        if start_node:
            self._start_node()

    def _start_node(self) -> None:
        """延迟加载 ROS 2，创建订阅并启动单独的 executor 线程。"""

        try:
            import rclpy
            from nav_msgs.msg import Odometry
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 Python is unavailable. Source /opt/ros/humble/setup.bash "
                "and use a Python environment that can import rclpy."
            ) from exc

        self._rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)

        try:
            self.node = rclpy.create_node(self.node_name)
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.subscription = self.node.create_subscription(
                Odometry,
                self.topic,
                self.ingest_odometry,
                qos,
            )
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.node)
            self.running = True
            self.spin_thread = threading.Thread(
                target=self._spin,
                name="unified-vln-ros2-odometry",
                daemon=True,
            )
            self.spin_thread.start()
        except Exception:
            self.close()
            raise

    def _spin(self) -> None:
        """处理 ROS 回调；异常被保存，``get_pose`` 随后会安全返回 ``None``。"""

        while self.running and self._rclpy is not None and self._rclpy.ok():
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as exc:
                self.last_spin_error = str(exc)
                self.running = False

    # 四元数先归一化再提取 yaw；输出只保留 x/y/yaw。
    # 新鲜度使用本机接收时间，不是消息 header 的采样时间，不能据此证明传感器端无延迟。
    def ingest_odometry(self, message, *, received_time: float | None = None) -> None:
        """验证并保存一条 ROS 2 Odometry 消息。

        与 Uni-LaViRA 一致，``|z| > 5m`` 被视为 SLAM 发散并永久停用本次 provider；
        特别大的平面坐标会以首次大值作为临时原点，只保留相对运动。
        """

        try:
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            x, y, z = float(position.x), float(position.y), float(position.z)
            qx = float(orientation.x)
            qy = float(orientation.y)
            qz = float(orientation.z)
            qw = float(orientation.w)
            values = np.array([x, y, z, qx, qy, qz, qw], dtype=np.float64)
            if not np.all(np.isfinite(values)):
                return
            quaternion_norm = float(np.linalg.norm(values[3:]))
            if quaternion_norm <= 1e-12:
                return
            qx, qy, qz, qw = (value / quaternion_norm for value in values[3:])
            yaw = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
            timestamp = time.monotonic() if received_time is None else float(received_time)
            if not math.isfinite(timestamp):
                return
        except (AttributeError, TypeError, ValueError):
            return

        with self.lock:
            if self.disabled_reason is not None:
                return
            if abs(z) > 5.0:
                self.disabled_reason = (
                    f"ROS 2 odometry disabled after SLAM divergence (z={z:.3f}m)."
                )
                self.latest_pose = None
                return
            if (
                not self.preserve_world_coordinates
                and (abs(x) > 1000.0 or abs(y) > 1000.0)
                and self.odom_offset_xy is None
            ):
                self.odom_offset_xy = np.array([x, y], dtype=np.float64)
            if self.odom_offset_xy is not None:
                x -= float(self.odom_offset_xy[0])
                y -= float(self.odom_offset_xy[1])
            self.latest_pose = Pose2D(x=x, y=y, yaw=yaw, timestamp=timestamp).validated()
            header = getattr(message, "header", None)
            self.frame_id = str(getattr(header, "frame_id", ""))
            self.child_frame_id = str(getattr(message, "child_frame_id", ""))

    def get_pose(self) -> Pose2D | None:
        """返回最新且未过期的位姿副本；不可用时返回 ``None``。"""

        with self.lock:
            pose = self.latest_pose
            disabled = self.disabled_reason is not None
        if disabled or pose is None:
            return None
        if time.monotonic() - pose.timestamp > self.pose_timeout_s:
            return None
        return Pose2D(pose.x, pose.y, pose.yaw, pose.timestamp)

    def close(self) -> None:
        """停止 executor，并只在本类负责初始化 rclpy 时关闭全局 ROS context。"""

        self.running = False
        if self.executor is not None:
            try:
                self.executor.wake()
            except Exception:
                pass
        if self.spin_thread is not None:
            self.spin_thread.join(timeout=1.0)
        if self.executor is not None and self.node is not None:
            try:
                self.executor.remove_node(self.node)
            except Exception:
                pass
        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
        if self._owns_rclpy and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
