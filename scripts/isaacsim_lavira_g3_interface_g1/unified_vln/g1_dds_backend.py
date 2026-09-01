from __future__ import annotations

"""可选的真实 G1 机器人 DDS 后端。

Unitree SDK 采用延迟导入，因此只运行 Isaac Sim 时无需安装 ``unitree_sdk2py``。
同一个后端负责持续发送速度命令，并从 DDS 状态消息中提供 IMU yaw。与
Uni-LaViRA 真机实现一致，二维位置不读取 ``SportModeState_.position``；轨迹
所需的 ``(x, y, yaw)`` 应由 ROS 2 ``/Odometry`` 提供。
"""

import math
import threading
import time


class UnitreeG1DDSBackend:
    """连接真实 G1 的速度控制与状态订阅适配器。

    与 Uni-LaViRA 真机代码一致，后台线程以固定频率持续重复发送最新速度，直到
    上层明确更新速度或调用 ``stop()``。
    """

    def __init__(
        self,
        network_interface: str,
        *,
        imu_timeout_s: float = 1.0,
        command_rate_hz: float = 50.0,
    ):
        """初始化 DDS、运动客户端、状态订阅器和命令发送线程。"""

        if not network_interface.strip():
            raise ValueError("Unitree network interface must not be empty.")
        if min(imu_timeout_s, command_rate_hz) <= 0.0:
            raise ValueError("DDS timeout/rate values must be positive.")
        # 只有实例化真实机器人后端时才导入 SDK，不影响纯仿真环境。
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

        ChannelFactoryInitialize(0, network_interface)
        self.client = LocoClient()
        self.client.SetTimeout(10.0)
        self.client.Init()

        self.imu_timeout_s = float(imu_timeout_s)
        self.command_period_s = 1.0 / float(command_rate_hz)
        # DDS 回调线程、命令线程和主控制线程共享以下状态，必须加锁访问。
        self.lock = threading.Lock()
        self.latest_yaw_rad: float | None = None
        self.latest_yaw_time = 0.0
        self.target_command = (0.0, 0.0, 0.0)
        self.running = True

        self.subscriber = ChannelSubscriber(
            "rt/sportmodestate", SportModeState_
        )
        self.subscriber.Init(self._state_callback, 10)
        self.command_thread = threading.Thread(
            target=self._command_loop, daemon=True
        )
        self.command_thread.start()

    def _state_callback(self, message) -> None:
        """接收 ``SportModeState_``，只提取 Uni-LaViRA 使用的 IMU yaw。"""

        try:
            yaw = float(message.imu_state.rpy[2])
            if not math.isfinite(yaw):
                return
        except Exception:
            # 单个损坏或字段不完整的状态包直接丢弃，不让 DDS 回调线程退出。
            return
        with self.lock:
            self.latest_yaw_rad = yaw
            self.latest_yaw_time = time.monotonic()

    def get_yaw(self) -> float | None:
        """返回尚未过期的 IMU yaw；没有消息或超过 1 秒时返回 ``None``。"""

        with self.lock:
            yaw = self.latest_yaw_rad
            timestamp = self.latest_yaw_time
        if yaw is None or time.monotonic() - timestamp > self.imu_timeout_s:
            return None
        return float(yaw)

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        """更新目标速度，由后台线程负责持续发送。"""

        with self.lock:
            self.target_command = (float(vx), float(vy), float(wz))

    def stop(self) -> None:
        """把目标速度归零，并请求 Unitree 运动客户端停止移动。"""

        self.set_velocity(0.0, 0.0, 0.0)
        try:
            self.client.StopMove()
        except Exception:
            pass
        # 与 Uni-LaViRA RobotController.stop_robot() 一致，给高层运动服务一个
        # 很短的时间处理零速度/StopMove 请求，然后再继续关闭其他资源。
        time.sleep(0.2)

    def high_stand(self) -> None:
        """像 Uni-LaViRA 一样在导航开始前请求一次 G1 高站立姿态。"""

        try:
            # 当前本地 unitree_sdk2py 的 HighStand() 内部调用
            # SetStandHeight(UINT32_MAX)，不是切换另一套行走/站立 policy。
            self.client.HighStand()
        except Exception as exc:
            raise RuntimeError("Unitree G1 HighStand request failed.") from exc
        # 对齐 Uni-LaViRA：命令发出后等待 1 秒，再允许导航开始。
        time.sleep(1.0)

    def _command_loop(self) -> None:
        """后台发送循环；与 Uni-LaViRA 一样持续发送最近一次目标速度。"""

        while self.running:
            with self.lock:
                command = self.target_command
            try:
                self.client.Move(*command)
            except Exception:
                pass
            time.sleep(self.command_period_s)

    def close(self) -> None:
        """停止后台线程和机器人运动，释放该后端。"""

        self.running = False
        self.stop()
        self.command_thread.join(timeout=1.0)
