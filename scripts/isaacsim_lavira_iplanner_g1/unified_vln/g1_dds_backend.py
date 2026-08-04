from __future__ import annotations

"""Optional real-G1 DDS backend.

Imports are lazy so the Isaac Sim Python environment does not need unitree_sdk2py.
The same object supplies velocity commands and odometry/IMU state.
"""

import threading
import time

from .odometry import Pose2D


class UnitreeG1DDSBackend:
    def __init__(
        self,
        network_interface: str,
        *,
        state_timeout_s: float = 0.5,
        command_ttl_s: float = 0.2,
        command_rate_hz: float = 50.0,
    ):
        if not network_interface.strip():
            raise ValueError("Unitree network interface must not be empty.")
        if min(state_timeout_s, command_ttl_s, command_rate_hz) <= 0.0:
            raise ValueError("DDS timeout/rate values must be positive.")
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

        self.state_timeout_s = float(state_timeout_s)
        self.command_ttl_s = float(command_ttl_s)
        self.command_period_s = 1.0 / float(command_rate_hz)
        self.lock = threading.Lock()
        self.latest_pose: Pose2D | None = None
        self.target_command = (0.0, 0.0, 0.0)
        self.last_command_update = 0.0
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
        try:
            pose = Pose2D(
                x=float(message.position[0]),
                y=float(message.position[1]),
                yaw=float(message.imu_state.rpy[2]),
                timestamp=time.monotonic(),
            ).validated()
        except Exception:
            return
        with self.lock:
            self.latest_pose = pose

    def get_pose(self) -> Pose2D | None:
        with self.lock:
            pose = self.latest_pose
        if pose is None or time.monotonic() - pose.timestamp > self.state_timeout_s:
            return None
        return pose

    def set_velocity(self, vx: float, vy: float, wz: float) -> None:
        now = time.monotonic()
        with self.lock:
            self.target_command = (float(vx), float(vy), float(wz))
            self.last_command_update = now

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0, 0.0)
        try:
            self.client.StopMove()
        except Exception:
            pass

    def _command_loop(self) -> None:
        while self.running:
            now = time.monotonic()
            with self.lock:
                command = self.target_command
                fresh = now - self.last_command_update <= self.command_ttl_s
            if not fresh:
                command = (0.0, 0.0, 0.0)
            try:
                self.client.Move(*command)
            except Exception:
                pass
            time.sleep(self.command_period_s)

    def close(self) -> None:
        self.running = False
        self.stop()
        self.command_thread.join(timeout=1.0)
