"""真机全景单独订阅rt/lowstate的IMU，保持与已通过的旋转探针相同来源。"""
import math
import threading
import time


class LowStateYaw:
    def __init__(self, timeout_s=0.5):
        # DDS工厂已由真机后端初始化；此处不创建运控客户端。
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._yaw = None
        self._received = 0.0
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        try:
            self._subscriber.Init(self._on_state, 1)
        except Exception:
            self._subscriber.Close()
            raise

    def _on_state(self, message):
        try:
            yaw = float(message.imu_state.rpy[2])
            if not math.isfinite(yaw):
                return
        except (AttributeError, IndexError, ValueError, TypeError):
            return
        with self._lock:
            self._yaw = yaw
            self._received = time.monotonic()

    def get_yaw(self):
        with self._lock:
            yaw, received = self._yaw, self._received
        if yaw is None or time.monotonic() - received > self.timeout_s:
            return None
        return yaw

    def wait_ready(self, timeout_s=3.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.get_yaw() is not None:
                return
            time.sleep(0.05)
        raise RuntimeError("No fresh rt/lowstate IMU for real panorama within 3 seconds.")

    def close(self):
        if self._subscriber is not None:
            self._subscriber.Close()
            self._subscriber = None
