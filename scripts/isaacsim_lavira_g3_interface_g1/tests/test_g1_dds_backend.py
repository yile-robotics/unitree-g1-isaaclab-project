from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.g1_dds_backend import UnitreeG1DDSBackend  # noqa: E402


class G1DDSBackendStateTest(unittest.TestCase):
    @staticmethod
    def _backend() -> UnitreeG1DDSBackend:
        # 绕过真实 SDK/DDS 初始化，只测试回调的数据职责和过期逻辑。
        backend = UnitreeG1DDSBackend.__new__(UnitreeG1DDSBackend)
        backend.lock = threading.Lock()
        backend.latest_yaw_rad = None
        backend.latest_yaw_time = 0.0
        backend.imu_timeout_s = 1.0
        return backend

    def test_callback_uses_imu_yaw_and_never_reads_position(self):
        class Message:
            imu_state = SimpleNamespace(rpy=[0.0, 0.0, 0.75])

            @property
            def position(self):
                raise AssertionError("SportModeState.position must not be odometry")

        backend = self._backend()
        backend._state_callback(Message())
        self.assertAlmostEqual(backend.get_yaw(), 0.75)

    def test_non_finite_or_stale_yaw_is_unavailable(self):
        backend = self._backend()
        backend._state_callback(
            SimpleNamespace(imu_state=SimpleNamespace(rpy=[0.0, 0.0, math.nan]))
        )
        self.assertIsNone(backend.get_yaw())

        backend.latest_yaw_rad = 0.5
        backend.latest_yaw_time = time.monotonic() - 2.0
        self.assertIsNone(backend.get_yaw())

    def test_stop_matches_uni_lavira_zero_stopmove_and_settle(self):
        backend = self._backend()
        backend.client = MagicMock()
        backend.target_command = (0.3, 0.0, 0.2)

        with patch("unified_vln.g1_dds_backend.time.sleep") as sleep:
            backend.stop()

        self.assertEqual(backend.target_command, (0.0, 0.0, 0.0))
        backend.client.StopMove.assert_called_once_with()
        sleep.assert_called_once_with(0.2)

    def test_high_stand_uses_installed_sdk_wrapper_and_waits_one_second(self):
        backend = self._backend()
        backend.client = MagicMock()

        with patch("unified_vln.g1_dds_backend.time.sleep") as sleep:
            backend.high_stand()

        backend.client.HighStand.assert_called_once_with()
        sleep.assert_called_once_with(1.0)

    def test_high_stand_failure_is_not_reported_as_ready(self):
        backend = self._backend()
        backend.client = MagicMock()
        backend.client.HighStand.side_effect = OSError("DDS unavailable")

        with self.assertRaisesRegex(RuntimeError, "HighStand request failed"):
            backend.high_stand()

    def test_command_loop_repeats_last_command_without_ttl(self):
        backend = self._backend()
        backend.client = MagicMock()
        backend.target_command = (0.3, 0.0, 0.1)
        backend.command_period_s = 0.02
        backend.running = True

        def stop_after_first_send(_period: float) -> None:
            backend.running = False

        with patch(
            "unified_vln.g1_dds_backend.time.sleep",
            side_effect=stop_after_first_send,
        ):
            backend._command_loop()

        backend.client.Move.assert_called_once_with(0.3, 0.0, 0.1)


if __name__ == "__main__":
    unittest.main()
