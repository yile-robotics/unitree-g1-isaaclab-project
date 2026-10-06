import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_vln.real_panorama_imu import LowStateYaw


class RealPanoramaImuTest(unittest.TestCase):
    def reader(self):
        reader = LowStateYaw.__new__(LowStateYaw)
        reader.timeout_s = 0.5
        reader._lock = threading.Lock()
        reader._yaw = None
        reader._received = 0.0
        reader._subscriber = Mock()
        return reader

    def test_lowstate_yaw_expires_and_does_not_read_robot_position(self):
        reader = self.reader()
        class LowState:
            imu_state = SimpleNamespace(rpy=[0.0, 0.0, 1.2])
            @property
            def position(self):
                raise AssertionError("IMU is not world position")
        with patch("unified_vln.real_panorama_imu.time.monotonic", return_value=10.0):
            reader._on_state(LowState())
            self.assertEqual(reader.get_yaw(), 1.2)
        with patch("unified_vln.real_panorama_imu.time.monotonic", return_value=10.6):
            self.assertIsNone(reader.get_yaw())

    def test_invalid_messages_do_not_refresh_last_valid_yaw(self):
        reader = self.reader()
        reader._on_state(SimpleNamespace(imu_state=SimpleNamespace(rpy=[0, 0, math.nan])))
        self.assertIsNone(reader.get_yaw())
        reader._on_state(SimpleNamespace())
        self.assertIsNone(reader.get_yaw())

    def test_close_releases_subscriber_once(self):
        reader = self.reader()
        subscriber = reader._subscriber
        reader.close()
        reader.close()
        subscriber.Close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
