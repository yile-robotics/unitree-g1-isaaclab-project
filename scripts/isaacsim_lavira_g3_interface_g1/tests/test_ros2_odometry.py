from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.ros2_odometry import Ros2OdometryProvider  # noqa: E402


def _message(x: float, y: float, z: float, yaw: float):
    orientation = SimpleNamespace(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )
    return SimpleNamespace(
        header=SimpleNamespace(frame_id="odom"),
        child_frame_id="base_link",
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=x, y=y, z=z),
                orientation=orientation,
            )
        ),
    )


class Ros2OdometryProviderTest(unittest.TestCase):
    def test_message_becomes_fresh_planar_pose(self):
        provider = Ros2OdometryProvider(start_node=False)
        provider.ingest_odometry(
            _message(1.25, -2.5, 0.2, math.pi / 2.0),
            received_time=time.monotonic(),
        )
        pose = provider.get_pose()
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose.x, 1.25)
        self.assertAlmostEqual(pose.y, -2.5)
        self.assertAlmostEqual(pose.yaw, math.pi / 2.0)
        self.assertEqual(provider.frame_id, "odom")
        self.assertEqual(provider.child_frame_id, "base_link")

    def test_large_coordinates_use_first_sample_as_relative_origin(self):
        provider = Ros2OdometryProvider(start_node=False)
        now = time.monotonic()
        provider.ingest_odometry(_message(2000.0, 3000.0, 0.0, 0.0), received_time=now)
        first = provider.get_pose()
        self.assertAlmostEqual(first.x, 0.0)
        self.assertAlmostEqual(first.y, 0.0)
        provider.ingest_odometry(_message(2001.0, 3002.0, 0.0, 0.0), received_time=now)
        second = provider.get_pose()
        self.assertAlmostEqual(second.x, 1.0)
        self.assertAlmostEqual(second.y, 2.0)

    def test_divergent_z_disables_provider_for_session(self):
        provider = Ros2OdometryProvider(start_node=False)
        provider.ingest_odometry(
            _message(0.0, 0.0, 5.1, 0.0),
            received_time=time.monotonic(),
        )
        self.assertIsNone(provider.get_pose())
        self.assertIn("SLAM divergence", provider.disabled_reason)

        provider.ingest_odometry(
            _message(1.0, 1.0, 0.0, 0.0),
            received_time=time.monotonic(),
        )
        self.assertIsNone(provider.get_pose())

    def test_stale_message_returns_none(self):
        provider = Ros2OdometryProvider(pose_timeout_s=0.1, start_node=False)
        provider.ingest_odometry(_message(1.0, 2.0, 0.0, 0.0), received_time=0.0)
        self.assertIsNone(provider.get_pose())


if __name__ == "__main__":
    unittest.main()
