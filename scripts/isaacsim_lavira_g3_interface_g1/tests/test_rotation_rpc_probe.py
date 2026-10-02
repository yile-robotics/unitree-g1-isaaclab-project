"""离线核对实际发送命令；SDK、时钟和姿态均替换为假对象，不连接机器人。"""

import contextlib
import io
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g1_rotation_rpc_probe as probe


class RotationRpcProbeTests(unittest.TestCase):
    def run_probe(self, target, stalled=False, observe_slam=False, degrees=90):
        clock = SimpleNamespace(now=0.0)
        calls = []

        def yaw():
            return math.radians(min(10.0 if stalled else 120.0, clock.now * 30.0))

        class Client:
            def SetTimeout(self, value): pass
            def Init(self): pass
            def GetFsmId(self): return 0, 802
            def SetVelocity(self, vx, vy, wz, duration=1.0):
                calls.append((clock.now, vx, vy, wz, duration))
                return 0
            def StopMove(self): self.SetVelocity(0.0, 0.0, 0.0)

        class Subscriber:
            def __init__(self, *args): pass
            def Init(self, callback, *args): pass
            def Close(self): pass

        class Monitor:
            def on_state(self, message): pass
            def require_fresh(self): return yaw()

        class Odom:
            frame_id = "map"
            child_frame_id = "base_link"
            def __init__(self, **kwargs): pass
            def get_pose(self): return SimpleNamespace(yaw=yaw())
            def close(self): pass

        class SlamMonitor:
            def on_message(self, message): pass
            def snapshot(self):
                half_yaw = yaw() * 0.45
                pose = {
                    "q_x": 0.0, "q_y": 0.0,
                    "q_z": math.sin(half_yaw), "q_w": math.cos(half_yaw),
                }
                return pose, clock.now, ""

        modules = {
            "unitree_sdk2py.core.channel": SimpleNamespace(
                ChannelFactoryInitialize=lambda *a: None, ChannelSubscriber=Subscriber),
            "unitree_sdk2py.g1.loco.g1_loco_client": SimpleNamespace(LocoClient=Client),
            "unitree_sdk2py.idl.unitree_hg.msg.dds_": SimpleNamespace(LowState_=object),
            "unitree_sdk2py.idl.std_msgs.msg.dds_": SimpleNamespace(String_=object),
            "unified_vln.ros2_odometry": SimpleNamespace(Ros2OdometryProvider=Odom),
            "monitor_g1_slam_info": SimpleNamespace(PoseMonitor=SlamMonitor),
        }
        argv = ["probe", "--network-interface", "fake", "--wz", "0.6",
                "--seconds", "20", target, str(degrees), "--execute"]
        if observe_slam:
            argv.append("--observe-slam")
        fake_time = SimpleNamespace(
            monotonic=lambda: clock.now,
            sleep=lambda dt: setattr(clock, "now", clock.now + dt),
        )
        output = io.StringIO()
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                patch.object(probe, "time", fake_time), \
                patch.object(probe, "StateMonitor", Monitor), \
                contextlib.redirect_stdout(output):
            result = probe.main()
        return result, calls, output.getvalue()

    def test_imu_90_continues_past_45_without_intermediate_zero(self):
        result, calls, _ = self.run_probe("--target-imu-deg")
        moving = [c for c in calls if c[3] != 0]
        stopped = [c for c in calls if c[3] == 0]
        self.assertEqual(result, 0)
        self.assertGreater(moving[-1][0], 2.8)
        self.assertGreaterEqual(stopped[0][0], 3.0)
        self.assertTrue(all(c[1:] == (0.0, 0.0, 0.6, 0.3) for c in moving))
        self.assertGreater(stopped[0][0], moving[-1][0])

    def test_stalled_map_10_keeps_nonzero_command_until_time_limit(self):
        result, calls, _ = self.run_probe("--target-map-deg", stalled=True)
        moving = [c for c in calls if c[3] != 0]
        stopped = [c for c in calls if c[3] == 0]
        self.assertEqual(result, 1)
        self.assertGreater(len(moving), 190)
        self.assertGreater(moving[-1][0], 19.8)
        self.assertGreaterEqual(stopped[0][0], 20.0)
        self.assertTrue(all(c[1:] == (0.0, 0.0, 0.6, 0.3) for c in moving))

    def test_slam_observer_does_not_change_imu_stop(self):
        result, calls, output = self.run_probe("--target-imu-deg", observe_slam=True)
        moving = [c for c in calls if c[3] != 0]
        stopped = [c for c in calls if c[3] == 0]
        self.assertEqual(result, 0)
        self.assertGreaterEqual(stopped[0][0], 3.0)
        self.assertGreater(stopped[0][0], moving[-1][0])
        self.assertIn("IMU reached +90° limit", output)
        self.assertIn("SLAM yaw after stop", output)

    def test_adjusted_imu_target_75_stops_at_requested_angle(self):
        result, calls, output = self.run_probe("--target-imu-deg", degrees=75)
        moving = [c for c in calls if c[3] != 0]
        stopped = [c for c in calls if c[3] == 0]
        self.assertEqual(result, 0)
        self.assertGreaterEqual(stopped[0][0], 2.5)
        self.assertLess(stopped[0][0], 2.7)
        self.assertGreater(stopped[0][0], moving[-1][0])
        self.assertIn("IMU reached +75° limit", output)


if __name__ == "__main__":
    unittest.main()
