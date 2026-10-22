import sys
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unified_vln.real_input_guard import RealInputGuard
from unified_vln.odometry import Pose2D


class RealInputGuardTest(unittest.TestCase):
    def fixture(self):
        backend=MagicMock()
        camera=SimpleNamespace(input_error=None,last_metadata={'client_received_monotonic_s':time.monotonic()})
        pose=SimpleNamespace(get_pose=lambda:Pose2D(0,0,0,time.monotonic()))
        low=SimpleNamespace(get_yaw=lambda:0.)
        return backend,camera,pose,low

    def test_stalled_camera_during_follow_latches_stop_and_blocks_late_http_command(self):
        backend,camera,pose,low=self.fixture()
        guard=RealInputGuard(backend,camera,pose,low)
        try:
            guard.set_velocity(.6,0,0)
            camera.last_metadata['client_received_monotonic_s']-=2.
            guard.thread.join(timeout=1.)
            self.assertIn('camera input stale',guard.error)
            backend.stop.assert_called_once()
            with self.assertRaisesRegex(RuntimeError,'latched'):
                guard.set_velocity(.6,0,0)
            self.assertEqual(backend.set_velocity.call_args.args,(0.,0.,0.))
        finally:guard.close()

    def test_panorama_rotation_does_not_require_continuous_rgb_but_requires_lowstate(self):
        backend,camera,pose,low=self.fixture()
        with patch.object(threading.Thread,'start'):
            guard=RealInputGuard(backend,camera,pose,low)
        camera.last_metadata['client_received_monotonic_s']-=10.
        guard.set_velocity(0,0,.8)
        guard.check()
        low.get_yaw=lambda:None
        with self.assertRaisesRegex(RuntimeError,'lowstate'):
            guard.check()

    def test_joint_camera_error_is_not_hidden_by_episode_replan_retry(self):
        backend,camera,pose,low=self.fixture()
        camera.input_error='joint/image mismatch'
        guard=RealInputGuard(backend,camera,pose,low)
        try:
            guard.thread.join(timeout=1.)
            self.assertEqual(guard.error,'joint/image mismatch')
            with self.assertRaises(RuntimeError):guard.set_velocity(0,0,.8)
        finally:guard.close()

    def test_slam_jump_is_checked_while_main_loop_is_blocked(self):
        backend,camera,pose,low=self.fixture()
        poses=iter([Pose2D(0,0,0,0),Pose2D(.4,0,0,0)])
        pose.get_pose=lambda:next(poses)
        guard=RealInputGuard(backend,camera,pose,low)
        try:
            guard.thread.join(timeout=1.)
            self.assertEqual(guard.error,'SLAM pose jumped')
        finally:guard.close()

    def test_manual_stop_wakes_independent_thread_and_rejects_late_plan_command(self):
        backend,camera,pose,low = self.fixture()
        stop_started,release = threading.Event(),threading.Event()
        def slow_rpc():
            stop_started.set()
            release.wait(2.)
        backend.stop.side_effect = slow_rpc
        guard = RealInputGuard(backend,camera,pose,low)
        try:
            guard.set_velocity(.6,0,0)
            # 模拟SIGINT发生在主线程持锁期间：此方法必须不取该锁也不等SDK。
            with guard.lock:
                guard.request_shutdown_stop('user_interrupt')
            self.assertTrue(stop_started.wait(1.),'independent stop worker did not run')
            with self.assertRaisesRegex(RuntimeError,'latched'):
                guard.set_velocity(.6,0,0)
            self.assertEqual(backend.set_velocity.call_args.args,(0.,0.,0.))
            release.set()
            guard.thread.join(timeout=1.)
            guard.stop()  # finally收尾不重复等待另一个同类RPC
            backend.stop.assert_called_once()
            times = guard.guard_report()['manual_shutdown_times']
            self.assertLessEqual(times['signal_requested_monotonic_s'],times['zero_target_monotonic_s'])
            self.assertLessEqual(times['zero_target_monotonic_s'],times['stop_call_returned_monotonic_s'])
        finally:
            release.set()
            guard.close()


if __name__=='__main__':unittest.main()
