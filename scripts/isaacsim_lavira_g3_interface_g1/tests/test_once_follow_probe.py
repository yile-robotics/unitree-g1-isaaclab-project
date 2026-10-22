"""单次诊断入口的运动隔离和停车测试；全部使用替身，不连接机器人。"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import g1_once_follow_probe as probe
from unified_vln.odometry import Pose2D
from unified_vln.types import ViewFrame


def frame():
    return ViewFrame('forward',1,0,0.,np.zeros((8,12,3),np.uint8),np.ones((8,12),np.float32),
                     np.array([[10.,0,6],[0,10.,4],[0,0,1]]))


class OnceFollowTests(unittest.TestCase):
    def test_check_only_never_constructs_motion_backend_or_rotation_process(self):
        camera=MagicMock();camera.capture_forward.return_value=frame()
        provider=MagicMock();provider.frame_id='map';provider.child_frame_id='base_link'
        provider.get_pose.return_value=Pose2D(0,0,0,time.monotonic())
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(probe,'UnitreeG1DDSBackend') as dds, \
             patch.object(probe.subprocess,'run') as rotate, \
             patch.object(probe,'Ros2OdometryProvider',return_value=provider), \
             patch.object(probe,'create_camera',return_value=camera), \
             patch.object(probe,'IPlannerClient') as planner, \
             patch.object(probe.G3SessionClient,'from_decision_url'), \
             contextlib.redirect_stdout(io.StringIO()):
            planner.return_value.reset.return_value=True
            output=Path(tmp)/'check'
            self.assertEqual(probe.main(['--output',str(output)]),0)
            self.assertFalse(json.loads((output/'follow_report.json').read_text())['robot_motion_sent'])
            dds.assert_not_called();rotate.assert_not_called()
            camera.close.assert_called_once();provider.close.assert_called_once()

    def test_replay_only_never_opens_ros_camera_or_motion_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'recorded';(source/'panorama').mkdir(parents=True)
            (source/'panorama/panorama.json').write_text(json.dumps({'return_view':{
                'frame_id':1,'sim_step':0,'timestamp':0.,'K':frame().K.tolist()}}))
            projection=SimpleNamespace(goal_after_turn_xy_m=np.array([2.,0]),to_dict=lambda:{})
            with patch.object(probe,'load_decision',return_value=projection), \
                 patch.object(probe.cv2,'imread',return_value=frame().rgb), \
                 patch.object(probe.np,'load',return_value=frame().depth_m), \
                 patch.object(probe,'IPlannerClient') as planner, \
                 patch.object(probe,'UnitreeG1DDSBackend') as dds, \
                 patch.object(probe,'Ros2OdometryProvider') as ros, \
                 patch.object(probe,'create_camera') as camera, \
                 patch.object(probe.subprocess,'run') as rotate:
                planner.return_value.reset.return_value=True
                planner.return_value.get_plan.return_value=(np.array([[0,0,0],[1,0,0],[2,0,0]]),.1)
                output=Path(tmp)/'replay'
                self.assertEqual(probe.main(['--replay-run',str(source),'--output',str(output)]),0)
                report=json.loads((output/'follow_report.json').read_text())
                self.assertEqual(report['status'],'REPLAY_ONLY')
                self.assertFalse(report['robot_motion_sent'])
                self.assertEqual(report['controller_config']['tracking_controller'],'pure_pursuit')
                for tool in (dds,ros,camera,rotate):tool.assert_not_called()

    def test_invalid_test_limits_rejected_before_hardware(self):
        with patch.object(probe,'UnitreeG1DDSBackend') as dds, contextlib.redirect_stderr(io.StringIO()):
            for args in (['--follow-seconds','nan'],['--follow-seconds','7'],['--max-travel-m','0.6'],['--max-travel-m','0'],
                         ['--complete-local-follow','--follow-seconds','nan'],
                         ['--complete-local-follow','--follow-seconds','61'],
                         ['--complete-local-follow','--max-travel-m','0.5']):
                with self.subTest(args=args), self.assertRaises(SystemExit):probe.main(args)
            dds.assert_not_called()

    def test_invalid_planner_path_is_rejected(self):
        for path in (None,[[0,0,0]],[[0,0,0],[float('nan'),0,0]]):
            with self.subTest(path=path),self.assertRaises(ValueError):probe.checked_path(path)

    def test_execute_refreshes_camera_after_dds_setup_before_follow(self):
        camera=MagicMock();camera.last_metadata={'client_received_monotonic_s':time.monotonic()-5.}
        captures=[]
        def capture(*args):
            captures.append(len(captures)+1)
            camera.last_metadata={'client_received_monotonic_s':time.monotonic()}
            return frame()
        camera.capture_forward.side_effect=capture
        provider=MagicMock();provider.frame_id='map';provider.child_frame_id='base_link'
        provider.get_pose.side_effect=lambda:Pose2D(0,0,0,time.monotonic())
        def initialize_dds(*args):
            # 模拟DDS初始化耗掉旧规划帧的新鲜度余量。
            camera.last_metadata={'client_received_monotonic_s':time.monotonic()-5.}
            return MagicMock()
        def follow(*args,**kwargs):
            self.assertEqual(len(captures),3)  # 预检查、首次规划、DDS就绪后新图。
            self.assertLess(time.monotonic()-camera.last_metadata['client_received_monotonic_s'],1.)
            self.assertIsNone(args[7].max_travel_m)
            self.assertEqual(args[7].follow_seconds,60.)
            return 'TIME_LIMIT',[]
        channel=SimpleNamespace(ChannelSubscriber=MagicMock())
        projection=SimpleNamespace(goal_after_turn_xy_m=np.array([2.,0]),to_dict=lambda:{})
        path=np.array([[0.,0,0],[2.,0,0]])
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(sys.modules,{'unitree_sdk2py.core.channel':channel}), \
             patch.object(probe,'UnitreeG1DDSBackend',side_effect=initialize_dds), \
             patch.object(probe.subprocess,'run'), \
             patch.object(probe,'Ros2OdometryProvider',return_value=provider), \
             patch.object(probe,'create_camera',return_value=camera), \
             patch.object(probe,'IPlannerClient'), \
             patch.object(probe,'StateMonitor'), \
             patch.object(probe.G3SessionClient,'from_decision_url'), \
             patch.object(probe,'load_decision',return_value=projection), \
             patch.object(probe,'initial_plan',return_value=(path,path,.1)), \
             patch.object(probe,'follow',side_effect=follow) as following, \
             contextlib.redirect_stdout(io.StringIO()):
            output=Path(tmp)/'execute'
            self.assertEqual(probe.main(['--execute','--complete-local-follow','--decision-display-s','0','--output',str(output)]),0)
            following.assert_called_once()
            report=json.loads((output/'follow_report.json').read_text())
            self.assertEqual(report['follow_limits'],{'timeout_s':60.,'slam_displacement_m':None})

    def test_refresh_does_not_relabel_old_frames_as_fresh(self):
        camera=MagicMock();camera.capture_forward.return_value=frame()
        camera.last_metadata={'client_received_monotonic_s':time.monotonic()-5.}
        odom=MagicMock();odom.get_pose.return_value=Pose2D(0,0,0,0)
        with self.assertRaisesRegex(RuntimeError,'图像已过期'):
            probe.refresh_follow_camera(camera,odom,Pose2D(0,0,0,0))
        camera.capture_forward.assert_called_once()

    def test_refresh_rejects_disconnection_and_movement_during_capture(self):
        camera=MagicMock();odom=MagicMock()
        camera.capture_forward.side_effect=ConnectionError('stream disconnected')
        with self.assertRaises(ConnectionError):
            probe.refresh_follow_camera(camera,odom,Pose2D(0,0,0,0))
        def capture(*args):
            camera.last_metadata={'client_received_monotonic_s':time.monotonic()}
            return frame()
        camera.capture_forward.side_effect=capture
        odom.get_pose.return_value=Pose2D(.1,0,0,0)
        with self.assertRaisesRegex(RuntimeError,'位姿变化'):
            probe.refresh_follow_camera(camera,odom,Pose2D(0,0,0,0))

    def test_safety_stop_response_does_not_become_navigation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model').mkdir()
            (root/'flow_report.json').write_text(json.dumps({'status':'PASS','rotation_capture_completed':True}))
            (root/'model/response.json').write_text(json.dumps({'control':'SAFE_STOP','action':'NAVIGATE'}))
            with self.assertRaisesRegex(ValueError,'停止/恢复'):probe.load_decision(root)

    def test_watchdog_stops_on_stale_data_and_rejects_late_velocity(self):
        dds=MagicMock();odom=MagicMock();odom.get_pose.return_value=Pose2D(0,0,0,0)
        low=MagicMock();low.require_fresh.side_effect=RuntimeError('lowstate stale')
        guard=probe.FollowWatchdog(dds,odom,low,[time.monotonic()],3.,.5)
        try:
            self.assertTrue(guard.stop.wait(timeout=1))
            guard.thread.join(timeout=1)
            self.assertIn('DATA_FAILURE',guard.reason)
            self.assertFalse(guard.send(np.array([.3,0,0])))
            dds.set_velocity.assert_called_once_with(0.,0.,0.)
            dds.stop.assert_called_once()
        finally:guard.close()

    def test_watchdog_distance_and_time_limits(self):
        for mode in ('distance','time'):
            dds=MagicMock();low=MagicMock();odom=MagicMock()
            positions=iter([0.,.1,.2,.3,.4,.5,.5])
            odom.get_pose.side_effect=lambda:Pose2D(next(positions,.5) if mode=='distance' else 0.,0,0,0)
            guard=probe.FollowWatchdog(dds,odom,low,[time.monotonic()],.01 if mode=='time' else 3.,.5)
            try:
                self.assertTrue(guard.stop.wait(timeout=1))
                guard.thread.join(timeout=1)
                self.assertEqual(guard.reason,'DISTANCE_LIMIT' if mode=='distance' else 'TIME_LIMIT')
                dds.stop.assert_called_once()
            finally:guard.close()

    def test_complete_follow_keeps_data_checks_and_timeout_without_distance_cap(self):
        for mode in ('timeout','camera_stale'):
            dds=MagicMock();low=MagicMock();odom=MagicMock()
            # 总位移超过短测的0.5m，单次增量仍小于定位跳变阈值。
            positions=iter([0.,.2,.4,.6,.8,1.])
            odom.get_pose.side_effect=lambda:Pose2D(next(positions,1.),0,0,0)
            camera_time=[time.monotonic()-(2. if mode=='camera_stale' else 0.)]
            guard=probe.FollowWatchdog(dds,odom,low,camera_time,.15,None)
            try:
                self.assertTrue(guard.stop.wait(timeout=1))
                guard.thread.join(timeout=1)
                expected='TIME_LIMIT' if mode=='timeout' else 'DATA_FAILURE: camera input stale'
                self.assertEqual(guard.reason,expected)
                self.assertFalse(guard.send(np.array([.6,0,0])))
                dds.set_velocity.assert_called_once_with(0.,0.,0.)
                dds.stop.assert_called_once()
            finally:guard.close()

    def test_complete_follow_uses_shared_goal_completion_beyond_half_metre(self):
        guard=SimpleNamespace(stop=MagicMock(),reason=None,started=time.monotonic(),close=MagicMock())
        guard.stop.is_set.return_value=False
        current_x=[0.]
        def send(command):
            current_x[0]+=.15
            return True
        guard.send=send
        camera=MagicMock();camera.last_metadata={'client_received_monotonic_s':time.monotonic()}
        odom=SimpleNamespace(get_pose=lambda:Pose2D(current_x[0],0,0,time.monotonic()))
        dds=MagicMock()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(probe,'FollowWatchdog',return_value=guard), \
             patch.object(probe.time,'time',return_value=1000.):
            reason,trace=probe.follow(dds,camera,odom,MagicMock(),MagicMock(),
                np.array([[0.,0,0],[1.,0,0],[2.,0,0],[3.,0,0]]),probe.LocalFollowerConfig(),
                SimpleNamespace(follow_seconds=60.,max_travel_m=None),Path(tmp))
            self.assertEqual(reason,'LOCAL_GOAL_REACHED')
            self.assertGreater(current_x[0],.5)
            self.assertGreater(len(trace),1)
            dds.stop.assert_called_once()
            self.assertEqual(json.loads((Path(tmp)/'follow_live.json').read_text())['state'],'STOPPED')

    def test_no_velocity_resumes_when_limit_occurs_during_replan(self):
        guard=SimpleNamespace(stop=threading.Event(),reason=None,started=time.monotonic(),
                              close=MagicMock(),send=MagicMock())
        camera=MagicMock();camera.last_metadata={'client_received_monotonic_s':time.monotonic()}
        camera.capture_forward.return_value=frame()
        follower=MagicMock();follower.update.return_value=SimpleNamespace(abort_reason=None,reached=False,
                                               command=np.array([.3,0,0]),distance_to_goal_m=2.)
        follower.needs_replan.return_value=True
        planner=MagicMock()
        def plan(*args):
            guard.reason='TIME_LIMIT';guard.stop.set()
            return np.array([[0.,0,0],[2.,0,0]]),.1
        planner.get_plan.side_effect=plan
        dds=MagicMock();odom=MagicMock();odom.get_pose.return_value=Pose2D(0,0,0,0)
        args=SimpleNamespace(follow_seconds=3.,max_travel_m=.5)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(probe,'FollowWatchdog',return_value=guard), \
             patch.object(probe,'LocalTrajectoryFollower',return_value=follower):
            reason,trace=probe.follow(dds,camera,odom,MagicMock(),planner,np.array([[0.,0,0],[2.,0,0]]),
                                     probe.LocalFollowerConfig(),args,Path(tmp))
            self.assertEqual(reason,'TIME_LIMIT');self.assertEqual(trace,[])
            guard.send.assert_not_called();dds.stop.assert_called_once();follower.stop.assert_called_once()

    def test_shared_follower_writes_actual_path_and_clears_it_after_stop(self):
        guard=SimpleNamespace(stop=threading.Event(),reason=None,started=time.monotonic(),close=MagicMock())
        sent=[]
        def send(command):
            sent.append(command.copy());guard.reason='TIME_LIMIT';guard.stop.set();return True
        guard.send=send
        camera=MagicMock();camera.last_metadata={'client_received_monotonic_s':time.monotonic()}
        odom=SimpleNamespace(get_pose=lambda:Pose2D(0,0,0,time.monotonic()))
        dds=MagicMock()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(probe,'FollowWatchdog',return_value=guard), \
             patch.object(probe,'write_json',wraps=probe.write_json) as write:
            reason,trace=probe.follow(dds,camera,odom,MagicMock(),MagicMock(),
                np.array([[0.,0,0],[1.,0,0],[2.,0,0]]),probe.LocalFollowerConfig(),
                SimpleNamespace(follow_seconds=3.,max_travel_m=.5),Path(tmp),initial_frame_id=23)
            self.assertEqual(reason,'TIME_LIMIT');self.assertEqual(len(sent),1)
            self.assertTrue(np.isfinite(sent[0]).all());self.assertGreater(sent[0][0],0)
            live=[call.args[1] for call in write.call_args_list if call.args[0].name=='.follow_live.tmp']
            self.assertEqual(len(live),1);self.assertEqual(live[0]['source_frame_id'],23)
            self.assertTrue(live[0]['executed_by_probe'])
            self.assertEqual(json.loads((Path(tmp)/'follow_live.json').read_text())['state'],'STOPPED')
            dds.stop.assert_called_once()


if __name__=='__main__':unittest.main()
