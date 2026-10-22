"""真机导航转向复用全景策略；假IMU、SLAM、运控，无硬件运动。"""
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unified_vln.real_navigation_rotation import RealNavigationRotation
from unified_vln.episode import EpisodeState, LocalEndToEndEpisode
from unified_vln.rotation import TimedFixedSpeedRotation
import test_real_episode_panorama as panorama_tests


class RealNavigationRotationTest(unittest.TestCase):
    def fixture(self):
        state=SimpleNamespace(now=0.,yaw=math.radians(170),imu=math.radians(170),stops=0)
        def stop():
            state.stops+=1
            state.yaw+=rotation.sign*math.radians(14)
            state.imu+=rotation.sign*math.radians(14)
            state.now+=.2
        rotation=RealNavigationRotation(.8,1.4,yaw_reader=lambda:state.yaw,
            imu_reader=lambda:state.imu,clock=lambda:state.now,on_stop=stop)
        return rotation,state

    def test_left_uses_point_eight_and_stops_at_imu75_then_waits_before_completion(self):
        rotation,state=self.fixture()
        rotation.start('left')
        command=rotation.update(30.)
        self.assertEqual(rotation.elapsed_s,0.)
        self.assertEqual(command.wz,.8)
        state.imu+=math.radians(74);state.yaw+=math.radians(74);state.now=3.
        self.assertEqual(rotation.update(.05).wz,.8)
        self.assertEqual(state.stops,0)
        state.imu+=math.radians(1);state.yaw+=math.radians(1);state.now=3.1
        command=rotation.update(.05)
        self.assertEqual(command.wz,0.)
        self.assertFalse(command.done)
        self.assertEqual(state.stops,1)
        state.now+=.49
        self.assertFalse(rotation.update(.05).done)
        state.now+=.02
        self.assertTrue(rotation.update(.05).done)
        self.assertAlmostEqual(rotation.verify_settled('left')['settled_error_deg'],-1.)
        self.assertAlmostEqual(rotation.snapshot()['quarter_turns'][0]['slam_deg'],89.)

    def test_stall_timeout_cannot_return_success(self):
        rotation,state=self.fixture()
        rotation.start('left');rotation.update(.05)
        state.yaw+=math.radians(5.7);state.imu+=math.radians(5.7);state.now=10.1
        with self.assertRaisesRegex(RuntimeError,'timed out'):rotation.update(.05)
        self.assertTrue(rotation.feedback_timed_out)
        self.assertEqual(rotation.status,'FAILED')
        with self.assertRaises(RuntimeError):rotation.verify_settled('left')

    def test_lost_either_sensor_never_falls_back_to_timed_turn(self):
        for sensor in ('yaw','imu'):
            with self.subTest(sensor=sensor):
                rotation,state=self.fixture();rotation.start('left');rotation.update(.05)
                setattr(state,sensor,None);state.now=.1
                with self.assertRaisesRegex(RuntimeError,'lost fresh'):rotation.update(.05)
                self.assertEqual(rotation.status,'FAILED')

    def test_imu75_alone_does_not_authorize_walk_when_slam_only_moved_a_few_degrees(self):
        rotation,state=self.fixture();rotation.start('left');rotation.update(.05)
        state.imu+=math.radians(75);state.yaw+=math.radians(5);state.now=3.
        self.assertFalse(rotation.update(.05).done)
        state.now+=.6
        with self.assertRaisesRegex(RuntimeError,'outside 70..110'):rotation.update(.05)

    def test_forward_skips_turn_and_behind_completes_two_right_quarters(self):
        rotation,state=self.fixture();rotation.start('forward')
        self.assertTrue(rotation.update(.1).done);self.assertEqual(state.stops,0)
        for direction,count in [('right',1),('behind',2)]:
            with self.subTest(direction=direction):
                rotation,state=self.fixture();rotation.start(direction)
                for quarter in range(count):
                    self.assertEqual(rotation.update(.05).wz,-.8)
                    state.imu-=math.radians(75);state.yaw-=math.radians(75);state.now+=3.
                    self.assertFalse(rotation.update(.05).done)
                    state.now+=.6
                    result=rotation.update(.05)
                    self.assertEqual(result.done,quarter==count-1)
                self.assertEqual(state.stops,count)
                self.assertAlmostEqual(rotation.verify_settled(direction)['settled_error_deg'],count)

    def test_wrong_direction_stops_as_failure(self):
        rotation,state=self.fixture();rotation.start('left');rotation.update(.05)
        state.imu-=math.radians(21);state.now=.5
        with self.assertRaisesRegex(RuntimeError,'wrong direction'):rotation.update(.05)

    def test_episode_timeout_stops_and_never_calls_planner(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode,state,tick=panorama_tests.RealPanoramaTest().fixture(output=Path(tmp))
            episode.state=EpisodeState.WAIT_ROTATION_LOCOMOTION
            episode.pending=SimpleNamespace(response=SimpleNamespace(direction='left'),post_rotation_forward=None)
            episode.rotation.start('left');result=tick()
            self.assertEqual(result.command[2],.8)
            state.command.fill(0.);state.yaw+=math.radians(5.7);state.now+=10.1
            with patch.object(LocalEndToEndEpisode,'_capture_forward_and_plan') as plan:result=tick()
            plan.assert_not_called();self.assertEqual(episode.state,EpisodeState.FAILED)
            np.testing.assert_allclose(result.command,np.zeros(3));self.assertTrue(state.stops)
            record=json.loads((episode.output_dir/'decision_000_navigation_rotation.json').read_text())
            self.assertTrue(record['feedback_timed_out'])

    def test_episode_stops_settles_checks_angle_then_plans(self):
        episode,state,tick=panorama_tests.RealPanoramaTest().fixture()
        episode.state=EpisodeState.WAIT_ROTATION_LOCOMOTION
        episode.pending=SimpleNamespace(response=SimpleNamespace(direction='left'),post_rotation_forward=None)
        episode.rotation.start('left');tick()
        state.yaw+=math.radians(75)
        with patch.object(LocalEndToEndEpisode,'_capture_forward_and_plan') as plan:
            result=tick()
            self.assertEqual(episode.rotation.status,'SETTLING')
            np.testing.assert_allclose(result.command,np.zeros(3))
            for _ in range(9):tick()
            plan.assert_not_called()
            for _ in range(5):
                tick()
                if plan.called:break
            plan.assert_called_once()
            self.assertEqual(episode.rotation.status,'SETTLED_VERIFIED')

    def test_simulation_keeps_original_rotation(self):
        episode=LocalEndToEndEpisode(panorama_tests.EpisodeConfig('sim','left'),
            panorama_tests.LocalFollowerConfig(),camera=None,model=None,planner=None)
        self.assertIs(type(episode.rotation),TimedFixedSpeedRotation)
        self.assertEqual(episode.rotation.speed_rad_s,.4)


if __name__=='__main__':unittest.main()
