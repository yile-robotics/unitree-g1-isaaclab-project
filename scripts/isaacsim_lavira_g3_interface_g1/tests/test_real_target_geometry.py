"""三维反投影与实际转向的独立几何检查，无硬件。"""
from pathlib import Path
from types import SimpleNamespace
import math
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_vln.real_target_geometry import backproject_optical, build_target_geometry, update_target_for_pose
from unified_vln.odometry import Pose2D
from unified_vln.real_episode import RealG1Episode
from unified_vln.episode import EpisodeState, LocalEndToEndEpisode

# 无俯仰安装：光学右->机器人右，光学下->机器人下，光轴->机器人前。
R0 = np.array([[0,0,1],[-1,0,0],[0,-1,0]], dtype=float)


class TargetGeometryTest(unittest.TestCase):
    def fixture(self, direction='forward', pixel=(100,70), z=2.):
        frame=SimpleNamespace(direction=direction, frame_id=8,sim_step=3,timestamp=1.,
            K=np.array([[100.,0,100],[0,100.,50],[0,0,1]]))
        projection=SimpleNamespace(direction=direction, pixel_uv=pixel, depth_m=z,
            goal_after_turn_xy_m=np.array([2.,0.]),used_forward_fallback=False)
        T=np.eye(4);T[:3,:3]=R0;T[:3,3]=[.1,.05,.3]
        return frame,projection,T

    def test_backprojection_uses_vertical_pixel_and_optical_z(self):
        frame,_,_=self.fixture()
        np.testing.assert_allclose(backproject_optical((120,70),2.,frame.K),[.4,.4,2.])

    def test_pitch_correction_includes_vertical_ray_and_translation(self):
        frame,projection,T=self.fixture()
        theta=math.radians(20)
        Ry=np.array([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]])
        T[:3,:3]=Ry@R0
        result=build_target_geometry(frame,projection,T,Pose2D(0,0,0,1))
        expected=2*math.cos(theta)-.4*math.sin(theta)+.1
        self.assertAlmostEqual(result['point_pelvis_at_capture_xyz_m'][0],expected)
        self.assertFalse(result['applied_to_planner'])

    def test_world_anchor_handles_left_capture_and_actual_85_degree_turn(self):
        frame,projection,T=self.fixture(direction='left',pixel=(100,50))
        T[:3,3]=0
        result=build_target_geometry(frame,projection,T,Pose2D(1,2,math.pi/2,1))
        np.testing.assert_allclose(result['target_map_xy_m'],[1,4])
        updated=update_target_for_pose(result,Pose2D(1.1,2.2,math.radians(85),2))
        a=math.radians(85)
        expected=[-.1*math.cos(a)+1.8*math.sin(a), .1*math.sin(a)+1.8*math.cos(a)]
        np.testing.assert_allclose(updated['target_current_pelvis_xy_m'],expected)
        np.testing.assert_allclose(updated['target_map_xy_m'],[1,4])
        self.assertNotEqual(updated['target_current_pelvis_xy_m'],[2.,0.])

    def test_missing_depth_does_not_turn_fixed_fallback_into_observation(self):
        frame,projection,T=self.fixture();projection.depth_m=None;projection.used_forward_fallback=True
        result=build_target_geometry(frame,projection,T,Pose2D(0,0,0,1))
        self.assertEqual(result['status'],'UNAVAILABLE');self.assertNotIn('target_map_xy_m',result)

    def test_invalid_intrinsic_or_transform_is_rejected(self):
        frame,projection,T=self.fixture();frame.K[1,1]=0
        with self.assertRaises(ValueError):build_target_geometry(frame,projection,T,Pose2D(0,0,0,1))
        frame,projection,T=self.fixture();T[0,0]=1
        with self.assertRaises(ValueError):build_target_geometry(frame,projection,T,Pose2D(0,0,0,1))

    def test_real_integration_freezes_selected_capture_and_updates_before_planner(self):
        frame,projection,T=self.fixture(direction='left')
        camera_record={'frame_id':8,'sim_step':3,'timestamp':1.,
            'pose_after_capture':{'x':0.,'y':0.,'yaw':math.pi/2,'timestamp':1.},
            'camera_metadata':{'T_pelvis_camera':T.tolist()}}
        episode=object.__new__(RealG1Episode);episode.decision_index=2
        episode._real_records={'left':camera_record}
        episode.pending=SimpleNamespace(projection=projection)
        episode.odometry=SimpleNamespace(get_pose=lambda:Pose2D(0,0,0,2))
        episode._save_json=lambda *args:None
        panorama=SimpleNamespace(views={'left':frame});response=SimpleNamespace(direction='left')
        episode._compute_target_geometry(panorama,response)
        target=np.array(episode._target_geometry['target_map_xy_m'])
        episode.state=EpisodeState.PLAN_AFTER_ROTATION
        pose=Pose2D(.1,.2,math.radians(85),3)
        with patch.object(LocalEndToEndEpisode,'_capture_forward_observation',return_value=(frame,pose)):
            episode._capture_forward_observation(3,3.)
        self.assertEqual(episode._target_geometry['current_reference_frame_id'],8)
        np.testing.assert_allclose(episode._target_geometry['target_map_xy_m'],target)
        self.assertIs(episode.pending.projection,projection)
        self.assertFalse(episode._target_geometry['applied_to_planner'])
        camera_record['frame_id']=999
        episode._compute_target_geometry(panorama,response)
        self.assertEqual(episode._target_geometry['status'],'UNAVAILABLE')
        self.assertNotIn('target_map_xy_m',episode._target_geometry)

if __name__ == '__main__':unittest.main()
