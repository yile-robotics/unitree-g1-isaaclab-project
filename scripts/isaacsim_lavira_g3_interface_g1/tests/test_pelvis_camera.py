"""pelvis坐标换算、名义离地高度与逐帧配对；不创建DDS或相机。"""
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
from camera_d415 import pelvis_camera as geometry
from unified_vln.map_progress import SparseMapConfig
from unified_vln.odometry import Pose2D
from unified_vln.types import ViewFrame


def frame(number=1):
    return ViewFrame('forward',number,number,float(number),np.zeros((1,1,3),np.uint8),
                     np.ones((1,1),np.float32),np.eye(3))


class PelvisGeometryTest(unittest.TestCase):
    def fixture(self):
        camera=geometry.PelvisCamera.__new__(geometry.PelvisCamera)
        camera.profile=geometry.load_profile(Path(geometry.__file__).with_name('pelvis.json'))
        camera.model=SimpleNamespace(sha256='fake',transform=MagicMock(return_value=np.eye(4)))
        state={'mode_pr':0,'mode_machine':5,'q':{'waist_yaw_joint':0.},
               'received_monotonic_s':time.monotonic()}
        camera.reader=SimpleNamespace(lock=threading.Lock(),samples=[state],latest=lambda **kw:state)
        camera.camera=MagicMock()
        camera.camera.last_metadata={'serial':camera.profile['camera_serial'],
                                    'client_received_monotonic_s':state['received_monotonic_s']}
        camera.camera.capture_forward.return_value=frame()
        camera.transforms=geometry.OrderedDict();camera.last_metadata={}
        return camera,state

    def test_profile_records_user_frame_confirmation_separately_from_camera_calibration(self):
        profile=geometry.load_profile(Path(geometry.__file__).with_name('pelvis.json'))
        self.assertIs(profile['slam_base_verified'],True)
        self.assertEqual(profile['physical_frame'],'pelvis')
        self.assertIn('2026-10-20',profile['slam_base_verification_source'])
        self.assertAlmostEqual(profile['pelvis_height_above_floor_m'],.80)
        self.assertNotIn('camera_height_above_floor_m',profile)
        self.assertEqual(profile['calibration_status'],'CANDIDATE_REQUIRES_PHYSICAL_VALIDATION')
        self.assertEqual(profile['camera_serial'],'211222063932')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'invalid.json'
            p.write_text(json.dumps({'status':'VERIFIED','slam_base_verified':True}))
            with self.assertRaises(ValueError):geometry.load_profile(p)

    def test_fk_composition_rotates_camera_offset_before_adding_pelvis_offset(self):
        camera,state=self.fixture()
        T=np.eye(4);T[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];T[:3,3]=[.1,.2,.044]
        camera.model.transform.return_value=T
        with patch.object(geometry,'model_for',return_value=camera.model):
            result=camera.transform_for_state(state)
        x,y,z=camera.profile['T_torso_camera'][:3,3]
        np.testing.assert_allclose(result[:3,3],[.1-y,.2+x,.044+z])
        camera.model.transform.assert_called_once_with('pelvis','torso_link',state['q'])

    def test_image_keeps_its_own_waist_transform_after_next_capture(self):
        camera,state=self.fixture()
        with patch.object(geometry,'model_for',return_value=camera.model):
            first=camera.capture_forward(1,1.)
            original=camera.transform_for_frame(first)
            self.assertIs(camera.last_metadata['slam_base_verified'],True)
            self.assertAlmostEqual(camera.last_metadata['derived_camera_height_above_floor_m'],.8+original[2,3])
            T=np.eye(4);T[0,3]=.2
            camera.model.transform.return_value=T
            camera.camera.capture_forward.return_value=frame(2)
            second=camera.capture_forward(2,2.)
        np.testing.assert_allclose(camera.transform_for_frame(first),original)
        self.assertAlmostEqual(camera.transform_for_frame(second)[0,3]-original[0,3],.2)
        with self.assertRaisesRegex(RuntimeError,'没有该图像'):
            camera.transform_for_frame(frame(3))

    def test_bad_serial_or_joint_pair_does_not_create_frame_geometry(self):
        for failure in ('serial','pair'):
            camera,state=self.fixture()
            if failure=='serial':camera.camera.last_metadata['serial']='other-camera'
            else:state['received_monotonic_s']-=.2
            with self.subTest(failure=failure),self.assertRaises((ValueError,RuntimeError)):
                camera.capture_forward(1,1.)
            self.assertFalse(camera.transforms)

    def test_pelvis_height_is_direct_and_camera_offset_is_added_once(self):
        T=np.eye(4);T[2,3]=.35
        config=geometry.map_config_for_transform(T,.80)
        self.assertAlmostEqual(config.nominal_base_height_m,.80)
        self.assertAlmostEqual(config.nominal_base_height_m+config.camera_offset_z_m,1.15)
        T[2,3]=.4
        updated=geometry.map_config_for_transform(T,.80,floor=2.)
        self.assertAlmostEqual(updated.nominal_base_height_m,2.80)
        self.assertAlmostEqual(updated.nominal_base_height_m+updated.camera_offset_z_m-updated.floor_z_world_m,1.20)
        for invalid in (0.,-.9,float('nan'),float('inf')):
            with self.subTest(height=invalid),self.assertRaises(ValueError):
                geometry.map_config_for_transform(T,invalid)

    def test_height_filter_uses_pelvis_height_camera_offset_and_floor_reference(self):
        T=np.eye(4);T[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];T[:3,3]=[.1,.02,.35]
        camera=SimpleNamespace(transform_for_frame=lambda f:T.copy())
        config=geometry.map_config_for_transform(T,.80,floor=2.,
                                               base_config=SparseMapConfig(depth_stride=1))
        for height,obstacle in ((0.,False),(.05,False),(.25,True),(1.25,True),(1.8,False)):
            with self.subTest(height=height):
                sample=frame()
                # 光学y向下；相机离地1.15m，点的高度为1.15+cy。
                sample.K[1,2]=height-1.15
                world=geometry.PelvisExplorationMap(config,pose_frame_id='map',camera=camera)
                result=world.integrate(sample,Pose2D(0,0,0,0))
                self.assertEqual(result.valid_depth_samples,1)
                self.assertEqual(bool(world.occupied_cells),obstacle)

    def test_real_map_uses_full_per_frame_optical_transform(self):
        # 一个光轴前方1米点，先由胸部/腰部外参转成pelvis前方，再随世界yaw旋转。
        T=np.eye(4);T[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];T[:3,3]=[.1,.02,.35]
        camera=SimpleNamespace(transform_for_frame=lambda f:T)
        config=SparseMapConfig(depth_stride=1,nominal_base_height_m=.75)
        world=geometry.PelvisExplorationMap(config,pose_frame_id='map',camera=camera)
        result=world.integrate(frame(),Pose2D(2.,3.,np.pi/2,0.))
        self.assertEqual(result.valid_depth_samples,1)
        expected=world.world_xy_to_cell(1.98,4.1)
        self.assertIn(expected,world.occupied_cells)
        self.assertIsNone(world.current_transform)
        with self.assertRaisesRegex(RuntimeError,'当前帧'):
            world._optical_points_to_world(np.array([[0.,0,1.]]),Pose2D(0,0,0,0))

    def test_successive_real_frames_accumulate_map_gain_without_counting_repeat_views(self):
        T=np.eye(4);T[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];T[:3,3]=[.1,.02,.35]
        camera=SimpleNamespace(transform_for_frame=lambda f:T.copy())
        world=geometry.PelvisExplorationMap(
            SparseMapConfig(depth_stride=1,nominal_base_height_m=.75),
            pose_frame_id='map',camera=camera)
        start=Pose2D(0.,0.,0.,0.)
        first=world.integrate(frame(1),start)
        initial_observed=set(world.observed_cells)
        initial_occupied=set(world.occupied_cells)
        baseline=world.explored_cells

        repeated=world.integrate(frame(2),start)
        self.assertEqual(repeated.new_explored_cells,0)
        self.assertEqual(repeated.new_occupied_cells,0)
        self.assertEqual(world.snapshot(explored_before=baseline).new_explored_cells,0)

        moved=world.integrate(frame(3),Pose2D(.5,0.,0.,1.))
        snapshot=world.snapshot(explored_before=baseline)
        self.assertGreater(moved.new_explored_cells,0)
        self.assertEqual(snapshot.new_explored_cells,moved.new_explored_cells)
        self.assertEqual(snapshot.update_count,3)
        self.assertEqual(snapshot.pose_frame_id,'map')
        self.assertTrue(initial_observed.issubset(world.observed_cells))
        self.assertTrue(initial_occupied.issubset(world.occupied_cells))
        self.assertIn(world.world_xy_to_cell(1.6,.02),world.occupied_cells)

    def test_incompatible_slam_message_is_rejected(self):
        camera,_=self.fixture()
        camera.validate_slam_frame('map','base_link')
        with self.assertRaises(ValueError):camera.validate_slam_frame('odom','pelvis')

    def test_real_planner_filters_far_depth_before_original_uint16_protocol(self):
        adapter=geometry.RealDepthPlanner.__new__(geometry.RealDepthPlanner)
        adapter.client=MagicMock();adapter.client.get_plan.return_value=(np.array([[0,0,0],[2,0,0]]),.1)
        source=frame();source.depth_m[0,0]=8.
        adapter.get_plan(source,np.array([2.,0]))
        forwarded=adapter.client.get_plan.call_args.args[0]
        self.assertEqual(forwarded.depth_m[0,0],0.)
        self.assertEqual(source.depth_m[0,0],8.)


if __name__=='__main__':unittest.main()
