"""Offline geometry tests; never import/create robot DDS objects."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from kinematics import RobotModel,fit_rigid,origin,rotation
from calibrate import board_pose,solve

class GeometryTests(unittest.TestCase):
    def test_stationary_uses_angle_span_and_records_velocity(self):
        import threading
        from unittest.mock import patch
        from robot_state import JointReader
        from kinematics import JOINT_NAMES
        reader=JointReader.__new__(JointReader)
        reader.lock=threading.Lock()
        reader.latest=lambda *args: reader.samples[-1]
        reader.samples=[dict(received_monotonic_s=100+i*.1,tick=i,mode_machine=5,mode_pr=0,
            q=dict.fromkeys(JOINT_NAMES,0.),dq=dict.fromkeys(JOINT_NAMES,.06)) for i in range(11)]
        with patch('robot_state.time.monotonic',return_value=100),patch('robot_state.time.sleep'):
            state=reader.stationary()
            self.assertEqual(state['sample_count'],11)
            self.assertEqual(state['max_joint_span_deg'],0)
            self.assertAlmostEqual(state['max_reported_dq_rad_s'],.06)
            name=next(n for n in JOINT_NAMES if n.startswith('left_shoulder'))
            reader.samples[-1]['q'][name]=np.deg2rad(1)
            with self.assertRaisesRegex(RuntimeError,'Arm/waist moved'):reader.stationary()
    def test_contact_sequence_keeps_check_points_and_resumes(self):
        from collect_points import points_to_collect
        pending=points_to_collect([])
        self.assertEqual(len(pending),9)
        self.assertEqual([p[2] for p in pending],['fit']*6+['check']*3)
        self.assertEqual(points_to_collect([dict(row=0,col=0,role='fit')]),pending[1:])
    def test_rigid_planar_points_transform_direction(self):
        a=np.array([[x,y,2.] for x in (0.,.2,.4) for y in (0.,.3,.6)])
        R=rotation([1,2,3],.7);t=np.array([.12,-.09,.2]);b=a@R.T+t
        T=fit_rigid(a,b)
        np.testing.assert_allclose(T[:3,:3],R,atol=1e-12)
        np.testing.assert_allclose(T[:3,3],t,atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(T[:3,:3]),1)
    def test_degenerate_rejected(self):
        a=np.array([[0,0,0],[1,0,0],[2,0,0]])
        with self.assertRaises(ValueError):fit_rigid(a,a)
    def test_nonfinite_rejected(self):
        a=np.array([[0,0,0],[1,0,0],[0,np.nan,0]])
        with self.assertRaises(ValueError):fit_rigid(a,a)
    def test_known_urdf_axis_origin_and_inverse(self):
        xml='''<robot name="test"><link name="pelvis"/><link name="torso"/><link name="wrist"/>
        <joint name="waist" type="revolute"><parent link="pelvis"/><child link="torso"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
        <joint name="arm" type="revolute"><parent link="torso"/><child link="wrist"/><origin xyz="0 2 0"/><axis xyz="1 0 0"/></joint></robot>'''
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'r.urdf';path.write_text(xml);m=RobotModel(path);q={'waist':np.pi/2,'arm':np.pi/2}
            np.testing.assert_allclose(m.point('pelvis','wrist',[0,1,0],q),[-1,0,1],atol=1e-12)
            np.testing.assert_allclose(m.point('torso','wrist',[0,1,0],q),[0,2,1],atol=1e-12)
            np.testing.assert_allclose(m.transform('pelvis','torso',q)@m.transform('torso','pelvis',q),np.eye(4),atol=1e-12)
            with self.assertRaises(ValueError):m.point('pelvis','wrist',[0,0,0],{'waist':0})
    def test_board_detection_metric_scale(self):
        import cv2
        # 7 x 6 squares -> 6 x 5 inner corners, fronto-parallel synthetic target.
        rgb=np.full((480,640,3),180,np.uint8)
        for row in range(6):
            for col in range(7):
                rgb[60+row*50:60+(row+1)*50,100+col*50:100+(col+1)*50]=255 if (row+col)%2 else 0
        K=np.array([[600.,0,320],[0,600.,240],[0,0,1]])
        pose,_=board_pose(rgb,K,np.zeros(5),6,5,.05)
        self.assertLess(pose['reprojection_rms_px'],.2)
        points=np.array(pose['points_camera_m'])
        self.assertAlmostEqual(np.linalg.norm(points[1]-points[0]),.05,places=8)
        self.assertAlmostEqual(np.median(points[:,2]),.6,places=2)
    def test_partial_board_without_full_grid(self):
        import cv2
        from partial_board import estimate
        ids=np.array([[1,1],[1,3],[2,2],[3,1],[3,3],[4,2]])
        obj=np.c_[ids[:,1]*.1,ids[:,0]*.1,np.zeros(len(ids))]
        K=np.array([[600.,0,320],[0,600.,240],[0,0,1]])
        rv=np.array([.12,-.18,.07]);tv=np.array([-.2,-.3,1.2])
        pixels=cv2.projectPoints(obj,rv,tv,K,np.zeros(5))[0].reshape(-1,2)
        pose=estimate(K,np.zeros(5),ids,pixels,5,8,.1)
        expected=origin(tv,[0,0,0]);expected[:3,:3]=cv2.Rodrigues(rv)[0]
        np.testing.assert_allclose(pose['T_camera_board'],expected,atol=1e-7)
        self.assertEqual(len(pose['points_camera_m']),40)
        self.assertEqual(pose['selected_corner_ids'],ids.tolist())
    def test_partial_board_rejects_bad_selection(self):
        from partial_board import estimate
        K=np.eye(3)
        for ids in ([[0,0],[0,1],[0,2],[0,3]],[[0,0],[0,0],[1,0],[1,1]],[[0,0],[0,1],[1,0],[9,1]]):
            with self.assertRaises(ValueError):
                estimate(K,np.zeros(5),ids,[[1,1],[2,1],[1,2],[2,2]],5,8,.1)
    def test_hand_mesh_candidate_export_is_not_measured(self):
        import subprocess,sys
        from robot_state import MODEL_DIR
        if not (MODEL_DIR/'g1_29dof_rev_1_0.urdf').exists():self.skipTest('Official model not available')
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'hand'
            subprocess.run([sys.executable,str(Path(__file__).with_name('inspect_hand.py')),'--output',str(output)],check=True,capture_output=True)
            candidate=json.loads((output/'model_contact_candidate.json').read_text())
            probe=json.loads((output/'probe.mesh_candidate.json').read_text())
            p=np.r_[candidate['p_left_rubber_hand_m'],1]
            expected=(np.array(candidate['T_wrist_hand'])@p)[:3]
            np.testing.assert_allclose(probe['tip_in_link_m'],expected,atol=1e-12)
            self.assertFalse(probe['tip_measured'])
            self.assertFalse(probe['mount_on_torso_confirmed'])
            self.assertTrue((output/'hand_model_views.png').exists())
    def test_model_tip_session_is_explicit_and_unmeasured(self):
        from calibrate import new_session
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);ref=base/'board.json';ref.write_text('{}')
            (base/'corners_numbered.png').touch()
            candidate=base/'candidate.json';candidate.write_text(json.dumps({'p_left_wrist_yaw_link_m':[.17,-.02,.006]}))
            config=base/'probe.json';config.write_text(json.dumps({'probe_link':'left_wrist_yaw_link',
                'tip_in_link_m':[.17,-.02,.006],'tip_source':'model','tip_measured':False,
                'mount_on_torso_confirmed':True,'model_candidate_path':str(candidate)}))
            args=SimpleNamespace(probe_config=config,allow_model_tip=False,reference_fixed=True,
                                 board_reference=ref,output=base/'session')
            with self.assertRaises(ValueError):new_session(args)
            args.allow_model_tip=True;new_session(args)
            record=json.loads((args.output/'session.json').read_text())
            self.assertEqual(record['tip_source'],'model')
            self.assertFalse(record['probe']['tip_measured'])
    def test_partial_high_reprojection_is_retained_for_review(self):
        import cv2
        from partial_board import estimate
        ids=np.array([[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]])
        obj=np.c_[ids[:,1]*.1,ids[:,0]*.1,np.zeros(len(ids))]
        K=np.array([[600.,0,320],[0,600.,240],[0,0,1]])
        pixels=cv2.projectPoints(obj,np.array([.1,-.2,.05]),np.array([-.1,-.1,.7]),K,np.zeros(5))[0].reshape(-1,2)
        pixels[4]+=[4,4]
        pose=estimate(K,np.zeros(5),ids,pixels,3,3,.1)
        self.assertGreater(pose['reprojection_rms_px'],1)
        self.assertEqual(pose['quality'],'REVIEW_REPROJECTION')
    def test_solver_requires_heldout_and_saves_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d);samples=[];R=rotation([0,1,0],.3);t=np.array([.1,.2,.3])
            for row in range(3):
                for col in range(3):
                    a=np.array([col*.1,row*.1,1.]);b=R@a+t
                    samples.append(dict(row=row,col=col,p_camera_m=a.tolist(),p_torso_m=b.tolist(),role='fit' if row<2 else 'check'))
            session=dict(samples=samples,urdf='synthetic',urdf_sha256='test',probe={},reference={'metadata':{}})
            (path/'session.json').write_text(json.dumps(session));solve(SimpleNamespace(session=path))
            result=json.loads((path/'extrinsics_candidate.json').read_text())
            self.assertLess(result['check_max_m'],1e-12)
            self.assertEqual(result['status'],'CANDIDATE_REQUIRES_PHYSICAL_VALIDATION')
            session['samples']=[p for p in samples if p['role']=='fit'];(path/'session.json').write_text(json.dumps(session))
            with self.assertRaises(ValueError):solve(SimpleNamespace(session=path))

if __name__=='__main__':unittest.main()
