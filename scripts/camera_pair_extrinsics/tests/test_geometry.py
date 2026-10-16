import sys
import unittest
from unittest.mock import patch
import cv2
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from board import points, centre, estimate, detect
from geometry import unpack, relative, fit, validate, compose, transform
from calibrate_pair import stable, read, HERE
from geometry import project
from estimate_from_urdf import estimate as estimate_urdf


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config=read(HERE/'board.json')
        self.K=np.array([[610.,0,320],[0,608.,240],[0,0,1]])
        self.T=unpack(np.array([.15,-.3,.02,.015,-.14,.09]))

    def pair(self,index,noise=0):
        pose=unpack(np.array([.07*index,.03*(index%3),-.08,.0,-.03,.8+.015*index]))
        result={'name':f'pair{index}'}
        rng=np.random.default_rng(index)
        # Intentionally no shared IDs. Both sets occupy a 2D board region.
        for side,T,ids in [('new',pose,[0,1,2,6,7,8,12,13,14]),
                           ('old',self.T@pose,[21,22,23,27,28,29,33,34,35])]:
            obj=points(self.config,ids)
            pixels=project(T,obj,self.K)+rng.normal(0,noise,(len(ids),2))
            result[side]={'T_camera_board':T.tolist(),'object_points':obj.tolist(),
                          'pixels':pixels.tolist(),'ids':ids,'K':self.K.tolist(),'error':None}
        return result

    def test_relative_direction(self):
        p=self.pair(2)
        np.testing.assert_allclose(relative(p),self.T,atol=1e-12)

    def test_fit_disjoint_partial_board_views(self):
        pairs=[self.pair(i,.05) for i in range(8)]
        # Initial independent board PnPs are slightly noisy, as in real data.
        for i,p in enumerate(pairs):
            p['old']['T_camera_board']=(unpack(np.array([.001,0,-.001,.001,0,-.001]))@np.array(p['old']['T_camera_board'])).tolist()
        solved,report=fit(pairs)
        np.testing.assert_allclose(solved,self.T,atol=.004)
        self.assertLess(report['joint_reprojection_rms_px'],.1)
        check=validate(solved,[self.pair(9)])
        self.assertLess(check['max_center_error_mm'],3)

    def test_pnp_centres_do_not_assume_marker_border_size(self):
        p=self.pair(2)['new']; markers={}
        obj=np.array(p['object_points']); T=np.array(p['T_camera_board'])
        for i,point in zip(p['ids'],obj):
            offsets=np.array([[-.015,-.015,0],[.015,-.015,0],[.015,.015,0],[-.015,.015,0]])
            markers[i]=project(T,point+offsets,self.K)
        observed=estimate(markers,self.K,self.config)
        np.testing.assert_allclose(observed['T_camera_board'],T,atol=1e-5)

    def test_perspective_center(self):
        corners=np.array([[0.,0.],[4.,0.],[3.,3.],[0.,2.]])
        point=centre(corners)
        np.testing.assert_allclose(point,[4/3,4/3])
        self.assertFalse(np.allclose(point,corners.mean(axis=0)))

    def test_upscaled_only_detection_preserves_original_pixel_coordinates(self):
        original=np.array([[10.,10.],[30,10],[30,30],[10,30]],np.float32)
        for scale,shift in ((2,.5),(3,1.)):
            class Detector:
                def detectMarkers(self,image):
                    if image.shape[0]!=60*scale:
                        return [],None,[]
                    return [(original*scale+shift).reshape(1,4,2)],np.array([[19]]),[]
            with patch.object(cv2.aruco,'ArucoDetector',return_value=Detector()):
                observed=detect(np.full((60,60,3),255,np.uint8),self.config)
            np.testing.assert_allclose(observed[19],original,atol=1e-6)

    def test_missing_anchor_and_wrong_frames_rejected(self):
        example=read(HERE/'old_rgb_to_torso.example.json')
        with self.assertRaises(ValueError):
            compose(example,self.T,'344422072128')
        example.update(verified=True,source='factory',T_parent_child=np.eye(4).tolist(),child_frame='d435_link')
        with self.assertRaises(ValueError):
            compose(example,self.T,'344422072128')

    def test_composition(self):
        anchor=unpack(np.array([0.,.8,0.,.05,.02,.43]))
        reference={'verified':True,'source':'synthetic','parent_frame':'torso_link',
                   'child_frame':'old_rgb_optical','old_camera_serial':'344422072128','T_parent_child':anchor.tolist()}
        np.testing.assert_allclose(compose(reference,self.T,'344422072128'),anchor@self.T)
        with self.assertRaises(ValueError):
            compose(reference,self.T,'wrong serial')

    def test_validation_does_not_refit(self):
        T=self.T.copy();T[0,3]+=.02;before=T.copy()
        report=validate(T,[self.pair(8),self.pair(9)])
        np.testing.assert_array_equal(T,before)
        self.assertAlmostEqual(report['center_rms_mm'],20,places=5)

    def test_view_diversity_and_count_required(self):
        with self.assertRaises(ValueError):
            fit([self.pair(1)]*8)
        with self.assertRaises(ValueError):
            fit([self.pair(i) for i in range(7)])

    def test_stationary_burst_and_motion_rejection(self):
        records=[]
        for i in range(5):
            p=self.pair(2); p['request_start_monotonic_s']=i*.25;p['receive_gap_ms']=30
            records.append(p)
        stable(records)
        records[-1]['old']['pixels'][0][0]+=5
        with self.assertRaises(ValueError):
            stable(records)

    def test_reflection_rejected(self):
        T=np.eye(4);T[0,0]=-1
        with self.assertRaises(ValueError):
            transform(T)

    def test_urdf_estimate_converts_optical_forward_and_remains_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.urdf'
            path.write_text('<robot name="test"><joint name="d435_joint" type="fixed">'
                            '<parent link="torso_link"/><child link="d435_link"/>'
                            '<origin xyz="0.1 0.2 0.3" rpy="0 0 0"/></joint></robot>')
            candidate={'parent_frame':'old_rgb_optical','child_frame':'new_rgb_optical',
                       'old_camera_serial':'344422072128','new_camera_serial':'211222063932',
                       'T_old_rgb_new_rgb':np.eye(4).tolist()}
            result=estimate_urdf(candidate,path)
            T=np.array(result['T_torso_new_rgb'])
            np.testing.assert_allclose(T@np.array([0.,0.,1.,1.]),[1.1,.2,.3,1.])
            np.testing.assert_allclose(result['mounting_flu_rpy_deg'],[0.,0.,0.])
            self.assertFalse(result['reference_verified'])
            self.assertFalse(result['applied_to_vln'])
            candidate['old_camera_serial']='different camera'
            with self.assertRaises(ValueError):
                estimate_urdf(candidate,path)


if __name__=='__main__':
    unittest.main()
