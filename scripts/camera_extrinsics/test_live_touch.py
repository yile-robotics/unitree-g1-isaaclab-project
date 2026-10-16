"""Synthetic tests for current-pose contacts with hidden targets."""
import unittest
import cv2
import numpy as np
from coded_board import board_point,marker_center,estimate,camera_point,stability
from live_touch import joint_stability,paired_state,validate_pairing,validate_coverage,TARGETS,CaptureKeys
from kinematics import JOINT_NAMES

CONFIG=dict(marker_outer_side_m=.055,marker_gap_x_m=.017,marker_gap_y_m=.017,
            marker_ids_by_position=[[6*c+r for c in range(6)] for r in range(6)])
K=np.array([[607.,0,320],[0,607.,240],[0,0,1]])


class LiveTouchTests(unittest.TestCase):
    def test_repeated_enter_cannot_confirm_next_target_or_retry(self):
        keys=CaptureKeys()
        self.assertFalse(keys.accept(13,True))
        self.assertFalse(keys.accept(32,True))
        self.assertTrue(keys.accept(13,True))
        self.assertFalse(keys.accept(13,True))
        next_target=CaptureKeys()
        for key in (13,13,10):
            self.assertFalse(next_target.accept(key,True))
        next_target.accept(32,True)
        self.assertTrue(next_target.accept(13,True))

    def test_losing_target_visibility_requires_ready_again(self):
        keys=CaptureKeys();keys.accept(32,True)
        self.assertFalse(keys.accept(255,False))
        self.assertFalse(keys.accept(13,True))
        keys.accept(32,True)
        self.assertTrue(keys.accept(13,True))

    def markers(self,rv,tv):
        # The encoded region is smaller than the measured outer brown square.
        offsets=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*.022
        markers={}
        for i in (0,6,12,18,19,20,24,25,26):
            centre=board_point(CONFIG,i,'center')
            corners=np.tile(centre,(4,1));corners[:,:2]+=offsets
            image=cv2.projectPoints(corners,rv,tv,K,np.zeros(5))[0].reshape(4,2)
            markers[i]=np.roll(image,1,axis=0)
        return markers

    def test_partial_hidden_target_and_moving_camera(self):
        target=board_point(CONFIG,14)
        np.testing.assert_allclose(target,[.144,.144,0])
        for rv,tv in ((np.array([.1,-.2,.03]),np.array([-.2,-.2,1.])),
                      (np.array([.2,-.1,-.03]),np.array([-.18,-.21,.9]))):
            pose=estimate(self.markers(rv,tv),K,np.zeros(5),CONFIG)
            expected=cv2.Rodrigues(rv)[0]@target+tv
            np.testing.assert_allclose(camera_point(pose,target),expected,atol=1e-7)
            self.assertNotIn(14,pose['decoded_ids'])
            self.assertLess(pose['reprojection_rms_px'],1e-5)

    def test_diagonal_intersection_correct_under_perspective(self):
        obj=np.array([[-1.,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])*.1
        rv=np.array([.3,.2,.1]);tv=np.array([0.,0.,1.])
        corners=cv2.projectPoints(obj,rv,tv,K,np.zeros(5))[0].reshape(4,2)
        expected=cv2.projectPoints(np.zeros((1,3)),rv,tv,K,np.zeros(5))[0].reshape(2)
        np.testing.assert_allclose(marker_center(corners),expected,atol=1e-9)

    def test_single_bad_marker_is_excluded_from_current_pose(self):
        rv=np.array([.1,-.2,.03]);tv=np.array([-.2,-.2,1.])
        markers=self.markers(rv,tv)
        markers[6]=markers[6]+np.array([40.,30.])
        pose=estimate(markers,K,np.zeros(5),CONFIG)
        self.assertIn(6,pose['rejected_marker_ids'])
        self.assertNotIn(6,pose['decoded_ids'])
        target=board_point(CONFIG,14)
        np.testing.assert_allclose(camera_point(pose,target),cv2.Rodrigues(rv)[0]@target+tv,atol=1e-7)

    def test_plane_rotation_is_diagnostic_when_target_is_still(self):
        target=board_point(CONFIG,14);pc=np.array([.1,.2,1.]);records=[]
        for degrees in np.linspace(-.6,.6,6):
            R=cv2.Rodrigues(np.array([0.,np.deg2rad(degrees),0.]))[0]
            T=np.eye(4);T[:3,:3]=R;T[:3,3]=pc-R@target
            records.append({'pose':{'T_camera_board':T.tolist()}})
        result=stability(records,target)
        self.assertLess(result['target_span_m'],1e-12)
        self.assertTrue(result['rotation_diagnostic_warning'])

    def test_valid_pose_coverage_rejects_long_missing_intervals(self):
        def records(times):return [{'metadata':{'client_received_monotonic_s':t}} for t in times]
        validate_coverage(records([.05,.25,.45,.65,.85,1.05]),0,1.2)
        with self.assertRaisesRegex(ValueError,'interrupted'):
            validate_coverage(records([.05,.1,.15,.9,1.,1.1]),0,1.2)
        with self.assertRaises(ValueError):validate_coverage(records([.9,1.]),0,1.2)

    def test_visual_motion_rejected(self):
        records=[]
        for i in range(6):
            T=np.eye(4);T[2,3]=1;T[0,3]=i*.0001
            records.append({'pose':{'T_camera_board':T.tolist()}})
        self.assertLess(stability(records,[0,0,0])['target_span_m'],.001)
        records[-1]['pose']['T_camera_board'][0][3]=.02
        with self.assertRaisesRegex(ValueError,'not still'):stability(records,[0,0,0])

    def test_joints_and_reception_pairing(self):
        import threading
        from types import SimpleNamespace
        states=[dict(tick=i,received_monotonic_s=100+i*.01,mode_machine=5,mode_pr=0,
                     q=dict.fromkeys(JOINT_NAMES,0.),dq=dict.fromkeys(JOINT_NAMES,.06)) for i in range(10)]
        self.assertEqual(joint_stability(states)['max_joint_span_deg'],0)
        reader=SimpleNamespace(lock=threading.Lock(),samples=states,latest=lambda timeout:states[-1])
        state,gap=paired_state(reader,100.052)
        self.assertEqual(state['tick'],5)
        self.assertLess(gap,.003)
        with self.assertRaises(ValueError):paired_state(reader,101.)
        states[-1]['q']['left_elbow_joint']=np.deg2rad(1)
        with self.assertRaisesRegex(ValueError,'moved'):joint_stability(states)

    def test_repeated_frames_and_check_partition(self):
        r=dict(metadata={'color_frame_number':1,'depth_frame_number':2},
               state={'mode_machine':5,'mode_pr':0})
        with self.assertRaisesRegex(ValueError,'Repeated'):validate_pairing([r,r])
        fit={i for i,role in TARGETS if role=='fit'}
        checks={i for i,role in TARGETS if role=='check'}
        self.assertEqual(len(fit),6);self.assertEqual(len(checks),3)
        self.assertFalse(fit&checks)


if __name__=='__main__':unittest.main()
