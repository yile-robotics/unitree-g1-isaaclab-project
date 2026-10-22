"""预览目标、深度传输和异步规划；不会连接相机、服务或机器人。"""
from dataclasses import replace
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import sys
import threading
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import g1_iplanner_viewer as viewer
from unified_vln.types import ViewFrame
from live_model_target import ModelTarget


def frame():
    return ViewFrame('forward',7,0,0.,np.zeros((8,12,3),np.uint8),
                     np.full((8,12),2.,np.float32),np.array([[10.,0,6],[0,10.,4],[0,0,1]]))


class FakePlanner:
    initialized = False
    def reset(self,K):
        self.K=K;self.initialized=True
        return True
    def get_plan(self,f,goal):
        self.frame=f;self.goal=goal
        return np.array([[0,0,0],[goal[0],goal[1],0]]),.2


class IPlannerViewerTests(unittest.TestCase):
    def test_closed_qt_receiver_is_normal_window_close(self):
        for message in ('NULL guiReceiver (please create a window)', 'NULL window handler'):
            with patch.object(viewer.cv2,'getWindowProperty',side_effect=viewer.cv2.error(message)):
                self.assertTrue(viewer.window_closed())

    def test_window_visibility_is_checked_and_other_errors_are_not_hidden(self):
        with patch.object(viewer.cv2,'getWindowProperty',return_value=1):
            self.assertFalse(viewer.window_closed())
        with patch.object(viewer.cv2,'getWindowProperty',return_value=0):
            self.assertTrue(viewer.window_closed())
        with patch.object(viewer.cv2,'getWindowProperty',side_effect=viewer.cv2.error('backend failed')):
            with self.assertRaises(viewer.cv2.error):
                viewer.window_closed()

    def test_close_window_during_wait_or_live_frame_cleans_up_without_failure(self):
        for live in (False,True):
            with self.subTest(live=live), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                feed = stack.enter_context(patch.object(viewer,'CameraFeed')).return_value
                feed.snapshot.return_value = (frame(),time.monotonic(),None,1) if live else (None,0.,'Connection refused',0)
                worker = stack.enter_context(patch.object(viewer,'PlanWorker')).return_value
                for name in ('namedWindow','resizeWindow','setMouseCallback','imshow','setWindowTitle','destroyAllWindows'):
                    stack.enter_context(patch.object(viewer.cv2,name))
                stack.enter_context(patch.object(viewer.cv2,'waitKey',return_value=-1))
                stack.enter_context(patch.object(viewer.cv2,'getWindowProperty',
                                                side_effect=viewer.cv2.error('NULL guiReceiver (please create a window)')))
                stack.enter_context(redirect_stdout(io.StringIO()))
                output = Path(folder)/'output'
                self.assertEqual(viewer.main(['--output',str(output),'--model-results',folder]),0)
                self.assertFalse((output/'error.json').exists())
                feed.close.assert_called_once()
                worker.close.assert_called_once()

    def test_model_goal_uses_clipped_bbox_bottom_centre(self):
        target=ModelTarget('NAVIGATE','forward','test','2',(2,2,10,8),12,8,'response.json')
        self.assertEqual(viewer.model_goal_pixel(frame(),target),(6,7))
        result=viewer.plan_once(FakePlanner(),frame(),(6,7),.5,model_target=target)
        self.assertEqual(result['model_decision']['decision_id'],'2')
        self.assertEqual(result['model_decision']['response_file'],'response.json')
        self.assertEqual(result['target']['pixel_uv'],[6,7])

    def test_non_navigation_or_mismatched_model_does_not_create_goal(self):
        target=ModelTarget('NAVIGATE','forward','test','2',(2,2,10,7),12,8,'response.json')
        for invalid in [None,replace(target,action='STOP'),replace(target,direction='left'),
                        replace(target,width=1280,height=720),replace(target,bbox=None)]:
            self.assertIsNone(viewer.model_goal_pixel(frame(),invalid))

    def test_camera_feed_reconnects_and_changes_generation_after_drop(self):
        class Camera:
            def __init__(self,flaky):self.flaky=flaky;self.count=0;self.closed=False
            def capture_forward(self,*args):
                self.count+=1
                if self.flaky and self.count>1:raise TimeoutError('stale frame')
                time.sleep(.01)
                return replace(frame(),frame_id=10 if self.flaky else 20)
            def close(self):self.closed=True
        first,second=Camera(True),Camera(False)
        with patch.object(viewer,'create_camera',side_effect=[first,second]):
            feed=viewer.CameraFeed('unused')
            try:
                deadline=time.monotonic()+2
                while time.monotonic()<deadline:
                    f,received,error,generation=feed.snapshot()
                    if generation==2:break
                    time.sleep(.01)
                self.assertEqual(generation,2)
                self.assertEqual(f.frame_id,20)
                self.assertIsNone(error)
                self.assertTrue(first.closed)
            finally:feed.close()
        self.assertTrue(second.closed)

    def test_720p_panel_preserves_aspect_and_click_coordinates(self):
        self.assertEqual(viewer.image_panel_geometry((720,1280,3)),(0,0,640,360))
        self.assertEqual(viewer.panel_pixel_to_source((720,1280,3),320,viewer.HEADER+180),(640,360))
        self.assertEqual(viewer.panel_pixel_to_source((720,1280,3),639,viewer.HEADER+359),(1278,718))
        self.assertIsNone(viewer.panel_pixel_to_source((720,1280,3),640,viewer.HEADER+180))
        self.assertIsNone(viewer.panel_pixel_to_source((720,1280,3),320,viewer.HEADER-1))

    def test_480p_replay_keeps_aspect_and_ignores_letterbox_clicks(self):
        self.assertEqual(viewer.image_panel_geometry((480,640,3)),(80,0,480,360))
        self.assertEqual(viewer.panel_pixel_to_source((480,640,3),320,viewer.HEADER+180),(320,240))
        self.assertIsNone(viewer.panel_pixel_to_source((480,640,3),79,viewer.HEADER+180))
        self.assertIsNone(viewer.panel_pixel_to_source((480,640,3),560,viewer.HEADER+180))

    def test_principal_ray_and_standoff_are_in_metres(self):
        r=viewer.goal_from_pixel(frame(),(6,4),.5)
        np.testing.assert_allclose(r['target_preview_xy_m'],[2,0])
        np.testing.assert_allclose(r['goal_preview_xy_m'],[1.5,0])

    def test_right_pixel_maps_to_negative_left(self):
        r=viewer.goal_from_pixel(frame(),(8,4),.5)
        np.testing.assert_allclose(r['target_preview_xy_m'],[2,-.4])
        self.assertAlmostEqual(np.linalg.norm(r['target_preview_xy_m'])-
                               np.linalg.norm(r['goal_preview_xy_m']),.5)

    def test_invalid_depth_never_creates_forward_fallback(self):
        empty=replace(frame(),depth_m=np.zeros((8,12),np.float32))
        with self.assertRaisesRegex(ValueError,'No valid'):
            viewer.goal_from_pixel(empty,(6,4),.5)

    def test_goal_closer_than_standoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'closer'):
            viewer.goal_from_pixel(frame(),(6,4),3.)

    def test_depth_range_is_cleaned_before_uint16_protocol(self):
        raw=frame().depth_m.copy()
        raw[0,:5]=[65.535,7.,float('nan'),float('inf'),-.1]
        f=replace(frame(),depth_m=raw)
        cleaned=viewer.planner_frame(f)
        np.testing.assert_array_equal(cleaned.depth_m[0,:5],np.zeros(5))
        self.assertEqual(cleaned.depth_m[1,1],2.)
        self.assertEqual(f.depth_m[0,0],raw[0,0])

    def test_plan_uses_actual_intrinsics_and_records_source_frame(self):
        client=FakePlanner()
        r=viewer.plan_once(client,frame(),(6,4),.5,version=3)
        np.testing.assert_allclose(client.K,frame().K)
        np.testing.assert_allclose(client.goal,[1.5,0])
        self.assertEqual((r['source_frame_id'],r['target_version']),(7,3))
        self.assertFalse(r['robot_motion_sent'])
        self.assertFalse(r['extrinsics_applied'])

    def test_invalid_path_is_not_displayed_as_success(self):
        client=FakePlanner()
        client.get_plan=lambda *a:(np.array([[0,0,0],[float('nan'),0,0]]),.1)
        with self.assertRaisesRegex(RuntimeError,'finite'):
            viewer.plan_once(client,frame(),(6,4),.5)

    def test_async_old_target_result_is_not_returned_for_new_selection(self):
        started=threading.Event();release=threading.Event();finished=threading.Event()
        class BlockingPlanner(FakePlanner):
            def get_plan(self,*args):
                started.set()
                release.wait(timeout=2)
                result=super().get_plan(*args)
                finished.set()
                return result
        with patch.object(viewer,'IPlannerClient',return_value=BlockingPlanner()):
            worker=viewer.PlanWorker('unused')
        try:
            worker.submit(viewer.Job(frame(),(6,4),.5,1))
            self.assertTrue(started.wait(timeout=1))
            self.assertEqual(worker.snapshot(2),(None,None))
            release.set()
            self.assertTrue(finished.wait(timeout=1))
            # 无论旧请求刚完成还是仍在发布结果，不能作为新选择的路径显示。
            self.assertEqual(worker.snapshot(2),(None,None))
        finally:
            release.set();worker.close()


if __name__=='__main__':
    unittest.main()
