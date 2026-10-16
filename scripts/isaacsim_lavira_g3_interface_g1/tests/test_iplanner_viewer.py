"""预览目标、深度传输和异步规划；不会连接相机、服务或机器人。"""
from dataclasses import replace
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import g1_iplanner_viewer as viewer
from unified_vln.types import ViewFrame


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
