"""决策换框、停止清框和实时图像隔离；不连接模型或运控。"""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from live_model_target import DecisionWatcher, draw_model_target, mark_display_motion, read_follow_preview


class LiveModelTargetTests(unittest.TestCase):
    def test_actual_follow_path_is_hidden_when_stopped_stale_or_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'follow_live.json'
            value={'state':'FOLLOWING','status':'PASS','executed_by_probe':True,
                   'received_monotonic_s':time.monotonic(),'path_local_xyz_m':[[0,0,0],[1,0,0]],
                   'target':{'target_preview_xy_m':[2,0],'goal_preview_xy_m':[1,0]}}
            path.write_text(json.dumps(value))
            self.assertIsNotNone(read_follow_preview(tmp))
            for change in ({'state':'STOPPED'},{'received_monotonic_s':time.monotonic()-1},
                           {'path_local_xyz_m':[[0,0,0],[float('nan'),0,0]]}):
                path.write_text(json.dumps({**value,**change}))
                self.assertIsNone(read_follow_preview(tmp))
            path.write_text('{')
            self.assertIsNone(read_follow_preview(tmp))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.version = 0
        self.watcher = DecisionWatcher(self.root, interval=0)

    def decision(self, *, bbox=(10,20,80,90), action='NAVIGATE', direction='forward', width=1280,
                 height=720, formal=False, partial=False):
        self.version += 1
        folder = self.root / ('formal' if formal else 'run/model')
        folder.mkdir(parents=True,exist_ok=True)
        request = folder/(f'decision_{self.version:03d}_request.json' if formal else 'request_metadata.json')
        request.write_text(json.dumps({'image_width':width,'image_height':height,'decision_index':self.version}))
        response = folder/(f'decision_{self.version:03d}_response.json' if formal else 'response.json')
        response.write_text('{' if partial else json.dumps({'action':action,'direction':direction,
                                                          'target':'黑色行李箱','bbox_2d':bbox}))
        stamp=1800000000000000000+self.version*1000000000
        os.utime(response,ns=(stamp,stamp))
        return response

    def test_new_decision_replaces_old_box(self):
        self.decision()
        first=self.watcher.poll()
        self.assertEqual(first.box_for((720,1280,3)),(10,20,80,90))
        self.decision(bbox=(100,200,300,400))
        second=self.watcher.poll()
        self.assertEqual(second.box_for((720,1280,3)),(100,200,300,400))
        self.assertNotEqual(first.decision_id,second.decision_id)
        self.assertIn('黑色行李箱',self.watcher.window_title('live'))

    def test_motion_clears_old_decision_until_a_new_response(self):
        response = self.decision()
        self.assertIsNotNone(self.watcher.poll())
        event = mark_display_motion(self.root/'next_run')
        stamp = response.stat().st_mtime_ns + 1
        os.utime(event, ns=(stamp,stamp))
        previous_revision = self.watcher.revision
        self.assertIsNone(self.watcher.poll())
        self.assertGreater(self.watcher.revision,previous_revision)
        self.assertTrue(self.watcher.hidden_by_motion)
        self.assertIsNone(self.watcher.poll())  # 不能在停车/重绘后恢复历史框。
        self.decision(bbox=(100,200,300,400))
        self.assertEqual(self.watcher.poll().bbox,(100,200,300,400))
        self.assertFalse(self.watcher.hidden_by_motion)

    def test_restart_during_motion_does_not_reload_old_response(self):
        response = self.decision()
        event = mark_display_motion(self.root/'run')
        stamp = response.stat().st_mtime_ns + 1
        os.utime(event, ns=(stamp,stamp))
        watcher = DecisionWatcher(self.root,interval=0)
        self.assertIsNone(watcher.poll())
        self.assertTrue(watcher.hidden_by_motion)

    def test_fresh_viewer_skips_history_and_waits_for_new_decision(self):
        response = self.decision()
        watcher = DecisionWatcher(self.root, interval=0, since_ns=response.stat().st_mtime_ns)
        self.assertIsNone(watcher.poll())
        self.decision()
        self.assertIsNotNone(watcher.poll())

    def test_partial_response_after_motion_does_not_restore_old_box(self):
        response = self.decision()
        self.watcher.poll()
        event = mark_display_motion(self.root/'run')
        stamp = response.stat().st_mtime_ns + 1
        os.utime(event, ns=(stamp,stamp))
        self.decision(partial=True)
        self.assertIsNone(self.watcher.poll())
        self.decision()
        self.assertIsNotNone(self.watcher.poll())

    def test_single_response_watcher_reads_parent_run_motion_event(self):
        response = self.decision()
        watcher = DecisionWatcher(response,interval=0)
        self.assertIsNotNone(watcher.poll())
        event = mark_display_motion(response.parent.parent)
        stamp = response.stat().st_mtime_ns + 1
        os.utime(event,ns=(stamp,stamp))
        self.assertIsNone(watcher.poll())

    def test_manual_hide_does_not_restore_history(self):
        response = self.decision()
        self.watcher.poll()
        with patch('live_model_target.time.time_ns', return_value=response.stat().st_mtime_ns+1):
            self.watcher.hide_until_next_decision()
        self.assertIsNone(self.watcher.poll())
        self.decision()
        self.assertIsNotNone(self.watcher.poll())

    def test_stop_and_backtrack_clear_previous_rectangle(self):
        for action in ('STOP','BACKTRACK','SAFE_STOP'):
            self.decision();self.watcher.poll()
            self.decision(action=action)
            self.assertIsNone(self.watcher.poll().box_for((720,1280,3)))

    def test_partial_json_keeps_previous_until_complete(self):
        self.decision();first=self.watcher.poll()
        self.decision(partial=True)
        self.assertEqual(self.watcher.poll(),first)
        self.decision(bbox=(200,100,400,300))
        self.assertEqual(self.watcher.poll().bbox,(200,100,400,300))

    def test_invalid_complete_bbox_clears_previous(self):
        self.decision();self.watcher.poll()
        self.decision(bbox=(100,20,10,90))
        self.assertIsNone(self.watcher.poll())
        self.assertIsNotNone(self.watcher.error)

    def test_missing_bbox_does_not_keep_old_rectangle(self):
        self.decision();self.watcher.poll()
        self.decision(bbox=None)
        self.assertIsNone(self.watcher.poll().box_for((720,1280,3)))

    def test_old_480p_crop_is_not_stretched_onto_720p(self):
        self.decision(width=640,height=480)
        self.assertIsNone(self.watcher.poll().box_for((720,1280,3)))

    def test_other_panorama_view_is_not_drawn_on_forward_rgb(self):
        for direction in ('left','right','behind'):
            self.decision(direction=direction)
            self.assertIsNone(self.watcher.poll().box_for((720,1280,3)))

    def test_formal_episode_response_and_request_pair(self):
        self.decision(formal=True)
        target=self.watcher.poll()
        self.assertEqual(target.decision_id,'1')
        self.assertEqual(target.box_for((720,1280,3)),(10,20,80,90))

    def test_side_view_box_waits_for_matching_real_turn_confirmation(self):
        response=self.decision(formal=True,direction='left')
        self.assertIsNone(self.watcher.poll().box_for((720,1280,3)))
        alignment=response.with_name(response.name.replace('_response.json','_overlay.json'))
        alignment.write_text(json.dumps({'view_aligned':True,'direction':'left','decision_index':1}))
        self.assertEqual(self.watcher.poll().box_for((720,1280,3)),(10,20,80,90))
        # 下一条决策不能沿用上一条转向完成标记。
        self.decision(formal=True,direction='right')
        self.assertIsNone(self.watcher.poll().box_for((720,1280,3)))

    def test_draw_updates_live_pixels_without_modifying_input(self):
        self.decision(bbox=(200,200,400,400))
        target=self.watcher.poll()
        black=np.zeros((720,1280,3),np.uint8)
        first=draw_model_target(black,target)
        white=np.full_like(black,255)
        second=draw_model_target(white,target)
        np.testing.assert_array_equal(first[600,600],[0,0,0])
        np.testing.assert_array_equal(second[600,600],[255,255,255])
        np.testing.assert_array_equal(first[200,200],[0,255,0])
        self.assertFalse(black.any())
        self.assertTrue((white==255).all())


if __name__=='__main__':
    unittest.main()
