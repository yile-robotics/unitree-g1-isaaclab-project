"""真机局部深度补采样：不连接相机、模型或机器人。"""
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unified_vln.episode import LocalEndToEndEpisode
from unified_vln.real_episode import RealG1Episode
from unified_vln.model_contract import NavigationDecisionResponse
from unified_vln.types import ViewFrame


class RealDepthSamplingTest(unittest.TestCase):
    def fixture(self, depth, bbox=(10, 10, 30, 39)):
        h, w = depth.shape
        frame = ViewFrame('forward', 1, 0, 0., np.zeros((h,w,3), dtype=np.uint8),
                          depth, np.array([[100.,0,w/2],[0,100.,h/2],[0,0,1.]]))
        response = NavigationDecisionResponse(session_id='test', observation_id='test_0',
            action='NAVIGATE', direction='forward', target='box', bbox_2d=bbox,
            waypoint=None, progress_analysis='', reasoning='')
        return frame, response

    def test_initial_valid_window_is_not_expanded(self):
        depth = np.full((40,40), 2., dtype=np.float32)
        frame, response = self.fixture(depth)
        result = RealG1Episode._project_selected_view_target(frame, response)
        self.assertEqual(result.depth_m, 2.)
        self.assertEqual(len(result.depth_sampling['attempts']), 1)
        self.assertFalse(result.depth_sampling['used_expanded_window'])
        self.assertEqual(result.depth_window_xyxy_exclusive, (17,36,24,40))

    def test_border_hole_uses_15_window_and_simulation_keeps_original_fallback(self):
        depth = np.zeros((40,40), dtype=np.float32)
        depth[32:36,23:28] = 1.15
        frame, response = self.fixture(depth)
        result = RealG1Episode._project_selected_view_target(frame, response)
        self.assertFalse(result.used_forward_fallback)
        self.assertAlmostEqual(result.depth_m, 1.15, places=5)
        self.assertEqual(result.valid_depth_count, 20)
        self.assertEqual(result.depth_window_xyxy_exclusive, (13,32,28,40))
        self.assertTrue(result.to_dict()['depth_sampling']['used_expanded_window'])
        original = LocalEndToEndEpisode._project_selected_view_target(frame, response)
        self.assertTrue(original.used_forward_fallback)
        self.assertNotIn('depth_sampling', original.to_dict())

    def test_outside_bbox_depth_cannot_rescue_small_target(self):
        depth = np.full((40,40), .5, dtype=np.float32)
        depth[15:26,18:23] = 0
        frame, response = self.fixture(depth, (18,15,22,25))
        result = RealG1Episode._project_selected_view_target(frame, response)
        self.assertTrue(result.used_forward_fallback)
        self.assertEqual(result.valid_depth_count, 0)
        self.assertEqual(result.depth_window_xyxy_exclusive, (18,18,23,26))
        for attempt in result.depth_sampling['attempts']:
            x0,y0,x1,y1=attempt['window_xyxy_exclusive']
            self.assertGreaterEqual(x0,18);self.assertLessEqual(x1,23)
            self.assertGreaterEqual(y0,15);self.assertLessEqual(y1,26)

    def test_expansion_preserves_depth_limits_and_percentile(self):
        depth = np.zeros((40,40), dtype=np.float32)
        depth[32,13:18] = [.05, 6., 1., 2., 3.]
        frame, response = self.fixture(depth)
        result = RealG1Episode._project_selected_view_target(frame, response)
        self.assertEqual(result.valid_depth_count, 3)
        self.assertAlmostEqual(result.depth_m, 1.6)
        self.assertEqual(result.pixel_uv, (20,39))


if __name__ == '__main__':
    unittest.main()
