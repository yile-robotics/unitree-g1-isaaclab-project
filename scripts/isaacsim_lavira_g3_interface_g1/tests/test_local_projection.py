from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.local_projection import project_selected_view_target  # noqa: E402
from unified_vln.model_contract import NavigationDecisionResponse  # noqa: E402
from unified_vln.types import ViewFrame  # noqa: E402


class LocalProjectionTest(unittest.TestCase):
    def test_left_bbox_and_left_depth_become_post_turn_goal(self):
        depth = np.full((6, 8), 2.0, dtype=np.float32)
        frame = ViewFrame(
            direction="left",
            frame_id=10,
            sim_step=5,
            timestamp=0.1,
            rgb=np.zeros((6, 8, 3), dtype=np.uint8),
            depth_m=depth,
            K=np.array([[4.0, 0.0, 4.0], [0.0, 4.0, 3.0], [0.0, 0.0, 1.0]]),
        )
        response = NavigationDecisionResponse(
            session_id="s",
            observation_id="s_decision_000",
            action="NAVIGATE",
            direction="left",
            target="door",
            bbox_2d=(4.0, 1.0, 6.0, 4.0),
            waypoint=None,
            progress_analysis="",
            reasoning="",
        )
        projection = project_selected_view_target(frame, response)
        np.testing.assert_allclose(
            projection.goal_after_turn_xy_m,
            np.array([2.0, -0.5]),
            atol=1.0e-8,
        )
        self.assertEqual(projection.pixel_uv, (5, 4))
        self.assertFalse(projection.used_forward_fallback)

    def test_missing_depth_uses_uni_forward_fallback(self):
        frame = ViewFrame(
            direction="forward",
            frame_id=1,
            sim_step=1,
            timestamp=0.1,
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth_m=np.zeros((4, 4), dtype=np.float32),
            K=np.eye(3),
        )
        response = NavigationDecisionResponse(
            session_id="s",
            observation_id="o",
            action="STOP",
            direction="forward",
            target="chair",
            bbox_2d=(0.0, 0.0, 2.0, 2.0),
            waypoint=None,
            progress_analysis="",
            reasoning="",
        )
        projection = project_selected_view_target(frame, response)

        np.testing.assert_allclose(projection.goal_after_turn_xy_m, [1.5, 0.0])
        self.assertIsNone(projection.depth_m)
        self.assertEqual(projection.valid_depth_count, 0)
        self.assertTrue(projection.used_forward_fallback)
        self.assertIsNone(projection.to_dict()["depth_m"])


if __name__ == "__main__":
    unittest.main()
