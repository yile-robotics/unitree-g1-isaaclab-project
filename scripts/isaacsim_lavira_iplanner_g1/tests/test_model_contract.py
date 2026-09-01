from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.model_client import (  # noqa: E402
    CombinedModelClient,
    CompletedWaypoint,
    build_model_history,
)
from unified_vln.types import DIRECTION_ORDER, PanoramaBundle, ViewFrame  # noqa: E402


def make_bundle() -> PanoramaBundle:
    views = {}
    for index, direction in enumerate(DIRECTION_ORDER):
        views[direction] = ViewFrame(
            direction=direction,
            frame_id=index,
            sim_step=7,
            timestamp=0.35,
            rgb=np.zeros((4, 6, 3), dtype=np.uint8),
            depth_m=np.ones((4, 6), dtype=np.float32),
            K=np.eye(3),
        )
    return PanoramaBundle(3, 7, 0.35, views)


class ModelContractTest(unittest.TestCase):
    def test_history_retention_is_independent_of_episode_length(self):
        image = np.zeros((4, 6, 3), dtype=np.uint8)
        records = [
            CompletedWaypoint(
                waypoint_id=index,
                decision_step=index * 10,
                direction="forward",
                target=f"target {index}",
                init_rgb=image,
                direction_rgb=image,
            )
            for index in range(6)
        ]
        history, history_images = build_model_history(records)

        # 文本历史不限制任务轮数；协议只给最近四个 waypoint 附两张图片。
        self.assertEqual(len(history), 6)
        self.assertFalse(history[0].has_images)
        self.assertFalse(history[1].has_images)
        self.assertTrue(all(entry.has_images for entry in history[2:]))
        self.assertEqual(len(history_images), 8)

        limited_history, limited_images = build_model_history(
            records, max_waypoints=3
        )
        self.assertEqual([entry.waypoint_id for entry in limited_history], [0, 1, 2])
        self.assertEqual(
            [entry.description for entry in limited_history],
            ["target 3", "target 4", "target 5"],
        )
        self.assertTrue(all(entry.has_images for entry in limited_history))
        self.assertEqual(len(limited_images), 6)

    def test_request_keeps_existing_metadata_and_order(self):
        request = CombinedModelClient.make_request(
            make_bundle(),
            session_id="robot_test",
            instruction="go left",
            decision_index=2,
        )
        self.assertEqual(request.schema_version, 2)
        self.assertEqual(request.request_type, "end2end_decision")
        self.assertEqual(request.observation_id, "robot_test_decision_002")
        self.assertEqual(tuple(request.current_panorama), DIRECTION_ORDER)
        self.assertEqual(
            request.required_image_fields,
            (
                "current_forward",
                "current_left",
                "current_behind",
                "current_right",
            ),
        )

    def test_multipart_field_order_matches_request(self):
        request = CombinedModelClient.make_request(
            make_bundle(),
            session_id="robot_test",
            instruction="go left",
            decision_index=0,
        )
        fake_png = b"\x89PNG\r\n\x1a\nplaceholder"
        images = {name: fake_png for name in request.required_image_fields}
        body, _boundary = CombinedModelClient._multipart_body(request, images)
        positions = [
            body.index(f'name="{name}"'.encode("ascii"))
            for name in request.required_image_fields
        ]
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
