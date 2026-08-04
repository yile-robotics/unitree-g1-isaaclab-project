from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.model_client import CombinedModelClient  # noqa: E402
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
