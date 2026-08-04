from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / "isaacsim_goal_tracking"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LEGACY))

try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = None

from mock_lavira_server import parse_multipart, png_dimensions  # noqa: E402
from unified_vln.model_client import CombinedModelClient  # noqa: E402
from unified_vln.model_contract import NavigationDecisionRequest  # noqa: E402
from unified_vln.types import DIRECTION_ORDER, PanoramaBundle, ViewFrame  # noqa: E402


@unittest.skipIf(cv2 is None, "OpenCV is required for the real PNG wire test")
class ModelMockRoundTripTest(unittest.TestCase):
    def test_new_wire_is_accepted_by_existing_mock_parser(self):
        views = {}
        for index, direction in enumerate(DIRECTION_ORDER):
            views[direction] = ViewFrame(
                direction=direction,
                frame_id=index,
                sim_step=2,
                timestamp=0.1,
                rgb=np.full((4, 6, 3), index * 20, dtype=np.uint8),
                depth_m=np.ones((4, 6), dtype=np.float32),
                K=np.eye(3),
            )
        bundle = PanoramaBundle(0, 2, 0.1, views)
        client = CombinedModelClient("http://127.0.0.1:1/v1/lavira/decision")
        request = client.make_request(
            bundle,
            session_id="wire_test",
            instruction="go through the doorway",
            decision_index=0,
        )
        images = client.image_fields(bundle, request)
        body, boundary = client._multipart_body(request, images)
        fields = parse_multipart(
            f"multipart/form-data; boundary={boundary}", body
        )
        metadata = NavigationDecisionRequest.from_metadata(
            json.loads(fields["metadata"][1].decode("utf-8"))
        )
        self.assertEqual(metadata.to_metadata(), request.to_metadata())
        self.assertEqual(
            set(fields), {"metadata", *request.required_image_fields}
        )
        for field_name in request.required_image_fields:
            filename, payload = fields[field_name]
            self.assertEqual(filename, f"{field_name}.png")
            self.assertEqual(png_dimensions(payload), (6, 4))


if __name__ == "__main__":
    unittest.main()
