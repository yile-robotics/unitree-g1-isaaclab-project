from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "isaacsim_goal_tracking"))
import g1_camera_model_probe as probe
from mock_lavira_server import parse_multipart
from unified_vln.types import DIRECTION_ORDER, PanoramaBundle, ViewFrame


def fixture_bundle():
    views = {}
    for index, direction in enumerate(DIRECTION_ORDER):
        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        rgb[:, :, 0] = 20 + index  # 通道不对称，能检测RGB/BGR保存错误。
        rgb[:, :, 2] = 170
        views[direction] = ViewFrame(direction, index + 1, 0, index * 0.1, rgb,
                                     np.full((8, 12), 1.25, np.float32),
                                     np.array([[10, 0, 6], [0, 10, 4], [0, 0, 1]], dtype=float))
    return PanoramaBundle(0, 0, 0.3, views).validated()


def save_fixture(output):
    bundle = fixture_bundle()
    records = {d: probe.save_frame(output, f, {}) for d, f in bundle.views.items()}
    probe.write_json(output / "panorama.json", {"format_version": 1, "views": records})
    return bundle


class CameraModelProbeTest(unittest.TestCase):
    def test_save_replay_preserves_color_depth_and_intrinsics(self):
        with tempfile.TemporaryDirectory() as name:
            original = save_fixture(Path(name))
            replay = probe.load_panorama(Path(name))
            for direction in DIRECTION_ORDER:
                for field in ("rgb", "depth_m", "K"):
                    np.testing.assert_array_equal(getattr(original.views[direction], field),
                                                  getattr(replay.views[direction], field))

    def test_rejects_incomplete_four_direction_capture(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            save_fixture(output)
            manifest = json.loads((output / "panorama.json").read_text())
            del manifest["views"]["right"]
            probe.write_json(output / "panorama.json", manifest)
            with self.assertRaisesRegex(ValueError, "完整四方向"):
                probe.load_panorama(output)

    def test_rejects_corrupt_depth(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            save_fixture(output)
            depth = np.full((8, 12), 1.25, np.float32)
            depth[0, 0] = np.nan
            np.save(output / "forward_depth_m.npy", depth)
            with self.assertRaisesRegex(ValueError, "非负有限"):
                probe.load_panorama(output)

    def test_ctrl_c_closes_camera_and_leaves_partial_capture_unusable(self):
        class Camera:
            closed = False
            last_metadata = {}

            def capture_forward(self, *args):
                return fixture_bundle().views["forward"]

            def close(self):
                self.closed = True

        camera = Camera()
        with tempfile.TemporaryDirectory() as name, patch.object(probe, "create_camera", return_value=camera), \
                patch("builtins.input", side_effect=["", KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                probe.capture_panorama(Path("unused"), Path(name), headless=True)
            self.assertTrue(camera.closed)
            with self.assertRaisesRegex(ValueError, "完整四方向"):
                probe.load_panorama(Path(name))

    def test_legacy_real_http_roundtrip_and_bbox_output(self):
        observed = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                fields = parse_multipart(self.headers["Content-Type"],
                                         self.rfile.read(int(self.headers["Content-Length"])))
                meta = json.loads(fields["metadata"][1])
                observed.update(meta)
                observed["images"] = sorted(set(fields) - {"metadata"})
                response = {"schema_version": 2, "response_type": "end2end_decision",
                            "session_id": meta["session_id"], "observation_id": meta["observation_id"],
                            "action": "NAVIGATE", "direction": "left", "target": "test door",
                            "bbox_2d": [1, 1, 8, 6], "waypoint": None,
                            "progress_analysis": "test fixture", "reasoning": "mock only"}
                payload = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as name:
                output = Path(name)
                raw = probe.request_decision(fixture_bundle(), output, "find the door",
                                             f"http://127.0.0.1:{server.server_port}/v1/lavira/decision",
                                             2, legacy=True)
                self.assertEqual(raw["direction"], "left")
                self.assertEqual(observed["instruction"], "find the door")
                self.assertEqual(observed["images"], sorted(f"current_{d}" for d in DIRECTION_ORDER))
                self.assertTrue((output / "decision_bbox.png").exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_g3_ends_cancelled_without_execution_reports(self):
        from unittest.mock import MagicMock
        session = MagicMock()
        session.health_check.return_value = {"status": "ok"}
        session.start_session.return_value = (None, {"status": "ACTIVE"})
        session.end_session.return_value = (None, {"status": "ENDED"})
        with tempfile.TemporaryDirectory() as name, \
                patch.object(probe.G3SessionClient, "from_decision_url", return_value=session), \
                patch.object(probe.CombinedModelClient, "decide", return_value=(None, {"control": "SAFE_STOP"})):
            probe.request_decision(fixture_bundle(), Path(name), "test", "http://localhost:1/v1/lavira/decision", 2)
        session.end_session.assert_called_once_with(status="CANCELLED", reason="static_probe_no_robot_execution")
        session.report_motion_window.assert_not_called()
        session.report_action_complete.assert_not_called()
        session.validate_decision_context.assert_called_once()


if __name__ == "__main__":
    unittest.main()
