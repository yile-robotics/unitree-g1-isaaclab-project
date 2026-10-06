"""真实文件保存与回正图保留；使用假相机，不发送机器人命令。"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rotation_capture import RotationCapture
from g1_camera_model_probe import load_panorama
from unified_vln.types import ViewFrame


class FakeCamera:
    last_metadata = {"serial": "test"}
    closed = False
    sequence = 0

    def capture_forward(self, step, timestamp):
        self.sequence += 1
        return ViewFrame("forward", self.sequence, step, timestamp,
                         np.full((8, 12, 3), self.sequence, np.uint8),
                         np.ones((8, 12), np.float32),
                         np.array([[10, 0, 6], [0, 10, 4], [0, 0, 1]], float))

    def close(self):
        self.closed = True


class RotationCaptureTest(unittest.TestCase):
    def record(self, root, desk=False, manual_follow=False):
        camera = FakeCamera()
        with patch("rotation_capture.create_camera", return_value=camera):
            recorder = RotationCapture(Path("unused"), root / "capture", desk, manual_follow=manual_follow)
        recorder.bind_monitors(SimpleNamespace(require_fresh=lambda: 0.1),
                               SimpleNamespace(snapshot=lambda: (
                                   {"x": 0.0, "y": 0.0, "z": 0.0,
                                    "q_x": 0.0, "q_y": 0.0, "q_z": 0.0, "q_w": 1.0},
                                   time.monotonic(), "")))
        return recorder, camera

    def test_forward_return_preserves_initial_image_and_mounted_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder, camera = self.record(Path(tmp))
            recorder.preflight()
            for quarter, direction in enumerate(("forward", "left", "behind", "right", "forward_return")):
                recorder.capture(direction, quarter)
            recorder.mark_rotation_started()
            recorder.finish("PASS", total_slam_turn_deg=359.9)
            recorder.close()
            self.assertTrue(camera.closed)
            bundle = load_panorama(recorder.output)
            self.assertEqual(bundle.views["forward"].frame_id, 2)
            self.assertTrue(np.all(bundle.views["forward"].rgb == 2))
            manifest = json.loads((recorder.output / "panorama.json").read_text())
            self.assertEqual(manifest["return_view"]["frame_id"], 6)
            self.assertIn("slam_currentPose", manifest["views"]["left"]["pose_after_capture"])

    def test_desk_capture_cannot_be_replayed_as_directional_panorama(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder, _ = self.record(Path(tmp), desk=True)
            recorder.close()
            with self.assertRaisesRegex(ValueError, "不能作为真实四方向"):
                load_panorama(recorder.output)

    def test_interruption_closes_camera_and_marks_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder, camera = self.record(Path(tmp))
            recorder.capture("forward", 0)
            recorder.close()
            self.assertTrue(camera.closed)
            self.assertEqual(recorder.manifest["status"], "INTERRUPTED")
            with self.assertRaisesRegex(ValueError, "流程未完成"):
                load_panorama(recorder.output)

    def test_manual_follow_is_not_recorded_as_verified_camera_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder, _ = self.record(Path(tmp), manual_follow=True)
            recorder.close()
            self.assertEqual(recorder.manifest["capture_mode"], "rotation_manual_follow")
            self.assertIsNone(recorder.manifest["camera_moves_with_robot"])
            with self.assertRaisesRegex(ValueError, "手动跟随旋转"):
                load_panorama(recorder.output)


if __name__ == "__main__":
    unittest.main()
