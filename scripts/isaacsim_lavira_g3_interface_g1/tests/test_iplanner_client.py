from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.iplanner_client import IPlannerClient  # noqa: E402
from unified_vln.types import ViewFrame  # noqa: E402


class _RecordingUniClient:
    initialized = True

    def __init__(self):
        self.arguments = None

    def get_plan(self, rgb_bgr, depth_mm, goal_local):
        self.arguments = (rgb_bgr, depth_mm, goal_local)
        return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), 0.25


class IPlannerClientAdapterTest(unittest.TestCase):
    def test_adapter_uses_uni_bgr_and_integer_millimetre_inputs(self):
        client = IPlannerClient("http://127.0.0.1:8888", timeout_s=5.0)
        recorder = _RecordingUniClient()
        client._client = recorder
        frame = ViewFrame(
            direction="forward",
            frame_id=1,
            sim_step=1,
            timestamp=0.0,
            rgb=np.array([[[10, 20, 30]]], dtype=np.uint8),
            depth_m=np.array([[1.2349]], dtype=np.float32),
            K=np.eye(3),
        )

        trajectory, fear = client.get_plan(frame, np.array([2.0, -0.5]))

        rgb_bgr, depth_mm, goal_local = recorder.arguments
        np.testing.assert_array_equal(rgb_bgr, [[[30, 20, 10]]])
        np.testing.assert_array_equal(depth_mm, [[1234]])
        self.assertEqual(depth_mm.dtype, np.uint16)
        np.testing.assert_allclose(goal_local, [2.0, -0.5])
        np.testing.assert_allclose(trajectory[-1], [1.0, 0.0, 0.0])
        self.assertEqual(fear, 0.25)

    def test_exact_uni_mode_rejects_a_different_timeout(self):
        with self.assertRaisesRegex(ValueError, "timeout_s=5.0"):
            IPlannerClient("http://127.0.0.1:8888", timeout_s=30.0)


if __name__ == "__main__":
    unittest.main()
