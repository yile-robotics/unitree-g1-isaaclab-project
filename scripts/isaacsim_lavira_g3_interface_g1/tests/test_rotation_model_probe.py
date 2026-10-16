"""仅使用替身，验证旋转和模型请求的先后及失败行为。"""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g1_rotation_model_probe as probe


class RotationModelProbeTests(unittest.TestCase):
    def run_flow(self, *, status="PASS", rotation_error=None, health_error=None, model_error=None):
        events = []
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "flow"

        def health():
            events.append("health")
            if health_error:
                raise health_error
            return {"status": "ok"}

        def rotate(interface, config, panorama):
            events.append("rotation")
            panorama.mkdir()
            probe.write_json(panorama / "panorama.json", {
                "status": status, "capture_mode": "robot_mounted_rotation",
                "total_slam_turn_deg": 359.5,
            })
            if rotation_error:
                raise rotation_error

        def decision(*args):
            events.append("decision")
            if model_error:
                raise model_error
            return {"action": "NAVIGATE", "direction": "forward"}

        with patch.object(probe.G3SessionClient, "from_decision_url") as client, \
                patch.object(probe, "run_rotation", side_effect=rotate) as rotation, \
                patch.object(probe, "load_panorama", return_value=object()) as load, \
                patch.object(probe, "request_decision", side_effect=decision) as request, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            client.return_value.health_check.side_effect = health
            code = probe.main([
                "--network-interface", "test_nic", "--instruction", "Go to the black suitcase.",
                "--output", str(output), "--execute",
            ])
        report = json.loads((output / "flow_report.json").read_text())
        return code, report, events, rotation.call_count, load.call_count, request.call_count, output

    def test_success_requests_exactly_one_decision_after_capture(self):
        code, report, events, _, _, calls, _ = self.run_flow()
        self.assertEqual(code, 0)
        self.assertEqual(events, ["health", "rotation", "decision"])
        self.assertEqual(calls, 1)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["iplanner_called"])
        self.assertFalse(report["model_action_executed"])
        self.assertEqual(report["total_slam_turn_deg"], 359.5)

    def test_health_failure_prevents_rotation_and_model_request(self):
        code, report, events, rotations, _, calls, _ = self.run_flow(health_error=RuntimeError("offline"))
        self.assertEqual(code, 1)
        self.assertEqual(events, ["health"])
        self.assertEqual((rotations, calls), (0, 0))
        self.assertEqual(report["stage"], "model_health")

    def test_zero_exit_with_interrupted_capture_does_not_upload(self):
        code, report, _, _, loads, calls, _ = self.run_flow(status="INTERRUPTED")
        self.assertEqual(code, 1)
        self.assertEqual((loads, calls), (0, 0))
        self.assertFalse(report["rotation_capture_completed"])

    def test_rotation_failure_prevents_upload(self):
        code, report, _, _, _, calls, _ = self.run_flow(
            rotation_error=subprocess.CalledProcessError(1, ["probe"]))
        self.assertEqual(code, 1)
        self.assertEqual(calls, 0)
        self.assertEqual(report["stage"], "rotation_capture")

    def test_ctrl_c_does_not_start_model_request(self):
        code, report, _, _, _, calls, _ = self.run_flow(rotation_error=KeyboardInterrupt())
        self.assertEqual(code, 130)
        self.assertEqual(calls, 0)
        self.assertEqual(report["status"], "CANCELLED")

    def test_model_failure_preserves_capture_for_replay(self):
        code, report, _, _, _, calls, output = self.run_flow(model_error=RuntimeError("model timeout"))
        self.assertEqual(code, 1)
        self.assertEqual(calls, 1)
        self.assertTrue(report["rotation_capture_completed"])
        self.assertTrue((output / "panorama/panorama.json").exists())
        self.assertEqual(report["stage"], "model_decision")


if __name__ == "__main__":
    unittest.main()
