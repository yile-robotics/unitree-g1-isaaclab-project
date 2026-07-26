from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from goal_tracking.frame_bundle import CameraFrame, FrameBundle  # noqa: E402
from goal_tracking.lavira_offline import (  # noqa: E402
    _multipart_body,
    encode_rgb_png,
    interpret_navigation_decision,
    make_first_navigation_decision_request,
    make_navigation_decision_request,
    panorama_png_fields,
    post_navigation_decision,
    save_navigation_decision_request,
)
from goal_tracking.lavira_protocol import (  # noqa: E402
    NavigationDecisionResponse,
    NavigationHistoryEntry,
)
from mock_lavira_server import (  # noqa: E402
    MockLaViRAHandler,
    parse_multipart,
    png_dimensions,
)


def make_bundle() -> FrameBundle:
    colors = {
        "forward": (255, 0, 0),
        "left": (0, 255, 0),
        "behind": (0, 0, 255),
        "right": (255, 255, 0),
    }
    views = {}
    for frame_id, (direction, color) in enumerate(colors.items(), start=1):
        rgb = np.empty((6, 8, 3), dtype=np.uint8)
        rgb[:, :] = color
        views[direction] = CameraFrame(
            camera_id=f"camera_{direction}",
            direction=direction,
            sensor_frame_id=frame_id,
            sim_step=5,
            timestamp=0.1,
            rgb=rgb,
            depth_z_m=np.ones((6, 8), dtype=np.float32),
            K=np.eye(3),
            T_world_camera_ros=np.eye(4),
            T_base_camera=np.eye(4),
        )
    return FrameBundle(
        bundle_id=0,
        env_index=0,
        sim_step=5,
        timestamp=0.1,
        T_world_base=np.eye(4),
        views=views,
    )


def make_history(count: int) -> tuple[NavigationHistoryEntry, ...]:
    image_start = max(0, count - 4)
    return tuple(
        NavigationHistoryEntry(
            waypoint_id=index,
            step=index * 10,
            turn_action="turn forward",
            description=f"target_{index}",
            init_image_field=(f"history_{index}_init" if index >= image_start else None),
            dir_image_field=(f"history_{index}_dir" if index >= image_start else None),
        )
        for index in range(count)
    )


class LaViRAHttpProbeTest(unittest.TestCase):
    def test_png_round_trip_preserves_rgb(self) -> None:
        rgb = np.array([[[12, 34, 56], [200, 150, 100]]], dtype=np.uint8)
        payload = encode_rgb_png(rgb)
        decoded_bgr = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(decoded_rgb, rgb)

    def test_mock_server_parses_generated_multipart_without_socket(self) -> None:
        bundle = make_bundle()
        request = make_first_navigation_decision_request(
            bundle,
            session_id="robot_01_task_001",
            instruction="Go to the bed.",
        )
        images = panorama_png_fields(bundle, request)
        body, boundary = _multipart_body(request, images)
        fields = parse_multipart(f"multipart/form-data; boundary={boundary}", body)

        restored = request.from_metadata(
            json.loads(fields["metadata"][1].decode("utf-8"))
        )
        self.assertEqual(restored, request)
        self.assertEqual(set(fields), {"metadata", *request.required_image_fields})
        for field_name in request.required_image_fields:
            self.assertEqual(png_dimensions(fields[field_name][1]), (8, 6))

    def test_six_history_waypoints_send_only_latest_four_image_pairs(self) -> None:
        bundle = make_bundle()
        history = make_history(6)
        request = make_navigation_decision_request(
            bundle,
            session_id="robot_01_task_001",
            instruction="Go to the bed.",
            decision_index=6,
            history=history,
        )
        history_images = {}
        for entry in history:
            if entry.has_images:
                history_images[entry.init_image_field] = np.zeros(
                    (6, 8, 3), dtype=np.uint8
                )
                history_images[entry.dir_image_field] = np.full(
                    (6, 8, 3), 127, dtype=np.uint8
                )
        images = panorama_png_fields(bundle, request, history_images)
        body, boundary = _multipart_body(request, images)
        fields = parse_multipart(f"multipart/form-data; boundary={boundary}", body)
        metadata = json.loads(fields["metadata"][1].decode("utf-8"))

        self.assertEqual(len(images), 12)
        self.assertNotIn("init_image_field", metadata["history"][0])
        self.assertNotIn("dir_image_field", metadata["history"][1])
        self.assertEqual(metadata["history"][2]["init_image_field"], "history_2_init")

    def test_response_binds_direction_to_same_frame_bundle(self) -> None:
        bundle = make_bundle()
        request = make_first_navigation_decision_request(
            bundle,
            session_id="robot_01_task_001",
            instruction="Go to the bed.",
        )
        response = NavigationDecisionResponse(
            session_id=request.session_id,
            observation_id=request.observation_id,
            action="STOP",
            direction="right",
            target="bed",
            bbox_2d=(0, 0, 8, 6),
            waypoint=None,
            progress_analysis="The final target is visible.",
            reasoning="Approach it and stop.",
        )
        interpreted = interpret_navigation_decision(request, bundle, response)

        self.assertEqual(interpreted["selected_image_field"], "current_right")
        self.assertEqual(interpreted["selected_camera_id"], "camera_right")
        self.assertEqual(interpreted["selected_sensor_frame_id"], 4)
        self.assertEqual(interpreted["bbox_response"], [0, 0, 8, 6])
        self.assertEqual(interpreted["bbox_clipped"], [0, 0, 7, 5])
        self.assertTrue(interpreted["final_approach"])

    def test_backtrack_interpretation_uses_only_waypoint(self) -> None:
        bundle = make_bundle()
        request = make_navigation_decision_request(
            bundle,
            session_id="robot_01_task_001",
            instruction="Go to the bed.",
            decision_index=1,
            history=make_history(1),
        )
        response = NavigationDecisionResponse(
            session_id=request.session_id,
            observation_id=request.observation_id,
            action="BACKTRACK",
            direction=None,
            target=None,
            bbox_2d=None,
            waypoint=0,
            progress_analysis="The current route is unproductive.",
            reasoning="Return to waypoint zero.",
        )
        interpreted = interpret_navigation_decision(request, bundle, response)

        self.assertEqual(interpreted["history_waypoint"], 0)
        self.assertEqual(interpreted["used_fields"], ["action", "waypoint"])
        self.assertNotIn("selected_direction", interpreted)

    def test_real_multipart_round_trip_and_offline_files(self) -> None:
        bundle = make_bundle()
        request = make_first_navigation_decision_request(
            bundle,
            session_id="robot_01_task_001",
            instruction="Go to the bed.",
        )
        images = panorama_png_fields(bundle, request)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "decision_000"
            save_navigation_decision_request(output_dir, request, images)
            self.assertTrue((output_dir / "metadata.json").is_file())
            for field_name in request.required_image_fields:
                self.assertTrue((output_dir / f"{field_name}.png").is_file())

            try:
                server = ThreadingHTTPServer(("127.0.0.1", 0), MockLaViRAHandler)
            except PermissionError:
                self.skipTest("Execution sandbox does not allow localhost sockets.")
            server.fake_action = "NAVIGATE"
            server.fake_direction = "left"
            server.fake_target = "doorway"
            server.fake_bbox = (0, 0, 12, 5)
            server.fake_waypoint = 3
            server.fake_delay_seconds = 0.0
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                response, raw = post_navigation_decision(
                    f"http://127.0.0.1:{port}/v1/lavira/decision",
                    request,
                    images,
                    timeout_seconds=2.0,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

        self.assertEqual(response.action, "NAVIGATE")
        self.assertEqual(response.direction, "left")
        self.assertEqual(response.target, "doorway")
        self.assertEqual(response.bbox_2d, (0, 0, 8, 5))
        self.assertIsNone(response.waypoint)
        self.assertEqual(raw["schema_version"], 2)
        self.assertEqual(raw["response_type"], "end2end_decision")
        self.assertEqual(raw["session_id"], request.session_id)
        self.assertEqual(raw["observation_id"], request.observation_id)


if __name__ == "__main__":
    unittest.main()
