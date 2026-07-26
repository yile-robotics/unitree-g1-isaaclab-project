from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from goal_tracking.camera import FOUR_VIEW_DIRECTIONS  # noqa: E402
from goal_tracking.frame_bundle import CameraFrame, FrameBundle  # noqa: E402
from goal_tracking.lavira_offline import (  # noqa: E402
    NavigationDecisionOfflineProbe,
    make_first_navigation_decision_request,
)
from goal_tracking.lavira_protocol import NavigationDecisionResponse  # noqa: E402
from goal_tracking.target_projection import (  # noqa: E402
    project_navigation_target,
    save_target_projection_debug,
)


def _make_transform(
    translation: tuple[float, float, float], yaw_deg: float = 0.0
) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform[:3, 3] = translation
    return transform


def _make_bundle(selected_depth: np.ndarray) -> FrameBundle:
    height, width = selected_depth.shape
    K = np.array(
        [[100.0, 0.0, 5.0], [0.0, 200.0, 4.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    T_world_base = _make_transform((10.0, 20.0, 0.0), yaw_deg=90.0)
    T_base_camera = _make_transform((0.1, -0.2, 1.0))
    T_world_camera = T_world_base @ T_base_camera
    views = {}
    for frame_id, direction in enumerate(FOUR_VIEW_DIRECTIONS, start=1):
        depth = (
            selected_depth.copy()
            if direction == "right"
            else np.full((height, width), 1.0, dtype=np.float32)
        )
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., frame_id % 3] = 40 * frame_id
        views[direction] = CameraFrame(
            camera_id=f"camera_{direction}",
            direction=direction,
            sensor_frame_id=frame_id,
            sim_step=5,
            timestamp=0.1,
            rgb=rgb,
            depth_z_m=depth,
            K=K.copy(),
            T_world_camera_ros=T_world_camera.copy(),
            T_base_camera=T_base_camera.copy(),
        )
    return FrameBundle(
        bundle_id=0,
        env_index=0,
        sim_step=5,
        timestamp=0.1,
        T_world_base=T_world_base,
        views=views,
    )


def _make_request_and_response(bundle: FrameBundle):
    request = make_first_navigation_decision_request(
        bundle,
        session_id="robot_01_projection_test",
        instruction="Navigate toward the bed.",
    )
    response = NavigationDecisionResponse(
        session_id=request.session_id,
        observation_id=request.observation_id,
        action="NAVIGATE",
        direction="right",
        target="bed",
        # x2/y2 follow the protocol's exclusive outer-bound convention.  y2 at
        # image height exercises the safe clamp before selecting the bottom point.
        bbox_2d=(2, 1, 8, 10),
        waypoint=None,
        progress_analysis="The bed is visible in the right view.",
        reasoning="Approach the bed.",
    )
    return request, response


class TargetProjectionTest(unittest.TestCase):
    def test_lavira_bottom_center_depth_and_full_transforms(self) -> None:
        depth = np.full((10, 12), np.nan, dtype=np.float32)
        # Safe-clipped bbox bottom centre is (5, 9).  The clipped 3x3 patch is
        # x=[4,7), y=[8,10), with five valid values whose median is 3 m.
        depth[8:10, 4:7] = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32
        )
        bundle = _make_bundle(depth)
        request, response = _make_request_and_response(bundle)

        result = project_navigation_target(
            request,
            bundle,
            response,
            min_depth_m=0.1,
            max_depth_m=5.0,
        )

        self.assertEqual(result.selected_pixel_uv, (5, 9))
        self.assertEqual(result.depth_window_size, 3)
        self.assertEqual(result.depth_window_xyxy_exclusive, (4, 8, 7, 10))
        self.assertEqual(result.valid_depth_count, 5)
        self.assertAlmostEqual(result.valid_depth_fraction, 5.0 / 6.0)
        self.assertAlmostEqual(result.selected_depth_median_m, 3.0)
        np.testing.assert_allclose(result.point_camera_ros_m, [0.0, 0.075, 3.0])
        np.testing.assert_allclose(result.point_base_m, [0.1, -0.125, 4.0])
        np.testing.assert_allclose(result.point_world_m, [10.125, 20.1, 4.0])
        self.assertFalse(result.to_dict()["motion_goal"])

    def test_depth_search_expands_in_lavira_window_order(self) -> None:
        depth = np.full((10, 12), np.nan, dtype=np.float32)
        # This sample is outside the clipped 3x3 patch but inside the 5x5 patch.
        depth[7, 7] = 2.25
        bundle = _make_bundle(depth)
        request, response = _make_request_and_response(bundle)

        result = project_navigation_target(
            request,
            bundle,
            response,
            min_depth_m=0.1,
            max_depth_m=5.0,
        )
        self.assertEqual(result.depth_window_size, 5)
        self.assertAlmostEqual(result.selected_depth_median_m, 2.25)

    def test_no_valid_depth_rejects_fabricated_fallback_goal(self) -> None:
        bundle = _make_bundle(np.full((10, 12), np.inf, dtype=np.float32))
        request, response = _make_request_and_response(bundle)

        with self.assertRaisesRegex(ValueError, "no fallback goal was fabricated"):
            project_navigation_target(
                request,
                bundle,
                response,
                min_depth_m=0.1,
                max_depth_m=5.0,
            )

    def test_projection_debug_files_are_local_and_reproducible(self) -> None:
        depth = np.full((10, 12), 2.0, dtype=np.float32)
        bundle = _make_bundle(depth)
        request, response = _make_request_and_response(bundle)
        result = project_navigation_target(
            request,
            bundle,
            response,
            min_depth_m=0.1,
            max_depth_m=5.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            files = save_target_projection_debug(
                Path(temp_dir),
                bundle,
                result,
                near_m=0.1,
                far_m=5.0,
            )
            self.assertEqual(
                set(files),
                {
                    "projection_json",
                    "annotated_rgb",
                    "metric_depth_npy",
                    "annotated_depth_preview",
                },
            )
            for filename in files.values():
                self.assertTrue((Path(temp_dir) / filename).is_file())
            restored_depth = np.load(Path(temp_dir) / files["metric_depth_npy"])
            np.testing.assert_array_equal(restored_depth, depth)

    def test_one_shot_probe_saves_projection_without_motion_output(self) -> None:
        bundle = _make_bundle(np.full((10, 12), 2.0, dtype=np.float32))

        class FakeCameraRig:
            def capture(self, sim_step: int, timestamp: float) -> FrameBundle:
                self.last_capture = (sim_step, timestamp)
                return bundle

        def fake_post(_url, request, _images, *, timeout_seconds):
            self.assertEqual(timeout_seconds, 90.0)
            response = NavigationDecisionResponse(
                session_id=request.session_id,
                observation_id=request.observation_id,
                action="NAVIGATE",
                direction="right",
                target="bed",
                bbox_2d=(2, 1, 8, 10),
                waypoint=None,
                progress_analysis="The bed is visible.",
                reasoning="Navigate toward it.",
            )
            return response, response.to_dict()

        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                lavira_decision_probe=True,
                lavira_decision_warmup_steps=5,
                lavira_session_id="robot_01_projection_probe",
                instruction="Navigate toward the bed.",
                lavira_output_dir=Path(temp_dir),
                lavira_server_url="http://127.0.0.1:8765/v1/lavira/decision",
                lavira_timeout=90.0,
                rgbd_camera_near=0.1,
                rgbd_camera_far=5.0,
                lavira_projection_debug_marker=False,
                lavira_local_map_probe=True,
                lavira_fmm_probe=True,
                nav_map_resolution_m=0.05,
                nav_map_size_m=24.0,
                nav_depth_stride=1,
                nav_nominal_base_height_m=0.80,
                nav_floor_search_half_range_m=0.30,
                nav_obstacle_min_height_m=0.10,
                nav_obstacle_max_height_m=1.60,
                nav_robot_radius_m=0.35,
                nav_target_retreat_step_m=0.10,
                nav_target_snap_max_m=1.00,
                fmm_step_size_cells=5,
                fmm_goal_tolerance_cells=1,
                fmm_waypoint_spacing_m=0.25,
                fmm_max_path_steps=20_000,
                headless=True,
            )
            probe = NavigationDecisionOfflineProbe(args)
            camera_rig = FakeCameraRig()
            with patch(
                "goal_tracking.lavira_offline.post_navigation_decision",
                side_effect=fake_post,
            ):
                probe.maybe_run(camera_rig, completed_step=5, step_dt=0.02)

            self.assertTrue(probe.completed)
            self.assertIsNotNone(probe.target_projection)
            self.assertEqual(camera_rig.last_capture, (5, 0.1))
            self.assertIsNotNone(probe.output_dir)
            for filename in (
                "metadata.json",
                "response.json",
                "response_interpretation.json",
                "target_projection.json",
                "target_projection_rgb.png",
                "target_projection_depth_m.npy",
                "target_projection_depth_preview.png",
                "navigation_map.json",
                "navigation_map.npz",
                "navigation_map.png",
                "fmm_plan.json",
                "fmm_distance.npy",
                "fmm_path.png",
            ):
                self.assertTrue((probe.output_dir / filename).is_file())
            self.assertIsNotNone(probe.navigation_map)
            self.assertIsNotNone(probe.fmm_plan)


if __name__ == "__main__":
    unittest.main()
