from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from goal_tracking.lavira_protocol import (  # noqa: E402
    NavigationDecisionRequest,
    NavigationDecisionResponse,
    NavigationHistoryEntry,
)
from goal_tracking.config import build_parser  # noqa: E402


def make_history(count: int) -> tuple[NavigationHistoryEntry, ...]:
    image_start = max(0, count - 4)
    entries = []
    for waypoint_id in range(count):
        has_images = waypoint_id >= image_start
        entries.append(
            NavigationHistoryEntry(
                waypoint_id=waypoint_id,
                step=10 + waypoint_id * 20,
                turn_action=("turn left" if waypoint_id % 2 else "turn forward"),
                description=f"target_{waypoint_id}",
                init_image_field=(
                    f"history_{waypoint_id}_init" if has_images else None
                ),
                dir_image_field=(
                    f"history_{waypoint_id}_dir" if has_images else None
                ),
            )
        )
    return tuple(entries)


def make_request(history_count: int = 2) -> NavigationDecisionRequest:
    return NavigationDecisionRequest(
        session_id="robot_01_task_001",
        observation_id="robot_01_task_001_decision_002",
        bundle_id=2,
        decision_index=2,
        sim_step=103,
        timestamp=2.06,
        instruction="Go to the bed.",
        image_width=640,
        image_height=480,
        history=make_history(history_count),
        current_panorama={
            "forward": "current_forward",
            "left": "current_left",
            "behind": "current_behind",
            "right": "current_right",
        },
    )


class NavigationDecisionProtocolTest(unittest.TestCase):
    def test_default_endpoint_timeout_and_legacy_cli_alias(self) -> None:
        class StubAppLauncher:
            @staticmethod
            def add_app_launcher_args(parser) -> None:
                return

        parser = build_parser(StubAppLauncher)
        defaults = parser.parse_args([])
        legacy = parser.parse_args(["--lavira_la_probe"])

        self.assertEqual(
            defaults.lavira_server_url,
            "http://127.0.0.1:8765/v1/lavira/decision",
        )
        self.assertEqual(defaults.lavira_timeout, 90.0)
        self.assertFalse(defaults.lavira_local_map_probe)
        self.assertFalse(defaults.lavira_fmm_probe)
        self.assertFalse(defaults.lavira_execute_fmm_path)
        self.assertEqual(defaults.nav_map_resolution_m, 0.05)
        self.assertEqual(defaults.nav_map_size_m, 24.0)
        self.assertEqual(defaults.nav_depth_stride, 4)
        self.assertEqual(defaults.fmm_step_size_cells, 5)
        self.assertEqual(defaults.fmm_goal_tolerance_cells, 1)
        self.assertEqual(defaults.fmm_waypoint_spacing_m, 0.25)
        self.assertEqual(defaults.fmm_execute_start_tolerance_m, 0.15)
        self.assertEqual(defaults.fmm_execute_max_path_m, 2.0)
        self.assertEqual(defaults.lavira_backtrack_max_path_m, 6.0)
        self.assertEqual(
            defaults.lavira_backtrack_strategy,
            "replan_world_goal",
        )
        self.assertEqual(defaults.fmm_execute_max_vx, 0.20)
        self.assertTrue(legacy.lavira_decision_probe)

    def test_request_round_trip_preserves_schema_and_order(self) -> None:
        request = make_request(history_count=6)
        restored = NavigationDecisionRequest.from_metadata(request.to_metadata())

        self.assertEqual(restored, request)
        self.assertEqual(request.schema_version, 2)
        self.assertEqual(request.request_type, "end2end_decision")
        self.assertEqual(request.decision_index, 2)
        self.assertEqual(
            restored.required_image_fields,
            (
                "history_2_init",
                "history_2_dir",
                "history_3_init",
                "history_3_dir",
                "history_4_init",
                "history_4_dir",
                "history_5_init",
                "history_5_dir",
                "current_forward",
                "current_left",
                "current_behind",
                "current_right",
            ),
        )
        old_entry = restored.to_metadata()["history"][0]
        self.assertNotIn("init_image_field", old_entry)
        self.assertNotIn("dir_image_field", old_entry)

    def test_history_image_budget_for_0_2_4_6_waypoints(self) -> None:
        expected_counts = {0: 4, 2: 8, 4: 12, 6: 12}
        for history_count, image_count in expected_counts.items():
            with self.subTest(history_count=history_count):
                request = make_request(history_count=history_count)
                self.assertEqual(len(request.required_image_fields), image_count)

    def test_first_request_supports_empty_history(self) -> None:
        request = make_request(history_count=0)
        self.assertEqual(request.to_metadata()["history"], [])
        self.assertEqual(len(request.required_image_fields), 4)

    def test_invalid_panorama_order_is_rejected(self) -> None:
        request = make_request()
        with self.assertRaisesRegex(ValueError, "ordered exactly"):
            NavigationDecisionRequest(
                **{
                    **request.__dict__,
                    "current_panorama": {
                        "forward": "current_forward",
                        "right": "current_right",
                        "behind": "current_behind",
                        "left": "current_left",
                    },
                }
            )

    def test_history_waypoint_ids_must_be_contiguous(self) -> None:
        request = make_request(history_count=1)
        bad_entry = NavigationHistoryEntry(
            waypoint_id=1,
            step=10,
            turn_action="turn forward",
            description="doorway",
            init_image_field="history_1_init",
            dir_image_field="history_1_dir",
        )
        with self.assertRaisesRegex(ValueError, "contiguous and zero-based"):
            NavigationDecisionRequest(
                **{**request.__dict__, "history": (bad_entry,)}
            )

    def test_only_latest_four_history_waypoints_may_have_images(self) -> None:
        request = make_request(history_count=6)
        old_with_images = NavigationHistoryEntry(
            waypoint_id=0,
            step=10,
            turn_action="turn forward",
            description="old target",
            init_image_field="history_0_init",
            dir_image_field="history_0_dir",
        )
        with self.assertRaisesRegex(ValueError, "only the latest 4"):
            NavigationDecisionRequest(
                **{
                    **request.__dict__,
                    "history": (old_with_images, *request.history[1:]),
                }
            )

    def test_navigate_response_is_normalized_and_matches_request(self) -> None:
        request = make_request()
        response = NavigationDecisionResponse.from_dict(
            {
                "schema_version": 2,
                "response_type": "end2end_decision",
                "session_id": request.session_id,
                "observation_id": request.observation_id,
                "action": "NAVIGATE",
                "direction": "forward",
                "target": "bedroom",
                "bbox_2d": [0, 0, 640, 480],
                "waypoint": None,
                "progress_analysis": "Continue out of the closet.",
                "reasoning": "The bedroom is ahead.",
            }
        )
        response.validate_matches(request)

        self.assertIsNone(response.waypoint)
        self.assertEqual(response.clipped_bbox(640, 480), (0, 0, 639, 479))
        self.assertEqual(response.to_dict()["response_type"], "end2end_decision")

    def test_backtrack_uses_zero_based_history_waypoint(self) -> None:
        request = make_request()
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
        response.validate_matches(request)

    def test_backtrack_rejects_waypoint_missing_from_request_history(self) -> None:
        request = make_request()
        response = NavigationDecisionResponse(
            session_id=request.session_id,
            observation_id=request.observation_id,
            action="BACKTRACK",
            direction=None,
            target=None,
            bbox_2d=None,
            waypoint=2,
            progress_analysis="",
            reasoning="",
        )
        with self.assertRaisesRegex(ValueError, "not present in request.history"):
            response.validate_matches(request)

    def test_stop_keeps_final_approach_fields(self) -> None:
        request = make_request()
        response = NavigationDecisionResponse(
            session_id=request.session_id,
            observation_id=request.observation_id,
            action="STOP",
            direction="left",
            target="bed",
            bbox_2d=(120, 100, 500, 470),
            waypoint=None,
            progress_analysis="The final target is visible.",
            reasoning="Approach the bed and then stop.",
        )
        response.validate_matches(request)
        self.assertEqual(response.target, "bed")

    def test_response_rejects_stale_observation(self) -> None:
        request = make_request()
        response = NavigationDecisionResponse(
            session_id=request.session_id,
            observation_id="stale_observation",
            action="NAVIGATE",
            direction="left",
            target="doorway",
            bbox_2d=(1, 2, 300, 400),
            waypoint=None,
            progress_analysis="",
            reasoning="",
        )
        with self.assertRaisesRegex(ValueError, "observation_id does not match"):
            response.validate_matches(request)

    def test_response_rejects_bbox_outside_normalized_image(self) -> None:
        request = make_request()
        response = NavigationDecisionResponse(
            session_id=request.session_id,
            observation_id=request.observation_id,
            action="NAVIGATE",
            direction="forward",
            target="bedroom",
            bbox_2d=(0, 0, 644, 476),
            waypoint=None,
            progress_analysis="",
            reasoning="",
        )
        with self.assertRaisesRegex(ValueError, "must lie inside"):
            response.validate_matches(request)

    def test_response_requires_schema_v2_envelope(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing fields"):
            NavigationDecisionResponse.from_dict(
                {
                    "action": "NAVIGATE",
                    "direction": "left",
                    "target": "doorway",
                    "bbox_2d": [1, 2, 3, 4],
                    "waypoint": None,
                    "progress_analysis": "",
                    "reasoning": "",
                }
            )


if __name__ == "__main__":
    unittest.main()
