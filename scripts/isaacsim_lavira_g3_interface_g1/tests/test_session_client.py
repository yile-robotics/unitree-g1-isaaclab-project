from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unified_vln.session_client import (  # noqa: E402
    G3SessionClient,
    G3SessionProtocolError,
    base_url_from_decision_url,
)


PLAN_ID = "sha256:test-plan"


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, _limit):
        return self.payload


def _start_payload(session_id="session-test"):
    return {
        "schema_version": 2,
        "response_type": "session_started",
        "session_id": session_id,
        "status": "ACTIVE",
        "stage_plan_id": PLAN_ID,
        "stage_plan_status": "READY",
        "frozen_stage_plan": {
            "stage_plan_id": PLAN_ID,
            "stage_plan_status": "READY",
            "stage_total": 2,
            "subgoals": ["Go through the door.", "Stop near the sofa."],
        },
        "next_action": "REQUEST_DECISION",
    }


def _health_payload():
    return {
        "status": "ok",
        "schema_version": 2,
        "framework": "G3",
        "commit": "b92abb3",
        "audit_interval": 2,
        "execution_protocol": "phase3_map_progress_v1",
        "map_progress_required": True,
        "stage_progress_protocol": "phase4_stage_progress_v2",
        "stop_gate_protocol": "phase4_stop_gate_v1",
        "recovery_control_protocol": "phase6_recovery_control_v1",
        "phase4": {
            "status": "wired",
            "stage_progress": {
                "status": "wired",
                "frequency": "once_per_decision_after_navigator",
                "model_role": "stage_progress_shadow",
                "seed_namespace_offset": 1000,
            },
            "stop_gate": {
                "status": "wired",
                "trigger": "navigator_action_STOP",
                "frequency": "once_per_stop_proposal",
                "model_role": "stop_evaluator",
                "model_backend": "local_role_isolated",
                "observation": "same_decision_four_direction_views",
            },
            "physical_monitor": False,
            "preempt": False,
            "failure_verifier": False,
            "recovery": False,
            "backtrack": False,
        },
        "phase5": {
            "status": "wired",
            "physical_monitor": True,
            "failure_verifier": True,
            "preempt": True,
            "model_backend": "strong_api",
            "node_size_m": 0.75,
            "fmm_available": False,
        },
        "phase6": {
            "status": "wired",
            "recovery": True,
            "backtrack": True,
            "escape_evaluator": True,
            "model_backend": "strong_api",
            "waypoint_source": "session_waypoint_registry",
            "escape_confirmation_policy": "single_valid_positive",
            "control_protocol": "phase6_recovery_control_v1",
            "safe_stop_control_precedence": True,
            "ordinary_navigator_stop_schema_unchanged": True,
        },
        "phase7": {
            "status": "wired",
            "semantic_audit": True,
            "audit_interval_decisions": 2,
            "model_backend": "strong_api",
        },
    }


def _stage_progress_payload(*, decision_index=0, parse_success=True):
    common = {
        "decision_index": decision_index,
        "role_call_index": decision_index + 1,
        "role_seed": None,
        "image_count": 4,
        "model_backend": "local_navigator_shadow",
    }
    if parse_success:
        return {
            "stage_plan_id": PLAN_ID,
            "stage_total": 2,
            "stage_completed": 0,
            "stage_id": 0,
            "completed_stage_ids": [],
            "current_stage": "Go through the door.",
            "evidence_of_completion": [],
            "final_target_visible": False,
            "parse_success": True,
            "parse_error": None,
            **common,
        }
    return {
        "stage_plan_id": None,
        "stage_total": None,
        "stage_completed": None,
        "stage_id": None,
        "completed_stage_ids": [],
        "current_stage": None,
        "evidence_of_completion": [],
        "final_target_visible": None,
        "parse_success": False,
        "parse_error": "No JSON object found in model response",
        **common,
    }


def _navigate_decision_context(*, decision_index=0, parse_success=True):
    return {
        "session_id": "session-test",
        "session_status": "ACTIVE",
        "stage_plan_id": PLAN_ID,
        "stage_plan_status": "READY",
        "action": "NAVIGATE",
        "stage_progress": _stage_progress_payload(
            decision_index=decision_index,
            parse_success=parse_success,
        ),
        "stop_gate": None,
        "stop_phase": None,
    }


def _stop_decision_context(stop_phase, *, decision_index=1):
    progress = {
        **_stage_progress_payload(decision_index=decision_index),
        "stage_completed": 2 if stop_phase != "PREMATURE_STOP" else 0,
        "stage_id": 2 if stop_phase != "PREMATURE_STOP" else 0,
        "completed_stage_ids": [0, 1] if stop_phase != "PREMATURE_STOP" else [],
        "current_stage": (
            "COMPLETED" if stop_phase != "PREMATURE_STOP" else "Go through the door."
        ),
        "final_target_visible": stop_phase != "PREMATURE_STOP",
    }
    if stop_phase == "STOP_CONFIRMED":
        verdict = "ALLOW"
        completed = True
        missing_subgoal = None
        evidence = ["both frozen stages are complete"]
        parse_success = True
        parse_error = None
        control = "STOP_CONFIRMED"
        next_action = "END_SESSION_SUCCESS"
    elif stop_phase == "PREMATURE_STOP":
        verdict = "PREMATURE"
        completed = False
        missing_subgoal = "Go through the door."
        evidence = ["the door was not crossed"]
        parse_success = True
        parse_error = None
        control = "CONTINUE"
        next_action = "REQUEST_DECISION"
    else:
        verdict = "UNCERTAIN"
        completed = None
        missing_subgoal = None
        evidence = []
        parse_success = False
        parse_error = "two frozen-plan Stage Progress observations are required"
        control = "CONTINUE"
        next_action = "REQUEST_DECISION"
    return {
        "schema_version": 2,
        "response_type": "end2end_decision",
        "session_id": "session-test",
        "observation_id": f"session-test_decision_{decision_index:03d}",
        "action": "STOP",
        "direction": "forward",
        "target": "open door",
        "bbox_2d": [3, 3, 4, 4],
        "waypoint": None,
        "progress_analysis": "progress",
        "reasoning": "reason",
        "session_status": "ACTIVE",
        "stage_plan_id": PLAN_ID,
        "stage_plan_status": "READY",
        "stage_progress": progress,
        "stop_gate": {
            "verdict": verdict,
            "completed": completed,
            "missing_subgoal": missing_subgoal,
            "evidence": evidence,
            "parse_success": parse_success,
            "parse_error": parse_error,
            "decision_index": decision_index,
            "role_call_index": None if stop_phase == "STOP_PENDING" else 1,
            "role_seed": None,
            "image_count": 0 if stop_phase == "STOP_PENDING" else 4,
            "model_backend": "strong_api",
            "stop_phase": stop_phase,
        },
        "stop_phase": stop_phase,
        "control": control,
        "next_action": next_action,
    }


def _control_payload(event_type, session_id="session-test", decision_index=0):
    return {
        "schema_version": 2,
        "response_type": "execution_control",
        "session_id": session_id,
        "decision_index": decision_index,
        "control": "CONTINUE",
        "reason": "execution_completed",
        "event_type": event_type,
        "stop_phase": None,
        "stage_plan_id": PLAN_ID,
        "stage_plan_status": "READY",
    }


def _preempt_control_payload(*, decision_index=0):
    return {
        "schema_version": 2,
        "response_type": "execution_control",
        "session_id": "session-test",
        "decision_index": decision_index,
        "event_type": "motion_window",
        "control": "PREEMPT",
        "reason": "failure_verifier_confirmed",
        "next_action": "ACTION_COMPLETE_PREEMPTED",
        "stage_plan_status": "READY",
        "failure_verification": {
            "model_backend": "strong_api",
            "need_recovery": True,
            "parse_success": True,
            "verdict": "FAILURE",
        },
        "phase5": {
            "status": "PREEMPT_REQUESTED",
            "recovery_phase": "RECOVERY_PLANNING",
        },
    }


def _recovery_decision_context(action="NAVIGATE", *, decision_index=1):
    backtrack = action == "BACKTRACK"
    return {
        "schema_version": 2,
        "response_type": "end2end_decision",
        "session_id": "session-test",
        "session_status": "ACTIVE",
        "stage_plan_status": "READY",
        "observation_id": f"recovery-{decision_index}",
        "action": action,
        "action_source": "RECOVERY",
        "direction": None if backtrack else "left",
        "target": None if backtrack else "alternate corridor",
        "bbox_2d": None if backtrack else [0, 0, 2, 2],
        "waypoint": 0 if backtrack else None,
        "control": "CONTINUE",
        "next_action": "EXECUTE_RECOVERY",
        "phase6": {
            "action_source": "RECOVERY",
            "recovery_phase": "RECOVERY_PLANNING",
            "status": "RECOVERY_PLAN_READY",
            "waypoint_validated": True if backtrack else None,
        },
        "recovery": {
            "model_backend": "strong_api",
            "parse_success": True,
            "role": "recovery_planner",
            "status": "READY",
            "waypoint_validated": True if backtrack else None,
        },
    }


def _safe_stop_decision_context(*, decision_index=2):
    return {
        "schema_version": 2,
        "response_type": "end2end_decision",
        "session_id": "session-test",
        "observation_id": f"recovery-{decision_index}",
        "decision_index": decision_index,
        "session_status": "ACTIVE",
        "stage_plan_id": PLAN_ID,
        "stage_plan_status": "READY",
        "action": "STOP",
        "direction": None,
        "target": None,
        "bbox_2d": None,
        "waypoint": None,
        "progress_analysis": "RECOVERY_PLANNER_REJECTED",
        "reasoning": "RECOVERY_PLANNER_REJECTED",
        "control": "SAFE_STOP",
        "reason": "RECOVERY_PLANNER_REJECTED",
        "next_action": "SAFE_STOP",
        "action_source": "RECOVERY",
        "stop_phase": "SAFE_STOP",
        "stop_gate": None,
        "recovery": {
            "status": "REJECTED",
            "parse_success": False,
            "parse_error": "Recovery Planner output is invalid",
            "role": "recovery_planner",
            "role_call_index": 2,
            "role_seed": None,
            "image_count": 4,
            "model_backend": "strong_api",
            "waypoint_validated": None,
            "transition": {"to": "SAFE_STOP"},
        },
        "phase6": {
            "status": "SAFE_STOP",
            "reason": "RECOVERY_PLANNER_REJECTED",
            "recovery_phase": "SAFE_STOP",
            "transition": {"to": "SAFE_STOP"},
        },
    }


def _semantic_preempt_context(*, decision_index=2):
    return {
        "schema_version": 2,
        "response_type": "end2end_decision",
        "session_id": "session-test",
        "session_status": "ACTIVE",
        "stage_plan_status": "READY",
        "action": "NAVIGATE",
        "bbox_2d": [3, 3, 4, 4],
        "direction": "forward",
        "control": "PREEMPT",
        "next_action": "ACTION_COMPLETE_PREEMPTED",
        "failure_verification": {
            "model_backend": "strong_api",
            "need_recovery": True,
            "parse_success": True,
            "verdict": "FAILURE",
        },
        "phase7": {
            "audit": {"candidate": True},
            "status": "PREEMPT_REQUESTED",
        },
    }


def _premature_stop_preempt_context(*, decision_index=3):
    payload = _stop_decision_context(
        "PREMATURE_STOP", decision_index=decision_index
    )
    candidate_id = f"session-test:premature_stop:d{decision_index}"
    payload.update(
        {
            "control": "PREEMPT",
            "next_action": "ACTION_COMPLETE_PREEMPTED",
            "failure_verification": {
                "model_backend": "strong_api",
                "need_recovery": True,
                "parse_success": True,
                "parse_error": "",
                "verdict": "FAILURE",
            },
            "phase5": {
                "status": "PREEMPT_REQUESTED",
                "candidate_source": "premature_stop",
                "candidate": {
                    "candidate_id": candidate_id,
                    "source": "premature_stop",
                    "candidate_type": "PREMATURE_STOP",
                    "trigger_type": "PREMATURE_STOP",
                    "decision_index": decision_index,
                },
                "arbiter": {
                    "accepted": True,
                    "selected_candidate_id": candidate_id,
                    "selected_source": "premature_stop",
                },
                "transition": {
                    "confirmed": True,
                    "candidate_id": candidate_id,
                    "phase": "RECOVERY_PLANNING",
                    "preemption_pending": True,
                },
                "preemption": {
                    "accepted": True,
                    "decision_index": decision_index,
                    "stop_phase": "PREMATURE_STOP",
                    "preempt_ack_pending": True,
                },
                "recovery_phase": "RECOVERY_PLANNING",
            },
        }
    )
    return payload


def _motion_kwargs(**overrides):
    values = {
        "decision_index": 0,
        "window_index": 0,
        "action": "NAVIGATE",
        "timestamp_start": 10.0,
        "timestamp_end": 11.0,
        "pose_frame_id": "isaac_world",
        "frame_epoch": 0,
        "pose_start": [1.0, 2.0, 0.1],
        "pose_end": [1.2, 2.0, 0.1],
        "displacement_m": 0.2,
        "local_planner_status": "RUNNING",
        "distance_to_local_goal_start": 2.0,
        "distance_to_local_goal_end": 1.8,
        "map_progress": {
            "resolution_m": 0.05,
            "explored_cells": 1200,
            "new_explored_cells": 32,
            "traversable_cells": 900,
        },
    }
    values.update(overrides)
    return values


def _complete_kwargs(**overrides):
    values = {
        "decision_index": 0,
        "action": "NAVIGATE",
        "status": "COMPLETED",
        "reached_local_goal": True,
        "timestamp": 14.0,
        "pose_frame_id": "isaac_world",
        "frame_epoch": 0,
        "decision_pose": [1.0, 2.0, 0.1],
        "final_pose": [2.0, 2.0, 0.1],
        "displacement_m": 1.0,
        "planner_result": "REACHED",
        "waypoint_id": 0,
    }
    values.update(overrides)
    return values


class G3SessionClientTest(unittest.TestCase):
    def test_derives_service_base_url(self):
        self.assertEqual(
            base_url_from_decision_url(
                "http://127.0.0.1:18765/v1/lavira/decision"
            ),
            "http://127.0.0.1:18765",
        )

    @patch("unified_vln.session_client.urlopen")
    def test_full_phase_one_request_sequence(self, mocked_urlopen):
        mocked_urlopen.side_effect = [
            _Response(_health_payload()),
            _Response(_start_payload()),
            _Response(_control_payload("motion_window")),
            _Response(_control_payload("action_complete")),
            _Response(
                {
                    "schema_version": 2,
                    "response_type": "session_ended",
                    "session_id": "session-test",
                    "status": "ENDED",
                    "reason": "episode_stopped",
                    "final_status": "SUCCESS",
                    "stage_plan_id": PLAN_ID,
                    "stage_plan_status": "READY",
                }
            ),
        ]
        client = G3SessionClient.from_decision_url(
            "http://127.0.0.1:18765/v1/lavira/decision"
        )

        client.health_check()
        started, _ = client.start_session(
            session_id="session-test", instruction="Go to the sofa."
        )
        client.validate_decision_context(
            _navigate_decision_context(), decision_index=0
        )
        motion, _ = client.report_motion_window(**_motion_kwargs())
        complete, _ = client.report_action_complete(**_complete_kwargs())
        ended, _ = client.end_session(status="SUCCESS", reason="episode_stopped")

        self.assertEqual(started.subgoals[-1], "Stop near the sofa.")
        self.assertEqual(motion.control, "CONTINUE")
        self.assertEqual(complete.event_type, "action_complete")
        self.assertEqual(ended.status, "ENDED")
        requests = [call.args[0] for call in mocked_urlopen.call_args_list]
        self.assertEqual(requests[0].full_url, "http://127.0.0.1:18765/health")
        self.assertEqual(
            requests[1].full_url,
            "http://127.0.0.1:18765/v1/lavira/session/start",
        )
        motion_request = json.loads(requests[2].data.decode("utf-8"))
        complete_request = json.loads(requests[3].data.decode("utf-8"))
        self.assertEqual(motion_request["request_type"], "report_execution")
        self.assertEqual(motion_request["event_type"], "motion_window")
        self.assertNotIn("status", motion_request)
        self.assertEqual(motion_request["event_id"], "session-test:d0:w0")
        self.assertEqual(motion_request["pose_start"], [1.0, 2.0, 0.1])
        self.assertEqual(
            motion_request["map_progress"],
            {
                "resolution_m": 0.05,
                "explored_cells": 1200,
                "new_explored_cells": 32,
                "traversable_cells": 900,
            },
        )
        self.assertEqual(complete_request["event_type"], "action_complete")
        self.assertEqual(
            complete_request["event_id"], "session-test:d0:complete"
        )
        self.assertEqual(complete_request["waypoint_id"], 0)
        forbidden = {
            "action_id",
            "path_length_m",
            "duration_s",
            "action_phase",
            "expected_translation",
            "collision_streak",
            "collision",
            "node_key",
        }
        self.assertTrue(forbidden.isdisjoint(motion_request))
        self.assertTrue(forbidden.isdisjoint(complete_request))

    @patch("unified_vln.session_client.urlopen")
    def test_health_rejects_old_map_progress_protocol(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(
            {
                "status": "ok",
                "schema_version": 2,
                "framework": "G3",
                "commit": "b92abb3",
                "audit_interval": 2,
            }
        )
        client = G3SessionClient("http://server")
        with self.assertRaisesRegex(G3SessionProtocolError, "execution_protocol"):
            client.health_check()

    @patch("unified_vln.session_client.urlopen")
    def test_health_rejects_missing_phase_four_protocol(self, mocked_urlopen):
        payload = _health_payload()
        del payload["stage_progress_protocol"]
        mocked_urlopen.return_value = _Response(payload)
        client = G3SessionClient("http://server")
        with self.assertRaisesRegex(G3SessionProtocolError, "stage_progress_protocol"):
            client.health_check()

    @patch("unified_vln.session_client.urlopen")
    def test_accepts_navigate_stage_progress_success_and_failure(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")

        success = client.validate_decision_context(
            _navigate_decision_context(), decision_index=0
        )
        failure = client.validate_decision_context(
            _navigate_decision_context(decision_index=1, parse_success=False),
            decision_index=1,
        )

        self.assertTrue(success.stage_progress.parse_success)
        self.assertIsNone(success.stop_gate)
        self.assertFalse(failure.stage_progress.parse_success)
        self.assertIn("No JSON", failure.stage_progress.parse_error)

    @patch("unified_vln.session_client.urlopen")
    def test_accepts_all_deployed_stop_gate_branches(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")

        expected = {
            "STOP_CONFIRMED": ("STOP_CONFIRMED", "END_SESSION_SUCCESS"),
            "PREMATURE_STOP": ("CONTINUE", "REQUEST_DECISION"),
            "STOP_PENDING": ("CONTINUE", "REQUEST_DECISION"),
        }
        for stop_phase, (control, next_action) in expected.items():
            with self.subTest(stop_phase=stop_phase):
                parsed = client.validate_decision_context(
                    _stop_decision_context(stop_phase), decision_index=1
                )
                self.assertEqual(parsed.stop_phase, stop_phase)
                self.assertEqual(parsed.control, control)
                self.assertEqual(parsed.next_action, next_action)
                self.assertEqual(parsed.stop_gate.stop_phase, stop_phase)

    @patch("unified_vln.session_client.urlopen")
    def test_accepts_uninvoked_strong_stop_gate_pending_metadata(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")

        parsed = client.validate_decision_context(
            _stop_decision_context("STOP_PENDING", decision_index=0),
            decision_index=0,
        )

        self.assertEqual(parsed.stop_phase, "STOP_PENDING")
        self.assertEqual(parsed.control, "CONTINUE")
        self.assertEqual(parsed.next_action, "REQUEST_DECISION")
        self.assertIsNone(parsed.stop_gate.role_call_index)
        self.assertEqual(parsed.stop_gate.image_count, 0)
        self.assertFalse(parsed.stop_gate.parse_success)
        self.assertIn("two frozen-plan", parsed.stop_gate.parse_error)

    @patch("unified_vln.session_client.urlopen")
    def test_rejects_null_role_index_after_stop_model_result(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        payload = _stop_decision_context("STOP_CONFIRMED", decision_index=1)
        payload["stop_gate"]["role_call_index"] = None
        payload["stop_gate"]["image_count"] = 0

        with self.assertRaisesRegex(G3SessionProtocolError, "uninvoked STOP Gate"):
            client.validate_decision_context(payload, decision_index=1)

    @patch("unified_vln.session_client.urlopen")
    def test_allow_accepts_legacy_empty_missing_subgoal(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        payload = _stop_decision_context("STOP_CONFIRMED", decision_index=1)
        payload["stop_gate"]["missing_subgoal"] = ""

        parsed = client.validate_decision_context(payload, decision_index=1)

        self.assertEqual(parsed.stop_phase, "STOP_CONFIRMED")
        self.assertEqual(parsed.stop_gate.missing_subgoal, "")

    @patch("unified_vln.session_client.urlopen")
    def test_allow_rejects_nonempty_missing_subgoal(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        payload = _stop_decision_context("STOP_CONFIRMED", decision_index=1)
        payload["stop_gate"]["missing_subgoal"] = "unfinished stage"

        with self.assertRaisesRegex(G3SessionProtocolError, "ALLOW STOP Gate"):
            client.validate_decision_context(payload, decision_index=1)

    @patch("unified_vln.session_client.urlopen")
    def test_phase_four_image_count_includes_history_images(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")

        navigate = _navigate_decision_context(decision_index=1)
        navigate["stage_progress"]["image_count"] = 6
        parsed_navigate = client.validate_decision_context(
            navigate, decision_index=1
        )
        self.assertEqual(parsed_navigate.stage_progress.image_count, 6)

        stop = _stop_decision_context("STOP_CONFIRMED", decision_index=1)
        stop["stage_progress"]["image_count"] = 6
        stop["stop_gate"]["image_count"] = 6
        parsed_stop = client.validate_decision_context(stop, decision_index=1)
        self.assertEqual(parsed_stop.stop_gate.image_count, 6)

    @patch("unified_vln.session_client.urlopen")
    def test_rejects_stop_control_mismatch(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        payload = _stop_decision_context("STOP_CONFIRMED")
        payload["control"] = "CONTINUE"
        with self.assertRaisesRegex(G3SessionProtocolError, "control/next_action"):
            client.validate_decision_context(payload, decision_index=1)

    @patch("unified_vln.session_client.urlopen")
    def test_stop_cannot_use_phase_three_execution_reports(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        with self.assertRaisesRegex(ValueError, "STOP decisions"):
            client.report_motion_window(**_motion_kwargs(action="STOP"))
        with self.assertRaisesRegex(ValueError, "STOP decisions"):
            client.report_action_complete(**_complete_kwargs(action="STOP"))
        self.assertEqual(mocked_urlopen.call_count, 1)

    @patch("unified_vln.session_client.urlopen")
    def test_end_session_uses_server_final_status_vocabulary(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        with self.assertRaisesRegex(ValueError, "FAILURE"):
            client.end_session(status="FAILED", reason="episode_failed")
        self.assertEqual(mocked_urlopen.call_count, 1)

    @patch("unified_vln.session_client.urlopen")
    def test_http_error_includes_server_error_code(self, mocked_urlopen):
        body = json.dumps(
            {
                "response_type": "error",
                "error_code": "SESSION_ALREADY_ACTIVE",
                "message": "session_id is already active",
            }
        ).encode("utf-8")
        mocked_urlopen.side_effect = HTTPError(
            "http://server/v1/lavira/session/start",
            409,
            "Conflict",
            {},
            io.BytesIO(body),
        )
        client = G3SessionClient("http://server")
        with self.assertRaisesRegex(
            G3SessionProtocolError, "SESSION_ALREADY_ACTIVE"
        ):
            client.start_session(session_id="duplicate", instruction="go")

    @patch("unified_vln.session_client.urlopen")
    def test_rejects_stage_plan_id_change(self, mocked_urlopen):
        mocked_urlopen.side_effect = [
            _Response(_start_payload()),
            _Response(
                {
                    **_control_payload("motion_window"),
                    "stage_plan_id": "sha256:changed",
                }
            ),
        ]
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        with self.assertRaisesRegex(G3SessionProtocolError, "stage_plan_id"):
            client.report_motion_window(**_motion_kwargs())

    @patch("unified_vln.session_client.urlopen")
    def test_motion_retry_uses_identical_event_id_and_payload(self, mocked_urlopen):
        mocked_urlopen.side_effect = [
            _Response(_start_payload()),
            _Response(_control_payload("motion_window")),
            _Response(_control_payload("motion_window")),
        ]
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        client.report_motion_window(**_motion_kwargs())
        client.report_motion_window(**_motion_kwargs())
        requests = [call.args[0] for call in mocked_urlopen.call_args_list]
        first = json.loads(requests[1].data.decode("utf-8"))
        second = json.loads(requests[2].data.decode("utf-8"))
        self.assertEqual(first, second)
        self.assertEqual(first["event_id"], "session-test:d0:w0")

    @patch("unified_vln.session_client.urlopen")
    def test_lost_motion_response_retries_same_window(self, mocked_urlopen):
        mocked_urlopen.side_effect = [
            _Response(_start_payload()),
            URLError("response lost"),
            _Response(_control_payload("motion_window")),
        ]
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        motion, _ = client.report_motion_window(**_motion_kwargs())
        self.assertEqual(motion.control, "CONTINUE")
        requests = [call.args[0] for call in mocked_urlopen.call_args_list]
        first_attempt = json.loads(requests[1].data.decode("utf-8"))
        retry_attempt = json.loads(requests[2].data.decode("utf-8"))
        self.assertEqual(first_attempt, retry_attempt)
        self.assertEqual(retry_attempt["event_id"], "session-test:d0:w0")

    @patch("unified_vln.session_client.urlopen")
    def test_rejects_object_pose_before_http(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        with self.assertRaisesRegex(ValueError, "pose_start"):
            client.report_motion_window(
                **_motion_kwargs(pose_start={"x": 1.0, "y": 2.0, "yaw": 0.1})
            )
        self.assertEqual(mocked_urlopen.call_count, 1)

    @patch("unified_vln.session_client.urlopen")
    def test_rejects_invalid_map_progress_before_http(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        invalid_values = [
            None,
            {
                "resolution_m": 0.05,
                "explored_cells": 10,
                "new_explored_cells": 2,
            },
            {
                "resolution_m": 0.05,
                "explored_cells": 10,
                "new_explored_cells": 2,
                "traversable_cells": 8,
                "available": True,
            },
            {
                "resolution_m": 0.05,
                "explored_cells": 10,
                "new_explored_cells": -1,
                "traversable_cells": 8,
            },
        ]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                client.report_motion_window(
                    **_motion_kwargs(map_progress=value)
                )
        self.assertEqual(mocked_urlopen.call_count, 1)

    @patch("unified_vln.session_client.urlopen")
    def test_phase_five_accepts_verified_motion_preempt(self, mocked_urlopen):
        mocked_urlopen.side_effect = [
            _Response(_start_payload()),
            _Response(_preempt_control_payload()),
        ]
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        control, _ = client.report_motion_window(**_motion_kwargs())

        self.assertEqual(control.control, "PREEMPT")
        self.assertEqual(control.next_action, "ACTION_COMPLETE_PREEMPTED")
        self.assertEqual(control.recovery_phase, "RECOVERY_PLANNING")

    @patch("unified_vln.session_client.urlopen")
    def test_phase_seven_preempt_has_priority_over_navigate(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")

        parsed = client.validate_decision_context(
            _semantic_preempt_context(), decision_index=2
        )

        self.assertEqual(parsed.control, "PREEMPT")
        self.assertEqual(parsed.next_action, "ACTION_COMPLETE_PREEMPTED")
        self.assertFalse(parsed.recovery)
        self.assertIsNone(parsed.stage_progress)
        self.assertEqual(parsed.preempt_source, "semantic_audit")

    @patch("unified_vln.session_client.urlopen")
    def test_p0_premature_stop_preempt_authorizes_one_stop_ack(self, mocked_urlopen):
        preempt_ack = {
            **_control_payload("action_complete", decision_index=3),
            "next_action": "REQUEST_RECOVERY_DECISION",
            "reason": "preemption_acknowledged",
            "phase5": {
                "status": "PREEMPT_ACKNOWLEDGED",
                "recovery_phase": "RECOVERY_PLANNING",
            },
        }
        mocked_urlopen.side_effect = [
            _Response(_start_payload()),
            _Response(preempt_ack),
        ]
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")

        parsed = client.validate_decision_context(
            _premature_stop_preempt_context(), decision_index=3
        )
        control, _ = client.report_action_complete(
            **_complete_kwargs(
                decision_index=3,
                action="STOP",
                waypoint_id=3,
                status="PREEMPTED",
                reached_local_goal=False,
                planner_result="PREEMPTED",
            )
        )

        self.assertEqual(parsed.preempt_source, "premature_stop")
        self.assertEqual(parsed.stop_phase, "PREMATURE_STOP")
        self.assertEqual(parsed.stop_gate.verdict, "PREMATURE")
        self.assertEqual(control.next_action, "REQUEST_RECOVERY_DECISION")
        with self.assertRaisesRegex(ValueError, "verified PREMATURE_STOP"):
            client.report_action_complete(
                **_complete_kwargs(
                    decision_index=3,
                    action="STOP",
                    waypoint_id=3,
                    status="PREEMPTED",
                    reached_local_goal=False,
                    planner_result="PREEMPTED",
                )
            )

    @patch("unified_vln.session_client.urlopen")
    def test_p0_rejects_incomplete_phase5_preempt_metadata(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        payload = _premature_stop_preempt_context()
        payload["phase5"]["arbiter"]["accepted"] = False

        with self.assertRaisesRegex(G3SessionProtocolError, "phase5 metadata"):
            client.validate_decision_context(payload, decision_index=3)

    @patch("unified_vln.session_client.urlopen")
    def test_phase_six_accepts_navigate_and_stable_backtrack(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")

        navigate = client.validate_decision_context(
            _recovery_decision_context("NAVIGATE"), decision_index=1
        )
        backtrack = client.validate_decision_context(
            _recovery_decision_context("BACKTRACK"), decision_index=1
        )

        self.assertTrue(navigate.recovery)
        self.assertEqual(navigate.action_source, "RECOVERY")
        self.assertEqual(navigate.next_action, "EXECUTE_RECOVERY")
        self.assertTrue(backtrack.recovery)

    @patch("unified_vln.session_client.urlopen")
    def test_phase_six_accepts_terminal_safe_stop_decision(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")

        parsed = client.validate_decision_context(
            _safe_stop_decision_context(), decision_index=2
        )

        self.assertEqual(parsed.control, "SAFE_STOP")
        self.assertEqual(parsed.next_action, "SAFE_STOP")
        self.assertEqual(parsed.stop_phase, "SAFE_STOP")
        self.assertEqual(parsed.action_source, "RECOVERY")
        self.assertTrue(parsed.recovery)

        invalid = _safe_stop_decision_context()
        invalid["direction"] = "forward"
        with self.assertRaisesRegex(
            G3SessionProtocolError, "visual navigation fields"
        ):
            client.validate_decision_context(invalid, decision_index=2)

    @patch("unified_vln.session_client.urlopen")
    def test_phase_six_accepts_safe_stop_after_action_complete(self, mocked_urlopen):
        safe_stop = {
            **_control_payload("action_complete", decision_index=3),
            "control": "SAFE_STOP",
            "next_action": "SAFE_STOP",
            "reason": "recovery_escape_not_proven",
            "phase6": {
                "status": "ESCAPE_EVALUATED",
                "recovery_phase": "SAFE_STOP",
            },
        }
        mocked_urlopen.side_effect = [
            _Response(_start_payload()),
            _Response(safe_stop),
        ]
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")

        control, _ = client.report_action_complete(
            **_complete_kwargs(decision_index=3, waypoint_id=3)
        )

        self.assertEqual(control.control, "SAFE_STOP")
        self.assertEqual(control.next_action, "SAFE_STOP")
        self.assertEqual(control.recovery_phase, "SAFE_STOP")

    @patch("unified_vln.session_client.urlopen")
    def test_phase_six_recovery_motion_window_continues_execution(
        self, mocked_urlopen
    ):
        recovery_window = {
            **_control_payload("motion_window", decision_index=3),
            "reason": "recovery_execution_window_recorded",
            "next_action": "CONTINUE_RECOVERY_EXECUTION",
            "phase6": {
                "status": "RECOVERY_EXECUTING",
                "recovery_phase": "RECOVERY_EXECUTING",
            },
        }
        mocked_urlopen.side_effect = [
            _Response(_start_payload()),
            _Response(recovery_window),
        ]
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")

        control, _ = client.report_motion_window(
            **_motion_kwargs(decision_index=3)
        )

        self.assertEqual(control.control, "CONTINUE")
        self.assertEqual(control.next_action, "CONTINUE_RECOVERY_EXECUTION")
        self.assertEqual(control.recovery_phase, "RECOVERY_EXECUTING")

    @patch("unified_vln.session_client.urlopen")
    def test_phase_six_recovery_continue_is_strictly_scoped_to_motion_window(
        self, mocked_urlopen
    ):
        invalid = {
            **_control_payload("action_complete", decision_index=3),
            "reason": "recovery_execution_window_recorded",
            "next_action": "CONTINUE_RECOVERY_EXECUTION",
            "phase6": {
                "status": "RECOVERY_EXECUTING",
                "recovery_phase": "RECOVERY_EXECUTING",
            },
        }
        mocked_urlopen.side_effect = [
            _Response(_start_payload()),
            _Response(invalid),
        ]
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")

        with self.assertRaisesRegex(
            G3SessionProtocolError, "Recovery motion window"
        ):
            client.report_action_complete(
                **_complete_kwargs(decision_index=3, waypoint_id=3)
            )

    @patch("unified_vln.session_client.urlopen")
    def test_preempt_ack_and_escape_transitions_use_frozen_next_actions(
        self, mocked_urlopen
    ):
        preempt_ack = {
            **_control_payload("action_complete", decision_index=2),
            "next_action": "REQUEST_RECOVERY_DECISION",
            "reason": "preemption_acknowledged",
            "phase5": {
                "status": "PREEMPT_ACKNOWLEDGED",
                "recovery_phase": "RECOVERY_PLANNING",
            },
        }
        handback = {
            **_control_payload("action_complete", decision_index=3),
            "next_action": "REQUEST_DECISION",
            "reason": "recovery_escape_confirmed",
            "phase6": {
                "status": "ESCAPE_EVALUATED",
                "recovery_phase": "NAVIGATOR",
            },
        }
        mocked_urlopen.side_effect = [
            _Response(_start_payload()),
            _Response(preempt_ack),
            _Response(handback),
        ]
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")

        acknowledged, _ = client.report_action_complete(
            **_complete_kwargs(
                decision_index=2,
                waypoint_id=2,
                status="PREEMPTED",
                reached_local_goal=False,
                planner_result="PREEMPTED",
            )
        )
        escaped, _ = client.report_action_complete(
            **_complete_kwargs(decision_index=3, waypoint_id=3)
        )

        self.assertEqual(
            acknowledged.next_action, "REQUEST_RECOVERY_DECISION"
        )
        self.assertEqual(acknowledged.recovery_phase, "RECOVERY_PLANNING")
        self.assertEqual(escaped.next_action, "REQUEST_DECISION")
        self.assertEqual(escaped.recovery_phase, "NAVIGATOR")

    @patch("unified_vln.session_client.urlopen")
    def test_rejects_decision_from_changed_stage_plan(self, mocked_urlopen):
        mocked_urlopen.return_value = _Response(_start_payload())
        client = G3SessionClient("http://server")
        client.start_session(session_id="session-test", instruction="go")
        with self.assertRaisesRegex(G3SessionProtocolError, "stage_plan_id"):
            client.validate_decision_context(
                {
                    **_navigate_decision_context(),
                    "stage_plan_id": "sha256:changed",
                },
                decision_index=0,
            )


if __name__ == "__main__":
    unittest.main()
