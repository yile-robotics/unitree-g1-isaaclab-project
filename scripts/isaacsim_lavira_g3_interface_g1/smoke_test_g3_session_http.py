#!/usr/bin/env python3
from __future__ import annotations

"""Run the phase-three G3 lifecycle with an existing offline four-image case."""

import argparse
import json
from pathlib import Path
import time

from unified_vln.model_client import CombinedModelClient
from unified_vln.model_contract import NavigationDecisionRequest
from unified_vln.session_client import G3SessionClient


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_CASE = (
    PROJECT_DIR
    / "outputs"
    / "isaacsim_goal_tracking"
    / "lavira_offline"
    / "run_20260721_120446_286340"
    / "robot_01_open_room_test_001"
    / "robot_01_open_room_test_001_decision_000"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise start, decision, reports, and end against G3/b92."
    )
    parser.add_argument(
        "--decision-url",
        default="http://127.0.0.1:18765/v1/lavira/decision",
    )
    parser.add_argument("--test-case", type=Path, default=DEFAULT_CASE)
    parser.add_argument(
        "--session-id",
        default=None,
        help="Must be unique while a server session is active.",
    )
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata_path = args.test_case / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    session_id = args.session_id or time.strftime("g3_http_smoke_%Y%m%d_%H%M%S")
    metadata["session_id"] = session_id
    metadata["observation_id"] = f"{session_id}_decision_000"
    metadata["decision_index"] = 0
    request = NavigationDecisionRequest.from_metadata(metadata)
    images = {
        field_name: (args.test_case / f"{field_name}.png").read_bytes()
        for field_name in request.required_image_fields
    }

    session = G3SessionClient.from_decision_url(
        args.decision_url, args.timeout_s
    )
    model = CombinedModelClient(
        args.decision_url,
        args.timeout_s,
        send_instruction=False,
    )
    active = False
    final_status = "FAILURE"
    final_reason = "g3_http_smoke_failed"
    summary = {}
    try:
        health = session.health_check()
        started, _ = session.start_session(
            session_id=session_id,
            instruction=request.instruction,
        )
        active = True
        decision, raw_decision = model.decide(request, images)
        session.validate_decision_context(
            raw_decision,
            decision_index=request.decision_index,
        )
        if decision.action.upper() != "NAVIGATE":
            raise RuntimeError(
                "Execution-report smoke test requires a NAVIGATE decision."
            )
        motion, _ = session.report_motion_window(
            decision_index=0,
            window_index=0,
            action=decision.action,
            timestamp_start=0.0,
            timestamp_end=1.0,
            pose_frame_id="smoke_local",
            frame_epoch=0,
            pose_start=[0.0, 0.0, 0.0],
            pose_end=[0.2, 0.0, 0.0],
            displacement_m=0.2,
            local_planner_status="RUNNING",
            distance_to_local_goal_start=2.0,
            distance_to_local_goal_end=1.8,
            map_progress={
                "resolution_m": 0.05,
                "explored_cells": 1200,
                "new_explored_cells": 32,
                "traversable_cells": 900,
            },
        )
        complete, _ = session.report_action_complete(
            decision_index=0,
            action=decision.action,
            status="COMPLETED",
            reached_local_goal=True,
            timestamp=2.0,
            pose_frame_id="smoke_local",
            frame_epoch=0,
            decision_pose=[0.0, 0.0, 0.0],
            final_pose=[1.0, 0.0, 0.0],
            displacement_m=1.0,
            planner_result="REACHED",
            waypoint_id=0,
        )
        final_status = "SUCCESS"
        final_reason = "g3_http_smoke_completed"
        summary = {
            "health": health["status"],
            "session_id": session_id,
            "stage_plan_id": started.stage_plan_id,
            "stage_total": started.stage_total,
            "action": decision.action,
            "direction": decision.direction,
            "target": decision.target,
            "motion_control": motion.control,
            "complete_control": complete.control,
        }
    finally:
        if active:
            ended, _ = session.end_session(
                status=final_status,
                reason=final_reason,
            )
            summary["end"] = ended.status
            summary["final_status"] = ended.final_status
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
