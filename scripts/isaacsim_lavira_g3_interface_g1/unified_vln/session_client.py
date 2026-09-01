from __future__ import annotations

"""HTTP client and response validation for the deployed LaViRA G3 adapter.

The model decision endpoint remains multipart and is handled by
``CombinedModelClient``.  This module validates its phase-four supervision and
owns the JSON lifecycle endpoints: health, start, execution reports, and end.
"""

from dataclasses import dataclass
import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


G3_SCHEMA_VERSION = 2
G3_FRAMEWORK = "G3"
G3_BASELINE_COMMIT = "b92abb3"
G3_AUDIT_INTERVAL = 2
G3_EXECUTION_PROTOCOL = "phase3_map_progress_v1"
G3_STAGE_PROGRESS_PROTOCOL = "phase4_stage_progress_v2"
G3_STOP_GATE_PROTOCOL = "phase4_stop_gate_v1"
G3_RECOVERY_CONTROL_PROTOCOL = "phase6_recovery_control_v1"
G3_MAX_REQUEST_IMAGES = 16
G3_CONTROLS = frozenset({"CONTINUE", "PREEMPT", "SAFE_STOP"})
G3_EVENT_TYPES = frozenset({"motion_window", "action_complete"})
G3_ACTION_STATUSES = frozenset({"COMPLETED", "FAILED", "PREEMPTED"})
G3_LOCAL_PLANNER_STATUSES = frozenset({"RUNNING", "FAILED", "PREEMPTED"})
G3_PLANNER_RESULTS = frozenset(
    {"REACHED", "TIMEOUT", "PLANNING_FAILED", "EXECUTION_FAILED", "PREEMPTED"}
)
G3_SESSION_FINAL_STATUSES = frozenset(
    {"SUCCESS", "FAILURE", "TIMEOUT", "CANCELLED"}
)


class G3SessionProtocolError(RuntimeError):
    """A transport error or a response that violates the G3 session contract."""


class G3TransportError(G3SessionProtocolError):
    """A retryable failure before an execution response was received."""


def _response_non_negative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise G3SessionProtocolError(
            f"G3 response field {field!r} must be a non-negative integer."
        )
    return int(value)


def _response_string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise G3SessionProtocolError(
            f"G3 response field {field!r} must be a string list."
        )
    return tuple(value)


def _response_optional_seed(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _response_non_negative_integer(value, field)


def base_url_from_decision_url(decision_url: str) -> str:
    """Return the service root for a configured ``/v1/lavira/decision`` URL."""

    value = str(decision_url).strip().rstrip("/")
    if not value:
        raise ValueError("LaViRA decision URL must not be empty.")
    suffix = "/v1/lavira/decision"
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    if not value:
        raise ValueError("LaViRA service base URL could not be derived.")
    return value


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise G3SessionProtocolError(f"G3 response field {field!r} must be non-empty.")
    return value


def _finite_number(value: Any, field: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    if non_negative and result < 0.0:
        raise ValueError(f"{field} must be non-negative.")
    return result


def _pose_array(value: Any, field: str) -> list[float]:
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(f"{field} must be [x, y, yaw].") from exc
    if len(items) != 3:
        raise ValueError(f"{field} must contain exactly [x, y, yaw].")
    return [
        _finite_number(item, f"{field}[{index}]")
        for index, item in enumerate(items)
    ]


def _map_progress_object(value: Any) -> dict[str, float | int]:
    """Validate the frozen phase-three map-progress wire object."""

    if not isinstance(value, dict):
        raise ValueError("map_progress must be a four-field object.")
    expected = {
        "resolution_m",
        "explored_cells",
        "new_explored_cells",
        "traversable_cells",
    }
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "map_progress must contain exactly resolution_m, explored_cells, "
            "new_explored_cells, and traversable_cells; "
            f"missing={missing}, extra={extra}."
        )
    resolution_m = _finite_number(value["resolution_m"], "map_progress.resolution_m")
    if resolution_m <= 0.0:
        raise ValueError("map_progress.resolution_m must be positive.")
    result: dict[str, float | int] = {"resolution_m": resolution_m}
    for field in ("explored_cells", "new_explored_cells", "traversable_cells"):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"map_progress.{field} must be a non-negative integer.")
        result[field] = int(item)
    return result


@dataclass(frozen=True)
class G3SessionStarted:
    session_id: str
    status: str
    stage_plan_id: str
    stage_plan_status: str
    stage_total: int
    subgoals: tuple[str, ...]
    next_action: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any], session_id: str) -> "G3SessionStarted":
        if payload.get("schema_version") != G3_SCHEMA_VERSION:
            raise G3SessionProtocolError("Session-start response schema_version is not 2.")
        if payload.get("response_type") != "session_started":
            raise G3SessionProtocolError("Expected response_type=session_started.")
        if payload.get("session_id") != session_id:
            raise G3SessionProtocolError("Session-start response session_id mismatch.")
        if payload.get("status") != "ACTIVE":
            raise G3SessionProtocolError("Started G3 session is not ACTIVE.")
        if payload.get("stage_plan_status") != "READY":
            raise G3SessionProtocolError("Frozen Stage Plan is not READY.")
        plan = payload.get("frozen_stage_plan")
        if not isinstance(plan, dict):
            raise G3SessionProtocolError("frozen_stage_plan must be an object.")
        plan_id = _required_string(payload, "stage_plan_id")
        if plan.get("stage_plan_id") != plan_id:
            raise G3SessionProtocolError("Frozen Stage Plan id mismatch.")
        if plan.get("stage_plan_status") != "READY":
            raise G3SessionProtocolError("Nested Frozen Stage Plan is not READY.")
        stage_total = plan.get("stage_total")
        subgoals = plan.get("subgoals")
        if not isinstance(stage_total, int) or isinstance(stage_total, bool) or stage_total <= 0:
            raise G3SessionProtocolError("stage_total must be a positive integer.")
        if (
            not isinstance(subgoals, list)
            or len(subgoals) != stage_total
            or any(not isinstance(item, str) or not item.strip() for item in subgoals)
        ):
            raise G3SessionProtocolError("Frozen Stage Plan subgoals are invalid.")
        return cls(
            session_id=session_id,
            status="ACTIVE",
            stage_plan_id=plan_id,
            stage_plan_status="READY",
            stage_total=stage_total,
            subgoals=tuple(subgoals),
            next_action=_required_string(payload, "next_action"),
        )


@dataclass(frozen=True)
class G3StageProgress:
    stage_plan_id: str | None
    stage_total: int | None
    stage_completed: int | None
    stage_id: int | None
    completed_stage_ids: tuple[int, ...]
    current_stage: str | None
    evidence_of_completion: tuple[str, ...]
    final_target_visible: bool | None
    parse_success: bool
    parse_error: str | None
    decision_index: int
    role_call_index: int
    role_seed: int | None
    image_count: int
    model_backend: str

    @classmethod
    def from_dict(
        cls,
        payload: Any,
        *,
        started: G3SessionStarted,
        decision_index: int,
    ) -> "G3StageProgress":
        if not isinstance(payload, dict):
            raise G3SessionProtocolError("stage_progress must be an object.")
        parse_success = payload.get("parse_success")
        if not isinstance(parse_success, bool):
            raise G3SessionProtocolError("stage_progress.parse_success must be boolean.")
        completed_value = payload.get("completed_stage_ids")
        if not isinstance(completed_value, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in completed_value
        ):
            raise G3SessionProtocolError(
                "stage_progress.completed_stage_ids must be a non-negative integer list."
            )
        completed_ids = tuple(int(item) for item in completed_value)
        evidence = _response_string_list(
            payload.get("evidence_of_completion"),
            "stage_progress.evidence_of_completion",
        )
        response_decision_index = _response_non_negative_integer(
            payload.get("decision_index"), "stage_progress.decision_index"
        )
        if response_decision_index != decision_index:
            raise G3SessionProtocolError("Stage Progress decision_index mismatch.")
        role_call_index = _response_non_negative_integer(
            payload.get("role_call_index"), "stage_progress.role_call_index"
        )
        role_seed = _response_optional_seed(
            payload.get("role_seed"), "stage_progress.role_seed"
        )
        image_count = _response_non_negative_integer(
            payload.get("image_count"), "stage_progress.image_count"
        )
        if not 4 <= image_count <= G3_MAX_REQUEST_IMAGES:
            raise G3SessionProtocolError(
                "Stage Progress image_count must be between 4 and 16."
            )
        model_backend = _required_string(payload, "model_backend")

        if parse_success:
            stage_plan_id = payload.get("stage_plan_id")
            if stage_plan_id != started.stage_plan_id:
                raise G3SessionProtocolError("Stage Progress stage_plan_id mismatch.")
            stage_total = _response_non_negative_integer(
                payload.get("stage_total"), "stage_progress.stage_total"
            )
            if stage_total != started.stage_total:
                raise G3SessionProtocolError("Stage Progress stage_total mismatch.")
            stage_completed = _response_non_negative_integer(
                payload.get("stage_completed"), "stage_progress.stage_completed"
            )
            stage_id = _response_non_negative_integer(
                payload.get("stage_id"), "stage_progress.stage_id"
            )
            if stage_completed > stage_total or stage_id > stage_total:
                raise G3SessionProtocolError("Stage Progress stage index exceeds stage_total.")
            current_stage = _required_string(payload, "current_stage")
            final_target_visible = payload.get("final_target_visible")
            if not isinstance(final_target_visible, bool):
                raise G3SessionProtocolError(
                    "stage_progress.final_target_visible must be boolean."
                )
            if payload.get("parse_error") is not None:
                raise G3SessionProtocolError(
                    "Successful Stage Progress parse_error must be null."
                )
            parse_error = None
        else:
            null_fields = (
                "stage_plan_id",
                "stage_total",
                "stage_completed",
                "stage_id",
                "current_stage",
                "final_target_visible",
            )
            if any(payload.get(field) is not None for field in null_fields):
                raise G3SessionProtocolError(
                    "Failed Stage Progress must use null stage fields."
                )
            if completed_ids or evidence:
                raise G3SessionProtocolError(
                    "Failed Stage Progress must use empty completion lists."
                )
            stage_plan_id = None
            stage_total = None
            stage_completed = None
            stage_id = None
            current_stage = None
            final_target_visible = None
            parse_error = _required_string(payload, "parse_error")

        return cls(
            stage_plan_id=stage_plan_id,
            stage_total=stage_total,
            stage_completed=stage_completed,
            stage_id=stage_id,
            completed_stage_ids=completed_ids,
            current_stage=current_stage,
            evidence_of_completion=evidence,
            final_target_visible=final_target_visible,
            parse_success=parse_success,
            parse_error=parse_error,
            decision_index=response_decision_index,
            role_call_index=role_call_index,
            role_seed=role_seed,
            image_count=image_count,
            model_backend=model_backend,
        )


@dataclass(frozen=True)
class G3StopGate:
    verdict: str
    completed: bool | None
    missing_subgoal: str | None
    evidence: tuple[str, ...]
    parse_success: bool
    parse_error: str | None
    decision_index: int
    role_call_index: int | None
    role_seed: int | None
    image_count: int
    model_backend: str
    stop_phase: str

    @classmethod
    def from_dict(cls, payload: Any, *, decision_index: int) -> "G3StopGate":
        if not isinstance(payload, dict):
            raise G3SessionProtocolError("stop_gate must be an object for STOP.")
        verdict = _required_string(payload, "verdict").upper()
        if verdict not in {"ALLOW", "PREMATURE", "UNCERTAIN"}:
            raise G3SessionProtocolError(f"Unsupported STOP Gate verdict {verdict!r}.")
        parse_success = payload.get("parse_success")
        if not isinstance(parse_success, bool):
            raise G3SessionProtocolError("stop_gate.parse_success must be boolean.")
        evidence = _response_string_list(payload.get("evidence"), "stop_gate.evidence")
        response_decision_index = _response_non_negative_integer(
            payload.get("decision_index"), "stop_gate.decision_index"
        )
        if response_decision_index != decision_index:
            raise G3SessionProtocolError("STOP Gate decision_index mismatch.")
        role_call_index_value = payload.get("role_call_index")
        role_call_index = (
            None
            if role_call_index_value is None
            else _response_non_negative_integer(
                role_call_index_value, "stop_gate.role_call_index"
            )
        )
        role_seed = _response_optional_seed(payload.get("role_seed"), "stop_gate.role_seed")
        image_count = _response_non_negative_integer(
            payload.get("image_count"), "stop_gate.image_count"
        )
        # The first STOP proposal is held as STOP_PENDING until two frozen-plan
        # Stage Progress observations exist.  In that state the server has not
        # called the STOP model yet, so its role index is null and image count
        # is zero.  Once a role call exists, retain the deployed 4..16 image
        # validation used by all actual STOP-model invocations.
        if role_call_index is None:
            if verdict != "UNCERTAIN" or parse_success or image_count != 0:
                raise G3SessionProtocolError(
                    "An uninvoked STOP Gate must be a failed UNCERTAIN result "
                    "with role_call_index=null and image_count=0."
                )
        elif not 4 <= image_count <= G3_MAX_REQUEST_IMAGES:
            raise G3SessionProtocolError(
                "STOP Gate image_count must be between 4 and 16."
            )
        model_backend = _required_string(payload, "model_backend")
        stop_phase = _required_string(payload, "stop_phase").upper()
        completed = payload.get("completed")
        missing_subgoal = payload.get("missing_subgoal")
        parse_error_value = payload.get("parse_error")

        if verdict == "ALLOW":
            expected_phase = "STOP_CONFIRMED"
            # The deployed strong backend uses JSON null when ALLOW has no
            # missing subgoal.  Keep accepting the former empty-string wire
            # representation so a backend restart cannot break an active
            # schema-v2 client, but reject any actual missing-subgoal text.
            if (
                not parse_success
                or completed is not True
                or missing_subgoal not in {None, ""}
            ):
                raise G3SessionProtocolError("ALLOW STOP Gate fields are inconsistent.")
            if parse_error_value is not None:
                raise G3SessionProtocolError("Successful STOP Gate parse_error must be null.")
            parse_error = None
        elif verdict == "PREMATURE":
            expected_phase = "PREMATURE_STOP"
            if not parse_success or completed is not False:
                raise G3SessionProtocolError("PREMATURE STOP Gate fields are inconsistent.")
            if not isinstance(missing_subgoal, str) or not missing_subgoal.strip():
                raise G3SessionProtocolError(
                    "PREMATURE stop_gate.missing_subgoal must be non-empty."
                )
            if parse_error_value is not None:
                raise G3SessionProtocolError("Successful STOP Gate parse_error must be null.")
            parse_error = None
        else:
            expected_phase = "STOP_PENDING"
            if completed is not None:
                raise G3SessionProtocolError("UNCERTAIN STOP Gate fields are inconsistent.")
            if missing_subgoal is not None and not isinstance(missing_subgoal, str):
                raise G3SessionProtocolError(
                    "UNCERTAIN stop_gate.missing_subgoal must be string or null."
                )
            if parse_success:
                if parse_error_value is not None:
                    raise G3SessionProtocolError(
                        "Successful STOP Gate parse_error must be null."
                    )
                parse_error = None
            else:
                if evidence:
                    raise G3SessionProtocolError(
                        "Failed UNCERTAIN STOP Gate evidence must be empty."
                    )
                parse_error = _required_string(payload, "parse_error")

        if stop_phase != expected_phase:
            raise G3SessionProtocolError("STOP Gate verdict/stop_phase mismatch.")
        return cls(
            verdict=verdict,
            completed=completed,
            missing_subgoal=missing_subgoal,
            evidence=evidence,
            parse_success=parse_success,
            parse_error=parse_error,
            decision_index=response_decision_index,
            role_call_index=role_call_index,
            role_seed=role_seed,
            image_count=image_count,
            model_backend=model_backend,
            stop_phase=stop_phase,
        )


@dataclass(frozen=True)
class G3DecisionSupervision:
    stage_progress: G3StageProgress | None
    stop_gate: G3StopGate | None
    stop_phase: str | None
    control: str | None
    next_action: str | None
    action_source: str
    recovery: bool
    preempt_source: str | None = None


@dataclass(frozen=True)
class G3ExecutionControl:
    session_id: str
    decision_index: int
    control: str
    reason: str
    event_type: str
    stage_plan_id: str
    stage_plan_status: str
    next_action: str | None
    recovery_phase: str | None

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        session_id: str,
        decision_index: int,
        event_type: str,
        stage_plan_id: str,
    ) -> "G3ExecutionControl":
        if payload.get("schema_version") != G3_SCHEMA_VERSION:
            raise G3SessionProtocolError("Execution response schema_version is not 2.")
        if payload.get("response_type") != "execution_control":
            raise G3SessionProtocolError("Expected response_type=execution_control.")
        if payload.get("session_id") != session_id:
            raise G3SessionProtocolError("Execution response session_id mismatch.")
        if payload.get("decision_index") != decision_index:
            raise G3SessionProtocolError("Execution response decision_index mismatch.")
        if payload.get("event_type") != event_type:
            raise G3SessionProtocolError("Execution response event_type mismatch.")
        control = _required_string(payload, "control").upper()
        if control not in G3_CONTROLS:
            raise G3SessionProtocolError(f"Unsupported execution control {control!r}.")
        response_plan_id = payload.get("stage_plan_id")
        if response_plan_id is not None and response_plan_id != stage_plan_id:
            raise G3SessionProtocolError("Execution response stage_plan_id mismatch.")
        if payload.get("stage_plan_status") != "READY":
            raise G3SessionProtocolError("Execution response Stage Plan is not READY.")
        reason = payload.get("reason", "")
        if not isinstance(reason, str):
            raise G3SessionProtocolError("Execution response reason must be a string.")
        next_action_value = payload.get("next_action")
        next_action = (
            None
            if next_action_value is None
            else _required_string(payload, "next_action").upper()
        )
        allowed_transitions = {
            "CONTINUE": {
                None,
                "REQUEST_RECOVERY_DECISION",
                "REQUEST_DECISION",
                "CONTINUE_RECOVERY_EXECUTION",
            },
            "PREEMPT": {"ACTION_COMPLETE_PREEMPTED"},
            "SAFE_STOP": {"SAFE_STOP"},
        }
        if next_action not in allowed_transitions[control]:
            raise G3SessionProtocolError(
                f"Execution control {control!r} is inconsistent with "
                f"next_action={next_action!r}."
            )
        if control == "PREEMPT":
            verification = payload.get("failure_verification")
            if (
                not isinstance(verification, dict)
                or verification.get("verdict") != "FAILURE"
                or verification.get("need_recovery") is not True
                or verification.get("parse_success") is not True
            ):
                raise G3SessionProtocolError(
                    "PREEMPT requires a successful FAILURE verification."
                )
        if next_action == "CONTINUE_RECOVERY_EXECUTION":
            phase6 = payload.get("phase6")
            if (
                control != "CONTINUE"
                or event_type != "motion_window"
                or reason != "recovery_execution_window_recorded"
                or not isinstance(phase6, dict)
                or phase6.get("status") != "RECOVERY_EXECUTING"
                or phase6.get("recovery_phase") != "RECOVERY_EXECUTING"
            ):
                raise G3SessionProtocolError(
                    "CONTINUE_RECOVERY_EXECUTION requires a recorded Recovery "
                    "motion window in phase6=RECOVERY_EXECUTING."
                )
        recovery_phase = None
        for phase_field in ("phase6", "phase5"):
            phase = payload.get(phase_field)
            if isinstance(phase, dict) and isinstance(
                phase.get("recovery_phase"), str
            ):
                recovery_phase = phase["recovery_phase"].upper()
                break
        return cls(
            session_id=session_id,
            decision_index=decision_index,
            control=control,
            reason=reason,
            event_type=event_type,
            stage_plan_id=stage_plan_id,
            stage_plan_status="READY",
            next_action=next_action,
            recovery_phase=recovery_phase,
        )


@dataclass(frozen=True)
class G3SessionEnded:
    session_id: str
    status: str
    reason: str
    final_status: str
    stage_plan_id: str
    stage_plan_status: str

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        session_id: str,
        stage_plan_id: str,
    ) -> "G3SessionEnded":
        if payload.get("schema_version") != G3_SCHEMA_VERSION:
            raise G3SessionProtocolError("Session-end response schema_version is not 2.")
        if payload.get("response_type") != "session_ended":
            raise G3SessionProtocolError("Expected response_type=session_ended.")
        if payload.get("session_id") != session_id:
            raise G3SessionProtocolError("Session-end response session_id mismatch.")
        if payload.get("status") != "ENDED":
            raise G3SessionProtocolError("Ended G3 session did not return ENDED.")
        if payload.get("stage_plan_id") != stage_plan_id:
            raise G3SessionProtocolError("Session-end stage_plan_id mismatch.")
        if payload.get("stage_plan_status") != "READY":
            raise G3SessionProtocolError("Session-end Stage Plan is not READY.")
        return cls(
            session_id=session_id,
            status="ENDED",
            reason=str(payload.get("reason", "")),
            final_status=_required_string(payload, "final_status"),
            stage_plan_id=stage_plan_id,
            stage_plan_status="READY",
        )


class G3SessionClient:
    """Strict client for the deployed b92 phase-three/four G3 adapter."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 180.0,
        *,
        expected_commit: str = G3_BASELINE_COMMIT,
    ):
        base_url = str(base_url).strip().rstrip("/")
        if not base_url:
            raise ValueError("G3 service base URL must not be empty.")
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("G3 session timeout must be finite and positive.")
        if not expected_commit.strip():
            raise ValueError("Expected G3 commit must not be empty.")
        self.base_url = base_url
        self.timeout_s = float(timeout_s)
        self.expected_commit = expected_commit
        self.started: G3SessionStarted | None = None
        self.ended = False
        # A Navigator STOP may report action_complete only after the deployed
        # P0 chain has confirmed PREMATURE_STOP and requested atomic PREEMPT.
        # Keep this authorization scoped to the exact decision index.
        self._premature_stop_preempt_decisions: set[int] = set()

    @classmethod
    def from_decision_url(
        cls,
        decision_url: str,
        timeout_s: float = 180.0,
        *,
        expected_commit: str = G3_BASELINE_COMMIT,
    ) -> "G3SessionClient":
        return cls(
            base_url_from_decision_url(decision_url),
            timeout_s,
            expected_commit=expected_commit,
        )

    def health_check(self) -> dict[str, Any]:
        payload = self._json_request("GET", "/health")
        expected = {
            "status": "ok",
            "schema_version": G3_SCHEMA_VERSION,
            "framework": G3_FRAMEWORK,
            "commit": self.expected_commit,
            "audit_interval": G3_AUDIT_INTERVAL,
            "execution_protocol": G3_EXECUTION_PROTOCOL,
            "map_progress_required": True,
            "stage_progress_protocol": G3_STAGE_PROGRESS_PROTOCOL,
            "stop_gate_protocol": G3_STOP_GATE_PROTOCOL,
            "recovery_control_protocol": G3_RECOVERY_CONTROL_PROTOCOL,
        }
        mismatches = {
            key: (value, payload.get(key))
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise G3SessionProtocolError(f"G3 health mismatch: {mismatches}.")
        phase4 = payload.get("phase4")
        if not isinstance(phase4, dict) or phase4.get("status") != "wired":
            raise G3SessionProtocolError("G3 phase4 is not wired.")
        stage_progress = phase4.get("stage_progress")
        expected_stage_progress = {
            "status": "wired",
            "frequency": "once_per_decision_after_navigator",
            "model_role": "stage_progress_shadow",
            "seed_namespace_offset": 1000,
        }
        if not isinstance(stage_progress, dict) or any(
            stage_progress.get(key) != value
            for key, value in expected_stage_progress.items()
        ):
            raise G3SessionProtocolError("G3 Stage Progress health fields mismatch.")
        stop_gate = phase4.get("stop_gate")
        expected_stop_gate = {
            "status": "wired",
            "trigger": "navigator_action_STOP",
            "frequency": "once_per_stop_proposal",
            "model_role": "stop_evaluator",
            "observation": "same_decision_four_direction_views",
        }
        if not isinstance(stop_gate, dict) or any(
            stop_gate.get(key) != value for key, value in expected_stop_gate.items()
        ):
            raise G3SessionProtocolError("G3 STOP Gate health fields mismatch.")
        _required_string(stop_gate, "model_backend")
        for disabled in (
            "physical_monitor",
            "preempt",
            "failure_verifier",
            "recovery",
            "backtrack",
        ):
            if phase4.get(disabled) is not False:
                raise G3SessionProtocolError(
                    f"G3 phase4 field {disabled!r} must be false."
                )
        phase5 = payload.get("phase5")
        if (
            not isinstance(phase5, dict)
            or phase5.get("status") != "wired"
            or phase5.get("physical_monitor") is not True
            or phase5.get("failure_verifier") is not True
            or phase5.get("preempt") is not True
            or phase5.get("model_backend") != "strong_api"
            or phase5.get("node_size_m") != 0.75
            or phase5.get("fmm_available") is not False
        ):
            raise G3SessionProtocolError("G3 phase5 is not the frozen online monitor.")
        phase6 = payload.get("phase6")
        if (
            not isinstance(phase6, dict)
            or phase6.get("status") != "wired"
            or phase6.get("recovery") is not True
            or phase6.get("backtrack") is not True
            or phase6.get("escape_evaluator") is not True
            or phase6.get("model_backend") != "strong_api"
            or phase6.get("waypoint_source") != "session_waypoint_registry"
            or phase6.get("escape_confirmation_policy") != "single_valid_positive"
            or phase6.get("control_protocol") != G3_RECOVERY_CONTROL_PROTOCOL
            or phase6.get("safe_stop_control_precedence") is not True
            or phase6.get("ordinary_navigator_stop_schema_unchanged") is not True
        ):
            raise G3SessionProtocolError("G3 phase6 recovery contract mismatch.")
        phase7 = payload.get("phase7")
        if (
            not isinstance(phase7, dict)
            or phase7.get("status") != "wired"
            or phase7.get("semantic_audit") is not True
            or phase7.get("audit_interval_decisions") != G3_AUDIT_INTERVAL
            or phase7.get("model_backend") != "strong_api"
        ):
            raise G3SessionProtocolError("G3 phase7 semantic audit contract mismatch.")
        return payload

    def start_session(
        self,
        *,
        session_id: str,
        instruction: str,
    ) -> tuple[G3SessionStarted, dict[str, Any]]:
        if self.started is not None and not self.ended:
            raise G3SessionProtocolError("This client already owns an active session.")
        if not session_id.strip() or not instruction.strip():
            raise ValueError("Session id and instruction must not be empty.")
        request_payload = {
            "schema_version": G3_SCHEMA_VERSION,
            "request_type": "start_session",
            "session_id": session_id,
            "instruction": instruction,
        }
        payload = self._json_request(
            "POST", "/v1/lavira/session/start", request_payload
        )
        parsed = G3SessionStarted.from_dict(payload, session_id)
        self.started = parsed
        self.ended = False
        self._premature_stop_preempt_decisions.clear()
        return parsed, payload

    def report_motion_window(
        self,
        *,
        decision_index: int,
        window_index: int,
        action: str,
        timestamp_start: float,
        timestamp_end: float,
        pose_frame_id: str,
        frame_epoch: int,
        pose_start: Any,
        pose_end: Any,
        displacement_m: float,
        local_planner_status: str,
        distance_to_local_goal_start: float,
        distance_to_local_goal_end: float,
        map_progress: Any,
    ) -> tuple[G3ExecutionControl, dict[str, Any]]:
        started = self._require_active()
        decision_index = self._decision_index(decision_index)
        window_index = self._non_negative_integer(window_index, "window_index")
        action = self._action(action)
        if action == "STOP":
            raise ValueError("Phase-four STOP decisions cannot report motion_window.")
        timestamp_start = _finite_number(timestamp_start, "timestamp_start")
        timestamp_end = _finite_number(timestamp_end, "timestamp_end")
        if timestamp_end < timestamp_start:
            raise ValueError("timestamp_end must be >= timestamp_start.")
        pose_frame_id = self._non_empty_string(pose_frame_id, "pose_frame_id")
        frame_epoch = self._non_negative_integer(frame_epoch, "frame_epoch")
        local_planner_status = self._enum_string(
            local_planner_status,
            "local_planner_status",
            G3_LOCAL_PLANNER_STATUSES,
        )
        map_progress = _map_progress_object(map_progress)
        request_payload = {
            "schema_version": G3_SCHEMA_VERSION,
            "request_type": "report_execution",
            "event_type": "motion_window",
            "session_id": started.session_id,
            "decision_index": decision_index,
            "event_id": (
                f"{started.session_id}:d{decision_index}:w{window_index}"
            ),
            "window_index": window_index,
            "action": action,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "pose_frame_id": pose_frame_id,
            "frame_epoch": frame_epoch,
            "pose_start": _pose_array(pose_start, "pose_start"),
            "pose_end": _pose_array(pose_end, "pose_end"),
            "displacement_m": _finite_number(
                displacement_m, "displacement_m", non_negative=True
            ),
            "local_planner_status": local_planner_status,
            "distance_to_local_goal_start": _finite_number(
                distance_to_local_goal_start,
                "distance_to_local_goal_start",
                non_negative=True,
            ),
            "distance_to_local_goal_end": _finite_number(
                distance_to_local_goal_end,
                "distance_to_local_goal_end",
                non_negative=True,
            ),
            "map_progress": map_progress,
        }
        return self._send_execution_report(
            request_payload,
            decision_index=decision_index,
            event_type="motion_window",
        )

    def validate_decision_context(
        self,
        payload: dict[str, Any],
        *,
        decision_index: int,
    ) -> G3DecisionSupervision:
        """Validate phase-four supervision attached to one model decision."""

        started = self._require_active()
        decision_index = self._decision_index(decision_index)
        if payload.get("session_id") != started.session_id:
            raise G3SessionProtocolError("Decision response session_id mismatch.")
        if payload.get("session_status") != "ACTIVE":
            raise G3SessionProtocolError("Decision response session_status is not ACTIVE.")
        if payload.get("stage_plan_status") != "READY":
            raise G3SessionProtocolError("Decision response Stage Plan is not READY.")
        action = _required_string(payload, "action").upper()
        action_source = str(payload.get("action_source", "NAVIGATOR")).upper()
        control_value = payload.get("control")
        control = (
            None
            if control_value is None
            else _required_string(payload, "control").upper()
        )
        next_action_value = payload.get("next_action")
        next_action = (
            None
            if next_action_value is None
            else _required_string(payload, "next_action").upper()
        )

        # Recovery SAFE_STOP is a terminal control response. It deliberately
        # carries action=STOP with no visual target, so validate it before the
        # ordinary Recovery NAVIGATE/BACKTRACK and Navigator STOP contracts.
        if control == "SAFE_STOP":
            if payload.get("decision_index") != decision_index:
                raise G3SessionProtocolError(
                    "SAFE_STOP response decision_index mismatch."
                )
            if payload.get("stage_plan_id") != started.stage_plan_id:
                raise G3SessionProtocolError(
                    "SAFE_STOP response stage_plan_id mismatch."
                )
            if action_source != "RECOVERY" or action != "STOP":
                raise G3SessionProtocolError(
                    "SAFE_STOP must be a Recovery STOP response."
                )
            if next_action != "SAFE_STOP":
                raise G3SessionProtocolError(
                    "SAFE_STOP control must request SAFE_STOP."
                )
            if str(payload.get("stop_phase", "")).upper() != "SAFE_STOP":
                raise G3SessionProtocolError(
                    "SAFE_STOP control must use stop_phase=SAFE_STOP."
                )
            if any(
                payload.get(field) is not None
                for field in ("direction", "target", "bbox_2d", "waypoint")
            ):
                raise G3SessionProtocolError(
                    "Recovery SAFE_STOP visual navigation fields must be null."
                )
            if payload.get("stop_gate") is not None:
                raise G3SessionProtocolError(
                    "Recovery SAFE_STOP must not contain a STOP Gate result."
                )
            recovery = payload.get("recovery")
            if (
                not isinstance(recovery, dict)
                or recovery.get("role") != "recovery_planner"
                or recovery.get("parse_success") is not False
                or not isinstance(recovery.get("status"), str)
                or not recovery.get("status")
            ):
                raise G3SessionProtocolError(
                    "Recovery SAFE_STOP metadata is invalid."
                )
            phase6 = payload.get("phase6")
            if (
                not isinstance(phase6, dict)
                or phase6.get("status") != "SAFE_STOP"
                or phase6.get("recovery_phase") != "SAFE_STOP"
                or not isinstance(phase6.get("transition"), dict)
            ):
                raise G3SessionProtocolError(
                    "Recovery SAFE_STOP phase6 metadata is invalid."
                )
            return G3DecisionSupervision(
                stage_progress=None,
                stop_gate=None,
                stop_phase="SAFE_STOP",
                control="SAFE_STOP",
                next_action="SAFE_STOP",
                action_source="RECOVERY",
                recovery=True,
            )

        # Recovery decisions intentionally do not run Stage Progress again.
        # They reuse the same /decision endpoint but carry an explicit source
        # and a frozen EXECUTE_RECOVERY transition.
        if action_source == "RECOVERY":
            if action not in {"NAVIGATE", "BACKTRACK"}:
                raise G3SessionProtocolError(
                    f"Unsupported Recovery action {action!r}."
                )
            if (control, next_action) != ("CONTINUE", "EXECUTE_RECOVERY"):
                raise G3SessionProtocolError(
                    "Recovery decision control/next_action mismatch."
                )
            recovery = payload.get("recovery")
            phase6 = payload.get("phase6")
            if (
                not isinstance(recovery, dict)
                or recovery.get("role") != "recovery_planner"
                or recovery.get("status") != "READY"
                or recovery.get("parse_success") is not True
                or not isinstance(recovery.get("model_backend"), str)
            ):
                raise G3SessionProtocolError("Recovery planner result is invalid.")
            if (
                not isinstance(phase6, dict)
                or phase6.get("action_source") != "RECOVERY"
                or phase6.get("status") != "RECOVERY_PLAN_READY"
            ):
                raise G3SessionProtocolError("Recovery phase6 metadata is invalid.")
            if action == "BACKTRACK":
                if phase6.get("waypoint_validated") is not True:
                    raise G3SessionProtocolError(
                        "Recovery BACKTRACK waypoint was not validated."
                    )
                _response_non_negative_integer(
                    payload.get("waypoint"), "waypoint"
                )
            elif payload.get("waypoint") is not None:
                raise G3SessionProtocolError(
                    "Recovery NAVIGATE must use waypoint=null."
                )
            return G3DecisionSupervision(
                stage_progress=None,
                stop_gate=None,
                stop_phase=None,
                control=control,
                next_action=next_action,
                action_source=action_source,
                recovery=True,
            )

        if action_source != "NAVIGATOR":
            raise G3SessionProtocolError(
                f"Unsupported decision action_source {action_source!r}."
            )

        # Verified supervision has higher authority than the Navigator action.
        # Both Semantic Audit and PREMATURE_STOP use the original LaViRA
        # Candidate Arbiter -> Failure Verifier -> atomic PREEMPT boundary.
        if control == "PREEMPT":
            if next_action != "ACTION_COMPLETE_PREEMPTED":
                raise G3SessionProtocolError(
                    "Decision PREEMPT must request ACTION_COMPLETE_PREEMPTED."
                )
            verification = payload.get("failure_verification")
            if (
                not isinstance(verification, dict)
                or verification.get("verdict") != "FAILURE"
                or verification.get("need_recovery") is not True
                or verification.get("parse_success") is not True
                or verification.get("model_backend") != "strong_api"
            ):
                raise G3SessionProtocolError(
                    "Decision PREEMPT requires a successful FAILURE verification."
                )
            if action == "STOP":
                if payload.get("stage_plan_id") != started.stage_plan_id:
                    raise G3SessionProtocolError(
                        "PREMATURE_STOP PREEMPT stage_plan_id mismatch."
                    )
                stage_progress = G3StageProgress.from_dict(
                    payload.get("stage_progress"),
                    started=started,
                    decision_index=decision_index,
                )
                stop_gate = G3StopGate.from_dict(
                    payload.get("stop_gate"), decision_index=decision_index
                )
                stop_phase = _required_string(payload, "stop_phase").upper()
                if stop_phase != "PREMATURE_STOP" or stop_gate.verdict != "PREMATURE":
                    raise G3SessionProtocolError(
                        "STOP PREEMPT requires a verified PREMATURE_STOP Gate."
                    )
                phase5 = payload.get("phase5")
                candidate = phase5.get("candidate") if isinstance(phase5, dict) else None
                arbiter = phase5.get("arbiter") if isinstance(phase5, dict) else None
                transition = phase5.get("transition") if isinstance(phase5, dict) else None
                preemption = phase5.get("preemption") if isinstance(phase5, dict) else None
                candidate_id = (
                    candidate.get("candidate_id") if isinstance(candidate, dict) else None
                )
                if (
                    not isinstance(phase5, dict)
                    or phase5.get("status") != "PREEMPT_REQUESTED"
                    or phase5.get("candidate_source") != "premature_stop"
                    or phase5.get("recovery_phase") != "RECOVERY_PLANNING"
                    or not isinstance(candidate, dict)
                    or not isinstance(candidate_id, str)
                    or not candidate_id
                    or candidate.get("source") != "premature_stop"
                    or candidate.get("candidate_type") != "PREMATURE_STOP"
                    or candidate.get("trigger_type") != "PREMATURE_STOP"
                    or candidate.get("decision_index") != decision_index
                    or not isinstance(arbiter, dict)
                    or arbiter.get("accepted") is not True
                    or arbiter.get("selected_candidate_id") != candidate_id
                    or arbiter.get("selected_source") != "premature_stop"
                    or not isinstance(transition, dict)
                    or transition.get("confirmed") is not True
                    or transition.get("candidate_id") != candidate_id
                    or transition.get("phase") != "RECOVERY_PLANNING"
                    or transition.get("preemption_pending") is not True
                    or not isinstance(preemption, dict)
                    or preemption.get("accepted") is not True
                    or preemption.get("decision_index") != decision_index
                    or preemption.get("stop_phase") != "PREMATURE_STOP"
                    or preemption.get("preempt_ack_pending") is not True
                ):
                    raise G3SessionProtocolError(
                        "PREMATURE_STOP PREEMPT lacks confirmed phase5 metadata."
                    )
                self._premature_stop_preempt_decisions.add(decision_index)
                return G3DecisionSupervision(
                    stage_progress=stage_progress,
                    stop_gate=stop_gate,
                    stop_phase=stop_phase,
                    control=control,
                    next_action=next_action,
                    action_source=action_source,
                    recovery=False,
                    preempt_source="premature_stop",
                )

            phase7 = payload.get("phase7")
            if action != "NAVIGATE" or (
                not isinstance(phase7, dict)
                or phase7.get("status") != "PREEMPT_REQUESTED"
            ):
                raise G3SessionProtocolError(
                    "Decision PREEMPT lacks Semantic Audit metadata."
                )
            return G3DecisionSupervision(
                stage_progress=None,
                stop_gate=None,
                stop_phase=None,
                control=control,
                next_action=next_action,
                action_source=action_source,
                recovery=False,
                preempt_source="semantic_audit",
            )

        if payload.get("stage_plan_id") != started.stage_plan_id:
            raise G3SessionProtocolError("Decision response stage_plan_id mismatch.")
        stage_progress = G3StageProgress.from_dict(
            payload.get("stage_progress"),
            started=started,
            decision_index=decision_index,
        )
        if action != "STOP":
            if payload.get("stop_gate") is not None or payload.get("stop_phase") is not None:
                raise G3SessionProtocolError(
                    "Non-STOP decision must use null stop_gate and stop_phase."
                )
            if payload.get("control") is not None or payload.get("next_action") is not None:
                raise G3SessionProtocolError(
                    "Non-STOP decision must not carry STOP control fields."
                )
            return G3DecisionSupervision(
                stage_progress=stage_progress,
                stop_gate=None,
                stop_phase=None,
                control=None,
                next_action=None,
                action_source=action_source,
                recovery=False,
            )

        stop_gate = G3StopGate.from_dict(
            payload.get("stop_gate"), decision_index=decision_index
        )
        stop_phase = _required_string(payload, "stop_phase").upper()
        control = _required_string(payload, "control").upper()
        next_action = _required_string(payload, "next_action").upper()
        expected_top_level = {
            "STOP_CONFIRMED": ("STOP_CONFIRMED", "END_SESSION_SUCCESS"),
            "PREMATURE_STOP": ("CONTINUE", "REQUEST_DECISION"),
            "STOP_PENDING": ("CONTINUE", "REQUEST_DECISION"),
        }
        if stop_phase not in expected_top_level:
            raise G3SessionProtocolError(f"Unsupported STOP phase {stop_phase!r}.")
        if stop_gate.stop_phase != stop_phase:
            raise G3SessionProtocolError("Nested/top-level stop_phase mismatch.")
        if (control, next_action) != expected_top_level[stop_phase]:
            raise G3SessionProtocolError("STOP phase control/next_action mismatch.")
        return G3DecisionSupervision(
            stage_progress=stage_progress,
            stop_gate=stop_gate,
            stop_phase=stop_phase,
            control=control,
            next_action=next_action,
            action_source=action_source,
            recovery=False,
        )

    def report_action_complete(
        self,
        *,
        decision_index: int,
        action: str,
        status: str,
        reached_local_goal: bool,
        timestamp: float,
        pose_frame_id: str,
        frame_epoch: int,
        decision_pose: Any,
        final_pose: Any,
        displacement_m: float,
        planner_result: str,
        waypoint_id: int,
    ) -> tuple[G3ExecutionControl, dict[str, Any]]:
        started = self._require_active()
        decision_index = self._decision_index(decision_index)
        action = self._action(action)
        if action == "STOP":
            if decision_index not in self._premature_stop_preempt_decisions:
                raise ValueError(
                    "STOP decisions cannot report action_complete without a "
                    "verified PREMATURE_STOP PREEMPT."
                )
        status = self._enum_string(status, "status", G3_ACTION_STATUSES)
        if not isinstance(reached_local_goal, bool):
            raise ValueError("reached_local_goal must be a boolean.")
        planner_result = self._enum_string(
            planner_result, "planner_result", G3_PLANNER_RESULTS
        )
        expected = {
            "COMPLETED": (True, "REACHED"),
            "PREEMPTED": (False, "PREEMPTED"),
        }
        if (
            status in expected
            and (reached_local_goal, planner_result) != expected[status]
        ):
            raise ValueError(
                f"{status} requires reached_local_goal={expected[status][0]} "
                f"and planner_result={expected[status][1]}."
            )
        if status == "FAILED" and (
            reached_local_goal or planner_result in {"REACHED", "PREEMPTED"}
        ):
            raise ValueError("FAILED requires a non-reached failure planner_result.")
        if action == "STOP" and status != "PREEMPTED":
            raise ValueError(
                "A verified PREMATURE_STOP may only report status=PREEMPTED."
            )
        pose_frame_id = self._non_empty_string(pose_frame_id, "pose_frame_id")
        frame_epoch = self._non_negative_integer(frame_epoch, "frame_epoch")
        waypoint_id = self._non_negative_integer(waypoint_id, "waypoint_id")
        if waypoint_id != decision_index:
            raise ValueError("Phase-three waypoint_id must equal decision_index.")
        request_payload = {
            "schema_version": G3_SCHEMA_VERSION,
            "request_type": "report_execution",
            "event_type": "action_complete",
            "session_id": started.session_id,
            "decision_index": decision_index,
            "event_id": f"{started.session_id}:d{decision_index}:complete",
            "action": action,
            "status": status,
            "reached_local_goal": reached_local_goal,
            "timestamp": _finite_number(timestamp, "timestamp"),
            "pose_frame_id": pose_frame_id,
            "frame_epoch": frame_epoch,
            "decision_pose": _pose_array(decision_pose, "decision_pose"),
            "final_pose": _pose_array(final_pose, "final_pose"),
            "displacement_m": _finite_number(
                displacement_m, "displacement_m", non_negative=True
            ),
            "planner_result": planner_result,
            "waypoint_id": waypoint_id,
        }
        result = self._send_execution_report(
            request_payload,
            decision_index=decision_index,
            event_type="action_complete",
        )
        if action == "STOP":
            self._premature_stop_preempt_decisions.discard(decision_index)
        return result

    def _send_execution_report(
        self,
        request_payload: dict[str, Any],
        *,
        decision_index: int,
        event_type: str,
    ) -> tuple[G3ExecutionControl, dict[str, Any]]:
        started = self._require_active()
        # One retry is safe because event_id is deterministic and the server is
        # idempotent.  Reuse the exact same object; never advance window_index
        # merely because the first response was lost.
        for attempt in range(2):
            try:
                payload = self._json_request(
                    "POST", "/v1/lavira/execution/report", request_payload
                )
                break
            except G3TransportError:
                if attempt == 1:
                    raise
        parsed = G3ExecutionControl.from_dict(
            payload,
            session_id=started.session_id,
            decision_index=int(decision_index),
            event_type=event_type,
            stage_plan_id=started.stage_plan_id,
        )
        return parsed, payload

    @staticmethod
    def _non_empty_string(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _non_negative_integer(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer.")
        return int(value)

    @classmethod
    def _decision_index(cls, value: Any) -> int:
        return cls._non_negative_integer(value, "decision_index")

    @classmethod
    def _action(cls, value: Any) -> str:
        return cls._non_empty_string(value, "action").upper()

    @classmethod
    def _enum_string(cls, value: Any, field: str, allowed: frozenset[str]) -> str:
        result = cls._non_empty_string(value, field).upper()
        if result not in allowed:
            raise ValueError(f"{field} must be one of {sorted(allowed)}.")
        return result

    def end_session(
        self,
        *,
        status: str,
        reason: str,
    ) -> tuple[G3SessionEnded, dict[str, Any]]:
        started = self._require_active()
        status = self._enum_string(
            status, "status", G3_SESSION_FINAL_STATUSES
        )
        if not reason.strip():
            raise ValueError("Session final reason must not be empty.")
        request_payload = {
            "schema_version": G3_SCHEMA_VERSION,
            "request_type": "end_session",
            "session_id": started.session_id,
            "status": status,
            "reason": reason,
        }
        payload = self._json_request(
            "POST", "/v1/lavira/session/end", request_payload
        )
        parsed = G3SessionEnded.from_dict(
            payload,
            session_id=started.session_id,
            stage_plan_id=started.stage_plan_id,
        )
        self.ended = True
        self._premature_stop_preempt_decisions.clear()
        return parsed, payload

    def _require_active(self) -> G3SessionStarted:
        if self.started is None or self.ended:
            raise G3SessionProtocolError("No active G3 session is owned by this client.")
        return self.started

    def _json_request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                response_bytes = response.read(1024 * 1024 + 1)
        except HTTPError as exc:
            error_body = exc.read(64 * 1024).decode("utf-8", errors="replace")
            try:
                error_payload = json.loads(error_body)
            except json.JSONDecodeError:
                error_payload = None
            if isinstance(error_payload, dict):
                code = error_payload.get("error_code", "UNKNOWN")
                message = error_payload.get("message", error_body)
                detail = f"{code}: {message}"
            else:
                detail = error_body
            raise G3SessionProtocolError(
                f"G3 service returned HTTP {exc.code} for {path}: {detail}"
            ) from exc
        except URLError as exc:
            raise G3TransportError(
                f"Could not reach G3 service at {self.base_url!r}: {exc.reason}"
            ) from exc
        except (TimeoutError, OSError) as exc:
            raise G3TransportError(
                f"Could not reach G3 service at {self.base_url!r}: {exc}"
            ) from exc
        if len(response_bytes) > 1024 * 1024:
            raise G3SessionProtocolError("G3 JSON response exceeded 1 MiB.")
        try:
            decoded = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise G3SessionProtocolError("G3 service did not return UTF-8 JSON.") from exc
        if not isinstance(decoded, dict):
            raise G3SessionProtocolError("G3 JSON response must be an object.")
        if decoded.get("response_type") == "error":
            raise G3SessionProtocolError(
                f"G3 service error {decoded.get('error_code')}: {decoded.get('message')}"
            )
        return decoded


__all__ = [
    "G3DecisionSupervision",
    "G3ExecutionControl",
    "G3SessionClient",
    "G3SessionEnded",
    "G3SessionProtocolError",
    "G3SessionStarted",
    "G3TransportError",
    "base_url_from_decision_url",
]
