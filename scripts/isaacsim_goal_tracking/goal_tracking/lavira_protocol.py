from __future__ import annotations

"""LaViRA 第四组 Qwen end-to-end 的 HTTP 数据契约。

模型 prompt 的语义以 ``lavira_code/vlnce_baselines/`` 中的
``lavira_main_qwen_end2end.py`` 为准：所有历史 waypoint 保留文字，只有最近
四个 waypoint 携带 init/dir 图片，当前 panorama 顺序固定为
forward -> left -> behind -> right。

Qwen 原文由远程适配层解析。Isaac Sim 只接收 schema v2 的稳定归一化响应，
并通过 session_id/observation_id 把每轮结果绑定到原始 FrameBundle。
"""

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping


LAVIRA_SCHEMA_VERSION = 2
LAVIRA_REQUEST_TYPE = "end2end_decision"
LAVIRA_RESPONSE_TYPE = "end2end_decision"
LAVIRA_DIRECTIONS = ("forward", "left", "behind", "right")
LAVIRA_ACTIONS = ("NAVIGATE", "BACKTRACK", "STOP")
LAVIRA_MAX_HISTORY_IMAGE_WAYPOINTS = 4
LAVIRA_MAX_REQUEST_IMAGES = 16
_IMAGE_FIELD_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _validate_image_field(value: Any, field_name: str) -> str:
    value = _require_non_empty_string(value, field_name)
    if _IMAGE_FIELD_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name}={value!r} is not a safe multipart field name."
        )
    return value


def _validate_bbox_tuple(value: Any, field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError(f"{field_name} must contain exactly four numbers.")
    for coordinate in value:
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, (int, float))
            or not math.isfinite(float(coordinate))
        ):
            raise ValueError(f"{field_name} must contain finite numbers.")
    x1, y1, x2, y2 = value
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"{field_name} must satisfy x1 < x2 and y1 < y2.")


@dataclass(frozen=True)
class NavigationHistoryEntry:
    """一个已完成、可作为 BACKTRACK 目标的 0-based waypoint。"""

    waypoint_id: int
    step: int
    turn_action: str
    description: str
    init_image_field: str | None = None
    dir_image_field: str | None = None

    def __post_init__(self) -> None:
        _require_non_negative_int(self.waypoint_id, "history.waypoint_id")
        _require_non_negative_int(self.step, "history.step")
        turn_action = _require_non_empty_string(
            self.turn_action, "history.turn_action"
        ).lower()
        valid_turn_actions = {f"turn {direction}" for direction in LAVIRA_DIRECTIONS}
        if turn_action not in valid_turn_actions:
            raise ValueError(
                f"history.turn_action must be one of {sorted(valid_turn_actions)}, "
                f"got {self.turn_action!r}."
            )
        _require_non_empty_string(self.description, "history.description")

        has_init = self.init_image_field is not None
        has_dir = self.dir_image_field is not None
        if has_init != has_dir:
            raise ValueError(
                "history init_image_field and dir_image_field must either both "
                "exist or both be omitted."
            )
        if has_init:
            init_field = _validate_image_field(
                self.init_image_field, "history.init_image_field"
            )
            dir_field = _validate_image_field(
                self.dir_image_field, "history.dir_image_field"
            )
            if init_field == dir_field:
                raise ValueError(
                    "History init and direction image fields must differ."
                )

    @property
    def has_images(self) -> bool:
        return self.init_image_field is not None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "waypoint_id": self.waypoint_id,
            "step": self.step,
            "turn_action": self.turn_action.lower(),
            "description": self.description,
        }
        if self.has_images:
            result["init_image_field"] = self.init_image_field
            result["dir_image_field"] = self.dir_image_field
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NavigationHistoryEntry":
        if not isinstance(value, Mapping):
            raise ValueError("Each history entry must be a JSON object.")
        return cls(
            waypoint_id=value.get("waypoint_id"),
            step=value.get("step"),
            turn_action=value.get("turn_action"),
            description=value.get("description"),
            init_image_field=value.get("init_image_field"),
            dir_image_field=value.get("dir_image_field"),
        )


@dataclass(frozen=True)
class NavigationDecisionRequest:
    """一次第四组单阶段 Qwen 请求的 multipart metadata。"""

    session_id: str
    observation_id: str
    bundle_id: int
    decision_index: int
    sim_step: int
    timestamp: float
    instruction: str
    image_width: int
    image_height: int
    history: tuple[NavigationHistoryEntry, ...]
    current_panorama: Mapping[str, str]
    schema_version: int = LAVIRA_SCHEMA_VERSION
    request_type: str = LAVIRA_REQUEST_TYPE

    def __post_init__(self) -> None:
        if self.schema_version != LAVIRA_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version={self.schema_version}; "
                f"expected {LAVIRA_SCHEMA_VERSION}."
            )
        if self.request_type != LAVIRA_REQUEST_TYPE:
            raise ValueError(
                f"request_type must be {LAVIRA_REQUEST_TYPE!r}."
            )
        _require_non_empty_string(self.session_id, "session_id")
        _require_non_empty_string(self.observation_id, "observation_id")
        _require_non_negative_int(self.bundle_id, "bundle_id")
        _require_non_negative_int(self.decision_index, "decision_index")
        _require_non_negative_int(self.sim_step, "sim_step")
        if not isinstance(self.timestamp, (int, float)) or not math.isfinite(
            float(self.timestamp)
        ):
            raise ValueError("timestamp must be finite.")
        _require_non_empty_string(self.instruction, "instruction")
        if (
            isinstance(self.image_width, bool)
            or not isinstance(self.image_width, int)
            or self.image_width <= 0
        ):
            raise ValueError("image_width must be a positive integer.")
        if (
            isinstance(self.image_height, bool)
            or not isinstance(self.image_height, int)
            or self.image_height <= 0
        ):
            raise ValueError("image_height must be a positive integer.")

        if tuple(self.current_panorama) != LAVIRA_DIRECTIONS:
            raise ValueError(
                "current_panorama keys must be ordered exactly as "
                f"{LAVIRA_DIRECTIONS}, got {tuple(self.current_panorama)}."
            )

        image_fields: list[str] = []
        waypoint_ids: list[int] = []
        image_history_start = max(
            0, len(self.history) - LAVIRA_MAX_HISTORY_IMAGE_WAYPOINTS
        )
        for index, entry in enumerate(self.history):
            if not isinstance(entry, NavigationHistoryEntry):
                raise ValueError(
                    "history must contain NavigationHistoryEntry values."
                )
            waypoint_ids.append(entry.waypoint_id)
            should_have_images = index >= image_history_start
            if entry.has_images != should_have_images:
                expected = "include images" if should_have_images else "omit images"
                raise ValueError(
                    f"history waypoint {entry.waypoint_id} must {expected}; only "
                    f"the latest {LAVIRA_MAX_HISTORY_IMAGE_WAYPOINTS} waypoints "
                    "carry images."
                )
            if entry.has_images:
                image_fields.extend(
                    (entry.init_image_field, entry.dir_image_field)
                )

        if waypoint_ids != list(range(len(waypoint_ids))):
            raise ValueError(
                "history waypoint_id values must be contiguous and zero-based."
            )

        for direction in LAVIRA_DIRECTIONS:
            image_fields.append(
                _validate_image_field(
                    self.current_panorama[direction],
                    f"current_panorama.{direction}",
                )
            )
        if len(set(image_fields)) != len(image_fields):
            raise ValueError("All history and panorama image fields must be unique.")
        if len(image_fields) > LAVIRA_MAX_REQUEST_IMAGES:
            raise ValueError(
                f"Request contains {len(image_fields)} images; maximum is "
                f"{LAVIRA_MAX_REQUEST_IMAGES}."
            )

    @property
    def required_image_fields(self) -> tuple[str, ...]:
        fields: list[str] = []
        for entry in self.history:
            if entry.has_images:
                fields.extend((entry.init_image_field, entry.dir_image_field))
        fields.extend(self.current_panorama[direction] for direction in LAVIRA_DIRECTIONS)
        return tuple(fields)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_type": self.request_type,
            "session_id": self.session_id,
            "observation_id": self.observation_id,
            "bundle_id": self.bundle_id,
            "decision_index": self.decision_index,
            "sim_step": self.sim_step,
            "timestamp": float(self.timestamp),
            "instruction": self.instruction,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "history": [entry.to_dict() for entry in self.history],
            "current_panorama": dict(self.current_panorama),
        }

    @classmethod
    def from_metadata(cls, value: Mapping[str, Any]) -> "NavigationDecisionRequest":
        history_value = value.get("history")
        if not isinstance(history_value, list):
            raise ValueError("history must be a JSON list.")
        panorama_value = value.get("current_panorama")
        if not isinstance(panorama_value, Mapping):
            raise ValueError("current_panorama must be a JSON object.")
        if set(panorama_value) != set(LAVIRA_DIRECTIONS):
            raise ValueError(
                f"current_panorama must contain exactly {LAVIRA_DIRECTIONS}."
            )
        ordered_panorama = {
            direction: panorama_value.get(direction) for direction in LAVIRA_DIRECTIONS
        }
        return cls(
            schema_version=value.get("schema_version"),
            request_type=value.get("request_type"),
            session_id=value.get("session_id"),
            observation_id=value.get("observation_id"),
            bundle_id=value.get("bundle_id"),
            decision_index=value.get("decision_index"),
            sim_step=value.get("sim_step"),
            timestamp=value.get("timestamp"),
            instruction=value.get("instruction"),
            image_width=value.get("image_width"),
            image_height=value.get("image_height"),
            history=tuple(
                NavigationHistoryEntry.from_dict(entry)
                for entry in history_value
            ),
            current_panorama=ordered_panorama,
        )


@dataclass(frozen=True)
class NavigationDecisionResponse:
    """远程适配层返回的 schema v2 归一化导航决策。"""

    session_id: str
    observation_id: str
    action: str
    direction: str | None
    target: str | None
    bbox_2d: tuple[float, float, float, float] | None
    waypoint: int | None
    progress_analysis: str
    reasoning: str
    schema_version: int = LAVIRA_SCHEMA_VERSION
    response_type: str = LAVIRA_RESPONSE_TYPE

    def __post_init__(self) -> None:
        if self.schema_version != LAVIRA_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported response schema_version={self.schema_version}."
            )
        if self.response_type != LAVIRA_RESPONSE_TYPE:
            raise ValueError(
                f"response_type must be {LAVIRA_RESPONSE_TYPE!r}."
            )
        _require_non_empty_string(self.session_id, "response.session_id")
        _require_non_empty_string(self.observation_id, "response.observation_id")
        action = _require_non_empty_string(self.action, "response.action").upper()
        if action not in LAVIRA_ACTIONS:
            raise ValueError(
                f"response.action must be one of {LAVIRA_ACTIONS}, got {action!r}."
            )
        if not isinstance(self.progress_analysis, str):
            raise ValueError("response.progress_analysis must be a string.")
        if not isinstance(self.reasoning, str):
            raise ValueError("response.reasoning must be a string.")

        if action == "BACKTRACK":
            _require_non_negative_int(self.waypoint, "response.waypoint")
            if self.direction is not None or self.target is not None or self.bbox_2d is not None:
                raise ValueError(
                    "BACKTRACK response.direction/target/bbox_2d must be null."
                )
            return

        if self.direction not in LAVIRA_DIRECTIONS:
            raise ValueError(
                f"{action} response.direction must be one of {LAVIRA_DIRECTIONS}."
            )
        _require_non_empty_string(self.target, f"{action} response.target")
        _validate_bbox_tuple(self.bbox_2d, f"{action} response.bbox_2d")
        if self.waypoint is not None:
            raise ValueError(f"{action} response.waypoint must be null.")

    def validate_matches(self, request: NavigationDecisionRequest) -> None:
        if self.session_id != request.session_id:
            raise ValueError(
                "Response session_id does not match request: "
                f"{self.session_id!r} != {request.session_id!r}."
            )
        if self.observation_id != request.observation_id:
            raise ValueError(
                "Response observation_id does not match request: "
                f"{self.observation_id!r} != {request.observation_id!r}."
            )
        if self.action.upper() == "BACKTRACK":
            if self.waypoint >= len(request.history):
                raise ValueError(
                    "BACKTRACK response.waypoint is not present in request.history: "
                    f"{self.waypoint} >= {len(request.history)}."
                )
            return

        x1, y1, x2, y2 = self.bbox_2d
        if not (
            0 <= x1 < x2 <= request.image_width
            and 0 <= y1 < y2 <= request.image_height
        ):
            raise ValueError(
                "Normalized response.bbox_2d must lie inside the request image "
                f"0..{request.image_width} x 0..{request.image_height}."
            )

    def clipped_bbox(
        self, image_width: int, image_height: int
    ) -> tuple[int, int, int, int]:
        """把 xyxy 框转换为可安全索引 RGB-D 数组的像素范围。"""
        if self.bbox_2d is None:
            raise ValueError("BACKTRACK response has no bbox_2d.")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive.")
        x1, y1, x2, y2 = (int(round(value)) for value in self.bbox_2d)
        x1 = min(max(x1, 0), image_width - 1)
        x2 = min(max(x2, 0), image_width - 1)
        y1 = min(max(y1, 0), image_height - 1)
        y2 = min(max(y2, 0), image_height - 1)
        if x1 > x2 or y1 > y2:
            raise ValueError("response.bbox_2d is empty after index clipping.")
        return x1, y1, x2, y2

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "response_type": self.response_type,
            "session_id": self.session_id,
            "observation_id": self.observation_id,
            "action": self.action.upper(),
            "direction": self.direction,
            "target": self.target,
            "bbox_2d": list(self.bbox_2d) if self.bbox_2d is not None else None,
            "waypoint": self.waypoint,
            "progress_analysis": self.progress_analysis,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NavigationDecisionResponse":
        required_fields = {
            "schema_version",
            "response_type",
            "session_id",
            "observation_id",
            "action",
            "direction",
            "target",
            "bbox_2d",
            "waypoint",
            "progress_analysis",
            "reasoning",
        }
        missing = sorted(required_fields - set(value))
        if missing:
            raise ValueError(f"Navigation response is missing fields: {missing}.")
        action_value = value.get("action")
        action = action_value.upper() if isinstance(action_value, str) else action_value
        direction_value = value.get("direction")
        direction = (
            direction_value.lower() if isinstance(direction_value, str) else direction_value
        )
        bbox_value = value.get("bbox_2d")
        bbox = tuple(bbox_value) if isinstance(bbox_value, (list, tuple)) else bbox_value
        return cls(
            schema_version=value.get("schema_version"),
            response_type=value.get("response_type"),
            session_id=value.get("session_id"),
            observation_id=value.get("observation_id"),
            action=action,
            direction=direction,
            target=value.get("target"),
            bbox_2d=bbox,
            waypoint=value.get("waypoint"),
            progress_analysis=value.get("progress_analysis"),
            reasoning=value.get("reasoning"),
        )
