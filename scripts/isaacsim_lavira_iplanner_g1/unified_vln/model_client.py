from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

import numpy as np

from .model_contract import (
    LAVIRA_MAX_HISTORY_IMAGE_WAYPOINTS,
    NavigationDecisionRequest,
    NavigationDecisionResponse,
    NavigationHistoryEntry,
)
from .types import DIRECTION_ORDER, PanoramaBundle


@dataclass(frozen=True)
class CompletedWaypoint:
    """Model-visible history for one physically completed NAVIGATE action."""

    waypoint_id: int
    decision_step: int
    direction: str
    target: str
    init_rgb: np.ndarray
    direction_rgb: np.ndarray

    def history_entry(self, *, include_images: bool) -> NavigationHistoryEntry:
        prefix = f"history_{self.waypoint_id}"
        return NavigationHistoryEntry(
            waypoint_id=int(self.waypoint_id),
            step=int(self.decision_step),
            turn_action=f"turn {self.direction}",
            description=self.target,
            init_image_field=f"{prefix}_init" if include_images else None,
            dir_image_field=f"{prefix}_dir" if include_images else None,
        )


def build_model_history(
    records: list[CompletedWaypoint] | tuple[CompletedWaypoint, ...],
) -> tuple[tuple[NavigationHistoryEntry, ...], dict[str, np.ndarray]]:
    records = tuple(records)
    image_start = max(0, len(records) - LAVIRA_MAX_HISTORY_IMAGE_WAYPOINTS)
    entries: list[NavigationHistoryEntry] = []
    images: dict[str, np.ndarray] = {}
    for index, record in enumerate(records):
        if record.waypoint_id != index:
            raise ValueError("Completed waypoint ids must be contiguous and zero-based.")
        entry = record.history_entry(include_images=index >= image_start)
        entries.append(entry)
        if entry.has_images:
            images[entry.init_image_field] = np.asarray(record.init_rgb).copy()
            images[entry.dir_image_field] = np.asarray(record.direction_rgb).copy()
    return tuple(entries), images


def encode_rgb_png(rgb: np.ndarray) -> bytes:
    """Match the deployed transport: RGB uint8, lossless full-resolution PNG."""
    import cv2

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Expected HxWx3 uint8 RGB, got {rgb.shape}/{rgb.dtype}.")
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("OpenCV failed to encode model RGB as PNG.")
    return encoded.tobytes()


class CombinedModelClient:
    """Exact schema-v2 multipart client for the existing combined model server."""

    def __init__(self, server_url: str, timeout_s: float = 90.0):
        if not server_url.strip():
            raise ValueError("Model server URL must not be empty.")
        if timeout_s <= 0.0:
            raise ValueError("Model timeout must be positive.")
        self.server_url = server_url
        self.timeout_s = float(timeout_s)

    @staticmethod
    def make_request(
        bundle: PanoramaBundle,
        *,
        session_id: str,
        instruction: str,
        decision_index: int,
        history: tuple[NavigationHistoryEntry, ...] = (),
    ) -> NavigationDecisionRequest:
        bundle.validated()
        first = bundle.views[DIRECTION_ORDER[0]]
        image_height, image_width = first.rgb.shape[:2]
        panorama = {
            direction: f"current_{direction}" for direction in DIRECTION_ORDER
        }
        return NavigationDecisionRequest(
            session_id=session_id,
            observation_id=(
                f"{session_id}_decision_{int(decision_index):03d}"
            ),
            bundle_id=int(bundle.bundle_id),
            decision_index=int(decision_index),
            sim_step=int(bundle.sim_step),
            timestamp=float(bundle.timestamp),
            instruction=instruction,
            image_width=int(image_width),
            image_height=int(image_height),
            history=history,
            current_panorama=panorama,
        )

    @staticmethod
    def image_fields(
        bundle: PanoramaBundle,
        request: NavigationDecisionRequest,
        history_images: Mapping[str, np.ndarray | bytes] | None = None,
    ) -> dict[str, bytes]:
        bundle.validated()
        history_images = {} if history_images is None else history_images
        expected_history: list[str] = []
        for entry in request.history:
            if entry.has_images:
                expected_history.extend(
                    (entry.init_image_field, entry.dir_image_field)
                )
        if set(history_images) != set(expected_history):
            raise ValueError(
                "History image fields do not match request metadata: "
                f"expected={sorted(expected_history)}, got={sorted(history_images)}."
            )

        images: dict[str, bytes] = {}
        for field_name in expected_history:
            value = history_images[field_name]
            images[field_name] = (
                value if isinstance(value, bytes) else encode_rgb_png(value)
            )
        for direction in DIRECTION_ORDER:
            field_name = request.current_panorama[direction]
            images[field_name] = encode_rgb_png(bundle.views[direction].rgb)
        CombinedModelClient._validate_images(request, images)
        return images

    def decide(
        self,
        request_metadata: NavigationDecisionRequest,
        images: Mapping[str, bytes],
    ) -> tuple[NavigationDecisionResponse, dict]:
        body, boundary = self._multipart_body(request_metadata, images)
        http_request = Request(
            self.server_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(http_request, timeout=self.timeout_s) as response:
                response_bytes = response.read(1024 * 1024 + 1)
        except HTTPError as exc:
            error_body = exc.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Combined model returned HTTP {exc.code}: {error_body}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach combined model at {self.server_url!r}: {exc.reason}"
            ) from exc

        if len(response_bytes) > 1024 * 1024:
            raise RuntimeError("Combined model response exceeded 1 MiB.")
        try:
            payload = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Combined model did not return UTF-8 JSON.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Combined model JSON response must be an object.")
        parsed = NavigationDecisionResponse.from_dict(payload)
        parsed.validate_matches(request_metadata)
        return parsed, payload

    @staticmethod
    def _validate_images(
        request: NavigationDecisionRequest,
        images: Mapping[str, bytes],
    ) -> None:
        expected = set(request.required_image_fields)
        actual = set(images)
        if actual != expected:
            raise ValueError(
                "Model image field mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
            )
        for field_name, payload in images.items():
            if not isinstance(payload, bytes) or not payload.startswith(
                b"\x89PNG\r\n\x1a\n"
            ):
                raise ValueError(f"{field_name!r} is not a PNG byte payload.")

    @classmethod
    def _multipart_body(
        cls,
        request: NavigationDecisionRequest,
        images: Mapping[str, bytes],
    ) -> tuple[bytes, str]:
        cls._validate_images(request, images)
        boundary = f"----IsaacLaViRA{uuid.uuid4().hex}"
        chunks: list[bytes] = []

        def append_part(headers: list[str], payload: bytes) -> None:
            chunks.append(f"--{boundary}\r\n".encode("ascii"))
            for header in headers:
                chunks.append(f"{header}\r\n".encode("utf-8"))
            chunks.append(b"\r\n")
            chunks.append(payload)
            chunks.append(b"\r\n")

        metadata = json.dumps(
            request.to_metadata(), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        append_part(
            [
                'Content-Disposition: form-data; name="metadata"',
                "Content-Type: application/json; charset=utf-8",
            ],
            metadata,
        )
        for field_name in request.required_image_fields:
            append_part(
                [
                    "Content-Disposition: form-data; "
                    f'name="{field_name}"; filename="{field_name}.png"',
                    "Content-Type: image/png",
                ],
                images[field_name],
            )
        chunks.append(f"--{boundary}--\r\n".encode("ascii"))
        return b"".join(chunks), boundary


def response_debug_dict(
    response: NavigationDecisionResponse,
    *,
    projected_goal_xy: np.ndarray | None = None,
) -> dict:
    result = response.to_dict()
    if projected_goal_xy is not None:
        goal = np.asarray(projected_goal_xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(goal)):
            raise ValueError("Projected goal must be finite.")
        result["goal_after_turn_robot_xy_m"] = goal.tolist()
        result["goal_distance_m"] = float(math.hypot(goal[0], goal[1]))
    return result
