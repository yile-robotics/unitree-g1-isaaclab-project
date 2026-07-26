#!/usr/bin/env python3
from __future__ import annotations

"""用于协议、history、BACKTRACK 和 STOP 测试的本机 LaViRA mock server。"""

import argparse
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import struct
import time
from typing import Any

from goal_tracking.lavira_protocol import (
    LAVIRA_ACTIONS,
    LAVIRA_DIRECTIONS,
    NavigationDecisionRequest,
    NavigationDecisionResponse,
)


MAX_REQUEST_BYTES = 64 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_multipart(content_type: str, body: bytes) -> dict[str, tuple[str | None, bytes]]:
    """使用标准库 email MIME parser 解析 multipart/form-data。"""
    mime_message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: "
        + content_type.encode("ascii")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    if not mime_message.is_multipart():
        raise ValueError("Request body is not multipart/form-data.")

    fields: dict[str, tuple[str | None, bytes]] = {}
    for part in mime_message.iter_parts():
        field_name = part.get_param("name", header="content-disposition")
        if not field_name:
            raise ValueError("Multipart part is missing its form field name.")
        if field_name in fields:
            raise ValueError(f"Duplicate multipart field {field_name!r}.")
        filename = part.get_filename()
        payload = part.get_payload(decode=True)
        if payload is None:
            payload = b""
        fields[field_name] = (filename, payload)
    return fields


def png_dimensions(payload: bytes) -> tuple[int, int]:
    if len(payload) < 24 or not payload.startswith(PNG_SIGNATURE):
        raise ValueError("Image payload is not a valid PNG header.")
    if payload[12:16] != b"IHDR":
        raise ValueError("PNG payload does not start with an IHDR chunk.")
    return struct.unpack(">II", payload[16:24])


class MockLaViRAHandler(BaseHTTPRequestHandler):
    server_version = "MockLaViRA/2.0"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/lavira/decision":
            self._send_json(404, {"error": f"Unknown endpoint {self.path!r}."})
            return
        try:
            request_metadata, image_count = self._read_navigation_decision_request()
            action = self.server.fake_action
            if self.server.fake_delay_seconds > 0.0:
                time.sleep(self.server.fake_delay_seconds)
            response = self._make_response(request_metadata, action)
            response.validate_matches(request_metadata)
            print(
                "[MOCK-LAVIRA] Valid request: "
                f"session={request_metadata.session_id} "
                f"observation={request_metadata.observation_id} "
                f"history={len(request_metadata.history)} images={image_count} "
                f"resolution={request_metadata.image_width}x{request_metadata.image_height}"
            )
            self._send_json(200, response.to_dict())
        except Exception as exc:
            print(f"[MOCK-LAVIRA] Rejected request: {exc}")
            self._send_json(400, {"error": str(exc)})

    def _make_response(
        self, request_metadata: NavigationDecisionRequest, action: str
    ) -> NavigationDecisionResponse:
        common = {
            "session_id": request_metadata.session_id,
            "observation_id": request_metadata.observation_id,
            "action": action,
            "progress_analysis": f"Mock analysis for {action}.",
            "reasoning": (
                "Mock response for validating Isaac Sim multipart transport; "
                "no model inference was performed."
            ),
        }
        if action == "BACKTRACK":
            return NavigationDecisionResponse(
                **common,
                direction=None,
                target=None,
                bbox_2d=None,
                waypoint=self.server.fake_waypoint,
            )

        bbox = self.server.fake_bbox
        if bbox is None:
            bbox = (
                request_metadata.image_width // 4,
                request_metadata.image_height // 4,
                3 * request_metadata.image_width // 4,
                3 * request_metadata.image_height // 4,
            )
        x1, y1, x2, y2 = (int(value) for value in bbox)
        x1 = max(0, min(x1, request_metadata.image_width - 1))
        y1 = max(0, min(y1, request_metadata.image_height - 1))
        x2 = max(x1 + 1, min(x2, request_metadata.image_width))
        y2 = max(y1 + 1, min(y2, request_metadata.image_height))
        return NavigationDecisionResponse(
            **common,
            direction=self.server.fake_direction,
            target=self.server.fake_target,
            bbox_2d=(x1, y1, x2, y2),
            waypoint=None,
        )

    def _read_navigation_decision_request(self) -> tuple[NavigationDecisionRequest, int]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise ValueError("Content-Type must be multipart/form-data.")
        content_length_value = self.headers.get("Content-Length")
        if content_length_value is None:
            raise ValueError("Content-Length is required.")
        content_length = int(content_length_value)
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            raise ValueError(
                f"Content-Length must be in [1, {MAX_REQUEST_BYTES}], got {content_length}."
            )
        body = self.rfile.read(content_length)
        if len(body) != content_length:
            raise ValueError("Request body ended before Content-Length bytes were received.")

        fields = parse_multipart(content_type, body)
        if "metadata" not in fields:
            raise ValueError("Multipart request is missing the metadata field.")
        try:
            metadata_value: Any = json.loads(fields["metadata"][1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("metadata must contain valid UTF-8 JSON.") from exc
        if not isinstance(metadata_value, dict):
            raise ValueError("metadata JSON must be an object.")
        request_metadata = NavigationDecisionRequest.from_metadata(metadata_value)

        expected_fields = {"metadata", *request_metadata.required_image_fields}
        if set(fields) != expected_fields:
            raise ValueError(
                "Multipart fields do not match metadata: "
                f"missing={sorted(expected_fields - set(fields))}, "
                f"extra={sorted(set(fields) - expected_fields)}."
            )
        for field_name in request_metadata.required_image_fields:
            filename, payload = fields[field_name]
            if filename != f"{field_name}.png":
                raise ValueError(
                    f"Image field {field_name!r} has unexpected filename {filename!r}."
                )
            width, height = png_dimensions(payload)
            if (width, height) != (
                request_metadata.image_width,
                request_metadata.image_height,
            ):
                raise ValueError(
                    f"Image {field_name!r} is {width}x{height}, expected "
                    f"{request_metadata.image_width}x{request_metadata.image_height}."
                )
        return request_metadata, len(request_metadata.required_image_fields)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        response = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate LaViRA schema v2 multipart requests and return one "
            "normalized end-to-end decision."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--action", choices=LAVIRA_ACTIONS, default="NAVIGATE")
    parser.add_argument("--direction", choices=LAVIRA_DIRECTIONS, default="left")
    parser.add_argument("--target", default="doorway")
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=None,
        help="Pixel xyxy box. Defaults to the center half of the request image.",
    )
    parser.add_argument("--waypoint", type=int, default=0)
    parser.add_argument(
        "--delay_seconds",
        type=float,
        default=0.0,
        help="Artificial response delay for timeout tests.",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MockLaViRAHandler)
    server.fake_action = args.action
    server.fake_direction = args.direction
    server.fake_target = args.target
    server.fake_bbox = tuple(args.bbox) if args.bbox is not None else None
    server.fake_waypoint = args.waypoint
    server.fake_delay_seconds = max(0.0, args.delay_seconds)
    print(
        "[MOCK-LAVIRA] Listening on "
        f"http://{args.host}:{args.port}/v1/lavira/decision "
        f"(action={args.action}, direction={args.direction}, "
        f"waypoint={args.waypoint}, delay={server.fake_delay_seconds:.1f}s). "
        "Press Ctrl-C to stop."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MOCK-LAVIRA] Stopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
