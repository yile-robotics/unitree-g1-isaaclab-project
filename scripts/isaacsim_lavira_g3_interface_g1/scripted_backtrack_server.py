#!/usr/bin/env python3
from __future__ import annotations

"""Deterministic schema-v2 decision server for physical BACKTRACK testing.

This is deliberately separate from the production G3 service.  It returns two
forward NAVIGATE decisions followed by BACKTRACK to wire waypoint 0, allowing
the Isaac executor to test stored-reverse motion without waiting for a model to
choose BACKTRACK randomly.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from typing import Any


DECISION_PATH = "/v1/lavira/decision"


def _metadata_from_multipart(body: bytes, content_type: str) -> dict[str, Any]:
    match = re.search(r"boundary=([^;]+)", content_type)
    if match is None:
        raise ValueError("multipart boundary is missing")
    boundary = match.group(1).strip().strip('"').encode("ascii")
    for part in body.split(b"--" + boundary):
        if b'name="metadata"' not in part:
            continue
        try:
            payload = part.split(b"\r\n\r\n", 1)[1]
            payload = payload.rsplit(b"\r\n", 1)[0]
            value = json.loads(payload.decode("utf-8"))
        except (IndexError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("metadata part is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("metadata must be a JSON object")
        return value
    raise ValueError("multipart request has no metadata part")


def _scripted_response(metadata: dict[str, Any]) -> dict[str, Any]:
    session_id = str(metadata.get("session_id", ""))
    observation_id = str(metadata.get("observation_id", ""))
    decision_index = metadata.get("decision_index")
    history = metadata.get("history")
    if not session_id or not observation_id:
        raise ValueError("session_id and observation_id are required")
    if not isinstance(decision_index, int) or isinstance(decision_index, bool):
        raise ValueError("decision_index must be an integer")
    if not isinstance(history, list):
        raise ValueError("history must be an array")

    base: dict[str, Any] = {
        "schema_version": 2,
        "response_type": "end2end_decision",
        "session_id": session_id,
        "observation_id": observation_id,
        "progress_analysis": "Deterministic Isaac BACKTRACK integration test.",
    }
    if decision_index == 0:
        base.update(
            action="NAVIGATE",
            direction="forward",
            target="dark landscape photograph",
            bbox_2d=[261, 148, 420, 227],
            waypoint=None,
            reasoning="Scripted first forward segment.",
        )
    elif decision_index == 1:
        base.update(
            action="NAVIGATE",
            direction="forward",
            target="dark landscape photograph",
            bbox_2d=[95, 0, 427, 308],
            waypoint=None,
            reasoning="Scripted second forward segment.",
        )
    elif decision_index == 2:
        if len(history) < 2:
            raise ValueError("decision 2 requires two completed waypoints")
        base.update(
            action="BACKTRACK",
            direction=None,
            target=None,
            bbox_2d=None,
            waypoint=0,
            reasoning="Scripted return to the first decision pose.",
        )
    else:
        raise ValueError("script is complete; run with --local_max_decisions 3")
    return base


class _Handler(BaseHTTPRequestHandler):
    server_version = "ScriptedBacktrackHTTP/1"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "service": "scripted_backtrack"})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != DECISION_PATH:
            self._send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024 * 1024:
                raise ValueError("invalid request size")
            metadata = _metadata_from_multipart(
                self.rfile.read(length), self.headers.get("Content-Type", "")
            )
            payload = _scripted_response(metadata)
        except ValueError as exc:
            self._send_json(400, {"error": "invalid_request", "message": str(exc)})
            return
        print(
            "[SCRIPTED-BACKTRACK] "
            f"decision={metadata['decision_index']} action={payload['action']} "
            f"history={len(metadata['history'])}"
        )
        self._send_json(200, payload)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18766)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(
        "[SCRIPTED-BACKTRACK] listening at "
        f"http://{args.host}:{args.port}{DECISION_PATH}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
