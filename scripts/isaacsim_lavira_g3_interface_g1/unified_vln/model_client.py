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
from .odometry import Pose2D
from .types import DIRECTION_ORDER, PanoramaBundle


@dataclass(frozen=True)
class CompletedWaypoint:
    """一次已由机器人实际完成的 NAVIGATE 动作记录。

    模型下一次决策时会看到这条文字历史；为了控制请求大小，只有最近若干条记录
    会附带动作前的前视图和所选方向图，较旧记录仍保留文字描述。
    """

    waypoint_id: int
    decision_step: int
    direction: str
    target: str
    init_rgb: np.ndarray
    direction_rgb: np.ndarray
    # These fields stay local.  The server continues to receive only the
    # schema-v2 history entry and images built by ``history_entry``.
    decision_pose: Pose2D | None = None
    arrival_pose: Pose2D | None = None
    executed_world_path_xy: np.ndarray | None = None

    def history_entry(
        self,
        *,
        include_images: bool,
        wire_waypoint_id: int | None = None,
    ) -> NavigationHistoryEntry:
        """生成协议中的历史元数据，并按需填写对应图片字段名。"""

        waypoint_id = (
            int(self.waypoint_id)
            if wire_waypoint_id is None
            else int(wire_waypoint_id)
        )
        prefix = f"history_{waypoint_id}"
        return NavigationHistoryEntry(
            waypoint_id=waypoint_id,
            step=int(self.decision_step),
            turn_action=f"turn {self.direction}",
            description=self.target,
            init_image_field=f"{prefix}_init" if include_images else None,
            dir_image_field=f"{prefix}_dir" if include_images else None,
        )


def select_model_history_records(
    records: list[CompletedWaypoint] | tuple[CompletedWaypoint, ...],
    *,
    max_waypoints: int | None = None,
) -> tuple[CompletedWaypoint, ...]:
    """Select the exact local records represented by wire waypoint ids."""

    checked = tuple(records)
    if max_waypoints is not None and max_waypoints <= 0:
        raise ValueError("History max waypoints must be >0 when configured.")
    for index, record in enumerate(checked):
        if record.waypoint_id != index:
            raise ValueError("Completed waypoint ids must be contiguous and zero-based.")
    return (
        checked if max_waypoints is None else checked[-max_waypoints:]
    )


def build_model_history(
    records: list[CompletedWaypoint] | tuple[CompletedWaypoint, ...],
    *,
    max_waypoints: int | None = None,
) -> tuple[tuple[NavigationHistoryEntry, ...], dict[str, np.ndarray]]:
    """把已完成路点转换为模型协议需要的历史条目和图片字段。

    返回值第一项是文字历史，第二项只包含协议允许的最近几组图片。默认保留全部
    文字历史；配置 ``max_waypoints`` 后只发送最近 N 项。内部完整记录仍保持原编号，
    截取后的 wire history 会从 0 重新编号，以满足 schema-v2 的连续编号要求。
    """

    selected_records = select_model_history_records(
        records, max_waypoints=max_waypoints
    )
    # 只从 image_start 开始附图；更早的记录通过纯文字保留上下文。
    image_start = max(
        0, len(selected_records) - LAVIRA_MAX_HISTORY_IMAGE_WAYPOINTS
    )
    entries: list[NavigationHistoryEntry] = []
    images: dict[str, np.ndarray] = {}
    for wire_index, record in enumerate(selected_records):
        entry = record.history_entry(
            include_images=wire_index >= image_start,
            wire_waypoint_id=wire_index,
        )
        entries.append(entry)
        if entry.has_images:
            images[entry.init_image_field] = np.asarray(record.init_rgb).copy()
            images[entry.dir_image_field] = np.asarray(record.direction_rgb).copy()
    return tuple(entries), images


def encode_rgb_png(rgb: np.ndarray) -> bytes:
    """按线上协议把 RGB uint8 原分辨率图像无损编码为 PNG 字节。"""
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
    """严格遵循 schema-v2 的组合模型 HTTP 客户端。

    请求使用 multipart/form-data：一个 JSON 元数据字段加若干 PNG 图片字段。
    客户端在发送前后都执行协议校验，尽早发现字段名、图片和响应不匹配的问题。
    """

    def __init__(
        self,
        server_url: str,
        timeout_s: float = 90.0,
        *,
        send_instruction: bool = True,
    ):
        """验证并保存模型服务地址和单次请求超时。"""

        if not server_url.strip():
            raise ValueError("Model server URL must not be empty.")
        if timeout_s <= 0.0:
            raise ValueError("Model timeout must be positive.")
        if not isinstance(send_instruction, bool):
            raise ValueError("send_instruction must be a boolean.")
        self.server_url = server_url
        self.timeout_s = float(timeout_s)
        self.send_instruction = send_instruction

    @staticmethod
    def make_request(
        bundle: PanoramaBundle,
        *,
        session_id: str,
        instruction: str,
        decision_index: int,
        history: tuple[NavigationHistoryEntry, ...] = (),
    ) -> NavigationDecisionRequest:
        """根据全景观察、导航指令和历史记录构造一次决策元数据。"""

        bundle.validated()
        first = bundle.views[DIRECTION_ORDER[0]]
        image_height, image_width = first.rgb.shape[:2]
        # 字典值是 multipart 中实际使用的图片字段名，而不是图片内容。
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
        """收集历史图和当前四视图，并编码成协议要求的 PNG 字段字典。"""

        bundle.validated()
        history_images = {} if history_images is None else history_images
        # 元数据声明了哪些历史图片，就必须不多不少地提供这些字段。
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
    ) -> tuple[NavigationDecisionResponse | None, dict]:
        """发送模型请求，校验并返回结构化响应及原始 JSON 字典。"""

        body, boundary = self._multipart_body(
            request_metadata,
            images,
            include_instruction=self.send_instruction,
        )
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
        # 最多读取 1 MiB + 1 字节，用额外的 1 字节判断响应是否超限。
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
        # Recovery SAFE_STOP is a terminal G3 control response, not a normal
        # Navigator STOP. Its visual navigation fields are intentionally null,
        # so it must reach the G3 session validator before the legacy Navigator
        # response class enforces direction/target/bbox requirements.
        control = payload.get("control")
        if isinstance(control, str) and control.upper() == "SAFE_STOP":
            if payload.get("schema_version") != 2:
                raise ValueError("SAFE_STOP response schema_version is not 2.")
            if payload.get("response_type") != "end2end_decision":
                raise ValueError(
                    "SAFE_STOP response_type must be 'end2end_decision'."
                )
            if payload.get("session_id") != request_metadata.session_id:
                raise ValueError("SAFE_STOP response session_id mismatch.")
            if payload.get("observation_id") != request_metadata.observation_id:
                raise ValueError("SAFE_STOP response observation_id mismatch.")
            return None, payload
        # 除了 JSON 格式，还要验证响应是否确实对应本次请求。
        parsed = NavigationDecisionResponse.from_dict(payload)
        parsed.validate_matches(request_metadata)
        return parsed, payload

    @staticmethod
    def _validate_images(
        request: NavigationDecisionRequest,
        images: Mapping[str, bytes],
    ) -> None:
        """检查图片字段集合完全匹配协议，且每个值都是真正的 PNG 字节。"""

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
        *,
        include_instruction: bool = True,
    ) -> tuple[bytes, str]:
        """手工组装 multipart 请求体，返回二进制正文和随机 boundary。"""

        cls._validate_images(request, images)
        boundary = f"----IsaacLaViRA{uuid.uuid4().hex}"
        chunks: list[bytes] = []

        def append_part(headers: list[str], payload: bytes) -> None:
            """向请求体追加一个符合 MIME 格式的表单分段。"""

            chunks.append(f"--{boundary}\r\n".encode("ascii"))
            for header in headers:
                chunks.append(f"{header}\r\n".encode("utf-8"))
            chunks.append(b"\r\n")
            chunks.append(payload)
            chunks.append(b"\r\n")

        metadata_payload = request.to_metadata()
        if not include_instruction:
            metadata_payload.pop("instruction", None)
        metadata = json.dumps(
            metadata_payload, ensure_ascii=False, separators=(",", ":")
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
    """生成便于落盘调试的响应字典，并可附加投影目标及其距离。"""

    result = response.to_dict()
    if projected_goal_xy is not None:
        goal = np.asarray(projected_goal_xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(goal)):
            raise ValueError("Projected goal must be finite.")
        result["goal_after_turn_robot_xy_m"] = goal.tolist()
        result["goal_distance_m"] = float(math.hypot(goal[0], goal[1]))
    return result
