from __future__ import annotations

"""可复用的单轮 FrameBundle -> HTTP 决策 -> 投影/地图/FMM 处理链。"""

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

import numpy as np

from .camera import FOUR_VIEW_DIRECTIONS
from .frame_bundle import FourViewCameraRig, FrameBundle
from .fmm_planner import (
    FMMPlan,
    build_fmm_plan,
    fmm_planner_config_from_args,
    save_fmm_plan_debug,
)
from .lavira_protocol import (
    NavigationDecisionRequest,
    NavigationDecisionResponse,
    NavigationHistoryEntry,
)
from .navigation_mapping import (
    NavigationGridMap,
    build_navigation_grid_map,
    navigation_map_config_from_args,
    save_navigation_map_debug,
)
from .target_projection import (
    TargetProjection,
    draw_target_projection_marker,
    project_navigation_target,
    save_target_projection_debug,
)


_SAFE_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_path_component(value: str) -> str:
    result = _SAFE_PATH_COMPONENT.sub("_", value.strip()).strip("._")
    if not result:
        raise ValueError(f"Cannot turn {value!r} into a safe output path component.")
    return result


def encode_rgb_png(rgb: np.ndarray) -> bytes:
    """将 Isaac RGB uint8 数组无损编码为 PNG；不写标注、不改变分辨率。"""
    import cv2

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape={rgb.shape}.")
    if rgb.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB image, got dtype={rgb.dtype}.")
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("OpenCV failed to encode RGB image as PNG.")
    return encoded.tobytes()


def make_navigation_decision_request(
    bundle: FrameBundle,
    *,
    session_id: str,
    instruction: str,
    decision_index: int,
    history: tuple[NavigationHistoryEntry, ...] = (),
) -> NavigationDecisionRequest:
    """从当前 FrameBundle 和已完成 history 构造 schema v2 metadata。"""
    first_frame = bundle.views[FOUR_VIEW_DIRECTIONS[0]]
    image_height, image_width = first_frame.rgb.shape[:2]
    for direction in FOUR_VIEW_DIRECTIONS:
        frame = bundle.views[direction]
        if frame.rgb.shape[:2] != (image_height, image_width):
            raise ValueError(
                "All panorama images must share one resolution, but "
                f"{direction} has {frame.rgb.shape[:2]} instead of "
                f"{(image_height, image_width)}."
            )

    observation_id = f"{session_id}_decision_{int(decision_index):03d}"
    panorama = {
        direction: f"current_{direction}" for direction in FOUR_VIEW_DIRECTIONS
    }
    return NavigationDecisionRequest(
        session_id=session_id,
        observation_id=observation_id,
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


def make_first_navigation_decision_request(
    bundle: FrameBundle,
    *,
    session_id: str,
    instruction: str,
) -> NavigationDecisionRequest:
    """为 decision 0 构造请求；第一轮 history 必须为空。"""
    return make_navigation_decision_request(
        bundle,
        session_id=session_id,
        instruction=instruction,
        decision_index=0,
        history=(),
    )


def panorama_png_fields(
    bundle: FrameBundle,
    request: NavigationDecisionRequest,
    history_images: Mapping[str, np.ndarray | bytes] | None = None,
) -> dict[str, bytes]:
    """编码最近四个历史点和当前四方向 RGB 的 multipart 字段。"""
    images: dict[str, bytes] = {}
    history_images = {} if history_images is None else history_images
    expected_history_fields: list[str] = []
    for entry in request.history:
        if entry.has_images:
            expected_history_fields.extend(
                (entry.init_image_field, entry.dir_image_field)
            )
    if set(history_images) != set(expected_history_fields):
        raise ValueError(
            "History image field mismatch: "
            f"missing={sorted(set(expected_history_fields) - set(history_images))}, "
            f"extra={sorted(set(history_images) - set(expected_history_fields))}."
        )
    for field_name in expected_history_fields:
        value = history_images[field_name]
        images[field_name] = value if isinstance(value, bytes) else encode_rgb_png(value)

    for direction in FOUR_VIEW_DIRECTIONS:
        field_name = request.current_panorama[direction]
        images[field_name] = encode_rgb_png(bundle.views[direction].rgb)
    _validate_image_field_set(request, images)
    return images


def _validate_image_field_set(
    request: NavigationDecisionRequest, images: Mapping[str, bytes]
) -> None:
    expected = set(request.required_image_fields)
    actual = set(images)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"Navigation image field mismatch: missing={missing}, extra={extra}."
        )
    for field_name, payload in images.items():
        if not isinstance(payload, bytes) or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Image field {field_name!r} is not a PNG byte payload.")


def save_navigation_decision_request(
    output_dir: Path,
    request: NavigationDecisionRequest,
    images: Mapping[str, bytes],
) -> None:
    """保存与 multipart 内容一一对应的离线请求副本。"""
    _validate_image_field_set(request, images)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(request.to_metadata(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for field_name in request.required_image_fields:
        (output_dir / f"{field_name}.png").write_bytes(images[field_name])


def _multipart_body(
    request: NavigationDecisionRequest,
    images: Mapping[str, bytes],
) -> tuple[bytes, str]:
    _validate_image_field_set(request, images)
    boundary = f"----IsaacLaViRA{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    def append_part(headers: list[str], payload: bytes) -> None:
        chunks.append(f"--{boundary}\r\n".encode("ascii"))
        for header in headers:
            chunks.append(f"{header}\r\n".encode("utf-8"))
        chunks.append(b"\r\n")
        chunks.append(payload)
        chunks.append(b"\r\n")

    metadata_bytes = json.dumps(
        request.to_metadata(), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    append_part(
        [
            'Content-Disposition: form-data; name="metadata"',
            "Content-Type: application/json; charset=utf-8",
        ],
        metadata_bytes,
    )
    for field_name in request.required_image_fields:
        append_part(
            [
                (
                    "Content-Disposition: form-data; "
                    f'name="{field_name}"; filename="{field_name}.png"'
                ),
                "Content-Type: image/png",
            ],
            images[field_name],
        )
    chunks.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), boundary


def post_navigation_decision(
    server_url: str,
    request_metadata: NavigationDecisionRequest,
    images: Mapping[str, bytes],
    *,
    timeout_seconds: float,
) -> tuple[NavigationDecisionResponse, dict]:
    """发送 multipart 请求并校验 schema v2 归一化响应。"""
    body, boundary = _multipart_body(request_metadata, images)
    http_request = Request(
        server_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(http_request, timeout=float(timeout_seconds)) as response:
            response_bytes = response.read(1024 * 1024 + 1)
    except HTTPError as exc:
        error_body = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LaViRA mock server returned HTTP {exc.code}: {error_body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach LaViRA server at {server_url!r}: {exc.reason}"
        ) from exc

    if len(response_bytes) > 1024 * 1024:
        raise RuntimeError("LaViRA server response exceeded 1 MiB.")
    try:
        response_payload = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("LaViRA server did not return valid UTF-8 JSON.") from exc
    if not isinstance(response_payload, dict):
        raise RuntimeError("LaViRA server JSON response must be an object.")

    parsed = NavigationDecisionResponse.from_dict(response_payload)
    parsed.validate_matches(request_metadata)
    return parsed, response_payload


def interpret_navigation_decision(
    request: NavigationDecisionRequest,
    bundle: FrameBundle,
    response: NavigationDecisionResponse,
) -> dict:
    """把模型字段绑定到发出该请求时的 FrameBundle；不产生运动命令。"""
    if int(bundle.bundle_id) != request.bundle_id:
        raise ValueError(
            f"FrameBundle id {bundle.bundle_id} does not match request "
            f"bundle_id {request.bundle_id}."
        )
    response.validate_matches(request)
    action = response.action.upper()
    result = {
        "observation_id": request.observation_id,
        "bundle_id": request.bundle_id,
        "action": action,
        "normalized_response": response.to_dict(),
    }
    if action == "BACKTRACK":
        result.update(
            {
                "used_fields": ["action", "waypoint"],
                "ignored_fields": ["direction", "target", "bbox_2d"],
                "history_waypoint": response.waypoint,
            }
        )
        return result

    frame = bundle.views[response.direction]
    image_height, image_width = frame.rgb.shape[:2]
    if (image_width, image_height) != (
        request.image_width,
        request.image_height,
    ):
        raise ValueError(
            "Selected FrameBundle view resolution does not match request metadata."
        )
    bbox_clipped = response.clipped_bbox(image_width, image_height)
    result.update(
        {
            "used_fields": ["action", "direction", "target", "bbox_2d"],
            "ignored_fields": ["waypoint"],
            "selected_direction": response.direction,
            "selected_image_field": request.current_panorama[response.direction],
            "selected_camera_id": frame.camera_id,
            "selected_sensor_frame_id": frame.sensor_frame_id,
            "target": response.target,
            "bbox_response": list(response.bbox_2d),
            "bbox_clipped": list(bbox_clipped),
            "final_approach": action == "STOP",
        }
    )
    return result


class NavigationDecisionOfflineProbe:
    """执行一轮真实抓图、离线导出和 HTTP 往返。

    类名保留 ``OfflineProbe`` 是为了兼容已有调用。它既服务于一次性只读
    ``--lavira_decision_probe``，也由 episode controller 为每轮显式传入
    decision_index/history，从而复用完全相同的 schema、投影、地图和 FMM
    代码，而不复制另一条接口链。
    """

    def __init__(
        self,
        args_cli,
        *,
        enabled: bool | None = None,
        decision_index: int = 0,
        history: tuple[NavigationHistoryEntry, ...] = (),
        history_images: Mapping[str, np.ndarray | bytes] | None = None,
        run_id: str | None = None,
        execution_requested: bool | None = None,
    ):
        self.enabled = (
            bool(getattr(args_cli, "lavira_decision_probe", False))
            if enabled is None
            else bool(enabled)
        )
        self.args_cli = args_cli
        self.decision_index = int(decision_index)
        self.history = tuple(history)
        self.history_images = (
            {} if history_images is None else dict(history_images)
        )
        self.execution_requested = (
            bool(getattr(args_cli, "lavira_execute_fmm_path", False))
            if execution_requested is None
            else bool(execution_requested)
        )
        self.attempted = False
        self.completed = False
        self.output_dir: Path | None = None
        self.bundle: FrameBundle | None = None
        self.request: NavigationDecisionRequest | None = None
        self.images: dict[str, bytes] | None = None
        self.response: NavigationDecisionResponse | None = None
        self.target_projection: TargetProjection | None = None
        self.navigation_map: NavigationGridMap | None = None
        self.fmm_plan: FMMPlan | None = None
        self._run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if run_id is None
            else str(run_id)
        )

    def maybe_run(
        self,
        camera_rig: FourViewCameraRig | None,
        *,
        completed_step: int,
        step_dt: float,
    ) -> None:
        if not self.enabled or self.attempted:
            return
        if completed_step < int(self.args_cli.lavira_decision_warmup_steps):
            return
        self.attempted = True
        if camera_rig is None:
            raise RuntimeError("LaViRA decision probe requires FourViewCameraRig.")

        bundle = camera_rig.capture(
            sim_step=int(completed_step),
            timestamp=float(completed_step) * float(step_dt),
        )
        self.bundle = bundle
        request_metadata = make_navigation_decision_request(
            bundle,
            session_id=str(self.args_cli.lavira_session_id),
            instruction=str(self.args_cli.instruction),
            decision_index=self.decision_index,
            history=self.history,
        )
        images = panorama_png_fields(
            bundle,
            request_metadata,
            self.history_images,
        )
        self.request = request_metadata
        self.images = images
        output_dir = (
            Path(self.args_cli.lavira_output_dir)
            / f"run_{self._run_id}"
            / _safe_path_component(request_metadata.session_id)
            / _safe_path_component(request_metadata.observation_id)
        )
        save_navigation_decision_request(output_dir, request_metadata, images)
        self.output_dir = output_dir
        print(f"[LAVIRA] Saved offline navigation request: {output_dir}")

        parsed, response_payload = post_navigation_decision(
            str(self.args_cli.lavira_server_url),
            request_metadata,
            images,
            timeout_seconds=float(self.args_cli.lavira_timeout),
        )
        (output_dir / "response.json").write_text(
            json.dumps(response_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        interpretation = interpret_navigation_decision(
            request_metadata, bundle, parsed
        )
        projection_error: str | None = None
        if parsed.action.upper() == "BACKTRACK":
            interpretation["target_projection"] = {
                "status": "not_applicable",
                "reason": "BACKTRACK uses only a history waypoint.",
            }
        else:
            try:
                projection = project_navigation_target(
                    request_metadata,
                    bundle,
                    parsed,
                    min_depth_m=float(self.args_cli.rgbd_camera_near),
                    max_depth_m=float(self.args_cli.rgbd_camera_far),
                )
                projection_files = save_target_projection_debug(
                    output_dir,
                    bundle,
                    projection,
                    near_m=float(self.args_cli.rgbd_camera_near),
                    far_m=float(self.args_cli.rgbd_camera_far),
                )
                interpretation["target_projection"] = {
                    "status": "ok",
                    "files": projection_files,
                    "selected_pixel_uv": list(projection.selected_pixel_uv),
                    "selected_depth_median_m": projection.selected_depth_median_m,
                    "point_camera_ros_m": projection.point_camera_ros_m.tolist(),
                    "point_base_m": projection.point_base_m.tolist(),
                    "point_world_m": projection.point_world_m.tolist(),
                    "motion_goal": False,
                }
                self.target_projection = projection
                if bool(self.args_cli.lavira_projection_debug_marker) and not bool(
                    self.args_cli.headless
                ):
                    draw_target_projection_marker(projection)
            except Exception as exc:
                projection_error = str(exc)
                interpretation["target_projection"] = {
                    "status": "error",
                    "error": projection_error,
                    "motion_goal": False,
                }
                (output_dir / "target_projection_error.json").write_text(
                    json.dumps(
                        interpretation["target_projection"],
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
        navigation_map_error: str | None = None
        if bool(getattr(self.args_cli, "lavira_local_map_probe", False)):
            if self.target_projection is None:
                interpretation["navigation_map"] = {
                    "status": "not_run",
                    "reason": "A successful NAVIGATE/STOP target projection is required.",
                    "motion_enabled": False,
                }
            else:
                try:
                    navigation_map = build_navigation_grid_map(
                        bundle,
                        self.target_projection,
                        navigation_map_config_from_args(self.args_cli),
                    )
                    navigation_map_files = save_navigation_map_debug(
                        output_dir, navigation_map
                    )
                    interpretation["navigation_map"] = {
                        "status": "ok",
                        "files": navigation_map_files,
                        "floor_z_world_m": navigation_map.floor_z_world_m,
                        "robot_cell_rc": list(navigation_map.robot_cell_rc),
                        "raw_target_cell_rc": (
                            list(navigation_map.raw_target_cell_rc)
                            if navigation_map.raw_target_cell_rc is not None
                            else None
                        ),
                        "safe_target_cell_rc": (
                            list(navigation_map.safe_target_cell_rc)
                            if navigation_map.safe_target_cell_rc is not None
                            else None
                        ),
                        "safe_target_world_xy": (
                            navigation_map.safe_target_world_xy.tolist()
                            if navigation_map.safe_target_world_xy is not None
                            else None
                        ),
                        "target_selection_strategy": (
                            navigation_map.target_selection_strategy
                        ),
                        "motion_enabled": False,
                    }
                    self.navigation_map = navigation_map
                except Exception as exc:
                    navigation_map_error = str(exc)
                    interpretation["navigation_map"] = {
                        "status": "error",
                        "error": navigation_map_error,
                        "motion_enabled": False,
                    }
                    (output_dir / "navigation_map_error.json").write_text(
                        json.dumps(
                            interpretation["navigation_map"],
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
        fmm_plan_error: str | None = None
        if bool(getattr(self.args_cli, "lavira_fmm_probe", False)):
            if self.navigation_map is None:
                interpretation["fmm_plan"] = {
                    "status": "not_run",
                    "reason": "A successful local navigation map is required.",
                    "motion_enabled": False,
                }
            elif self.navigation_map.safe_target_cell_rc is None:
                interpretation["fmm_plan"] = {
                    "status": "not_run",
                    "reason": "The local navigation map contains no safe target.",
                    "motion_enabled": False,
                }
            else:
                try:
                    fmm_plan = build_fmm_plan(
                        self.navigation_map,
                        fmm_planner_config_from_args(self.args_cli),
                    )
                    fmm_plan_files = save_fmm_plan_debug(
                        output_dir, self.navigation_map, fmm_plan
                    )
                    interpretation["fmm_plan"] = {
                        "status": "ok",
                        "files": fmm_plan_files,
                        "start_cell_rc": list(fmm_plan.start_cell_rc),
                        "goal_cell_rc": list(fmm_plan.goal_cell_rc),
                        "start_distance_m": (
                            fmm_plan.start_distance_cells * fmm_plan.resolution_m
                        ),
                        "path_length_m": fmm_plan.path_length_m,
                        "path_cell_count": int(fmm_plan.path_cells_rc.shape[0]),
                        "waypoint_count": int(fmm_plan.waypoint_cells_rc.shape[0]),
                        "lavira_short_term_goal_cell_rc": list(
                            fmm_plan.lavira_short_term_goal_cell_rc
                        ),
                        "motion_enabled": False,
                    }
                    self.fmm_plan = fmm_plan
                except Exception as exc:
                    fmm_plan_error = str(exc)
                    interpretation["fmm_plan"] = {
                        "status": "error",
                        "error": fmm_plan_error,
                        "motion_enabled": False,
                    }
                    (output_dir / "fmm_plan_error.json").write_text(
                        json.dumps(
                            interpretation["fmm_plan"],
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
        (output_dir / "response_interpretation.json").write_text(
            json.dumps(interpretation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.response = parsed
        self.completed = True
        action = parsed.action.upper()
        if action == "BACKTRACK":
            suffix = f"waypoint={parsed.waypoint}"
        else:
            suffix = (
                f"direction={parsed.direction} target={parsed.target!r} "
                f"bbox_response={list(parsed.bbox_2d)} "
                f"bbox_clipped={interpretation['bbox_clipped']}"
            )
        print(
            "[LAVIRA] Valid navigation response: "
            f"action={action} {suffix} "
            f"observation={request_metadata.observation_id}"
        )
        if action == "STOP":
            print(
                "[LAVIRA] STOP means final approach. A bounded episode controller "
                "may execute this same projected/FMM target before stopping."
            )
        if self.target_projection is not None:
            projection = self.target_projection
            print(
                "[LAVIRA] Projected semantic target: "
                f"pixel={projection.selected_pixel_uv} "
                f"depth={projection.selected_depth_median_m:.3f}m "
                f"window={projection.depth_window_size}x{projection.depth_window_size} "
                f"base=({projection.point_base_m[0]:+.3f},"
                f"{projection.point_base_m[1]:+.3f},"
                f"{projection.point_base_m[2]:+.3f})m "
                f"world=({projection.point_world_m[0]:+.3f},"
                f"{projection.point_world_m[1]:+.3f},"
                f"{projection.point_world_m[2]:+.3f})m."
            )
            print(
                "[LAVIRA] Projection is a semantic surface point only; "
                "it is not a collision-free motion goal."
            )
        elif projection_error is not None:
            print(f"[WARN] LaViRA target projection failed: {projection_error}")
        if self.navigation_map is not None:
            navigation_map = self.navigation_map
            safe_target = (
                "none"
                if navigation_map.safe_target_world_xy is None
                else (
                    f"({navigation_map.safe_target_world_xy[0]:+.3f},"
                    f"{navigation_map.safe_target_world_xy[1]:+.3f})m"
                )
            )
            print(
                "[LAVIRA] Built local navigation map: "
                f"shape={navigation_map.shape} "
                f"resolution={navigation_map.resolution_m:.3f}m "
                f"floor_z={navigation_map.floor_z_world_m:+.3f}m "
                f"traversable={int(np.count_nonzero(navigation_map.traversable))} "
                f"target_strategy={navigation_map.target_selection_strategy} "
                f"safe_target={safe_target}."
            )
            print(
                "[LAVIRA] Map probe only: no robot command was generated."
            )
        elif navigation_map_error is not None:
            print(f"[WARN] LaViRA local navigation map failed: {navigation_map_error}")
        if self.fmm_plan is not None:
            fmm_plan = self.fmm_plan
            print(
                "[LAVIRA] Planned FMM probe path: "
                f"length={fmm_plan.path_length_m:.3f}m "
                f"cells={fmm_plan.path_cells_rc.shape[0]} "
                f"waypoints={fmm_plan.waypoint_cells_rc.shape[0]} "
                f"start={fmm_plan.start_cell_rc} "
                f"goal={fmm_plan.goal_cell_rc} "
                f"short_term_goal={fmm_plan.lavira_short_term_goal_cell_rc}."
            )
            print(
                (
                    "[LAVIRA] FMM path is ready for the runner's guarded locomotion "
                    "handoff; no movement has occurred yet."
                    if self.execution_requested
                    else (
                        "[LAVIRA] FMM probe only: path was not sent to pure-pursuit "
                        "or G1."
                    )
                )
            )
        elif fmm_plan_error is not None:
            print(f"[WARN] LaViRA FMM planning failed: {fmm_plan_error}")
        if self.execution_requested:
            print(
                "[LAVIRA] Decision/planning complete; runner must still pass all "
                "execution safety checks before movement."
            )
        else:
            print("[LAVIRA] Probe only: response will not move the robot.")

    def report_status(self) -> None:
        if not self.enabled or self.completed:
            return
        if not self.attempted:
            print(
                "[WARN] LaViRA decision probe did not run; simulation ended before "
                f"warm-up step {self.args_cli.lavira_decision_warmup_steps}."
            )
