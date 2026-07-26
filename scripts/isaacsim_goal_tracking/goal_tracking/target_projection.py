from __future__ import annotations

"""把 LaViRA 的方向与二维 bbox 投影为 Isaac Sim 中的三维语义目标。

像素和深度选择刻意对齐 ``lavira_code`` 的现有执行路径：使用 bbox 底边中点，
依次检查 3/5/7/9 像素邻域，并取第一个含有效值邻域的深度中位数。与 Habitat
实现不同之处只有坐标变换：这里使用 FrameBundle 中该相机的完整内参与外参，
而不是用二维 agent heading 近似。

本模块只产生、保存和显示语义表面点。它不生成速度命令，也不把物体表面点直接
当作可行走目标；安全落点、occupancy map 和 FMM 由后续处理模块完成。
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .frame_bundle import FrameBundle
from .lavira_protocol import NavigationDecisionRequest, NavigationDecisionResponse


LAVIRA_DEPTH_WINDOW_SIZES = (3, 5, 7, 9)


@dataclass(frozen=True)
class TargetProjection:
    """一次 bbox-depth 反投影的完整可审计结果。"""

    observation_id: str
    bundle_id: int
    sim_step: int
    action: str
    direction: str
    target: str
    camera_id: str
    sensor_frame_id: int
    bbox_response: tuple[float, float, float, float]
    bbox_clipped: tuple[int, int, int, int]
    selected_pixel_uv: tuple[int, int]
    depth_window_size: int
    depth_window_xyxy_exclusive: tuple[int, int, int, int]
    valid_depth_count: int
    depth_window_pixel_count: int
    valid_depth_fraction: float
    valid_depth_min_m: float
    selected_depth_median_m: float
    valid_depth_max_m: float
    point_camera_ros_m: np.ndarray
    point_base_m: np.ndarray
    point_world_m: np.ndarray
    horizontal_distance_base_m: float
    bearing_base_rad: float
    K: np.ndarray
    T_base_camera: np.ndarray
    T_world_camera_ros: np.ndarray

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "projection_type": "lavira_bbox_depth_semantic_surface_point",
            "source_algorithm": (
                "lavira_code bbox bottom-center + first valid 3/5/7/9 window "
                "depth median"
            ),
            "motion_goal": False,
            "observation_id": self.observation_id,
            "bundle_id": self.bundle_id,
            "sim_step": self.sim_step,
            "action": self.action,
            "direction": self.direction,
            "target": self.target,
            "camera_id": self.camera_id,
            "sensor_frame_id": self.sensor_frame_id,
            "bbox_response": list(self.bbox_response),
            "bbox_clipped": list(self.bbox_clipped),
            "selected_pixel_uv": list(self.selected_pixel_uv),
            "depth_sampling": {
                "window_sizes_tried": list(LAVIRA_DEPTH_WINDOW_SIZES),
                "selected_window_size": self.depth_window_size,
                "window_xyxy_exclusive": list(self.depth_window_xyxy_exclusive),
                "valid_depth_count": self.valid_depth_count,
                "window_pixel_count": self.depth_window_pixel_count,
                "valid_depth_fraction": self.valid_depth_fraction,
                "valid_depth_min_m": self.valid_depth_min_m,
                "selected_depth_median_m": self.selected_depth_median_m,
                "valid_depth_max_m": self.valid_depth_max_m,
            },
            "camera_frame_convention": "ROS optical: +X right, +Y down, +Z forward",
            "base_frame_axes": "navigation forward=+X, left=+Y, up=+Z",
            "point_camera_ros_m": self.point_camera_ros_m.tolist(),
            "point_base_m": self.point_base_m.tolist(),
            "point_world_m": self.point_world_m.tolist(),
            "horizontal_distance_base_m": self.horizontal_distance_base_m,
            "bearing_base_rad": self.bearing_base_rad,
            "K": self.K.tolist(),
            "T_base_camera": self.T_base_camera.tolist(),
            "T_world_camera_ros": self.T_world_camera_ros.tolist(),
            "note": (
                "This is a depth-derived semantic surface point, not yet a collision-free "
                "robot destination. A traversability map and stand-off rule must choose the "
                "later motion goal."
            ),
        }


@dataclass(frozen=True)
class _DepthSample:
    depth_m: float
    window_size: int
    window_xyxy_exclusive: tuple[int, int, int, int]
    valid_count: int
    pixel_count: int
    valid_fraction: float
    valid_min_m: float
    valid_max_m: float


def project_navigation_target(
    request: NavigationDecisionRequest,
    bundle: FrameBundle,
    response: NavigationDecisionResponse,
    *,
    min_depth_m: float,
    max_depth_m: float,
    window_sizes: Iterable[int] = LAVIRA_DEPTH_WINDOW_SIZES,
) -> TargetProjection:
    """将 NAVIGATE/STOP 的 bbox 反投影到 camera/base/world 三个坐标系。"""
    if int(bundle.bundle_id) != int(request.bundle_id):
        raise ValueError(
            f"FrameBundle id {bundle.bundle_id} does not match request "
            f"bundle_id {request.bundle_id}."
        )
    response.validate_matches(request)
    action = response.action.upper()
    if action == "BACKTRACK":
        raise ValueError("BACKTRACK has no bbox target to project.")
    if not 0.0 < float(min_depth_m) < float(max_depth_m):
        raise ValueError(
            f"Invalid target projection depth range ({min_depth_m}, {max_depth_m})."
        )

    frame = bundle.views[response.direction]
    image_height, image_width = frame.depth_z_m.shape
    if (image_width, image_height) != (request.image_width, request.image_height):
        raise ValueError(
            "Selected FrameBundle depth resolution does not match request metadata: "
            f"{image_width}x{image_height} != "
            f"{request.image_width}x{request.image_height}."
        )

    bbox_clipped = response.clipped_bbox(image_width, image_height)
    x1, _y1, x2, y2 = bbox_clipped
    # This is the same target-pixel convention used by lavira_main.py.  Clipping
    # y2 first fixes its edge case where a model may legally return y2 == height.
    selected_pixel_uv = (int((x1 + x2) / 2.0), int(y2))
    sample = _sample_lavira_depth(
        frame.depth_z_m,
        selected_pixel_uv,
        min_depth_m=float(min_depth_m),
        max_depth_m=float(max_depth_m),
        window_sizes=window_sizes,
    )

    u, v = selected_pixel_uv
    pixel_h = np.array([float(u), float(v), 1.0], dtype=np.float64)
    try:
        camera_ray = np.linalg.solve(np.asarray(frame.K, dtype=np.float64), pixel_h)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Selected camera intrinsic matrix K is singular.") from exc
    point_camera = camera_ray * sample.depth_m
    point_base = _transform_point(frame.T_base_camera, point_camera)
    point_world = _transform_point(frame.T_world_camera_ros, point_camera)

    expected_world_camera = (
        np.asarray(bundle.T_world_base, dtype=np.float64)
        @ np.asarray(frame.T_base_camera, dtype=np.float64)
    )
    if not np.allclose(
        expected_world_camera,
        np.asarray(frame.T_world_camera_ros, dtype=np.float64),
        rtol=1.0e-6,
        atol=1.0e-6,
    ):
        raise ValueError(
            "FrameBundle transforms are inconsistent: "
            "T_world_base @ T_base_camera != T_world_camera_ros."
        )
    point_world_via_base = _transform_point(bundle.T_world_base, point_base)
    if not np.allclose(point_world_via_base, point_world, rtol=1.0e-6, atol=1.0e-6):
        raise ValueError("Projected camera/base/world target transforms are inconsistent.")

    return TargetProjection(
        observation_id=request.observation_id,
        bundle_id=int(bundle.bundle_id),
        sim_step=int(bundle.sim_step),
        action=action,
        direction=str(response.direction),
        target=str(response.target),
        camera_id=frame.camera_id,
        sensor_frame_id=int(frame.sensor_frame_id),
        bbox_response=tuple(float(value) for value in response.bbox_2d),
        bbox_clipped=bbox_clipped,
        selected_pixel_uv=selected_pixel_uv,
        depth_window_size=sample.window_size,
        depth_window_xyxy_exclusive=sample.window_xyxy_exclusive,
        valid_depth_count=sample.valid_count,
        depth_window_pixel_count=sample.pixel_count,
        valid_depth_fraction=sample.valid_fraction,
        valid_depth_min_m=sample.valid_min_m,
        selected_depth_median_m=sample.depth_m,
        valid_depth_max_m=sample.valid_max_m,
        point_camera_ros_m=point_camera,
        point_base_m=point_base,
        point_world_m=point_world,
        horizontal_distance_base_m=float(np.linalg.norm(point_base[:2])),
        bearing_base_rad=float(math.atan2(point_base[1], point_base[0])),
        K=np.asarray(frame.K, dtype=np.float64).copy(),
        T_base_camera=np.asarray(frame.T_base_camera, dtype=np.float64).copy(),
        T_world_camera_ros=np.asarray(frame.T_world_camera_ros, dtype=np.float64).copy(),
    )


def save_target_projection_debug(
    output_dir: Path,
    bundle: FrameBundle,
    projection: TargetProjection,
    *,
    near_m: float,
    far_m: float,
) -> dict[str, str]:
    """保存原始选中 depth、bbox/采样点标注和完整投影 JSON。"""
    import cv2

    output_dir = Path(output_dir)
    frame = bundle.views[projection.direction]
    json_name = "target_projection.json"
    rgb_name = "target_projection_rgb.png"
    depth_name = "target_projection_depth_m.npy"
    depth_preview_name = "target_projection_depth_preview.png"

    (output_dir / json_name).write_text(
        json.dumps(projection.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(output_dir / depth_name, frame.depth_z_m, allow_pickle=False)

    rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    rgb_annotated = _draw_projection_overlay(rgb_bgr, projection)
    if not cv2.imwrite(str(output_dir / rgb_name), rgb_annotated):
        raise OSError(f"Failed to write {output_dir / rgb_name}.")

    depth_preview = _metric_depth_preview(frame.depth_z_m, near_m, far_m)
    depth_annotated = _draw_projection_overlay(depth_preview, projection)
    if not cv2.imwrite(str(output_dir / depth_preview_name), depth_annotated):
        raise OSError(f"Failed to write {output_dir / depth_preview_name}.")

    return {
        "projection_json": json_name,
        "annotated_rgb": rgb_name,
        "metric_depth_npy": depth_name,
        "annotated_depth_preview": depth_preview_name,
    }


def draw_target_projection_marker(projection: TargetProjection) -> None:
    """在 GUI stage 中画一次性的紫色语义表面点；失败不影响投影结果。"""
    try:
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("no USD stage")
        root_path = "/World/LaViRATargetProjection"
        if stage.GetPrimAtPath(root_path):
            stage.RemovePrim(root_path)
        UsdGeom.Xform.Define(stage, root_path)
        sphere = UsdGeom.Sphere.Define(stage, f"{root_path}/SemanticSurfacePoint")
        sphere.CreateRadiusAttr(0.08)
        UsdGeom.XformCommonAPI(sphere).SetTranslate(
            Gf.Vec3d(*(float(value) for value in projection.point_world_m))
        )
        sphere.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 1.0)])
        print(
            "[LAVIRA] Drew magenta semantic target marker at world="
            f"{_format_xyz(projection.point_world_m)}."
        )
    except Exception as exc:
        print(f"[WARN] Could not draw LaViRA target projection marker: {exc}")


def _sample_lavira_depth(
    depth_z_m: np.ndarray,
    pixel_uv: tuple[int, int],
    *,
    min_depth_m: float,
    max_depth_m: float,
    window_sizes: Iterable[int],
) -> _DepthSample:
    """复现 LaViRA 的逐级邻域中位数策略，但不制造虚假 fallback 目标。"""
    depth = np.asarray(depth_z_m)
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be HxW, got shape={depth.shape}.")
    height, width = depth.shape
    u, v = (int(pixel_uv[0]), int(pixel_uv[1]))
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError(f"Target pixel {(u, v)} is outside {width}x{height} depth image.")

    tried: list[int] = []
    for raw_size in window_sizes:
        size = int(raw_size)
        if size <= 0 or size % 2 == 0:
            raise ValueError(f"Depth window size must be a positive odd integer, got {size}.")
        tried.append(size)
        half = size // 2
        x1 = max(0, u - half)
        x2 = min(width, u + half + 1)
        y1 = max(0, v - half)
        y2 = min(height, v + half + 1)
        patch = np.asarray(depth[y1:y2, x1:x2], dtype=np.float64)
        valid_mask = (
            np.isfinite(patch)
            & (patch >= float(min_depth_m))
            & (patch <= float(max_depth_m))
        )
        valid = patch[valid_mask]
        if valid.size == 0:
            continue
        return _DepthSample(
            depth_m=float(np.median(valid)),
            window_size=size,
            window_xyxy_exclusive=(x1, y1, x2, y2),
            valid_count=int(valid.size),
            pixel_count=int(patch.size),
            valid_fraction=float(valid.size / patch.size),
            valid_min_m=float(valid.min()),
            valid_max_m=float(valid.max()),
        )

    raise ValueError(
        f"No finite depth in [{min_depth_m}, {max_depth_m}] m around pixel "
        f"{(u, v)} for LaViRA windows {tried}; no fallback goal was fabricated."
    )


def _transform_point(transform: np.ndarray, point_xyz: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    point_xyz = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Point transform must be a finite 4x4 matrix.")
    result_h = transform @ np.append(point_xyz, 1.0)
    if not np.all(np.isfinite(result_h)) or abs(float(result_h[3])) < 1.0e-12:
        raise ValueError("Point transform produced invalid homogeneous coordinates.")
    return result_h[:3] / result_h[3]


def _draw_projection_overlay(image_bgr: np.ndarray, projection: TargetProjection) -> np.ndarray:
    import cv2

    result = np.asarray(image_bgr).copy()
    x1, y1, x2, y2 = projection.bbox_clipped
    wx1, wy1, wx2, wy2 = projection.depth_window_xyxy_exclusive
    u, v = projection.selected_pixel_uv
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(result, (wx1, wy1), (wx2 - 1, wy2 - 1), (255, 255, 0), 1)
    cv2.drawMarker(
        result,
        (u, v),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
    )
    label = (
        f"{projection.direction} {projection.target} "
        f"z={projection.selected_depth_median_m:.2f}m"
    )
    text_y = max(22, y1 - 8)
    cv2.putText(
        result,
        label,
        (max(4, x1), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def _metric_depth_preview(depth_z_m: np.ndarray, near_m: float, far_m: float) -> np.ndarray:
    import cv2

    depth = np.asarray(depth_z_m, dtype=np.float32)
    valid = (
        np.isfinite(depth)
        & (depth >= float(near_m))
        & (depth <= float(far_m))
    )
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        scaled = 1.0 - (
            np.clip(depth[valid], near_m, far_m) - float(near_m)
        ) / max(float(far_m) - float(near_m), 1.0e-6)
        normalized[valid] = np.round(scaled * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def _format_xyz(value: np.ndarray) -> str:
    xyz = np.asarray(value, dtype=np.float64).reshape(3)
    return f"({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f})"
