from __future__ import annotations

"""四方向 RGB-D 快照的数据结构、同步抓取和调试落盘。

这个模块只负责本机 Isaac Sim 数据，不包含模型请求。后续 HTTP/episode 逻辑只接收
这里已经复制好的 FrameBundle，从而不会长期引用会被下一次渲染覆盖的 GPU buffer。
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .camera import (
    FOUR_VIEW_DIRECTIONS,
    FOUR_VIEW_PARENT_BODY_NAME,
    FOUR_VIEW_SENSOR_NAMES,
    get_four_view_local_poses,
)


# Columns are the ROS optical axes expressed in the CameraCfg "world"
# convention: x_ros=right=-y_world, y_ros=down=-z_world, z_ros=forward=x_world.
_R_WORLD_CONVENTION_FROM_ROS = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CameraFrame:
    """一台相机在一次四视图快照中的完整本机数据。"""

    camera_id: str
    direction: str
    sensor_frame_id: int
    sim_step: int
    timestamp: float
    rgb: np.ndarray
    depth_z_m: np.ndarray
    K: np.ndarray
    T_world_camera_ros: np.ndarray
    T_base_camera: np.ndarray


@dataclass(frozen=True)
class FrameBundle:
    """同一仿真 step 下四方向相机和机器人位姿的快照。"""

    bundle_id: int
    env_index: int
    sim_step: int
    timestamp: float
    T_world_base: np.ndarray
    views: dict[str, CameraFrame]


class FourViewCameraRig:
    """从 ``raw_env.scene`` 同步抓取四个 IsaacLab Camera sensor。

    四台 Camera prim 使用相对 ``torso_link`` 的固定局部外参，由标准 USD/Fabric
    父子层级随机器人运动。``capture()`` 必须由仿真主线程调用；下游请求与 episode
    逻辑只消费已复制完成的 ``FrameBundle``，不能直接读取 Camera sensor 或调用 render。
    """

    def __init__(self, raw_env, args_cli, env_index: int = 0):
        self.raw_env = raw_env
        self.args_cli = args_cli
        self.env_index = int(env_index)
        self._next_bundle_id = 0
        self._debug_saved = False
        self._debug_attempts = 0
        self._debug_session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._local_poses = get_four_view_local_poses(
            rig_height=float(args_cli.camera_rig_height),
            rig_radius=float(args_cli.camera_rig_radius),
            down_tilt_deg=float(args_cli.camera_down_tilt_deg),
        )

        if self.env_index < 0 or self.env_index >= int(raw_env.num_envs):
            raise ValueError(
                f"Camera env_index {self.env_index} is outside [0, {int(raw_env.num_envs) - 1}]."
            )

        available = set(raw_env.scene.keys())
        missing = [
            sensor_name
            for sensor_name in FOUR_VIEW_SENSOR_NAMES.values()
            if sensor_name not in available
        ]
        if missing:
            raise RuntimeError(
                "Four-view RGB-D sensors are missing from raw_env.scene: "
                f"{missing}. Available scene entities: {sorted(available)}"
            )

        robot = raw_env.scene["robot"]
        body_ids, body_names = robot.find_bodies(
            FOUR_VIEW_PARENT_BODY_NAME, preserve_order=True
        )
        if len(body_ids) != 1 or body_names != [FOUR_VIEW_PARENT_BODY_NAME]:
            raise RuntimeError(
                f"Expected exactly one {FOUR_VIEW_PARENT_BODY_NAME!r} body, "
                f"got ids={body_ids}, names={body_names}."
            )
        self._parent_body_id = int(body_ids[0])

        print(
            "[INFO] FourViewCameraRig ready: "
            f"env_index={self.env_index}, sensors={FOUR_VIEW_SENSOR_NAMES}."
        )

    def capture(self, sim_step: int, timestamp: float) -> FrameBundle:
        """在不推进物理仿真的情况下重渲染并读取同一 step 的四路 RGB-D。

        相机的固定局部外参由 ``torso_link`` 父子层级更新。这里额外 render 一次只为
        刷新当前 physics state 的图像，不再覆盖 Camera 的 world pose。对外相机外参
        由当前 torso tensor 和固定局部外参计算，避免混用 USD 与 Fabric pose cache。
        """
        requested_sim_step = int(sim_step)
        timestamp = float(timestamp)
        common_step_before = int(
            getattr(self.raw_env, "common_step_counter", requested_sim_step)
        )
        if requested_sim_step != common_step_before:
            raise ValueError(
                "Requested camera sim_step does not match raw_env.common_step_counter: "
                f"{requested_sim_step} != {common_step_before}."
            )
        sim_step = common_step_before
        robot = self.raw_env.scene["robot"]

        camera_poses_ros = self._camera_poses_from_current_torso()
        self.raw_env.sim.render()
        # Mark update_period=0 sensors stale and synchronously copy all four
        # annotator buffers produced by the render above.
        for sensor_name in FOUR_VIEW_SENSOR_NAMES.values():
            self.raw_env.scene[sensor_name].update(dt=0.0, force_recompute=True)

        root_pos_w = _indexed_numpy_copy(robot.data.root_link_pos_w, self.env_index)
        root_quat_wxyz = _indexed_numpy_copy(
            robot.data.root_link_quat_w, self.env_index
        )
        T_world_base = _pose_matrix(root_pos_w, root_quat_wxyz)

        views: dict[str, CameraFrame] = {}
        for direction in FOUR_VIEW_DIRECTIONS:
            sensor_name = FOUR_VIEW_SENSOR_NAMES[direction]
            sensor = self.raw_env.scene[sensor_name]
            data = sensor.data

            rgb = _indexed_numpy_copy(data.output["rgb"], self.env_index)
            depth = _indexed_numpy_copy(
                data.output["distance_to_image_plane"], self.env_index
            )
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0].copy()

            K = _indexed_numpy_copy(data.intrinsic_matrices, self.env_index)
            T_world_camera_ros = camera_poses_ros[direction].copy()
            T_base_camera = np.linalg.inv(T_world_base) @ T_world_camera_ros
            sensor_frame_id = int(sensor.frame[self.env_index].detach().cpu().item())

            frame = CameraFrame(
                camera_id=sensor_name,
                direction=direction,
                sensor_frame_id=sensor_frame_id,
                sim_step=sim_step,
                timestamp=timestamp,
                rgb=rgb,
                depth_z_m=depth,
                K=K,
                T_world_camera_ros=T_world_camera_ros,
                T_base_camera=T_base_camera,
            )
            _validate_camera_frame(frame)
            views[direction] = frame

        common_step_after = int(
            getattr(self.raw_env, "common_step_counter", common_step_before)
        )
        if common_step_after != common_step_before:
            raise RuntimeError(
                "Simulation advanced during four-view capture: "
                f"step {common_step_before} -> {common_step_after}."
            )

        bundle = FrameBundle(
            bundle_id=self._next_bundle_id,
            env_index=self.env_index,
            sim_step=sim_step,
            timestamp=timestamp,
            T_world_base=T_world_base,
            views=views,
        )
        _validate_frame_bundle(bundle)
        self._next_bundle_id += 1
        return bundle

    def _camera_poses_from_current_torso(self) -> dict[str, np.ndarray]:
        """根据当前 torso tensor 与固定安装外参计算四台 ROS optical 世界位姿。"""
        robot = self.raw_env.scene["robot"]
        torso_pos_w = _indexed_body_numpy_copy(
            robot.data.body_link_pos_w,
            self.env_index,
            self._parent_body_id,
        )
        torso_quat_wxyz = _indexed_body_numpy_copy(
            robot.data.body_link_quat_w,
            self.env_index,
            self._parent_body_id,
        )
        T_world_torso = _pose_matrix(torso_pos_w, torso_quat_wxyz)

        camera_poses_ros: dict[str, np.ndarray] = {}
        for direction in FOUR_VIEW_DIRECTIONS:
            local_pos, local_quat = self._local_poses[direction]
            T_torso_camera_world = _pose_matrix(local_pos, local_quat)
            T_world_camera_world = T_world_torso @ T_torso_camera_world

            T_world_camera_ros = np.eye(4, dtype=np.float64)
            T_world_camera_ros[:3, :3] = (
                T_world_camera_world[:3, :3] @ _R_WORLD_CONVENTION_FROM_ROS
            )
            T_world_camera_ros[:3, 3] = T_world_camera_world[:3, 3]
            camera_poses_ros[direction] = T_world_camera_ros

        return camera_poses_ros

    def maybe_save_debug_snapshot(self, completed_step: int, step_dt: float) -> None:
        """按命令行配置在 warm-up 后安全地保存一次 FrameBundle。"""
        if self._debug_saved or not bool(self.args_cli.camera_debug_save_once):
            return
        if int(completed_step) < max(int(self.args_cli.camera_debug_warmup_steps), 0):
            return
        if self._debug_attempts >= 3:
            return

        self._debug_attempts += 1
        try:
            bundle = self.capture(
                sim_step=int(completed_step),
                timestamp=float(completed_step) * float(step_dt),
            )
            output_dir = self.save_debug_snapshot(
                bundle, Path(self.args_cli.camera_output_dir)
            )
            self._debug_saved = True
            self.print_summary(bundle)
            print(f"[INFO] Saved four-view RGB-D debug bundle: {output_dir}")
        except Exception as exc:
            print(
                "[WARN] Four-view RGB-D debug capture failed "
                f"(attempt {self._debug_attempts}/3): {exc}"
            )

    def report_debug_status(self) -> None:
        """在 runner 退出时说明一次性抓图是否执行，便于发现 warm-up 太长。"""
        if not bool(self.args_cli.camera_debug_save_once):
            return
        if self._debug_saved:
            return
        if self._debug_attempts == 0:
            print(
                "[WARN] Four-view RGB-D debug snapshot was not captured; "
                "the run ended before --camera_debug_warmup_steps."
            )
        else:
            print(
                "[WARN] Four-view RGB-D debug snapshot was not saved after "
                f"{self._debug_attempts} attempt(s)."
            )

    def save_debug_snapshot(self, bundle: FrameBundle, output_root: Path) -> Path:
        """保存 RGB、原始米制 depth、预览图和几何元数据。"""
        import cv2

        output_dir = (
            output_root
            / f"run_{self._debug_session}"
            / f"bundle_{bundle.bundle_id:06d}_step_{bundle.sim_step:06d}"
        )
        output_dir.mkdir(parents=True, exist_ok=False)

        montage_tiles = []
        metadata = {
            "schema_version": 1,
            "bundle_id": bundle.bundle_id,
            "env_index": bundle.env_index,
            "sim_step": bundle.sim_step,
            "timestamp": bundle.timestamp,
            "directions": list(FOUR_VIEW_DIRECTIONS),
            "depth_type": "distance_to_image_plane_m",
            "depth_unit": "meter",
            "configured_depth_range_m": [
                float(self.args_cli.rgbd_camera_near),
                float(self.args_cli.rgbd_camera_far),
            ],
            "invalid_depth": "non_finite_or_non_positive",
            "camera_frame_convention": "ROS optical: +X right, +Y down, +Z forward",
            "base_frame": "robot articulation root_link",
            "base_frame_axes": (
                "torso/root axes with calibrated navigation "
                "forward=+X, left=+Y, up=+Z"
            ),
            "quaternion_order": "wxyz",
            "transform_convention": (
                "T_A_B maps homogeneous point coordinates from frame B into frame A"
            ),
            "T_world_base": bundle.T_world_base.tolist(),
            "views": {},
        }

        for direction in FOUR_VIEW_DIRECTIONS:
            frame = bundle.views[direction]
            rgb_path = output_dir / f"{direction}_rgb.png"
            depth_path = output_dir / f"{direction}_depth.npy"
            depth_preview_path = output_dir / f"{direction}_depth_preview.png"

            rgb_bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(rgb_path), rgb_bgr):
                raise OSError(f"Failed to write {rgb_path}")
            np.save(depth_path, frame.depth_z_m, allow_pickle=False)
            if not cv2.imwrite(
                str(depth_preview_path),
                _depth_preview(
                    frame.depth_z_m,
                    near=float(self.args_cli.rgbd_camera_near),
                    far=float(self.args_cli.rgbd_camera_far),
                ),
            ):
                raise OSError(f"Failed to write {depth_preview_path}")

            tile = rgb_bgr.copy()
            cv2.putText(
                tile,
                direction.upper(),
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            montage_tiles.append(tile)

            valid_depth = np.isfinite(frame.depth_z_m) & (frame.depth_z_m > 0.0)
            valid_values = frame.depth_z_m[valid_depth]
            metadata["views"][direction] = {
                "camera_id": frame.camera_id,
                "sensor_frame_id": frame.sensor_frame_id,
                "sim_step": frame.sim_step,
                "timestamp": frame.timestamp,
                "image_width": int(frame.rgb.shape[1]),
                "image_height": int(frame.rgb.shape[0]),
                "rgb_dtype": str(frame.rgb.dtype),
                "depth_dtype": str(frame.depth_z_m.dtype),
                "valid_depth_fraction": float(valid_depth.mean()),
                "valid_depth_min_m": float(valid_values.min()),
                "valid_depth_median_m": float(np.median(valid_values)),
                "valid_depth_max_m": float(valid_values.max()),
                "K": frame.K.tolist(),
                "T_world_camera_ros": frame.T_world_camera_ros.tolist(),
                "T_base_camera": frame.T_base_camera.tolist(),
            }

        top_row = np.concatenate(montage_tiles[:2], axis=1)
        bottom_row = np.concatenate(montage_tiles[2:], axis=1)
        montage = np.concatenate((top_row, bottom_row), axis=0)
        montage_path = output_dir / "montage.png"
        if not cv2.imwrite(str(montage_path), montage):
            raise OSError(f"Failed to write {montage_path}")

        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return output_dir

    @staticmethod
    def print_summary(bundle: FrameBundle) -> None:
        """打印足够核对 shape、dtype、frame id 和 depth 有效率的信息。"""
        print(
            "[CAMERA] "
            f"bundle={bundle.bundle_id} sim_step={bundle.sim_step} "
            f"timestamp={bundle.timestamp:.3f}s env={bundle.env_index}"
        )
        for direction in FOUR_VIEW_DIRECTIONS:
            frame = bundle.views[direction]
            valid = np.isfinite(frame.depth_z_m) & (frame.depth_z_m > 0.0)
            print(
                "[CAMERA] "
                f"{direction:7s} frame={frame.sensor_frame_id:04d} "
                f"rgb={frame.rgb.shape}/{frame.rgb.dtype} "
                f"depth={frame.depth_z_m.shape}/{frame.depth_z_m.dtype} "
                f"valid_depth={float(valid.mean()):.3f}"
            )


def _indexed_numpy_copy(value, index: int) -> np.ndarray:
    """从 IsaacLab tensor 的某个 env 复制为独立、连续的 numpy array。"""
    selected = value[index]
    if hasattr(selected, "detach"):
        selected = selected.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(np.asarray(selected).copy())


def _indexed_body_numpy_copy(value, env_index: int, body_index: int) -> np.ndarray:
    """从 (env, body, ...) IsaacLab tensor 复制一个 body 的数据。"""
    selected = value[env_index, body_index]
    if hasattr(selected, "detach"):
        selected = selected.detach().cpu().contiguous().numpy()
    return np.ascontiguousarray(np.asarray(selected).copy())


def _pose_matrix(position_xyz: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """将 world 中的 wxyz pose 转成 4x4 齐次变换。"""
    position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError(f"Invalid pose quaternion: {quat.tolist()}")
    w, x, y, z = quat / norm

    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def _validate_camera_frame(frame: CameraFrame) -> None:
    """尽早拒绝 shape、dtype、内参或位姿错误，避免错误帧进入模型缓存。"""
    if frame.rgb.ndim != 3 or frame.rgb.shape[-1] != 3:
        raise ValueError(f"{frame.direction} RGB shape must be HxWx3, got {frame.rgb.shape}.")
    if frame.rgb.dtype != np.uint8:
        raise ValueError(f"{frame.direction} RGB dtype must be uint8, got {frame.rgb.dtype}.")
    if frame.depth_z_m.shape != frame.rgb.shape[:2]:
        raise ValueError(
            f"{frame.direction} depth shape {frame.depth_z_m.shape} does not match RGB {frame.rgb.shape[:2]}."
        )
    if not np.issubdtype(frame.depth_z_m.dtype, np.floating):
        raise ValueError(f"{frame.direction} depth must be floating point, got {frame.depth_z_m.dtype}.")
    if frame.K.shape != (3, 3) or not np.all(np.isfinite(frame.K)):
        raise ValueError(f"{frame.direction} camera intrinsics are invalid: shape={frame.K.shape}.")
    if frame.K[0, 0] <= 0.0 or frame.K[1, 1] <= 0.0:
        raise ValueError(f"{frame.direction} camera focal lengths must be positive: {frame.K}.")
    if frame.T_world_camera_ros.shape != (4, 4) or not np.all(
        np.isfinite(frame.T_world_camera_ros)
    ):
        raise ValueError(f"{frame.direction} T_world_camera_ros is invalid.")
    if frame.T_base_camera.shape != (4, 4) or not np.all(
        np.isfinite(frame.T_base_camera)
    ):
        raise ValueError(f"{frame.direction} T_base_camera is invalid.")

    valid_depth = np.isfinite(frame.depth_z_m) & (frame.depth_z_m > 0.0)
    if not np.any(valid_depth):
        raise ValueError(f"{frame.direction} depth image contains no finite positive values.")


def _validate_frame_bundle(bundle: FrameBundle) -> None:
    if tuple(bundle.views.keys()) != FOUR_VIEW_DIRECTIONS:
        raise ValueError(
            f"FrameBundle directions must be {FOUR_VIEW_DIRECTIONS}, got {tuple(bundle.views.keys())}."
        )
    if bundle.T_world_base.shape != (4, 4) or not np.all(np.isfinite(bundle.T_world_base)):
        raise ValueError("FrameBundle T_world_base is invalid.")
    for direction, frame in bundle.views.items():
        if frame.direction != direction:
            raise ValueError(
                f"FrameBundle key {direction!r} contains frame direction {frame.direction!r}."
            )
        if frame.sim_step != bundle.sim_step or frame.timestamp != bundle.timestamp:
            raise ValueError(f"{direction} frame metadata does not match its FrameBundle.")


def _depth_preview(depth_z_m: np.ndarray, near: float = 0.1, far: float = 10.0) -> np.ndarray:
    """把原始米制 depth 转成仅用于观察的彩色预览；原始值始终另存为 .npy。"""
    import cv2

    depth = np.asarray(depth_z_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth[valid], near, far)
        normalized[valid] = np.round(
            (1.0 - (clipped - near) / max(far - near, 1.0e-6)) * 255.0
        ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored
