from __future__ import annotations

import math

import numpy as np

from .odometry import Pose2D
from .types import DIRECTION_ORDER, PanoramaBundle, ViewFrame


SENSOR_NAMES = {
    "forward": "camera_forward",
    "left": "camera_left",
    "behind": "camera_behind",
    "right": "camera_right",
}


def _indexed_numpy_copy(value, index: int) -> np.ndarray:
    item = value[index]
    if hasattr(item, "detach"):
        item = item.detach()
    if hasattr(item, "cpu"):
        item = item.cpu()
    if hasattr(item, "numpy"):
        item = item.numpy()
    return np.asarray(item).copy()


class IsaacLocalFourViewCamera:
    """Read Isaac RGB-D/K only; intentionally never reads robot/world pose tensors."""

    def __init__(self, raw_env, env_index: int = 0):
        self.raw_env = raw_env
        self.env_index = int(env_index)
        self.next_bundle_id = 0
        if not 0 <= self.env_index < int(raw_env.num_envs):
            raise ValueError("Isaac camera env index is outside the environment batch.")
        missing = [
            sensor_name
            for sensor_name in SENSOR_NAMES.values()
            if sensor_name not in set(raw_env.scene.keys())
        ]
        if missing:
            raise RuntimeError(f"Isaac four-view camera sensors are missing: {missing}.")

    def capture_panorama(self, sim_step: int, timestamp: float) -> PanoramaBundle:
        self.raw_env.sim.render()
        for sensor_name in SENSOR_NAMES.values():
            self.raw_env.scene[sensor_name].update(dt=0.0, force_recompute=True)
        views = {
            direction: self._copy_view(direction, sim_step, timestamp)
            for direction in DIRECTION_ORDER
        }
        bundle = PanoramaBundle(
            bundle_id=self.next_bundle_id,
            sim_step=int(sim_step),
            timestamp=float(timestamp),
            views=views,
        ).validated()
        self.next_bundle_id += 1
        return bundle

    def capture_forward(self, sim_step: int, timestamp: float) -> ViewFrame:
        self.raw_env.sim.render()
        sensor = self.raw_env.scene[SENSOR_NAMES["forward"]]
        sensor.update(dt=0.0, force_recompute=True)
        return self._copy_view("forward", sim_step, timestamp).validated()

    def _copy_view(
        self,
        direction: str,
        sim_step: int,
        timestamp: float,
    ) -> ViewFrame:
        sensor = self.raw_env.scene[SENSOR_NAMES[direction]]
        data = sensor.data
        rgb = _indexed_numpy_copy(data.output["rgb"], self.env_index)
        depth = _indexed_numpy_copy(
            data.output["distance_to_image_plane"], self.env_index
        )
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0].copy()
        K = _indexed_numpy_copy(data.intrinsic_matrices, self.env_index)
        try:
            frame_id = int(sensor.frame[self.env_index].detach().cpu().item())
        except Exception:
            frame_id = int(sim_step)
        return ViewFrame(
            direction=direction,
            frame_id=frame_id,
            sim_step=int(sim_step),
            timestamp=float(timestamp),
            rgb=rgb,
            depth_m=depth,
            K=K,
        ).validated()


class IsaacRootOdometryProvider:
    """Explicit opt-in odometry adapter; never construct it in no-odom runs."""

    def __init__(self, raw_env, env_index: int = 0):
        self.raw_env = raw_env
        self.env_index = int(env_index)

    def get_pose(self) -> Pose2D:
        robot = self.raw_env.scene["robot"]
        position = _indexed_numpy_copy(robot.data.root_link_pos_w, self.env_index)
        quat = _indexed_numpy_copy(robot.data.root_link_quat_w, self.env_index)
        qw, qx, qy, qz = (float(value) for value in quat)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        timestamp = float(
            getattr(self.raw_env, "common_step_counter", 0)
        ) * float(self.raw_env.step_dt)
        return Pose2D(
            x=float(position[0]),
            y=float(position[1]),
            yaw=yaw,
            timestamp=timestamp,
        ).validated()
