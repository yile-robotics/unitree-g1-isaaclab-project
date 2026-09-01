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
    """从 Isaac 的批量张量中取出一个环境的数据，并安全复制为 NumPy 数组。

    Isaac 数据可能是 GPU 上的 PyTorch 张量，也可能已经是 NumPy 数组，因此按
    ``detach → cpu → numpy`` 的顺序逐步兼容。最后 ``copy``，避免后续仿真更新
    原始缓冲区时悄悄改掉已经保存的相机帧或位姿。
    """

    item = value[index]
    if hasattr(item, "detach"):
        item = item.detach()
    if hasattr(item, "cpu"):
        item = item.cpu()
    if hasattr(item, "numpy"):
        item = item.numpy()
    return np.asarray(item).copy()


class IsaacLocalFourViewCamera:
    """Isaac Sim 四方向相机适配器，只读取 RGB、深度和相机内参。

    该类刻意不读取机器人世界位姿，以保证“无里程计”导航实验不会通过相机后端
    偷用真值坐标。Isaac 的环境通常是批量运行的，``env_index`` 指定读取哪一个。
    """

    def __init__(self, raw_env, env_index: int = 0):
        """保存 Isaac 环境并确认四个约定名称的相机传感器全部存在。"""

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
        """强制刷新四个相机，复制出同一时刻的全景观察并分配 bundle 编号。"""

        # 先渲染场景，再强制传感器重算，避免读到上一个仿真步的旧画面。
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
        """只刷新并读取正前方相机，供转向后的 iPlanner 重规划使用。"""

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
        """把指定方向的 Isaac 传感器输出整理成通用 ``ViewFrame``。"""

        sensor = self.raw_env.scene[SENSOR_NAMES[direction]]
        data = sensor.data
        rgb = _indexed_numpy_copy(data.output["rgb"], self.env_index)
        depth = _indexed_numpy_copy(
            data.output["distance_to_image_plane"], self.env_index
        )
        # Isaac 有时把深度保存为 H×W×1；统一压成后续算法要求的 H×W。
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0].copy()
        K = _indexed_numpy_copy(data.intrinsic_matrices, self.env_index)
        # 某些传感器版本没有可用 frame 计数，此时退化为仿真步编号。
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
    """显式启用的 Isaac 根节点里程计适配器。

    它读取仿真中的机器人世界真值，因此无里程计实验绝不能创建这个对象。只有调用
    方主动传入它时，局部轨迹跟随器才会使用固定坐标系修正累计误差。
    """

    def __init__(self, raw_env, env_index: int = 0):
        self.raw_env = raw_env
        self.env_index = int(env_index)

    def get_pose(self) -> Pose2D:
        """读取机器人根节点位置和四元数，并转换成平面 ``Pose2D``。"""

        robot = self.raw_env.scene["robot"]
        position = _indexed_numpy_copy(robot.data.root_link_pos_w, self.env_index)
        quat = _indexed_numpy_copy(robot.data.root_link_quat_w, self.env_index)
        # Isaac 四元数顺序为 (w, x, y, z)，下面只提取绕竖直轴的 yaw。
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
