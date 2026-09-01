from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model_contract import NavigationDecisionResponse
from .types import ViewFrame


@dataclass(frozen=True)
class LocalTargetProjection:
    """把模型选中的图像目标表示为“转向后机器人坐标系”中的二维目标。

    坐标约定为 x 向前、y 向左。该对象还保留检测框、采样像素、深度窗口等
    调试信息，方便追查一个目标点是如何从图像中计算出来的。
    """

    direction: str
    bbox_xyxy: tuple[int, int, int, int]
    pixel_uv: tuple[int, int]
    depth_m: float | None
    depth_window_xyxy_exclusive: tuple[int, int, int, int]
    valid_depth_count: int
    goal_after_turn_xy_m: np.ndarray
    used_forward_fallback: bool = False

    def to_dict(self) -> dict:
        """转换为可写入 JSON 的普通字典；NumPy 数组会转换成列表。"""

        return {
            "direction": self.direction,
            "bbox_xyxy": list(self.bbox_xyxy),
            "pixel_uv": list(self.pixel_uv),
            "depth_m": self.depth_m,
            "depth_window_xyxy_exclusive": list(
                self.depth_window_xyxy_exclusive
            ),
            "valid_depth_count": self.valid_depth_count,
            "goal_after_turn_xy_m": self.goal_after_turn_xy_m.tolist(),
            "used_forward_fallback": self.used_forward_fallback,
            "coordinate_convention": "robot_after_turn: x=forward,y=left",
        }


def project_selected_view_target(
    frame: ViewFrame,
    response: NavigationDecisionResponse,
    *,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
    window_radius_px: int = 3,
    depth_percentile: float = 30.0,
) -> LocalTargetProjection:
    """利用检测框和深度图，把视觉目标投影成转向后的局部二维目标。

    这里不使用世界位姿。算法假设机器人完成理想转向后，被选中相机原来的光轴
    就成为机器人正前方，因此只需针孔相机的水平投影关系，就能得到 iPlanner
    所需的 ``(向前距离, 向左距离)``。

    深度并非只取单个像素，而是在检测框底边中心附近取一个小窗口，并使用有效
    深度的指定百分位数。这样比单点采样更能抵抗空洞和背景干扰。
    """

    frame.validated()
    action = response.action.upper()
    if action == "BACKTRACK":
        raise ValueError("BACKTRACK has no bbox target to project.")
    if response.direction != frame.direction:
        raise ValueError(
            f"Response selected {response.direction!r}, got {frame.direction!r} frame."
        )
    if not 0.0 < min_depth_m < max_depth_m:
        raise ValueError("Depth range must satisfy 0 < min < max.")
    if window_radius_px < 0:
        raise ValueError("Depth window radius must be non-negative.")
    if not 0.0 <= depth_percentile <= 100.0:
        raise ValueError("Depth percentile must lie in [0, 100].")

    # 检测框先裁剪到图像内部，再取“底边中心”作为目标接地点。
    height, width = frame.depth_m.shape
    bbox = response.clipped_bbox(width, height)
    x1, _y1, x2, y2 = bbox
    u = int((x1 + x2) / 2.0)
    v = int(y2)

    # 构造不越过图像边界的深度采样窗口；右边界和下边界采用 Python 独占格式。
    x_min = max(0, u - window_radius_px)
    x_max = min(width, u + window_radius_px + 1)
    y_min = max(0, v - window_radius_px)
    y_max = min(height, v + window_radius_px + 1)
    patch = np.asarray(
        frame.depth_m[y_min:y_max, x_min:x_max], dtype=np.float64
    )
    # 丢弃 NaN、Inf、过近和过远的深度值，只对可信像素计算百分位数。
    valid = patch[
        np.isfinite(patch)
        & (patch >= float(min_depth_m))
        & (patch <= float(max_depth_m))
    ]
    if valid.size == 0:
        # 与 Uni-LaViRA G1 的 tasks/vln.py 一致：bbox/depth 无法给出可靠
        # 三维目标时，不终止任务，而是让机器人转向后尝试向正前方走 1.5m。
        print(
            "[LOCAL-VLN WARN] No valid selected-view depth around pixel "
            f"{(u, v)}; using Uni-LaViRA fallback goal [1.5, 0.0]m."
        )
        return LocalTargetProjection(
            direction=frame.direction,
            bbox_xyxy=bbox,
            pixel_uv=(u, v),
            depth_m=None,
            depth_window_xyxy_exclusive=(x_min, y_min, x_max, y_max),
            valid_depth_count=0,
            goal_after_turn_xy_m=np.array([1.5, 0.0], dtype=np.float64),
            used_forward_fallback=True,
        )
    depth_m = float(np.percentile(valid, depth_percentile))

    K = np.asarray(frame.K, dtype=np.float64)
    fx = float(K[0, 0])
    principal_x = float(K[0, 2])
    if not np.isfinite(fx) or fx <= 0.0 or not np.isfinite(principal_x):
        raise ValueError("Selected camera has invalid fx/cx intrinsics.")

    # 针孔模型给出的水平偏移以“相机向右”为正；机器人目标坐标以“向左”为正，
    # 所以写入 goal 的第二项时需要取反。
    camera_x_right_m = (float(u) - principal_x) * depth_m / fx
    goal = np.array(
        [depth_m, -camera_x_right_m],
        dtype=np.float64,
    )
    return LocalTargetProjection(
        direction=frame.direction,
        bbox_xyxy=bbox,
        pixel_uv=(u, v),
        depth_m=depth_m,
        depth_window_xyxy_exclusive=(x_min, y_min, x_max, y_max),
        valid_depth_count=int(valid.size),
        goal_after_turn_xy_m=goal,
        used_forward_fallback=False,
    )
