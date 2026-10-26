"""真机目标三维几何。前两阶段独立记录，尚不接入旧iPlanner坐标接口。"""
from __future__ import annotations

import numpy as np
from .odometry import Pose2D, local_points_to_fixed, fixed_points_to_local


def pose_dict(pose):
    pose.validated()
    return {k: float(getattr(pose, k)) for k in ('x', 'y', 'yaw', 'timestamp')}


def pose_from_record(record):
    return Pose2D(**{k: float(record[k]) for k in ('x', 'y', 'yaw', 'timestamp')}).validated()


def backproject_optical(pixel_uv, depth_m, intrinsic):
    """RGB光学坐标：x右、y下、z前。深度为光轴Z，而非射线长度。"""
    K = np.asarray(intrinsic, dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('目标反投影需要有效fx、fy和3x3内参')
    if not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-9 or abs(K[1, 0]) > 1e-9:
        raise ValueError('当前反投影要求零skew针孔内参')
    z = float(depth_m)
    uv = np.asarray(pixel_uv, dtype=float)
    if uv.shape != (2,) or not np.isfinite(uv).all() or not np.isfinite(z) or z <= 0:
        raise ValueError('像素和深度必须有效，禁止将前进兜底当作深度')
    return np.array([(uv[0]-K[0, 2])*z/K[0, 0],
                     (uv[1]-K[1, 2])*z/K[1, 1], z])


def optical_to_pelvis(point, transform):
    T = np.asarray(transform, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0,0,0,1]):
        raise ValueError('相机到pelvis外参必须是有效4x4刚体变换')
    R = T[:3, :3]
    if not np.allclose(R.T@R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1., atol=1e-5):
        raise ValueError('相机外参旋转矩阵不合法')
    return R@np.asarray(point, dtype=float)+T[:3, 3]


def build_target_geometry(frame, projection, transform, capture_pose):
    """冻结所选图像对应的外参和SLAM位姿；不假设left一定旋转90°。"""
    result = {'status': 'UNAVAILABLE', 'mode': 'geometry_comparison_only',
              'applied_to_planner': False, 'direction': frame.direction,
              'source_frame_id': frame.frame_id, 'source_sim_step': frame.sim_step,
              'source_timestamp': frame.timestamp, 'pixel_uv': list(projection.pixel_uv),
              'depth_m': projection.depth_m,
              'nominal_goal_after_turn_xy_m': projection.goal_after_turn_xy_m.tolist(),
              'coordinate_model': 'per-frame optical->pelvis SE3; map planar xy/yaw',
              'limitations': 'candidate mount calibration; capture-time host pose, not exposure sync; pelvis roll/pitch in map not compensated; no ground intersection or obstacle clearance check'}
    if projection.used_forward_fallback or projection.depth_m is None:
        result['reason'] = 'no_measured_depth; fixed forward fallback is not a 3D observation'
        return result
    if projection.direction != frame.direction:
        raise ValueError('投影方向与图像不一致')
    capture_pose.validated()
    pc = backproject_optical(projection.pixel_uv, projection.depth_m, frame.K)
    pb = optical_to_pelvis(pc, transform)
    world_xy = local_points_to_fixed(pb[:2].reshape(1,2), capture_pose)[0]
    result.update(status='COMPUTED', point_camera_optical_xyz_m=pc.tolist(),
                  point_pelvis_at_capture_xyz_m=pb.tolist(), target_map_xy_m=world_xy.tolist(),
                  capture_pose=pose_dict(capture_pose), T_pelvis_camera=np.asarray(transform).tolist(),
                  pose_pairing='SLAM pose observed after selected-frame capture; not exposure interpolation')
    return result


def update_target_for_pose(geometry, current_pose, *, reference_frame_id=None):
    """地图中目标保持不动，以实际当前位置和yaw重新表达；同时补偿转向平移。"""
    result = dict(geometry)
    if geometry.get('status') != 'COMPUTED':
        return result
    current_pose.validated()
    point = fixed_points_to_local(np.asarray(geometry['target_map_xy_m']).reshape(1,2), current_pose)[0]
    nominal = np.asarray(geometry['nominal_goal_after_turn_xy_m'])
    result.update(current_pose=pose_dict(current_pose), current_reference_frame_id=reference_frame_id,
                  target_current_pelvis_xy_m=point.tolist(),
                  target_current_planar_distance_m=float(np.linalg.norm(point)),
                  corrected_minus_nominal_xy_m=(point-nominal).tolist(),
                  correction_norm_m=float(np.linalg.norm(point-nominal)))
    return result
