"""真机D415：SLAM基座由用户确认为pelvis，逐帧用URDF/FK换算胸前相机外参。

只读关节和相机，不创建运动客户端。地图仍用原稀疏栅格积分；这里只替换坐标变换。
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from camera_d415.network_camera import create_camera as create_network_camera
from unified_vln.map_progress import SparseEpisodeExplorationMap, SparseMapConfig

# 复用已用于手触碰标定的关节映射和URDF正运动学，不另建一套关节编号。
_KINEMATICS_DIR = Path(__file__).resolve().parents[2] / 'camera_extrinsics'
if str(_KINEMATICS_DIR) not in sys.path:
    sys.path.insert(0, str(_KINEMATICS_DIR))
from robot_state import JointReader, model_for


def load_profile(path):
    path = Path(path).resolve()
    profile = json.loads(path.read_text(encoding='utf-8'))
    if (profile.get('status') != 'USER_CONFIRMED_SLAM_BASE_IS_PELVIS'
            or profile.get('physical_frame') != 'pelvis'
            or profile.get('slam_child_frame') != 'base_link'
            or profile.get('slam_base_verified') is not True
            or not str(profile.get('slam_base_verification_source', '')).strip()):
        raise ValueError('需要base_link=pelvis确认记录及其来源')
    for key in ('pelvis_height_above_floor_m', 'max_joint_pair_delta_s'):
        value = float(profile[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid {key}')
    if not math.isfinite(float(profile['floor_z_world_m'])):
        raise ValueError('Invalid floor height')
    profile['network_camera_config'] = (path.parent / profile['network_camera_config']).resolve()
    profile['torso_calibration'] = (path.parent / profile['torso_calibration']).resolve()
    calibration = json.loads(profile['torso_calibration'].read_text(encoding='utf-8'))
    if calibration.get('target_frame') != 'torso_link' or calibration.get('source_frame') != 'color_optical':
        raise ValueError('需要color_optical → torso_link标定候选')
    T = np.asarray(calibration['T_torso_camera'], dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1., atol=1e-6)):
        raise ValueError('Invalid T_torso_camera')
    profile['T_torso_camera'] = T
    profile['camera_serial'] = str(calibration['reference_camera']['serial'])
    profile['calibration_status'] = calibration.get('status')
    return profile


def map_config_for_transform(T, pelvis_height, floor=0., base_config=None):
    """骨盆离地高度直接配置；相机高度由骨盆高度加逐帧外参的z分量得到。"""
    pelvis_height = float(pelvis_height)
    if not math.isfinite(pelvis_height) or pelvis_height <= 0:
        raise ValueError('Invalid pelvis_height_above_floor_m')
    rotation = np.asarray(T)[:3, :3]
    optical_forward = rotation[:, 2]
    yaw = math.atan2(optical_forward[1], optical_forward[0])
    down = -math.atan2(optical_forward[2], math.hypot(*optical_forward[:2]))
    return replace(base_config or SparseMapConfig(),
                   camera_offset_x_m=float(T[0, 3]), camera_offset_y_m=float(T[1, 3]),
                   camera_offset_z_m=float(T[2, 3]), camera_yaw_rad=yaw,
                   camera_down_tilt_rad=down,
                   nominal_base_height_m=float(floor + pelvis_height),
                   floor_z_world_m=float(floor)).validated()


class PelvisCamera:
    def __init__(self, profile_path, interface=None):
        self.profile = load_profile(profile_path)
        self.reader = self.camera = None
        self.last_metadata = {}
        self.input_error = None
        self.transforms = OrderedDict()
        try:
            self.reader = JointReader(interface or self.profile['network_interface'])
            state = self.reader.latest(timeout=3.)
            self.model = model_for(state)
            self.initial_transform = self.transform_for_state(state)
            self.nominal_config = map_config_for_transform(
                self.initial_transform, self.profile['pelvis_height_above_floor_m'],
                self.profile['floor_z_world_m'])
            self.initial_state = state
            self.camera = create_network_camera(self.profile['network_camera_config'])
        except BaseException:
            self.close()
            raise

    def transform_for_state(self, state):
        if state['mode_pr'] != 0 or model_for(state).sha256 != self.model.sha256:
            raise RuntimeError('关节模式/URDF发生变化，不沿用旧映射')
        return self.model.transform('pelvis', 'torso_link', state['q']) @ self.profile['T_torso_camera']

    def capture_forward(self, sim_step, timestamp):
        try:
            return self._capture_forward(sim_step, timestamp)
        except Exception as error:
            self.input_error = str(error)
            raise

    def _capture_forward(self, sim_step, timestamp):
        frame = self.camera.capture_forward(sim_step, timestamp)
        meta = dict(self.camera.last_metadata)
        if str(meta['serial']) != self.profile['camera_serial']:
            raise ValueError('相机序列号与胸部标定候选不匹配')
        # 对齐的是本机接收时刻，不能声称与相机曝光硬同步。
        self.reader.latest(timeout=.5)
        received = float(meta['client_received_monotonic_s'])
        with self.reader.lock:
            state = min(self.reader.samples, key=lambda s: abs(s['received_monotonic_s'] - received))
        delta = abs(state['received_monotonic_s'] - received)
        if delta > self.profile['max_joint_pair_delta_s']:
            raise RuntimeError(f'图像/腰关节接收时间差过大：{delta:.3f}s')
        T = self.transform_for_state(state)
        key = (frame.frame_id, frame.sim_step, frame.timestamp)
        self.transforms[key] = T.copy()
        while len(self.transforms) > 512:
            self.transforms.popitem(last=False)
        self.last_metadata = {**meta, 'slam_base_relation': 'base_link=pelvis',
                              'slam_base_verified': self.profile['slam_base_verified'],
                              'slam_base_verification_source': self.profile['slam_base_verification_source'],
                              'T_pelvis_camera': T.tolist(),
                              'pelvis_height_above_floor_m': self.profile['pelvis_height_above_floor_m'],
                              'derived_camera_height_above_floor_m': float(self.profile['pelvis_height_above_floor_m'] + T[2, 3]),
                              'waist_joint_rad': {k: v for k, v in state['q'].items() if k.startswith('waist_')},
                              'joint_image_receipt_delta_s': delta,
                              'geometry_time_pairing': 'nearest host receipt, not exposure synchronisation'}
        return frame

    def transform_for_frame(self, frame):
        key = (frame.frame_id, frame.sim_step, frame.timestamp)
        if key not in self.transforms:
            raise RuntimeError('没有该图像对应的pelvis外参，拒绝使用另一帧的腰角')
        return self.transforms[key].copy()

    def validate_slam_frame(self, parent, child):
        if parent != 'map' or child != self.profile['slam_child_frame']:
            raise ValueError('pelvis入口要求SLAM消息为map → base_link')

    def guard_motion_backend(self, backend, pose_source, lowstate):
        from unified_vln.real_input_guard import RealInputGuard
        return RealInputGuard(backend, self, pose_source, lowstate)

    def create_planner(self, url, timeout_s):
        return RealDepthPlanner(url, timeout_s)

    def provenance(self):
        path = self.profile['torso_calibration']
        return {'status': self.profile['status'],
                'slam_base_verified': self.profile['slam_base_verified'],
                'slam_base_relation': 'base_link=pelvis',
                'slam_base_verification_source': self.profile['slam_base_verification_source'],
                'calibration_status': self.profile['calibration_status'],
                'calibration_file': str(path), 'calibration_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'urdf': str(self.model.path), 'urdf_sha256': self.model.sha256,
                'T_torso_camera': self.profile['T_torso_camera'].tolist(),
                'initial_T_pelvis_camera': self.initial_transform.tolist(),
                'initial_waist_joint_rad': {k: v for k, v in self.initial_state['q'].items() if k.startswith('waist_')},
                'pelvis_height_above_floor_m': self.profile['pelvis_height_above_floor_m'],
                'initial_derived_camera_height_above_floor_m': float(self.profile['pelvis_height_above_floor_m'] + self.initial_transform[2, 3]),
                'height_source': self.profile['pelvis_height_source'],
                'projection_model': 'full per-frame waist FK; SLAM planar xy/yaw; fixed nominal pelvis height',
                'local_navigation_projection': 'planner retains nominal camera axes; real target_geometry independently records SE3 and measured-turn correction; not applied to planner'}

    def create_exploration_map(self, config, *, pose_frame_id, frame_epoch=0):
        config = map_config_for_transform(self.initial_transform,
            self.profile['pelvis_height_above_floor_m'], self.profile['floor_z_world_m'], config)
        return PelvisExplorationMap(config, pose_frame_id=pose_frame_id,
                                    frame_epoch=frame_epoch, camera=self)

    def close(self):
        try:
            if self.camera is not None:
                self.camera.close()
        finally:
            if self.reader is not None:
                self.reader.close()


class PelvisExplorationMap(SparseEpisodeExplorationMap):
    """原稀疏地图的真机坐标适配，不改变栅格、膨胀或连通性算法。"""
    def __init__(self, *args, camera, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera = camera
        self.current_transform = None

    def integrate(self, frame, robot_pose):
        self.current_transform = self.camera.transform_for_frame(frame)
        try:
            return super().integrate(frame, robot_pose)
        finally:
            self.current_transform = None

    def _optical_points_to_world(self, points_optical, robot_pose):
        if self.current_transform is None:
            raise RuntimeError('缺少当前帧pelvis外参')
        T = self.current_transform
        c, s = math.cos(robot_pose.yaw), math.sin(robot_pose.yaw)
        world_rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        base = np.array([robot_pose.x, robot_pose.y, self.config.nominal_base_height_m])
        points_pelvis = np.asarray(points_optical) @ T[:3, :3].T + T[:3, 3]
        return points_pelvis @ world_rotation.T + base, world_rotation @ T[:3, 3] + base

    def save_debug(self, output_prefix, snapshot):
        result = super().save_debug(output_prefix, snapshot)
        path = Path(result['metadata'])
        payload = json.loads(path.read_text())
        payload['real_geometry'] = self.camera.provenance()
        payload['real_geometry']['last_camera_metadata'] = self.camera.last_metadata
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return result


class RealDepthPlanner:
    """沿用单轮真机已验证的0.1～5m输入适配，防止旧uint16协议的远距离回绕。"""
    def __init__(self, url, timeout_s):
        from unified_vln.iplanner_client import IPlannerClient
        self.client = IPlannerClient(url, timeout_s)
        self.last_frame_id = 0
        self.request_duration_s = 0.

    def __getattr__(self, name):
        return getattr(self.client, name)

    def get_plan(self, frame, goal):
        import time
        from g1_iplanner_viewer import planner_frame
        started = time.monotonic()
        result = self.client.get_plan(planner_frame(frame), goal)
        self.last_frame_id = frame.frame_id
        self.request_duration_s = time.monotonic()-started
        return result


def create_camera(config_path):
    return PelvisCamera(config_path)
