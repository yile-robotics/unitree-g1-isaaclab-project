"""真机专用全景采集；共享Episode及Isaac入口保持原来的旋转行为。"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from .episode import EpisodeState, LocalEndToEndEpisode, _PanoramaSweep
from .odometry import wrap_to_pi
from .types import DIRECTION_ORDER


@dataclass(frozen=True)
class RealPanoramaConfig:
    speed_rad_s: float = 0.8
    imu_stop_deg: float = 75.0
    settle_s: float = 0.5
    quarter_timeout_s: float = 10.0

    def validated(self):
        values = (self.speed_rad_s, self.imu_stop_deg, self.settle_s, self.quarter_timeout_s)
        if any(not math.isfinite(v) or v <= 0 for v in values) or self.imu_stop_deg > 90:
            raise ValueError("Real panorama parameters must be positive; IMU stop must be <=90°.")
        return self


class RealG1Episode(LocalEndToEndEpisode):
    """只替换真机四段全景子流程，模型决策、导航跟随与仿真实现继续复用。

    stop_robot由真机入口注入DDS.stop。到达阈值时同步主动停车；从停车调用返回后
    用单调时钟等待，再采图。每次update等待期间返回零速度，不sleep阻塞控制循环。
    地图位置和模型选定方向的旋转仍使用既有SLAM来源；全景提前停车单独读取IMU。
    """

    def __init__(self, *args, panorama_imu, stop_robot,
                 real_panorama_config: RealPanoramaConfig | None = None,
                 clock=time.monotonic, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_panorama_config = (real_panorama_config or RealPanoramaConfig()).validated()
        self.panorama_imu = panorama_imu
        self.stop_robot = stop_robot
        self._clock = clock
        self._real_phase = "idle"
        self._real_records = {}
        self._real_turns = []
        self._real_trace_status = "IDLE"
        self._real_stop_times = {}

    def _imu_yaw(self):
        yaw = self.panorama_imu.get_yaw()
        if yaw is None or not math.isfinite(yaw):
            raise RuntimeError("Real panorama IMU missing or stale; stopping.")
        return float(yaw)

    def _map_pose(self):
        pose = self.odometry.get_pose()
        if pose is None:
            raise RuntimeError("Real panorama requires fresh SLAM pose; stopping.")
        return pose.validated()

    def _stop_and_wait(self):
        started = self._clock()
        self.stop_robot()
        # 等待从StopMove/零速度调用返回之后开始，不把RPC耗时算作0.5秒等待。
        returned = self._clock()
        self._real_stop_times = {"stop_requested_monotonic_s": started,
                                 "stop_returned_monotonic_s": returned}
        self._real_capture_after = returned + self.real_panorama_config.settle_s
        self._real_phase = "settling"

    def _start_single_camera_panorama(self, completed_step, timestamp):
        if self.panorama_sweep is not None:
            raise RuntimeError("A real panorama sweep is already active.")
        self._imu_yaw()
        self._map_pose()
        self.panorama_sweep = _PanoramaSweep({}, None, {})
        self._real_records = {}
        self._real_turns = []
        self._real_trace_status = "RUNNING"
        self._stop_and_wait()
        self.state = EpisodeState.WAIT_PANORAMA_LOCOMOTION
        self._write_real_trace()
        print("[REAL VLN] panorama: stop → settle → forward; four left turns, "
              f"wz={self.real_panorama_config.speed_rad_s:.2f}, "
              f"IMU stop={self.real_panorama_config.imu_stop_deg:.1f}°", flush=True)

    def _begin_quarter(self):
        self._real_initial_imu = self._imu_yaw()
        self._real_previous_imu = self._real_initial_imu
        self._real_accumulated_imu = 0.0
        self._real_initial_map = self._map_pose().yaw
        self._real_turn_started = self._clock()
        self._real_phase = "turning"
        print(f"[REAL VLN] quarter {self.panorama_sweep.quarter_turns_completed + 1}/4 started", flush=True)

    def _take_view(self, direction, completed_step, timestamp):
        pose_before = self._map_pose()
        imu_before = self._imu_yaw()
        requested = self._clock()
        frame, _ = self._capture_forward_observation(completed_step, timestamp)
        pose_after = self._map_pose()
        imu_after = self._imu_yaw()
        record = {"frame_id": frame.frame_id, "sim_step": frame.sim_step,
                  "timestamp": frame.timestamp, "K": frame.K.tolist(),
                  **self._real_stop_times,
                  "capture_requested_monotonic_s": requested,
                  "capture_returned_monotonic_s": self._clock(),
                  "pose_before_capture": self._pose_dict(pose_before),
                  "pose_after_capture": self._pose_dict(pose_after),
                  "imu_before_capture_rad": imu_before, "imu_after_capture_rad": imu_after,
                  "camera_metadata": dict(getattr(self.camera, "last_metadata", {}))}
        if self.output_dir is not None:
            stem = f"decision_{self.decision_index:03d}_{direction}"
            if not cv2.imwrite(str(self.output_dir / f"{stem}.png"),
                               cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to save real panorama {direction} RGB.")
            np.save(self.output_dir / f"{stem}_depth_m.npy", frame.depth_m)
        self._real_records[direction] = record
        if direction != "forward_return":
            self.panorama_sweep.views[direction] = self._relabel_forward_frame(frame, direction)
            self.panorama_sweep.capture_poses[direction] = pose_after
            if direction == "forward":
                self.panorama_sweep.decision_pose = pose_after
        self._write_real_trace()
        print(f"[REAL VLN] captured {direction}; frame={frame.frame_id}", flush=True)

    def _update_single_camera_panorama(self, *, completed_step, step_dt, timestamp, locomotion_ready):
        if self.state == EpisodeState.PANORAMA_DECIDE:
            # 原有逻辑冻结四张图并在后台调用模型；额外回正图只保留诊断。
            return super()._update_single_camera_panorama(
                completed_step=completed_step, step_dt=step_dt, timestamp=timestamp,
                locomotion_ready=locomotion_ready)
        if self.panorama_sweep is None:
            raise RuntimeError("No real panorama sweep.")
        zero = np.zeros(3, dtype=np.float64)
        imu = self._imu_yaw()
        pose = self._map_pose()
        if not locomotion_ready:
            if self._real_phase == "turning":
                raise RuntimeError("Locomotion lost during real panorama.")
            return zero
        self.state = EpisodeState.PANORAMA_ROTATING

        if self._real_phase == "settling":
            if self._clock() < self._real_capture_after:
                return zero
            if self.panorama_sweep.views:
                self._real_accumulated_imu += wrap_to_pi(imu - self._real_previous_imu)
                map_deg = math.degrees(wrap_to_pi(pose.yaw - self._real_initial_map))
                self._real_turns.append({"imu_deg": math.degrees(self._real_accumulated_imu),
                                         "slam_deg": map_deg})
                if not 70.0 <= map_deg <= 110.0:
                    raise RuntimeError(f"Real panorama SLAM turn {map_deg:+.1f}° outside 70°–110°.")
                self.panorama_sweep.quarter_turns_completed += 1
            quarter = self.panorama_sweep.quarter_turns_completed
            direction = DIRECTION_ORDER[quarter] if quarter < 4 else "forward_return"
            self._take_view(direction, completed_step, timestamp)
            if quarter == 4:
                self._real_trace_status = "PASS"
                self._real_phase = "idle"
                self.state = EpisodeState.PANORAMA_DECIDE
                self._write_real_trace()
            else:
                # 采图结束这一轮仍返回零速度；下一轮才记录新基线并开始下一段。
                self._real_phase = "ready_to_turn"
            return zero

        if self._real_phase == "ready_to_turn":
            self._begin_quarter()
            return np.array([0.0, 0.0, self.real_panorama_config.speed_rad_s])

        if self._real_phase == "turning":
            self._real_accumulated_imu += wrap_to_pi(imu - self._real_previous_imu)
            self._real_previous_imu = imu
            imu_deg = math.degrees(self._real_accumulated_imu)
            if imu_deg <= -20.0:
                raise RuntimeError("Wrong-direction IMU turn during real panorama.")
            if self._clock() - self._real_turn_started >= self.real_panorama_config.quarter_timeout_s:
                raise RuntimeError("Real panorama quarter time limit reached before IMU target.")
            if imu_deg >= self.real_panorama_config.imu_stop_deg:
                self._stop_and_wait()
                self._write_real_trace()
                return zero
            return np.array([0.0, 0.0, self.real_panorama_config.speed_rad_s])
        raise RuntimeError(f"Unexpected real panorama phase: {self._real_phase}")

    def _write_real_trace(self):
        self._save_json(f"decision_{self.decision_index:03d}_panorama_capture.json", {
            "camera_mode": "real_single_forward_rgbd_imu75_rotation",
            "status": self._real_trace_status, "phase": self._real_phase,
            "rotation_direction": "left",
            "speed_rad_s": self.real_panorama_config.speed_rad_s,
            "imu_stop_deg": self.real_panorama_config.imu_stop_deg,
            "settle_s": self.real_panorama_config.settle_s,
            "quarter_turns_completed": (self.panorama_sweep.quarter_turns_completed
                                        if self.panorama_sweep else 4),
            "quarter_turns": self._real_turns,
            "total_slam_turn_deg": sum(t["slam_deg"] for t in self._real_turns),
            "pose_timing": "poses observed around capture; not interpolated to exposure time",
            "captures": self._real_records,
        })

    def _save_single_camera_panorama_trace(self, sweep):
        self._write_real_trace()

    def _fail(self, reason):
        # 包括相机阻塞/异常、IMU和SLAM失效，先主动停车，再写失败日志。
        try:
            self.stop_robot()
        finally:
            self._real_trace_status = "FAILED"
            super()._fail(reason)
            if self.panorama_sweep is not None:
                self._write_real_trace()
