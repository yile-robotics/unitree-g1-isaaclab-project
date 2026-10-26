"""真机专用全景采集；共享Episode及Isaac入口保持原来的旋转行为。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time

import cv2
import numpy as np

from .episode import EpisodeState, LocalEndToEndEpisode, _PanoramaSweep
from .odometry import wrap_to_pi, local_points_to_fixed, fixed_points_to_local
from .types import DIRECTION_ORDER
from .real_run_records import RealRunRecords, RecordingSessionClient, RecordingPlanner, atomic_json
from .real_navigation_rotation import RealNavigationRotation
from .local_projection import project_selected_view_target
from .real_target_geometry import build_target_geometry, update_target_for_pose, pose_from_record
from . import real_console


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
    """真机全景及导航转向适配，模型决策、导航跟随继续复用共享实现。

    stop_robot由真机入口注入DDS.stop。到达阈值时同步主动停车；从停车调用返回后
    用单调时钟等待，再采图。每次update等待期间返回零速度，不sleep阻塞控制循环。
    地图位置使用SLAM；全景和导航选定方向都用lowstate IMU提前停车、SLAM核验。
    """

    def __init__(self, *args, panorama_imu, stop_robot,
                 real_panorama_config: RealPanoramaConfig | None = None,
                 clock=time.monotonic, decision_display_s=0., **kwargs):
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
        self._display_moving = False
        self._display_following = False
        self._display_decision = None
        self._target_geometry = None
        self.decision_display_s = float(decision_display_s)
        if not math.isfinite(self.decision_display_s) or not 0 <= self.decision_display_s <= 10:
            raise ValueError('decision display time must be within 0..10 seconds')
        self._decision_display_until = 0.
        self._navigation_settle_after = 0.
        self.rotation = RealNavigationRotation(
            self.config.rotation_speed_rad_s, self.config.rotation_duration_scale,
            yaw_reader=lambda:self._map_pose().yaw, clock=self._clock,
            imu_reader=self._imu_yaw, on_stop=self._stop_navigation_rotation,
            stop_deg=self.real_panorama_config.imu_stop_deg,
            settle_s=self.real_panorama_config.settle_s,
            quarter_timeout_s=self.real_panorama_config.quarter_timeout_s)
        self.run_records = RealRunRecords(self.output_dir)
        if self.planner is not None:
            self.planner = RecordingPlanner(self.planner, self.run_records)
        if self.session_client is not None:
            self.session_client = RecordingSessionClient(self.session_client, self.run_records)

    @staticmethod
    def _project_selected_view_target(frame, response, **kwargs):
        projection = project_selected_view_target(
            frame, response, expanded_window_radius_px=7, limit_to_bbox=True, **kwargs)
        sampling = projection.depth_sampling
        if sampling["used_expanded_window"]:
            last = sampling["attempts"][-1]
            print(f'[REAL VLN DEPTH] 7x7无有效深度，尝试框内15x15；'
                  f'有效点={last["valid_depth_count"]}/{last["pixel_count"]} '
                  f'深度={projection.depth_m}m 兜底={projection.used_forward_fallback}', flush=True)
        return projection

    def _save_json(self, name, payload):
        if self.output_dir is not None:
            atomic_json(self.output_dir/name, payload)
        if name == 'g3_session_started.json':
            real_console.emit(real_console.stage_plan_lines,payload)
        elif name.startswith('map_progress_decision_'):
            real_console.emit(real_console.map_lines,payload)

    def _report_motion_window_if_due(self):
        # 只改变真机窗口触发边界：用窗口起止SLAM接收时间，避免把进入执行前
        # 的step_dt计入首个窗口；保留共享协议、证据计算和跟随控制。
        if self.state not in (EpisodeState.EXECUTING, EpisodeState.BACKTRACK_EXECUTING):
            return None
        start = self._motion_window_start_pose
        end = self.odometry.get_pose()
        if start is None or end is None:
            raise RuntimeError('Real motion window requires fresh SLAM poses.')
        if end.timestamp-start.timestamp + 1e-9 < self.config.motion_window_s:
            return None
        self.next_motion_window_elapsed_s = self.action_elapsed_s
        return super()._report_motion_window_if_due()

    def update(self, *args, **kwargs):
        self.run_records.sample(self, 'update_begin')
        if self.state == EpisodeState.ROTATION_SETTLE:
            if self._clock() < self._navigation_settle_after:
                return self._result(np.zeros(3))
            self.rotation_settle_elapsed_s = self.config.rotation_settle_s
        if self.state == EpisodeState.WAIT_EXECUTION_LOCOMOTION and self._clock() < self._decision_display_until:
            return self._result(np.zeros(3))
        result = super().update(*args, **kwargs)
        if self.output_dir is not None:
            moving = result.desired_mode == 'locomotion' and bool(np.any(result.command))
            if moving and not self._display_moving:
                from live_model_target import mark_display_motion
                mark_display_motion(self.output_dir)
            self._display_moving = moving
            if moving and self.state == EpisodeState.EXECUTING and self.follower.active and self.pending is not None:
                self._publish_follow_preview(result)
            elif self._display_following:
                self.save_follow_stopped()
        return result

    def _stop_navigation_rotation(self):
        self.stop_robot()
        self._navigation_settle_after = self._clock()+self.config.rotation_settle_s
        self._save_json(f'decision_{self.decision_index:03d}_navigation_rotation.json',
                        self.rotation.snapshot())
        print(f'[REAL VLN] navigation {self.rotation.direction}: IMU early-stop; '
              f'wz={self.rotation.sign*self.rotation.speed_rad_s:+.2f}, '
              f'settling {self.real_panorama_config.settle_s:.1f}s',flush=True)

    def _publish_follow_preview(self, result):
        pose = self.odometry.get_pose()
        if pose is None:
            return
        if self._display_decision != self.decision_index:
            self._display_decision = self.decision_index
            self._display_target_world = local_points_to_fixed(
                self.pending.projection.goal_after_turn_xy_m.reshape(1, 2), pose)
        # 只读原跟随器实际持有的局部路径，不为显示重算另一条路径或修改参考状态。
        path = np.asarray(self.follower._reference.path_local)
        path = np.column_stack((path[:, :2], np.zeros(len(path))))
        payload = {'state':'FOLLOWING','status':'PASS','execution_source':'formal_real_episode',
                   'executed_by_probe':False, 'target_version':self.decision_index,
                   'received_monotonic_s':time.monotonic(), 'path_local_xyz_m':path.tolist(),
                   'source_frame_id':getattr(self.planner,'last_frame_id',0),
                   'target':{'target_preview_xy_m':fixed_points_to_local(self._display_target_world,pose)[0].tolist(),
                             'goal_preview_xy_m':self.follower.current_goal_local_xy.tolist()},
                   'fear':float(self.last_fear or 0.),
                   'request_duration_s':getattr(self.planner,'request_duration_s',0.),
                   'command_vx_vy_wz':result.command.tolist(), 'extrinsics_applied':False,
                   'coordinate_convention':'original follower local path; nominal camera target',
                   'note':'Requested by formal follower; not an SDK acknowledgement or collision check'}
        temporary = self.output_dir/'.follow_live.tmp'
        temporary.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
        temporary.replace(self.output_dir/'follow_live.json')
        self._display_following = True

    def save_follow_stopped(self):
        self._save_json('follow_live.json',{'state':'STOPPED','received_monotonic_s':time.monotonic(),
                                          'execution_source':'formal_real_episode'})
        self._display_following = False

    def _apply_decision_response(self, panorama, *, decision_pose, response, raw_response):
        # 仅真机增加目标输出；仍由共享状态机校验并执行原决策。
        index = self.decision_index
        direction=raw_response.get('direction')
        frame=panorama.views.get(direction) if isinstance(direction,str) else None
        real_console.emit(real_console.model_lines,index,raw_response,frame)
        apply_error=None
        self._target_geometry = None
        try:
            result = super()._apply_decision_response(
                panorama, decision_pose=decision_pose, response=response, raw_response=raw_response)
            if (raw_response.get('action') == 'NAVIGATE'
                    and raw_response.get('control') not in ('PREEMPT', 'SAFE_STOP')
                    and self.pending is not None
                    and getattr(self.pending, 'panorama', None) is panorama):
                self._compute_target_geometry(panorama, response)
            return result
        except Exception as error:
            apply_error=str(error)
            raise
        finally:
            accepted=(self.pending is not None and getattr(self.pending,'panorama',None) is panorama
                      and raw_response.get('action') == 'NAVIGATE'
                      and raw_response.get('control') not in ('PREEMPT','SAFE_STOP'))
            projection=None
            if accepted:
                try:
                    projection=self.pending.projection.to_dict()
                except Exception:
                    pass  # 调试摘要不能替换原导航异常。
            real_console.emit(real_console.execution_lines,index,raw_response,self.state,
                              accepted=accepted,projection=projection,
                              failure_reason=apply_error or self.failure_reason)
            if self.output_dir is not None:
                try:
                    from .real_decision_artifacts import save_decision_target
                    frame = panorama.views.get(raw_response.get('direction'), panorama.views['forward'])
                    save_decision_target(self.output_dir, index, raw_response, frame,
                                         projection=projection, navigation_accepted=accepted)
                except Exception as error:
                    print(f'[REAL VLN ARTIFACT WARN] target preview not saved: {error}', flush=True)

    def _compute_target_geometry(self, panorama, response):
        # 前两处几何修正独立记录；第三处规划接口适配前，不替换旧名义目标。
        geometry = {'status': 'UNAVAILABLE', 'mode': 'geometry_comparison_only',
                    'applied_to_planner': False}
        try:
            frame = panorama.views[response.direction]
            record = self._real_records.get(response.direction)
            if record is None or (record['frame_id'], record['sim_step'], record['timestamp']) != (
                    frame.frame_id, frame.sim_step, frame.timestamp):
                raise ValueError('缺少所选图像对应的采图位姿/外参，拒绝借用其他帧')
            transform = record['camera_metadata']['T_pelvis_camera']
            pose = pose_from_record(record['pose_after_capture'])
            geometry = build_target_geometry(frame, self.pending.projection, transform, pose)
            current = self.odometry.get_pose()
            if current is not None:
                geometry = update_target_for_pose(geometry, current)
        except Exception as error:
            geometry['reason'] = str(error)
            print(f'[REAL VLN GEOMETRY] 修正目标不可用：{error}', flush=True)
        self._target_geometry = geometry
        self._save_json(f'decision_{self.decision_index:03d}_target_geometry.json', geometry)

    def _capture_forward_observation(self, completed_step, timestamp):
        frame, pose = super()._capture_forward_observation(completed_step, timestamp)
        if self.state == EpisodeState.PLAN_AFTER_ROTATION and self._target_geometry is not None:
            if pose is None:
                self._target_geometry = {**self._target_geometry, 'status': 'UNAVAILABLE',
                                         'reason': 'planning-time SLAM pose unavailable'}
                self._target_geometry.pop('target_current_pelvis_xy_m', None)
            else:
                self._target_geometry = update_target_for_pose(
                    self._target_geometry, pose, reference_frame_id=frame.frame_id)
            self._save_json(f'decision_{self.decision_index:03d}_target_geometry.json', self._target_geometry)
            point = self._target_geometry.get('target_current_pelvis_xy_m')
            if point is not None:
                print(f'[REAL VLN GEOMETRY] 转向后pelvis目标={point}m；'
                      '已补偿采图外参及实际SLAM位姿，仅记录，尚未传入iPlanner', flush=True)
        return frame, pose

    def _capture_forward_and_plan(self, completed_step, timestamp):
        if self.pending is not None and self.pending.response.direction != 'forward':
            if self._clock() < self._navigation_settle_after:
                raise RuntimeError('Real navigation rotation has not settled; planning blocked.')
            settled = self.rotation.verify_settled(self.pending.response.direction)
            self._save_json(f'decision_{self.decision_index:03d}_navigation_rotation.json',
                            {**self.rotation.snapshot(), **settled})
            print(f'[REAL VLN] navigation {self.rotation.direction} verified: '
                  f'target={math.degrees(self.rotation.target_yaw_rad):+.1f}deg, '
                  f'error={settled["settled_error_deg"]:+.1f}deg; planning allowed',flush=True)
        super()._capture_forward_and_plan(completed_step, timestamp)
        if self.state == EpisodeState.WAIT_EXECUTION_LOCOMOTION:
            self._decision_display_until = self._clock() + self.decision_display_s
        # 给只读预览标记“已按选中方向转向并重新取前视帧”；不改变共享/仿真控制逻辑。
        if self.pending is not None and self.pending.post_rotation_forward is not None:
            self._save_json(f"decision_{self.decision_index:03d}_overlay.json", {
                "view_aligned": True,
                "decision_index": self.decision_index,
                "direction": self.pending.response.direction,
                "forward_frame_id": self.pending.post_rotation_forward.frame_id,
                "note": "Navigation turn and fresh forward capture completed; bbox is held, not tracked.",
            })

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
            if self.rotation.status == 'FAILED':
                self._save_json(f'decision_{self.decision_index:03d}_navigation_rotation.json',
                                self.rotation.snapshot())
            if self.panorama_sweep is not None:
                self._write_real_trace()
