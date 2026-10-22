"""真机追加记录与原子落盘；不计算控制命令，不启动硬件。"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from . import real_console


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def atomic_json(path, payload):
    """同目录临时文件+fsync+replace，旧的完整记录不会被半写文件覆盖。"""
    path = Path(path)
    data = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class RealRunRecords:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.last_pose = None
        self.last_command = [0., 0., 0.]
        self.last_sync = 0.
        self.timings = {}

    def timing(self, kind, duration_s, **details):
        self.timings.setdefault(kind, []).append(float(duration_s))
        self.append('timing', {'kind':kind, 'duration_s':float(duration_s), **details})

    def save_timings(self):
        self.save('timing_summary.json', {kind:{'count':len(values),
            'mean_s':float(np.mean(values)), 'max_s':max(values),
            'p95_s':float(np.percentile(values,95))}
            for kind,values in self.timings.items()})

    def save(self, name, payload):
        if self.output_dir is not None:
            atomic_json(Path(self.output_dir)/name, payload)

    def append(self, event, payload, *, durable=False):
        if self.output_dir is None:
            return
        now = time.monotonic()
        row = {'event':event, 'monotonic_s':now, 'wall_time_s':time.time(), **payload}
        with (Path(self.output_dir)/'execution_events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=_json_default)+'\n')
            stream.flush()
            if durable or now-self.last_sync >= 1.:
                os.fsync(stream.fileno())
                self.last_sync = now

    def pose(self, episode):
        try:
            pose = episode.odometry.get_pose()
            if pose is not None:
                self.last_pose = episode._pose_dict(pose.validated())
                return self.last_pose, 'fresh_provider_read'
        except Exception:
            pass
        return self.last_pose, 'last_recorded_pose' if self.last_pose else 'unavailable'

    def sample(self, episode, phase, command=None):
        pose, source = self.pose(episode)
        if phase == 'command_requested':
            self.last_command = np.asarray(command, dtype=float).tolist()
        payload = {'state':episode.state, 'decision_index':episode.decision_index,
                   'phase':phase, 'pose':pose, 'pose_source':source,
                   'pose_frame_id':episode.pose_frame_id, 'frame_epoch':episode.frame_epoch,
                   'last_requested_command':self.last_command,
                   'action_elapsed_s':episode.action_elapsed_s,
                   'window_index':episode.motion_window_index,
                   'camera_frame_id':getattr(episode.planner, 'last_frame_id', None)}
        if command is not None:
            payload['command'] = np.asarray(command, dtype=float).tolist()
        if phase == 'command_requested' and episode.follower.active:
            # 保存实际跟随器的路径，避免follow_live.json被停车状态覆盖后丢失依据。
            payload['path_local_xyz_m'] = episode.follower._reference.path_local
            try:
                payload['goal_local_xy_m'] = episode.follower.current_goal_local_xy
            except Exception as error:
                payload['goal_read_error'] = str(error)
        self.append('control', payload)

    def local_exit(self, episode, reason, error, stop_record):
        pose, source = self.pose(episode)
        now = time.monotonic()
        summary = {'session_id':episode.config.session_id, 'exit_reason':reason,
                   'error':error, 'state':episode.state, 'decision_index':episode.decision_index,
                   'failure_reason':episode.failure_reason,
                   'session_success_reason':episode.session_success_reason,
                   'final_pose':pose, 'final_pose_source':source,
                   'pose_frame_id':episode.pose_frame_id, 'frame_epoch':episode.frame_epoch,
                   'final_pose_age_s':None if pose is None else max(0., now-pose['timestamp']),
                   'pose_timestamp_convention':'host monotonic receive time, not exposure time',
                   'last_command_before_shutdown':self.last_command,
                   'stop_request':stop_record, 'physical_stop_confirmed':False,
                   'phase':'LOCAL_RECORDS_SAVED', 'remote_end_status':'PENDING'}
        # 包含尚未凑满一秒的窗口和未完成动作；只在本地保存，不伪造action_complete。
        start = episode._motion_window_start_pose
        partial = {'decision_index':episode.decision_index,
                   'window_index':episode.motion_window_index,
                   'partial':True, 'uploaded':False, 'exit_reason':reason,
                   'state':episode.state,
                   'action_in_progress':episode.state in ('executing','backtrack_executing'),
                   'pose_start':episode._pose_dict(start), 'pose_end':pose,
                   'world_trace_xy_m':episode._active_world_trace,
                   'map_explored_before':episode._map_window_explored_before,
                   'map_explored_at_exit':None if episode.exploration_map is None
                       else episode.exploration_map.explored_cells}
        if start is not None and pose is not None:
            partial['duration_s'] = pose['timestamp']-start.timestamp
            partial['displacement_m'] = math.hypot(pose['x']-start.x, pose['y']-start.y)
        self.save('execution_tail.json', partial)
        self.save_timings()
        self.save('run_exit.json', summary)
        self.append('shutdown_local_saved', summary, durable=True)
        return summary


class RecordingSessionClient:
    """保留原协议/重试；HTTP前保存窗口请求，退出HTTP单独限制等待时间。"""
    def __init__(self, client, records):
        self.client, self.records = client, records

    def __getattr__(self, name):
        return getattr(self.client, name)

    def report_motion_window(self, **kwargs):
        stem = f"g3_decision_{kwargs['decision_index']:03d}_motion_{kwargs['window_index']:04d}_request.json"
        self.records.save(stem, kwargs)
        self.records.append('motion_window_request', kwargs, durable=True)
        self.records.timing('motion_window_span_s',
                            kwargs.get('timestamp_end',0)-kwargs.get('timestamp_start',0),
                            decision_index=kwargs['decision_index'],window_index=kwargs['window_index'])
        started = time.monotonic()
        try:
            result = self.client.report_motion_window(**kwargs)
        except Exception as error:
            self.records.timing('motion_window_http_s',time.monotonic()-started,
                                status='ERROR', decision_index=kwargs['decision_index'],
                                window_index=kwargs['window_index'])
            self.records.append('motion_window_interrupted', {'request_file':stem,
                                'error':f'{type(error).__name__}: {error}'}, durable=True)
            raise
        self.records.timing('motion_window_http_s',time.monotonic()-started,
                            decision_index=kwargs['decision_index'],window_index=kwargs['window_index'])
        self.records.append('motion_window_response', {'request_file':stem, 'response':result[1]})
        real_console.emit(real_console.window_lines,kwargs,result[1],time.monotonic()-started)
        return result

    def report_action_complete(self, **kwargs):
        stem = f"g3_decision_{kwargs['decision_index']:03d}_action_complete_request.json"
        self.records.save(stem, kwargs)
        self.records.append('action_complete_request', kwargs, durable=True)
        started = time.monotonic()
        result = self.client.report_action_complete(**kwargs)
        self.records.timing('action_complete_http_s',time.monotonic()-started,
                            decision_index=kwargs['decision_index'])
        self.records.append('action_complete_response', {'request_file':stem, 'response':result[1]})
        return result

    def end_session(self, **kwargs):
        self.records.save('g3_session_end_request.json', kwargs)
        timeout = self.client.timeout_s
        self.client.timeout_s = min(timeout, 5.)
        try:
            result = self.client.end_session(**kwargs)
        except Exception as error:
            self.records.save('g3_session_end_attempt.json', {
                'status':'UNCONFIRMED', 'request':kwargs, 'error':str(error)})
            raise
        else:
            self.records.save('g3_session_end_attempt.json', {
                'status':'CONFIRMED', 'request':kwargs, 'response':result[1]})
            return result
        finally:
            self.client.timeout_s = timeout


class RecordingPlanner:
    """只测量原规划调用的耗时与帧号，不改变同步方式、路径或跟随器。"""
    def __init__(self, planner, records):
        self.planner, self.records = planner, records
        self.console_last_time = -math.inf
        self.plan_call_index = 0

    def __getattr__(self, name):
        return getattr(self.planner, name)

    def get_plan(self, frame, goal):
        started = time.monotonic()
        self.plan_call_index+=1
        result = self.planner.get_plan(frame, goal)
        ended=time.monotonic()
        duration=ended-started
        self.records.timing('iplanner_call_s',duration,
                            frame_id=frame.frame_id, goal_local_xy_m=np.asarray(goal).tolist())
        path,fear=result
        if ended-self.console_last_time>=1. or path is None or len(path)<2:
            self.console_last_time=ended
            real_console.emit(real_console.planner_lines,self.plan_call_index,frame.frame_id,
                              np.asarray(goal).tolist(),path,fear,duration)
        return result
