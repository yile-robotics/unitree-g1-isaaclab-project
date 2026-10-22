#!/usr/bin/env python3
"""真机单次诊断：原旋转采图/模型入口 → 仿真共用投影、iPlanner、局部跟随器。

默认只读检查；--execute 才旋转和行走。--complete-local-follow 按原到达条件完成本次局部跟随。
没有完整G3执行上报/地图监督，不伪造执行证据。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import cv2
import numpy as np

from camera_d415.network_camera import create_camera
from g1_camera_model_probe import load_panorama, write_json
from g1_iplanner_viewer import planner_frame
from g1_real_smoke import StateMonitor
from live_model_target import mark_display_motion
from run_g1_real import _SlamPoseSource, _wait_for_odometry
from unified_vln.g1_dds_backend import UnitreeG1DDSBackend
from unified_vln.iplanner_client import IPlannerClient
from unified_vln.local_projection import project_selected_view_target
from unified_vln.local_trajectory import LocalFollowerConfig, LocalTrajectoryFollower, truncate_trajectory_for_safety
from unified_vln.model_contract import NavigationDecisionResponse
from unified_vln.odometry import Pose2D, wrap_to_pi, local_points_to_fixed, fixed_points_to_local
from unified_vln.ros2_odometry import Ros2OdometryProvider
from unified_vln.session_client import G3SessionClient
from unified_vln.types import ViewFrame

ROOT = Path(__file__).resolve().parent


def load_decision(run):
    report = json.loads((run/'flow_report.json').read_text())
    if report.get('status') != 'PASS' or not report.get('rotation_capture_completed'):
        raise ValueError('需要完整成功的旋转采集和模型结果')
    raw = json.loads((run/'model/response.json').read_text())
    if raw.get('control') not in (None, 'CONTINUE'):
        raise ValueError('模型要求停止/恢复，本诊断不执行该动作')
    decision = NavigationDecisionResponse.from_dict(raw)
    if decision.action != 'NAVIGATE' or decision.direction != 'forward':
        raise ValueError('首次诊断仅执行NAVIGATE/forward；STOP或侧后方结果不行走')
    bundle = load_panorama(run/'panorama')
    projection = project_selected_view_target(bundle.views[decision.direction], decision)
    if projection.used_forward_fallback:
        raise ValueError('选中视图没有有效目标深度，本诊断不执行1.5米前进兜底')
    return projection


def checked_path(path):
    result = np.asarray(path, dtype=float)
    if result.ndim != 2 or len(result) < 2 or result.shape[1] < 2 or not np.isfinite(result).all():
        raise ValueError('iPlanner没有有效有限路径；不发送行走速度')
    return result


def initial_plan(client, frame, goal):
    # 与episode.py的首次规划相同：原目标投影→iPlanner→末端保留0.5m→同一个follower。
    if not client.reset(frame.K.tolist()):
        raise RuntimeError('iPlanner reset失败')
    # 使用已实测预览入口的同一深度范围适配；不让>5m的真机值在旧uint16协议回绕。
    path, fear = client.get_plan(planner_frame(frame), goal)
    path = checked_path(path)
    safe = truncate_trajectory_for_safety(path, .5)
    return path, safe, fear


def refresh_follow_camera(camera, odometry, plan_pose):
    """DDS初始化后、非零速度前重新取图；不能用重置时钟冒充新图。

    首次路径仍对应原规划帧，因此取图后还要确认机器人没有离开规划位姿。
    """
    frame = camera.capture_forward(0, time.monotonic())
    received = float(camera.last_metadata['client_received_monotonic_s'])
    age = time.monotonic() - received
    if not math.isfinite(age) or not 0. <= age <= 1.:
        raise RuntimeError('起步前相机图像已过期，不开始行走')
    pose = odometry.get_pose()
    if math.hypot(pose.x-plan_pose.x, pose.y-plan_pose.y) > .05 or abs(wrap_to_pi(pose.yaw-plan_pose.yaw)) > math.radians(3):
        raise RuntimeError('规划后机器人位姿变化，旧路径不执行')
    return frame, age


class FollowWatchdog:
    """测试外层的距离/时长/数据失效停车；不计算路径或跟随速度。

    同步相机/HTTP调用期间也检查，防止后台DDS无限重复上一速度。限制为软件请求停车阈值，
    不是物理刹停距离保证。
    """
    def __init__(self, dds, odometry, lowstate, camera_time, seconds, distance):
        self.dds, self.odom, self.low = dds, odometry, lowstate
        self.camera_time = camera_time
        self.seconds, self.distance = seconds, distance
        self.start = odometry.get_pose()
        self.previous = self.start
        self.started = time.monotonic()
        self.reason = None
        self.stop = threading.Event()
        self.command_lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def check(self):
        self.low.require_fresh()
        pose = self.odom.get_pose()  # 缺失/过期抛异常，不退回命令积分。
        if math.hypot(pose.x-self.previous.x, pose.y-self.previous.y) > .25 or abs(wrap_to_pi(pose.yaw-self.previous.yaw)) > math.radians(30):
            raise RuntimeError('SLAM pose jumped')
        self.previous = pose
        if time.monotonic()-self.camera_time[0] > 1.:
            raise RuntimeError('camera input stale')
        if self.distance is not None and math.hypot(pose.x-self.start.x, pose.y-self.start.y) >= self.distance:
            return 'DISTANCE_LIMIT'
        if time.monotonic()-self.started >= self.seconds:
            return 'TIME_LIMIT'
        return None

    def run(self):
        while not self.stop.wait(.02):
            try:
                reason = self.check()
            except Exception as error:
                reason = 'DATA_FAILURE: '+str(error)
            if reason:
                with self.command_lock:
                    self.reason = reason
                    self.stop.set()
                    self.dds.set_velocity(0., 0., 0.)
                self.dds.stop()
                return

    def send(self, command):
        with self.command_lock:
            if self.stop.is_set():
                return False
            self.dds.set_velocity(*command.tolist())
            return True

    def close(self):
        self.stop.set()
        self.thread.join(timeout=1.)


def follow(dds, camera, odometry, lowstate, client, safe_path, config, args, output,
           *, model_goal=None, initial_frame_id=0, initial_fear=0.):
    follower = LocalTrajectoryFollower(config, odometry)
    follower.start(safe_path, safe_path[-1,:2])
    start_pose=odometry.get_pose()
    display_path_world=local_points_to_fixed(safe_path[:,:2],start_pose)
    display_target_world=local_points_to_fixed(np.asarray(
        model_goal if model_goal is not None else safe_path[-1,:2]).reshape(1,2),start_pose)
    source_frame_id=initial_frame_id
    last_fear=initial_fear
    request_duration=0.
    camera_time = [camera.last_metadata['client_received_monotonic_s']]
    if time.monotonic()-camera_time[0] > 1.:
        raise RuntimeError('初始规划图像已过期，不开始行走')
    # 模型框和独立预览路径在实际跟随开始前失效；原响应、图像仍保留供离线查看。
    mark_display_motion(output)
    guard = FollowWatchdog(dds, odometry, lowstate, camera_time, args.follow_seconds, args.max_travel_m)
    trace = []
    last = time.monotonic()
    command = np.zeros(3)
    reason = 'UNKNOWN'
    try:
        while not guard.stop.is_set():
            now = time.monotonic()
            lowstate.require_fresh()
            pose = odometry.get_pose()
            update = follower.update(max(now-last,1e-6), command)
            last = now
            if update.abort_reason:
                raise RuntimeError(update.abort_reason)
            if update.reached:
                reason = 'LOCAL_GOAL_REACHED'
                break
            # 原episode同样在执行阶段按needs_replan()重新取深度、更新当前局部目标并replace_path。
            if follower.needs_replan():
                started = time.time()
                frame = camera.capture_forward(len(trace), now)
                camera_time[0] = float(camera.last_metadata['client_received_monotonic_s'])
                path, fear = client.get_plan(planner_frame(frame), follower.current_goal_local_xy)
                follower.replace_path(checked_path(path))
                display_path_world=local_points_to_fixed(np.asarray(path)[:,:2],odometry.get_pose())
                source_frame_id=frame.frame_id;last_fear=fear
                request_duration=time.time()-started
                follower.mark_replan_attempt(started)
                write_json(output/'latest_replan.json', {'frame_id':frame.frame_id,
                           'path':np.asarray(path).tolist(),'fear':fear})
            # HTTP/取图期间可能已经由watchdog停车，不能再把旧非零命令写回DDS。
            if guard.stop.is_set():
                break
            command = np.asarray(update.command,float)
            if command.shape != (3,) or not np.isfinite(command).all():
                raise RuntimeError('跟随器返回无效速度')
            if not guard.send(command):
                break
            trace.append({'t_s':time.monotonic()-guard.started,'pose':[pose.x,pose.y,pose.yaw],
                          'command_vx_vy_wz':command.tolist(),'goal_distance_m':update.distance_to_goal_m})
            display_path=fixed_points_to_local(display_path_world,pose)
            preview={'state':'FOLLOWING','status':'PASS','executed_by_probe':True,
                     'source_frame_id':source_frame_id,'target_version':1,
                     'received_monotonic_s':time.monotonic(),
                     'path_local_xyz_m':np.column_stack((display_path,np.zeros(len(display_path)))).tolist(),
                     'target':{'target_preview_xy_m':fixed_points_to_local(display_target_world,pose)[0].tolist(),
                               'goal_preview_xy_m':update.goal_local_xy.tolist()},
                     'fear':last_fear,'request_duration_s':request_duration,
                     'command_vx_vy_wz':command.tolist(),'extrinsics_applied':False,
                     'target_source':'actual shared follower; nominal camera axes'}
            temporary=output/'.follow_live.tmp'
            write_json(temporary,preview);temporary.replace(output/'follow_live.json')
            if len(trace)%10 == 0:
                print(f'[FOLLOW] t={trace[-1]["t_s"]:.1f}s goal={update.distance_to_goal_m:.2f}m command={command.round(3).tolist()}',flush=True)
            guard.stop.wait(.05)
        reason = guard.reason or reason
        if reason.startswith('DATA_FAILURE'):
            raise RuntimeError(reason)
        return reason, trace
    except (Exception, KeyboardInterrupt) as error:
        reason = 'ERROR: '+('Ctrl+C' if isinstance(error,KeyboardInterrupt) else str(error))
        raise
    finally:
        guard.close()
        dds.stop()
        follower.stop()
        write_json(output/'follow_live.json',{'state':'STOPPED','stop_reason':guard.reason or reason,
                   'received_monotonic_s':time.monotonic()})
        write_json(output/'follow_trace.json',{'stop_reason':guard.reason or reason,'trace':trace,
                   'commands_requested':any(any(v != 0 for v in row['command_vx_vy_wz']) for row in trace)})


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true',help='由用户手动执行：旋转、一次决策、局部跟随')
    mode.add_argument('--replay-run',type=Path,help='只用记录图像规划并算第一帧速度；不初始化DDS/ROS')
    parser.add_argument('--complete-local-follow',action='store_true',
                        help='取消短测位移截止，按原局部到达条件完成；仍保留动作超时和数据失效停车')
    parser.add_argument('--network-interface',default='enxf8e43bbea486')
    parser.add_argument('--instruction',default='向前走到一个黑色的行李箱前')
    parser.add_argument('--camera-config',type=Path,default=ROOT/'camera_d415/network_camera_d415_local.json')
    parser.add_argument('--model-url',default='http://127.0.0.1:18765/v1/lavira/decision')
    parser.add_argument('--iplanner-url',default='http://127.0.0.1:8888')
    parser.add_argument('--tracking-controller',choices=('pure_pursuit','kp'),default=LocalFollowerConfig().tracking_controller)
    parser.add_argument('--follow-seconds',type=float,
                        help='跟随超时；默认短测3秒，complete-local-follow模式60秒')
    parser.add_argument('--max-travel-m',type=float,
                        help='短测SLAM位移上限，默认0.5米；不与complete-local-follow同时使用')
    parser.add_argument('--decision-display-s',type=float,default=2.,help='停车显示新决策的时间，不改变跟随器')
    parser.add_argument('--output',type=Path,default=ROOT.parents[1]/'outputs/d415_rotation_model'/
                        (datetime.now().strftime('once_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]))
    return parser


def main(argv=None):
    parser=build_parser();args=parser.parse_args(argv)
    if args.complete_local_follow:
        if args.max_travel_m is not None:
            parser.error('--complete-local-follow不与--max-travel-m同时使用')
        args.follow_seconds=60. if args.follow_seconds is None else args.follow_seconds
        if not math.isfinite(args.follow_seconds) or not 0<args.follow_seconds<=60:
            parser.error('完整局部跟随要求0<follow-seconds<=60')
    else:
        args.follow_seconds=3. if args.follow_seconds is None else args.follow_seconds
        args.max_travel_m=.5 if args.max_travel_m is None else args.max_travel_m
        if not math.isfinite(args.follow_seconds) or not 0<args.follow_seconds<=6 or not math.isfinite(args.max_travel_m) or not 0<args.max_travel_m<=.5:
            parser.error('短程诊断要求0<follow-seconds<=6、0<max-travel-m<=0.5')
    if not math.isfinite(args.decision_display_s) or not 0<=args.decision_display_s<=10:
        parser.error('decision-display-s必须在0～10秒内')
    if args.output.exists():parser.error('--output必须是新目录')
    args.output.mkdir(parents=True)
    config=LocalFollowerConfig(tracking_controller=args.tracking_controller,
                              target_speed_m_s=.6,max_forward_speed_m_s=.6)
    report={'robot_motion_sent':False,'scope':'one_local_follow_diagnostic','full_g3_execution_reporting':False,
            'camera_extrinsics_applied':False,'controller_config':asdict(config),
            'completion_mode':'local_goal' if args.complete_local_follow else 'short_probe',
            'follow_limits':{'timeout_s':args.follow_seconds,'slam_displacement_m':args.max_travel_m}}
    provider=camera=dds=sub=None
    try:
        client=IPlannerClient(args.iplanner_url)
        if args.replay_run:
            projection=load_decision(args.replay_run)
            record=json.loads((args.replay_run/'panorama/panorama.json').read_text())['return_view']
            bgr=cv2.imread(str(args.replay_run/'panorama/current_forward_return.png'))
            frame=ViewFrame('forward',record['frame_id'],record['sim_step'],record['timestamp'],
                            cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),np.load(args.replay_run/'panorama/forward_return_depth_m.npy'),np.array(record['K']))
            path,safe,fear=initial_plan(client,frame,projection.goal_after_turn_xy_m)
            class RecordedPose:
                def get_pose(self):return Pose2D(0.,0.,0.,0.)
            if safe is not None:
                follower=LocalTrajectoryFollower(config,RecordedPose())
                follower.start(safe,safe[-1,:2]);update=follower.update(.05,np.zeros(3))
                report['first_computed_command']=update.command.tolist()
            report.update(status='REPLAY_ONLY',source_run=str(args.replay_run),projection=projection.to_dict(),
                          path=np.asarray(path).tolist(),safe_path=None if safe is None else safe.tolist(),fear=fear)
            return 0
        G3SessionClient.from_decision_url(args.model_url,5.).health_check()
        provider=Ros2OdometryProvider(topic='/unitree/slam_relocation/odom',pose_timeout_s=.5,
                                     preserve_world_coordinates=True,node_name='g1_once_follow_probe')
        frame_id=_wait_for_odometry(provider,3.)
        if frame_id!='map' or provider.child_frame_id!='base_link':
            raise ValueError('需要map → base_link里程计')
        odometry=_SlamPoseSource(provider,frame_id)
        camera=create_camera(args.camera_config)
        frame=camera.capture_forward(0,0.)
        if not client.reset(frame.K.tolist()):raise RuntimeError('iPlanner reset失败')
        if not args.execute:
            report.update(status='CHECK_ONLY',rgb_shape=list(frame.rgb.shape),slam_frame=frame_id,
                          note='只读检查完成；没有实例化运动后端。--execute才会运动')
            print('[CHECK] 相机、模型、iPlanner和SLAM可用；没有发送运动命令。',flush=True)
            return 0
        camera.close();camera=None
        run=args.output/'rotation_model'
        # 子进程可能在旋转中失败/中断，不能在异常报告中写成“没有运动”。
        report.update(robot_motion_sent=None,rotation_process_started=True)
        subprocess.run([sys.executable,str(ROOT/'g1_rotation_model_probe.py'),
                        '--network-interface',args.network_interface,'--instruction',args.instruction,
                        '--camera-config',str(args.camera_config.resolve()),'--model-url',args.model_url,
                        '--output',str(run.resolve()),'--execute'],check=True)
        report['robot_motion_sent']=True
        projection=load_decision(run)
        print(f'[FOLLOW] 新决策已返回，停车显示{args.decision_display_s:.1f}秒；随后仅跟随一次。Ctrl+C取消。',flush=True)
        until=time.monotonic()+args.decision_display_s
        while time.monotonic()<until:
            odometry.get_pose()
            time.sleep(min(.05,max(0.,until-time.monotonic())))
        camera=create_camera(args.camera_config)
        frame=camera.capture_forward(0,0.)
        plan_pose=odometry.get_pose()
        path,safe,fear=initial_plan(client,frame,projection.goal_after_turn_xy_m)
        report.update(projection=projection.to_dict(),path=path.tolist(),safe_path=None if safe is None else safe.tolist(),fear=fear)
        write_json(args.output/'initial_plan.json',report)
        if safe is None:
            report.update(status='NO_FOLLOW',stop_reason='path_inside_stand_off');return 0
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        dds=UnitreeG1DDSBackend(args.network_interface)
        low=StateMonitor();sub=ChannelSubscriber('rt/lowstate',LowState_);sub.Init(low.on_state)
        deadline=time.monotonic()+3
        while True:
            try:low.require_fresh();break
            except RuntimeError:
                if time.monotonic()>deadline:raise
                time.sleep(.05)
        # DDS/订阅器初始化可能耗时接近1秒。此时必须实际再收一帧，不能让旧规划图
        # 的接收时间耗尽watchdog余量，或仅把时间戳改成当前时刻来掩盖断流。
        ready_frame, ready_age=refresh_follow_camera(camera,odometry,plan_pose)
        report.update(follow_ready_frame_id=ready_frame.frame_id,follow_ready_image_age_s=ready_age)
        print(f'[FOLLOW] 起步前新图就绪：frame={ready_frame.frame_id}，接收后{ready_age:.3f}s。',flush=True)
        if args.complete_local_follow:
            print(f'[FOLLOW] 使用仿真共用{args.tracking_controller}；按原局部到达条件结束，无短测位移截止；动作超时{args.follow_seconds}s。Ctrl+C停车。',flush=True)
        else:
            print(f'[FOLLOW] 使用仿真共用{args.tracking_controller}；最多{args.max_travel_m}m / {args.follow_seconds}s。Ctrl+C停车。',flush=True)
        reason,trace=follow(dds,camera,odometry,low,client,safe,
                            config,args,args.output,
                            model_goal=projection.goal_after_turn_xy_m,initial_frame_id=frame.frame_id,initial_fear=fear)
        report.update(status='STOPPED',stop_reason=reason,follow_command_count=len(trace))
        print(f'[FOLLOW] 已请求停车：{reason}；结果：{args.output}',flush=True)
        return 0
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='CANCELLED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=str(error))
        print('[FOLLOW] '+str(error),file=sys.stderr,flush=True)
        return 130 if isinstance(error,KeyboardInterrupt) else 1
    finally:
        try:
            if dds is not None:dds.close()
        finally:
            if sub is not None:sub.Close()
            if camera is not None:camera.close()
            if provider is not None:provider.close()
            write_json(args.output/'follow_report.json',report)


if __name__=='__main__':
    raise SystemExit(main())
