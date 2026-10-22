#!/usr/bin/env python3
"""按用户确认的base_link=pelvis准备完整G3入口；默认只读，--execute才启动真机。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from urllib.request import urlopen
import uuid

from camera_d415.pelvis_camera import PelvisCamera, load_profile
from g1_camera_model_probe import write_json
from live_model_target import DEFAULT_RESULTS, mark_display_motion
from run_g1_real import _wait_for_odometry
from unified_vln.ros2_odometry import Ros2OdometryProvider
from unified_vln.session_client import G3SessionClient
from unified_vln.real_run_records import atomic_json

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]


def _run_formal_runner(command, log_path):
    # 不用subprocess.run/context manager：其KeyboardInterrupt路径可能kill仍在收尾的子进程。
    with log_path.open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace',
                                 env={**os.environ, 'PYTHONUNBUFFERED':'1'})

        def copy_output():
            for line in child.stdout:
                log.write(line)
                log.flush()
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except (OSError, ValueError):
                    pass

        reader = threading.Thread(target=copy_output, name='formal-runner-log')
        reader.start()
        interrupted = False
        try:
            try:
                code = child.wait()
            except KeyboardInterrupt:
                interrupted = True
                print('[PELVIS VLN] 已请求停车，等待正式程序保存记录；请保留此终端。', flush=True)
                previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    if child.poll() is None:
                        try:
                            child.send_signal(signal.SIGINT)
                        except ProcessLookupError:
                            pass
                    code = child.wait()
                finally:
                    signal.signal(signal.SIGINT, previous)
        finally:
            reader.join()
            child.stdout.close()
            log.flush()
            os.fsync(log.fileno())
    return code, interrupted


def runner_command(args, profile, map_path, session_id):
    return [sys.executable, str(ROOT/'run_g1_real.py'),
            '--instruction', args.instruction, '--session-id', session_id,
            '--model-url', args.model_url, '--iplanner-url', args.iplanner_url,
            '--network-interface', profile['network_interface'],
            '--camera-factory', 'camera_d415.pelvis_camera:create_camera',
            '--camera-config', str(args.profile.resolve()),
            '--odometry-topic', '/unitree/slam_relocation/odom',
            '--map-config', str(map_path.resolve()), '--rotation-duration-scale', '1.4',
            '--rotation-speed-rad-s', '0.8',
            '--walk-speed-m-s', '0.6', '--max-forward-speed-m-s', '0.6',
            '--decision-display-s','2',
            '--tracking-controller', 'pure_pursuit',
            '--max-decisions', str(args.max_decisions),
            '--output-dir', str(args.results.resolve())]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='用户手动启动：多轮全景、决策和行走')
    parser.add_argument('--instruction', default='向前走到一个黑色的行李箱前')
    parser.add_argument('--profile', type=Path, default=ROOT/'camera_d415/pelvis.json')
    parser.add_argument('--model-url', default='http://127.0.0.1:18765/v1/lavira/decision')
    parser.add_argument('--iplanner-url', default='http://127.0.0.1:8888')
    parser.add_argument('--max-decisions', type=int, default=0, help='0沿用原入口：持续至STOP确认或失败；正数为决策上限')
    parser.add_argument('--results', type=Path, default=PROJECT/'outputs/g1_real_unified_vln')
    parser.add_argument('--output', type=Path, default=PROJECT/'outputs/g1_pelvis_vln_prepare'/
                        (datetime.now().strftime('check_%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]))
    return parser


def main(argv=None):
    parser=build_parser();args=parser.parse_args(argv)
    if not args.instruction.strip() or args.max_decisions < 0:
        parser.error('指令不能为空，max-decisions必须>=0')
    if args.output.exists():
        parser.error('--output必须是新目录')
    profile=load_profile(args.profile)
    args.output.mkdir(parents=True)
    report={'robot_motion_sent':False, 'status':'CHECKING',
            'slam_base_relation':'base_link=pelvis',
            'slam_base_verified':profile['slam_base_verified'],
            'slam_base_verification_source':profile['slam_base_verification_source'],
            'pelvis_height_above_floor_m':profile['pelvis_height_above_floor_m']}
    camera=odometry=None
    try:
        health=G3SessionClient.from_decision_url(args.model_url,5.).health_check()
        report['g3_health']={k:health.get(k) for k in ('status','framework','commit')}
        with urlopen(args.iplanner_url.rstrip('/')+'/health', timeout=5.) as response:
            planner=json.load(response)
        if planner.get('status') != 'ok' or planner.get('model_loaded') is not True:
            raise RuntimeError('iPlanner尚未就绪')
        report['iplanner_health']=planner
        camera=PelvisCamera(args.profile)
        odometry=Ros2OdometryProvider(topic='/unitree/slam_relocation/odom',pose_timeout_s=.5,
                                      preserve_world_coordinates=True,node_name='g1_pelvis_vln_check')
        frame_id=_wait_for_odometry(odometry,5.)
        camera.validate_slam_frame(frame_id,odometry.child_frame_id)
        frame=camera.capture_forward(0,0.)
        pose=odometry.get_pose()
        if pose is None:
            raise RuntimeError('SLAM位姿过期')
        config=camera.nominal_config
        exploration=camera.create_exploration_map(config,pose_frame_id=frame_id)
        integration=exploration.integrate(frame,pose)
        map_path=args.output/'map_config.json'
        write_json(map_path,asdict(config))
        write_json(args.output/'geometry_assumptions.json',camera.provenance())
        write_json(args.output/'camera_metadata.json',camera.last_metadata)
        report.update(status='PREFLIGHT_PASS', rgb_shape=list(frame.rgb.shape),
                      slam_frame=frame_id, slam_child_frame=odometry.child_frame_id,
                      map_integration=integration.to_dict())
        session_id='g1_pelvis_'+args.output.name
        command=runner_command(args,profile,map_path,session_id)
        write_json(args.output/'runner_command.json',{'argv':command})
        print('[PELVIS CHECK] 相机、腰关节、SLAM、G3、iPlanner及地图坐标换算检查通过。',flush=True)
        print(f"[PELVIS CHECK] base_link=pelvis已由用户确认；名义骨盆离地高度约{profile['pelvis_height_above_floor_m']:.2f}m，非实时测高。",flush=True)
        print('[PELVIS CHECK] 相机外参仍保留候选状态；身份确认不代表相机标定精度已验证。',flush=True)
        print(f'[PELVIS CHECK] 记录：{args.output}',flush=True)
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='CANCELLED' if isinstance(error,KeyboardInterrupt) else 'FAILED',error=str(error))
        print(f'[PELVIS CHECK] {error}',file=sys.stderr,flush=True)
        return 130 if isinstance(error,KeyboardInterrupt) else 1
    finally:
        try:
            if camera is not None:camera.close()
        finally:
            if odometry is not None:odometry.close()
            write_json(args.output/'preflight_report.json',report)
    if not args.execute:
        print('[PELVIS CHECK] 仅只读检查；没有创建运动客户端。--execute才会启动真机。',flush=True)
        return 0
    mark_display_motion(DEFAULT_RESULTS)
    mark_display_motion(args.results)
    report.update(status='RUNNER_STARTED',robot_motion_sent=None)
    write_json(args.output/'preflight_report.json',report)
    print('[PELVIS VLN] 启动原正式G3状态机；Ctrl+C请求停车。',flush=True)
    try:
        code, interrupted = _run_formal_runner(command,args.output/'runner_console.log')
        report.update(status='CANCELLED' if interrupted or code in (130,143) else 'RUNNER_EXITED',
                      runner_returncode=code, runner_cleanup_waited=True,
                      console_log=str(args.output/'runner_console.log'))
        return 130 if interrupted else code
    except KeyboardInterrupt:
        report.update(status='CANCELLED',error='Ctrl+C; formal runner handles its own motor cleanup')
        return 130
    finally:
        # 子进程退出码不等于语义成功，最终结论看原状态机的STOP_CONFIRMED/会话结束记录。
        atomic_json(args.output/'preflight_report.json',report)


if __name__ == '__main__':
    raise SystemExit(main())
