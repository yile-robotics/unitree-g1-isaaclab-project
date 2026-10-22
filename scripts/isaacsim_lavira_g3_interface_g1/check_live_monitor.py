#!/usr/bin/env python3
"""只读检查 D415、G3 和 iPlanner，不加载 DDS、不发送机器人命令。"""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from camera_d415.network_camera import create_camera
from unified_vln.session_client import G3SessionClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-config', type=Path, default=Path(__file__).parent/
                        'camera_d415/network_camera_d415_local.json')
    parser.add_argument('--model-url', default='http://127.0.0.1:18765/v1/lavira/decision')
    parser.add_argument('--iplanner-url', default='http://127.0.0.1:8888')
    args = parser.parse_args()
    report = {'robot_motion_sent': False}
    try:
        health = G3SessionClient.from_decision_url(args.model_url, 5.).health_check()
        report['model'] = {key:health.get(key) for key in ('status','framework','commit')}
        with urlopen(args.iplanner_url.rstrip('/')+'/health', timeout=5.) as response:
            report['iplanner'] = json.load(response)
        if report['iplanner'].get('status') != 'ok' or report['iplanner'].get('model_loaded') is not True:
            raise RuntimeError('iPlanner health reports model not ready')
        camera = create_camera(args.camera_config)
        try:
            for index in range(5):
                frame = camera.capture_forward(index, 0.)
            report['camera'] = {'rgb_shape':list(frame.rgb.shape), 'depth_shape':list(frame.depth_m.shape),
                                'frame_id':frame.frame_id, **{key:camera.last_metadata[key] for key in
                                ('serial','server_frame_age_s','client_round_trip_s','pair_delta_ms',
                                 'valid_depth_fraction')}}
        finally:
            camera.close()
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='FAILED', error=str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
