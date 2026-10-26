#!/usr/bin/env python3
"""读取真机结果重算目标几何，不导入运控、不访问相机或模型接口。"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from unified_vln.model_contract import NavigationDecisionResponse
from unified_vln.real_episode import RealG1Episode
from unified_vln.real_target_geometry import build_target_geometry, update_target_for_pose, pose_from_record
from unified_vln.types import ViewFrame


def replay(source):
    # 历史日志没有规划瞬间的精确配对位姿；仅使用首个规划调用前最近控制日志位姿。
    references = {}
    last = None
    for line in (source/'execution_events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['event'] == 'control' and event.get('pose') is not None:
            last = event
        if event['event'] == 'timing' and event.get('kind') == 'iplanner_call_s' and last is not None:
            index = last['decision_index']
            if index not in references:
                references[index] = {'pose': last['pose'], 'frame_id': event['frame_id'],
                    'pose_to_call_end_s': event['monotonic_s']-last['monotonic_s'],
                    'pose_source': 'last control log before first planner-call timing; approximate, not exact image exposure pose'}
    results = []
    for path in sorted(source.glob('decision_*_response.json')):
        index = int(path.name.split('_')[1]); raw = json.loads(path.read_text())
        if raw.get('action') != 'NAVIGATE':
            continue
        direction = raw['direction']; stem = f'decision_{index:03d}'
        capture = json.loads((source/f'{stem}_panorama_capture.json').read_text())['captures'][direction]
        meta = capture['camera_metadata']
        image = cv2.imread(str(source/f'{stem}_{direction}.png'))
        if image is None:
            raise ValueError(f'缺少图像：{stem}/{direction}')
        frame = ViewFrame(direction,capture['frame_id'],capture['sim_step'],capture['timestamp'],
            cv2.cvtColor(image,cv2.COLOR_BGR2RGB),np.load(source/f'{stem}_{direction}_depth_m.npy',allow_pickle=False),np.array(capture['K']))
        response = NavigationDecisionResponse(session_id=raw['session_id'],observation_id=raw['observation_id'],
            action=raw['action'],direction=direction,target=raw['target'],bbox_2d=tuple(raw['bbox_2d']),
            waypoint=raw.get('waypoint'),progress_analysis=raw.get('progress_analysis',''),reasoning=raw.get('reasoning',''))
        projection = RealG1Episode._project_selected_view_target(frame,response)
        geometry = build_target_geometry(frame,projection,meta['T_pelvis_camera'],pose_from_record(capture['pose_after_capture']))
        if index in references:
            ref=references[index]
            geometry=update_target_for_pose(geometry,pose_from_record(ref['pose']),reference_frame_id=ref['frame_id'])
            geometry['replay_reference']=ref
        results.append({'decision_index':index,'target':raw['target'],
            'historical_projection':json.loads((source/f'{stem}_projection.json').read_text()),
            'resampled_projection':projection.to_dict(),'geometry':geometry})
    return {'scope':'offline_saved_data_geometry_only','robot_motion_sent':False,
            'source':str(source.resolve()),'planner_modified':False,'decisions':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('输出文件已存在，请换一个新文件，保留历史结果')
    result=replay(args.run_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for entry in result['decisions']:
        print(f"decision={entry['decision_index']} {entry['target']} "
              f"old={entry['historical_projection']['goal_after_turn_xy_m']} "
              f"new={entry['geometry'].get('target_current_pelvis_xy_m')} "
              f"status={entry['geometry']['status']}")
    print(args.output)

if __name__=='__main__':main()
