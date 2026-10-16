#!/usr/bin/env python3
"""Read-only live-board contacts: SPACE arms the current point, Enter records it."""
import argparse
import time
import select
import sys
import os
from pathlib import Path
import cv2
import numpy as np
from calibrate import DEFAULT_CAMERA, read, save, save_image
from coded_board import board_point, camera_point, detect_pose, stability, positions
from robot_state import JointReader, model_for
from kinematics import JOINT_NAMES

HERE = Path(__file__).resolve().parent
FONT_DIR=Path('/usr/share/fonts/truetype/dejavu')
if FONT_DIR.is_dir():
    # OpenCV may overwrite this environment variable during import; set after it.
    os.environ['QT_QPA_FONTDIR']=str(FONT_DIR)
# First two marker rows fit; third row is held out. Start at the central target.
TARGETS = [(14,'fit'),(7,'fit'),(13,'fit'),(19,'fit'),(8,'fit'),(20,'fit'),
           (9,'check'),(15,'check'),(21,'check')]


def terminal_key():
    if sys.stdin.isatty() and select.select([sys.stdin],[],[],0)[0]:
        # Read a burst once: repeated newlines must not remain in readline's
        # buffer and later confirm a different target.
        line=os.read(sys.stdin.fileno(),4096).decode(errors='replace')
        if line.strip().lower()=='q':
            return ord('q')
        if line.strip().lower()=='r':
            return 32
        if line and not line.strip():
            return 13
    return None


class CaptureKeys:
    """A new target or attempt requires a separate ready action."""
    def __init__(self):
        self.armed=False

    def accept(self,key,visible):
        if not visible:
            self.armed=False
            return False
        if key==32:
            self.armed=True
        if key in (10,13) and self.armed:
            self.armed=False
            return True
        return False


def joint_stability(states, side='left'):
    if len(states)<5 or len({s['tick'] for s in states})<5:
        raise ValueError('Insufficient updating joint states')
    if len({(s['mode_machine'],s['mode_pr']) for s in states})!=1:
        raise ValueError('Robot mode changed')
    if any(b['received_monotonic_s']-a['received_monotonic_s']>.2 for a,b in zip(states,states[1:])):
        raise ValueError('Joint stream interrupted')
    names=[n for n in JOINT_NAMES if n.startswith('waist_') or
           (n.startswith(side+'_') and any(x in n for x in ('shoulder','elbow','wrist')))]
    angles=np.array([[s['q'][n] for n in names] for s in states])
    span=float(np.rad2deg(np.max(np.ptp(angles,axis=0))))
    if span>.5:
        raise ValueError(f'Arm/waist moved {span:.3f}deg; retry when still')
    return dict(max_joint_span_deg=span,sample_count=len(states),
                max_reported_dq_rad_s=max(abs(s['dq'][n]) for s in states for n in names))


def paired_state(reader, received):
    reader.latest(.5)
    with reader.lock:
        states=list(reader.samples)
    state=min(states,key=lambda s:abs(s['received_monotonic_s']-received))
    delta=abs(state['received_monotonic_s']-received)
    if delta>.05:
        raise ValueError(f'Image/joint reception gap {delta*1000:.1f}ms >50ms')
    return state,delta


def validate_pairing(records):
    keys={(r['metadata']['color_frame_number'],r['metadata']['depth_frame_number']) for r in records}
    if len(keys)!=len(records):
        raise ValueError('Repeated camera frame during contact')
    if len({(r['state']['mode_machine'],r['state']['mode_pr']) for r in records})!=1:
        raise ValueError('Robot mode changed during contact')


def validate_coverage(records, start, end):
    times=[r['metadata']['client_received_monotonic_s'] for r in records]
    if len(times)<5 or times[-1]-times[0]<.8:
        raise ValueError('Not enough valid current poses covering contact window')
    gaps=[times[0]-start,end-times[-1]]+[b-a for a,b in zip(times,times[1:])]
    if max(gaps)>.35:
        raise ValueError('Valid board pose stream interrupted >350ms')


def capture_contact(camera, reader, config, marker_id, output=None):
    start=time.monotonic();records=[];rejections=[]
    while time.monotonic()-start<1.2:
        frame=camera.capture_forward(0,time.monotonic())
        metadata=dict(camera.last_metadata)
        try:
            pose=detect_pose(frame.rgb,frame.K,metadata,config)
        except (ValueError,cv2.error,np.linalg.LinAlgError) as e:
            rejections.append(dict(error=str(e),metadata=metadata))
            if output is not None:
                save_image(output/'rejected_contact_frame.png',frame.rgb)
                save(output/'last_window_rejections.json',rejections)
            continue
        state,delta=paired_state(reader,metadata['client_received_monotonic_s'])
        records.append(dict(pose=pose,state=state,metadata=metadata,
                            reception_gap_s=delta,rgb=frame.rgb,depth_m=frame.depth_m))
    validate_coverage(records,start,time.monotonic())
    validate_pairing(records)
    target=board_point(config,marker_id)
    visual=stability(records,target)
    with reader.lock:
        states=[s for s in reader.samples if s['received_monotonic_s']>=start]
    joint=joint_stability(states)
    chosen=records[len(records)//2]
    chosen['visual_stability']=visual;chosen['joint_stability']=joint
    chosen['pose_window']=[{k:r[k] for k in ('pose','metadata','reception_gap_s')} for r in records]
    chosen['rejected_frames']=rejections
    return chosen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--board-config',type=Path,default=HERE/'board_marker_measurements.json')
    parser.add_argument('--probe-config',type=Path,required=True)
    parser.add_argument('--allow-model-tip',action='store_true')
    parser.add_argument('--network-interface',default='enxf8e43bbea486')
    parser.add_argument('--camera-config',type=Path,default=DEFAULT_CAMERA)
    args=parser.parse_args()
    config=read(args.board_config);probe=read(args.probe_config)
    if config.get('dictionary')!='APRILTAG_36h11' or not config.get('marker_ids_by_position'):
        parser.error('Need verified AprilTag 36h11 layout')
    evidence=read(config['layout_evidence_path'])
    if evidence['status']!='ID_LAYOUT_CONFIRMED' or evidence['candidate_id_grid']!=config['marker_ids_by_position']:
        parser.error('Board layout differs from confirmation evidence')
    if probe.get('mount_on_torso_confirmed') is not True or probe.get('probe_link')!='left_wrist_yaw_link':
        parser.error('Confirm torso mount and use the configured left wrist frame')
    if probe.get('tip_measured') is not True:
        if not (args.allow_model_tip and probe.get('tip_source')=='model'):
            parser.error('Unmeasured model tip requires --allow-model-tip')
        source=read(probe['model_candidate_path'])
        if not np.allclose(probe['tip_in_link_m'],source['p_left_wrist_yaw_link_m'],atol=1e-12):
            parser.error('Model tip differs from referenced geometry')
    tip=np.asarray(probe['tip_in_link_m'],float)
    if tip.shape!=(3,) or not np.isfinite(tip).all():
        parser.error('Invalid contact offset')
    session_path=args.session/'session.json'
    if session_path.exists():
        session=read(session_path)
        # Allow correction of decoder-only border width in an empty session;
        # never silently change geometry after a contact has been saved.
        previous=dict(session.get('board_config',{}));previous.pop('marker_border_bits',None)
        current=dict(config);current.pop('marker_border_bits',None)
        if not session['samples'] and previous==current:
            session['board_config']=config;save(session_path,session)
        if session.get('sampling_method')!='live_board_per_contact' or session['board_config']!=config or session['probe']!=probe:
            parser.error('Existing session uses a different board/probe/method; choose new directory')
    else:
        if args.session.exists():
            parser.error('Output directory exists without valid session')
        args.session.mkdir(parents=True)
        session=dict(sampling_method='live_board_per_contact',board_config=config,probe=probe,
                     reference={'metadata':{}},samples=[],
                     time_pairing='Nearest joint reception time to image reception, <=50ms; '
                     'quasi-static 1.2s contact. Not hardware exposure-time synchronisation.')
        save(session_path,session)
    from camera_d415.network_camera import create_camera
    camera=None;reader=None;window='D415 extrinsics: SPACE ready | ENTER records | Q exits'
    print('[LIVE TOUCH] 只读，不控制机器人。使用左手中指同一固定点触碰红圈。',flush=True)
    print('[LIVE TOUCH] 模型指尖偏移尚未实测；此流程只产生待验证外参候选。',flush=True)
    try:
        reader=JointReader(args.network_interface);reader.latest(15)
        camera=create_camera(args.camera_config)
        done={(s['marker_id'],s['role']) for s in session['samples']}
        pending=[t for t in TARGETS if t not in done]
        for index,(marker_id,role) in enumerate(pending):
            keys=CaptureKeys()
            print(f'[TARGET] ID {marker_id} 棕色大方块左上外角，{role}；换到这个点并停稳后，图像窗口先按空格，再按 Enter。终端可输入 r 回车准备，再单独回车采集。',flush=True)
            while True:
                frame=camera.capture_forward(0,time.monotonic());meta=dict(camera.last_metadata)
                show=cv2.resize(cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR),None,fx=2,fy=2)
                ready=False
                reason=None
                try:
                    pose=detect_pose(frame.rgb,frame.K,meta,config)
                    p=camera_point(pose,board_point(config,marker_id))
                    xy=cv2.projectPoints(p.reshape(1,3),np.zeros(3),np.zeros(3),frame.K,
                         np.array(meta['color_intrinsics']['coeffs']))[0].reshape(2)*2
                    x,y=np.rint(xy).astype(int)
                    ready=0<=x<show.shape[1] and 0<=y<show.shape[0]
                    if not ready:
                        reason=f'Target projects outside image: x={x/2:.1f}, y={y/2:.1f}'
                    if ready:
                        cv2.circle(show,(x,y),10,(0,0,255),3)
                        cv2.putText(show,f'ID {marker_id} OUTER top-left',(max(5,x-80),max(75,y-20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
                    message=f'Visible {len(pose["decoded_ids"])} tags; RMS {pose["reprojection_rms_px"]:.2f}px'
                except (ValueError,cv2.error,np.linalg.LinAlgError) as e:
                    message='Pose unavailable: '+str(e)
                    reason=message
                cv2.rectangle(show,(0,0),(show.shape[1],62),(20,20,20),-1)
                instruction='ARMED: ENTER records' if keys.armed else 'SPACE ready, then ENTER'
                cv2.putText(show,f'Target ID {marker_id} | {role} | {instruction} | Q exits',
                            (10,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                cv2.putText(show,message,(10,47),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,220,255),1)
                cv2.imshow(window,show);key=cv2.waitKey(1)&255
                terminal=terminal_key()
                if terminal is not None:key=terminal
                if key in (27,ord('q')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                    return 130
                capture_requested=keys.accept(key,ready)
                if key in (10,13) and not ready:
                    save_image(args.session/'rejected_preview.png',frame.rgb)
                    save(args.session/'last_contact_error.json',dict(marker_id=marker_id,error=reason))
                    print(f'[RETRY] Enter已收到，但未保存：{reason}。调整后重试。',flush=True)
                    continue
                if not capture_requested:
                    if key in (10,13):
                        print('[READY] 未采集：先换到当前红圈并停稳，按空格准备，再按Enter。',flush=True)
                    continue
                print('[CONTACT] 保持接触约1.2秒，采当前图像和关节数据……',flush=True)
                try:
                    contact=capture_contact(camera,reader,config,marker_id,args.session)
                    model=model_for(contact['state'])
                    if session.get('urdf_sha256') and session['urdf_sha256']!=model.sha256:
                        raise ValueError('Robot URDF changed within session')
                    pc=camera_point(contact['pose'],board_point(config,marker_id))
                    pt=model.point('torso_link',probe['probe_link'],tip,contact['state']['q'])
                    r,c=positions(config)[marker_id]
                    sample={k:v for k,v in contact.items() if k not in ('rgb','depth_m')}
                    number=len(session['samples'])+1
                    image=f'contact_{number:02d}_id{marker_id}.png'
                    save_image(args.session/image,contact['rgb'])
                    np.save(args.session/f'contact_{number:02d}_depth_m.npy',contact['depth_m'])
                    sample.update(row=r,col=c,marker_id=marker_id,role=role,contact_corner='outer_top_left',
                        p_board_m=board_point(config,marker_id).tolist(),p_camera_m=pc.tolist(),p_torso_m=pt.tolist(),
                        image=image,operator_confirmed_contact_by_enter=True,
                        created_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                        T_pelvis_torso=model.transform('pelvis','torso_link',contact['state']['q']).tolist())
                    session.update(urdf=str(model.path),urdf_sha256=model.sha256,reference={'metadata':contact['metadata']})
                    session['samples'].append(sample);save(session_path,session)
                    print(f'[SAVED] ID{marker_id}，共{len(session["samples"])}条；视觉波动 '
                          f'{contact["visual_stability"]["target_span_m"]*1000:.1f}mm；'
                          f'板朝向估计波动{contact["visual_stability"]["rotation_span_deg"]:.2f}°（诊断）。',flush=True)
                    break
                except (ValueError,RuntimeError,cv2.error,np.linalg.LinAlgError) as e:
                    save(args.session/'last_contact_error.json',dict(marker_id=marker_id,error=str(e)))
                    print(f'[RETRY] 未保存：{e}。停稳并确认红圈位置，先按空格，再按Enter。',flush=True)
        print('[DONE] 9点已保存；待离线求解和验证，不自动应用VLN。',flush=True)
        return 0
    finally:
        if camera is not None:camera.close()
        if reader is not None:reader.close()
        cv2.destroyAllWindows()


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n[CANCEL] 保存记录保留，没有发送机器人命令。');raise SystemExit(130)
