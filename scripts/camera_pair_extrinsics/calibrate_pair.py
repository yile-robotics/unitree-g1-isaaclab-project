#!/usr/bin/env python3
"""双相机独立采集/求解入口。只读图，不导入机器人运动或手触碰工具。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from board import detect, estimate
from geometry import angle, fit, validate, compose

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'isaacsim_lavira_g3_interface_g1'))
from camera_d415.network_camera import create_camera


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


class PairCamera:
    def __init__(self, old, new, board):
        self.cameras = []; self.pool = ThreadPoolExecutor(max_workers=2); self.board = board
        try:
            for config in (old,new):
                self.cameras.append(create_camera(config))
        except BaseException:
            self.close(); raise

    def close(self):
        for camera in self.cameras:
            camera.close()
        self.pool.shutdown(wait=True, cancel_futures=True)

    def next(self):
        start = time.monotonic()
        futures = [self.pool.submit(c.capture_forward,0,start) for c in self.cameras]
        frames = []
        for side, future in zip(('old','new'), futures):
            try:
                frames.append(future.result())
            except (ValueError,OSError) as exc:
                raise ValueError(f'{side} camera: {exc}') from exc
        pair = {'request_start_monotonic_s':start}
        for side, frame, camera in zip(('old','new'),frames,self.cameras):
            markers = detect(frame.rgb, self.board)
            obs = {'metadata':dict(camera.last_metadata), 'frame_id':frame.frame_id,
                   'K':frame.K.tolist(), 'detected_ids':sorted(markers)}
            try:
                obs.update(estimate(markers,frame.K,self.board))
                obs['error'] = None
            except (ValueError, np.linalg.LinAlgError) as exc:
                obs['error'] = str(exc)
            pair[side] = obs
        pair['receive_gap_ms'] = abs(pair['old']['metadata']['client_received_monotonic_s']-
                                     pair['new']['metadata']['client_received_monotonic_s'])*1000
        return frames, pair


def stable(records):
    if len(records) < 5:
        raise ValueError('Need 5 fresh pairs')
    if records[-1]['request_start_monotonic_s']-records[0]['request_start_monotonic_s'] < .8:
        raise ValueError('Hold the board still for at least 0.8 seconds')
    for side in ('old','new'):
        for p in records:
            if p[side]['error']:
                raise ValueError(f'{side}: {p[side]["error"]}')
        poses = [np.asarray(p[side]['T_camera_board']) for p in records]
        shared = set.intersection(*[set(p[side]['ids']) for p in records])
        if len(shared) < 6:
            raise ValueError(f'{side}: need 6 stable IDs throughout the burst')
        pixel_lookup = [{i:xy for i,xy in zip(p[side]['ids'],p[side]['pixels'])} for p in records]
        max_pixel_span = max(float(np.linalg.norm(np.asarray(a[i])-b[i])) for a in pixel_lookup for b in pixel_lookup for i in shared)
        max_rotation = max(angle(a[:3,:3]@b[:3,:3].T) for a in poses for b in poses)
        if max_pixel_span > 2 or max_rotation > 2:
            raise ValueError(f'{side}: board/camera moved or pose unstable: {max_pixel_span:.1f}px, {max_rotation:.1f}deg')
    return {'max_receive_gap_ms':max(p['receive_gap_ms'] for p in records),
            'burst_duration_s':records[-1]['request_start_monotonic_s']-records[0]['request_start_monotonic_s'],
            'note':'Concurrent software requests, not hardware exposure synchronization; static board required'}


def display(frames, pair, role, armed, count, preview_only=False):
    images = []
    for side, frame in zip(('old','new'),frames):
        image = cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR)
        obs = pair[side]
        for marker_id, xy in zip(obs.get('ids',[]),obs.get('pixels',[])):
            point = tuple(np.rint(xy).astype(int))
            cv2.circle(image,point,3,(0,255,0),-1)
            cv2.putText(image,str(marker_id),point,cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1)
        cv2.rectangle(image,(0,0),(image.shape[1],65),(15,15,15),-1)
        label = 'OLD robot D435i' if side=='old' else 'NEW D415'
        cv2.putText(image,f'{label} | decoded={len(obs["detected_ids"])} valid={len(obs.get("ids",[]))}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.putText(image,(obs['error'] or 'Board pose ready')[:85],(8,48),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,220,255),1)
        images.append(image)
    canvas = np.hstack(images)
    banner = np.zeros((40,canvas.shape[1],3),np.uint8)
    instruction = ('Preview only | adjust board until BOTH poses ready | Q: exit' if preview_only else
                   f'{count} saved | role={role} | '+('ENTER: hold still + save' if armed else 'SPACE: arm capture')+' | V: toggle fit/check | Q: exit')
    cv2.putText(banner,instruction,
                (8,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    return np.vstack((banner,canvas))


def stream(args):
    board = read(args.board); camera = None
    session = {'schema':1, 'board':board, 'old_camera':read(args.old_camera), 'new_camera':read(args.new_camera),
               'pairs':[], 'method':'stationary dual RGB coded board; no hand contacts',
               'synchronization':'concurrent TCP requests; no hardware exposure synchronization'}
    output = args.output
    if output and output.exists():
        raise ValueError('Output already exists; choose a new directory')
    role = 'fit'; armed = False; deadline = time.monotonic()+args.seconds
    window = ('Dual camera preview: OLD left | NEW right | Q exits' if args.command=='check' else
              'Dual camera capture: OLD left | NEW right | SPACE then Enter')
    try:
        camera = PairCamera(args.old_camera,args.new_camera,board)
        if output:
            output.mkdir(parents=True)
            save(output/'session.json',session)
        print('[PAIR] 左=原装D435i；右=D415。只读图，不发运动命令。',flush=True)
        print('[PAIR] 固定两台相机，只移动标定板。'+('当前仅预览，Q退出。' if args.command=='check' else
              '图像窗口 Space 准备，Enter 保存；V 切换独立验证样本。'),flush=True)
        next_log = 0
        while time.monotonic() < deadline:
            frames, pair = camera.next()
            if time.monotonic() >= next_log:
                print('[PAIR] '+ ' | '.join(f'{side}: frame={pair[side]["frame_id"]} IDs={len(pair[side].get("ids",[]))} '+
                       (pair[side]['error'] or 'ready') for side in ('old','new')),flush=True)
                next_log = time.monotonic()+5
            if args.command == 'check' and output:
                for side, frame in zip(('old','new'),frames):
                    cv2.imwrite(str(output/f'{side}_rgb.png'),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
                save(output/'check.json',pair)
                cv2.imwrite(str(output/'preview.jpg'),display(frames,pair,role,armed,0,preview_only=True))
            if args.headless:
                continue
            cv2.imshow(window,display(frames,pair,role,armed,len(session['pairs']),preview_only=args.command=='check'))
            key = cv2.waitKey(1)&255
            if key in (27,ord('q')) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                break
            if args.command != 'capture':
                continue
            if key == ord('v'):
                role = 'check' if role=='fit' else 'fit'; armed=False
                print(f'[PAIR] 下一组 role={role}',flush=True)
            if key == 32:
                armed = True
                print('[PAIR] 已准备：板和相机停稳，在图像窗口 Enter 采集。',flush=True)
            if key in (10,13) and armed:
                armed = False
                print('[PAIR] 保持静止，采集5组新图…',flush=True)
                burst = []
                for _ in range(5):
                    frames,pair = camera.next(); burst.append(pair); time.sleep(.2)
                try:
                    quality = stable(burst)
                except ValueError as exc:
                    print(f'[RETRY] 未保存：{exc}；重新 Space 再 Enter。',flush=True)
                    continue
                index = len(session['pairs'])+1
                pair.update(name=f'pair_{index:03d}',role=role,stability=quality,burst=burst[:-1])
                for side, frame in zip(('old','new'),frames):
                    filename=f'{pair["name"]}_{side}_rgb.png'
                    if not cv2.imwrite(str(output/filename),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR)):
                        raise OSError(f'Cannot write {filename}')
                    pair[side]['image']=filename
                    np.save(output/f'{pair["name"]}_{side}_depth_m.npy',frame.depth_m)
                session['pairs'].append(pair); save(output/'session.json',session)
                print(f'[SAVED] {pair["name"]} role={role}；移动/倾斜板后继续。',flush=True)
                for _ in range(5):
                    cv2.waitKey(1)
    except BaseException as exc:
        if output and output.exists():
            save(output/'interruption.json',{'error':str(exc),'type':type(exc).__name__,
                                            'saved_pairs':len(session['pairs'])})
        raise
    finally:
        if camera:
            camera.close()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


def solve(args):
    session = read(args.session/'session.json')
    fitting = [p for p in session['pairs'] if p['role']=='fit']
    checking = [p for p in session['pairs'] if p['role']=='check']
    if len(checking)<3:
        raise ValueError('Capture at least 3 separate check views (V toggles role); these are not fitted')
    T, diagnostics = fit(fitting)
    report = validate(T,checking)
    candidate = {'schema':1,'status':'RELATIVE_CANDIDATE', 'parent_frame':'old_rgb_optical','child_frame':'new_rgb_optical',
                 'old_camera_serial':session['old_camera']['serial'],'new_camera_serial':session['new_camera']['serial'],
                 'T_old_rgb_new_rgb':T.tolist(), 'translation_m':T[:3,3].tolist(),
                 'rotation_vector_rad':cv2.Rodrigues(T[:3,:3])[0].ravel().tolist(),
                 'fit':diagnostics,'held_out_validation':report,'board':session['board'],'session':str(args.session.resolve()),
                 'torso_reference_verified':False,'applied_to_vln':False,
                 'notes':'Both RGB optical frames: X right, Y down, Z forward. Transform maps new coordinates into old. Validation uses independently estimated board poses, not physical ground truth.'}
    save(args.session/'relative_candidate.json',candidate)
    print(f'[SOLVE] fit={len(fitting)} check={len(checking)}；joint RMS={diagnostics["joint_reprojection_rms_px"]:.2f}px',flush=True)
    print(f'[CHECK] centre RMS={report["center_rms_mm"]:.1f}mm max={report["max_center_error_mm"]:.1f}mm；候选未应用VLN。',flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    for command in ('check','capture'):
        p = sub.add_parser(command)
        p.add_argument('--old-camera',type=Path,default=HERE/'old_camera.json')
        p.add_argument('--new-camera',type=Path,default=HERE/'new_camera.json')
        p.add_argument('--board',type=Path,default=HERE/'board.json')
        p.add_argument('--seconds',type=float,default=600 if command=='capture' else 30)
        p.add_argument('--output',type=Path,required=command=='capture')
        p.add_argument('--headless',action='store_true')
    p = sub.add_parser('solve'); p.add_argument('--session',type=Path,required=True)
    p = sub.add_parser('validate'); p.add_argument('--candidate',type=Path,required=True); p.add_argument('--session',type=Path,required=True)
    p = sub.add_parser('compose'); p.add_argument('--candidate',type=Path,required=True); p.add_argument('--reference',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try:
        if args.command in ('check','capture'):
            if not np.isfinite(args.seconds) or args.seconds<=0:
                parser.error('seconds must be positive and finite')
            if args.command=='capture' and args.headless:
                parser.error('capture needs the image window for Space / Enter')
            return stream(args)
        if args.command=='solve':
            return solve(args)
        candidate = read(args.candidate)
        T = candidate['T_old_rgb_new_rgb']
        if args.command=='validate':
            session = read(args.session/'session.json')
            original = Path(candidate['session']).resolve()
            if original==args.session.resolve():
                pairs = [p for p in session['pairs'] if p['role']=='check']
            else:
                pairs = session['pairs']
            if not pairs:
                raise ValueError('No independent validation views')
            if (candidate['old_camera_serial'],candidate['new_camera_serial']) != (session['old_camera']['serial'],session['new_camera']['serial']):
                raise ValueError('Camera serials differ')
            if candidate['board'] != session['board']:
                raise ValueError('Board layout or dimensions differ')
            report = validate(T,pairs); save(args.session/'relative_validation.json',report)
            print(json.dumps(report,indent=2)); return 0
        if args.output.exists():
            raise ValueError('Output exists; choose a new file')
        reference = read(args.reference)
        result = compose(reference,T,candidate['old_camera_serial'])
        save(args.output,{'status':'COMPOSED_CANDIDATE','parent_frame':'torso_link','child_frame':'new_rgb_optical',
                          'T_torso_new_rgb':result.tolist(),'relative_candidate':str(args.candidate.resolve()),
                          'reference':reference,'applied_to_vln':False})
        print(f'[COMPOSE] saved {args.output}；未应用VLN。'); return 0
    except KeyboardInterrupt:
        print('[PAIR] Ctrl+C：关闭相机连接，已保存数据保留。'); return 130
    except (ValueError,OSError,KeyError) as exc:
        print(f'[PAIR ERROR] {exc}',file=sys.stderr); return 1


if __name__=='__main__':
    raise SystemExit(main())
