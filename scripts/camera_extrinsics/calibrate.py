#!/usr/bin/env python3
"""Read-only D415 / G1 touch-point extrinsic calibration, independent of VLN."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
from kinematics import fit_rigid
from robot_state import JointReader, model_for

HERE=Path(__file__).resolve().parent
INTERFACE_DIR=HERE.parent/'isaacsim_lavira_g3_interface_g1'
sys.path.insert(0,str(INTERFACE_DIR))
DEFAULT_CAMERA=INTERFACE_DIR/'camera_d415/network_camera_d415_local.json'

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def read(path):return json.loads(Path(path).read_text())

def camera_frame(config):
    from camera_d415.network_camera import create_camera
    camera=create_camera(Path(config))
    try:
        frame=camera.capture_forward(0,time.monotonic())
        return frame,camera.last_metadata
    finally:camera.close()

def save_image(path,rgb):
    import cv2
    if not cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise IOError(f'Cannot save {path}')

def board_pose(rgb,K,dist,cols,rows,square):
    import cv2
    if cols<3 or rows<3 or not np.isfinite(square) or square<=0:raise ValueError('Invalid board dimensions')
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    ok,corners=cv2.findChessboardCornersSB(gray,(cols,rows),flags=cv2.CALIB_CB_EXHAUSTIVE|cv2.CALIB_CB_ACCURACY)
    if not ok:raise RuntimeError('Full chessboard inner-corner grid not detected; check grid size, focus, visibility and flatness')
    obj=np.zeros((rows*cols,3),np.float64)
    obj[:,:2]=np.mgrid[0:cols,0:rows].T.reshape(-1,2)*square
    ok,rvec,tvec=cv2.solvePnP(obj,corners,K,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:raise RuntimeError('PnP failed')
    R=cv2.Rodrigues(rvec)[0];points=obj@R.T+tvec.reshape(3)
    if np.any(points[:,2]<=0):raise RuntimeError('PnP board behind camera')
    projected=cv2.projectPoints(obj,rvec,tvec,K,dist)[0].reshape(-1,2)
    error=float(np.sqrt(np.mean(np.sum((projected-corners.reshape(-1,2))**2,axis=1))))
    if error>1.0:raise RuntimeError(f'Reprojection RMS {error:.2f}px >1px; inspect board/intrinsics')
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=tvec.reshape(3)
    annotated=rgb.copy()
    for i,(x,y) in enumerate(corners.reshape(-1,2)):
        cv2.circle(annotated,(round(float(x)),round(float(y))),3,(255,0,0),-1)
        cv2.putText(annotated,f'{i//cols},{i%cols}',(round(float(x))+3,round(float(y))-3),
                    cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1,cv2.LINE_AA)
    return dict(T_camera_board=T.tolist(),points_camera_m=points.tolist(),
                image_corners_px=corners.reshape(-1,2).tolist(),reprojection_rms_px=error,
                cols=cols,rows=rows,square_m=square),annotated

def check(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    report={'read_only':True,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    try:
        f,m=camera_frame(args.camera_config)
        save_image(out/'camera_rgb.png',f.rgb)
        np.save(out/'camera_depth_m.npy',f.depth_m)
        report['camera']={'status':'PASS','metadata':m,'rgb_shape':list(f.rgb.shape)}
        print(f'[CAMERA] PASS serial={m["serial"]} RGB={f.rgb.shape} valid_depth={m["valid_depth_fraction"]:.1%}')
    except Exception as e:
        report['camera']={'status':'FAIL','error':str(e)};print('[CAMERA] FAIL',e)
    reader=None
    try:
        reader=JointReader(args.network_interface);s=reader.latest(args.wait)
        report['robot']={'status':'PASS','state':s}
        print(f'[JOINTS] PASS mode_machine={s["mode_machine"]} mode_pr={s["mode_pr"]}')
        model=model_for(s)
        report['robot']['urdf']=str(model.path);report['robot']['urdf_sha256']=model.sha256
        report['robot']['T_pelvis_torso']=model.transform('pelvis','torso_link',s['q']).tolist()
        for side in ('left','right'):
            link=side+'_wrist_yaw_link'
            print(f'[FK] {link} origin in torso = {model.point("torso_link",link,[0,0,0],s["q"])} m (NOT probe tip)')
        print('[FK] Model selected from mode_machine; physical mount/probe still require confirmation')
    except Exception as e:
        report['robot']={'status':'FAIL','error':str(e)};print('[JOINTS/FK] FAIL',e)
    finally:
        if reader:reader.close()
    save(out/'check.json',report)
    return 0 if all(report[k]['status']=='PASS' for k in ('camera','robot')) else 1

def watch(args):
    reader=JointReader(args.network_interface)
    try:
        state=reader.latest(args.wait);model=model_for(state)
        print(f'[WATCH] Read only: {model.path.name}; wrist origins are NOT probe tips; Ctrl+C exits')
        end=time.monotonic()+args.seconds
        while time.monotonic()<end:
            state=reader.latest(.5)
            if model_for(state).sha256!=model.sha256:raise ValueError('Robot model changed')
            waist=[np.rad2deg(state['q'][f'waist_{axis}_joint']) for axis in ('yaw','roll','pitch')]
            left=model.point('torso_link','left_wrist_yaw_link',[0,0,0],state['q'])
            right=model.point('torso_link','right_wrist_yaw_link',[0,0,0],state['q'])
            print(f'{time.strftime("%H:%M:%S")} waist yaw/roll/pitch={np.round(waist,1)} deg; '
                  f'left wrist={np.round(left,4)} m right wrist={np.round(right,4)} m',flush=True)
            time.sleep(.5)
    finally:reader.close()

def board(args):
    out=args.output
    if out.exists():raise ValueError('Output exists; use a new reference directory')
    f,m=camera_frame(args.camera_config)
    out.mkdir(parents=True)
    save_image(out/'rgb.png',f.rgb)
    pose,annotated=board_pose(f.rgb,f.K,np.asarray(m['color_intrinsics']['coeffs']),args.cols,args.rows,args.square_mm/1000)
    save_image(out/'corners_numbered.png',annotated)
    record={'board':pose,'metadata':m,'camera_config':read(args.camera_config),'K':f.K.tolist(),
            'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    save(out/'board.json',record)
    print(f'[BOARD] PASS RMS={pose["reprojection_rms_px"]:.3f}px; inspect {out/"corners_numbered.png"}')
    print('[BOARD] Row,col labels in this image define contact points. This is not a camera-to-body extrinsic yet.')

def new_session(args):
    config=read(args.probe_config)
    using_model = config.get('tip_source') == 'model' and getattr(args,'allow_model_tip',False)
    if config.get('mount_on_torso_confirmed') is not True:
        raise ValueError('Confirm physical torso mount in probe_config first')
    if config.get('tip_measured') is not True and not using_model:
        raise ValueError('Need measured tip, or explicit --allow-model-tip with tip_source=model for provisional calibration')
    if using_model:
        source=read(config['model_candidate_path'])
        if not np.allclose(config['tip_in_link_m'],source['p_left_wrist_yaw_link_m'],atol=1e-12):
            raise ValueError('Model tip differs from referenced candidate')
        print('[MODEL TIP] Provisional model-derived contact offset; not physically measured.')
    tip=np.asarray(config.get('tip_in_link_m'),float)
    if tip.shape!=(3,) or not np.isfinite(tip).all():raise ValueError('tip_in_link_m must be 3 measured values in metres')
    if config.get('probe_link') not in ('left_wrist_yaw_link','right_wrist_yaw_link'):
        raise ValueError('Use wrist_yaw_link as probe frame; rigid probe must not depend on movable fingers')
    if not args.reference_fixed:raise ValueError('Need --reference-fixed: camera and board must remain fixed relative to each other')
    ref_path=Path(args.board_reference).resolve();ref=read(ref_path)
    if not (ref_path.parent/'corners_numbered.png').exists():raise ValueError('Numbered reference image missing')
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    save(out/'session.json',{'reference':ref,'reference_path':str(ref_path),'reference_sha256':hashlib.sha256(ref_path.read_bytes()).hexdigest(),
                            'probe':config,'tip_source':('model' if using_model else 'measured'),'reference_fixed_asserted':True,'samples':[]})
    print('[SESSION] Created. Inspect numbered reference before touching. Arm motion may occlude the board; body/camera/board must not move.')

def touch(args):
    if not args.reference_fixed:raise ValueError('Need --reference-fixed; if reference moved create a new board reference/session')
    path=args.session/'session.json';session=read(path);b=session['reference']['board']
    if not(0<=args.row<b['rows'] and 0<=args.col<b['cols']):raise ValueError('Corner row,col outside reference')
    reader=JointReader(args.network_interface)
    try:
        reader.latest(args.wait)
        print(f'[CONTACT] Hold the configured contact point on reference corner ({args.row},{args.col}); role={args.role}.')
        input('Press Enter IN THIS TERMINAL when contact is steady (Ctrl+C cancels): ')
        side=session['probe']['probe_link'].split('_')[0]
        s=reader.stationary(side=side);model=model_for(s)
        print(f'[CONTACT] joint span={s["max_joint_span_deg"]:.4f}deg; reported speed peak={s["max_reported_dq_rad_s"]:.4f}rad/s')
        if session.get('urdf_sha256') and session['urdf_sha256']!=model.sha256:raise ValueError('Robot model changed within session')
        p=model.point('torso_link',session['probe']['probe_link'],session['probe']['tip_in_link_m'],s['q'])
        sample={'row':args.row,'col':args.col,'role':args.role,'p_camera_m':b['points_camera_m'][args.row*b['cols']+args.col],
                'p_torso_m':p.tolist(),'state':s,'T_pelvis_torso':model.transform('pelvis','torso_link',s['q']).tolist(),
                'reference_fixed_asserted':True,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        session['urdf']=str(model.path);session['urdf_sha256']=model.sha256;session['samples'].append(sample)
        save(path,session);print(f'[CONTACT] SAVED {len(session["samples"])}; torso tip={p} m')
    finally:reader.close()

def solve(args):
    s=read(args.session/'session.json')
    fit=[p for p in s['samples'] if p['role']=='fit'];checks=[p for p in s['samples'] if p['role']=='check']
    fit_ids={(p['row'],p['col']) for p in fit}
    if len(fit_ids)<6:raise ValueError('Need >=6 different fit corners; recommend 8-12')
    if len({(p['row'],p['col']) for p in checks}-fit_ids)<3:
        raise ValueError('Need >=3 distinct held-out check corners not used for fitting')
    a=np.array([p['p_camera_m'] for p in fit]);b=np.array([p['p_torso_m'] for p in fit]);T=fit_rigid(a,b)
    def errors(points):
        return np.array([np.linalg.norm(T[:3,:3]@np.array(p['p_camera_m'])+T[:3,3]-p['p_torso_m']) for p in points])
    ef,ec=errors(fit),errors(checks)
    result={'status':'CANDIDATE_REQUIRES_PHYSICAL_VALIDATION','target_frame':'torso_link','source_frame':'color_optical',
            'convention':'p_torso = T_torso_camera @ p_camera; column homogeneous coordinates; metres',
            'T_torso_camera':T.tolist(),'T_camera_torso':np.linalg.inv(T).tolist(),'fit_rms_m':float(np.sqrt(np.mean(ef**2))),
            'check_rms_m':float(np.sqrt(np.mean(ec**2))),'check_max_m':float(ec.max()),
            'fit_errors_m':ef.tolist(),'check_errors_m':ec.tolist(),'urdf':s['urdf'],'urdf_sha256':s['urdf_sha256'],
            'probe':s['probe'],'reference_camera':s['reference']['metadata'],
            'sampling_method':s.get('sampling_method','fixed_board_reference'),
            'time_pairing':s.get('time_pairing','static fixed-reference collection'),
            'notes':
            'Contact offset source is recorded in probe config; model offsets require physical verification. Assumes correct URDF/joint mapping, flat metric board, fixed camera-board relationship. '
            'Not auto-installed in VLN. Dynamic pelvis transform requires time-matched waist state; SLAM frame unverified.'}
    if s.get('sampling_method')=='live_board_per_contact':
        result['board_config']=s['board_config']
        result['notes']='Each contact uses its own camera-board pose and quasi-static image/joint reception pairing. Camera-board pose may change between contacts; camera must remain rigid relative to torso. Model contact offset and physical metric accuracy require independent validation. Not auto-installed in VLN; hardware exposure synchronisation and SLAM body frame remain unverified.'
    save(args.session/'extrinsics_candidate.json',result)
    print(f'[SOLVE] fit RMS={result["fit_rms_m"]*1000:.1f}mm; held-out RMS={result["check_rms_m"]*1000:.1f}mm max={result["check_max_m"]*1000:.1f}mm')
    print('[SOLVE] Saved candidate; inspect and validate at other board poses/distances before VLN use')

def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name,fun in [('check',check),('watch',watch),('board',board),('new-session',new_session),('touch',touch),('solve',solve)]:
        a=sub.add_parser(name);a.set_defaults(run=fun)
        if name in ('check','board'):a.add_argument('--camera-config',type=Path,default=DEFAULT_CAMERA)
        if name in ('check','touch','watch'):
            a.add_argument('--network-interface',default='enxf8e43bbea486');a.add_argument('--wait',type=float,default=8)
        if name in ('check','board','new-session'):a.add_argument('--output',type=Path,required=True)
        if name=='watch':a.add_argument('--seconds',type=float,default=30)
        if name=='board':
            a.add_argument('--cols',type=int,required=True);a.add_argument('--rows',type=int,required=True)
            a.add_argument('--square-mm',type=float,required=True)
        if name=='new-session':
            a.add_argument('--board-reference',type=Path,required=True)
            a.add_argument('--probe-config',type=Path,default=HERE/'probe.example.json')
            a.add_argument('--allow-model-tip',action='store_true',help='Use an explicitly model-derived tip as provisional, without marking it measured')
        if name in ('new-session','touch'):a.add_argument('--reference-fixed',action='store_true')
        if name in ('touch','solve'):a.add_argument('--session',type=Path,required=True)
        if name=='touch':
            a.add_argument('--row',type=int,required=True);a.add_argument('--col',type=int,required=True)
            a.add_argument('--role',choices=['fit','check'],required=True)
    return p

if __name__=='__main__':
    try:
        args=parser().parse_args();raise SystemExit(args.run(args) or 0)
    except KeyboardInterrupt:
        print('\n[CANCEL] Read-only process ended; no motion command was sent.');raise SystemExit(130)
    except Exception as e:
        print(f'[ERROR] {e}',file=sys.stderr)
        if 'args' in globals() and getattr(args,'command',None)=='touch':
            save(args.session/'last_contact_error.json',{'error':str(e),'type':type(e).__name__,
                 'row':args.row,'col':args.col,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
        raise SystemExit(1)
