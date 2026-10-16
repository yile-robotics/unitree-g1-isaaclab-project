#!/usr/bin/env python3
"""Estimate a partially visible board from explicitly identified corners. No robot commands."""
import argparse
from pathlib import Path
import time
import numpy as np
from calibrate import DEFAULT_CAMERA, camera_frame, read, save, save_image


def estimate(K,dist,ids,pixels,cols,rows,square):
    import cv2
    ids=np.asarray(ids,dtype=int);pixels=np.asarray(pixels,dtype=float)
    if cols<2 or rows<2 or not np.isfinite(square) or square<=0:raise ValueError('Invalid board geometry')
    if ids.ndim!=2 or ids.shape[1]!=2 or len(ids)<4 or pixels.shape!=(len(ids),2):
        raise ValueError('Need >=4 explicit row,col IDs with matching image pixels')
    if len({tuple(p) for p in ids})!=len(ids):raise ValueError('Duplicate corner IDs')
    if np.any(ids<0) or np.any(ids[:,0]>=rows) or np.any(ids[:,1]>=cols):raise ValueError('Corner ID outside board')
    if not np.isfinite(pixels).all():raise ValueError('Nonfinite pixels')
    obj=np.c_[ids[:,1]*square,ids[:,0]*square,np.zeros(len(ids))]
    if np.linalg.matrix_rank(obj-obj.mean(0),tol=1e-8)<2:raise ValueError('Selected board corners are collinear')
    if np.linalg.matrix_rank(pixels-pixels.mean(0),tol=1e-8)<2:raise ValueError('Selected image corners are collinear')
    ok,rv,tv=cv2.solvePnP(obj,pixels,K,dist,flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:raise ValueError('PnP failed')
    proj=cv2.projectPoints(obj,rv,tv,K,dist)[0].reshape(-1,2)
    rms=float(np.sqrt(np.mean(np.sum((proj-pixels)**2,axis=1))))
    if not np.isfinite(rms):raise ValueError('Nonfinite reprojection error')
    all_obj=np.zeros((rows*cols,3),float)
    all_obj[:,:2]=np.mgrid[0:cols,0:rows].T.reshape(-1,2)*square
    R=cv2.Rodrigues(rv)[0];points=all_obj@R.T+tv.reshape(3)
    if np.any((obj@R.T+tv.reshape(3))[:,2]<=0):raise ValueError('Selected board points behind camera')
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=tv.reshape(3)
    return dict(T_camera_board=T.tolist(),points_camera_m=points.tolist(),
                selected_corner_ids=ids.tolist(),image_corners_px=pixels.tolist(),
                reprojection_rms_px=rms,cols=cols,rows=rows,square_m=square,
                detection='manual_partial',quality=('PASS_REPROJECTION' if rms<=1 else 'REVIEW_REPROJECTION'),
                warning='Only selected corners observed; others inferred from planar metric board. Low reprojection error does not verify labels or flatness.')


def parse_ids(values):
    ids=[]
    for value in values:
        try:
            pair=tuple(map(int,value.split(',')))
            if len(pair)!=2:raise ValueError()
        except ValueError:raise ValueError('Each corner ID must be row,col, e.g. 2,1')
        ids.append(pair)
    if len(ids)<4 or len(set(ids))!=len(ids):raise ValueError('Need >=4 unique corner IDs')
    return ids


def select_pixels(rgb,ids):
    import cv2
    name='Partial board: click requested corner | right-click undo | Enter accept | Q cancel'
    selected=[]
    def mouse(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and len(selected)<len(ids):
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            # Refine only within a small neighbourhood; never repair torn/hidden corners by guessing.
            if 6<=x<rgb.shape[1]-6 and 6<=y<rgb.shape[0]-6:
                p=np.array([[[x,y]]],np.float32)
                cv2.cornerSubPix(gray,p,(5,5),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.001))
                px=p.reshape(2)
                if np.linalg.norm(px-[x,y])>5:
                    print('[CLICK] Refinement moved too far; select a clear actual corner',flush=True);return
                selected.append(px.tolist())
            else:print('[CLICK] Too near image edge; use another corner',flush=True)
        elif event==cv2.EVENT_RBUTTONDOWN and selected:selected.pop()
    cv2.namedWindow(name,cv2.WINDOW_AUTOSIZE);cv2.setMouseCallback(name,mouse)
    try:
        while True:
            show=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            for rc,xy in zip(ids,selected):
                xy=tuple(np.rint(xy).astype(int));cv2.circle(show,xy,4,(0,0,255),1)
                cv2.putText(show,str(tuple(rc)),xy,cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1)
            instruction=f'Click row,col {ids[len(selected)]}' if len(selected)<len(ids) else 'Enter in THIS IMAGE WINDOW to confirm'
            cv2.putText(show,instruction,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.imshow(name,show);key=cv2.waitKey(30)&255
            if key in (27,ord('q')) or cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:raise KeyboardInterrupt()
            if key in (10,13) and len(selected)==len(ids):return selected
    finally:cv2.destroyAllWindows()


def live_freeze(config):
    import cv2
    from camera_d415.network_camera import create_camera
    camera=create_camera(Path(config));name='Board preview: SPACE freeze | Q cancel'
    try:
        while True:
            f=camera.capture_forward(0,time.monotonic());m=dict(camera.last_metadata)
            cv2.imshow(name,cv2.cvtColor(f.rgb,cv2.COLOR_RGB2BGR));key=cv2.waitKey(1)&255
            if key==32:return f.rgb,f.K,m
            if key in (27,ord('q')) or cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1:raise KeyboardInterrupt()
    finally:camera.close();cv2.destroyAllWindows()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cols',type=int,default=5,help='Inner corner columns, NOT squares')
    p.add_argument('--rows',type=int,default=8,help='Inner corner rows, NOT squares')
    p.add_argument('--square-mm',type=float,default=100)
    p.add_argument('--ids',nargs='+',required=True,help='Physical inner corner row,col IDs in click order, zero based')
    p.add_argument('--camera-config',type=Path,default=DEFAULT_CAMERA)
    p.add_argument('--from-check',type=Path,help='Annotate saved check camera_rgb.png instead of live camera')
    p.add_argument('--pixels-json',type=Path,help='Offline pixel pairs matching --ids; requires --from-check')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();ids=parse_ids(args.ids)
    if any(not(0<=r<args.rows and 0<=c<args.cols) for r,c in ids):p.error('Corner ID outside board')
    if args.output.exists():p.error('Output exists; use new directory')
    if args.pixels_json and not args.from_check:p.error('--pixels-json requires --from-check to bind pixels to exact saved image')
    if args.from_check:
        import cv2
        report=read(args.from_check/'check.json')
        m=report['camera']['metadata'];K=np.asarray(m['color_intrinsics']['K'],float)
        image_path=args.from_check/'camera_rgb.png'
        if not image_path.exists():image_path=args.from_check/'rgb.png'
        bgr=cv2.imread(str(image_path))
        if bgr is None:raise ValueError('Saved check image missing')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    else:rgb,K,m=live_freeze(args.camera_config)
    pixels=read(args.pixels_json) if args.pixels_json else select_pixels(rgb,ids)
    if any(not(0<=x<rgb.shape[1] and 0<=y<rgb.shape[0]) for x,y in pixels):raise ValueError('Pixel outside image')
    # Preserve the exact snapshot and clicks even if pose estimation fails.
    args.output.mkdir(parents=True)
    save_image(args.output/'rgb.png',rgb)
    save(args.output/'selected_pixels.json',pixels)
    save(args.output/'selected_ids.json',ids)
    save(args.output/'check.json',{'camera':{'status':'PASS','metadata':m}})
    pose=estimate(K,np.asarray(m['color_intrinsics']['coeffs'],float),ids,pixels,args.cols,args.rows,args.square_mm/1000)
    import cv2
    annotated=rgb.copy()
    for rc,xy in zip(ids,pixels):
        xy=tuple(np.rint(xy).astype(int));cv2.circle(annotated,xy,4,(255,0,0),1)
        cv2.putText(annotated,str(tuple(rc)),xy,cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1)
    save_image(args.output/'corners_numbered.png',annotated)
    save(args.output/'board.json',{'board':pose,'metadata':m,'K':K.tolist(),
                                 'camera_config':read(args.camera_config),'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    print(f'[PARTIAL BOARD] Saved {len(ids)} identified corners; RMS={pose["reprojection_rms_px"]:.3f}px; {args.output}')
    if pose['quality']!='PASS_REPROJECTION':
        print('[REVIEW] RMS exceeds 1px. Snapshot and clicks retained; review corner placement/board flatness before using reference.')
    print('Body/camera/board must remain fixed relative to this reference during later touches. Not an extrinsic yet.')

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\n[CANCEL] No robot command sent.');raise SystemExit(130)
    except Exception as e:print(f'[ERROR] {e}');raise SystemExit(1)
