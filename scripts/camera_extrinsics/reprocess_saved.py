#!/usr/bin/env python3
"""Re-detect saved contact RGBs without robot/camera access or changing source data."""
import argparse
import copy
import hashlib
from pathlib import Path
import cv2
import numpy as np
from calibrate import read, save
from coded_board import detect_pose, board_point, camera_point
from kinematics import RobotModel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reprocess(source, output):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; source records must remain unchanged')
    session=read(source/'session.json')
    if session.get('sampling_method')!='live_board_per_contact':
        raise ValueError('Expected current-image coded-board contacts')
    model=RobotModel(session['urdf'])
    if model.sha256!=session['urdf_sha256']:
        raise ValueError('URDF changed since acquisition; cannot isolate detector correction')
    output.mkdir(parents=True)
    new=copy.deepcopy(session);new['samples']=[];rows=[]
    provenance={'source_session':str(source/'session.json'),'source_session_sha256':digest(source/'session.json'),
                'opencv_version':cv2.__version__,'raw_acquisition_unchanged':True,
                'corrections':['projective centre for layout and multi-scale checks','half-pixel-aware resize inverse'],
                'geometry_unchanged':True,'probe_unchanged':True,'fit_check_roles_unchanged':True,
                'code_sha256':{name:digest(Path(__file__).with_name(name)) for name in
                              ['reprocess_saved.py','verify_marker_layout.py','coded_board.py','kinematics.py']},
                'note':'Only the saved selected RGB is re-detected. Original acquisition-window diagnostics are retained as original, not rerun.'}
    try:
        for sample in session['samples']:
            image=(source/sample['image']).resolve()
            bgr=cv2.imread(str(image))
            if bgr is None:
                raise ValueError(f'Cannot read {image}')
            meta=sample['metadata'];K=np.array(meta['color_intrinsics']['K'],float)
            if np.any(np.array(meta['color_intrinsics']['coeffs'])!=0):
                raise ValueError('This reprocessing expects original zero-distortion RGB')
            pose=detect_pose(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),K,meta,session['board_config'])
            p=board_point(session['board_config'],sample['marker_id'],sample['contact_corner'].removeprefix('outer_'))
            pc=camera_point(pose,p)
            if sample['state']['mode_pr']!=0:
                raise ValueError('Saved state is not PR joint coordinates')
            pt=model.point('torso_link',session['probe']['probe_link'],session['probe']['tip_in_link_m'],sample['state']['q'])
            if not np.allclose(pt,sample['p_torso_m'],atol=1e-10,rtol=0):
                raise ValueError('Saved FK differs despite unchanged model; inspect before fitting')
            new_sample=copy.deepcopy(sample)
            new_sample['original_pose']=new_sample['pose']
            new_sample['original_p_camera_m']=new_sample['p_camera_m']
            for key in ['pose_window','visual_stability','joint_stability','rejected_frames']:
                if key in new_sample:
                    new_sample[key+'_original']=new_sample.pop(key)
            new_sample.update(image=str(image),pose=pose,p_board_m=p.tolist(),p_camera_m=pc.tolist(),
                              p_torso_m=pt.tolist(),offline_reprocessed=True,source_image_sha256=digest(image))
            new['samples'].append(new_sample)
            oldT=np.array(sample['pose']['T_camera_board']);newT=np.array(pose['T_camera_board'])
            angle=float(np.degrees(np.arccos(np.clip((np.trace(newT[:3,:3]@oldT[:3,:3].T)-1)/2,-1,1))))
            rows.append({'id':sample['marker_id'],'role':sample['role'],'image':str(image),
                         'p_camera_change_mm':float(np.linalg.norm(pc-np.array(sample['p_camera_m']))*1000),
                         'board_rotation_change_deg':angle,'old_ids':sample['pose']['decoded_ids'],
                         'new_ids':pose['decoded_ids'],'new_reprojection_rms_px':pose['reprojection_rms_px']})
    except Exception as exc:
        save(output/'reprocess_failed.json',{'error':str(exc),'completed':rows,'provenance':provenance})
        raise
    new['reprocessing']=provenance
    save(output/'session.json',new)
    save(output/'reprocess_report.json',{'provenance':provenance,'samples':rows,'source_samples':len(session['samples']),
                                        'output_samples':len(new['samples']),'dropped_samples':0})
    print(f'[REPROCESS HAND] {len(rows)} contacts; same roles/FK; saved {output}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reprocess(args.source_session,args.output)


if __name__=='__main__':
    main()
