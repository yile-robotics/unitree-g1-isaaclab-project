#!/usr/bin/env python3
"""Re-detect both saved RGBs with the original fit/check split; no devices."""
import argparse
import copy
import hashlib
from pathlib import Path
import cv2
import numpy as np
from board import detect, estimate
from calibrate_pair import read, save


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reprocess(source, output):
    source=Path(source).resolve();output=Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; preserve original candidates')
    session=read(source/'session.json');new=copy.deepcopy(session);rows=[]
    output.mkdir(parents=True)
    provenance={'source_session':str(source/'session.json'),'source_session_sha256':digest(source/'session.json'),
                'opencv_version':cv2.__version__,'corrections':['half-pixel-aware resize inverse'],
                'geometry_unchanged':True,'fit_check_roles_unchanged':True,
                'fit_names':[p['name'] for p in session['pairs'] if p['role']=='fit'],
                'check_names':[p['name'] for p in session['pairs'] if p['role']=='check'],
                'code_sha256':{name:digest(Path(__file__).with_name(name)) for name in ['reprocess_saved.py','board.py','geometry.py']},
                'note':'Reprocess saved selected RGB pair only; burst checks are historical acquisition checks, not rerun.'}
    try:
        for pair in new['pairs']:
            record={'name':pair['name'],'role':pair['role']}
            if 'burst' in pair:
                pair['burst_original']=pair.pop('burst')
            if 'stability' in pair:
                pair['stability_original']=pair.pop('stability')
            for side in ['old','new']:
                obs=pair[side];image=(source/obs['image']).resolve();bgr=cv2.imread(str(image))
                if bgr is None:
                    raise ValueError(f'Cannot read {image}')
                if np.any(np.array(obs['metadata']['color_intrinsics']['coeffs'])!=0):
                    raise ValueError('Expected original zero-distortion RGB')
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);K=np.array(obs['metadata']['color_intrinsics']['K'],float)
                markers=detect(rgb,session['board']);pose=estimate(markers,K,session['board'])
                record[side]={'image':str(image),'old_ids':obs['ids'],'new_ids':pose['ids'],
                              'old_reprojection_rms_px':obs['reprojection_rms_px'],
                              'new_reprojection_rms_px':pose['reprojection_rms_px']}
                original={key:obs[key] for key in ['T_camera_board','ids','pixels','object_points','reprojection_rms_px']}
                obs.update(pose,image=str(image),detected_ids=sorted(markers),error=None,original_pose=original,
                           source_image_sha256=digest(image),offline_reprocessed=True)
            rows.append(record)
    except Exception as exc:
        save(output/'reprocess_failed.json',{'error':str(exc),'completed':rows,'provenance':provenance})
        raise
    new['reprocessing']=provenance
    save(output/'session.json',new)
    save(output/'reprocess_report.json',{'provenance':provenance,'pairs':rows,
                                        'source_pairs':len(session['pairs']),'output_pairs':len(new['pairs']),'dropped_pairs':0})
    print(f'[REPROCESS PAIR] {len(rows)} pairs; roles preserved; saved {output}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    reprocess(args.source_session,args.output)


if __name__=='__main__':
    main()
