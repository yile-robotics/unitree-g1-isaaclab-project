#!/usr/bin/env python3
"""Evaluate an unchanged camera extrinsic on separately acquired contacts."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from calibrate import read, save


def evaluate(candidate, session, original=None):
    if candidate['target_frame'] != 'torso_link' or candidate['source_frame'] != 'color_optical':
        raise ValueError('Expected color_optical to torso_link transform')
    for key in ('urdf_sha256', 'probe', 'board_config'):
        if candidate.get(key) is None or candidate[key] != session.get(key):
            raise ValueError(f'Candidate and validation {key} differ')
    if session.get('sampling_method') != 'live_board_per_contact':
        raise ValueError('Use live_board_per_contact validation samples')
    samples = session['samples']
    if len({(s['row'], s['col']) for s in samples}) < 3:
        raise ValueError('Need at least 3 different validation corners')
    serial = candidate['reference_camera']['serial']
    def frame_identity(s):
        m = s['metadata']
        return (m['serial'], m['color_timestamp_ms'], m['depth_timestamp_ms'])
    if any(s['metadata']['serial'] != serial for s in samples):
        raise ValueError('Validation camera serial differs')
    keys = [frame_identity(s) for s in samples]
    if len(set(keys)) != len(keys):
        raise ValueError('Repeated validation camera exposure')
    if original and set(keys) & {frame_identity(s) for s in original['samples']}:
        raise ValueError('Original calibration exposures reused; acquire new contacts')
    T = np.asarray(candidate['T_torso_camera'], float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Invalid candidate transform')
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1, atol=1e-6):
        raise ValueError('Candidate is not a rigid rotation')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-12):
        raise ValueError('Invalid homogeneous transform')
    a = np.asarray([s['p_camera_m'] for s in samples], float)
    b = np.asarray([s['p_torso_m'] for s in samples], float)
    if a.shape != (len(samples), 3) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid validation points')
    if np.linalg.matrix_rank(a - a.mean(0), tol=1e-8) < 2:
        raise ValueError('Validation points are collinear; sample a wider area')
    predictions = a @ R.T + T[:3, 3]
    differences = predictions - b
    errors = np.linalg.norm(differences, axis=1)
    return dict(status='FIXED_CANDIDATE_EVALUATED_NOT_AUTO_APPROVED', refitted=False,
                count=len(samples), rms_m=float(np.sqrt(np.mean(errors**2))),
                max_m=float(errors.max()), mean_residual_torso_m=differences.mean(0).tolist(),
                T_torso_camera=T.tolist(),
                model_tip_measured=session['probe'].get('tip_measured') is True,
                points=[dict(marker_id=s['marker_id'], original_role=s['role'],
                             predicted_torso_m=p.tolist(), fk_tip_torso_m=s['p_torso_m'],
                             residual_torso_m=d.tolist(), error_m=float(e), image=s['image'])
                        for s, p, d, e in zip(samples, predictions, differences, errors)],
                notes='Every new contact is a validation point, including those labelled fit by the sampler. '
                      'The candidate is unchanged. Same board/kinematics/model tip can share systematic errors; '
                      'confirm actual finger offset and validate at other board poses/distances. Not applied to VLN.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    args = parser.parse_args()
    candidate_path = args.candidate.resolve(); session_dir = args.session.resolve()
    if candidate_path.parent == session_dir:
        parser.error('Use a separately acquired session, not the original calibration directory')
    candidate = read(candidate_path); session = read(session_dir / 'session.json')
    original_path = candidate_path.parent / 'session.json'
    original = read(original_path) if original_path.exists() else None
    try:
        result = evaluate(candidate, session, original)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    result.update(candidate_path=str(candidate_path),
                  candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                  validation_session=str(session_dir))
    save(session_dir / 'validation_report.json', result)
    for point in result['points']:
        print(f'[VALIDATE] ID {point["marker_id"]}: error={point["error_m"]*1000:.1f} mm')
    print(f'[VALIDATE] {result["count"]} new contacts; RMS={result["rms_m"]*1000:.1f} mm; max={result["max_m"]*1000:.1f} mm')
    print('[VALIDATE] 外参未重新拟合。报告已保存；模型指尖偏移仍需实物核实，不自动应用VLN。')


if __name__ == '__main__':
    main()
