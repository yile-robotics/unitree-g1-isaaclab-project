"""Metric board pose from decoded centres; outer brown squares define contacts."""
import cv2
import numpy as np
from verify_marker_layout import detect, check_grid, marker_center


def positions(config):
    grid = config['marker_ids_by_position']
    return {int(i):(r,c) for r,row in enumerate(grid) for c,i in enumerate(row)}


def board_point(config, marker_id, corner='top_left'):
    r,c = positions(config)[marker_id]
    side = float(config['marker_outer_side_m'])
    px = side+float(config['marker_gap_x_m'])
    py = side+float(config['marker_gap_y_m'])
    offsets = {'top_left':(0,0), 'center':(.5,.5)}
    dx,dy = offsets[corner]
    return np.array([c*px+dx*side, r*py+dy*side, 0.])


def estimate(markers, K, dist, config):
    accepted, _, error = check_grid(markers, config['marker_ids_by_position'],allow_outliers=True)
    if error:
        raise ValueError(error)
    ids = sorted(accepted)
    obj = np.array([board_point(config,i,'center') for i in ids],np.float64)
    pixels = np.array([marker_center(accepted[i]) for i in ids],np.float64)
    # Centres are invariant to marker rotation. Coded-region borders are NOT
    # assumed to be the measured 55mm outer brown-square borders.
    ok,rv,tv = cv2.solvePnP(obj,pixels,np.asarray(K,float),np.asarray(dist,float),
                           flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('Board PnP failed')
    R = cv2.Rodrigues(rv)[0]
    if np.any((obj@R.T+tv.reshape(3))[:,2] <= 0):
        raise ValueError('Board behind camera')
    projected = cv2.projectPoints(obj,rv,tv,K,dist)[0].reshape(-1,2)
    errors = np.linalg.norm(projected-pixels,axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    if not np.isfinite(rms) or rms > 1.5 or errors.max() > 3:
        raise ValueError(f'Centre reprojection error too large: RMS={rms:.2f}px')
    T = np.eye(4);T[:3,:3]=R;T[:3,3]=tv.reshape(3)
    return dict(T_camera_board=T.tolist(),reprojection_rms_px=rms,
                decoded_ids=ids,marker_centres_px=pixels.tolist(),
                rejected_marker_ids=sorted(set(markers)-set(accepted)),
                corners_px={str(i):accepted[i].tolist() for i in ids})


def camera_point(pose, point):
    T = np.array(pose['T_camera_board'])
    return T[:3,:3]@np.asarray(point)+T[:3,3]


def stability(records, target, max_span_m=.005):
    if len(records) < 5:
        raise ValueError('Need at least 5 current board poses')
    points = np.array([camera_point(x['pose'],target) for x in records])
    span = float(np.max(np.linalg.norm(points-points[:,None,:],axis=2)))
    Rs = [np.array(x['pose']['T_camera_board'])[:3,:3] for x in records]
    rotation = max(float(np.rad2deg(np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1))))
                   for a in Rs for b in Rs)
    if span > max_span_m:
        raise ValueError(f'Board/camera not still: target span={span*1000:.1f}mm, rotation={rotation:.2f}deg')
    # Calibration consumes this target's 3D position, not the fitted plane
    # orientation. Keep the orientation diagnostic without rejecting stable
    # target coordinates merely because the plane normal estimate fluctuates.
    return dict(target_span_m=span,rotation_span_deg=rotation,
                rotation_diagnostic_warning=rotation>1.)


def detect_pose(rgb, K, metadata, config):
    markers = detect(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),config.get('marker_border_bits',2))
    return estimate(markers,K,np.array(metadata['color_intrinsics']['coeffs']),config)
