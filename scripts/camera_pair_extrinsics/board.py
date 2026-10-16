"""Independent coded-board detector. No hand contacts or robot kinematics."""
import cv2
import numpy as np


def points(config, ids):
    layout = {int(i): (r, c) for r, row in enumerate(config['id_grid']) for c, i in enumerate(row)}
    side = config['outer_square_side_m']
    return np.asarray([[layout[i][1]*(side+config['gap_x_m'])+side/2,
                        layout[i][0]*(side+config['gap_y_m'])+side/2, 0] for i in ids], float)


def centre(corners):
    c = np.asarray(corners, float)
    uv = np.linalg.solve(np.column_stack((c[2]-c[0], c[1]-c[3])), c[1]-c[0])
    return c[0]+uv[0]*(c[2]-c[0])


def detect(rgb, config):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    observations = {}
    for scale in (1, 2, 3):
        params = cv2.aruco.DetectorParameters()
        params.markerBorderBits = config['marker_border_bits']
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        image = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(image)
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                original = (c.reshape(4, 2)+.5)/scale-.5
                observations.setdefault(int(i), []).append(original)
    result = {}
    for i, corners in observations.items():
        centers = np.asarray([centre(c) for c in corners])
        if np.max(np.linalg.norm(centers-centers[0], axis=1)) <= 3:
            result[i] = corners[0]
    return result


def estimate(markers, K, config):
    known = {int(i) for row in config['id_grid'] for i in row}
    ids = sorted(set(markers) & known)
    if len(ids) < 6:
        raise ValueError('Need 6 decoded IDs in each camera, spanning rows and columns')
    obj = points(config, ids)
    pixels = np.asarray([centre(markers[i]) for i in ids])
    if np.linalg.matrix_rank(obj[:, :2]-obj[:, :2].mean(axis=0)) < 2:
        raise ValueError('IDs are collinear')
    H, mask = cv2.findHomography(obj[:, :2], pixels, cv2.RANSAC, 1.5)
    if H is None:
        raise ValueError('ID layout has no consistent homography')
    errors = np.linalg.norm(cv2.perspectiveTransform(obj[:, :2].reshape(-1,1,2), H).reshape(-1,2)-pixels, axis=1)
    good = mask.ravel().astype(bool) & (errors <= 2)
    obj, pixels = obj[good], pixels[good]
    accepted = [i for i, keep in zip(ids, good) if keep]
    if len(accepted) < 6 or np.linalg.matrix_rank(obj[:, :2]-obj[:, :2].mean(axis=0)) < 2:
        raise ValueError('Too few consistent IDs after layout check')
    ok, rv, tv = cv2.solvePnP(obj, pixels, np.asarray(K, float), None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('PnP failed')
    R = cv2.Rodrigues(rv)[0]
    if np.any((obj@R.T+tv.reshape(3))[:, 2] <= 0):
        raise ValueError('Board behind camera')
    projected = cv2.projectPoints(obj, rv, tv, K, None)[0].reshape(-1,2)
    error = np.linalg.norm(projected-pixels, axis=1)
    rms = float(np.sqrt(np.mean(error**2)))
    if rms > 1.5 or error.max() > 3:
        raise ValueError(f'Board reprojection RMS={rms:.2f}px')
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tv.ravel()
    return {'T_camera_board': T.tolist(), 'ids': accepted, 'pixels': pixels.tolist(),
            'object_points': obj.tolist(), 'K': np.asarray(K).tolist(),
            'reprojection_rms_px': rms, 'rejected_ids': sorted(set(markers)-set(accepted))}
