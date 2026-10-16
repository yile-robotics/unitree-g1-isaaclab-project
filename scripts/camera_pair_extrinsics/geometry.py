"""T_A_B maps points expressed in B into A. Translations are metres."""
import cv2
import numpy as np


def transform(value):
    T = np.asarray(value, float)
    if T.shape != (4,4) or not np.isfinite(T).all() or not np.allclose(T[3], [0,0,0,1]):
        raise ValueError('Invalid homogeneous transform')
    R = T[:3,:3]
    if not np.allclose(R.T@R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
        raise ValueError('Rotation must be in SO(3)')
    return T


def angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))


def pack(T):
    return np.r_[cv2.Rodrigues(T[:3,:3])[0].ravel(), T[:3,3]]


def unpack(p):
    T = np.eye(4); T[:3,:3] = cv2.Rodrigues(p[:3])[0]; T[:3,3] = p[3:6]
    return T


def project(T, obj, K):
    xyz = np.asarray(obj)@T[:3,:3].T+T[:3,3]
    if np.any(xyz[:,2] <= .01):
        raise ValueError('Optimized board behind camera')
    return cv2.projectPoints(np.asarray(obj, float), pack(T)[:3], T[:3,3], np.asarray(K,float), None)[0].reshape(-1,2)


def relative(pair):
    return transform(pair['old']['T_camera_board'])@np.linalg.inv(transform(pair['new']['T_camera_board']))


def average(Ts):
    U, _, V = np.linalg.svd(sum(T[:3,:3] for T in Ts))
    D = np.diag([1,1,np.linalg.det(U@V)])
    T = np.eye(4); T[:3,:3] = U@D@V; T[:3,3] = np.median([a[:3,3] for a in Ts],axis=0)
    return T


def fit(pairs, require_diversity=True):
    """Joint fixed-intrinsic reprojection fit, including non-overlapping ID sets.

    Unknowns: one new->old transform and one board->new pose per view.
    Centres use the measured board pitch; no coded-border size assumption.
    Huber pixel residuals reduce the influence of marginal detections.
    """
    if len(pairs) < 8:
        raise ValueError('Need at least 8 fit pairs; validation pairs do not enter fitting')
    poses = [transform(p['new']['T_camera_board']) for p in pairs]
    normal_span = max(float(np.degrees(np.arccos(np.clip(a[:3,2]@b[:3,2],-1,1)))) for a in poses for b in poses)
    if require_diversity and normal_span < 10:
        raise ValueError('Tilt the board: fit views need at least 10 degrees of board-normal diversity')
    Ts = [relative(p) for p in pairs]
    x = np.concatenate([pack(average(Ts))]+[pack(T) for T in poses])

    def residual(v):
        rel = unpack(v[:6]); out = []
        for i, pair in enumerate(pairs):
            new = unpack(v[6+i*6:12+i*6])
            for side, T in [('new',new),('old',rel@new)]:
                obs = pair[side]
                out.extend((project(T,obs['object_points'],obs['K'])-np.asarray(obs['pixels'])).ravel())
        return np.asarray(out)

    def cost(r):
        a = np.abs(r)
        return float(np.sum(np.where(a<=2, .5*r*r, 2*(a-1))))

    r = residual(x); damping = 1e-3
    for _ in range(35):
        J = np.empty((len(r),len(x)))
        for j in range(len(x)):
            xp = x.copy(); xp[j] += 1e-6
            J[:,j] = (residual(xp)-r)/1e-6
        w = np.minimum(1,2/np.maximum(np.abs(r),1e-12))
        H = J.T@(w[:,None]*J); g = J.T@(w*r)
        step = np.linalg.solve(H+damping*np.diag(np.maximum(np.diag(H),1)), -g)
        try:
            trial = residual(x+step)
        except ValueError:
            damping *= 10; continue
        if cost(trial) < cost(r):
            x += step; r = trial; damping = max(damping/3,1e-9)
            if np.linalg.norm(step) < 1e-7:
                break
        else:
            damping *= 10
    T = transform(unpack(x[:6]))
    summary = {'fit_views':len(pairs), 'board_normal_span_deg':normal_span,
               'joint_reprojection_rms_px':float(np.sqrt(np.mean(r*r))),
               'per_pair_initial_translation_difference_mm':[float(np.linalg.norm(t[:3,3]-T[:3,3])*1000) for t in Ts],
               'per_pair_initial_rotation_difference_deg':[angle(t[:3,:3]@T[:3,:3].T) for t in Ts]}
    return T, summary


def validate(T, pairs):
    T = transform(T); records = []
    for p in pairs:
        old, new = transform(p['old']['T_camera_board']), transform(p['new']['T_camera_board'])
        predicted = T@new
        center = np.mean(np.asarray(p['old']['object_points']),axis=0)
        delta = predicted[:3,:3]@center+predicted[:3,3]-(old[:3,:3]@center+old[:3,3])
        errors = []
        for side, pose in [('old',predicted),('new',np.linalg.inv(T)@old)]:
            obs = p[side]
            errors.extend(np.linalg.norm(project(pose,obs['object_points'],obs['K'])-np.asarray(obs['pixels']),axis=1))
        records.append({'pair':p.get('name'), 'board_center_error_mm':float(np.linalg.norm(delta)*1000),
                        'rotation_error_deg':angle(predicted[:3,:3]@old[:3,:3].T),
                        'cross_projection_rms_px':float(np.sqrt(np.mean(np.asarray(errors)**2)))})
    if not records:
        return {'status':'NO_HELD_OUT_VALIDATION','views':[]}
    return {'status':'HELD_OUT_REPORT', 'views':records,
            'center_rms_mm':float(np.sqrt(np.mean([r['board_center_error_mm']**2 for r in records]))),
            'max_center_error_mm':max(r['board_center_error_mm'] for r in records),
            'max_rotation_error_deg':max(r['rotation_error_deg'] for r in records),
            'max_cross_projection_rms_px':max(r['cross_projection_rms_px'] for r in records)}


def compose(reference, T, serial):
    if reference.get('verified') is not True or not reference.get('source'):
        raise ValueError('A verified old RGB optical -> torso reference with source is required')
    if (reference.get('parent_frame'),reference.get('child_frame')) != ('torso_link','old_rgb_optical'):
        raise ValueError('Reference frames must be torso_link and old_rgb_optical; d435_link alone is insufficient')
    if reference.get('old_camera_serial') != serial:
        raise ValueError('Reference serial differs from original camera')
    return transform(reference['T_parent_child'])@transform(T)
