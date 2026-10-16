#!/usr/bin/env python3
"""Read-only live verification of the candidate AprilTag ID layout."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from calibrate import DEFAULT_CAMERA, save, save_image, read

HERE = Path(__file__).resolve().parent


def marker_center(corners):
    """Projective centre: intersection of opposite-corner diagonals."""
    c = np.asarray(corners, float)
    a, b = c[2]-c[0], c[3]-c[1]
    uv = np.linalg.solve(np.column_stack((a, -b)), c[1]-c[0])
    return c[0] + uv[0]*a


def detect(gray, border_bits=2):
    """Decode without supplying expected IDs; enlarging helps small markers."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    observations = {}
    for scale in (1, 2, 3):
        image = gray if scale == 1 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        params = cv2.aruco.DetectorParameters()
        params.markerBorderBits = border_bits
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(image)
        if ids is not None:
            for corner, marker_id in zip(corners, ids.flatten()):
                # Cubic resize maps pixel centres, including a half-pixel shift.
                original = (corner.reshape(4, 2) + .5) / scale - .5
                observations.setdefault(int(marker_id), []).append(original)
    # The same ID at different image locations is ambiguous; do not count it.
    result = {}
    for marker_id, corners in observations.items():
        centres = np.array([marker_center(c) for c in corners])
        if np.max(np.linalg.norm(centres - centres[0], axis=1)) <= 3:
            result[marker_id] = corners[0]
    return result


def check_grid(markers, grid, allow_outliers=False):
    """Check the proposed positions against a common planar homography."""
    positions = {int(i): (c, r) for r, row in enumerate(grid) for c, i in enumerate(row)}
    unknown = set(markers) - set(positions)
    if unknown:
        if not allow_outliers:
            return {}, None, f'Unknown IDs: {sorted(unknown)}'
        markers={i:c for i,c in markers.items() if i in positions}
    if len(markers) < 6:
        return {}, None, 'Need at least 6 decoded markers spread across rows and columns'
    ids = list(markers)
    source = np.array([positions[i] for i in ids], np.float32)
    image = np.array([marker_center(markers[i]) for i in ids], np.float32)
    if np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        return {}, None, 'Markers lie in a single row/column'
    H, mask = cv2.findHomography(source, image, cv2.RANSAC, 1.5)
    if H is None:
        return {}, None, 'Candidate layout could not be projected'
    projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - image, axis=1)
    if allow_outliers:
        good=np.isfinite(errors)&(errors<=2)&mask.reshape(-1).astype(bool)
        selected=source[good]
        if good.sum()<6 or np.linalg.matrix_rank(selected-selected.mean(axis=0))<2:
            return {}, H, 'Too few consistent marker positions after rejecting outliers'
        if len(np.unique(selected[:,0]))<2 or len(np.unique(selected[:,1]))<2:
            return {}, H, 'Consistent markers do not span rows and columns'
        return {i:markers[i] for i,keep in zip(ids,good) if keep}, H, None
    if not np.isfinite(errors).all() or errors.max() > 2:
        return {}, H, 'Decoded IDs conflict with candidate grid; frame not counted'
    return markers, H, None


class Evidence:
    def __init__(self, grid):
        self.counts = {int(i): 0 for row in grid for i in row}
        self.frames = set()

    def add(self, frame_number, markers):
        if frame_number in self.frames:
            return []
        self.frames.add(frame_number)
        changed = []
        for marker_id in markers:
            if self.counts[marker_id] < 2:
                self.counts[marker_id] += 1
                changed.append(marker_id)
        return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera-config', type=Path, default=DEFAULT_CAMERA)
    parser.add_argument('--candidate-layout', type=Path, required=True)
    parser.add_argument('--measurements', type=Path, default=HERE/'board_marker_measurements.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('seconds must be finite and positive')
    if args.output.exists():
        parser.error('Use a new output directory')
    candidate = read(args.candidate_layout)
    grid = candidate['candidate_id_grid']
    if len(grid) != 6 or any(len(row) != 6 for row in grid) or len({i for row in grid for i in row}) != 36:
        parser.error('This verifier expects the proposed 6x6 unique-ID board')
    measurements = read(args.measurements)
    args.output.mkdir(parents=True)
    evidence = Evidence(grid)
    record = dict(status='PARTIAL_ID_LAYOUT', dictionary='APRILTAG_36h11',
                  candidate_id_grid=grid, measurements=measurements,
                  candidate_layout_source=str(args.candidate_layout.resolve()),
                  required_distinct_rgb_frames_per_id=2, observations=[],
                  notes='Image-space ID layout only. Not a camera-body extrinsic; '
                  'marker corner orientation and metric board pose still require validation.')

    def persist():
        confirmed = sorted(i for i, n in evidence.counts.items() if n >= 2)
        missing = sorted(i for i, n in evidence.counts.items() if n < 2)
        record.update(confirmed_ids=confirmed, pending_ids=missing,
                      observation_counts=evidence.counts,
                      status='ID_LAYOUT_CONFIRMED' if not missing else 'PARTIAL_ID_LAYOUT')
        save(args.output/'layout_progress.json', record)
        if not missing:
            save(args.output/'board_layout_confirmed.json', record)

    from camera_d415.network_camera import create_camera
    camera = None
    window = 'D415 ID layout: green=confirmed | yellow=pending | Q=exit'
    deadline = time.monotonic() + args.seconds
    old_summary = None
    print('[LAYOUT] 只读取图，不发送机器人命令。保持板完整或让待确认区域清晰入镜。', flush=True)
    print('[LAYOUT] 无需按Enter；每个ID需要在两张不同RGB帧中解码并通过布局检查。', flush=True)
    try:
        camera = create_camera(args.camera_config)
        while time.monotonic() < deadline:
            frame = camera.capture_forward(0, time.monotonic())
            meta = dict(camera.last_metadata)
            decoded = detect(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY),
                             measurements.get('marker_border_bits',2))
            accepted, H, error = check_grid(decoded, grid)
            changed = evidence.add(meta['color_frame_number'], accepted)
            if changed:
                filename = f'rgb_{meta["color_frame_number"]}.png'
                save_image(args.output/filename, frame.rgb)
                record['observations'].append(dict(image=filename, metadata=meta,
                    accepted_markers=[dict(id=i, corners_px=c.tolist()) for i, c in accepted.items()]))
            persist()
            summary = (len(record['confirmed_ids']), tuple(record['pending_ids']), error)
            if summary != old_summary:
                print(f'[LAYOUT] 确认 {summary[0]}/36；待确认 {record["pending_ids"]}' +
                      (f'；当前帧未计入：{error}' if error else ''), flush=True)
                if not record['pending_ids']:
                    print('[LAYOUT] 全部编号通过核对并保存。按图像窗口 Q 退出；尚未开始触碰标定。', flush=True)
                old_summary = summary
            if not args.headless:
                show = cv2.resize(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), None, fx=2, fy=2)
                for marker_id, corners in decoded.items():
                    if marker_id not in evidence.counts:
                        continue
                    color = (0, 255, 0) if evidence.counts[marker_id] >= 2 else (0, 220, 255)
                    cv2.polylines(show, [np.rint(corners*2).astype(np.int32)], True, color, 2)
                    x, y = np.rint(corners.mean(axis=0)*2).astype(int)
                    cv2.putText(show, str(marker_id), (x-10, y), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
                cv2.rectangle(show, (0, 0), (show.shape[1], 62), (20, 20, 20), -1)
                cv2.putText(show, f'Confirmed {len(record["confirmed_ids"])}/36 | Pending: ' +
                    ','.join(map(str, record['pending_ids'])), (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
                cv2.putText(show, error or 'Move board to show pending tags clearly. Q exits.',
                    (10, 47), cv2.FONT_HERSHEY_SIMPLEX, .48, (0,220,255), 1)
                cv2.imshow(window, show)
                key = cv2.waitKey(1) & 255
                if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        persist()
        if camera is not None:
            camera.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f'[LAYOUT] 保存到 {args.output}；状态 {record["status"]}', flush=True)
    return 0 if record['status'] == 'ID_LAYOUT_CONFIRMED' else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n[CANCEL] 已退出，进度保留，没有发送机器人命令。')
        raise SystemExit(130)
