#!/usr/bin/env python3
"""Export a self-contained, read-only URDF waist viewer; no robot connection."""
import argparse
import base64
import json
import struct
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from kinematics import origin, RobotModel, JOINT_NAMES

HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE.parents[2] / 'unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf'


def transform(element):
    if element is None:
        return np.eye(4)
    return origin(np.fromstring(element.get('xyz', '0 0 0'), sep=' '),
                  np.fromstring(element.get('rpy', '0 0 0'), sep=' '))


def column_matrix(matrix):
    return matrix.T.reshape(-1).tolist()


def rpy_degrees(matrix):
    r = np.asarray(matrix)[:3, :3]
    return np.rad2deg([np.arctan2(r[2, 1], r[2, 2]),
                      np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0])),
                      np.arctan2(r[1, 0], r[0, 0])]).tolist()


def export(urdf, output, extrinsics=None):
    root = ET.parse(urdf).getroot()
    meshes = []
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is None:
                continue
            path = urdf.parent / mesh.get('filename')
            raw = path.read_bytes()
            count = struct.unpack_from('<I', raw, 80)[0]
            if len(raw) != 84 + count * 50:
                raise ValueError(f'Expected binary STL: {path}')
            triangles = np.frombuffer(raw, dtype=np.dtype([
                ('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]),
                count=count, offset=84)
            vertices = np.empty((count, 3, 6), dtype='<f4')
            vertices[:, :, :3] = triangles['vertices']
            vertices[:, :, :3] *= np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            # Derive normals from the actual scaled triangles.
            normals = np.cross(vertices[:, 1, :3] - vertices[:, 0, :3],
                               vertices[:, 2, :3] - vertices[:, 0, :3])
            normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
            vertices[:, :, 3:] = normals[:, None, :]
            name = link.get('name')
            color = [.66, .7, .75]
            if name == 'pelvis': color = [.42, .48, .59]
            if name == 'torso_link': color = [.82, .85, .89]
            if name == 'waist_yaw_link': color = [.33, .55, .95]
            if name == 'waist_roll_link': color = [.94, .39, .36]
            meshes.append(dict(link=name, origin=column_matrix(transform(visual.find('origin'))),
                               color=color, data=base64.b64encode(vertices.tobytes()).decode()))
    joints = []
    for joint in root.findall('joint'):
        axis = joint.find('axis')
        joints.append(dict(name=joint.get('name'), parent=joint.find('parent').get('link'),
                           child=joint.find('child').get('link'),
                           origin=column_matrix(transform(joint.find('origin'))),
                           axis=np.fromstring(axis.get('xyz') if axis is not None else '1 0 0', sep=' ').tolist(),
                           type=joint.get('type')))
    model = RobotModel(urdf)
    checks = []
    for angles in ([0, 0, 0], [25, 0, 0], [0, 12, 0], [0, 0, -15], [25, 12, -15]):
        q = dict.fromkeys(JOINT_NAMES, 0.)
        for name, angle in zip(('yaw', 'roll', 'pitch'), angles):
            q[f'waist_{name}_joint'] = np.deg2rad(angle)
        checks.append(dict(angles=angles, torso=column_matrix(model.root_pose('torso_link', q))))
    data = dict(urdf=str(urdf), meshes=meshes, joints=joints, fk_checks=checks)
    if extrinsics is not None:
        candidate = json.loads(extrinsics.read_text())
        if candidate['target_frame'] != 'torso_link' or candidate['source_frame'] != 'color_optical':
            raise ValueError('Expected color_optical to torso_link extrinsics')
        camera = np.array(candidate['T_torso_camera'], dtype=float)
        if camera.shape != (4, 4) or not np.isfinite(camera).all():
            raise ValueError('Invalid camera matrix')
        # Forward/left/up camera axes, expressed in the original optical frame.
        optical_from_flu = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        flu = camera.copy(); flu[:3, :3] = camera[:3, :3] @ optical_from_flu
        data['camera'] = dict(matrix=column_matrix(camera), translation_m=camera[:3, 3].tolist(),
                              optical_rpy_deg=rpy_degrees(camera), mounting_rpy_deg=rpy_degrees(flu),
                              source=str(extrinsics), status=candidate['status'])
    template = (HERE / 'waist_viewer.html').read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('__ROBOT_DATA__', json.dumps(data, separators=(',', ':'))))
    print(f'Exported {len(meshes)} actual URDF meshes: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extrinsics', type=Path, help='Display a torso/color-optical extrinsics candidate')
    args = parser.parse_args()
    export(args.urdf.resolve(), args.output.resolve(), args.extrinsics.resolve() if args.extrinsics else None)
