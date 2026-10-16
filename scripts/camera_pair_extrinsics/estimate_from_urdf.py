#!/usr/bin/env python3
"""Compose an explicitly unverified URDF anchor with the relative RGB candidate."""
import argparse
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from calibrate_pair import read, save
from geometry import transform


def rotation_rpy(r,p,y):
    Rx=np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
    Ry=np.array([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]])
    Rz=np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
    return Rz@Ry@Rx


def rpy_degrees(R):
    pitch=math.atan2(-R[2,0],math.hypot(R[0,0],R[1,0]))
    if abs(math.cos(pitch))<1e-6:
        raise ValueError('RPY singularity; use rotation matrix')
    return np.degrees([math.atan2(R[2,1],R[2,2]),pitch,math.atan2(R[1,0],R[0,0])]).tolist()


def estimate(candidate, urdf):
    joint=ET.parse(urdf).getroot().find("joint[@name='d435_joint']")
    if joint is None or joint.get('type')!='fixed':
        raise ValueError('Expected fixed d435_joint')
    if (joint.find('parent').get('link'),joint.find('child').get('link'))!=('torso_link','d435_link'):
        raise ValueError('Unexpected reference frames')
    if (candidate['parent_frame'],candidate['child_frame'])!=('old_rgb_optical','new_rgb_optical'):
        raise ValueError('Expected relative RGB optical transform')
    if candidate['old_camera_serial']!='344422072128':
        raise ValueError('URDF assumption here is for the originally inspected D435i')
    origin=joint.find('origin')
    xyz=np.array([float(v) for v in origin.get('xyz').split()])
    rpy=[float(v) for v in origin.get('rpy').split()]
    anchor=np.eye(4);anchor[:3,:3]=rotation_rpy(*rpy);anchor[:3,3]=xyz
    A=np.eye(4);A[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]
    T=transform(anchor@A@transform(candidate['T_old_rgb_new_rgb']))
    flu=transform(T@np.linalg.inv(A))
    return {'status':'URDF_ASSUMPTIONS_UNVERIFIED_ESTIMATE','camera_model':'D415',
            'camera_serial':candidate['new_camera_serial'],'parent_frame':'torso_link','child_frame':'new_rgb_optical',
            'T_torso_new_rgb':T.tolist(),'translation_m':T[:3,3].tolist(),
            'rgb_optical_rpy_deg':rpy_degrees(T[:3,:3]),'mounting_flu_rpy_deg':rpy_degrees(flu[:3,:3]),
            'T_torso_new_camera_flu':flu.tolist(),'T_d435_old_rgb_assumed':A.tolist(),
            'T_torso_d435_from_urdf':anchor.tolist(),'urdf':str(Path(urdf).resolve()),
            'urdf_sha256':hashlib.sha256(Path(urdf).read_bytes()).hexdigest(),
            'convention':'T_A_B maps B into A; RPY is Rz(yaw) @ Ry(pitch) @ Rx(roll)',
            'assumptions':['d435_link axes assumed X forward/Y left/Z up',
                           'd435_link origin assumed coincident with original RGB optical origin',
                           'Mounting remains as during the paired captures'],
            'reference_verified':False,'applied_to_vln':False,
            'note':'Dual-camera validation does not validate this assumed torso anchor.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--relative-candidate',type=Path,required=True)
    parser.add_argument('--urdf',type=Path,default=Path(__file__).resolve().parents[3]/'unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf')
    parser.add_argument('--assume-rgb-at-link-origin',action='store_true',required=True,
                        help='Explicitly accept an UNVERIFIED coincident-origin and FLU-axis assumption')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a new file; preserve previous estimates')
    result=estimate(read(args.relative_candidate),args.urdf)
    result['relative_candidate']=str(args.relative_candidate.resolve())
    save(args.output,result)
    print('[URDF ESTIMATE] xyz m:',result['translation_m'],'mounting RPY deg:',result['mounting_flu_rpy_deg'])
    print('[URDF ESTIMATE] reference remains UNVERIFIED; not applied to VLN')


if __name__=='__main__':
    main()
