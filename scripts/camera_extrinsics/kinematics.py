"""URDF serial-chain FK. T_target_source maps source coordinates into target."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

JOINT_NAMES = [f'{side}_{name}_joint' for side in ('left','right') for name in
               ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
JOINT_NAMES += ['waist_yaw_joint','waist_roll_joint','waist_pitch_joint']
JOINT_NAMES += [f'{side}_{name}_joint' for side in ('left','right') for name in
                ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]
MODE_FILES = {2:'g1_29dof',5:'g1_29dof_rev_1_0',**{i:f'g1_29dof_mode_{i}' for i in (11,13,15)}}

def rotation(axis, angle):
    axis = np.asarray(axis,dtype=float)
    norm=np.linalg.norm(axis)
    if norm < 1e-12: raise ValueError('zero joint axis')
    x,y,z=axis/norm
    W=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*W+(1-np.cos(angle))*(W@W)

def origin(xyz, rpy):
    T=np.eye(4)
    T[:3,3]=xyz
    T[:3,:3]=rotation([0,0,1],rpy[2])@rotation([0,1,0],rpy[1])@rotation([1,0,0],rpy[0])
    return T

class RobotModel:
    def __init__(self,path):
        self.path=Path(path).resolve()
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        root=ET.parse(self.path).getroot()
        self.links={e.attrib['name'] for e in root.findall('link')}
        self.parents={}
        for j in root.findall('joint'):
            child=j.find('child').attrib['link']
            if child in self.parents: raise ValueError('multiple parents')
            self.parents[child]=j
    def root_pose(self,link,q):
        if link not in self.links: raise ValueError(f'Unknown link: {link}')
        chain=[]; seen=set()
        while link in self.parents:
            if link in seen: raise ValueError('URDF cycle')
            seen.add(link)
            j=self.parents[link];chain.append(j);link=j.find('parent').attrib['link']
        T=np.eye(4)
        for j in reversed(chain):
            o=j.find('origin')
            xyz=np.fromstring(o.get('xyz','0 0 0') if o is not None else '0 0 0',sep=' ')
            rpy=np.fromstring(o.get('rpy','0 0 0') if o is not None else '0 0 0',sep=' ')
            T=T@origin(xyz,rpy)
            typ=j.get('type')
            if typ!='fixed':
                if typ not in ('revolute','continuous'): raise ValueError(f'Unsupported joint: {typ}')
                if j.find('mimic') is not None: raise ValueError('Mimic chain unsupported')
                name=j.get('name')
                if name not in q: raise ValueError(f'Missing measured joint: {name}')
                value=float(q[name])
                if not np.isfinite(value): raise ValueError('Nonfinite joint')
                axis=j.find('axis');v=np.fromstring(axis.get('xyz') if axis is not None else '1 0 0',sep=' ')
                motion=np.eye(4);motion[:3,:3]=rotation(v,value);T=T@motion
        return T
    def transform(self,target,source,q):
        return np.linalg.inv(self.root_pose(target,q))@self.root_pose(source,q)
    def point(self,target,source,offset,q):
        return (self.transform(target,source,q)@np.r_[offset,1])[:3]

def fit_rigid(source,target):
    a,b=np.asarray(source,float),np.asarray(target,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or len(a)<3: raise ValueError('Need >=3 matched 3D points')
    if not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError('Nonfinite points')
    ac,bc=a-a.mean(0),b-b.mean(0)
    if min(np.linalg.matrix_rank(ac,tol=1e-8),np.linalg.matrix_rank(bc,tol=1e-8))<2:
        raise ValueError('Points collinear or repeated')
    U,s,Vt=np.linalg.svd(ac.T@bc)
    D=np.eye(3);D[2,2]=np.linalg.det(Vt.T@U.T)
    R=Vt.T@D@U.T
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=b.mean(0)-R@a.mean(0)
    return T
