#!/usr/bin/env python3
"""Extract and visualize a MODEL contact-point candidate; does not confirm physical fingertip."""
import argparse
import hashlib
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from calibrate import save
from kinematics import RobotModel,origin
from robot_state import MODEL_DIR


def vertices(path):
    raw=Path(path).read_bytes()
    if len(raw)<84:raise ValueError('STL too short')
    count=struct.unpack_from('<I',raw,80)[0]
    if len(raw)!=84+count*50:raise ValueError('Expected binary STL')
    dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')])
    v=np.frombuffer(raw,dtype,count=count,offset=84)['v'].reshape(-1,3)
    if not np.isfinite(v).all():raise ValueError('Nonfinite mesh')
    return np.unique(v.astype(float),axis=0)


def draw_views(v,p,path):
    import cv2
    canvas=np.full((650,1650,3),250,np.uint8)
    for panel,(a,b,title) in enumerate([(0,1,'XY view'),(0,2,'XZ view'),(1,2,'YZ view')]):
        lo=v[:,[a,b]].min(0);hi=v[:,[a,b]].max(0);scale=440/max(hi-lo)
        mid=(lo+hi)/2
        def pixels(points):
            xy=(points[:,[a,b]]-mid)*scale
            xy[:,1]*=-1
            return np.rint(xy+[panel*550+275,300]).astype(int)
        for x,y in pixels(v):cv2.circle(canvas,(int(x),int(y)),1,(125,125,125),-1)
        x,y=pixels(p[None])[0];cv2.circle(canvas,(int(x),int(y)),9,(0,0,255),2)
        cv2.drawMarker(canvas,(int(x),int(y)),(0,0,255),cv2.MARKER_CROSS,25,2)
        cv2.putText(canvas,title,(panel*550+20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
        cv2.putText(canvas,f'Horizontal +{"XYZ"[a]} right; vertical +{"XYZ"[b]} up',
                    (panel*550+20,570),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
    cv2.putText(canvas,'Red: maximum-X mesh vertex. MODEL CANDIDATE ONLY; confirm actual finger/contact surface.',
                (20,625),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,180),1)
    if not cv2.imwrite(str(path),canvas):raise IOError('Cannot save model views')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf',type=Path,default=MODEL_DIR/'g1_29dof_rev_1_0.urdf')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists; choose new directory')
    model=RobotModel(args.urdf)
    root=ET.parse(args.urdf).getroot()
    visual=root.find("link[@name='left_rubber_hand']/visual")
    if visual is None:raise ValueError('Model does not contain fixed rubber left hand')
    mesh=visual.find('geometry/mesh');mesh_path=args.urdf.parent/mesh.get('filename')
    v=vertices(mesh_path);scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
    o=visual.find('origin')
    xyz=np.fromstring(o.get('xyz','0 0 0') if o is not None else '0 0 0',sep=' ')
    rpy=np.fromstring(o.get('rpy','0 0 0') if o is not None else '0 0 0',sep=' ')
    T=origin(xyz,rpy);v=(v*scale)@T[:3,:3].T+T[:3,3]
    p=v[np.argmax(v[:,0])]
    joint=root.find("joint[@name='left_hand_palm_joint']")
    if joint is None or joint.get('type')!='fixed' or joint.find('parent').get('link')!='left_wrist_yaw_link' or joint.find('child').get('link')!='left_rubber_hand':
        raise ValueError('Expected direct fixed wrist-to-hand connection')
    jo=joint.find('origin')
    jxyz=np.fromstring(jo.get('xyz','0 0 0') if jo is not None else '0 0 0',sep=' ')
    jrpy=np.fromstring(jo.get('rpy','0 0 0') if jo is not None else '0 0 0',sep=' ')
    T_wrist_hand=origin(jxyz,jrpy)
    wrist=(T_wrist_hand@np.r_[p,1])[:3]
    args.output.mkdir(parents=True)
    record={'status':'MODEL_CANDIDATE_NOT_PHYSICALLY_CONFIRMED','selection':'maximum X surface vertex, not an anatomical finger label',
            'urdf':str(args.urdf.resolve()),'urdf_sha256':model.sha256,
            'mesh':str(mesh_path.resolve()),'mesh_sha256':hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            'p_left_rubber_hand_m':p.tolist(),'p_left_wrist_yaw_link_m':wrist.tolist(),
            'T_wrist_hand':T_wrist_hand.tolist(),
            'mesh_bounds_m':[v.min(0).tolist(),v.max(0).tolist()]}
    save(args.output/'model_contact_candidate.json',record)
    save(args.output/'probe.mesh_candidate.json',{
        'probe_link':'left_wrist_yaw_link','tip_in_link_m':wrist.tolist(),'tip_measured':False,
        'mount_on_torso_confirmed':False,'notes':'MODEL CANDIDATE ONLY. Confirm real hand shape, exact contact point, dimensions and covering thickness before use.'})
    draw_views(v,p,args.output/'hand_model_views.png')
    print('Model maximum-X point in left wrist frame (m):',wrist)
    print('Candidate only; NOT a verified real fingertip and NOT camera extrinsics. Saved:',args.output)

if __name__=='__main__':main()
