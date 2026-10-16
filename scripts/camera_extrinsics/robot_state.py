"""Only subscribes rt/lowstate; never creates a publisher or motion client."""
import threading
import time
import numpy as np
from kinematics import JOINT_NAMES, MODE_FILES, RobotModel
from pathlib import Path

MODEL_DIR=Path(__file__).resolve().parents[3]/'unitree_ros/robots/g1_description'

class JointReader:
    def __init__(self,interface):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        ChannelFactoryInitialize(0,interface)
        self.lock=threading.Lock();self.samples=[]
        self.sub=ChannelSubscriber('rt/lowstate',LowState_)
        self.sub.Init(self.receive,1)
    def receive(self,m):
        try:
            q=[float(s.q) for s in m.motor_state[:29]]
            dq=[float(s.dq) for s in m.motor_state[:29]]
            if len(q)!=29 or not np.isfinite(q+dq).all():return
            s=dict(received_monotonic_s=time.monotonic(),tick=int(m.tick),mode_machine=int(m.mode_machine),
                   mode_pr=int(m.mode_pr),q=dict(zip(JOINT_NAMES,q)),dq=dict(zip(JOINT_NAMES,dq)))
            with self.lock:
                self.samples.append(s);self.samples=self.samples[-2000:]
        except (ValueError,TypeError,AttributeError):return
    def latest(self,timeout=8):
        end=time.monotonic()+timeout
        while True:
            with self.lock:s=self.samples[-1] if self.samples else None
            if s and time.monotonic()-s['received_monotonic_s']<.5:return s
            if time.monotonic()>=end:raise RuntimeError('No fresh rt/lowstate; ping alone does not prove DDS/joints available')
            time.sleep(.03)
    def stationary(self,seconds=1,side='left'):
        self.latest();start=time.monotonic();time.sleep(seconds);self.latest()
        with self.lock:ss=[s for s in self.samples if s['received_monotonic_s']>=start]
        if len(ss)<5:raise RuntimeError('Insufficient fresh joint samples')
        if any(b['received_monotonic_s']-a['received_monotonic_s']>.2 for a,b in zip(ss,ss[1:])):
            raise RuntimeError('Joint samples interrupted')
        if len({s['tick'] for s in ss})<5:raise RuntimeError('Robot tick is not updating')
        if len({(s['mode_machine'],s['mode_pr']) for s in ss})!=1:raise RuntimeError('Robot mode changed during sample')
        names=[n for n in JOINT_NAMES if n.startswith('waist_') or
               (n.startswith(side+'_') and any(k in n for k in ('shoulder','elbow','wrist')))]
        a=np.array([[s['q'][n] for n in names] for s in ss])
        span=float(np.max(np.ptp(a,axis=0)))
        if span>np.deg2rad(.5):raise RuntimeError(f'Arm/waist moved {np.rad2deg(span):.3f} deg (>0.5) during contact; repeat when still')
        # A single velocity spike can occur while encoder positions remain still.
        # Keep raw velocities for audit; observed position variation decides steadiness.
        max_dq=max(abs(s['dq'][n]) for s in ss for n in names)
        result=dict(ss[-1]);result['q']={n:float(np.median([s['q'][n] for s in ss])) for n in JOINT_NAMES}
        result['stationary_window_s']=seconds;result['sample_count']=len(ss)
        result['max_joint_span_deg']=float(np.rad2deg(span));result['max_reported_dq_rad_s']=float(max_dq)
        return result
    def close(self):self.sub.Close()

def model_for(state):
    if state['mode_pr']!=0:raise ValueError('mode_pr is not PR; motor angles cannot be used as URDF joint angles')
    mode=state['mode_machine']
    if mode not in MODE_FILES:raise ValueError(f'Unsupported mode_machine={mode}; confirm correct model and joint mapping first')
    return RobotModel(MODEL_DIR/(MODE_FILES[mode]+'.urdf'))
