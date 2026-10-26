"""独立按键阻尼请求及本机导航锁存；不包含站立、行走或恢复命令。"""
import json
from pathlib import Path
import re
import time
import fcntl
import uuid
from contextlib import contextmanager


def latch_path(interface):
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+', interface):
        raise ValueError('无效网络接口名')
    return Path.home()/'.local/state/g1_safety'/f'damping_{interface}.json'


@contextmanager
def _file_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix(path.suffix+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def snapshot_latch(path):
    try:return Path(path).read_bytes()
    except FileNotFoundError:return None


def write_latch(path, *, interface):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 每次请求生成新版本；启动自动清锁不能误删刚按下的阻尼请求。
    with _file_lock(path):
        # 先创建禁止标记；即使后续写JSON失败，也留下锁存。
        path.touch(exist_ok=True)
        with path.open('w',encoding='utf-8') as f:
            json.dump({'reason':'keyboard_damping','interface':interface,
                       'request_id':uuid.uuid4().hex,'requested_epoch_s':time.time()},f)
            f.write('\n')


def auto_clear_previous_latch(client,path,expected_snapshot):
    """仅新任务启动调用。只读确认802后清旧锁，绝不切换机器人模式。"""
    path=Path(path)
    current=snapshot_latch(path)
    if current is None:return {'status':'NO_LATCH','robot_mode_changed':False}
    if expected_snapshot is None or current!=expected_snapshot:
        raise RuntimeError('检测到本次启动期间的新阻尼请求，不自动清除')
    # RPC不持文件锁，避免阻塞独立按键建立锁存。
    code,fsm=client.GetFsmId()
    if code!=0 or fsm!=802:
        raise RuntimeError(f'保留阻尼锁存：FSM查询=({code}, {fsm})；请先手动恢复已验证的运控模式802')
    with _file_lock(path):
        if snapshot_latch(path)!=expected_snapshot:
            raise RuntimeError('查询状态期间发生新的阻尼请求，取消自动清锁')
        path.unlink()
    result={'status':'AUTO_CLEARED','readback_code':code,'fsm_id':fsm,
            'robot_mode_changed':False,'cleared_epoch_s':time.time()}
    print('[DAMPING KEY] 已确认FSM=802，自动清除上次阻尼锁存；未切换机器人模式。',flush=True)
    return result


def request_damping(client, path, *, interface, attempts=3, sleep=time.sleep, clock=time.monotonic):
    report={'requested_fsm_id':1,'latch_file':str(path),'automatic_recovery':False,
            'confirmed':False,'attempts':[],'requested_monotonic_s':clock()}
    try:
        write_latch(path, interface=interface)
        report['local_latch_written']=True
    except OSError as error:
        report['local_latch_written']=False
        report['local_latch_error']=str(error)
    for _ in range(attempts):
        attempt={'requested_monotonic_s':clock()}
        report['attempts'].append(attempt)
        try:
            # Damp()丢弃返回码；使用同一官方调用并保留返回码。
            attempt['request_code']=client.SetFsmId(1)
            for _ in range(4):
                code, fsm=client.GetFsmId()
                attempt.setdefault('observations',[]).append({'code':code,'fsm_id':fsm,'monotonic_s':clock()})
                if code==0 and fsm==1:
                    report['confirmed']=True
                    report['finished_monotonic_s']=clock()
                    return report
                sleep(.1)
        except Exception as error:
            attempt['error']=str(error)
        sleep(.1)
    report['finished_monotonic_s']=clock()
    return report
