"""运行终端空格准备、Enter确认阻尼；独立进程避免模型/规划HTTP阻塞。"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty

from .damping_safety import latch_path, request_damping, snapshot_latch, auto_clear_previous_latch


class TerminalDampingMonitor:
    """父进程保存终端属性；子进程读取按键并独立调用SDK。"""
    def __init__(self, interface, output_dir):
        self.interface=interface;self.output_dir=Path(output_dir)
        self.process=None;self.fd=None;self.original=None

    def start(self):
        if not sys.stdin.isatty():
            print('[DAMPING KEY] 无交互终端，空格监听未启用；不要将其视为可用保护。',flush=True)
            return False
        self.fd=sys.stdin.fileno();self.original=termios.tcgetattr(self.fd)
        read_fd,write_fd=os.pipe()
        try:
            root=Path(__file__).resolve().parents[1]
            self.process=subprocess.Popen([sys.executable,'-m','unified_vln.terminal_damping',
                '--network-interface',self.interface,'--output',str(self.output_dir.resolve()),
                '--parent-pid',str(os.getpid()),'--ready-fd',str(write_fd)],
                cwd=root,pass_fds=(write_fd,))
            os.close(write_fd);write_fd=None
            ready,_,_=select.select([read_fd],[],[],10.)
            message=os.read(read_fd,4096).decode(errors='replace') if ready else 'startup timed out'
            if message.strip()!='READY':raise RuntimeError('终端阻尼监听启动失败：'+message)
            return True
        except BaseException:
            self.close();raise
        finally:
            os.close(read_fd)
            if write_fd is not None:os.close(write_fd)

    def close(self):
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:self.process.wait(timeout=20.)
                    except subprocess.TimeoutExpired:
                        self.process.kill();self.process.wait()
                        print('[DAMPING KEY] 监听进程未正常退出；请核对最后阻尼请求状态。',flush=True)
                self.process=None
        finally:
            if self.fd is not None and self.original is not None:
                try:termios.tcsetattr(self.fd,termios.TCSANOW,self.original)
                except termios.error:pass
                self.original=None

    def wait_in_standby(self):
        """导航资源收尾后仍保持前台保护；用户q+Enter才让父程序退出。"""
        if self.process is None:
            return
        if self.process.poll() is not None:
            print('[DAMPING KEY] 监听进程已退出，阻尼按键不可用。',flush=True)
            return
        self.process.send_signal(signal.SIGUSR1)
        print('[DAMPING KEY] 导航资源已收尾，保护仍启用；空格+Enter=阻尼，q+Enter=退出保护。',flush=True)
        self.process.wait()
        if self.process.returncode != 0:
            print('[DAMPING KEY] 保护监听异常退出，请核对机器人状态。',flush=True)


def listen(fd,client,interface,output,stop,*,parent_alive=lambda:True,confirmation_timeout_s=5.,
           standby=None,stop_requested=None,on_stop=None):
    """空格只准备，随后Enter确认。Ctrl+C保留原信号处理，不触发阻尼。"""
    deadline=None;quit_deadline=None;announced=False
    standby=standby if standby is not None else threading.Event()
    stop_requested=stop_requested if stop_requested is not None else threading.Event()
    while not stop.is_set() and parent_alive():
        if stop_requested.is_set():
            stop_requested.clear();deadline=None;quit_deadline=None
            if not standby.is_set():
                try:
                    if on_stop is not None:on_stop()
                except Exception as error:
                    print(f'[DAMPING KEY] 停车请求异常：{error}；仍保留阻尼按键，请观察机器人。',flush=True)
                standby.set()
        if standby.is_set() and not announced:
            deadline=None;quit_deadline=None;announced=True
            print('\n[DAMPING KEY] 保护待命：空格→Enter仍可请求阻尼；q→Enter退出保护。'
                  'Ctrl+C不关闭保护，不自动恢复站立。',flush=True)
        if quit_deadline is not None and time.monotonic()>=quit_deadline:
            quit_deadline=None
            print('[DAMPING KEY] 退出确认超时，保护继续。',flush=True)
        if deadline is not None and time.monotonic()>=deadline:
            deadline=None
            print('\n[DAMPING KEY] 确认超时，已取消准备；未请求阻尼。',flush=True)
        ready,_,_=select.select([fd],[],[],.1)
        if not ready:continue
        key=os.read(fd,1)
        if not key:
            print('[DAMPING KEY] 终端输入已关闭，按键保护退出。',flush=True)
            return
        if standby.is_set() and key in (b'q',b'Q'):
            deadline=None;quit_deadline=time.monotonic()+confirmation_timeout_s
            print('[DAMPING KEY] 5秒内按Enter关闭保护；关闭后此终端不能再触发阻尼。',flush=True)
            continue
        if quit_deadline is not None:
            if key in (b'\r',b'\n') and time.monotonic()<quit_deadline:
                print('[DAMPING KEY] 已明确退出保护；未切换机器人模式。',flush=True)
                return
            quit_deadline=None
            print('[DAMPING KEY] 已取消退出，保护继续。',flush=True)
        if key==b' ':
            if deadline is None:
                deadline=time.monotonic()+confirmation_timeout_s
                print('\n[DAMPING KEY] 已准备：5秒内按Enter请求DAMPING；'+
                      ('保护待命中。' if standby.is_set() else '仅准备未停车，任务仍运行。')+
                      '其他键取消，Ctrl+C停车并保留保护。',flush=True)
            continue
        if deadline is None:continue
        if key not in (b'\r',b'\n') or time.monotonic()>=deadline:
            deadline=None
            print('\n[DAMPING KEY] 已取消准备；未请求阻尼。',flush=True)
            continue
        deadline=None
        print('\n[DAMPING KEY] Enter已确认：锁存导航并独立请求damping…',flush=True)
        report=request_damping(client,latch_path(interface),interface=interface)
        report['trigger']='terminal_space_then_enter'
        report['standby_after_navigation']=standby.is_set()
        output=Path(output)
        try:
            output.mkdir(parents=True,exist_ok=True)
            path=output/(datetime.now().strftime('terminal_%Y%m%d_%H%M%S_%f')+'.json')
            path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print('[DAMPING KEY] 记录：'+str(path),flush=True)
        except OSError as error:print(f'[DAMPING KEY] 日志保存失败：{error}',flush=True)
        print('[DAMPING KEY] '+('已确认FSM=1：DAMPING。' if report['confirmed'] else
              '未确认进入阻尼！请使用原有紧急操作，可再按空格+Enter重试。')+
              (' 本机导航锁存保持。' if report['local_latch_written'] else
               ' 锁存写入失败，其他控制命令可能仍在发送！'),flush=True)
        # 丢弃请求期间积累的按键，避免长按空格导致请求不断重复。
        termios.tcflush(fd,termios.TCIFLUSH)


def request_standby_stop(client,interface,output):
    """仅用于不启动导航的吊起测试；Ctrl+C请求零速度，不切换FSM/站立。"""
    report={'trigger':'terminal_ctrl_c','automatic_recovery':False,
            'requested_command':[0.,0.,0.],'physical_stop_confirmed':False}
    if latch_path(interface).exists():
        report['skipped']='damping_latched; no velocity RPC sent'
    else:
        try:
            report['request_code']=client.SetVelocity(0.,0.,0.)
            report['fsm_readback']=client.GetFsmId()
        except Exception as error:report['error']=str(error)
    try:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        path=output/(datetime.now().strftime('standby_stop_%Y%m%d_%H%M%S_%f')+'.json')
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print('[DAMPING KEY] Ctrl+C停车请求记录：'+str(path),flush=True)
    except OSError as error:print(f'[DAMPING KEY] 停车记录保存失败：{error}',flush=True)
    print('[DAMPING KEY] '+str(report)+'；请观察机器人，下面仍保留阻尼保护。',flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--network-interface',default='enxf8e43bbea486')
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[3]/'outputs/damping_key')
    p.add_argument('--parent-pid',type=int)
    p.add_argument('--ready-fd',type=int)
    args=p.parse_args(argv)
    ready_fd=args.ready_fd
    stop=threading.Event();standby=threading.Event();stop_requested=threading.Event();original=None;fd=None
    # 正式任务Ctrl+C交给runner停车收尾；独立吊起测试先停车，再进入保护待命。
    signal.signal(signal.SIGINT,signal.SIG_IGN if args.parent_pid is not None else
                  lambda *_:stop_requested.set() if not standby.is_set() else None)
    signal.signal(signal.SIGUSR1,lambda *_:standby.set())
    signal.signal(signal.SIGTERM,lambda *_:stop.set())
    try:
        if not sys.stdin.isatty():raise RuntimeError('需要真实交互终端')
        previous_latch=snapshot_latch(latch_path(args.network_interface))
        fd=sys.stdin.fileno();original=termios.tcgetattr(fd)
        tty.setcbreak(fd,termios.TCSANOW)  # 逐字符读空格/Enter；保留ISIG，使Ctrl+C仍产生SIGINT。
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        ChannelFactoryInitialize(0,args.network_interface)
        client=LocoClient();client.SetTimeout(1.);client.Init()
        if args.parent_pid is None:
            try:auto_clear_previous_latch(client,latch_path(args.network_interface),previous_latch)
            except Exception as error:
                # 单独保护测试仍可请求阻尼；这里不创建导航任务，也不恢复模式。
                print(f'[DAMPING KEY] {error}；保留锁存，仍启用阻尼按键。',flush=True)
        print('[DAMPING KEY] 已启用：空格→5秒内Enter=阻尼；Ctrl+C=停车并保留保护；待命时q→Enter退出保护。',flush=True)
        if ready_fd is not None:
            os.write(ready_fd,b'READY\n');os.close(ready_fd);ready_fd=None
        listen(fd,client,args.network_interface,args.output,stop,
               parent_alive=lambda:args.parent_pid is None or os.getppid()==args.parent_pid,
               standby=standby,stop_requested=stop_requested,
               on_stop=lambda:request_standby_stop(client,args.network_interface,args.output))
        return 0
    except Exception as error:
        if ready_fd is not None:
            os.write(ready_fd,str(error).encode());os.close(ready_fd);ready_fd=None
        print(f'[DAMPING KEY] 监听不可用：{error}',flush=True);return 1
    finally:
        if original is not None:
            try:termios.tcsetattr(fd,termios.TCSANOW,original)
            except termios.error:pass

if __name__=='__main__':raise SystemExit(main())
