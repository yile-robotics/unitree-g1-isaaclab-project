"""真实伪终端+假SDK验证同终端空格，无机器人命令。"""
import json
import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import unittest
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unified_vln.terminal_damping import TerminalDampingMonitor,listen,request_standby_stop

class TerminalDampingTest(unittest.TestCase):
    def test_only_space_then_enter_triggers_fake_sdk_and_writes_confirmation(self):
        master,slave=pty.openpty();original=termios.tcgetattr(slave)
        stop=threading.Event();client=Mock();client.SetFsmId.return_value=0
        def confirmed():stop.set();return 0,1
        client.GetFsmId.side_effect=confirmed
        errors=[]
        def run(out):
            try:listen(slave,client,'eth0',out,stop)
            except Exception as e:errors.append(e)
        try:
            tty.setcbreak(slave,termios.TCSANOW)
            self.assertTrue(termios.tcgetattr(slave)[3]&termios.ISIG)
            with tempfile.TemporaryDirectory() as tmp,patch('unified_vln.terminal_damping.latch_path',return_value=Path(tmp)/'latch'):
                thread=threading.Thread(target=run,args=(Path(tmp)/'logs',));thread.start()
                os.write(master,b'x\n')
                self.assertFalse(client.SetFsmId.called)
                os.write(master,b' ')
                time.sleep(.05)
                client.SetFsmId.assert_not_called()
                self.assertFalse((Path(tmp)/'latch').exists())
                os.write(master,b'\n')
                thread.join(timeout=3.)
                self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
                client.SetFsmId.assert_called_once_with(1)
                report=json.loads(next((Path(tmp)/'logs').glob('*.json')).read_text())
                self.assertTrue(report['confirmed']);self.assertTrue((Path(tmp)/'latch').exists())
                self.assertEqual(report['trigger'],'terminal_space_then_enter')
                client.HighStand.assert_not_called();client.Move.assert_not_called()
        finally:
            stop.set();termios.tcsetattr(slave,termios.TCSANOW,original)
            os.close(master);os.close(slave)

    def test_lifecycle_restores_terminal_and_subprocess_inherits_stdin(self):
        master,slave=pty.openpty();original=termios.tcgetattr(slave)
        stream=Mock();stream.isatty.return_value=True;stream.fileno.return_value=slave
        child=Mock();child.poll.return_value=None
        def spawn(*args,**kwargs):
            self.assertNotIn('stdin',kwargs)
            self.assertIn('unified_vln.terminal_damping',args[0])
            tty.setcbreak(slave,termios.TCSANOW)
            return child
        monitor=TerminalDampingMonitor('eth0',Path('/unused'))
        try:
            with patch('unified_vln.terminal_damping.sys.stdin',stream),\
                 patch('unified_vln.terminal_damping.subprocess.Popen',side_effect=spawn),\
                 patch('unified_vln.terminal_damping.select.select',return_value=([0],[],[])),\
                 patch('unified_vln.terminal_damping.os.read',return_value=b'READY\n'):
                self.assertTrue(monitor.start())
                self.assertFalse(termios.tcgetattr(slave)[3]&termios.ICANON)
                monitor.close()
            child.terminate.assert_called_once();child.wait.assert_called_once()
            self.assertEqual(termios.tcgetattr(slave),original)
        finally:monitor.close();os.close(master);os.close(slave)

    def test_cancel_or_timeout_never_sends_damping(self):
        for expire in (False,True):
            with self.subTest(expire=expire),tempfile.TemporaryDirectory() as tmp:
                master,slave=pty.openpty();original=termios.tcgetattr(slave)
                stop=threading.Event();client=Mock()
                tty.setcbreak(slave,termios.TCSANOW)
                thread=threading.Thread(target=listen,args=(slave,client,'eth0',Path(tmp),stop),
                    kwargs={'confirmation_timeout_s':.1})
                try:
                    thread.start();os.write(master,b' ')
                    if expire:time.sleep(.25)
                    else:os.write(master,b'x')
                    os.write(master,b'\n');time.sleep(.15)
                    client.SetFsmId.assert_not_called();client.GetFsmId.assert_not_called()
                    self.assertEqual(list(Path(tmp).iterdir()),[])
                finally:
                    stop.set();thread.join(timeout=2.)
                    termios.tcsetattr(slave,termios.TCSANOW,original)
                    os.close(master);os.close(slave)

    def test_noninteractive_input_does_not_claim_keyboard_is_enabled(self):
        stream=Mock();stream.isatty.return_value=False
        with patch('unified_vln.terminal_damping.sys.stdin',stream),\
             patch('unified_vln.terminal_damping.subprocess.Popen') as spawn:
            monitor=TerminalDampingMonitor('eth0',Path('/unused'))
            self.assertFalse(monitor.start());monitor.close();spawn.assert_not_called()

    def test_stop_then_standby_still_accepts_space_enter_damping(self):
        master,slave=pty.openpty();original=termios.tcgetattr(slave)
        standby=threading.Event();stop_requested=threading.Event();stop=threading.Event()
        stop_requested.set();parked=threading.Event();client=Mock()
        client.SetFsmId.return_value=0
        client.GetFsmId.side_effect=lambda:(stop.set() or (0,1))
        try:
            tty.setcbreak(slave,termios.TCSANOW)
            with tempfile.TemporaryDirectory() as tmp,patch('unified_vln.terminal_damping.latch_path',return_value=Path(tmp)/'latch'):
                on_stop=Mock(side_effect=parked.set)
                thread=threading.Thread(target=listen,args=(slave,client,'eth0',Path(tmp)/'logs',stop),
                    kwargs={'standby':standby,'stop_requested':stop_requested,'on_stop':on_stop})
                thread.start();self.assertTrue(parked.wait(1.))
                self.assertTrue(thread.is_alive());client.SetFsmId.assert_not_called()
                os.write(master,b' \n');thread.join(timeout=3.)
                self.assertFalse(thread.is_alive());on_stop.assert_called_once()
                client.SetFsmId.assert_called_once_with(1)
                report=json.loads(next((Path(tmp)/'logs').glob('*.json')).read_text())
                self.assertTrue(report['standby_after_navigation'])
        finally:
            stop.set();termios.tcsetattr(slave,termios.TCSANOW,original);os.close(master);os.close(slave)

    def test_q_enter_can_only_exit_protection_in_standby(self):
        master,slave=pty.openpty();original=termios.tcgetattr(slave)
        standby=threading.Event();stop=threading.Event();client=Mock()
        try:
            tty.setcbreak(slave,termios.TCSANOW)
            thread=threading.Thread(target=listen,args=(slave,client,'eth0',Path('/unused'),stop),kwargs={'standby':standby})
            thread.start();os.write(master,b'q\n');time.sleep(.15)
            self.assertTrue(thread.is_alive())
            standby.set();time.sleep(.15);os.write(master,b'q\n');thread.join(timeout=2.)
            self.assertFalse(thread.is_alive());client.SetFsmId.assert_not_called()
        finally:
            stop.set();termios.tcsetattr(slave,termios.TCSANOW,original);os.close(master);os.close(slave)

    def test_standalone_ctrl_c_zero_velocity_is_not_highstand_or_damping(self):
        client=Mock();client.SetVelocity.return_value=0;client.GetFsmId.return_value=(0,802)
        with tempfile.TemporaryDirectory() as tmp,patch('unified_vln.terminal_damping.latch_path',return_value=Path(tmp)/'latch'):
            report=request_standby_stop(client,'eth0',Path(tmp)/'logs')
        self.assertEqual(report['request_code'],0)
        client.SetVelocity.assert_called_once_with(0.,0.,0.)
        client.HighStand.assert_not_called();client.SetFsmId.assert_not_called()

    def test_monitor_standby_waits_for_listener_exit_without_terminating_it(self):
        child=Mock();child.poll.return_value=None;child.returncode=0
        monitor=TerminalDampingMonitor('eth0',Path('/unused'));monitor.process=child
        monitor.wait_in_standby()
        import signal
        child.send_signal.assert_called_once_with(signal.SIGUSR1)
        child.wait.assert_called_once();child.terminate.assert_not_called()

    def test_real_sigint_then_damping_then_quit_with_fake_sdk_in_child_pty(self):
        # 真实子进程、SIGINT和TTY；SDK全用内存替身，绝不连接机器人。
        code='''
import sys,types
from pathlib import Path
for name in ['unitree_sdk2py','unitree_sdk2py.core','unitree_sdk2py.g1','unitree_sdk2py.g1.loco']:
    m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
channel=types.ModuleType('unitree_sdk2py.core.channel')
channel.ChannelFactoryInitialize=lambda *a:None
sys.modules[channel.__name__]=channel
sdk=types.ModuleType('unitree_sdk2py.g1.loco.g1_loco_client')
class Fake:
    def __init__(self):self.fsm=802
    def Init(self):pass
    def SetTimeout(self,t):pass
    def SetVelocity(self,*v):assert v==(0.,0.,0.);return 0
    def SetFsmId(self,fsm):assert fsm==1;self.fsm=1;return 0
    def GetFsmId(self):return 0,self.fsm
sdk.LocoClient=Fake;sys.modules[sdk.__name__]=sdk
from unified_vln import terminal_damping as t
t.latch_path=lambda _:Path(sys.argv[1])/'latch'
raise SystemExit(t.main(['--network-interface','fake0','--output',str(Path(sys.argv[1])/'reports')]))
'''
        master,slave=pty.openpty();original=termios.tcgetattr(slave);child=None;received=b''
        def until(text):
            nonlocal received
            deadline=time.monotonic()+5
            while text.encode() not in received and time.monotonic()<deadline:
                ready,_,_=select.select([master],[],[],.1)
                if ready:received+=os.read(master,8192)
            self.assertIn(text.encode(),received)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                child=subprocess.Popen([sys.executable,'-c',code,tmp],cwd=Path(__file__).resolve().parents[1],
                    stdin=slave,stdout=slave,stderr=slave)
                until('已启用');os.kill(child.pid,signal.SIGINT)
                until('保护待命');self.assertIsNone(child.poll())
                os.kill(child.pid,signal.SIGINT);os.write(master,b' \n')
                until('已确认FSM=1');self.assertIsNone(child.poll())
                reports=list((Path(tmp)/'reports').glob('terminal_*.json'))
                self.assertEqual(len(reports),1)
                self.assertTrue(json.loads(reports[0].read_text())['standby_after_navigation'])
                os.write(master,b'q\n');child.wait(timeout=3)
                self.assertEqual(child.returncode,0)
                self.assertEqual(termios.tcgetattr(slave),original)
        finally:
            if child is not None and child.poll() is None:child.kill();child.wait()
            os.close(master);os.close(slave)

if __name__=='__main__':unittest.main()
