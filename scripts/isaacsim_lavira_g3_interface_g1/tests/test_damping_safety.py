"""假SDK验证阻尼请求、跨进程锁存及命令禁止；不连接硬件。"""
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unified_vln.damping_safety import latch_path,write_latch,request_damping,snapshot_latch,auto_clear_previous_latch
from unified_vln.g1_dds_backend import UnitreeG1DDSBackend

class DampingSafetyTest(unittest.TestCase):
    def test_request_latches_before_rpc_then_confirms_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'latch.json';client=Mock()
            def send(fsm):
                self.assertTrue(path.exists());self.assertEqual(fsm,1);return 0
            client.SetFsmId.side_effect=send
            client.GetFsmId.side_effect=[(0,802),(0,1)]
            report=request_damping(client,path,interface='eth0',sleep=lambda _:None)
            self.assertTrue(report['confirmed']);self.assertTrue(report['local_latch_written'])
            client.SetFsmId.assert_called_once_with(1)
            client.HighStand.assert_not_called();client.Move.assert_not_called()

    def test_failure_keeps_latch_and_does_not_claim_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'latch.json';client=Mock()
            client.SetFsmId.side_effect=TimeoutError('offline')
            report=request_damping(client,path,interface='eth0',sleep=lambda _:None)
            self.assertFalse(report['confirmed']);self.assertTrue(path.exists())
            self.assertEqual(client.SetFsmId.call_count,3)

    def test_latch_io_failure_does_not_prevent_direct_damping_request(self):
        client=Mock();client.SetFsmId.return_value=0;client.GetFsmId.return_value=(0,1)
        with patch('unified_vln.damping_safety.write_latch',side_effect=OSError('disk unavailable')):
            report=request_damping(client,Path('/unused'),interface='eth0',sleep=lambda _:None)
        self.assertFalse(report['local_latch_written']);self.assertTrue(report['confirmed'])
        client.SetFsmId.assert_called_once_with(1)

    def test_path_is_interface_specific_and_rejects_traversal(self):
        self.assertNotEqual(latch_path('eth0'),latch_path('eth1'))
        with self.assertRaises(ValueError):latch_path('../x')

    def test_backend_suppresses_commands_and_cleanup_and_stays_latched_after_file_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=UnitreeG1DDSBackend.__new__(UnitreeG1DDSBackend)
            b.lock=threading.Lock();b.shutdown_requested=threading.Event();b.damping_requested=threading.Event()
            b.damping_latch_path=Path(tmp)/'latch.json';b.client=Mock()
            b.target_command=(.6,0,0);b.command_period_s=.02;b.running=True
            write_latch(b.damping_latch_path,interface='eth0')
            with self.assertRaisesRegex(RuntimeError,'Damping latched'):b.set_velocity(.6,0,0)
            with self.assertRaisesRegex(RuntimeError,'HighStand rejected'):b.high_stand()
            b.stop();b.client.StopMove.assert_not_called()
            with patch('unified_vln.g1_dds_backend.time.sleep',side_effect=lambda _:setattr(b,'running',False)):
                b._command_loop()
            b.client.Move.assert_not_called();b.client.HighStand.assert_not_called()
            b.damping_latch_path.unlink()
            with self.assertRaises(RuntimeError):b.set_velocity(.6,0,0)

    def test_backend_startup_is_blocked_before_sdk_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'latched';path.touch()
            with patch('unified_vln.g1_dds_backend.latch_path',return_value=path):
                with self.assertRaisesRegex(RuntimeError,'startup blocked'):
                    UnitreeG1DDSBackend('eth0')

    def test_new_start_auto_clears_old_latch_only_after_reading_802(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'latch.json';write_latch(p,interface='eth0');old=snapshot_latch(p)
            client=Mock();client.GetFsmId.return_value=(0,802)
            report=auto_clear_previous_latch(client,p,old)
            self.assertEqual(report['status'],'AUTO_CLEARED');self.assertFalse(p.exists())
            client.GetFsmId.assert_called_once();client.SetFsmId.assert_not_called()
            client.HighStand.assert_not_called();client.Move.assert_not_called()

    def test_damping_unknown_mode_or_read_failure_keeps_latch(self):
        for state in [(0,1),(0,0),(0,999),(3102,None)]:
            with self.subTest(state=state),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'latch.json';write_latch(p,interface='eth0');old=snapshot_latch(p)
                client=Mock();client.GetFsmId.return_value=state
                with self.assertRaises(RuntimeError):auto_clear_previous_latch(client,p,old)
                self.assertEqual(snapshot_latch(p),old)

    def test_new_emergency_during_fsm_read_cannot_be_auto_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'latch.json';write_latch(p,interface='eth0');old=snapshot_latch(p)
            client=Mock()
            def racing_read():write_latch(p,interface='eth0');return 0,802
            client.GetFsmId.side_effect=racing_read
            with self.assertRaisesRegex(RuntimeError,'新的阻尼'):auto_clear_previous_latch(client,p,old)
            self.assertTrue(p.exists());self.assertNotEqual(snapshot_latch(p),old)

    def test_latch_created_after_start_snapshot_is_not_old_latch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'latch.json';old=snapshot_latch(p);write_latch(p,interface='eth0')
            client=Mock()
            with self.assertRaisesRegex(RuntimeError,'新阻尼'):auto_clear_previous_latch(client,p,old)
            client.GetFsmId.assert_not_called();self.assertTrue(p.exists())

    def test_no_latch_needs_no_rpc_or_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'latch.json';client=Mock()
            self.assertEqual(auto_clear_previous_latch(client,p,None)['status'],'NO_LATCH')
            client.GetFsmId.assert_not_called()

    def test_backend_auto_clear_happens_before_sender_and_never_recovers_mode(self):
        for fsm in (802,1):
            with self.subTest(fsm=fsm),tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'latch.json';write_latch(p,interface='eth0');old=snapshot_latch(p)
                modules={}
                for name in ('unitree_sdk2py','unitree_sdk2py.core','unitree_sdk2py.core.channel',
                             'unitree_sdk2py.g1','unitree_sdk2py.g1.loco','unitree_sdk2py.g1.loco.g1_loco_client',
                             'unitree_sdk2py.idl','unitree_sdk2py.idl.unitree_go','unitree_sdk2py.idl.unitree_go.msg',
                             'unitree_sdk2py.idl.unitree_go.msg.dds_'):
                    modules[name]=types.ModuleType(name);modules[name].__path__=[]
                client=Mock();client.GetFsmId.return_value=(0,fsm)
                channel=modules['unitree_sdk2py.core.channel']
                channel.ChannelFactoryInitialize=Mock();channel.ChannelSubscriber=Mock()
                modules['unitree_sdk2py.g1.loco.g1_loco_client'].LocoClient=lambda:client
                modules['unitree_sdk2py.idl.unitree_go.msg.dds_'].SportModeState_=object
                with patch.dict(sys.modules,modules),\
                     patch('unified_vln.g1_dds_backend.latch_path',return_value=p),\
                     patch('unified_vln.g1_dds_backend.threading.Thread') as sender:
                    if fsm==802:
                        b=UnitreeG1DDSBackend('eth0',auto_clear_damping_latch=True,startup_latch_snapshot=old)
                        self.assertFalse(b.damping_requested.is_set());self.assertFalse(p.exists())
                        sender.return_value.start.assert_called_once()
                    else:
                        with self.assertRaises(RuntimeError):
                            UnitreeG1DDSBackend('eth0',auto_clear_damping_latch=True,startup_latch_snapshot=old)
                        sender.assert_not_called();self.assertTrue(p.exists())
                client.Move.assert_not_called();client.SetFsmId.assert_not_called();client.HighStand.assert_not_called()

if __name__=='__main__':unittest.main()
