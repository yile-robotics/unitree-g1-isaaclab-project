from __future__ import annotations

from pathlib import Path
import signal
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_g1_real  # noqa: E402


class _Resource:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events

    def stop(self) -> None:
        self.events.append(f"{self.name}.stop")

    def close(self) -> None:
        self.events.append(f"{self.name}.close")


class RealG1SignalTest(unittest.TestCase):
    def tearDown(self) -> None:
        run_g1_real._active_episode = None
        run_g1_real._active_dds = None
        run_g1_real._active_camera = None
        run_g1_real._active_odometry = None
        run_g1_real._active_panorama_imu = None

    def test_sigint_unwinds_into_run_finally_instead_of_force_exit(self):
        with self.assertRaises(KeyboardInterrupt):
            run_g1_real._signal_handler(signal.SIGINT, None)

    def test_signal_latches_independent_stop_before_unwinding_or_logging(self):
        events = []
        class Backend:
            def request_shutdown_stop(self,reason):events.append(reason)
        run_g1_real._active_dds = Backend()
        with patch('builtins.print',side_effect=lambda *a,**kw:events.append('print')):
            with self.assertRaises(KeyboardInterrupt):
                run_g1_real._signal_handler(signal.SIGINT,None)
        self.assertEqual(events,['user_interrupt','print'])

    def test_repeated_sigint_does_not_interrupt_cleanup(self):
        with patch.object(run_g1_real, '_shutdown_in_progress', True):
            run_g1_real._signal_handler(signal.SIGINT, None)

    def test_shutdown_stops_before_closing_resources(self):
        events = []
        run_g1_real._active_dds = _Resource('dds', events)
        run_g1_real._active_camera = _Resource('camera', events)
        run_g1_real._active_odometry = _Resource('odometry', events)
        run_g1_real._active_panorama_imu = _Resource('imu', events)
        run_g1_real._shutdown_active_resources(reason='user_interrupt')
        self.assertEqual(events, ['dds.stop','dds.close','camera.close','imu.close','odometry.close'])

    def test_main_registers_sigint_before_running(self):
        parser = MagicMock()
        parser.parse_args.return_value = object()
        with (
            patch.object(run_g1_real.signal, "signal") as register,
            patch.object(run_g1_real, "build_parser", return_value=parser),
            patch.object(run_g1_real, "run", return_value=0) as run,
        ):
            result = run_g1_real.main()

        self.assertEqual(register.call_args_list, [
            unittest.mock.call(signal.SIGINT, run_g1_real._signal_handler),
            unittest.mock.call(signal.SIGTERM, run_g1_real._signal_handler)])
        run.assert_called_once_with(parser.parse_args.return_value)
        self.assertEqual(result, 0)

    def test_ctrl_c_finishes_cleanup_and_records_before_waiting_in_protection(self):
        events=[]
        dds=_Resource('dds',events)
        monitor=MagicMock()
        monitor.wait_in_standby.side_effect=lambda:events.append('standby')
        monitor.close.side_effect=lambda:events.append('monitor.close')
        records=MagicMock()
        records.local_exit.side_effect=lambda *a:events.append('local_exit') or {}
        records.save.side_effect=lambda *a:events.append('saved')
        records.append.side_effect=lambda *a,**k:events.append('journal')
        episode=SimpleNamespace(run_records=records,remote_session_active=True,
            failure_reason=None,session_success_reason=None,session_failure_reason=None,
            end_remote_session=lambda **kw:events.append('remote_end'))
        with patch.object(run_g1_real,'_active_dds',dds),\
             patch.object(run_g1_real,'_active_episode',episode),\
             patch.object(run_g1_real,'_active_terminal_damping',monitor):
            run_g1_real._shutdown_active_resources(reason='user_interrupt')
        self.assertLess(events.index('dds.stop'),events.index('local_exit'))
        for completed in ('dds.close','local_exit','remote_end','saved','journal'):
            self.assertLess(events.index(completed),events.index('standby'))
        self.assertEqual(events[-2:],['standby','monitor.close'])

    def test_motion_to_stand_edge_calls_stopmove_once(self):
        dds = MagicMock()
        moving = SimpleNamespace(
            desired_mode="locomotion",
            command=[0.3, 0.0, 0.1],
        )
        standing = SimpleNamespace(
            desired_mode="stand",
            command=[9.0, 9.0, 9.0],
        )

        command, mode = run_g1_real._apply_episode_command(dds, moving, "stand")
        self.assertEqual(command.tolist(), [0.3, 0.0, 0.1])
        self.assertEqual(mode, "locomotion")
        dds.set_velocity.assert_called_once_with(0.3, 0.0, 0.1)
        dds.stop.assert_not_called()

        dds.reset_mock()
        command, mode = run_g1_real._apply_episode_command(
            dds, standing, "locomotion"
        )
        self.assertEqual(command.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(mode, "stand")
        dds.stop.assert_called_once_with()
        dds.set_velocity.assert_not_called()

        dds.reset_mock()
        run_g1_real._apply_episode_command(dds, standing, "stand")
        dds.stop.assert_not_called()
        dds.set_velocity.assert_called_once_with(0.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
