"""离线验证中断落盘、HTTP失败、最后半窗口及父子进程收尾；不创建硬件。"""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g1_pelvis_vln as entry
import run_g1_real as runner
import test_real_episode_panorama as panorama_tests
from unified_vln.episode import EpisodeState, LocalEndToEndEpisode
from unified_vln.odometry import Pose2D
from unified_vln.real_run_records import atomic_json, RecordingSessionClient, RecordingPlanner


class RunRecordsTest(unittest.TestCase):
    def tearDown(self):
        for name in ('_active_dds','_active_camera','_active_odometry',
                     '_active_episode','_active_panorama_imu'):
            setattr(runner, name, None)
        runner._shutdown_in_progress = False

    def test_atomic_replacement_failure_keeps_previous_complete_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'record.json'
            atomic_json(path, {'old':True})
            with patch('unified_vln.real_run_records.os.replace', side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    atomic_json(path, {'new':True})
            self.assertEqual(json.loads(path.read_text()), {'old':True})
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def shutdown_case(self, reason, *, http_error=False, close_error=False):
        with tempfile.TemporaryDirectory() as tmp:
            episode, state, _ = panorama_tests.RealPanoramaTest().fixture(output=Path(tmp))
            episode.state = EpisodeState.EXECUTING
            episode.action_elapsed_s = .45
            episode._motion_window_start_pose = Pose2D(.8,2,0,0)
            episode._active_world_trace = [[.8,2], [1,2]]
            state.now = .45
            episode.run_records.sample(episode, 'command_requested', [.6,0,.1])
            episode._remote_session_active = True
            events = []
            outer = self
            class Client:
                timeout_s = 180.
                def end_session(self, **kwargs):
                    events.append('http')
                    # 验证HTTP阻塞/失败之前本地退出和不足一秒数据已经存在。
                    outer.assertTrue((episode.output_dir/'run_exit.json').is_file())
                    outer.assertTrue((episode.output_dir/'execution_tail.json').is_file())
                    outer.assertLessEqual(self.timeout_s, 5.)
                    if http_error:
                        raise RuntimeError('server offline')
                    return SimpleNamespace(final_status=kwargs['status'], reason=kwargs['reason']), kwargs
            client = Client()
            episode.session_client = RecordingSessionClient(client, episode.run_records)
            class Backend:
                def stop(self):
                    events.append('stop')
                    outer.assertFalse((episode.output_dir/'run_exit.json').exists())
                def close(self):
                    events.append('close')
                    if close_error:
                        raise RuntimeError('close failed')
                def guard_report(self):
                    return {'last_requested_command':[0,0,0]}
            runner._active_dds = Backend()
            runner._active_episode = episode
            runner._active_camera = SimpleNamespace(close=lambda:events.append('camera.close'))
            if reason == 'runner_completed':
                episode.state = EpisodeState.STOPPED
                episode.session_success_reason = 'stop_confirmed'
            runner._shutdown_active_resources(reason=reason, error='control failure' if reason=='runner_exception' else None)
            saved = json.loads((episode.output_dir/'run_exit.json').read_text())
            tail = json.loads((episode.output_dir/'execution_tail.json').read_text())
            self.assertEqual(saved['exit_reason'], reason)
            self.assertEqual(saved['phase'], 'CLEANUP_FINISHED')
            self.assertEqual(saved['last_command_before_shutdown'], [.6,0,.1])
            self.assertFalse(saved['physical_stop_confirmed'])
            self.assertTrue(saved['stop_request']['call_returned'])
            self.assertEqual(saved['final_pose']['x'], 1.)
            self.assertTrue(tail['partial'])
            self.assertFalse(tail['uploaded'])
            self.assertAlmostEqual(tail['duration_s'], .45)
            self.assertAlmostEqual(tail['displacement_m'], .2)
            self.assertEqual(events, ['stop','close','camera.close','http'])
            self.assertTrue((episode.output_dir/'real_input_guard.json').is_file())
            attempt = json.loads((episode.output_dir/'g3_session_end_attempt.json').read_text())
            self.assertEqual(attempt['request']['status'], 'SUCCESS' if reason=='runner_completed' else 'FAILURE')
            self.assertEqual(saved['remote_end_status'], 'UNCONFIRMED' if http_error else 'CONFIRMED')
            self.assertEqual(client.timeout_s, 180.)
            self.assertEqual((episode.output_dir/'g3_session_ended.json').exists(), not http_error)
            # 重复清理不会重复发HTTP或覆盖完整记录。
            runner._shutdown_active_resources()
            self.assertEqual(events.count('http'), 1)

    def test_manual_stop_retains_partial_window_even_when_remote_unavailable(self):
        self.shutdown_case('user_interrupt', http_error=True)

    def test_normal_confirmed_stop_records_success(self):
        self.shutdown_case('runner_completed')

    def test_exception_and_close_failure_still_save_and_end_session(self):
        self.shutdown_case('runner_exception', close_error=True)

    def test_stale_final_pose_is_explicitly_marked_as_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode,state,_ = panorama_tests.RealPanoramaTest().fixture(output=Path(tmp))
            episode.run_records.sample(episode, 'update_begin')
            state.map_live = False
            saved = episode.run_records.local_exit(episode,'user_interrupt',None,{})
            self.assertEqual(saved['final_pose_source'],'last_recorded_pose')
            self.assertIsNotNone(saved['final_pose'])

    def test_request_survives_interrupt_before_http_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode,_,_ = panorama_tests.RealPanoramaTest().fixture(output=Path(tmp))
            client = SimpleNamespace(report_motion_window=lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
            recording = RecordingSessionClient(client,episode.run_records)
            with self.assertRaises(KeyboardInterrupt):
                recording.report_motion_window(decision_index=2,window_index=3,displacement_m=.4)
            saved = json.loads((episode.output_dir/'g3_decision_002_motion_0003_request.json').read_text())
            self.assertEqual(saved['displacement_m'],.4)
            events = [json.loads(line) for line in (episode.output_dir/'execution_events.jsonl').read_text().splitlines()]
            self.assertEqual([e['event'] for e in events if e['event']!='timing'],['motion_window_request'])

    def test_first_motion_window_uses_actual_pose_interval_not_pre_execution_dt(self):
        episode,state,_ = panorama_tests.RealPanoramaTest().fixture()
        episode.state = EpisodeState.EXECUTING
        episode.action_elapsed_s = 5.  # 包含进入执行前的大循环间隔
        episode._motion_window_start_pose = Pose2D(1,2,0,0)
        state.now = .7
        with patch.object(LocalEndToEndEpisode,'_report_motion_window_if_due',return_value='original') as report:
            self.assertIsNone(episode._report_motion_window_if_due())
            report.assert_not_called()
            state.now = 1.05
            self.assertEqual(episode._report_motion_window_if_due(),'original')
            report.assert_called_once()

    def test_planner_timing_keeps_original_frame_goal_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            episode,_,_ = panorama_tests.RealPanoramaTest().fixture(output=Path(tmp))
            frame,goal,result = SimpleNamespace(frame_id=42),[2.,.1],([[0,0],[2,0]],.1)
            calls = []
            planner = SimpleNamespace(get_plan=lambda f,g:(calls.append((f,g)),result)[1])
            recorder = RecordingPlanner(planner,episode.run_records)
            self.assertIs(recorder.get_plan(frame,goal),result)
            self.assertIs(calls[0][0],frame)
            self.assertIs(calls[0][1],goal)
            episode.run_records.save_timings()
            summary = json.loads((episode.output_dir/'timing_summary.json').read_text())
            self.assertEqual(summary['iplanner_call_s']['count'],1)
            self.assertGreaterEqual(summary['iplanner_call_s']['p95_s'],0.)

    def test_parent_waits_for_child_cleanup_longer_than_subprocess_interrupt_grace(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            ready, done = folder/'ready',folder/'saved'
            script = """import signal,time,pathlib,sys
def stop(sig,frame):
    time.sleep(.6)
    pathlib.Path(sys.argv[2]).write_text('local data saved')
    print('CLEANUP DONE',flush=True)
    raise SystemExit(130)
signal.signal(signal.SIGINT,stop)
pathlib.Path(sys.argv[1]).write_text('ready')
print('READY',flush=True)
time.sleep(10)
"""
            original_popen = subprocess.Popen
            children = []
            def launch(*args,**kwargs):
                child = original_popen(*args,**kwargs)
                children.append(child)
                original_wait = child.wait
                calls = 0
                def wait(*a,**kw):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        deadline = time.monotonic()+5
                        while not ready.exists() and time.monotonic()<deadline:
                            time.sleep(.01)
                        if not ready.exists():
                            raise RuntimeError('offline child did not start')
                        raise KeyboardInterrupt()
                    return original_wait(*a,**kw)
                child.wait = wait
                return child
            try:
                with patch.object(entry.subprocess,'Popen',side_effect=launch), redirect_stdout(StringIO()):
                    code,interrupted = entry._run_formal_runner(
                        [sys.executable,'-c',script,str(ready),str(done)],folder/'console.log')
                self.assertEqual(code,130)
                self.assertTrue(interrupted)
                self.assertEqual(done.read_text(),'local data saved')
                self.assertIn('CLEANUP DONE',(folder/'console.log').read_text())
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.wait()


if __name__ == '__main__':
    unittest.main()
