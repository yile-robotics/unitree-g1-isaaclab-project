"""完整真机入口默认只读和显式执行隔离；所有外设均使用替身。"""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import g1_pelvis_vln as entry
import run_g1_real as runner
from unified_vln.map_progress import SparseMapConfig
from unified_vln.odometry import Pose2D
from test_pelvis_camera import frame


class PelvisEntryTest(unittest.TestCase):
    def exercise(self,execute=False,fail=False):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            camera=MagicMock();camera.nominal_config=SparseMapConfig()
            camera.capture_forward.return_value=frame()
            camera.provenance.return_value={'slam_base_verified':True}
            camera.last_metadata={}
            camera.create_exploration_map.return_value.integrate.return_value.to_dict.return_value={'valid_depth_samples':1}
            if fail:camera.capture_forward.side_effect=RuntimeError('no frame')
            odom=MagicMock();odom.frame_id='map';odom.child_frame_id='base_link'
            odom.get_pose.return_value=Pose2D(0,0,0,0)
            stack.enter_context(patch.object(entry,'PelvisCamera',return_value=camera))
            stack.enter_context(patch.object(entry,'Ros2OdometryProvider',return_value=odom))
            session=stack.enter_context(patch.object(entry.G3SessionClient,'from_decision_url'))
            session.return_value.health_check.return_value={'status':'ok','framework':'G3','commit':'test'}
            stack.enter_context(patch.object(entry,'urlopen',return_value=BytesIO(b'{"status":"ok","model_loaded":true}')))
            child=stack.enter_context(patch.object(entry,'_run_formal_runner',return_value=(0,False)))
            dds=stack.enter_context(patch.object(runner,'UnitreeG1DDSBackend'))
            stack.enter_context(patch.object(entry,'mark_display_motion'))
            stack.enter_context(redirect_stdout(StringIO()));stack.enter_context(redirect_stderr(StringIO()))
            output=Path(tmp)/'check'
            args=['--output',str(output)] + (['--execute'] if execute else [])
            code=entry.main(args)
            report=json.loads((output/'preflight_report.json').read_text())
            dds.assert_not_called();camera.close.assert_called_once();odom.close.assert_called_once()
            session.return_value.start_session.assert_not_called()
            if execute and not fail:
                command=child.call_args.args[0]
                parsed=runner.build_parser().parse_args(command[2:])
                self.assertEqual(parsed.walk_speed_m_s,.6)
                self.assertEqual(parsed.max_forward_speed_m_s,.6)
                self.assertEqual(parsed.rotation_speed_rad_s,.8)
                self.assertEqual(parsed.max_decisions,0)
                self.assertTrue(parsed.g3_session)
                self.assertEqual(parsed.camera_factory,'camera_d415.pelvis_camera:create_camera')
                self.assertEqual(report['status'],'RUNNER_EXITED')
            else:
                child.assert_not_called()
                self.assertIs(report['robot_motion_sent'],False)
            return code,report

    def test_default_only_checks_without_starting_motion_or_model_session(self):
        code,report=self.exercise()
        self.assertEqual(code,0);self.assertEqual(report['status'],'PREFLIGHT_PASS')
        self.assertIs(report['slam_base_verified'],True)
        self.assertIn('2026-10-20',report['slam_base_verification_source'])
        self.assertAlmostEqual(report['pelvis_height_above_floor_m'],.80)

    def test_explicit_execute_reuses_original_g3_runner(self):
        self.assertEqual(self.exercise(execute=True)[0],0)

    def test_preflight_failure_blocks_execute(self):
        self.assertEqual(self.exercise(execute=True,fail=True)[0],1)


if __name__=='__main__':unittest.main()
