"""终端摘要忠实显示模型/地图/规划数据，不介入控制。"""
from contextlib import redirect_stdout
import copy
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unified_vln import real_console as console
from unified_vln.real_run_records import RecordingPlanner


class ConsoleTest(unittest.TestCase):
    def raw(self):
        return {'action':'NAVIGATE','direction':'left','target':'绿色瓶子',
                'bbox_2d':[320,468,960,706], 'action_source':'RECOVERY',
                'reasoning':'turn left\nthen walk',
                'stage_progress':{'parse_success':True,'stage_total':2,'stage_completed':1,
                    'stage_id':1,'current_stage':'向左转走到绿色瓶子前停下','completed_stage_ids':[0]}}

    def test_raw_message_box_frame_and_zero_based_stage_are_unambiguous(self):
        raw=self.raw();before=copy.deepcopy(raw)
        frame=SimpleNamespace(frame_id=9159,rgb=np.zeros((720,1280,3),np.uint8))
        lines=console.model_lines(3,raw,frame)
        text='\n'.join(lines)
        for item in ['来源=RECOVERY','方向=left(左方)','bbox_xyxy=[320,468,960,706]',
                     '所选原图=left','尺寸=1280x720','帧=9159','解析=成功',
                     '已完成=1/2','第2阶段(stage_id=1)','向左转走到绿色瓶子前停下']:
            self.assertIn(item,text)
        self.assertEqual(raw,before)
        self.assertTrue(all('\n' not in line for line in lines))

    def test_parse_error_does_not_invent_valid_stage(self):
        raw=self.raw();raw['stage_progress']={'parse_success':False,'parse_error':'stage mismatch'}
        text='\n'.join(console.model_lines(2,raw))
        self.assertIn('解析=失败',text)
        self.assertIn('未知阶段',text)
        self.assertIn('stage mismatch',text)
        self.assertNotIn('第1阶段',text)

    def test_completed_sentinel_is_not_a_third_subgoal(self):
        raw=self.raw()
        raw['stage_progress'].update(stage_completed=2,stage_id=2,current_stage='COMPLETED')
        text='\n'.join(console.model_lines(4,raw))
        self.assertIn('全部阶段完成',text)
        self.assertIn('已完成=2/2',text)
        self.assertNotIn('第3阶段',text)

    def test_confirmed_stop_message_still_distinguishes_pending_remote_end(self):
        raw={'action':'STOP','control':'STOP_CONFIRMED'}
        text='\n'.join(console.execution_lines(4,raw,'stopped'))
        self.assertIn('STOP已确认',text)
        self.assertIn('等待会话结束确认',text)
        self.assertNotIn('此行不表示任务成功',text)

    def test_stop_gate_audit_and_recovery_are_visible(self):
        raw={**self.raw(),'action':'STOP','control':'PREEMPT',
             'stop_gate':{'verdict':'PREMATURE','completed':False,'missing_subgoal':'绿色瓶子'},
             'failure_verification':{'verdict':'FAILURE','need_recovery':True},
             'phase7':{'status':'AUDIT_RECORDED','audit':{'candidate':False,'moved':False,
                        'sustained':False,'contradiction_evidence':['方向不对']}},
             'recovery':{'status':'READY','parse_success':True}}
        text='\n'.join(console.model_lines(2,raw))
        for item in ['PREMATURE','需要恢复=True','candidate=False','moved=False','方向不对','[Recovery]']:
            self.assertIn(item,text)
        result='\n'.join(console.execution_lines(2,raw,'wait_action_stand',accepted=True))
        self.assertIn('不执行上述模型运动动作',result)
        self.assertNotIn('已接受导航',result)

    def test_zero_new_cells_are_real_map_stats_and_window_span_is_actual(self):
        lines=console.map_lines({'decision_index':1,'window_index':2,'new_explored_cells':0,
            'explored_cells':21257,'occupied_cells':5000,'traversable_cells':2516,
            'resolution_m':.05,'update_count':20,'map_update_failures':0})
        self.assertIn('新增探索cell=0',lines[0])
        self.assertIn('来源=RGB-D探索地图',lines[0])
        result=console.window_lines({'decision_index':1,'window_index':2,
            'timestamp_start':10.,'timestamp_end':11.3,'displacement_m':.485,
            'distance_to_local_goal_start':1.82,'distance_to_local_goal_end':1.39},
            {'control':'CONTINUE','phase5':{'status':'MONITORING','candidate':None}},.054)
        self.assertIn('实际时长=1.30s',result[0])
        self.assertIn('位移=0.485m',result[0])
        self.assertIn('候选=无',result[0])

    def test_console_failures_do_not_escape(self):
        with patch('builtins.print',side_effect=BrokenPipeError()):
            console.emit(lambda:['line'])
        console.emit(lambda:1/0)

    def test_planner_print_is_throttled_and_returns_original_result(self):
        now=SimpleNamespace(value=10.)
        result=(np.array([[0,0],[1,0]]),.085)
        def plan(frame,goal):
            now.value+=.05
            return result
        planner=RecordingPlanner(SimpleNamespace(get_plan=plan),Mock())
        output=StringIO();frame=SimpleNamespace(frame_id=42)
        with patch('unified_vln.real_run_records.time.monotonic',side_effect=lambda:now.value),redirect_stdout(output):
            self.assertIs(planner.get_plan(frame,np.array([1.,0.])),result)
            now.value+=.3;planner.get_plan(frame,np.array([1.,0.]))
            now.value+=1.;planner.get_plan(frame,np.array([1.,0.]))
        self.assertEqual(output.getvalue().count('[iPlanner规划]'),2)
        self.assertIn('路径点=2 fear=0.085',output.getvalue())

    def test_plan_and_fallback_have_distinct_labels(self):
        raw=self.raw()
        lines=console.execution_lines(1,raw,'plan_after_rotation',accepted=True,projection={
            'pixel_uv':[1104,719],'depth_m':None,'valid_depth_count':0,
            'used_forward_fallback':True,'goal_after_turn_xy_m':[1.5,0.]})
        self.assertIn('兜底=True',lines[-1])
        self.assertIn('深度=--m',lines[-1])
        self.assertIn('通过后才可能行走',lines[0])


if __name__=='__main__':unittest.main()
