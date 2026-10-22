"""真机终端的只读摘要；不解释或改写导航控制。"""
import json


def text(value):
    # 单行输出，避免模型文本中的换行/终端控制字符伪装成其它日志。
    return json.dumps(value,ensure_ascii=False,separators=(',',':'))


def number(value, digits=2):
    return '--' if value is None else f'{float(value):.{digits}f}'


def emit(formatter, *args, **kwargs):
    # 观察输出不能替代或阻止停车、原协议校验和异常处理。
    try:
        for line in formatter(*args,**kwargs):
            print(line,flush=True)
    except Exception:
        pass


def stage_plan_lines(payload):
    plan=payload.get('frozen_stage_plan') or {}
    goals=plan.get('subgoals') or []
    return [f'[任务阶段] 共{plan.get("stage_total","--")}阶段；以下为服务器冻结计划']+[
        f'[任务阶段] 第{i+1}阶段(stage_id={i})：{text(goal)}' for i,goal in enumerate(goals)]


def model_lines(index, raw, frame=None):
    direction=raw.get('direction')
    names={'forward':'前方','left':'左方','behind':'后方','right':'右方'}
    rows=[f'[模型返回] decision={index} 来源={raw.get("action_source","NAVIGATOR")} '
          f'动作={raw.get("action")} 方向={direction}({names.get(direction,"无")}) '
          f'目标={text(raw.get("target"))} control={raw.get("control","--")}']
    shape='--' if frame is None else f'{frame.rgb.shape[1]}x{frame.rgb.shape[0]}'
    rows.append(f'[目标框] decision={index} bbox_xyxy={text(raw.get("bbox_2d"))} '
                f'所选原图={direction} 尺寸={shape} 帧={getattr(frame,"frame_id","--")}')
    stage=raw.get('stage_progress')
    if isinstance(stage,dict):
        parsed=stage.get('parse_success')
        status='成功' if parsed is True else '失败' if parsed is False else '未提供'
        sid=stage.get('stage_id');total=stage.get('stage_total')
        current=f'第{sid+1}阶段(stage_id={sid})' if isinstance(sid,int) else '未知阶段'
        if (parsed is True and stage.get('current_stage') == 'COMPLETED'
                and stage.get('stage_completed') == total):
            current=f'全部阶段完成(终态stage_id={sid})'
        rows.append(f'[阶段进度] decision={index} 解析={status} '
                    f'已完成={stage.get("stage_completed")}/{total} 当前={current} '
                    f'内容={text(stage.get("current_stage"))} 完成ID={text(stage.get("completed_stage_ids"))}')
        if stage.get('parse_error'):
            rows.append(f'[阶段解析错误] {text(stage["parse_error"])}')
    else:
        rows.append(f'[阶段进度] decision={index} 服务器未提供阶段解析结果')
    for key,label in [('progress_analysis','模型进度说明'),('reasoning','模型理由')]:
        if raw.get(key):rows.append(f'[{label}] {text(raw[key])}')
    gate=raw.get('stop_gate')
    if isinstance(gate,dict):
        rows.append(f'[STOP Gate] 判定={gate.get("verdict")} completed={gate.get("completed")} '
                    f'缺失子任务={text(gate.get("missing_subgoal"))} 解析={gate.get("parse_success")}')
    verification=raw.get('failure_verification')
    if isinstance(verification,dict):
        rows.append(f'[Failure Verifier] 判定={verification.get("verdict")} '
                    f'需要恢复={verification.get("need_recovery")} '
                    f'依据={text(verification.get("evidence_summary"))}')
    phase7=raw.get('phase7') or {};audit=phase7.get('audit') or {}
    if audit:
        rows.append(f'[Semantic Audit] 状态={phase7.get("status")} candidate={audit.get("candidate")} '
                    f'moved={audit.get("moved")} sustained={audit.get("sustained")} '
                    f'候选类型={audit.get("candidate_type")}')
        if audit.get('contradiction_evidence'):
            rows.append(f'[Semantic Audit 矛盾证据] {text(audit["contradiction_evidence"])}')
    if raw.get('recovery'):
        recovery=raw['recovery']
        rows.append(f'[Recovery] 状态={recovery.get("status")} 解析={recovery.get("parse_success")} '
                    f'下一步={raw.get("next_action")}')
    return rows


def execution_lines(index, raw, state, accepted=False, projection=None, failure_reason=None):
    control=raw.get('control')
    if control in ('PREEMPT','SAFE_STOP'):
        outcome=f'{control}：保持停车，不执行上述模型运动动作'
    elif failure_reason:
        outcome=f'失败：{failure_reason}'
    elif (control == 'STOP_CONFIRMED' or raw.get('stop_phase') == 'STOP_CONFIRMED') and state == 'stopped':
        outcome='STOP已确认；本地已停车，等待会话结束确认'
    elif accepted:
        outcome='已接受导航；转向/停稳/规划通过后才可能行走'
    else:
        outcome='按原状态机处理；此行不表示任务成功'
    rows=[f'[执行判定] decision={index} {outcome}；当前状态={state}']
    if projection:
        rows.append(f'[目标投影] 像素={text(projection.get("pixel_uv"))} '
                    f'深度={number(projection.get("depth_m"),3)}m '
                    f'有效深度点={projection.get("valid_depth_count")} '
                    f'兜底={projection.get("used_forward_fallback",False)} '
                    f'转向后名义局部目标={text(projection.get("goal_after_turn_xy_m"))}m')
    return rows


def map_lines(payload):
    return [f'[地图窗口] decision={payload.get("decision_index")} window={payload.get("window_index")} '
            f'新增探索cell={payload.get("new_explored_cells")} 累计探索={payload.get("explored_cells")} '
            f'占用={payload.get("occupied_cells")} 可通行={payload.get("traversable_cells")} '
            f'融合次数={payload.get("update_count")} 分辨率={payload.get("resolution_m")}m '
            f'融合失败={payload.get("map_update_failures")} 来源=RGB-D探索地图']


def window_lines(request, response, duration_s):
    phase5=response.get('phase5') or {};candidate=phase5.get('candidate') or {}
    span=request.get('timestamp_end',0)-request.get('timestamp_start',0)
    return [f'[执行窗口] decision={request["decision_index"]} window={request["window_index"]} '
            f'实际时长={number(span)}s 位移={number(request.get("displacement_m"),3)}m '
            f'距局部目标={number(request.get("distance_to_local_goal_start"))}'
            f'→{number(request.get("distance_to_local_goal_end"))}m '
            f'上报={duration_s*1000:.0f}ms control={response.get("control")} '
            f'PhysicalMonitor={phase5.get("status","--")} '
            f'候选={candidate.get("candidate_type","无")} 原因={text(response.get("reason"))}']


def planner_lines(call_index, frame_id, goal, path, fear, duration_s):
    size=0 if path is None else len(path)
    return [f'[iPlanner规划] 调用={call_index} 帧={frame_id} 局部目标={text(goal)}m '
            f'路径点={size} fear={number(fear,3)} 耗时={duration_s*1000:.0f}ms '
            f'结果={"已返回路径" if size>1 else "无有效路径"}']
