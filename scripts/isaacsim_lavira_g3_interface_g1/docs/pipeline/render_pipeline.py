"""Generate reviewable vector diagrams from the current robot/G3 interface.

Run: python render_pipeline.py
Artifacts: three SVG/PNG pages and one multipage PDF. No robot is contacted.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/g1_g3_pipeline_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({'font.family': font_name, 'svg.fonttype': 'path',
                     'pdf.fonttype': 42, 'axes.unicode_minus': False})
C = {'ink':'#142D43', 'muted':'#52687A', 'blue':'#2267AD', 'teal':'#148575',
     'purple':'#7154AD', 'orange':'#B96919', 'line':'#A8B8C7'}
FIGS = []

def page(title, subtitle, number):
    fig, ax = plt.subplots(figsize=(22, 15))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0,22), ylim=(0,15)); ax.axis('off')
    fig.patch.set_facecolor('#FFFFFF')
    ax.text(.6,14.5,title,fontsize=27,weight='bold',color=C['ink'],va='center')
    ax.text(.6,13.94,subtitle,fontsize=12.2,color=C['muted'],va='center')
    ax.plot([.6,21.4],[13.55,13.55],color='#DCE5ED',lw=1)
    ax.text(.6,.2,'G1 × LaViRA G3  |  基于本地代码核对 · 2026-09-20  |  真机接口 ≠ 已完成硬件验收',
            fontsize=10,color=C['muted'])
    ax.text(21.4,.2,f'{number} / 3',ha='right',fontsize=10,color=C['muted'])
    FIGS.append(fig)
    return ax

def box(ax,x,y,w,h,title,body,color='blue',fs=12):
    colors={'blue':'#F0F6FC','teal':'#EFF9F6','purple':'#F5F2FB','orange':'#FFF7EC'}
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.035,rounding_size=0.11',
                 linewidth=1.15,edgecolor=C[color],facecolor=colors[color],zorder=2))
    ax.text(x+.19,y+h-.20,title,fontsize=14,weight='bold',color=C[color],va='top',zorder=3)
    ax.text(x+.19,y+h-.58,body,fontsize=fs,color=C['ink'],va='top',linespacing=1.55,zorder=3)

def arrow(ax,a,b,color='blue',label=None,rad=0,dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,lw=1.65,
         color=C[color],connectionstyle=f'arc3,rad={rad}',linestyle='--' if dashed else '-',zorder=4))
    if label:
        x=(a[0]+b[0])/2; y=(a[1]+b[1])/2
        ax.text(x,y+.08,label,fontsize=10,color=C[color],ha='center',va='bottom',
                bbox=dict(facecolor='white',edgecolor='none',pad=1),zorder=5)

ax=page('机器人端整体 Pipeline：从指令到运动闭环',
        '阅读方向：中列自上而下；左列提供传感与位姿；右列通过 HTTP 返回高层决策和监督控制。',1)
ax.text(.65,13.1,'机器人端：感知与状态',fontsize=16,weight='bold',color=C['teal'])
ax.text(7.05,13.1,'机器人端：共享 Episode 状态机',fontsize=16,weight='bold',color=C['blue'])
ax.text(15.25,13.1,'远端：G3 服务（8765）',fontsize=16,weight='bold',color=C['purple'])
box(ax,7.0,11.35,6.0,1.25,'① 初始化与会话启动',
    '相机 / SLAM / 标定检查 → health → start_session\n提交 instruction；保存冻结阶段计划',fs=11.7)
box(ax,15.2,11.35,6.1,1.25,'Session + Stage Planner',
    '返回 ACTIVE、stage_plan_id、READY\nFrozen Stage Plan：本轮任务的有序子目标','purple',11.7)
arrow(ax,(13.05,12.22),(15.15,12.22),label='instruction / session_id')
arrow(ax,(15.15,11.62),(13.05,11.62),'purple',label='冻结计划')
box(ax,.65,9.25,5.5,2.25,'RGB-D 观测（真机计划用 D435i）',
    'RGB + depth_m + K + 采集时间\n单个前向相机配合机器人原地旋转\n依次采集前 / 左 / 后 / 右\n四视图有时间先后，并非同时拍摄','teal',12)
box(ax,7.0,9.15,6.0,1.65,'② 采集全景并请求高层决策',
    '发送四方向 RGB PNG + 历史图像/动作\n本轮 ID、图像尺寸、时间等元数据\n等待高层模型期间保持零速度',fs=11.7)
arrow(ax,(10,11.31),(10,10.85))
arrow(ax,(6.2,10.02),(6.95,10.02),'teal')
box(ax,15.2,9.15,6.1,1.65,'Navigator + Stage Progress（弱模型侧）',
    'Navigator：选择动作、方向、目标与 bbox\nStage Progress：独立估计子目标进度\n两者是不同 prompt / 独立调用','purple',11.7)
arrow(ax,(13.05,10.32),(15.15,10.32),label='decision：RGB + metadata')
box(ax,.65,6.95,5.5,1.65,'世界位姿：SLAM ROS 2 Odometry',
    '基座 x、y、yaw + frame_id\n跟随 / 旋转 / 地图 / 历史 / 上报共用\n位姿超时或坐标系名称变化 → 停车','teal',11.6)
box(ax,7.0,6.85,6.0,1.65,'③ 校验响应并分派动作',
    'NAVIGATE → 目标投影 → 转向 → 规划\nBACKTRACK → 历史 waypoint / 世界轨迹\nPREEMPT → 停车确认；STOP 按服务控制处理',fs=11.35)
arrow(ax,(10,9.11),(10,8.54))
arrow(ax,(6.2,7.65),(6.95,7.65),'teal')
box(ax,15.2,6.85,6.1,1.65,'条件触发的强模型监督',
    'Semantic Audit：按周期检查持续语义异常\nSTOP Gate：核验 STOP；Verifier：确认故障\nRecovery Planner / Escape：恢复与交还','purple',11.35)
arrow(ax,(18.25,9.11),(18.25,8.54),'purple')
arrow(ax,(15.15,7.25),(13.05,7.25),'purple',label='一次 decision 回复')
ax.text(14.1,6.93,'动作 + bbox + 阶段 + 控制',ha='center',fontsize=9.2,color=C['purple'])
arrow(ax,(3.4,6.91),(3.4,6.29),'teal')
box(ax,.65,4.15,5.5,2.1,'Map Progress：本轮探索证据',
    '深度 → 三维点 → 世界二维栅格（5 cm）\n累计 observed；高度筛选障碍并膨胀\n统计 explored / new_explored / traversable\n提供监督证据；不作为 iPlanner 输入','teal',11.4)
box(ax,7.0,4.35,6.0,1.85,'④ 本地目标与路径规划',
    'bbox 底边中心 + 对应深度 → 局部目标\n转向后新前视 RGB-D → iPlanner（8888）\n返回轨迹 + fear；路径末尾留安全距离\n默认 safe_distance = 0.5 m；fear 在客户端记录',fs=11.1)
arrow(ax,(10,6.81),(10,6.24))
box(ax,15.2,4.35,6.1,1.85,'执行报告处理与 Physical Monitor',
    '接收运动窗口 / 动作完成及 Map Progress\n检查位移、目标距离变化、探索增长等\n按条件验证故障、抢占、评估恢复结果\n返回 CONTINUE / PREEMPT / SAFE_STOP 等','purple',11.1)
box(ax,7.0,2.05,6.0,1.65,'⑤ 路径跟随 → 机器人运动',
    'Pure Pursuit：当前位置 + 轨迹 → vx、vy、wz\n默认 lookahead = 0.5 m；支持 KP 替代\nSLAM 位姿持续反馈，更新跟随与完成条件',fs=11.4)
arrow(ax,(10,4.31),(10,3.74))
box(ax,.65,1.0,5.5,2.45,'两个执行后端',
    '真机：LocoClient.Move(vx,vy,wz)\n导航目标频率 20 Hz；DDS 重发目标频率 50 Hz\n停止使用零速度 / StopMove\nIsaacSim：速度命令 → ONNX 步态/站立策略\n仿真世界位姿替代 SLAM','teal',10.9)
arrow(ax,(6.95,2.55),(6.2,2.55),'teal')
box(ax,7.0,.65,6.0,.9,'⑥ 执行证据与下一轮',
    '运动窗口上报 → 动作结束上报 → 更新历史 → 回到②',fs=10.9)
arrow(ax,(10,2.01),(10,1.59))
arrow(ax,(13.05,1.15),(14.35,1.15),label=None)
ax.plot([14.35,14.35],[1.15,4.85],color=C['blue'],lw=1.65)
arrow(ax,(14.35,4.85),(15.15,4.85),label=None)
ax.text(14.12,2.5,'执行上报',rotation=90,fontsize=11,color=C['blue'])
box(ax,15.2,1.0,6.1,2.65,'终止与成功条件',
    'STOP_CONFIRMED → 停车 → end_session(SUCCESS)\nSAFE_STOP / 本地异常 / 人工中断 → 停车\n按失败原因尽力 end_session(FAILURE)\n局部 COMPLETED 不等于语义任务成功\n正常闭环以 Session ENDED 为准','purple',11.1)
ax.text(.8,12.45,'机器人本地：深度、点云、地图、局部轨迹\nG3 远端：语言决策、监督、恢复与会话',
        fontsize=11.6,color=C['ink'],va='top')

ax=page('机器人 ↔ G3：完整通信时序与载荷',
        '蓝色箭头为机器人发送；紫色箭头为服务器回复。所有字段为当前接口摘要，完整字段见配套说明。',2)
for x,label,col in [(2.5,'机器人端 Episode / Client','blue'),(19.3,'G3 Server :8765','purple')]:
    ax.text(x,12.95,label,ha='center',fontsize=16,weight='bold',color=C[col])
    ax.plot([x,x],[.8,12.5],color=C['line'],lw=1.3,ls='--')
def msg(y,send,title,fields):
    a,b=((2.5,y),(19.3,y)) if send else ((19.3,y),(2.5,y))
    col='blue' if send else 'purple'
    arrow(ax,a,b,col)
    ax.text(10.9,y+.31,title,ha='center',fontsize=13.2,weight='bold',color=C[col])
    ax.text(10.9,y-.16,fields,ha='center',va='top',fontsize=11.1,color=C['ink'])
msg(12.0,True,'0  GET /health → 校验协议与阶段能力','检查 G3 / schema_version / execution、stage、stop、recovery 协议与强模型后端')
msg(11.0,True,'1  POST /v1/lavira/session/start','schema_version=2 · request_type · session_id · instruction（完整任务在这里提交一次）')
msg(10.0,False,'start 回复：ACTIVE + Frozen Stage Plan','stage_plan_id · stage_plan_status=READY · stage_total · subgoals · next_action')
msg(8.9,True,'2  POST /v1/lavira/decision 〔multipart〕','JSON 元数据 + 当前四方向 RGB PNG + 历史初始/选中方向 RGB；历史动作/目标/bbox/waypoint')
ax.text(10.9,8.27,'元数据：session_id、observation_id、decision_index、bundle_id、时间、图像尺寸及图片字段映射',
        ha='center',fontsize=10.5,color=C['muted'])
msg(7.7,False,'decision 回复：导航动作 + 阶段进度 + 条件监督结果','action=NAVIGATE/STOP/BACKTRACK · direction · target · bbox_2d / waypoint · reasoning / progress_analysis')
ax.text(10.9,7.06,'附带：stage_progress、stop_phase/control；按触发包含 phase5/6/7、failure_verification、recovery 等',
        ha='center',fontsize=10.5,color=C['muted'])
msg(6.45,True,'3  POST /v1/lavira/execution/report 〔motion_window〕','窗口 ID/时间 · 坐标系/epoch · 起终位姿(x,y,yaw) · 位移 · 起终目标距离 · local_planner_status')
ax.text(10.9,5.80,'map_progress = {resolution_m, explored_cells, new_explored_cells, traversable_cells}；默认约每 1 s 一个执行窗口',
        ha='center',fontsize=10.5,color=C['teal'])
msg(5.2,False,'执行控制回复：CONTINUE / PREEMPT / SAFE_STOP 等','机器人解析 control 与恢复状态；需要抢占时先停车，再回报 PREEMPTED')
msg(4.05,True,'4  POST /v1/lavira/execution/report 〔action_complete〕','status=COMPLETED/PREEMPTED/FAILED · reached_local_goal · planner_result · waypoint_id · 起终位姿与位移')
msg(3.05,False,'完成回复：继续导航 / 恢复规划 / Escape 结果与 Handback','继续时重新观察并请求下一轮 decision；Recovery 动作走同一套本地规划与执行链路')
msg(1.95,True,'5  POST /v1/lavira/session/end','session_id · status=SUCCESS/FAILURE · reason；成功仅由 STOP_CONFIRMED 路径触发')
msg(.95,False,'结束回复：status=ENDED','final_status · reason · stage_plan_id；本地保存 JSON、图像、地图快照与执行证据')

ax=page('机器人本地处理：几何、地图与部署边界',
        '同一帧 RGB-D 供三条用途：高层视觉、局部规划、探索地图。SLAM 提供位姿；不直接读取 SLAM PCD 地图。',3)
box(ax,.65,10.8,6.1,2.0,'A  高层视觉 → 目标投影',
    '四方向 RGB → G3；深度保留在机器人端\n收到 direction + bbox → 选择同方向 RGB-D\nbbox 底边中心邻域：深度过滤 → 第 30 百分位\n用 K 投影为理想转向后的机器人平面局部目标','blue',11.6)
box(ax,7.95,10.8,6.1,2.0,'B  新前视 RGB-D → iPlanner',
    '转向、等待稳定、再采一帧前视 RGB-D\nRGB 转 BGR；depth_m 转 uint16 毫米\n向本机 :8888 发送图像、深度、局部目标\n返回局部轨迹与 fear；内参用于 planner.reset','blue',11.6)
box(ax,15.25,10.8,6.1,2.0,'C  路径 → Pure Pursuit → 速度',
    '轨迹尾部留 0.5 m（可配置）→ 安全路径\nSLAM 当前位姿 → 更新局部目标与跟踪点\n默认 lookahead 0.5 m → vx、vy、wz\n到达容差 / passed-goal guard 等结束本段','blue',11.6)
arrow(ax,(6.8,11.8),(7.9,11.8))
arrow(ax,(14.1,11.8),(15.2,11.8))
ax.text(.65,10.05,'探索地图分支：由本地深度和位姿维护，输出统计作为 G3 监督证据',fontsize=17,weight='bold',color=C['teal'])
steps=[('① 深度采样 / 有效性','横纵每隔 8 像素采样\n有限值且 0.10–5.0 m'),
       ('② 三维点云','深度 + K → 相机点云\n外参 + SLAM → 世界点云'),
       ('③ 世界二维栅格','5 cm 格；射线路径加入 observed\n按格坐标去重，允许负坐标'),
       ('④ 障碍 / 连通统计','离地 0.10–1.60 m 标障碍\n膨胀 0.35 m，再算连通区域')]
for i,(title,body) in enumerate(steps):
    x=.65+i*5.35
    box(ax,x,7.85,4.7,1.65,title,body,'teal',11.4)
    if i<3: arrow(ax,(x+4.75,8.65),(x+5.3,8.65),'teal')
box(ax,.65,5.45,10.1,1.7,'Map Progress 的统计含义',
    'explored_cells：本轮累计观察格数；new_explored_cells：窗口新增格数\ntraversable_cells：与机器人连通、未被膨胀障碍阻挡的已观察格数\n完整栅格与点云留在本地；执行报告发送统计量和世界位姿','teal',12)
box(ax,11.3,5.45,10.05,1.7,'三个距离参数不能混为一谈',
    'safe_distance：截短轨迹尾部；lookahead：沿路径选择跟踪点\ngoal_tolerance：局部动作结束容差，实验常用 0.5 m 或 1.0 m\n局部动作完成 ≠ 到达语义目标 ≠ 强模型确认任务完成','orange',12)
box(ax,.65,2.35,10.1,2.4,'当前已实现 / 仿真已见证的链路',
    '机器人状态机、目标投影、规划、跟随、地图与 G3 协议已连接\n仿真日志中已有：早停拦截 → PREEMPT → Recovery → Handback\n也已有 STOP_CONFIRMED → end_session(SUCCESS) 的完整记录\n真机入口：SLAM ROS 2 Odometry + 外部 CameraBackend + Unitree SDK\n当前真机入口调用机载运动接口；不在该入口运行仿真 ONNX 步态','blue',11.8)
box(ax,11.3,2.35,10.05,2.4,'真机仍需完成的适配与测量',
    'D435i 工厂/驱动接入：对齐 RGB-D、米单位深度、匹配内参与时间戳\nSLAM 话题、基座定义、相机外参需实测；图像与位姿需时间对齐\n现位姿新鲜度使用接收时间；尚无按图像时间查询/插值位姿\n地图仅用 x/y/yaw + 固定高度，未补偿实时 roll/pitch/高度变化\n障碍单调累积；噪声点不会自动清除，需真机验证与改进','orange',11.8)
ax.text(.7,1.65,'数据边界',fontsize=14,weight='bold',color=C['ink'])
ax.text(2.4,1.65,'G3：RGB + 历史 + 位姿/地图统计   |   iPlanner：RGB-D + 局部目标   |   Unitree：速度指令',fontsize=12,color=C['ink'])
ax.text(.7,1.05,'参数说明：以上数字为当前代码默认值 / 已用实验值；实际运行以启动参数和真实标定为准。',fontsize=11.5,color=C['muted'])

names=['01_robot_g3_pipeline','02_message_sequence','03_local_geometry_and_readiness']
with PdfPages(OUT/'G1_G3_pipeline.pdf') as pdf:
    for fig,name in zip(FIGS,names):
        pdf.savefig(fig)
        fig.savefig(OUT/f'{name}.svg')
        fig.savefig(OUT/f'{name}.png',dpi=160)
        plt.close(fig)
print('Generated',OUT/'G1_G3_pipeline.pdf')
