# 2026-10-20：真机两阶段 VLN 完整链路

实验日期为10月20日，10月22日整理上传。指令：**向前走到黑色箱子前，然后向左转走到绿色瓶子前停下**。

本次用户手动启动真机运动。系统完成5次决策、4次局部动作上报、4个运动监控窗口、22次iPlanner调用，最终STOP Gate允许停止，会话结束为SUCCESS。实验验证正常任务链路；没有验证精确停靠、实际避障或故障恢复效果。

原始目录：`outputs/g1_real_unified_vln/g1_pelvis_check_20261020_182930_fb929806/run_20261020_182933_574664/`。
[证据目录](evidence/)保存原始响应、投影、规划、监控、收尾JSON、控制事件日志与5张目标图；[来源及SHA256](evidence_manifest.json)用于追溯。完整RGB-D采集仍在本机原始目录，本次未上传深度数组及所有全景原图。

## 链路和各层职责

```text
D415 RGB-D → 四段左转、停车采图 → Navigator选择方向/目标框
→ Stage Progress解析阶段 → Semantic Audit按策略审查
→ 真机选定方向转向、停车核验 → 深度目标投影
→ iPlanner重规划 → 原Pure Pursuit跟随
→ RGB-D探索地图更新 → Physical Monitor运动窗口上报
→ 动作完成上报 → 下一轮采图决策 → STOP Gate → 停车及结束会话
```

胸前D415通过笔记本USB读取，自己的TCP服务传输1280×720、15fps的RGB和对齐深度。SLAM的base_link按用户确认对应pelvis，标称离地0.80m；相机到pelvis结合胸部候选外参与腰关节正运动学。探索地图使用这套几何，局部目标投影仍有名义相机轴假设，不能把本次成功当作外参精度验证。仿真共享跟随算法保持原实现。

## 五次决策和执行

| 决策 | 模型动作/方向/目标 | 阶段进度 | 深度投影 | 动作首尾SLAM位移 |
|---|---|---|---|---|
| 0 | NAVIGATE / forward / 黑色箱子 | 0/2，第一阶段 | 2.590m，49个有效点 | 1.134m |
| 1 | NAVIGATE / forward / 黑色箱子 | 1/2，已进入瓶子阶段 | 框底无有效深度，使用原1.5m前进兜底 | 0.0065m |
| 2 | NAVIGATE / left / 绿色瓶子 | 1/2，第二阶段 | 1.933m，49个有效点 | 0.564m |
| 3 | NAVIGATE / forward / 绿色瓶子 | 1/2，第二阶段 | 1.304m，49个有效点 | 0.0019m |
| 4 | STOP / forward / 绿色瓶子 | 2/2，COMPLETED | 停止决策不执行新导航 | — |

位移是动作首尾SLAM位置差，不是轨迹弧长。决策2的位移还包含转向期间的身体位置变化。决策1、3截短后的局部目标距离分别约0.785m、0.858m，小于原1m局部到达容差，几乎未额外行走。局部动作COMPLETED不等于任务语义阶段完成。

决策2确实先左转再规划前进：[导航转向证据](evidence/decision_002_navigation_rotation.json)记录速度0.8rad/s、IMU累计75°提前请求停车、停车后等待0.5秒，最终IMU约81.08°、SLAM约85.54°，相对90°误差−4.46°，状态SETTLED_VERIFIED。它验证本次转向分支通过，不意味着每次精确90°。

目标框：

| 决策 | 原图bbox_xyxy | 目标图 |
|---|---|---|
| 0 | [457,384,611,549] | [黑色箱子](evidence/decision_000_target.png) |
| 1 | [228,515,632,720] | [仍选箱子且框触底](evidence/decision_001_target.png) |
| 2 | [571,314,607,505] | [左侧绿色瓶子](evidence/decision_002_target.png) |
| 3 | [426,377,495,485] | [前方绿色瓶子](evidence/decision_003_target.png) |
| 4 | [322,301,394,550] | [停止时绿色瓶子](evidence/decision_004_target.png) |

这些框对应各次模型实际输入的所选视图，不是运动中的目标跟踪。全景最初forward与转完后的实时画面可能有朝向差。

## Physical layer：实际判断了什么

Physical Monitor接收执行窗口的位移、地图进展等指标，记录物理执行异常候选；它不负责辨认瓶子或箱子。以下4次返回均为`MONITORING`、`candidate=null`、`CONTINUE`，动作完成的4次上报也均继续。没有触发PREEMPT、Failure Verifier、Recovery或Escape Evaluator。

| 决策/窗口 | 实际跨度 | 首尾位移 | 距局部目标起→止 | 新探索cell | 累计探索cell | 可通行cell |
|---|---|---|---|---|---|---|
| 0/0 | 1.205s | 0.110m | 2.100→2.039m | 45 | 15414 | 2262 |
| 0/1 | 1.104s | 0.406m | 2.039→1.657m | 370 | 15784 | 2301 |
| 0/2 | 1.211s | 0.495m | 1.657→1.144m | 69 | 15853 | 2301 |
| 2/0 | 1.122s | 0.065m | 1.323→1.315m | 321 | 23577 | 3295 |

新cell来自本地RGB-D探索地图，不是iPlanner检测到的物体数量。窗口新增合计805，退出时累计探索27356；全景采集及动作尾段也更新地图，所以两者不能直接相减求初始地图。噪声和定位漂移也可能增加cell。表中目标距离是执行证据，不代表所有指标都参与了远端候选触发规则。

窗口并非严格1秒：本次平均1.160s；窗口HTTP平均48ms，二者分别计时。iPlanner共22次调用，均值131ms、p95约156ms、最大170ms，包含客户端调用与适配耗时，不是纯GPU推理耗时。

`CONTINUE`只说明本次没有触发物理层干预，不能证明路径净空或避障可靠。近零fear也不是安全保证。原始证据见`evidence/g3_decision_*_motion_*.json`及对应request、`timing_summary.json`。

## Semantic layer：发现过一次目标与阶段不一致

| 审查 | 结果 | 证据解释 | 实际干预 |
|---|---|---|---|
| 决策1，Audit 1 | candidate=true，SEMANTIC_WANDERING，HIGH | 第一阶段已完成，当前阶段要左转找瓶子，但Navigator仍选择前方黑箱；审查指出瓶子在LEFT | moved=false、sustained=false、有效窗口0；没有PREEMPT |
| 决策3，Audit 2 | candidate=false，HIGH | 已完成左转，瓶子位于FORWARD，朝向与第二阶段一致 | 没有干预 |

两次均为`AUDIT_RECORDED`、解析成功。原文见[决策1响应](evidence/decision_001_response.json)、[决策3响应](evidence/decision_003_response.json)。

因此不能说Semantic层没有发现问题：它提出过语义游走候选，但没有确认持续异常或触发恢复。后续决策2由Navigator选择left/绿色瓶子；本次记录不足以证明这是审查直接导致的纠正。

`moved=false`是远端语义证据窗口中的门控标记，不能解读为机器人整个任务没有移动；动作记录明确显示1.134m及0.564m首尾位移。仅凭该标记不能推翻实际位移，也不能认定审查已经形成有效、持续的失败证据。

阶段解析5次均成功，进度为0/2→1/2→1/2→1/2→2/2。最终`stage_id=2`是两阶段之后的终态编号，不代表第三个任务；终端显示已改为“全部阶段完成”。Stage Progress记录后端为`local_navigator_shadow`，Semantic Audit和STOP Gate记录后端为`strong_api`。

## STOP Gate与收尾

[最终决策](evidence/decision_004_response.json)中Navigator请求STOP，Stage Progress为2/2，STOP Gate返回ALLOW、completed=true、无缺失子任务，理由包括箱子阶段、左转和接近瓶子均已完成。随后STOP_CONFIRMED，本地进入stopped，远端[会话结束](evidence/g3_session_ended.json)为SUCCESS，原因stop_confirmed。

[退出记录](evidence/run_exit.json)：CLEANUP_FINISHED、remote_end_status=CONFIRMED、cleanup_errors为空，退出前最后命令为[0,0,0]，最终位姿来源为新鲜SLAM读取。`physical_stop_confirmed=false`表明没有独立物理停车反馈确认，不能把软件请求成功当作实测停车延迟。本次是正常结束，没有再次测试Ctrl+C分支。

## 已验证与未验证

已验证：真实720p取图、五轮全景观察、两阶段模型决策、真实左转核验、iPlanner与原Pure Pursuit执行、连续探索地图更新、Physical上报、Semantic审查、STOP Gate及正常结束会话。300项接口离线测试于10月22日通过，使用替身，不产生真机运动。

仍需验证：障碍物净空与避障、精确停靠、深度缺失处理、长期物理异常检测、语义持续异常后恢复。尤其决策1使用了无有效深度的原兜底；本次成功不表示该兜底可靠。不能将本次没有触发的恢复分支记为实机验证通过。

代码与启动说明见[真机PELVIS入口README](../../../scripts/isaacsim_lavira_g3_interface_g1/README_G1_PELVIS_VLN.md)。运动命令仍由用户手动执行，上传及离线验证没有启动机器人。
