# Isaac Sim LaViRA G3 接口适配（G1 + 本机 iPlanner）

本文档是 `isaacsim_lavira_g3_interface_g1` 的当前唯一运行说明。旧版
`isaacsim_lavira_iplanner_g1`、2026-08-11/13 的临时命令和早期单阶段接口结论不再作为本目录的
运行依据。

最后更新：2026-08-28。

## 1. 目标与职责边界

本项目负责让 Isaac Sim G1（以后是真机 G1）调用同学服务器上的 LaViRA G3，并在机器人端完成
视觉采集、深度投影、iPlanner 和实际运动。

```text
Isaac Sim / 真机 G1
  ├── 单前向 RGB-D 采集
  ├── SLAM / odometry / 世界位姿
  ├── bbox + depth 局部目标投影
  ├── 本机 iPlanner
  ├── G1 运动执行与安全停止
  ├── 1 秒 Motion Window
  └── 本地 Map Progress
             │
             │ HTTP（本机通过 SSH 隧道访问）
             ▼
同学的 LaViRA G3 服务
  ├── Session
  ├── Stage Planner
  ├── Navigator
  ├── Stage Progress
  ├── STOP Gate
  ├── Physical Monitor / Failure Verifier / PREEMPT
  ├── Recovery Planner / Escape Evaluator
  └── Semantic Auditor
```

模型服务器不负责相机、深度、SLAM、iPlanner 和底层控制；机器人端也不直接连接 Navigator 或强模型
的内部端口。机器人端只连接统一 LaViRA 服务。

### 1.1 后续适配原则

服务器侧模型流程、Prompt、Parser、输入输出、调用频率、counter/seed 和状态转换，以同学现有
LaViRA G3 代码及七阶段方案为准。机器人端能原样提供的内容不自行增删；Isaac/G1 客观无法直接复现
时，先记录原实现、差异和影响，再确认替代方案后修改。

当前确认保留的机器人端适配只有：单前向 RGB-D 旋转采集四方向图像、无边界 5 cm 稀疏 Episode
Map、连续 iPlanner 约 1 秒汇总为 Motion Window、本地诊断日志、幂等执行上报重试和安全失败保护。
这些适配不改变服务器模型的 Prompt、Parser 或决策逻辑。

## 2. 当前完成状态

### 2.1 已完成并实际验证

| 部分 | 当前状态 | 已验证内容 |
|---|---|---|
| G3 健康检查 | 完成 | b92/G3/schema-v2，`execution_protocol=phase3_map_progress_v1`、`map_progress_required=true` |
| Session 生命周期 | 完成 | `start_session`、重复 Session `409`、`end_session` |
| Stage Planner | 完成 | `start_session` 真实生成 Frozen Stage Plan，整个 Episode 保持同一 `stage_plan_id` |
| Navigator | 完成 | 不重复发送 instruction 也能从 Session 读取任务，返回 `NAVIGATE/STOP/BACKTRACK` 协议字段 |
| 单前向 RGB-D 全景 | 完成 | G1 原地连续左转，通过同一个 forward 相机采集 forward/left/behind/right；其他相机保留但不读取 |
| bbox + depth 投影 | 完成 | 使用所选方向图像的 bbox 底边中心深度生成 `x=forward,y=left` 局部目标 |
| iPlanner | 完成 | 本机 `127.0.0.1:8888` 规划、重规划和 Pure Pursuit 执行 |
| Isaac root odometry | 完成（Isaac） | 记录 decision pose、arrival pose 和世界路径 |
| 本地 BACKTRACK 执行器 | 阶段6受控完成 | stored-reverse 曾在 Isaac 实测，稳定Registry ID映射已自动化验证；远端Recovery真实触发后的多段实走尚未完成 |
| 阶段3 Motion Window | 完成 | 仅 `EXECUTING` 平移期间约1秒一次，真实 Isaac 连续窗口已被远端接受 |
| 稀疏 Map Progress | 完成（Isaac） | 四字段对象已接入远端；地图无固定边界，分辨率 5 cm |
| 阶段3严格执行上报 | 完成 | 成功、重复、乱序、三种失败结果和真实 Isaac 请求均已通过 |
| 阶段4 Stage Progress | 完成 | 真实Isaac持续返回有效`1/2`及`2/2 COMPLETED`；服务器`phase4_stage_progress_v2`已将失败语义字段归一化为空 |
| 阶段4 STOP Pipeline | 历史闭环完成；待回归 | 真实Isaac曾闭环`STOP_PENDING → STOP_CONFIRMED → end_session(SUCCESS)`；Stage Progress阻塞已修复，需用新Session重跑照片STOP |
| STOP独立强模型 | 完成 | `/health`和真实响应均为`model_backend=strong_api`；真实`ALLOW`含有效role call与4张图 |
| 决策等待控制 | 客户端完成 | 单相机实际路径的`/decision`在后台线程执行；Isaac/G1控制循环继续运行并保持locomotion零速度 |
| 阶段5 PREEMPT | PREMATURE真实闭环完成 | 真实Isaac已走通`PREMATURE_STOP → Strong Failure Verifier → PREEMPT → PREEMPTED ack`；在线Physical/Semantic候选直接PREEMPT仍待实测 |
| 阶段6 Recovery | 失败安全闭环完成 | 真实Isaac已通过Recovery NAVIGATE、Strong Escape拒绝假成功、多次重试、预算耗尽SAFE_STOP及`end_session(FAILURE)`；BACKTRACK实走和成功Handback待验证 |
| 阶段7 Semantic | 基础真实调用完成 | Strong Semantic Audit在真实Isaac中能返回正常结果并识别`SEMANTIC_WANDERING/HIGH`；`sustained=true → PREEMPT`尚未触发 |

### 2.2 尚未完成，不能提前宣称成功

| 部分 | 尚缺内容 |
|---|---|
| 阶段5～7真实Isaac | PREMATURE候选、Strong Failure Verifier、PREEMPT、Recovery NAVIGATE、Strong Escape失败、重试和SAFE_STOP已通过；Physical/Semantic在线PREEMPT与成功Handback仍待验证 |
| Recovery BACKTRACK实走 | 稳定Registry ID与stored-reverse映射已受控验证；仍需Isaac实际走完多段路径 |
| SAFE_STOP真实链路 | 已在真实Isaac中走通`Recovery预算耗尽 → control=SAFE_STOP → 零速度站立 → end_session(FAILURE/recovery_safe_stop)` |
| 真机 G1 | DDS、D435、SLAM 位姿和在线 Map Progress 尚未端到端联调 |
| Map traversable | Isaac 阶段按当前 RGB-D 稀疏地图结果作为可靠输入；真机迁移时重新校准和验证 |

当前阶段的准确表述是：**阶段1至阶段4已完成主要真实Isaac链路；阶段5至阶段7服务器已部署，
真实Isaac已走通PREMATURE_STOP → Strong Failure Verifier → PREEMPT → Recovery NAVIGATE →
Strong Escape拒绝假成功 → 重试 → SAFE_STOP → Session安全失败结束。当前112项客户端自动化
测试通过。尚待真实验证的是修复后的Physical candidate完整PREEMPT、Recovery BACKTRACK、
Escape成功Handback、真实Isaac `STOP_CONFIRMED → SUCCESS`最终闭环以及真机G1。**

### 2.3 阶段3最终验收记录（2026-08-26）

真实 Isaac Session：

```text
session_id: scene200_map_v1_20260826_132311_12501
stage_plan_id: sha256:2a18df401b6c960fe4aefe6411070b0dd0f9783a2d1cebd12f3f5f059136c731
```

本地证据目录：

```text
outputs/isaacsim_lavira_g3_interface_g1/scene200_map_v1/
  scene200_map_v1_20260826_132311_12501/run_20260826_132326_424941/
```

验收结果：

- decision 0 的 `window_index=0...12` 共 13 条真实 Motion Window 连续被远端接受；
- 每条窗口响应均为 `CONTINUE/execution_window_recorded`；
- w0：`explored=13590`、`new=397`、`traversable=6753`；
- w12：`explored=15339`、`new=0`、`traversable=29`；
- `action_complete` 为 `COMPLETED/REACHED`，总位移 `3.0964654970393903 m`，`waypoint_id=0`；
- 重发真实 `action_complete` 返回原结果，没有重复记录；
- 乱序 `w999` 被拒绝，没有新增记录；
- 独立 Session `phase3_failure_paths_20260826_b` 的 `FAILED/TIMEOUT`、
  `FAILED/PLANNING_FAILED`、`FAILED/EXECUTION_FAILED` 均返回
  `CONTINUE/execution_result_recorded` 并被服务器保存；
- 服务器阶段3测试 21 项通过；本目录自动化测试 73 项通过。

服务器没有公开 `execution_history`/Waypoint Registry 查询接口；不为此扩大阶段3公开协议。
Waypoint坐标的实际取回和 BACKTRACK 端到端执行留到阶段6验证。

### 2.4 阶段4最终验收记录（2026-08-27）

受控真实 Isaac Session：

```text
session_id: phase4_strong_null_20260827_164006_10033
stage_plan_id: sha256:a91038566f2ec5ac56cb68f2e5380a4066c1efe2fab7a44464a4281f63df09e2
```

本地证据目录：

```text
outputs/isaacsim_lavira_g3_interface_g1/phase4_strong_null/
  phase4_strong_null_20260827_164006_10033/
  run_20260827_164021_455538/
```

本次instruction明确说明机器人已经位于目标处并要求面对前方黑色风景照片停止，因此用于稳定触发
STOP协议分支；它证明阶段4接口闭环，不作为正常长距离导航成功率证据。

真实执行顺序：

```text
start_session → Frozen Stage Plan(1 stage)
→ Isaac单前向RGB-D旋转采集第一组四方向图像
→ Navigator STOP
→ Stage Progress decision=0：1/1 COMPLETED
→ STOP_PENDING（强模型尚未调用：role_call_index=null、image_count=0）
→ 不调用iPlanner，不发送motion_window/action_complete
→ Isaac重新采集第二组四方向图像
→ Navigator STOP
→ Stage Progress decision=1：1/1 COMPLETED
→ strong_api STOP Gate：verdict=ALLOW、role_call_index=3、image_count=4
→ STOP_CONFIRMED / END_SESSION_SUCCESS
→ end_session：final_status=SUCCESS、reason=stop_confirmed
```

第二次真实响应中的关键字段：

```json
{
  "stop_gate": {
    "verdict": "ALLOW",
    "completed": true,
    "missing_subgoal": null,
    "parse_success": true,
    "role_call_index": 3,
    "image_count": 4,
    "model_backend": "strong_api",
    "stop_phase": "STOP_CONFIRMED"
  },
  "control": "STOP_CONFIRMED",
  "next_action": "END_SESSION_SUCCESS"
}
```

最终本地状态为：

```text
finished: state=stopped history=0 failure=None
session ENDED: final_status=SUCCESS reason='stop_confirmed'
```

其中`history=0`是正确结果：STOP受控测试没有执行NAVIGATE，因此没有生成导航Waypoint。今天同时根据
真实服务器响应补齐两项客户端兼容：未调用强模型的STOP_PENDING允许`role_call_index=null`和
`image_count=0`；strong API的ALLOW使用`missing_subgoal=null`，并兼容早期空字符串表示。

### 2.5 阶段5/6真实Isaac部分验收与SAFE_STOP阻塞（2026-08-28）

真实 Isaac Session：

```text
session_id: phase567_recovery_retest_20260828_140432_4076
stage_plan_id: sha256:2a18df401b6c960fe4aefe6411070b0dd0f9783a2d1cebd12f3f5f059136c731
```

本地证据目录：

```text
outputs/isaacsim_lavira_g3_interface_g1/phase567_recovery_retest/
  phase567_recovery_retest_20260828_140432_4076/
  run_20260828_140448_143128/
```

本轮已真实验证的链路：

```text
Navigator NAVIGATE
→ decision 0连续上报window 0...59
→ iPlanner执行60秒后TIMEOUT
→ action_complete=FAILED/TIMEOUT
→ Failure Monitor生成action_complete_failed候选
→ Strong Failure Verifier：FAILURE、need_recovery=true、parse_success=true
→ 服务器内部确认并完成原子Preemption
→ control=CONTINUE、next_action=REQUEST_RECOVERY_DECISION
→ Recovery Planner：NAVIGATE/left/open doorway、model_backend=strong_api
→ 客户端bbox + depth + iPlanner执行Recovery
→ action_complete=COMPLETED/REACHED
→ Strong Escape Evaluator：recovery_escape_not_proven、semantic_alignment=NOT_ALIGNED
→ next_action=REQUEST_RECOVERY_DECISION
```

关键真实数据：

- decision 0总首尾位移为`4.182465368290126 m`，局部动作按`local_action_timeout_s=60`结束；
- 后段窗口位移约为`0.003...0.027 m`，局部目标距离约从`1.46 m`恶化至`1.51 m`；
- 所有60条在线Motion Window均为`MONITORING/candidate=null/CONTINUE`，所以本轮不是在线Physical
  Monitor提前PREEMPT，而是TIMEOUT兜底触发；
- TIMEOUT的Failure Verifier真实使用`model_backend=strong_api`并返回`verdict=FAILURE`；
- Recovery decision 1真实使用`recovery_planner/strong_api`；
- Recovery投影目标约为`[1.570,-0.837] m`；同一次iPlanner结果包含`fear≈0.99995`和一条短轨迹，
  再经固定`safe_distance_m=0.5`截短后安全目标约为`[0.163,0.201] m`。客户端没有使用fear计算
  截断长度；`COMPLETED`只表示到达短局部轨迹终点，不表示Escape成功；
- Strong Escape Evaluator正确拒绝假成功并要求继续Recovery。

关键证据文件：

```text
g3_decision_000_action_complete.json  # TIMEOUT、Verifier、Preemption和Recovery请求
decision_001_response.json            # Strong Recovery NAVIGATE
decision_001_plan.json                # 同时记录fear、原轨迹及固定安全距离截短结果
g3_decision_001_action_complete.json  # Strong Escape失败及继续Recovery
```

随后decision 2在客户端普通导航响应解析阶段失败：

```text
STOP response.direction must be one of ('forward', 'left', 'behind', 'right')
```

客户端在返回raw response之前先调用旧`NavigationDecisionResponse.from_dict()`，所以没有生成
`decision_002_response.json`；服务器当时也没有落盘响应体。根据服务器代码，最可能是Recovery出口
返回了`action=STOP、direction=null、control=SAFE_STOP、action_source=RECOVERY`，但这必须由服务器
受控测试保存完整JSON后才能作为事实。当前禁止全局放宽普通STOP方向校验；若受控JSON确认，应只在
G3适配层优先识别`control=SAFE_STOP`，再执行零速度、清轨迹、稳定站立和
`end_session(FAILURE/recovery_safe_stop)`。

2026-08-31处理结果：服务器受控测试确认上述响应确为Recovery SAFE_STOP，并部署
`phase6_recovery_control_v1`，补齐Session/RecoveryState状态、缓存、幂等和trace。客户端现已在旧
`NavigationDecisionResponse`之前识别该控制响应，并复用已有`_enter_safe_stop()`；普通Navigator
STOP的方向、目标和bbox校验保持不变。

### 2.6 普通照片STOP回归与Stage Progress阻塞（2026-08-28）

真实 Isaac Session：

```text
session_id: photo_stop_retest_20260828_142643_11007
stage_plan_id: sha256:a6bc1397e23d0f2bccc5b2f1424dfe142228e71e833019e8a49099dd4f995bea
```

本地证据目录：

```text
outputs/isaacsim_lavira_g3_interface_g1/photo_stop_retest/
  photo_stop_retest_20260828_142643_11007/
  run_20260828_142658_884658/
```

decision 0真实返回普通Navigator STOP：

```text
action=STOP
direction=forward
target=dark landscape photograph
Stage Progress=1/1、current_stage=COMPLETED、parse_success=true
STOP_PENDING、control=CONTINUE、next_action=REQUEST_DECISION
```

这证明普通STOP的`direction/target/bbox`字段仍然正常，不能为了Recovery SAFE_STOP而全局允许
`STOP.direction=null`。

decision 1的Navigator仍返回合法`STOP/forward`，但Stage Progress返回了矛盾组合：

```text
parse_success=false
parse_error=current_stage must be COMPLETED after the final stage
```

同时`stage_plan_id/stage_total/stage_completed/stage_id/current_stage/final_target_visible`仍非null，
`completed_stage_ids/evidence_of_completion`仍非空。客户端按冻结Schema安全拒绝：

```text
Failed Stage Progress must use null stage fields.
```

服务器应二选一归一化：解析失败时所有stage语义字段为null且两个列表为空；或把最终阶段合法归一化为
`current_stage=COMPLETED、parse_success=true、parse_error=null`。当前响应两者都不满足。由于第二个
Stage Progress无效，Strong STOP Gate没有实际调用，只返回`UNCERTAIN/valid frozen-plan Stage Progress
is required/STOP_PENDING`。Session已正常`end_session(FAILURE)`，无需手动清理。

完整原始响应已保存：

```text
decision_000_response.json  # 合法STOP_PENDING
decision_001_response.json  # Stage Progress失败Schema矛盾
```

2026-08-31处理结果：服务器已部署`phase4_stage_progress_v2`。解析失败时对外语义字段统一为null/空
列表，原始模型输出仅保留在trace；失败记录不覆盖上一次有效进度，也不进入STOP Gate有效证据窗口。
客户端原有失败隔离校验可直接接受该结果。

### 2.7 2026-08-31当前冻结点

服务器与客户端协议阻塞均已完成受控修复：

1. 服务器已生成并落盘完整Recovery SAFE_STOP响应；
2. `phase6_recovery_control_v1`已冻结`control=SAFE_STOP`的终止控制；
3. `phase4_stage_progress_v2`已修正失败字段归一化和有效窗口隔离；
4. 客户端已优先解析SAFE_STOP，且没有放宽普通STOP；
5. 新目录99项自动化测试通过。

下一步使用全新的Session ID分别回归照片STOP和Recovery SAFE_STOP。仍不修改原Habitat、Navigator、
bbox投影、iPlanner、普通STOP和运动控制协议。

### 2.8 双画任务完整真实Isaac记录（2026-08-31）

本节只记录落盘JSON能够证明的事实，不把接口成功等同于语义任务成功。

真实Session与证据目录：

```text
session_id: paintings_full_g3_20260831_121007_17925
stage_plan_id: sha256:1daa4cc2af3727bc09e0dd906ea2ad58e557643eaa7ec82870abd86cadbe79c3

outputs/isaacsim_lavira_g3_interface_g1/paintings_full_g3/
  paintings_full_g3_20260831_121007_17925/
  run_20260831_121024_772949/
```

任务为先接近正前方的暗色风景照片，再接近右侧相邻墙上的彩虹画并停止，且不能进入门洞。
Stage Planner真实生成并冻结了两个正确子目标：

```text
0. Move straight toward the dark landscape photograph on the far wall.
1. Move toward the rainbow-colored painting on the adjacent wall to the right and stop near it.
```

最终落盘规模：

```text
14个decision response（decision 0...13）
84个motion_window
7个action_complete
6次Strong STOP Gate PREMATURE判定
4次Strong Semantic Audit
1次普通Navigator BACKTRACK真实执行
0次Physical Monitor candidate
0次PREEMPT
0次Failure Verifier / Recovery Planner / Escape Evaluator
```

完整高层过程：

| decision | Navigator输出 | 本地执行/服务器结果 |
| --- | --- | --- |
| 0 | `NAVIGATE/forward/dark landscape photograph` | 2个窗口，动作首尾位移`0.569m`，`COMPLETED/REACHED` |
| 1 | `NAVIGATE/behind/dark landscape photograph` | 5个窗口，位移`1.369m`，完成；Strong Semantic Audit已运行，但`moved=false、sustained=false` |
| 2 | `NAVIGATE/forward/dark landscape photograph` | 46个窗口，位移`3.561m`，完成；局部目标距离先降后升，存在越过目标现象 |
| 3 | `NAVIGATE/forward/dark landscape photograph` | 投影深度约`0.10m`，按当时`0.5m` tolerance近乎原地完成；Semantic Audit判断`SEMANTIC_WANDERING/HIGH`，但仅一条持续证据 |
| 4～6 | 三次普通`STOP/forward/dark landscape photograph` | Strong STOP Gate三次均判定`PREMATURE`，返回`CONTINUE + REQUEST_DECISION` |
| 7 | 普通Navigator `BACKTRACK waypoint=2` | 本地stored-reverse真实回退，10个窗口，首尾位移`3.121m`，完成；这不是Recovery Planner的BACKTRACK |
| 8 | `NAVIGATE/forward/dark landscape photograph` | 3个窗口，位移`0.948m`，完成 |
| 9～11 | 三次普通`STOP/forward/dark landscape photograph` | Strong STOP Gate继续判定`PREMATURE`，没有错误结束任务 |
| 12 | `NAVIGATE/left/dark landscape photograph` | 安全轨迹终点约`0.241m`，按当时`0.5m` tolerance短距离完成，位移`0.108m` |
| 13 | `NAVIGATE/left/dark landscape photograph` | 18个窗口后测试停止；目标距离最低约`0.555m`后回升至`1.040m`，未生成action_complete |

本轮已真实验证成功的功能：

1. `start_session`、Frozen Stage Plan和两阶段子目标生成正确；
2. 单前向RGB-D旋转采集四方向图像以及后台异步`/decision`正常工作；
3. Navigator的`NAVIGATE、STOP、BACKTRACK`三种动作均被客户端正确解析；
4. bbox+depth投影、iPlanner规划、连续控制、约1秒Motion Window及四字段Map Progress持续上报；
5. 所有已完成动作的`action_complete`均被服务器接受并返回`CONTINUE`；
6. Stage Progress在decision 0～13全部`parse_success=true`，始终正确报告`1/2`和当前彩虹画子目标；
7. Strong STOP Gate真实运行6次，均识别出只完成第一阶段，正确拒绝过早STOP；
8. Strong Semantic Auditor真实运行4次，多次识别`SEMANTIC_WANDERING`，其中decision 3明确指出
   彩虹画在LEFT视图中，而Navigator仍追踪暗色照片；
9. 普通Navigator返回的稳定Waypoint索引能够驱动本地stored-reverse BACKTRACK真实执行；
10. 本轮没有出现Stage Progress v2字段矛盾、普通STOP解析错误或Recovery SAFE_STOP误解析。

本轮没有验证成功、不能宣称完成的功能：

1. 机器人没有完成第二个彩虹画子目标，Session仍为`ACTIVE`，目录中没有`g3_session_ended.json`；
2. Physical Monitor的84个窗口全部为`MONITORING/candidate=null/CONTINUE`，没有在线PREEMPT；
3. Semantic Audit虽然发现语义游走，但`sustained`始终为false，没有进入Failure Verifier；
4. 没有触发Failure Verifier、Recovery Planner、Recovery BACKTRACK、Escape Evaluator、Navigator
   Handback或Recovery SAFE_STOP；
5. decision 7的BACKTRACK来自普通Navigator，不可作为Recovery Planner链路的验收证据；
6. 本轮没有得到`STOP_CONFIRMED`，所以也没有验证成功结束Session。

本轮暴露的主要问题：

- Stage Progress已经切换到第二阶段，但Navigator持续把暗色照片作为目标，形成
  `追暗色照片 → PREMATURE_STOP → 再次追暗色照片`的循环；
- Semantic Auditor能看出彩虹画和错误目标，但现有持续证据窗口没有形成`sustained=true`；
- decision 2和13都出现接近局部目标后距离重新增大。启用Isaac odometry时当前跟踪器没有“已越过
  目标”保护，靠旧`0.5m` tolerance没有及时结束；
- Physical Monitor按整个高层动作累计运动量判断。机器人先移动数米后再卡住时，累计
  `motion_m`已经远大于低运动阈值，因此后段低位移窗口没有形成候选；
- Map Progress继续记录少量新格也会削弱“无进展”证据；Map Progress记录成功不代表iPlanner会
  使用该地图避障。

参数说明：上述运行发生在`goal_tolerance_m=0.5`时期。随后测试`1.0m`；对应iPlanner结果虽
同时记录了高fear，但客户端只按固定`safe_distance_m=0.5`截短，所得安全轨迹终点约为
`0.965m/0.991m`，导致机器人仅移动`0.021m/0.002m`就被判为
`COMPLETED`。根据当前联调选择，正式默认值现改为`1.0m`，并保留odometry越过目标保护；
这会更早结束局部动作，上述短安全轨迹近乎原地完成的风险仍需在测试时观察。服务器Physical Monitor
和Recovery逻辑没有因此改变。

### 2.9 PREMATURE_STOP→Recovery→SAFE_STOP真实Isaac验收（2026-09-01）

真实Session和证据目录：

```text
session_id: terminal_stage_recovery_20260901_134756_15160
stage_plan_id: sha256:4d79b01bb65de0ecb72f21bd879e7e9bee678866824299a5107087ce082fa54d

outputs/isaacsim_lavira_g3_interface_g1/terminal_stage_recovery/
  terminal_stage_recovery_20260901_134756_15160/
  run_20260901_134812_495030/
```

任务为先接近暗色风景照片，再接近右侧彩虹画并停止。本轮真实走通：

```text
Navigator错误追踪第一阶段目标
→ Navigator STOP
→ Stage Progress=1/2
→ Strong STOP Gate=PREMATURE
→ Candidate Arbiter选中premature_stop候选
→ Strong Failure Verifier=FAILURE, need_recovery=true
→ control=PREEMPT, next_action=ACTION_COMPLETE_PREEMPTED
→ 客户端零速度、稳定站立并上报action_complete=PREEMPTED
→ preemption_acknowledged
→ Strong Recovery Planner连续生成3个Recovery NAVIGATE
→ Strong Escape Evaluator拒绝3次局部假成功
→ Recovery预算耗尽
→ control=SAFE_STOP
→ end_session(FAILURE, recovery_safe_stop)
```

关键事实：

- decision 4的STOP Gate和Failure Verifier均为`model_backend=strong_api`、`parse_success=true`；
- 客户端没有执行错误STOP动作，而是先完成原子PREEMPT确认；
- decision 5～7均为`action_source=RECOVERY`，Recovery Planner使用强模型；
- Escape Evaluator使用真实位移、Stage、Map和语义证据，没有把客户端的
  `COMPLETED/REACHED`直接当成Escape成功；
- decision 7服务器返回`SAFE_STOP`，客户端正确结束Session，最终失败是受控安全失败，
  不是协议断链或异常崩溃。

关键证据文件：

```text
decision_004_response.json                # PREMATURE + Strong Verifier + PREEMPT
g3_decision_004_action_complete.json      # PREEMPTED ack与原子抢占
decision_005_response.json                # 第一次Strong Recovery NAVIGATE
g3_decision_005_action_complete.json      # 第一次Escape未证明
g3_decision_006_action_complete.json      # 第二次Escape未证明
g3_decision_007_action_complete.json      # 预算耗尽并返回SAFE_STOP
g3_session_ended.json                     # FAILURE/recovery_safe_stop
```

本轮没有验证：

- Recovery不是由Physical Monitor触发，而是由PREMATURE_STOP触发；
- 修复后的`stage_id=stage_total`终态Physical candidate尚未在真实Isaac中再次触发；
- Recovery Planner未返回BACKTRACK；
- Escape没有成功，所以没有Navigator Handback；
- 未得到`STOP_CONFIRMED → end_session(SUCCESS)`的本轮成功结束。

本轮同时验证了`goal_tolerance_m=1.0`的副作用。Recovery decision 5～7的iPlanner
安全目标距离分别约为`0.289m / 0.176m / 0.476m`，均小于`1.0m`，因此客户端几乎
原地返回`COMPLETED`。Escape Evaluator正确拒绝了这些假成功，但Recovery也因缺乏真实
位移最终进入SAFE_STOP。当前依用户选择保持`1.0m`，每次实验必须在结论中标注这一
执行层影响。

#### 可复用的真实Isaac回归命令

终端1保持SSH隧道，终端2使用`run_iplanner_local.sh`启动iPlanner。终端3每次生成全新
Session ID后执行：

```bash
conda activate isaacsim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
cd /home/yile/projects/unitree-g1-isaaclab-project

SESSION_ID="terminal_stage_recovery_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
echo "Using new Session ID: $SESSION_ID"

python scripts/isaacsim_lavira_g3_interface_g1/run_isaacsim.py \
  --max_steps -1 \
  --scene_usd /home/yile/scene/House/scene_200/mujoco/usd/scene_scene_200.usda \
  --spawn 4.6 3.5 0.8 \
  --yaw -0.72 \
  --device cuda:0 \
  --real-time \
  --no-four_rgbd_set_viewport \
  --instruction "Stay inside this exhibition room. First move straight toward the dark landscape photograph directly ahead on the far wall. Then move toward the rainbow-colored painting on the adjacent wall to the right and stop near it. Do not enter any doorway." \
  --lavira_session_id "$SESSION_ID" \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 120 \
  --g3_session_timeout_s 180 \
  --g3_motion_window_s 1.0 \
  --iplanner_url "http://127.0.0.1:8888" \
  --iplanner_timeout_s 5 \
  --local_use_isaac_odometry \
  --local_map_progress \
  --local_map_resolution_m 0.05 \
  --local_map_depth_stride 8 \
  --local_map_robot_radius_m 0.35 \
  --local_goal_tolerance_m 1.0 \
  --local_blind_yaw_radius_m 1.5 \
  --local_max_decisions 0 \
  --local_action_timeout_s 60 \
  --local_output_dir outputs/isaacsim_lavira_g3_interface_g1/terminal_stage_recovery
```

若要更换对象并测试跨房间任务，仅替换instruction和输出目录：

```text
instruction: Leave the exhibition room through the open doorway. After entering the next room, move toward the sofa and stop near it.
output: outputs/isaacsim_lavira_g3_interface_g1/door_sofa_g3
```

## 3. 当前服务器与 SSH 接口

### 3.1 地址关系

```text
同学服务器内部服务：127.0.0.1:8765
同学服务器 SSH：wangchu@131.159.60.188
本机 SSH 隧道：127.0.0.1:18765 -> 远端 127.0.0.1:8765
本机 iPlanner：127.0.0.1:8888
```

远端 `8765` 才是 LaViRA 服务；本机 `18765` 只是 SSH 进程创建的临时转发端口。关闭 SSH 终端、
断网或重启机器后，隧道都会消失。

### 3.2 终端 1：启动 SSH 隧道

```bash
ssh -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -N \
  -L 18765:127.0.0.1:8765 \
  wangchu@131.159.60.188
```

输入 SSH 密码后没有新输出是正常的。这个终端必须保持打开。

如果出现 `Address already in use`，先检查旧隧道：

```bash
ss -ltnp | grep ':18765'
```

### 3.3 每次启动 Isaac 前必须检查健康接口

```bash
curl -sS --max-time 10 \
  http://127.0.0.1:18765/health \
  | python -m json.tool
```

期望至少包含：

```json
{
  "status": "ok",
  "schema_version": 2,
  "framework": "G3",
  "commit": "b92abb3",
  "audit_interval": 2,
  "execution_protocol": "phase3_map_progress_v1",
  "map_progress_required": true,
  "stage_progress_protocol": "phase4_stage_progress_v2",
  "stop_gate_protocol": "phase4_stop_gate_v1",
  "recovery_control_protocol": "phase6_recovery_control_v1",
  "phase4": {
    "status": "wired"
  },
  "phase5": {
    "status": "wired",
    "model_backend": "strong_api"
  },
  "phase6": {
    "status": "wired",
    "escape_confirmation_policy": "single_valid_positive"
  },
  "phase7": {
    "status": "wired",
    "audit_interval_decisions": 2
  }
}
```

客户端严格检查阶段3协议、阶段4的Stage Progress/STOP Gate，以及顶层阶段5～7的启用状态、
`strong_api` backend、稳定Waypoint Registry和一次Escape确认策略。`phase4`中的历史false字段保持
兼容但不代表顶层阶段5～7未启用。
Stage Planner 是否可用由随后真实
`start_session` 返回的 `stage_plan_status=READY` 和 Frozen Stage Plan 共同验证，
不再依赖旧版 `/health.stage_planner` 字段。

健康检查失败时不要启动 Isaac，先恢复 SSH 隧道或让同学检查远端容器
`lavira-end2end-b92` 和端口 `8765`。

## 4. Session ID 规则（每次测试必须阅读）

### 4.1 每一轮必须使用全新的 Session ID

服务器会保护活动 Session。复用仍为 ACTIVE 的 ID 会返回：

```text
HTTP 409 SESSION_ALREADY_ACTIVE
```

即使旧 Session 已 ENDED，也不要复用。推荐每次完整执行下面这一行：

```bash
SESSION_ID="scene200_doorway_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
echo "$SESSION_ID"
```

不能只重复执行 Python 命令而不重新生成变量，否则 shell 会继续使用旧 ID。正式保存实验时，也可以
手工填写一个从未使用过的固定 ID，例如：

```text
scene200_doorway_20260825_008
```

### 4.2 正常结束与异常清理

正常模型 STOP、执行失败或正常 `Ctrl+C` 会尽量调用 `end_session`。日志必须出现：

```text
[LOCAL-VLN G3] session ENDED
```

如果强制 `kill -9`、网络中断，或没有生成 `g3_session_ended.json`，需要手动结束旧 Session：

```bash
curl -sS -i --max-time 30 \
  -X POST \
  http://127.0.0.1:18765/v1/lavira/session/end \
  -H "Content-Type: application/json" \
  --data-raw '{
    "schema_version": 2,
    "request_type": "end_session",
    "session_id": "<OLD_SESSION_ID>",
    "status": "FAILURE",
    "reason": "manual_restart_for_new_test"
  }'
```

- `200` 和 `status=ENDED`：清理成功。
- `404 SESSION_NOT_FOUND`：服务内已经没有该 Session，可以继续。
- 清理完成后仍必须使用新的 Session ID。

删除本地输出目录不会清除服务器 Session。

## 5. 标准运行顺序

### 5.1 终端 1：SSH 隧道

使用第 3.2 节命令，并保持运行。

### 5.2 终端 2：启动本机 iPlanner

```bash
conda activate isaacsim
cd /home/yile/projects/unitree-g1-isaaclab-project

bash scripts/isaacsim_lavira_g3_interface_g1/run_iplanner_local.sh
```

iPlanner 使用：

```text
代码：/home/yile/projects/uni-lavira-code/real-world-code/unitree_g1/iplanner
checkpoint：scripts/isaacsim_lavira_g3_interface_g1/checkpoints/iplanner.pth
地址：http://127.0.0.1:8888
```

### 5.3 终端 3：scene_200 无限轮次标准命令

先重新生成 Session ID，再运行整个命令：

```bash
conda activate isaacsim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

cd /home/yile/projects/unitree-g1-isaaclab-project

SESSION_ID="scene200_doorway_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
echo "Using new Session ID: $SESSION_ID"

python scripts/isaacsim_lavira_g3_interface_g1/run_isaacsim.py \
  --max_steps -1 \
  --scene_usd /home/yile/scene/House/scene_200/mujoco/usd/scene_scene_200.usda \
  --spawn 4.6 3.5 0.8 \
  --yaw -0.72 \
  --device cuda:0 \
  --real-time \
  --no-four_rgbd_set_viewport \
  --instruction "Go through the open doorway directly ahead and continue into the next room. Do not stop before crossing the doorway." \
  --lavira_session_id "$SESSION_ID" \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 120 \
  --g3_session_timeout_s 180 \
  --g3_motion_window_s 1.0 \
  --iplanner_url "http://127.0.0.1:8888" \
  --iplanner_timeout_s 5 \
  --local_use_isaac_odometry \
  --local_map_progress \
  --local_map_resolution_m 0.05 \
  --local_map_depth_stride 8 \
  --local_map_robot_radius_m 0.35 \
  --local_goal_tolerance_m 1.0 \
  --local_blind_yaw_radius_m 1.5 \
  --local_max_decisions 0 \
  --local_action_timeout_s 60 \
  --local_output_dir outputs/isaacsim_lavira_g3_interface_g1/scene200_doorway_map_progress
```

`--local_max_decisions 0` 和 `--max_steps -1` 表示不使用人为轮次/仿真步上限。运行会持续到：

- 模型完成 STOP 流程；
- 本地执行失败或超时；
- 用户按 `Ctrl+C`。

机器人明显顶墙、贴障碍或循环时立即按 `Ctrl+C`。当前 iPlanner 不读取本地 Map Progress，因此
Map Progress 只是记录证据，不是碰撞规避器。

### 5.4 Headless 诊断

需要排除 GUI 窗口状态时，在标准命令中增加：

```bash
--headless
```

相机、深度、iPlanner、模型调用和 Map Progress 仍会运行，但不会显示 Isaac 窗口。

## 6. 当前接口流程

### 6.1 Episode 开始

客户端调用：

```text
GET /health
POST /v1/lavira/session/start
```

`start_session` 只发送一次 instruction。服务器真实运行 Stage Planner 并返回 Frozen Stage Plan：

```json
{
  "schema_version": 2,
  "request_type": "start_session",
  "session_id": "<NEW_SESSION_ID>",
  "instruction": "Go through the doorway and continue into the next room."
}
```

只有 `status=ACTIVE` 且 `stage_plan_status=READY` 才能继续。

### 6.2 Navigator 决策

客户端上传当前四方向图像和最近历史：

```text
POST /v1/lavira/decision（multipart/form-data）
```

instruction 不需要每轮重复发送；服务器从 Session 中读取。当前典型响应：

```json
{
  "action": "NAVIGATE",
  "direction": "forward",
  "target": "open doorway",
  "bbox_2d": [248, 55, 350, 350],
  "waypoint": null,
  "session_status": "ACTIVE",
  "stage_plan_status": "READY"
}
```

### 6.3 阶段4 Stage Progress 与 STOP Gate

服务器每次按以下顺序处理同一个 `/decision`：

```text
Navigator → Stage Progress → Navigator为STOP时调用STOP Gate → 统一响应
```

`stage_progress` 始终位于响应顶层。`parse_success=false` 时仍保存完整对象和 `parse_error`，但正常
`NAVIGATE` 继续执行，不因监督解析失败而停止。

`image_count` 是本次监督请求的总图片数：首轮通常为4，后续会加上历史waypoint图片，例如一条
带图历史时为6；客户端按现有协议接受4到16，不能固定写成4。

单前向相机完成360度采集并回到参考朝向后，客户端把现有`/decision`请求提交给单个后台线程，
进入`WAITING_DECISION`。该状态不修改任何服务器字段，不重拍已经提交的图片，不产生执行上报，
并在每个Isaac/G1控制周期保持`desired_mode=locomotion`和速度`[0,0,0]`。模型响应完成后才执行
原有Stage Progress、STOP Gate、bbox和动作状态转换。旧四相机同时采集仅作为诊断回退，保持同步。

`NAVIGATE` 必须满足：

```text
stop_gate=null
stop_phase=null
```

`STOP` 时虽然仍保留 `direction/target/bbox_2d`，机器人端不使用这些字段，不做bbox投影、不调用
iPlanner，也不发送 `motion_window/action_complete`：

```text
STOP_CONFIRMED → 保持stand → end_session(SUCCESS, reason=stop_confirmed)
PREMATURE_STOP → decision_index+1 → 重新采集并请求decision
STOP_PENDING   → decision_index+1 → 重新采集并请求decision
```

顶层状态严格对应：

| stop_phase | control | next_action |
|---|---|---|
| `STOP_CONFIRMED` | `STOP_CONFIRMED` | `END_SESSION_SUCCESS` |
| `PREMATURE_STOP` | `CONTINUE` | `REQUEST_DECISION` |
| `STOP_PENDING` | `CONTINUE` | `REQUEST_DECISION` |

`STOP_PENDING`包含一种“强模型尚未调用”的合法状态：服务器需要先积累两个Frozen-Plan
Stage Progress观测时，会返回`parse_success=false`、`role_call_index=null`、`image_count=0`，并通过
`parse_error`说明等待原因。客户端接受该状态，不执行运动或阶段3执行上报，直接递增
`decision_index`并重新采集下一轮观测。强模型一旦实际调用，`role_call_index`仍必须是非负整数，
`image_count`仍必须在4到16之间。2026-08-27的真实Isaac受控STOP测试首次发现并修复了这一协议兼容点。

独立`strong_api`返回`ALLOW`时使用`completed=true`和`missing_subgoal=null`表示没有未完成子任务；
客户端以`null`作为当前正式表示，并兼容早期阶段四样例使用的空字符串。任何非空
`missing_subgoal`都不能与`ALLOW/STOP_CONFIRMED`同时出现。

### 6.4 阶段3执行上报

仅在 iPlanner 轨迹真正处于 `EpisodeState.EXECUTING` 平移期间约每1秒发送：

```json
{
  "schema_version": 2,
  "request_type": "report_execution",
  "event_type": "motion_window",
  "session_id": "scene200_test_001",
  "decision_index": 0,
  "event_id": "scene200_test_001:d0:w0",
  "window_index": 0,
  "action": "NAVIGATE",
  "timestamp_start": 10.0,
  "timestamp_end": 11.02,
  "pose_frame_id": "isaac_world",
  "frame_epoch": 0,
  "pose_start": [4.6, 3.5, -0.72],
  "pose_end": [4.75, 3.39, -0.72],
  "displacement_m": 0.186,
  "local_planner_status": "RUNNING",
  "distance_to_local_goal_start": 2.0,
  "distance_to_local_goal_end": 1.82,
  "map_progress": {
    "resolution_m": 0.05,
    "explored_cells": 13761,
    "new_explored_cells": 39,
    "traversable_cells": 7033
  }
}
```

`motion_window` 绝对不含 `status`。全景旋转、方向旋转、模型/iPlanner等待和 stand 阶段不发送窗口。
相同 `event_id` 重发时请求体保持一致；只有成功收到响应后才递增 `window_index`。

一个高层动作结束后只发送一次：

```json
{
  "schema_version": 2,
  "request_type": "report_execution",
  "event_type": "action_complete",
  "session_id": "scene200_test_001",
  "decision_index": 0,
  "event_id": "scene200_test_001:d0:complete",
  "action": "NAVIGATE",
  "status": "COMPLETED",
  "reached_local_goal": true,
  "timestamp": 14.3,
  "pose_frame_id": "isaac_world",
  "frame_epoch": 0,
  "decision_pose": [4.6, 3.5, -0.72],
  "final_pose": [5.2, 3.0, -0.71],
  "displacement_m": 0.781,
  "planner_result": "REACHED",
  "waypoint_id": 0
}
```

阶段3结果映射：正常到达为 `COMPLETED/REACHED`，超时为 `FAILED/TIMEOUT`，无轨迹为
`FAILED/PLANNING_FAILED`，跟随失败为 `FAILED/EXECUTION_FAILED`，外部中断为
`PREEMPTED/PREEMPTED`。当前服务器只允许返回 `CONTINUE`。

`map_progress` 必须是上述四字段对象，不接受 `null`，也不允许增加 `available` 等字段。
`new_explored_cells` 是当前 Motion Window 起止之间的累计探索格数差，不是最后一帧的新增量。
三个 cell 字段均为非负整数，`resolution_m` 必须是正的有限数。启用 G3 Session 时必须同时启用
`--local_map_progress` 和 `--local_use_isaac_odometry`。

## 7. 单前向 RGB-D 全景

Isaac 中仍保留 forward/left/behind/right 四个相机 prim，但当前模式只读取 forward 相机。每轮决策：

```text
当前朝向拍 forward
  ↓ 连续左转约 90°并拍 left
  ↓ 再转约 90°并拍 behind
  ↓ 再转约 90°并拍 right
  ↓ 完成第四段转动，回到参考朝向
  ↓ 四张图发给 Navigator
```

机器人不用在每个 90°位置切换到 stand；转动到对应方向时直接抓取 forward RGB-D，然后继续旋转。
模型选择 left/behind/right 后，执行链会再转到所选方向，用最新 forward RGB-D 投影并调用 iPlanner。

扫描证据保存在：

```text
decision_NNN_panorama_capture.json
```

当前 `local_blind_yaw_radius_m=1.5` 只表示距离局部安全终点小于 1.5 m 后停止 Pure Pursuit yaw
修正，以减少近目标绕圈；它不是目标到达距离。当前Isaac目标到达由
`local_goal_tolerance_m=1.0`判断；启用odometry时还会使用本节后述的越过目标保护。

如果 bbox 周围没有有效深度，当前兼容逻辑会使用 `[1.5, 0.0]m` fallback。该 fallback 可能把机器人
继续推向墙面，正式真机前必须加入更安全的失败处理，不能直接沿用。

### 7.1 Odometry越过局部目标保护（2026-08-31）

Isaac root pose或真机SLAM odometry可用时，局部安全目标会先固定到对应坐标系，再随机器人最新位姿
持续转换回机器人坐标。当前跟随器在正常`1.0m`到达判断之外，维护：

```text
best_goal_distance_m          本动作曾经到达的最小二维目标距离
goal_pass_guard_armed         是否曾进入 tolerance+0.2m，即0.7m范围
goal_distance_increasing_s    明显远离历史最近点的持续时间
```

触发条件全部满足时才认为已经越过局部目标：

```text
odometry可用
且 best_goal_distance_m <= 0.7m
且 current_distance >= best_goal_distance_m + 0.2m
且上述远离状态持续至少0.3s
```

触发后立即输出零速度，并沿用现有正常完成路径：

```text
follower reached=true
→ 清除当前局部轨迹
→ 等待稳定站立
→ action_complete status=COMPLETED
→ reached_local_goal=true
→ planner_result=REACHED
→ 请求下一次Navigator decision
```

它不新增服务器JSON字段，也不直接启动Recovery。若机器人从未进入0.7m范围，例如卡在目标外
`1.52m`，保护不会误判成功，仍由Physical Monitor或动作TIMEOUT处理。无odometry时继续保留原来的
`goal.x < -0.1m` dead-reckoning保护。重新规划同一固定目标不会清除本动作已经积累的最近距离证据；
开始新动作或停止跟随时才重置。

实现与测试位置：

```text
unified_vln/local_trajectory.py
tests/test_local_trajectory.py
```

### 7.2 可选Kp轨迹跟踪器（2026-09-01更新）

为保留现有实验的可复现性，原Pure Pursuit分支没有被替换。Isaac和真机入口现在都提供两个完全隔离的
机器人端局部跟踪模式：

```text
pure_pursuit  默认值；保持原有1.5m blind-yaw行为
kp            复用旧isaacsim_goal_tracking的固定坐标路径投影与全向比例跟踪
```

Isaac选择方式：

```bash
--local_tracking_controller pure_pursuit
--local_tracking_controller kp
```

真机选择方式：

```bash
--tracking-controller pure_pursuit
--tracking-controller kp
```

Kp分支不搬入旧的Session或Waypoint状态机，不修改Navigator、iPlanner、G3 JSON、Motion Window或
Recovery。它继续使用当前iPlanner安全轨迹、odometry固定目标、到达判断和越过目标保护，只替换局部
追踪算法：将每次iPlanner安全路径固定到odometry坐标系，把机器人投影到整条折线路径，再沿路径插值
取得lookahead点，最后转回机器人坐标生成`vx/vy/wz`。

默认参数来自旧Kp跟踪器的已用范围：

```text
kp_xy=0.7
kp_yaw=1.0
kp_slow_radius_m=1.0
kp_max_lateral_speed=0.12m/s
kp_max_yaw_speed=0.35rad/s
```

其中`vx/vy`指向插值lookahead点，`wz`追踪当前路径切线方向；终点附近继续按固定目标距离减速。
横向速度第一版限制为`0.12m/s`，低于locomotion policy训练命令上限，供单前向RGB-D的Isaac与真机
保守验证。历史`kp_yaw_deadband`参数仅为旧命令兼容保留，新Kp算法不再使用`blind_yaw_radius`或该
deadband。默认不传控制器参数时仍走原`pure_pursuit`，便于同一代码直接做A/B回归。

这次替换没有加入最终yaw到达要求：G3动作仍只以安全目标的二维距离、现有odometry越过保护和原
Episode收尾逻辑判断`COMPLETED/REACHED`。iPlanner每次重规划产生的新局部路径会按当时位姿重新固定，
不会复用旧路径的Session或FMM状态机。

### 7.3 scene_200去除第一展厅长椅的可恢复场景

为隔离验证长椅碰撞问题，新增场景覆盖层：

```text
scenes/scene_scene_200_without_exhibition_bench.usda
```

它加载原始`scene_scene_200.usda`，只将以下prim设为`active=false`：

```text
/scene_scene_200/Geometry/exhibition_room_bench_0
```

因此该长椅的视觉网格和碰撞/物理子节点会一起消失，墙体、画作及其他家具不变。原始场景文件没有
被修改；把运行参数`--scene_usd`切回原路径即可恢复长椅。该变体只用于Isaac测试，不改变真机、
iPlanner、G3接口或控制器逻辑。

### 7.4 PREMATURE_STOP P0原子抢占适配（2026-09-01）

服务器部署P0后，真实Isaac Session
`p0_new_kp_20260901_115909_19533`的decision 3已经证明服务器链路为：

```text
Navigator STOP
→ STOP Gate PREMATURE
→ Candidate Arbiter接受premature_stop候选
→ Strong Failure Verifier返回FAILURE/need_recovery=true
→ RecoveryState进入RECOVERY_PLANNING
→ control=PREEMPT
→ next_action=ACTION_COMPLETE_PREEMPTED
```

客户端现在按同一Pipeline处理，不另建恢复逻辑：`control`优先于`action`，所以这个`STOP`不表示任务
成功，也不执行iPlanner。客户端仅在完整验证`stop_gate + phase5 candidate/arbiter/transition/preemption
+ strong failure_verification`后，授权对应decision一次`STOP/PREEMPTED`回报；随后清除路径、保持零
速度、稳定站立并发送：

```text
action=STOP
status=PREEMPTED
reached_local_goal=false
planner_result=PREEMPTED
waypoint_id=decision_index
```

服务器回报`REQUEST_RECOVERY_DECISION`后，客户端才递增decision并请求Recovery Planner。普通
`PREMATURE_STOP/CONTINUE`、`STOP_PENDING`、`STOP_CONFIRMED`、Semantic Audit PREEMPT和在线
Physical Monitor PREEMPT的原有分支均保持不变。普通STOP仍不能发送motion window或
action_complete；P0授权在成功使用一次后立即清除，不能重复或跨decision使用。

新增自动化覆盖包括：完整P0元数据接受、不完整Arbiter元数据拒绝、一次性STOP/PREEMPTED授权、
零运动收尾、Recovery请求，以及原Semantic/Physical/STOP分支回归。真实Isaac还需重新触发一次同类
PREMATURE_STOP，验证远端对`STOP/PREEMPTED`的实际确认和随后的Recovery决策。

真实回归`p0_pipeline_retest_20260901_123157_22535`随后已经验证上述P0确认及Strong Recovery
Planner的`NAVIGATE/right/rainbow-colored painting`输出。Recovery开始执行后的第一个motion window
由服务器返回其既有状态：

```text
control=CONTINUE
next_action=CONTINUE_RECOVERY_EXECUTION
phase6.status=RECOVERY_EXECUTING
phase6.recovery_phase=RECOVERY_EXECUTING
```

客户端现已补齐该冻结分支，并严格限定它只能用于`event_type=motion_window`且reason为
`recovery_execution_window_recorded`的Phase6响应。接受后继续当前iPlanner路径，不清轨迹、不站立、
不递增decision，也不请求新模型决策；动作结束后仍按原Pipeline上报action_complete并进入Escape
Evaluator。

## 8. 稀疏 Map Progress

### 8.1 当前实现

`unified_vln/map_progress.py` 使用无固定边界的 5 cm 稀疏集合，不预分配 24 m × 24 m 地图：

```python
grid_x = floor(world_x / 0.05)
grid_y = floor(world_y / 0.05)
```

机器人移动时，已有格子的世界 key 不动，新区域只增加新的正/负整数 key。单前向深度图经过固定
相机外参和 Isaac root pose 投影到 `isaac_world`，使用 Bresenham 射线更新：

```text
observed_cells
occupied_cells
inflated_obstacle_cells
obstacle_hits
```

窗口字段定义：

```text
explored_cells       Episode 中观察过的唯一栅格数，单调不减
new_explored_cells   当前运动窗口内新增的唯一栅格数
traversable_cells    从机器人格可连通的已观察且未被膨胀障碍阻塞的格子数
```

其中 `traversable_cells` 是每次快照从当前机器人栅格重新执行连通搜索得到的**当前值**，不是累计值，
可以随当前位置和新融合障碍增加或减少。`explored_cells` 才是单调不减的累计值。

每个约 1 秒窗口保存：

```text
map_progress_decision_NNN_window_MMMM.json
map_progress/decision_NNN_window_MMMM.json
map_progress/decision_NNN_window_MMMM.png
```

### 8.2 2026-08-25 实测结论

scene_200 无限轮次测试目录：

```text
outputs/isaacsim_lavira_g3_interface_g1/
  sparse_map_doorway_test_007/
  sparse_map_doorway_test_20260825_007/
  run_20260825_185801_228531/
```

正常移动的 decision 0：

| window | new | explored | traversable | failures |
|---:|---:|---:|---:|---:|
| 0 | 423 | 13616 | 6779 | 0 |
| 1 | 239 | 13855 | 6823 | 0 |
| 2 | 171 | 14026 | 6835 | 0 |
| 3 | 323 | 14349 | 6925 | 0 |

机器人卡墙后的 decision 3：

| window | new | explored | traversable | occupied |
|---:|---:|---:|---:|---:|
| 0 | 9 | 16260 | 29 | 3581 |
| 1 | 340 | 16600 | 29 | 5050 |
| 2 | 43 | 16643 | 29 | 5247 |
| 3 | 0 | 16643 | 29 | 5296 |
| 4 | 0 | 16643 | 29 | 5302 |
| 5 | 0 | 16643 | 29 | 5302 |
| 6 | 0 | 16643 | 29 | 5302 |
| 7 | 0 | 16643 | 29 | 5302 |

因此已经验证：

- `explored_cells` 单调不减；
- 新区域运动时 `new_explored_cells > 0`；
- 卡住并重复观察时连续窗口变为 `new_explored_cells = 0`；
- 地图更新没有报错；
- 5 cm 地图下，移动窗口的 map gain 明显高于原 LaViRA `20 cells` 阈值。

已知在机器人卡墙的仿真样例中，`traversable_cells` 曾降到 29。当前阶段按用户确定的方案，
Isaac 中仍将该值作为可靠输入原样上报；不在客户端修改或平滑统计结果。迁移真机时需使用 D435
真实外参和 SLAM 位姿重新校准，但不改变四字段 HTTP Schema。

## 9. BACKTRACK 当前范围

阶段6已经启用stored-reverse BACKTRACK。服务器Recovery响应中的`waypoint`按冻结协议解释为稳定的
Waypoint Registry ID；客户端维护`server waypoint id → local history index`映射，再使用已验证的
世界路径面包屑生成反向分段，并将每段交给iPlanner。BACKTRACK真实平移期间同样发送约1秒
`motion_window`，动作结束发送一次`action_complete`。

已用受控测试验证：稳定Registry ID不依赖当前请求中截断后的临时history索引、到达当前waypoint时
不发送多余速度、完成后按Escape结果进入`REQUEST_RECOVERY_DECISION`或`REQUEST_DECISION`。

尚未完成的是远端真实Recovery Planner触发后的Isaac多段实走，以及真机SLAM全局路径版本。因此本地
受控BACKTRACK成功仍不能等同于完整真机Recovery成功。

## 10. 输出与实验判断

每轮输出根目录包含：

```text
g3_health.json
g3_session_started.json
g3_session_ended.json                 # 正常结束时才有
decision_NNN_request.json
decision_NNN_response.json             # 通过基础响应解析后才会生成
decision_NNN_projection.json
decision_NNN_plan.json
decision_NNN_completed.json
decision_NNN_panorama_capture.json
g3_decision_NNN_motion_MMMM.json       # 阶段3请求与服务器响应
g3_decision_NNN_action_complete.json   # 阶段3请求、结果原因与服务器响应
map_progress_decision_NNN_window_MMMM.json
map_progress/
images/iplanner/
```

实验不能只看程序回到命令行。判断标准：

- `finished: state=stopped ... failure=None`：本地流程正常结束；
- `Configured decision limit reached`：测试轮数上限结束，不代表模型完成任务；
- `session ENDED final_status=SUCCESS`：协议正常结束，不自动证明机器人真的到达语义目标；
- 机器人卡墙但程序继续：导航失败，Map Progress 记录可能仍然成功；
- 没有 `g3_session_ended.json`：检查并手动清理服务器 Session。
- 有`decision_NNN_request.json`但没有对应response：HTTP JSON可能在旧Navigation响应基础解析阶段已被
  拒绝；此时本地拿不到raw response，必须读取服务器落盘或容器日志。2026-08-28的Recovery decision 2
  即属于这种情况。

## 11. 当前已知问题

1. iPlanner 深度转 `uint16` 时仍可能打印 `invalid value encountered in cast`。Map Progress 已过滤
   NaN/Inf，但 iPlanner client 的转换还需单独修正。
2. bbox 底边深度可能落在门后地面或无效区域；固定 1.5 m fallback 可能导致顶墙。
3. 本地 Map Progress 不参与 iPlanner 避障，所以记录到障碍不表示机器人会避开障碍。
4. 当前 `motion_window` 已包含真实位姿、位移和四字段 `map_progress`；阶段5服务器可能在Verifier确认
   后返回`PREEMPT`，客户端会停止当前路径并进入Recovery，不再假定永远为`CONTINUE`。
5. 单相机连续旋转依赖 yaw/odometry；真机必须使用稳定 SLAM yaw，而不是仅靠命令时间估算。
6. 当前正式默认参数为`goal_tolerance_m=1.0`、`safe_distance_m=0.5`和
   `blind_yaw_radius_m=1.5`。`1.0m`实测会把不足1m的短安全轨迹在起点直接判为完成；这是当前依
   用户选择保留的已知取舍，不得在分析Recovery成败时忽略。odometry越过目标保护只在曾接近到
   `tolerance+0.2m`后、当前距离比历史最小值增加至少
   `0.2m`并持续`0.3s`时触发，随后按`COMPLETED/REACHED`结束当前局部动作并请求下一次Navigator。
7. Physical Monitor已在真实Isaac中生成过`REPEATED_FAILED_EDGE`候选（`navigation_windows=10`、
   `low_displacement_streak=8`、`map_gain_cells=1`）。当时因终态`stage_id=stage_total`使
   `expected_subgoal=None`而被服务器拒绝；该服务器问题已修复并部署，但修复后的
   `Physical candidate → Strong Verifier → PREEMPT`尚需再次真实触发。
8. Recovery SAFE_STOP已经由服务器受控确认并落盘；客户端只在
   `control=SAFE_STOP/action_source=RECOVERY`时绕过旧Navigator解析，普通STOP仍保持严格校验。
9. Stage Progress失败字段已经由`phase4_stage_progress_v2`归一化，失败结果不会进入STOP Gate有效窗口；
   仍需用真实模型回归一次照片STOP，确认部署后的运行路径与受控测试一致。
10. Recovery NAVIGATE可能收到iPlanner生成的短轨迹，再被固定`safe_distance_m=0.5`进一步截短；
    fear当前只记录、不参与客户端截断。本地`COMPLETED/REACHED`只表示到达截短轨迹终点，必须以
    服务器Escape Evaluator结果决定是否Handback。

## 12. 下一步实现顺序

阶段5至阶段7的服务器与客户端代码均已接入，PREMATURE_STOP后的真实PREEMPT、Recovery
NAVIGATE、Escape失败重试和SAFE_STOP已通过。下一步继续真实运动联调：

1. 检查`/health`包含`phase4_stage_progress_v2`和`phase6_recovery_control_v1`；
2. 使用独立新Session验证`STOP_CONFIRMED → end_session(SUCCESS)`的真实Isaac成功结束；
3. 在Isaac中重新触发修复后的Physical candidate，验证Strong Verifier和在线PREEMPT；
4. 验证`sustained=true`的Semantic candidate能否进入同一PREEMPT/Recovery入口；
5. 分别验证Recovery BACKTRACK实际执行、一次有效Escape后的Navigator Handback；
6. 最后将同一状态转换迁移到真机DDS + D435 + 预建SLAM地图重定位。

当前明确禁止提前加入自定义 `STOP_HOLD`、`STOP_EVALUATING`、`stop_attempt_id`、`SAFE_HOLD`、
`node_key` 或其他尚未由同学现有代码/冻结协议确认的字段和状态。

## 13. 真机迁移必须满足的条件

当前 Isaac 代码可以提供接口结构，但真机还需要：

- D435 RGB、米制 depth、真实内参；
- D435 相对 G1/SLAM frame 的标定外参；
- 在线 SLAM 输出稳定 `[x,y,yaw]`，并标明 `pose_frame_id`；
- SLAM 重定位/重置时递增 `frame_epoch`；
- G1 高层 DDS 速度命令、持续发送线程、可靠 StopMove 和硬件急停；
- 机器人安全层独立于模型、iPlanner 和 Physical Monitor；
- BACKTRACK 使用 SLAM/全局路径，而不是只依赖局部深度；
- 实际碰撞字段与“连续低位移”字段分开。

真机 Physical Monitor 中的 LaViRA `collision_streak` 应继续定义为 NAVIGATION 平移窗口的连续低
位移次数，不代表真实物理接触。真实接触、急停或人工 `Ctrl+C` 必须使用独立安全字段和处理路径。

## 14. 自动化测试

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project

conda run -n isaacsim python -m unittest discover \
  -s scripts/isaacsim_lavira_g3_interface_g1/tests \
  -v
```

测试覆盖 Session client、协议校验、单相机全景、bbox 投影、iPlanner client、轨迹跟踪、控制调度、
BACKTRACK、DDS backend、ROS2 odometry、信号处理和稀疏 Map Progress。自动化测试不替代 Isaac
场景实测或真机安全测试。

2026-09-01 当前结果：112项客户端自动化测试通过。除阶段4原有覆盖外，新增验证Physical
`motion_window → PREEMPT`、Semantic `action=NAVIGATE/control=PREEMPT`优先级、原子清除路径、
`action_complete=PREEMPTED`、Recovery NAVIGATE、稳定Registry ID BACKTRACK、失败后继续Recovery、
一次Escape成功后的Navigator Handback、Stage Progress v2健康协议，以及decision SAFE_STOP在旧
Navigator解析前进入失败闭环。P0新增覆盖`PREMATURE_STOP → Candidate Arbiter → Strong Verifier
→ STOP/PREEMPTED确认 → Recovery请求`的客户端边界。自动化结果不替代远端真实强模型和Isaac场景
联调。
