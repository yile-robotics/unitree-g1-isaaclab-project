# LaViRA G3 统一接口适配记录

> 历史文档：本文件保留早期阶段1～3的接口记录，不再表示当前完成状态。2026-08-28之后的唯一运行、
> 阶段5～7客户端实现和测试说明请以同目录`README.md`为准。

本文档记录 `isaacsim_lavira_g3_interface_g1` 与远端 LaViRA G3/b92 服务的接口边界、
第一阶段协议、截至 2026-08-25 的真实验证结果，以及后续接入顺序。

本目录是以下已跑通工程的工作副本：

```text
scripts/isaacsim_lavira_iplanner_g1
```

原目录保留不动。后续 G3 接口适配只应在本目录中进行。

> 当前状态：远端 Session、真实 Stage Planner 和原 Navigator 已经联通；本目录已经实现并
> 实机验证 `health / start_session / motion_window / action_complete / end_session` 自动生命周期。
> 下一项是多决策Episode以及Stage Progress接入。

## 1. 系统职责

### IsaacSim/G1 执行端（本项目）

负责：

- 前、左、后、右四方向 RGB-D 采集；
- bbox 与深度投影，生成机器人局部目标；
- 本机 iPlanner 调用；
- IsaacSim locomotion 或真机 G1 DDS 控制；
- SLAM/odometry、世界位姿、`node_key` 和 waypoint；
- 约 1 秒一次的运动证据汇总；
- BACKTRACK 的 stored-reverse 真实运动执行；SLAM全局重规划版本后续接入。

以下原始数据原则上不发送给远端算法服务：

- 完整深度图；
- 相机内外参；
- 完整 SLAM 地图；
- iPlanner 完整轨迹；
- G1 底层速度控制命令。

### 远端 LaViRA G3 算法服务

统一入口由朋友的服务器提供。内部最终负责：

- Stage Planner；
- Frozen Stage Plan；
- Navigator；
- Stage Progress；
- Physical/Semantic/STOP 监督；
- Failure Verifier；
- Recovery Planner；
- Escape Evaluator。

当前第一阶段真正接入的是 Session、Stage Planner 和原 Navigator；其余监督与恢复模块尚未
接入真实算法。

## 2. 当前远端服务

```text
remote host: 131.159.60.188
ssh user:    wangchu
remote port: 8765
framework:   G3
commit:      b92abb3
schema:      2
audit:       interval=2
container:   lavira-end2end-b92
```

服务器公网端口当前不能直接访问，使用 SSH 本地端口转发：

```bash
ssh \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -N \
  -L 18765:127.0.0.1:8765 \
  wangchu@131.159.60.188
```

该终端必须在实验期间保持运行。执行端统一访问：

```text
http://127.0.0.1:18765
```

本机 iPlanner 是独立服务，不经过远端 8765：

```text
http://127.0.0.1:8888
```

## 3. 第一阶段 Episode 流程

```text
GET /health
    ↓
POST /v1/lavira/session/start
    ↓ 服务器真实调用 Stage Planner
Frozen Stage Plan READY
    ↓
POST /v1/lavira/decision
    ↓ 原 b92 Navigator
NAVIGATE / STOP / BACKTRACK
    ↓
本地 bbox 投影 → iPlanner → IsaacSim/G1 执行
    ↓
POST /v1/lavira/execution/report
    ├── motion_window（约 1 秒一次）
    └── action_complete（一个高层动作结束一次）
    ↓
CONTINUE / PREEMPT
    ↓
POST /v1/lavira/session/end
```

第一阶段中的 `CONTINUE/PREEMPT/STOP` 目前主要是会话状态机结果，不代表 Physical Monitor、
Failure Verifier 或 STOP Evaluator 已经接入真实判断。

## 4. 健康检查

```bash
curl -sS --max-time 10 \
  http://127.0.0.1:18765/health \
  | python -m json.tool
```

2026-08-26 阶段3 Map Progress 协议部署后的执行端返回：

```json
{
  "status": "ok",
  "schema_version": 2,
  "framework": "G3",
  "commit": "b92abb3",
  "audit_interval": 2,
  "execution_protocol": "phase3_map_progress_v1",
  "map_progress_required": true
}
```

客户端使用这两个新增字段区分旧协议。Stage Planner 由紧随其后的
`start_session` 中 `stage_plan_status=READY` 和 Frozen Stage Plan 验证。

## 5. `start_session` 与 Stage Planner

`start_session` 是 Episode 初始化接口，Stage Planner 是它在服务器内部真实调用的模型角色。
instruction 在这里首次提交一次。

```bash
curl -sS --max-time 180 \
  -X POST \
  http://127.0.0.1:18765/v1/lavira/session/start \
  -H "Content-Type: application/json" \
  -d '{
    "schema_version": 2,
    "request_type": "start_session",
    "session_id": "yile_stageplan_test_001",
    "instruction": "Go through the door. Stop near the sofa."
  }' \
  | python -m json.tool
```

`request_type=start_session` 是必填字段。缺失时服务器返回：

```text
INVALID_SESSION_REQUEST: request_type must be start_session
```

2026-08-25 真实生成并返回：

```json
{
  "response_type": "session_started",
  "session_id": "yile_stageplan_test_001",
  "status": "ACTIVE",
  "stage_plan_id": "sha256:8fce7642e6ecab6fb2d6f22be6b33342e852053fa17c333c46d465c7e2347d85",
  "stage_plan_status": "READY",
  "frozen_stage_plan": {
    "stage_total": 2,
    "subgoals": [
      "Go through the door.",
      "Stop near the sofa."
    ]
  },
  "next_action": "REQUEST_DECISION"
}
```

`stage_plan_id` 是 Frozen Stage Plan 的内容标识。后续同一 Episode 应始终引用同一个 ID。

活动 Session 重复启动会返回：

```text
HTTP 409 Conflict
SESSION_ALREADY_ACTIVE
```

因此每次实验应使用新的 `session_id`，并在正常退出和异常清理路径中调用 `end_session`。

## 6. Session 内的 Navigator Decision

现有接口保持 multipart 四图协议：

```text
POST /v1/lavira/decision
```

Session 已启动后，decision metadata 可以不再重复携带 instruction。服务器根据 `session_id`
恢复保存的 instruction。2026-08-25 已用离线四图从执行端真实验证：

```json
{
  "session_id": "yile_stageplan_test_001",
  "observation_id": "yile_stageplan_test_001_decision_000",
  "action": "NAVIGATE",
  "direction": "right",
  "target": "door",
  "bbox_2d": [306, 178, 340, 305],
  "session_status": "ACTIVE",
  "stage_plan_status": "READY"
}
```

模型 reasoning 明确恢复了 Session 中保存的任务：

```text
The goal is to go through the door and stop near the sofa.
```

原 Navigator Prompt、图片处理、Parser 和 bbox 返回逻辑没有修改。Frozen Stage Plan 当前保存于
Session，主要用于后续 Stage Progress 和监督链；不能仅根据上述结果宣称 Stage Plan 已直接加入
Navigator Prompt。

为了降低第一轮客户端接入风险，本目录现有 decision 请求可以暂时继续携带与 Session 一致的
instruction。待 Session 自动流程稳定后，再决定是否从 decision metadata 中移除该重复字段。

## 7. `execution/report`

服务端已经声明支持两类记录：

```text
motion_window
action_complete
```

计划语义如下：

- `motion_window`：只在 NAVIGATION 或 RECOVERY 真实运动期间约每 1 秒汇总一次；
- `action_complete`：一个 Navigator/Recovery 高层动作结束时发送一次；
- 同一个 `decision_index` 可以对应多条 `motion_window`；
- 模型等待、相机采集、模式切换和原地停车期间不生成虚假运动窗口；
- `motion_index` 应在 Episode 内单调递增；
- `report_id` 应唯一，以支持重试和幂等处理。

第一阶段服务器的准确最小请求已经固定为：

```json
{
  "schema_version": 2,
  "request_type": "report_execution",
  "session_id": "episode_id",
  "decision_index": 0,
  "action": "NAVIGATE",
  "status": "COMPLETED",
  "event_type": "motion_window"
}
```

`action_complete` 使用相同结构，只把 `event_type` 改为 `action_complete`。服务器返回
`response_type=execution_control` 和 `control=CONTINUE/PREEMPT`。

本目录已实现：

- `unified_vln/session_client.py`：严格校验 health、Session、Stage Plan 和执行控制响应；
- `EXECUTING` 阶段每约 1 秒发送一条 `motion_window`；
- 站稳并提交动作前发送一条 `action_complete`；
- 收到 `PREEMPT` 时立即清零并终止第一阶段执行器，绝不误进入下一次 Navigator；
- runner 正常结束、失败或外层终止时尽力发送 `end_session`；
- Session、报告和结束响应写入本次运行输出目录。

当前最小 `motion_window` 仍没有真实位移、碰撞、地图进展、`report_id` 和 `motion_index` 字段，
因此只能支撑第一阶段生命周期，不能冒充 Physical Monitor 的真实证据接口。

## 8. 已验证的 IsaacSim 最小导航链路

在 Session 协议接入之前，本目录已经通过一次实时 IsaacSim 单决策测试：

```text
IsaacSim 四方向 RGB-D
→ 远端 b92 Navigator
→ bbox 深度投影
→ 本机 iPlanner
→ IsaacSim G1 执行
```

测试结果：

```text
action:                 NAVIGATE
direction:              left
target:                 nearest visible doorway
bbox:                   [305, 194, 343, 469]
projection depth:       1.763 m
goal after turn:        [1.763, -0.018] m
iPlanner fear:          0.00133
iPlanner raw trajectory: 51 points
safe local goal:        [1.246, -0.132] m
result:                 history=1, failure=None
```

对应输出：

```text
outputs/isaacsim_lavira_iplanner_g1/
  b92_g3_single_decision_001/
  run_20260825_104337_097128/
```

上述旧测试输出仍使用继承自原工程的 `isaacsim_lavira_iplanner_g1` 名称。当前新 runner 的
默认输出已经改为：

```text
outputs/isaacsim_lavira_g3_interface_g1/
```

### 8.1 新客户端真实HTTP验证

2026-08-25 使用本目录新客户端和离线四图，对远端服务真实运行：

```text
health
→ start_session
→ decision
→ motion_window
→ action_complete
→ end_session
```

结果：

```text
health:            ok
stage_total:       1
action:            NAVIGATE
direction:         right
target:            bed
motion_control:    CONTINUE
complete_control:  CONTINUE
end:               ENDED
final_status:      SUCCESS
```

可重复运行：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/smoke_test_g3_session_http.py \
  --decision-url http://127.0.0.1:18765/v1/lavira/decision
```

### 8.2 自动IsaacSim单决策命令

SSH隧道和本机 iPlanner 启动后运行：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/run_isaacsim.py \
  --house \
  --spawn 1.15 5.25 0.8 \
  --device cuda:0 \
  --real-time \
  --no-four_rgbd_set_viewport \
  --instruction "Move toward the nearest visible doorway and stop near it." \
  --lavira_session_id "g3_auto_episode_001" \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 90 \
  --g3_session_timeout_s 180 \
  --g3_motion_window_s 1.0 \
  --iplanner_url "http://127.0.0.1:8888" \
  --iplanner_timeout_s 5 \
  --local_max_decisions 1 \
  --local_goal_tolerance_m 0.4 \
  --local_blind_yaw_radius_m 0.2 \
  --local_safe_distance_m 0.5 \
  --local_replan_interval_s 0.1 \
  --local_dead_reckoning_linear_scale_sim 1.0 \
  --local_dead_reckoning_angular_scale_sim 1.0
```

新目录默认启用 G3 Session。只有诊断旧单阶段服务时才使用 `--no-g3_session`。

2026-08-25 已使用该命令真实完成：

```text
Session ACTIVE，Frozen Stage Plan共2阶段
→ Navigator: NAVIGATE/right/doorway
→ bbox+depth局部目标: [1.712, -0.803] m
→ iPlanner连接并执行
→ 3条motion_window，全部CONTINUE
→ action_complete(COMPLETED)，CONTINUE
→ 单决策安全上限停止
→ end_session: ENDED/SUCCESS
```

Frozen Stage Plan：

```text
1. Locate the nearest visible doorway.
2. Move toward the doorway and stop near it.
```

所有Session、Decision、motion、action complete和end响应都保持同一个`stage_plan_id`：

```text
sha256:00818225d39618519c900e29cb512ff5d6289bf6aae9ad5456f0201009ba62d3
```

完整实验记录：

```text
outputs/isaacsim_lavira_g3_interface_g1/
  g3_auto_episode_20260825_002/
  run_20260825_124453_266330/
```

## 9. 当前实现边界

### 已经完成或验证

- G3/b92 health；
- SSH 隧道访问；
- Session 生命周期骨架；
- 真实 Stage Planner；
- Frozen Stage Plan 保存与哈希标识；
- 活动 Session 重复请求返回 409；
- decision 不重复发送 instruction；
- 原 Navigator 与 bbox；
- 本地 bbox/depth 投影；
- 本地 iPlanner 与 IsaacSim 单次执行；
- G3 Session 严格HTTP客户端；
- 自动 health 和 `start_session`；
- 自动 1 秒 `motion_window`；
- 自动 `action_complete`；
- 自动清理 `end_session`；
- `PREEMPT` 原子停车；
- 完整离线四图真实HTTP闭环；
- 第一版 stored-reverse BACKTRACK：决策位姿、实测路径、历史分支截断、分段 iPlanner、G3
  运动/完成上报和无位姿安全停车；
- 自动化测试 `59 passed`。

### 尚未完成

- 含真实位移/碰撞/地图增量的完整运动记录器；
- Stage Progress；
- Physical Monitor 的真实证据判断；
- Semantic Auditor；
- STOP Evaluator 的真实二阶段判断；
- Failure Verifier；
- Recovery Planner；
- Escape Evaluator；
- SLAM OccupancyGrid/FMM 全局绕行 BACKTRACK（stored-reverse 第一版已完成 scene_200 实测）；
- 真机 SLAM、DDS 与整套 G3 联调。

## 10. 后续实现顺序

1. 用多决策Episode验证同一Frozen Stage Plan始终不变；
2. 与服务器共同扩展含真实证据的 `motion_window` v2 字段；
3. 接入 Stage Progress；
4. 在已通过 scene_200 stored-reverse BACKTRACK 的基础上接入SLAM全局重规划；
5. 接入监督、Failure、Recovery 与 Escape；
6. 替换位姿后端，在真实 G1 上使用在线 SLAM 和 DDS 验证。

计划中的本地客户端结构：

```text
G3SessionClient
├── health_check()
├── start_session()
├── request_decision()          # 复用现有客户端
├── report_motion_window()
├── report_action_complete()
└── end_session()
```

## 11. b92 40% 基线说明

此次服务端改动没有修改：

```text
vlnce_baselines/end2end_decision.py
vlnce_baselines/lavira_main_qwen_end2end.py
```

原 Navigator Prompt、采样、图片输入、Parser 和 bbox 输出路径保持不变。Stage Planner 使用独立
role seed，不占用 Navigator 采样序号。因此可以表述为“原 b92 Navigator 代码和输入路径未改”。

但新增 Session 适配层后尚未重新运行完整 100 集评测，不能表述为“新服务重新测得 40%”。

## 12. 常见问题

### `request_type must be start_session`

`start_session` JSON 缺少：

```json
"request_type": "start_session"
```

### `SESSION_ALREADY_ACTIVE`

同名 Session 仍处于 ACTIVE。使用新的 `session_id`，或者按协议结束旧 Session。

### iPlanner `Connection refused`

本机 iPlanner 未启动。运行：

```bash
conda activate isaacsim
cd /home/yile/projects/unitree-g1-isaaclab-project
bash scripts/isaacsim_lavira_g3_interface_g1/run_iplanner_local.sh
```

确认：

```bash
curl -sS http://127.0.0.1:8888/health | python -m json.tool
```

### 深度转换出现 `invalid value encountered in cast`

Isaac 深度图包含 NaN/Inf。该警告在已完成的单决策实验中不是停止原因；后续应在不改变
iPlanner 协议语义的前提下明确清洗无效深度。

## 13. 2026-08-25：stored-reverse BACKTRACK 第一版

当前执行端不再把合法 `BACKTRACK` 直接判为不支持。每个完成的 NAVIGATE 在本地额外保存：

- 拍摄该次模型全景时的 `decision_pose`；
- 动作完成后的 `arrival_pose`；
- 执行期间由 Isaac root pose 或真机 SLAM pose 测得的世界坐标 breadcrumb 路径。

模型返回 wire `waypoint` 后，客户端先映射到本地稳定 waypoint id，再复用旧
`isaacsim_goal_tracking` 已验证的语义：接受动作时立即截断目标后的错误历史分支，把成功路径倒序
拼接成世界返回路线。路线按默认 `1.0m` 分段；每段先转向目标，再用当前 forward RGB-D 调用
iPlanner，期间继续发送 `action=BACKTRACK` 的 `motion_window`，最终发送一次 `action_complete`。

相关安全参数：

```text
--local_backtrack_max_path_m 6.0
--local_backtrack_start_tolerance_m 1.0
--local_backtrack_segment_length_m 1.0
--local_backtrack_goal_tolerance_m 0.35
--local_backtrack_heading_tolerance_rad 0.20
--local_backtrack_breadcrumb_spacing_m 0.15
```

Isaac正式测试必须使用 `--local_use_isaac_odometry`；真机必须提供新鲜、稳定的 SLAM odometry。
没有世界位姿、历史路径、有效目标编号或 iPlanner 路径时统一清零速度并 fail closed。该版本尚未在
OccupancyGrid 上搜索全新绕行路线，原路完全堵塞时需要后续全局规划版本。

## 14. 2026-08-25：scene_200 BACKTRACK 实测与参数记录

为避免等待远程模型随机产生 BACKTRACK，使用独立的
`scripted_backtrack_server.py` 固定生成两次 NAVIGATE 和一次 `BACKTRACK waypoint=0`；运行时使用
`--no-g3_session`，所以该实验只验证本地物理执行器，不冒充完整 Recovery Planner 联调。

成功输出：

```text
outputs/isaacsim_lavira_g3_interface_g1/
  scene200_scripted_backtrack_001/
  run_20260825_135609_321383/
```

结果为：倒序路径 `3.107 m`、起点漂移 `0.054 m`、3 个 iPlanner 分段、历史 `2 -> 1`；目标
decision pose 为 `[4.601, 3.498]`，完成 pose 为 `[4.505, 3.802]`，二维误差约 `0.319 m`，小于
`--local_backtrack_goal_tolerance_m 0.35`。最终日志为：

```text
BACKTRACK completed: waypoint=0 history=2->1 segments=3
state=stopped
failure=None
```

当前后续 scene_200 和 G3 接口测试统一显式使用：

```text
--local_goal_tolerance_m 1.0
--local_blind_yaw_radius_m 1.5
```

2026-08-31起，局部轨迹控制器已隔离为可选模式。未指定时继续使用上述原Pure Pursuit行为；测试旧Kp
跟踪规律时显式增加：

```text
--local_tracking_controller kp
--local_kp_xy 0.7
--local_kp_yaw 1.0
--local_kp_slow_radius_m 1.0
--local_kp_max_lateral_speed_m_s 0.12
--local_kp_max_yaw_speed_rad_s 0.35
```

Kp模式复用旧`isaacsim_goal_tracking`的固定坐标路径投影、插值lookahead和`vx/vy/wz`比例控制，
`vy`默认限制为`0.12m/s`。它不改变G3服务器接口、iPlanner、Motion Window、Physical Monitor或
Recovery，也不增加最终yaw到达要求。删除`--local_tracking_controller kp`即可回到原控制器。

在默认`pure_pursuit`模式，`1.5 m`是停止yaw修正半径：进入该范围后继续直行，进入`0.5 m`后完成
普通NAVIGATE。在可选`kp`模式，追踪器使用路径切线控制yaw，不使用`1.5m blind-yaw`。旧文档
命令中的`0.2 m`是早期小场景历史参数，不再用于后续G3联调。真机第一次测试两种模式都必须保留
急停，并根据实际SLAM、相机外参与底盘响应复核，且不得在没有记录的情况下静默改参数。

同日远程 G3 实验已运行到 `STOP_PROPOSED`，证明 Stage Planner、Frozen Stage Plan、多轮
Navigator 和基础执行上报可用；由于 STOP 二阶段 control 枚举尚未对齐，该实验不证明 STOP
Evaluator 或独立 Stage Progress 已经完成。

## 15. 2026-08-25：单前向 RGB-D 连续旋转全景

Isaac 与真机统一改为一个 forward RGB-D 输入。每次高层决策前先拍 forward，随后在 locomotion
模式下连续完成四个左转 90°分段；前三段结束时立即把同一 forward 相机的最新帧依次标记为
left、behind、right，第四段结束后回到参考朝向并请求 Navigator。拍摄点不切换 stand，不等待
额外稳定时间，也没有单独的角度 tolerance 配置。

Isaac 机器人上的 left/behind/right 虚拟相机没有删除，场景配置和资产保持不变；默认状态机只调用
`capture_forward()`。`capture_panorama()` 保留为 `--no-local_single_camera_panorama` 诊断回退。

运行日志应出现：

```text
single-camera panorama started
single-camera panorama captured: direction=left
single-camera panorama captured: direction=behind
single-camera panorama captured: direction=right
single-camera panorama complete
```

每轮额外保存 `decision_NNN_panorama_capture.json`，记录四张图的 frame id、时间戳和可用位姿。
默认 blind-yaw 半径也已在新 runner 和 follower 中固定为 `1.5 m`。
