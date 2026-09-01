# Isaac Sim + Unitree G1 + LaViRA 导航系统

更新时间：2026-07-29

本文是 `scripts/isaacsim_goal_tracking` 的唯一权威说明，覆盖当前已经实现的功能、服务器接口、
运行方法、输出文件、已验证结果和尚未完成的部分。

如果文档与代码发生冲突，以以下文件为准：

```text
goal_tracking/lavira_protocol.py
goal_tracking/lavira_offline.py
goal_tracking/lavira_episode.py
goal_tracking/lavira_global_mapping.py
goal_tracking/config.py
isaacsim_path_follwing.py
```

## 1. 当前完成度

当前已经跑通单轮完整导航链，并实现了带 history 的有限多轮 controller：

```text
Isaac Sim 同步四方向 RGB-D
  -> multipart/form-data 上传四张 RGB
  -> SSH 隧道
  -> 远程服务器本地 Qwen/LaViRA
  -> action / direction / target / bbox_2d / waypoint
  -> direction 对应的原 FrameBundle depth + K + 相机位姿
  -> bbox 底边中点三维投影
  -> 四路 depth 当前观测
  -> 固定世界坐标的 episode 累计全局地图
  -> obstacle / explored / current / past 通道 max 融合
  -> G1 尺寸障碍膨胀和安全目标修正
  -> scikit-fmm 距离场和无碰撞 waypoint
  -> 原 WaypointPathFollower
  -> 原 SwitchCommandController
  -> 原 29DOF locomotion ONNX
  -> G1 运动
  -> 可选：行走期间低频抓取四方向 RGB-D 并融合到同一 full map
  -> 可选：持续命令但无位移时把机器人前方写入独立 collision_map
  -> 可选：对同一个 NAVIGATE/BACKTRACK/STOP 世界目标周期重新运行 FMM
  -> 可选：运动中热替换路径，不触发 stand/locomotion 往返切换
  -> 到达短期目标后归零并平滑切回 stand
  -> 稳定 0.8s 后提交当前 waypoint
  -> 重新抓取四视角
  -> 自动发送下一轮 decision + 全部已完成 history
  -> 默认允许 decision_000、decision_001 的普通 NAVIGATE/BACKTRACK 执行
  -> 普通 decision_002 只校验保存，不再启动新路径
  -> 任意一轮 STOP 复用同一 FMM 路径完成最终接近并结束 episode
```

当前准确定位：

```text
已实现：一次模型决策 + 累计全局地图规划 + 一次短路径执行
已实现：decision_000 执行成功后生成 history 并自动请求 decision_001
已实现：有限多轮 NAVIGATE/BACKTRACK/STOP controller，默认三次请求
已实现：BACKTRACK 默认按历史 waypoint 世界坐标在累计全局地图重新 FMM，并立即截断 history
已实现：LaViRA 风格 STOP 最终接近、稳定 stand 和 episode 终止
已实现：可选的执行期四视图在线全局地图融合
已实现：适配 G1 连续控制的在线 collision map
已实现：三种 action 共用的稳定活动目标、周期 FMM 和运动中路径热替换
已验证：累计全局地图下两次 NAVIGATE、history 提交、Qwen 主动 STOP 和稳定 stand
已验证：累计全局地图下真实 Qwen 两次选择 waypoint=0，均重新 FMM、截断 history 并成功返回
已验证（累计全局地图加入前）：默认 `replan_world_goal` 下 Qwen 两次选择 waypoint=0 并成功返回
已验证：旧 `stored_reverse` 策略下真实服务器返回 waypoint=1，G1 沿 1.962m 反向路径到达
已验证：mock STOP 下的“目标已在阈值内”无移动终止
已验证：mock STOP 下的 FMM 最终接近、真实 G1 locomotion、0.75m 停止和 stand
未实现：无限 episode、ground-truth task success 和 SPL 等评测
```

默认 `--lavira_decision_probe` 仍然只请求 `decision_000`；需要多轮执行时使用
`--lavira_history_probe` 的 LaViRA 风格有限循环。默认最多请求 3 次：执行前两个普通
NAVIGATE/BACKTRACK，每次成功后更新 history，第三次普通响应只保存。STOP 是终止动作，
即使出现在第三次也会执行最终接近。`--lavira_history_max_decisions` 可以调整边界。

### 1.1 2026-07-27：加入 LaViRA 兼容的累计全局地图

bounded episode 的默认地图模式现在是：

```text
--nav_map_mode lavira_compatible_global
```

第一轮 `FrameBundle` 自动读取实际机器人世界坐标，并把该位置放在固定 full map 的中心。之后
机器人移动不会改变 full map 原点；每轮四路 RGB-D 当前观测会转换到这个固定坐标系，并用
channel-wise `max` 融合。因此更换 Isaac 场景或修改 `--spawn` 不需要改地图代码。

当前全局通道与 LaViRA 前四个通道对应：

```text
channel 0 = obstacle
channel 1 = explored
channel 2 = current robot location
channel 3 = past robot locations
```

默认 `24m / 0.05m = 480 × 480` full map，`--nav_global_downscaling 2` 产生
`240 × 240` local window；local window 每 25 次地图更新重新居中。地图物理尺寸和分辨率都
是运行时参数，不依赖固定 house 或固定起点。

三种模型 action 使用同一个累计状态：

```text
NAVIGATE / STOP:
  Qwen bbox -> depth 世界坐标 -> 累计全局 traversability -> FMM

BACKTRACK:
  Qwen 明确 waypoint id -> history decision_world_pose -> 同一累计全局图 -> FMM
```

Qwen 仍负责决定是否 `BACKTRACK` 以及返回哪个 waypoint；地图模块不会自行把 0/1/2 中的某个
编号替模型选出来。接受 BACKTRACK 后 history 截断语义没有变化。

新增参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--nav_map_mode` | `lavira_compatible_global` | 累计全局图；可切回 `local_current_bundle` |
| `--nav_global_origin_mode` | `spawn_center` | 第一帧实际起点作为地图中心 |
| `--nav_global_origin_world_x_m/y_m` | `None` | manual 模式下指定 full map 左下角 |
| `--nav_global_downscaling` | `2` | full/local 边长比例 |
| `--nav_global_center_reset_steps` | `25` | local window 重置频率 |
| `--nav_global_unknown_space_policy` | `blocked` | G1 默认不进入未观测格；`lavira` 更接近原实现 |

手动固定地图边界的例子：

```bash
--nav_global_origin_mode manual \
--nav_global_origin_world_x_m -12.0 \
--nav_global_origin_world_y_m -12.0
```

这里的 manual X/Y 是 full map 左下角，不是机器人的出生点；机器人出生点仍由 `--spawn`
独立设置。如果机器人走出固定边界，controller 会明确报错并提示增大 `--nav_map_size_m` 或
修改 manual origin。

每轮输出新增：

```text
lavira_global_map.json
lavira_global_map.npz
lavira_global_map.png
global_planning/fmm_plan.json
global_planning/fmm_distance.npy
global_planning/fmm_path.png
```

NPZ 保存 `full_map`、`one_step_full_map`、`local_map` 和累计 obstacle hits，便于核对旧地图
是否保留、当前/历史位置通道和 FMM 输入。旧行为可以显式恢复：

```bash
--nav_map_mode local_current_bundle
```

与原版 LaViRA 仍有两个明确差异：当前只有几何四通道，没有 Grounded-SAM semantic category
通道；Isaac 的地图更新使用独立低频 navigation tick，而不是把四相机建图塞进每个高频 G1
physics step。代码没有伪造语义通道，默认也继续阻止 G1 进入未知区域。

### 1.1.1 2026-07-29：执行期在线地图、collision map 和周期 FMM

新增的在线闭环默认关闭，必须显式使用：

```bash
--lavira_online_navigation
```

开启后，controller 在 `EpisodeState.EXECUTING` 中保持模型选定的同一个世界目标：

```text
NAVIGATE -> 首次规划得到的 safe_target_world_xy
BACKTRACK -> Qwen waypoint 对应的 decision_world_pose[:2,3]
STOP -> 首次最终接近的 stop_goal_world_xy
```

在线更新不会增加 `decision_index`、不会发送新的 Qwen 请求、不会增加或截断 history。它只执行：

```text
每隔 mapping interval 抓取当前四路 RGB-D
  -> 融合到固定 full map
  -> 每隔 replan interval 对原活动目标重新运行 FMM
  -> 原子热替换 WaypointPathFollower 路径
```

G1 collision 判定要求 locomotion 已接管、切换结束、实际平移命令超过阈值，并且在完整时间窗口
内根节点 XY 进度仍低于阈值。纯旋转、低速 ramp、stand 和 policy 切换会重置检测窗口。确认的
碰撞写入独立 `collision_map`，规划时执行：

```python
planning_traversable = lavira_traversable & ~collision_map
```

相机障碍通道仍保留 LaViRA 风格的 episode 内 `max` 累计，碰撞推断不会污染 `full_map[0]`。
在线 BACKTRACK 只支持默认 `replan_world_goal`，防止显式选择 `stored_reverse` 后又被周期世界
目标规划静默替换。

新增参数：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--lavira_online_navigation` | false | 开启执行期地图/collision/FMM 闭环 |
| `--lavira_online_mapping_interval_s` | 1.0 | 四视图地图融合周期 |
| `--lavira_online_replan_interval_s` | 1.0 | 周期 FMM 最小间隔 |
| `--lavira_collision_command_speed_m_s` | 0.12 | 进入停滞检测的最小平移命令 |
| `--lavira_collision_window_s` | 0.75 | 连续无进度检测窗口 |
| `--lavira_collision_min_progress_m` | 0.04 | 窗口内最小实际 XY 进度 |
| `--lavira_collision_mark_distance_m` | 0.45 | 碰撞圆心在运动方向前方距离 |
| `--lavira_collision_mark_radius_m` | 0.15 | collision mask 标记半径 |

每个正在执行的 decision 目录会新增或更新：

```text
online_navigation.json
online_latest/lavira_global_map.json
online_latest/lavira_global_map.npz
online_latest/lavira_global_map.png
online_latest/fmm_plan.json
online_latest/fmm_distance.npy
online_latest/fmm_path.png
```

`lavira_global_map.npz` 现在也包含独立的 `collision_map`。

#### 2026-07-27 真实 Qwen + 累计全局地图运行审计

运行目录：

```text
outputs/isaacsim_goal_tracking/lavira_offline/
run_20260727_124833_865149/robot_01_global_map_test_001
```

任务：

```text
Go through the doorway, then turn left and stop near the bed.
```

真实执行序列为：

```text
decision_000 NAVIGATE left -> 到达并提交 waypoint 0
decision_001 NAVIGATE left -> 到达并提交 waypoint 1
decision_002 STOP forward  -> 已在 0.75m 阈值内，稳定 stand 并结束 controller
```

全局地图数组审计结果：

| decision | action | updates | obstacle | explored | current | past |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `000` | NAVIGATE | 1 | 1908 | 7044 | 29 | 0 |
| `001` | NAVIGATE | 2 | 2483 | 7860 | 29 | 29 |
| `002` | STOP | 3 | 2587 | 7908 | 29 | 58 |

三轮 full map 原点始终为：

```text
[-10.848760962486267, -6.751875877380371]
```

第一帧实际机器人位置为：

```text
[1.151239037513733, 5.248124122619629]
```

两轴均相差 12m，证明默认 24m full map 正确地以第一帧实际起点为中心。直接读取三个
`lavira_global_map.npz` 后确认：

- `full_map.shape == (4, 480, 480)`，通道均为真实二值数组。
- obstacle/explored 在 `000 -> 001 -> 002` 中单调不减。
- 每一轮 previous current 都完整进入下一轮 past。
- `local_map` 与 `local_bounds_rc_exclusive` 指定的 full-map 裁剪完全一致。
- 当前观测局部原点随机器人改变，而 full-map 原点保持不变。

实际交给 follower 的是 `global_planning/fmm_plan.json`：

```text
decision_000:
  global path = 1.585m / 8 waypoints
  next decision root distance to goal = 0.147m

decision_001:
  local probe path  = 0.429m
  global execution = 0.135m / 2 waypoints
  next decision root distance to goal = 0.117m

decision_002:
  global STOP goal distance = 0.143m
  threshold = 0.75m
  result = STOPPED, robot_standing=true, model_stop_completed=true
```

第二轮局部和全局路径不同是预期行为：累计地图把床表面目标判为不可通行，并通过
`global_nearest_traversable` 选择床旁的安全格，runner 日志也明确接受了 0.135m 的全局路径，
不是 0.429m 的局部 probe 路径。

本轮发现的主要风险是 Qwen 三轮都返回近似整图 bbox：

```text
[0, 0, 640, 476]
```

decision 000 的 bbox 底边中点落在地板上，投影高度约 `z=0.002m`，而不是床本身；后两轮底边
中点因为床占据画面下部才落到约 `z=0.503m` 的床面。这说明方向、目标和 STOP 判断可以正确，
但当前服务器/Qwen 的 bbox grounding 不够精确，换场景后可能导致错误三维目标。

这轮没有返回 BACKTRACK，因此它只验证了累计全局图上的 NAVIGATE、history 和 STOP。累计
全局图上的真实 Qwen BACKTRACK 随后已经通过下面的专项运行完成验证。

#### 2026-07-27 真实 Qwen + 累计全局地图 BACKTRACK 验证

运行目录：

```text
outputs/isaacsim_goal_tracking/lavira_offline/
run_20260727_131434_233382/robot_01_global_backtrack_test_003
```

该运行使用 10 轮执行窗口。真实 Qwen 分别在 decision 003 和 decision 005 返回：

```json
{
  "action": "BACKTRACK",
  "waypoint": 0
}
```

两次动作都完成了同一条默认实现链：

```text
Qwen 选择 waypoint 0
  -> 从 history_commit.json 读取 waypoint 0 的 decision_world_pose[:2,3]
  -> 把本轮四视图融合进 episode 累计 full map
  -> 在固定全局坐标图上设置 historical waypoint 目标
  -> 重新运行 FMM，而不是反向重放旧路径
  -> 接受动作时立即把 history 截断到 waypoint 0
  -> 复用现有 pure-pursuit 和 G1 locomotion
  -> 路径到达、切回 stand、稳定后写入 status=arrived
```

waypoint 0 保存的原始历史世界坐标为：

```text
[1.151239037513733, 5.248124122619629]
```

固定全局地图原点始终为：

```text
[-10.848760962486267, -6.751875877380371]
```

FMM 目标格均为：

```text
goal_cell_rc = [240, 240]
target_selection_strategy = global_historical_waypoint_traversable
```

对应的格中心世界坐标为：

```text
[1.1762390375137333, 5.273124122619629]
```

它与原始历史坐标每轴相差半个 `0.05m` cell，属于正常的世界坐标到栅格中心量化，不是选错
waypoint。

第一次真实全局 BACKTRACK：

```text
decision_index        = 3
strategy              = replan_world_goal
target_waypoint       = 0
history_count_before  = 3
history_count_after   = 1
path_length_m         = 0.2795666502
completion_step       = 2965
status                = arrived
execution_source      = lavira_global_replanned_backtrack_waypoint_000_decision_003
```

第二次真实全局 BACKTRACK：

```text
decision_index        = 5
strategy              = replan_world_goal
target_waypoint       = 0
history_count_before  = 2
history_count_after   = 1
path_length_m         = 0.2115347327
completion_step       = 4002
status                = arrived
execution_source      = lavira_global_replanned_backtrack_waypoint_000_decision_005
```

对应日志均完整出现：

```text
BACKTRACK accepted
BACKTRACK path reached
BACKTRACK completed
```

两个 decision 目录都包含：

```text
response.json
response_interpretation.json
lavira_global_map.json
lavira_global_map.npz
global_planning/fmm_plan.json
global_planning/fmm_distance.npy
global_planning/fmm_path.png
backtrack_execution.json
```

`response_interpretation.json` 中的 `global_execution_plan.status` 为 `ok`，action 为
`BACKTRACK`；`backtrack_execution.json` 最终状态均为 `arrived`。这证明累计全局地图版本的
Qwen waypoint 选择、历史世界坐标解析、全局 FMM、立即截断和真实机器人运动已经端到端跑通。

因为专项命令配置了 10 轮，第一次 BACKTRACK 完成后 controller 会根据截断后的 history 继续
请求模型。本次运行后续 decision 006 的新 NAVIGATE 因 cross-track 安全阈值而中止；它发生在
两次 BACKTRACK 都完成以后，不属于 BACKTRACK 执行失败，也不改变上述验证结论。

### 1.2 2026-07-26：BACKTRACK 对齐 `qwen_end2end`

本次修改把 Isaac 侧 BACKTRACK 的默认执行方式从“反向重放历史 FMM 路径”改为与
`lavira_code/vlnce_baselines/lavira_main_qwen_end2end.py` 相同的世界坐标目标重规划。
Qwen 仍然同时决定是否返回以及返回哪个零起始 waypoint；Isaac controller 不替模型选择
waypoint，只负责校验和执行。

#### 代码改动

- `goal_tracking/navigation_mapping.py`
  - `build_navigation_grid_map()` 现在支持当前 bbox 投影目标和历史世界坐标目标两种互斥输入。
  - 新增 `build_navigation_grid_map_for_world_goal()`，用于把历史 waypoint 的世界坐标放进
    当前四视图构建的 traversability map。
  - 历史 waypoint 表示机器人曾经站立的位置，不执行 bbox 表面目标使用的沿相机射线退让。
  - 如果历史坐标所在栅格不可通行，只允许在 `target_snap_max_m` 内吸附到最近的可通行栅格；
    找不到则把本次 BACKTRACK 判为不可达。
- `goal_tracking/lavira_episode.py`
  - 新增 `build_replanned_backtrack_execution_request()`。
  - 从 Qwen 返回的 `waypoint` 找到对应 history record，读取该 record 的
    `decision_world_pose[:2, 3]` 作为返回目标。
  - 使用机器人当前四视图重新建图、重新运行 FMM，再复用现有 follower 和 locomotion 执行。
  - 与 LaViRA `qwen_end2end` 一致：BACKTRACK 一旦通过校验并被接受，就立即执行
    `history = history[:waypoint + 1]`，不再等机器人实际到达后才截断。
  - `backtrack_execution.json` 现在记录策略、目标 waypoint、目标世界坐标、history 截断前后
    数量、FMM 路径以及最终到达或失败状态。
  - 到达后的 `_complete_backtrack()` 只确认执行结果，不再二次截断 history。
- `goal_tracking/config.py`
  - 新增 `--lavira_backtrack_strategy`。
  - 默认值为 `replan_world_goal`。
  - 可显式设置为 `stored_reverse`，复现实验用的旧安全路径反向重放行为。
- `tests/test_navigation_mapping.py`
  - 增加历史世界坐标目标的栅格选择测试。
- `tests/test_lavira_episode.py`
  - 增加 Qwen waypoint 对应 `decision_world_pose`、默认世界目标重规划、接受动作时立即截断
    history，以及旧 `stored_reverse` 策略测试。
- `tests/test_lavira_protocol.py`
  - 增加默认 BACKTRACK 策略配置测试。

默认执行流程现在是：

```text
Qwen 返回 BACKTRACK + waypoint
  -> 校验 waypoint 属于已提交 history
  -> 读取该 waypoint 决策时的机器人世界坐标
  -> 使用当前四视图更新固定坐标累计全局地图
  -> 在累计全局地图上重新运行 FMM
  -> 接受动作并立即截断 waypoint 后面的 history
  -> 沿新 FMM 路径返回
  -> 到达、切回 stand、稳定后记录 arrived
```

旧策略仍可这样运行：

```bash
--lavira_backtrack_strategy stored_reverse
```

#### 真实 Qwen + Isaac Sim 验证

本次运行目录：

```text
outputs/isaacsim_goal_tracking/lavira_offline/
run_20260726_180432_703105/robot_01_backtrack_world_replan_test_001
```

运行中默认 `replan_world_goal` 成功执行了两次 BACKTRACK：

```text
decision_003:
  Qwen action       = BACKTRACK
  Qwen waypoint     = 0
  replanned path    = 2.224m / 10 waypoints
  history           = 3 -> 1
  result            = arrived
  world target      = (1.151239, 5.248124)

decision_005:
  Qwen action       = BACKTRACK
  Qwen waypoint     = 0
  replanned path    = 0.370m
  history           = 2 -> 1
  result            = arrived
```

这证明以下链路已经真实跑通：

```text
Qwen 选择明确 waypoint
  -> Isaac 读取历史世界坐标
  -> 当时版本使用当前地图重新 FMM
  -> G1 返回该 waypoint
  -> history 在接受动作时按 LaViRA 语义截断
```

本轮最终失败不是 BACKTRACK 失败。`decision_008` 返回 NAVIGATE 后生成了 `2.784m` 路径，
超过普通 NAVIGATE/STOP 的默认 `2.5m` 安全上限，因此 controller 正确拒绝并进入 `FAILED`。

#### 为什么回到 waypoint 0 后 Qwen 没有返回 STOP

测试 instruction 要求机器人先移动再返回起点。Qwen 能够根据 history 返回
`BACKTRACK waypoint=0`，但 BACKTRACK 到达以后发送给下一轮 Qwen 的 schema v2 请求里目前
没有以下执行状态：

```text
last action = BACKTRACK
execution status = arrived
current waypoint = 0
waypoint 0 = starting position
distance to starting position
```

同时，接受 BACKTRACK 时立即截断 history 会删除出发后的 waypoint；机器人返回时的朝向也
可能与最初拍照朝向不同。因此下一轮 Qwen 只能看到当前四张 RGB、任务 instruction 和截断后的
history，无法可靠判断“已经完成出去再返回”这一时序任务，于是可能继续返回 NAVIGATE。

把 instruction 改成下面这样可以提高模型输出 STOP 的概率，但不能代替执行状态反馈：

```text
Move forward to explore the clear open floor. Then return to waypoint 0 using BACKTRACK.
Once you have returned to waypoint 0, the task is complete and you must output STOP immediately.
Do not start a new navigation action after returning to waypoint 0.
```

后续要可靠触发模型 STOP，需要同时修改客户端请求 schema 和远程 Qwen adapter prompt，在下一轮
请求中加入类似信息：

```json
{
  "last_execution": {
    "action": "BACKTRACK",
    "status": "ARRIVED",
    "target_waypoint": 0
  },
  "current_location": {
    "waypoint": 0,
    "is_starting_position": true
  }
}
```

当前代码尚未发送这部分字段，因此本次修改只完成 BACKTRACK 执行语义对齐，没有把
“回到起点后的任务完成状态”硬编码成自动 STOP。

#### 验证

本次修改后执行：

```bash
python -m compileall -q goal_tracking tests
python -m unittest discover -s tests
```

结果：

```text
Ran 55 tests
OK (skipped=1)
```

### 1.3 2026-07-23 完成内容

当日完成的不是单独一个 STOP，而是把单轮导航扩展成了有限多轮
`NAVIGATE / BACKTRACK / STOP` episode controller。

#### History

已经完成：

- 每个 decision 使用新的同步四方向 FrameBundle。
- `decision_index` 在同一 session 内单调递增，BACKTRACK 后也不会复用旧 observation id。
- NAVIGATE 做出决策时先创建 candidate waypoint。
- 只有 FMM 路径真正到达、切回 stand 并稳定后，candidate 才提交到 history。
- 执行失败、超时或安全中止的 candidate 不会进入服务器 history。
- 每个 waypoint 保存决策时位姿、方向、目标、bbox、模型分析、投影目标、安全目标和实际 FMM
  世界路径。
- `init_rgb` 固定使用该决策点的 forward 图；`dir_rgb` 使用模型选择方向的图。
- 所有历史点保留文字；只有最近四个 waypoint 携带两张原始历史图片，与 LaViRA 图片预算一致。
- 下一轮 multipart 会自动携带 history metadata 和对应图片。

真实服务器已经观察到：

```text
decision_000 -> history=0, images=4
decision_001 -> history=1, images=6
decision_002 -> history=2, images=8
decision_003 -> history=3, images=10
```

这证明多轮 history 不只是离线生成，真实 Qwen 服务已经接收并基于它返回后续动作。

#### BACKTRACK

已经完成：

- 严格读取模型返回的零起始 `waypoint`。
- 校验 waypoint 必须属于已经成功提交的 history。
- 默认读取目标 waypoint 的 `decision_world_pose`，在当前四视图深度地图上重新运行 FMM。
- 历史 waypoint 是机器人曾站立的坐标，不使用 bbox 表面目标的相机方向退让；不可通行时只允许
  在配置距离内吸附到附近 traversable cell。
- 继续复用同一个路径执行入口、pure-pursuit、G1 locomotion policy 和安全检查。
- 与 `lavira_code/qwen_end2end` 一致，在接受 BACKTRACK、启动运动前立即截断后续 history。
- 可用 `--lavira_backtrack_strategy stored_reverse` 保留原先的历史 FMM 路径反向策略作对照。
- 保存 `backtrack_execution.json`，记录策略、目标 waypoint、目标世界坐标、路径和执行状态。

旧 `stored_reverse` 策略曾由真实服务器与 Isaac Sim 成功验证一次：

```text
session              = robot_01_backtrack_execute_test_003
decision             = 002
action               = BACKTRACK
target waypoint      = 1
reverse path length  = 1.9615925547m
status               = arrived
completion step      = 3165
history              = 2 -> 2
```

本次返回 waypoint 1，即当时最新的已提交 waypoint，因此到达后 history 长度保持 `2` 是正确
结果，并不是没有执行 BACKTRACK。

旧 `stored_reverse` 策略有一个已知鲁棒性问题：历史反向路径起点是之前保存的理论终点，而
机器人到达容差和切换 stand 后可能产生约 `0.17m` 的实际偏差。后续两次运行分别出现：

```text
0.172m > 默认 0.150m
0.173m > 默认 0.150m
```

执行前保护因此正确拒绝了路径。默认 `replan_world_goal` 已消除对旧理论路径起点的依赖；
`--fmm_execute_start_tolerance_m 0.25` 只再用于复现实验或旧策略对照。

#### STOP

已经完成：

- STOP 使用与 NAVIGATE 相同的 direction、bbox、depth 投影、安全目标回退、地图和 FMM。
- 保持 LaViRA 默认 `15 cells × 0.05m = 0.75m` 的最终到达语义。
- 如果开始时已经在阈值内，不启动 locomotion。
- 如果尚未进入阈值，沿同一 FMM 路径执行最终接近。
- 进入阈值后停止 follower、命令归零、平滑切回 stand。
- stand 稳定后状态变为 `STOPPED`，不再发送模型请求。
- STOP 即使出现在有限决策预算的最后一轮也会执行。
- 安全中止、超时或路径失败不会被记录成 STOP 成功。
- 保存 `stop_execution.json`，并区分 `model_stop_completed` 与尚无 ground truth 的
  `task_success`。

今天完成两次 mock + Isaac Sim 实测：

```text
无移动 STOP：
  initial distance=0.991m, diagnostic threshold=5.0m
  -> 保持 stand -> STOP completed

真实最终接近：
  initial distance=0.991m, LaViRA threshold=0.75m
  -> FMM -> locomotion -> 移动约 0.24m
  -> threshold reached -> stand -> STOP completed
```

第二次已经验证 G1 的真实 locomotion 最终接近；尚未验证的是远程真实 Qwen 能否在正确语义
位置主动返回 STOP。

### 1.4 仍未完成

- 真实 Qwen 主动选择 STOP 的端到端验证；目前 STOP 运动执行使用 mock 强制触发。
- 真正“持续运行直到 STOP”的开放 episode。当前仍由
  `--lavira_history_max_decisions` 限制决策数，提高到较大值只能近似开放运行。
- 普通最后一轮响应仍是只读；BACKTRACK 如果恰好出现在最后一轮不会执行。STOP 已做终止动作
  例外。
- 路径卡住、FMM 不可达、cross-track 或单段超时后的自动恢复。目前会安全进入 FAILED，不会
  自动重新拍照请求模型。
- BACKTRACK 执行过程中的逐控制步动态障碍更新和在线重新规划；当前全局图在决策抓图时累计。
- Grounded-SAM semantic category 通道；当前只有 obstacle/explored/current/past。
- 异步 HTTP worker；当前请求模型时仿真主线程同步等待。
- 跨进程 episode/history 保存与恢复。
- ground-truth goal region、真正 task success、SPL、碰撞率和超时统计。

## 2. 目录结构

```text
scripts/isaacsim_goal_tracking/
├── README.md
├── config.yaml
├── isaacsim_path_follwing.py
├── mock_lavira_server.py
├── goal_tracking/
│   ├── camera.py
│   ├── config.py
│   ├── control.py
│   ├── fmm_planner.py
│   ├── frame_bundle.py
│   ├── lavira_episode.py
│   ├── lavira_offline.py
│   ├── lavira_protocol.py
│   ├── navigation_mapping.py
│   ├── path.py
│   ├── runners.py
│   └── target_projection.py
└── tests/
    ├── test_camera_geometry.py
    ├── test_fmm_planner.py
    ├── test_lavira_http_probe.py
    ├── test_lavira_protocol.py
    ├── test_navigation_mapping.py
    ├── test_path_follower.py
    └── test_target_projection.py
```

主程序文件名当前是 `isaacsim_path_follwing.py`，其中 `follwing` 是现有文件名，运行命令必须
保持这个拼写。

## 3. G1 仿真与 policy

### 3.1 三种运行模式

| 参数 | 功能 |
| --- | --- |
| `--mode stand` | 单独运行 lower-body stand policy |
| `--mode locomotion` | 单独运行 29DOF locomotion ONNX |
| `--mode switch` | stand/locomotion 平滑切换、键盘和路径执行 |

房间场景默认使用：

```text
/home/yile/scene/House/scene_047/mujoco/usd/scene_scene_047.usda
```

`--house` 只在运行时把训练地形替换成房间 USD，不修改源 USD。房间模式默认固定 reset pose、
关闭部署测试不需要的随机化，并关闭 episode 自动 reset。

`--mode switch` 当前只支持：

```text
--num_envs 1
```

### 3.2 当前 policy 文件

Locomotion：

```text
/home/yile/projects/unitree_rl_lab/deploy/robots/
g1_29dof_goal_tracking/policies/locomotion/exported/policy.onnx
```

它与 MuJoCo 使用的：

```text
/home/yile/projects/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/
velocity/model_33000_yawrew25_mujoco/exported/policy.onnx
```

二进制完全一致，SHA-256 为：

```text
dd10c680e132c5d1c3018972d96c7a4bf2ccdbec871b1c0a9c7a842c8596c3a2
```

Stand：

```text
/home/yile/projects/unitree_rl_lab/deploy/robots/
g1_29dof_goal_tracking/policies/stand/exported/policy.onnx
```

它与 `model_26999_lower_body_stand` 的部署模型一致，SHA-256 为：

```text
0d1a2a9296f6785c1d90dd852872c2e48df6193a58b84c6d7788609bbe2af666
```

### 3.3 控制链

Locomotion 输出完整 29 维 action。Stand 输出 12 维腿部 action，再按关节名称填回 29 维 action；
其余上半身关节保持默认 action offset。

policy 切换使用 smoothstep 混合：

- stand → locomotion：直接开始平滑混合。
- locomotion → stand：先把速度命令降为零。
- 等待安全接管窗口。
- 冻结当前 action。
- 再平滑混合到 stand。

路径跟踪器不直接控制关节，只生成机体系：

```text
[vx, vy, wz]
```

然后继续使用原 IsaacLab command manager、action manager 和 actuator stack。

### 3.4 坐标和键盘

速度命令是机器人机体系：

```text
vx > 0：机器人前方
vy > 0：机器人左方
wz > 0：逆时针旋转
```

默认出生角：

```text
--yaw 3.141592653589793
```

即机器人相对世界坐标旋转约 `180°`。因此机体系 `vx/vy` 不能与世界系
`root_lin_vel_w[x/y]` 直接逐项比较。

Switch 模式键盘：

| 按键 | 功能 |
| --- | --- |
| `1` | 请求平滑切换到 stand |
| `2` | 请求平滑切换到 locomotion |
| `3` | 跳过等待窗口，强制开始平滑切到 stand |
| `G` | 启动当前 waypoint 路径 |
| `W/S` | 增减 `vx` |
| `A/D` | 增减 `vy` |
| `Q/E` | 增减 `wz` |
| `X/L/Space` | 停止路径、速度归零并请求 stand |

Locomotion 固定速度测试必须加：

```text
--no-keyboard
```

否则默认键盘零命令会在下一周期覆盖命令行的 `--vx/--vy/--wz` 初值。

## 4. 四方向 RGB-D 相机

### 4.1 相机安装

四台相机是 IsaacLab 的真实 `Camera` sensor，固定挂在 G1 `torso_link` 下。它们通过 USD
父子关系随机器人运动，不会在每次抓图前手工覆盖世界位姿。

当前方向以 G1 torso 坐标为准：

| 语义视图 | torso 方向 | 默认相对位置（米） |
| --- | --- | --- |
| `forward` | `+X` | `(0.085, 0.000, 0.56)` |
| `left` | `+Y` | `(0.000, 0.085, 0.56)` |
| `behind` | `-X` | `(-0.085, 0.000, 0.56)` |
| `right` | `-Y` | `(0.000, -0.085, 0.56)` |

四台相机统一向下倾斜约 `12°`。

默认成像参数：

| 参数 | 默认值 |
| --- | ---: |
| RGB/depth 宽度 | `640` |
| RGB/depth 高度 | `480` |
| 水平 FOV | `79°` |
| near | `0.1m` |
| far | `5.0m` |

旧的单目 `head_camera` 挂载函数仍保留在 `camera.py`，但当前主运行链不会调用。

### 4.2 FrameBundle

一次 `FourViewCameraRig.capture()` 在同一个仿真 step 得到：

- 四路 `uint8` RGB，shape 为 `(480, 640, 3)`。
- 四路米制 `float32` optical-Z depth，shape 为 `(480, 640)`。
- 每台相机内参矩阵 `K`。
- 抓图时机器人位姿 `T_world_base`。
- 每台相机世界位姿 `T_world_camera_ros`。
- 每台相机相对机器人根节点外参 `T_base_camera`。
- `bundle_id`、`sim_step`、`timestamp`。
- 每台相机的 sensor frame id。

四张 RGB、四张 depth 和所有位姿严格绑定到同一个 FrameBundle。服务器返回 bbox 后，客户端
必须使用原 FrameBundle 的 depth，不能重新抓取另一帧。

### 4.3 相机调试

保存一次 RGB-D bundle：

```bash
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch --house --headless --device cuda:0 \
  --four_rgbd_cameras --camera_debug_save_once \
  --no-show_path --max_steps 12 --print_every 5
```

输出：

```text
outputs/isaacsim_goal_tracking/camera_probe/
└── run_<timestamp>/
    └── bundle_<id>_step_<step>/
        ├── forward_rgb.png
        ├── left_rgb.png
        ├── behind_rgb.png
        ├── right_rgb.png
        ├── <direction>_depth.npy
        ├── <direction>_depth_preview.png
        ├── montage.png
        └── metadata.json
```

`depth.npy` 是精确米制深度；彩色 depth preview 只供人眼查看，不能用于几何计算。

显示相机光心调试点：

```bash
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch --house --device cuda:0 \
  --four_rgbd_cameras --four_rgbd_debug_points \
  --no-four_rgbd_set_viewport --no-show_path
```

调试点颜色：

```text
forward：红
left：绿
behind：蓝
right：黄
```

正式模型输入不要开启调试点和路径标记，避免污染 RGB。

### 4.4 GUI viewport

`--four_rgbd_set_viewport` 默认开启，会尝试把 active viewport 切到
`lavira_camera_forward`。这只影响 GUI，不影响相机数组。

需要固定世界观察视角时使用：

```text
--no-four_rgbd_set_viewport
```

并在 Isaac Sim 中选择：

```text
Perspective
Follow Mode: World
Environment Index: 0
```

## 5. LaViRA/Qwen HTTP 接口

### 5.1 双方职责

Isaac 客户端负责：

- 同步抓取四方向 RGB-D。
- 管理 session、observation 和 instruction。
- 把 metadata 与原始 RGB PNG 组成 multipart 请求。
- 校验响应。
- 在本机完成 depth 投影、地图、FMM 和机器人控制。

模型服务器负责：

- 提供统一 endpoint。
- 解析 metadata 和 RGB PNG。
- 按 history 和四个当前视图构造 Qwen 输入。
- 调用本地部署的统一 Qwen/LaViRA。
- 保存模型原始输出。
- 把模型输出归一化成 schema v2。
- 回显请求的 session 和 observation ID。

服务器不需要接收 depth、K、相机外参，不需要运行 FMM，也不控制机器人。

### 5.2 Endpoint

```text
POST /v1/lavira/decision
Content-Type: multipart/form-data
Accept: application/json
```

当前客户端：

- 没有 Authorization header。
- 没有自动重试。
- 默认 timeout 为 90 秒。
- 请求是同步阻塞式。
- 最大响应 body 为 1 MiB。

### 5.3 第一轮请求

```json
{
  "schema_version": 2,
  "request_type": "end2end_decision",
  "session_id": "robot_01_task_001",
  "observation_id": "robot_01_task_001_decision_000",
  "bundle_id": 0,
  "decision_index": 0,
  "sim_step": 5,
  "timestamp": 0.1,
  "instruction": "Go straight, then turn left and stop by the bed.",
  "image_width": 640,
  "image_height": 480,
  "history": [],
  "current_panorama": {
    "forward": "current_forward",
    "left": "current_left",
    "behind": "current_behind",
    "right": "current_right"
  }
}
```

同一个 multipart 请求包含：

```text
metadata
current_forward
current_left
current_behind
current_right
```

图片是二进制 PNG，不是放在 JSON 中的 base64。所有图片为未画 bbox、文字、路径线和调试点的
原始 RGB。

服务器必须通过 metadata 中的 field name 查找图片，不能依赖 multipart part 顺序。

当前四视图 prompt 顺序固定为：

```text
forward -> left -> behind -> right
```

### 5.4 history 契约

协议和 `--lavira_history_probe` 已支持自动生成第一个 history point。默认一次性 probe 仍发送
空 history。

一个 history entry：

```json
{
  "waypoint_id": 0,
  "step": 35,
  "turn_action": "turn forward",
  "description": "doorway",
  "init_image_field": "history_0_init",
  "dir_image_field": "history_0_dir"
}
```

含义：

| 字段 | 含义 |
| --- | --- |
| `waypoint_id` | 从 0 开始、连续递增，也是 BACKTRACK 返回值 |
| `step` | 创建该 waypoint 时的仿真 step |
| `turn_action` | `turn forward/left/behind/right` |
| `description` | 当时模型返回的 target |
| `init_image_field` | 刚到决策点时的前方图 |
| `dir_image_field` | 当时选中方向的图 |

history 只允许包含已经完成的 waypoint。当前尚未执行完成的决策点不能进入 history，也不能成为
BACKTRACK 目标。

这对应原版 `lavira_code` 的：

```python
history_targets = visited_targets[:-1]
```

当前决策点先作为本地 candidate 保存；只有 FMM 路径完成、切回 stand 并稳定后才提交。路径规划
失败、超时、cross-track 中止或倾角中止都不会产生服务器 history。

图片预算：

- 所有 history waypoint 都保留文字。
- 最近最多 4 个 waypoint 携带 init/dir 两张图。
- 更早的 waypoint 必须省略两个 image field。
- 当前四张图片始终发送。

```text
total_images = 4 + 2 * min(history_count, 4)
```

标准请求最多 12 张图。

服务器构造 Qwen 输入的顺序：

```text
instruction
  -> waypoint 0 init image
  -> waypoint 0 Action
  -> waypoint 0 dir image
  -> waypoint 0 Navigate to description
  -> 后续历史 waypoint
  -> 当前 FORWARD
  -> 当前 LEFT
  -> 当前 BEHIND
  -> 当前 RIGHT
  -> JSON 输出要求
```

`session_id/observation_id/bundle_id/sim_step/timestamp` 只用于配对和本机状态管理，不应作为导航
语义写入 Qwen prompt。

### 5.5 统一响应

HTTP 成功响应固定包含：

```text
schema_version
response_type
session_id
observation_id
action
direction
target
bbox_2d
waypoint
progress_analysis
reasoning
```

NAVIGATE 示例：

```json
{
  "schema_version": 2,
  "response_type": "end2end_decision",
  "session_id": "robot_01_task_001",
  "observation_id": "robot_01_task_001_decision_000",
  "action": "NAVIGATE",
  "direction": "left",
  "target": "doorway",
  "bbox_2d": [180, 120, 420, 460],
  "waypoint": null,
  "progress_analysis": "The route continues through the doorway.",
  "reasoning": "The left view contains the doorway."
}
```

STOP 使用同样的 direction、target 和 bbox，`waypoint=null`。

BACKTRACK 归一化响应：

```json
{
  "schema_version": 2,
  "response_type": "end2end_decision",
  "session_id": "robot_01_task_001",
  "observation_id": "robot_01_task_001_decision_003",
  "action": "BACKTRACK",
  "direction": null,
  "target": null,
  "bbox_2d": null,
  "waypoint": 1,
  "progress_analysis": "The current route is unproductive.",
  "reasoning": "Return to waypoint 1."
}
```

Qwen 原始输出可以让三种 action 都使用完整字段模板，但服务器 adapter 必须归一化：

- NAVIGATE/STOP：使用 direction、target、bbox，waypoint 设为 null。
- BACKTRACK：只使用 waypoint，direction、target、bbox 设为 null。

### 5.6 bbox 规则

```text
bbox_2d = [x1, y1, x2, y2]
```

- 原点在图片左上角。
- x 向右，y 向下。
- 是像素坐标，不是 0~1 归一化坐标。
- 不是 `[x, y, width, height]`。
- 属于 `direction` 指定的当前图片。
- 必须满足 `x1 < x2`、`y1 < y2`。

合法范围：

```text
0 <= x1 < x2 <= image_width
0 <= y1 < y2 <= image_height
```

`640x480` 图片允许 `x2=640`、`y2=480` 作为排他边界；客户端索引数组时会安全裁剪到
`639/479`。

### 5.7 客户端严格校验

客户端会拒绝：

- schema/version/type 错误。
- session 或 observation ID 不匹配。
- action 不属于 NAVIGATE/BACKTRACK/STOP。
- direction 不属于四个方向。
- NAVIGATE/STOP 缺少 target 或合法 bbox。
- NAVIGATE/STOP waypoint 不为 null。
- BACKTRACK waypoint 越界。
- BACKTRACK direction/target/bbox 不为 null。
- bbox 越界、空框或包含非有限数。
- metadata 引用图片缺失、重复或存在未引用图片。
- 响应不是 UTF-8 JSON object 或超过 1 MiB。

服务端请求错误应返回 HTTP 400，模型/GPU/内部错误应返回 HTTP 500，不要用字段缺失的 HTTP 200
冒充成功响应。

## 6. bbox + depth 三维投影

NAVIGATE/STOP 通过校验后，客户端从 `direction` 对应的同一 FrameBundle 取 depth。

算法：

1. 选择 bbox 底边中点。
2. 依次尝试 `3x3/5x5/7x7/9x9` 邻域。
3. 使用第一个存在有效深度的邻域。
4. 对有效深度取中位数。
5. 用真实 `K` 反投影到 ROS optical camera 坐标。
6. 用 `T_base_camera` 得到机器人坐标。
7. 用 `T_world_camera_ros` 得到世界坐标。
8. 检查 `T_world_base @ T_base_camera` 的变换一致性。

没有有效深度时明确失败，不生成固定的虚假远距离目标。

输出点是模型框选目标表面的语义点，不是机器人可以直接站立的无碰撞点。

输出：

```text
target_projection.json
target_projection_rgb.png
target_projection_depth_m.npy
target_projection_depth_preview.png
```

## 7. 四路 depth 当前观测与累计全局地图

增加：

```text
--lavira_local_map_probe
```

后，客户端先把本轮四路 depth 合并为以机器人为中心的当前观测栅格。bounded episode 默认
再由 `lavira_global_mapping.py` 把它变换到 episode 固定世界栅格并累计；一次性 probe 仍会
保留当前观测文件，便于分别检查投影错误和融合错误。

默认约定：

| 参数 | 默认值 |
| --- | ---: |
| 分辨率 | `0.05m/cell` |
| 物理范围 | `24m x 24m` |
| shape | `480 x 480` |
| depth stride | `4` |
| G1 水平障碍膨胀 | 约 `0.35m` |
| 目标回退步长 | `0.1m` |
| 最近安全点搜索半径 | 约 `1.0m` |

通道：

```text
observed
free
occupied
inflated_obstacle
traversable
```

当前观测规则：

- 未观测区域不可通行。
- 障碍按照 G1 水平包络膨胀。
- traversable 只保留与机器人当前位置连通的区域。
- bbox 原始目标不可通行时，先沿相机射线向机器人方向回退。
- 回退失败后才搜索有限距离内的最近 traversable 点。
- 没有安全目标时停止规划。

当前观测输出：

```text
navigation_map.json
navigation_map.npz
navigation_map.png
```

bounded episode 默认额外生成固定坐标累计地图。它不是读取完整 USD 的先验地图，而是机器人
从第一轮开始实际看到的 RGB-D 观测历史，因此能适配未来更换场景：

```text
第一帧实际 root XY
  -> 确定一次 full map 原点
  -> 后续每轮局部 observed/occupied 转换到 full-map cell
  -> max 融合，旧区域不丢失
  -> 根据当前 root cell 生成 connected traversable
  -> NAVIGATE / BACKTRACK / STOP 共用该 FMM 输入
```

累计地图输出：

```text
lavira_global_map.json
lavira_global_map.npz
lavira_global_map.png
```

## 8. FMM 路径规划

依赖：

```text
scikit-fmm 2025.06.23
```

验证：

```bash
conda activate isaacsim
python -c "import skfmm; print(skfmm.__version__)"
```

增加：

```text
--lavira_fmm_probe
```

后会使用 `navigation_map.traversable`、机器人栅格和安全目标运行：

- masked traversable level set。
- `skfmm.distance(dx=1)`。
- LaViRA 风格 `step_size=5` short-term-goal 圆环。
- 沿距离场严格单调下降提取路径。
- 8 邻域移动。
- 禁止从两个障碍角之间对角穿越。
- 路径简化后的直线 traversability 检查。
- 约 `0.25m` 间距的世界坐标 waypoint。
- 根据下一段路径切线生成 yaw。
- 很近目标的单独终止处理。
- 不可达时明确失败，不伪造直线路径。

输出：

```text
fmm_plan.json
fmm_distance.npy
fmm_path.png
```

失败时保存：

```text
fmm_plan_error.json
```

## 9. 一次性 FMM locomotion 执行

只有显式增加：

```text
--lavira_execute_fmm_path
```

才会让机器人执行本轮 FMM 路径。

执行链：

```text
FMMPlan.waypoints_world_xy
  -> Waypoint(x, y, yaw)
  -> WaypointPathFollower
  -> SwitchCommandController
  -> IsaacLab base_velocity
  -> locomotion ONNX
  -> 原 29 维 action/actuator stack
```

默认安全参数：

| 项目 | 默认值 |
| --- | ---: |
| 抓图后最大起点漂移 | `0.15m` |
| 最大可执行路径长度 | `2.0m` |
| FMM lookahead | `0.20m` |
| 最大 `vx` | `0.20m/s` |
| 最大 `vy` | `0.12m/s` |
| 最大 `wz` | `0.25rad/s` |
| 最大 cross-track error | `0.40m` |
| 最大身体倾角 | `0.50rad` |

路径被拒绝或运动中触发保护时，command 归零并请求 stand。正常到达也会归零并切回 stand。

已实际验证的一次执行：

```text
FMM path length = 1.579m
waypoints       = 8
start drift     = 0.000m
goal dist       = 0.120m
yaw error       = -0.001rad
cross track     = 0.008m
final mode      = stand
```

### 9.1 有限多轮 history probe

`--lavira_history_probe` 复用同一套决策、投影、地图、FMM 和路径执行代码，按原版
`lavira_main.py::rollout()` 的 `visited_targets[:-1]` 语义运行：

```text
decision_000，history=[]
  -> NAVIGATE
  -> 创建候选 waypoint 0
  -> 执行 FMM 路径
  -> 到达并切回 stand
  -> 稳定 0.8s
  -> waypoint 0 标记 arrived 并提交

decision_001，history=[waypoint 0]
  -> NAVIGATE
  -> 执行并提交 waypoint 1

decision_002，history=[waypoint 0, waypoint 1]
  -> 保存并校验第三次响应
  -> 保持 stand，不执行第三条路径
```

每个候选 waypoint 的 `decision_world_pose`、`init_rgb` 和 `dir_rgb` 都来自该 decision 的同一个
FrameBundle。`init_rgb` 固定取 forward，`dir_rgb` 取模型 `direction` 指定的视图。到达时的
sim step 只记录执行结果，不会错误替换拍摄图片时的决策点位姿。

默认 `--lavira_history_max_decisions 3`。前 `N-1` 个 NAVIGATE/BACKTRACK 可以执行，第 `N`
个普通响应只读。收到 BACKTRACK 时，程序默认读取目标 waypoint 的历史世界坐标，在累计
全局 traversability map 上重新运行 FMM，并在接受动作时立即把 history 截断到该 waypoint；
decision index 仍保持单调递增。BACKTRACK 路径上限默认为 `6.0m`，可用
`--lavira_backtrack_max_path_m` 调整。旧反向路径策略可用
`--lavira_backtrack_strategy stored_reverse` 显式开启。

STOP 严格复用 NAVIGATE 已有的 bbox 底边中心投影、安全目标回退、累计全局地图、FMM 和
pure-pursuit，不建立第二套规划器。原版 LaViRA 的到达阈值为 `15` 个地图格，默认地图分辨率
为 `0.05m/cell`，因此 Isaac 默认使用：

```text
15 cells × 0.05 m/cell = 0.75 m
```

机器人进入 STOP 的 FMM 安全目标 `0.75m` 范围后会停止 follower、清零 `vx/vy/wz`、
平滑切回 stand，并把 episode 设为 `STOPPED`，不再发送模型请求。可用
`--lavira_stop_reached_threshold_m` 调整该阈值。该状态只证明模型 STOP 已正确执行，
不等同于有 simulator ground truth 支持的 VLN task success。

2026-07-23 已完成两次 Isaac Sim mock STOP 实测：

```text
测试 A：threshold=5.0m，initial distance=0.991m
  -> 不启动 locomotion
  -> 保持 stand
  -> STOP completed

测试 B：threshold=0.75m，initial distance=0.991m
  -> 接受 1.070m FMM path
  -> stand -> locomotion
  -> 实际执行约 0.24m 最终接近
  -> distance < 0.75m 时停止 follower
  -> locomotion -> stand
  -> STOP completed
```

第二次测试证明 STOP 不只是 JSON 解析或静态状态切换，而是已经真实通过现有 G1 locomotion
policy 执行了一段 FMM 路径。两次测试使用 mock 强制返回 STOP，因此尚未证明远程真实 Qwen
能在正确语义位置主动选择 STOP。

当前实现使用同步 HTTP：请求期间仿真主线程会等待服务器，机器人在请求前保持 stand。异步网络
worker 仍属于后续工作。

## 10. 运行方法

所有命令从项目根目录运行：

```bash
cd ~/projects/unitree-g1-isaaclab-project
conda activate isaacsim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
```

### 10.1 Stand

```bash
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode stand --house --device cuda:0 \
  --spawn 1.15 5.25 0.8 --real-time
```

### 10.2 Locomotion 固定前进

```bash
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode locomotion --house --device cuda:0 \
  --spawn 1.15 5.25 0.8 \
  --vx 0.25 --vy 0.0 --wz 0.0 \
  --no-keyboard --real-time
```

纯侧向测试：

```bash
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode locomotion --house --device cuda:0 \
  --spawn 1.15 5.25 0.8 \
  --vx 0.0 --vy 0.25 --wz 0.0 \
  --no-keyboard --real-time
```

### 10.3 Switch 和预设路径

```bash
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch --house --device cuda:0 \
  --spawn 2.45 1.15 0.8 --real-time
```

按 `G` 启动预设 waypoint 路径，也可以：

```text
--start_path_on_enter
```

自定义世界坐标 waypoint：

```text
--path_waypoints "x0,y0,yaw0;x1,y1,yaw1;x2,y2,yaw2"
```

### 10.4 本机 mock server

终端 1：

```bash
python scripts/isaacsim_goal_tracking/mock_lavira_server.py \
  --host 127.0.0.1 --port 8765 \
  --action NAVIGATE --direction left --target doorway \
  --bbox 100 80 540 470
```

终端 2：

```bash
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch --house --device cuda:0 \
  --four_rgbd_cameras \
  --lavira_decision_probe \
  --lavira_local_map_probe \
  --lavira_fmm_probe \
  --instruction "Go straight, then turn left and stop by the bed." \
  --no-show_path
```

没有 `--lavira_execute_fmm_path` 时只保存和检查结果，不移动机器人。

### 10.5 远程 Qwen SSH 隧道

单独终端保持运行：

```bash
ssh -N \
  -L 127.0.0.1:18765:127.0.0.1:8765 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  wangchu@131.159.60.188
```

本机 URL：

```text
http://127.0.0.1:18765/v1/lavira/decision
```

### 10.6 真实模型只读 probe

```bash
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch --house --device cuda:0 \
  --spawn 1.15 5.25 0.8 \
  --four_rgbd_cameras \
  --lavira_decision_probe \
  --lavira_local_map_probe \
  --lavira_fmm_probe \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 90 \
  --lavira_session_id "robot_01_open_room_test_001" \
  --instruction "Navigate toward the bed." \
  --real-time --no-show_path
```

### 10.7 真实模型 + FMM + G1 运动

```bash
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch --house --device cuda:0 \
  --spawn 1.15 5.25 0.8 \
  --four_rgbd_cameras \
  --lavira_decision_probe \
  --lavira_local_map_probe \
  --lavira_fmm_probe \
  --lavira_execute_fmm_path \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 90 \
  --lavira_session_id "robot_01_open_room_fmm_move_001" \
  --instruction "Navigate toward the bed." \
  --real-time --no-show_path
```

重要：完整运动测试不要增加：

```text
--disable_fabric
```

已经确认该参数会造成 PhysX tensor 状态与 USD/Viewport 可视状态不同步，表现为 follower 日志
显示机器人已经移动或到达，但 GUI 中机器人模型看起来没有平移。删除后可视运动恢复正常。

### 10.8 有限多轮真实 history 闭环

保持 SSH 隧道运行后执行：

```bash
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch --house --device cuda:0 \
  --spawn 1.15 5.25 0.8 \
  --four_rgbd_cameras \
  --lavira_history_probe \
  --lavira_history_max_decisions 3 \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 90 \
  --lavira_session_id "robot_01_history_test_001" \
  --instruction "Go through the doorway, then turn left and stop near the bed." \
  --real-time --no-show_path
```

这个模式会自动启用本地 target projection、occupancy map、FMM 和路径执行，不需要再写
四个底层 probe/execute flag。默认允许前两次普通 NAVIGATE/BACKTRACK 执行，第三次普通
模型响应只验证和保存；任意一轮 STOP 仍会按终止动作完成最终接近。
该模式还会自动关闭 projection USD marker，防止 decision 0 的调试球出现在 decision 1 原始
RGB 中。

成功标志：

```text
[LAVIRA EPISODE] decision_000 accepted and path started
[LAVIRA EPISODE] decision_000 path reached
[LAVIRA EPISODE] committed history waypoint: waypoint=0
[LAVIRA EPISODE] sending decision request: decision=001 history=1 images=6
[LAVIRA EPISODE] decision_001 accepted and path started
[LAVIRA EPISODE] committed history waypoint: waypoint=1
[LAVIRA EPISODE] sending decision request: decision=002 history=2 images=8
[LAVIRA EPISODE] decision_002 response valid
```

### 10.9 确定性 STOP 测试

为了把“Isaac 客户端是否能正确执行 STOP”和“真实 Qwen 是否会判断 STOP”分开验证，可以先
使用本机 mock 强制返回 STOP。

终端一：

```bash
cd ~/projects/unitree-g1-isaaclab-project
conda activate isaacsim

python scripts/isaacsim_goal_tracking/mock_lavira_server.py \
  --host 127.0.0.1 \
  --port 18766 \
  --action STOP \
  --direction left \
  --target "bed" \
  --bbox 0 0 504 475
```

终端二：

```bash
cd ~/projects/unitree-g1-isaaclab-project
conda activate isaacsim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch \
  --house \
  --device cuda:0 \
  --spawn 1.15 5.25 0.8 \
  --four_rgbd_cameras \
  --lavira_history_probe \
  --lavira_history_max_decisions 2 \
  --lavira_history_execution_timeout 30 \
  --lavira_stop_reached_threshold_m 0.75 \
  --fmm_execute_start_tolerance_m 0.25 \
  --fmm_execute_max_path_m 2.5 \
  --lavira_server_url "http://127.0.0.1:18766/v1/lavira/decision" \
  --lavira_timeout 30 \
  --lavira_session_id "robot_01_forced_stop_move_test_001" \
  --instruction "Stop now." \
  --max_steps 2000 \
  --real-time \
  --no-show_path
```

这组已验证配置会先生成同一条 FMM 路径，再在机器人与安全目标距离小于 `0.75m` 时提前结束
路径。成功日志为：

```text
[LAVIRA EPISODE] decision_000 STOP final approach started ...
[INFO] Begin smooth switch to locomotion.
[INFO] Path follower stopped (LaViRA STOP target threshold reached).
[LAVIRA EPISODE] STOP final-approach threshold reached ...
[INFO] Switch complete: stand.
[LAVIRA EPISODE] STOP completed ...
```

若只想验证无移动 STOP 状态切换，可以临时使用
`--lavira_stop_reached_threshold_m 5.0`。这只是诊断值；正常 LaViRA 测试必须恢复默认
`0.75m`。

### 10.10 真实 Qwen + 累计全局地图 BACKTRACK 专项测试

这个测试通过 history 数量向 Qwen 明确规定动作顺序：

```text
history=0 -> NAVIGATE，创建 waypoint 0
history=1 -> NAVIGATE，创建 waypoint 1
history>=2 -> BACKTRACK waypoint 0
```

保持 10.5 的 SSH 隧道运行，然后执行：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
conda activate isaacsim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch \
  --house \
  --device cuda:0 \
  --spawn 1.15 5.25 0.8 \
  --four_rgbd_cameras \
  --lavira_history_probe \
  --lavira_history_max_decisions 10 \
  --lavira_history_execution_timeout 45 \
  --lavira_backtrack_strategy replan_world_goal \
  --lavira_backtrack_max_path_m 6.0 \
  --fmm_execute_max_path_m 2.5 \
  --nav_map_mode lavira_compatible_global \
  --nav_map_resolution_m 0.05 \
  --nav_map_size_m 24.0 \
  --nav_global_origin_mode spawn_center \
  --nav_global_downscaling 2 \
  --nav_global_center_reset_steps 25 \
  --nav_global_unknown_space_policy blocked \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 90 \
  --lavira_session_id "robot_01_global_backtrack_test_003" \
  --instruction "This is a controller verification task. Create exactly two navigation waypoints before returning. If history contains 0 waypoints, output NAVIGATE toward a visible clear traversable floor region away from the starting position. If history contains 1 waypoint, output NAVIGATE again toward another visible clear traversable floor region farther from the starting position. As soon as history contains 2 or more waypoints, output BACKTRACK with waypoint 0. Do not output STOP before issuing BACKTRACK to waypoint 0." \
  --max_steps 10000 \
  --real-time \
  --no-show_path
```

这里使用 `--lavira_history_max_decisions 10`，decision 000–008 都属于可执行轮，只有
decision 009 是最后一次只读响应。这样即使 Qwen 没有在 history=2 时立刻遵循指令，而是晚几轮
才返回 BACKTRACK，仍有足够大的执行窗口。看到 `BACKTRACK completed` 后可以直接按
`Ctrl+C` 结束专项测试；因为 BACKTRACK 接受时会把 history 截断到 waypoint 0，继续运行时
Qwen 可能根据缩短后的 history 再次开始 NAVIGATE。

预期关键日志：

```text
[LAVIRA EPISODE] committed history waypoint: waypoint=0
[LAVIRA EPISODE] committed history waypoint: waypoint=1
[LAVIRA] Valid navigation response: action=BACKTRACK waypoint=0
[LAVIRA GLOBAL MAP] fused decision observation: updates=N origin=[...]
[LAVIRA EPISODE] decision_NNN BACKTRACK accepted:
waypoint=0 strategy=replan_world_goal history=M->1
[LAVIRA EPISODE] decision_NNN BACKTRACK path reached
[LAVIRA EPISODE] BACKTRACK completed: waypoint=0 history=M->1
```

必须核对实际返回 BACKTRACK 的 `decision_NNN` 目录：

```text
response.json
response_interpretation.json
lavira_global_map.json
lavira_global_map.npz
global_planning/fmm_plan.json
global_planning/fmm_path.png
backtrack_execution.json
```

`backtrack_execution.json` 的合格条件：

```text
strategy = replan_world_goal
target_waypoint = 0
history_count_before >= 2
history_count_after = 1
status = arrived
```

同时，`global_planning/fmm_plan.json` 的 `goal_world_xy` 应接近
waypoint 0 的 `history_commit.json -> decision_world_pose[:2,3]`，而不是反向重放旧路径。

### 10.11 在线地图、collision map 和周期 FMM 测试

先保持 10.10 的服务器和 SSH 隧道运行。在同一条仿真命令中增加：

```bash
--lavira_online_navigation \
--lavira_online_mapping_interval_s 1.0 \
--lavira_online_replan_interval_s 1.0 \
--lavira_collision_command_speed_m_s 0.12 \
--lavira_collision_window_s 0.75 \
--lavira_collision_min_progress_m 0.04 \
--lavira_collision_mark_distance_m 0.45 \
--lavira_collision_mark_radius_m 0.15
```

同时必须保留：

```bash
--nav_map_mode lavira_compatible_global \
--lavira_backtrack_strategy replan_world_goal
```

预期正常路径至少出现：

```text
[LAVIRA ONLINE] fused execution observation: bundle=...
[LAVIRA ONLINE] Hot-swapped FMM path: ...
[LAVIRA ONLINE] replanned unchanged goal: action=NAVIGATE|BACKTRACK|STOP ...
```

只有持续平移命令没有产生足够 XY 进度时才应出现：

```text
[LAVIRA ONLINE] commanded non-progress marked collision: ...
```

检查执行 decision 目录下的 `online_navigation.json`：周期重规划前后的
`goal_world_xy` 必须不变，`map_update_count` 和 `replan_count` 应递增；发生碰撞时
`collision_count` 和 NPZ 中的 `collision_map` 非零。在线更新不应产生额外 Qwen 请求，
也不应改变本轮 history 数量。

## 11. 一次请求的输出文件

```text
outputs/isaacsim_goal_tracking/lavira_offline/
└── run_<timestamp>/
    └── <session_id>/
        └── <observation_id>/
            ├── metadata.json
            ├── current_forward.png
            ├── current_left.png
            ├── current_behind.png
            ├── current_right.png
            ├── response.json
            ├── response_interpretation.json
            ├── target_projection.json
            ├── target_projection_rgb.png
            ├── target_projection_depth_m.npy
            ├── target_projection_depth_preview.png
            ├── navigation_map.json
            ├── navigation_map.npz
            ├── navigation_map.png
            ├── fmm_plan.json
            ├── fmm_distance.npy
            ├── fmm_path.png
            ├── history_commit.json
            ├── backtrack_execution.json
            ├── stop_execution.json
            ├── online_navigation.json
            ├── online_latest/
            │   ├── lavira_global_map.json
            │   ├── lavira_global_map.npz
            │   ├── lavira_global_map.png
            │   ├── fmm_plan.json
            │   ├── fmm_distance.npy
            │   └── fmm_path.png
            └── bounded_episode_status.json
```

错误会写入对应的：

```text
target_projection_error.json
navigation_map_error.json
fmm_plan_error.json
```

所有文件通过同一个 `observation_id` 关联，可以审计本轮输入、模型响应、三维投影、地图和路径。
`history_commit.json` 只在本 decision 真正到达并稳定后生成。后续 decision 目录会携带此前
已完成 waypoint 的 history 图片。BACKTRACK 请求、重规划路径和完成/失败状态保存在
`backtrack_execution.json`。STOP 目标、阈值、实际停止距离、stand 状态和完成/失败结果保存在
`stop_execution.json`。终止只读响应、STOP 或失败的控制器状态保存在
`bounded_episode_status.json`。累计 full/local/one-step 地图保存在
`lavira_global_map.json/.npz/.png`，真正交给执行器的全局 FMM 结果保存在
`global_planning/`，不会覆盖同轮当前观测的局部 FMM 调试文件。开启在线闭环后，
`online_navigation.json` 保存执行期事件，`online_latest/` 始终保存最新一次在线地图和路径。

## 12. 关键代码

| 文件 | 作用 |
| --- | --- |
| `isaacsim_path_follwing.py` | 程序入口，装配环境、相机和 runner |
| `goal_tracking/config.py` | 默认资源路径和所有 CLI 参数 |
| `goal_tracking/runners.py` | stand、locomotion、switch、FMM 启动和在线路径热替换接入 |
| `goal_tracking/control.py` | 命令写入、速度斜坡、stand/locomotion 状态机 |
| `goal_tracking/path.py` | waypoint、pure-pursuit、动态/在线热替换 FMM 路径和安全中止 |
| `goal_tracking/camera.py` | 四相机固定安装、方向、俯角和调试显示 |
| `goal_tracking/frame_bundle.py` | 同步 RGB-D、内参、外参和位姿 |
| `goal_tracking/lavira_protocol.py` | schema v2 请求、history、响应和严格校验 |
| `goal_tracking/lavira_offline.py` | 可复用任意 decision/history 的 multipart、HTTP、落盘和单轮流程 |
| `goal_tracking/lavira_episode.py` | waypoint/history、活动世界目标、在线 collision/FMM 和有限多轮状态机 |
| `goal_tracking/lavira_global_mapping.py` | 固定世界原点、full/local 通道、max 融合、collision mask 和全局 FMM 输入 |
| `goal_tracking/target_projection.py` | bbox-depth 采样和三坐标系投影 |
| `goal_tracking/navigation_mapping.py` | 四路 depth 栅格、障碍膨胀和安全目标 |
| `goal_tracking/fmm_planner.py` | FMM 距离场、短期目标、路径和世界 waypoint |
| `mock_lavira_server.py` | 本机测试服务器 |

## 13. 测试

运行：

```bash
python -m unittest discover \
  -s scripts/isaacsim_goal_tracking/tests -v
```

最近验证结果：

```text
Ran 64 tests
OK (skipped=1)
```

跳过的是执行沙箱不允许 localhost socket 的真实 socket 测试。其他测试覆盖：

- 四相机方向、位置、俯角和坐标变换。
- 请求/响应 schema 和 ID。
- history 图片预算、BACKTRACK waypoint、世界坐标重规划和旧多段路径反向策略。
- candidate waypoint、两段到达提交、第三轮 history 和路径失败不提交。
- 有限决策边界、BACKTRACK 接受时 history 截断。
- STOP 复用 FMM、`0.75m` 阈值提前停止、稳定 stand、最终轮 STOP 仍执行。
- multipart PNG 生成与解析。
- bbox-depth 投影。
- 地图坐标、障碍膨胀、目标回退。
- 全局原点固定、跨帧 obstacle/explored 累计、current/past 通道、manual origin 和越界报错。
- 稳定世界目标与 collision mask 共用全局 FMM 网格。
- FMM 绕障、不可达和路径安全。
- 在线命令无进度碰撞窗口、周期地图融合、同一世界目标 FMM 重规划。
- FMM 路径执行准备、运动中热替换、起点漂移、超长、cross-track 和倾角中止。

## 14. 已知限制

目前尚未实现：

- 不阻塞仿真主线程的异步网络 worker。
- 跨进程退出/重启后的 episode 和 history 恢复。
- Grounded-SAM semantic category 通道；当前累计的是 LaViRA 前四个几何/位置通道。
- 在线地图目前由主线程按低频 navigation tick 同步抓取，尚未移动到独立建图 worker。
- 完整 VLN episode success、SPL、碰撞率和超时评估。

默认 `--lavira_decision_probe` 仍固定发送：

```json
{
  "decision_index": 0,
  "history": []
}
```

`--lavira_history_probe` 已是可配置的有限多轮 NAVIGATE/BACKTRACK/STOP controller，但
普通最后一个响应仍只读，并且尚无 ground-truth success、SPL 和跨进程恢复，因此仍不是完整
的 VLN 评测框架。

## 15. 下一步

真实服务器早期 history 测试曾记录：

```text
decision_000 -> NAVIGATE，执行并提交 waypoint 0
decision_001 -> history=1, images=6，NAVIGATE 并提交 waypoint 1
decision_002 -> history=2, images=8，NAVIGATE 并提交 waypoint 2
decision_003 -> history=3, images=10，返回 BACKTRACK waypoint=1
```

上述 `decision_003` 当时恰好是有限测试的最后一个普通响应，因此该次 BACKTRACK 只验证和
保存。另一次 `robot_01_backtrack_execute_test_003` 已在旧 `stored_reverse` 策略下真实执行
BACKTRACK waypoint 1，沿 `1.962m` 反向路径在 step 3165 到达。当前默认策略已改为按该
waypoint 的世界坐标在累计全局地图重新 FMM。

此后已经进一步完成：

```text
robot_01_global_map_test_001：
  累计全局地图两次 NAVIGATE
  真实 Qwen 主动 STOP
  STOPPED + robot standing

robot_01_global_backtrack_test_003：
  真实 Qwen 两次 BACKTRACK waypoint 0
  默认 replan_world_goal
  累计全局地图 FMM
  history 分别 3->1、2->1
  两次 status=arrived
```

BACKTRACK 和 STOP 都已按原版 LaViRA action 语义接入，并在累计全局地图模式下真实完成。
接下来需要：

```text
BACKTRACK：使用数米级返回距离进一步压力测试累计地图遮挡、未知区域和长路径执行
MAPPING：把当前“每次决策更新”扩展为运动过程按步更新，并加入 semantic category 通道
EPISODE：支持持续运行直到 STOP，并为卡住/不可达增加重新决策恢复
评测：加入 ground-truth goal region、success、SPL 和碰撞统计
```


测试过的代码：


ssh -N \
  -L 127.0.0.1:18765:127.0.0.1:8765 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  wangchu@131.159.60.188


cd /home/yile/projects/unitree-g1-isaaclab-project

conda activate isaacsim

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch \
  --house \
  --device cuda:0 \
  --spawn 1.15 5.25 0.8 \
  --four_rgbd_cameras \
  --lavira_history_probe \
  --lavira_history_max_decisions 10 \
  --lavira_history_execution_timeout 45 \
  --lavira_backtrack_strategy replan_world_goal \
  --lavira_backtrack_max_path_m 6.0 \
  --fmm_execute_max_path_m 2.5 \
  --nav_map_mode lavira_compatible_global \
  --nav_map_resolution_m 0.05 \
  --nav_map_size_m 24.0 \
  --nav_global_origin_mode spawn_center \
  --nav_global_downscaling 2 \
  --nav_global_center_reset_steps 25 \
  --nav_global_unknown_space_policy blocked \
  --no-lavira_online_navigation \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 90 \
  --lavira_session_id "robot_01_global_backtrack_smooth_test_004" \
  --instruction "This is a controller verification task. Create exactly two navigation waypoints before returning. If history contains 0 waypoints, output NAVIGATE toward a visible clear traversable floor region away from the starting position. If history contains 1 waypoint, output NAVIGATE again toward another visible clear traversable floor region farther from the starting position. As soon as history contains 2 or more waypoints, output BACKTRACK with waypoint 0. Do not output STOP before issuing BACKTRACK to waypoint 0." \
  --max_steps 10000 \
  --real-time \
  --no-show_path

加了不停更新的FMM和不停的累计全局地图 碰到碰撞重新规划:
  cd /home/yile/projects/unitree-g1-isaaclab-project
  conda activate isaacsim
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch \
  --house \
  --device cuda:0 \
  --spawn 1.15 5.25 0.8 \
  --four_rgbd_cameras \
  --lavira_history_probe \
  --lavira_history_max_decisions 10 \
  --lavira_history_execution_timeout 45 \
  --lavira_backtrack_strategy replan_world_goal \
  --lavira_backtrack_max_path_m 6.0 \
  --fmm_execute_max_path_m 2.5 \
  --nav_map_mode lavira_compatible_global \
  --nav_map_resolution_m 0.05 \
  --nav_map_size_m 24.0 \
  --nav_global_origin_mode spawn_center \
  --nav_global_downscaling 2 \
  --nav_global_center_reset_steps 25 \
  --nav_global_unknown_space_policy blocked \
  --lavira_online_navigation \
  --lavira_online_mapping_interval_s 1.0 \
  --lavira_online_replan_interval_s 1.0 \
  --lavira_collision_command_speed_m_s 0.12 \
  --lavira_collision_window_s 0.75 \
  --lavira_collision_min_progress_m 0.04 \
  --lavira_collision_mark_distance_m 0.45 \
  --lavira_collision_mark_radius_m 0.15 \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 90 \
  --lavira_session_id "robot_01_online_navigation_test_001" \
  --instruction "Go through the doorway, then turn left and stop near the bed." \
  --max_steps 10000 \
  --real-time \
  --no-show_path