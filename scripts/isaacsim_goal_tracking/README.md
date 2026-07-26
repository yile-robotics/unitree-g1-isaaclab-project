# Isaac Sim + Unitree G1 + LaViRA 导航系统

更新时间：2026-07-26

本文是 `scripts/isaacsim_goal_tracking` 的唯一权威说明，覆盖当前已经实现的功能、服务器接口、
运行方法、输出文件、已验证结果和尚未完成的部分。

如果文档与代码发生冲突，以以下文件为准：

```text
goal_tracking/lavira_protocol.py
goal_tracking/lavira_offline.py
goal_tracking/lavira_episode.py
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
  -> 四路 depth 局部 occupancy map
  -> G1 尺寸障碍膨胀和安全目标修正
  -> scikit-fmm 距离场和无碰撞 waypoint
  -> 原 WaypointPathFollower
  -> 原 SwitchCommandController
  -> 原 29DOF locomotion ONNX
  -> G1 运动
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
已实现：一次模型决策 + 一次局部规划 + 一次短路径执行
已实现：decision_000 执行成功后生成 history 并自动请求 decision_001
已实现：有限多轮 NAVIGATE/BACKTRACK/STOP controller，默认三次请求
已实现：BACKTRACK 默认按历史 waypoint 世界坐标在当前地图重新 FMM，并立即截断 history
已实现：LaViRA 风格 STOP 最终接近、稳定 stand 和 episode 终止
已验证：默认 `replan_world_goal` 下 Qwen 两次选择 waypoint=0，G1 均重新 FMM 并成功返回
已验证：旧 `stored_reverse` 策略下真实服务器返回 waypoint=1，G1 沿 1.962m 反向路径到达
已验证：mock STOP 下的“目标已在阈值内”无移动终止
已验证：mock STOP 下的 FMM 最终接近、真实 G1 locomotion、0.75m 停止和 stand
未实现：无限 episode、ground-truth task success 和 SPL 等评测
```

默认 `--lavira_decision_probe` 仍然只请求 `decision_000`；需要多轮执行时使用
`--lavira_history_probe` 的 LaViRA 风格有限循环。默认最多请求 3 次：执行前两个普通
NAVIGATE/BACKTRACK，每次成功后更新 history，第三次普通响应只保存。STOP 是终止动作，
即使出现在第三次也会执行最终接近。`--lavira_history_max_decisions` 可以调整边界。

### 1.1 2026-07-26：BACKTRACK 对齐 `qwen_end2end`

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
  -> 使用当前四视图重新构建局部地图
  -> 在当前地图上重新运行 FMM
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
  -> 当前地图重新 FMM
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

### 1.2 2026-07-23 完成内容

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

### 1.3 仍未完成

- 真实 Qwen 主动选择 STOP 的端到端验证；目前 STOP 运动执行使用 mock 强制触发。
- 真正“持续运行直到 STOP”的开放 episode。当前仍由
  `--lavira_history_max_decisions` 限制决策数，提高到较大值只能近似开放运行。
- 普通最后一轮响应仍是只读；BACKTRACK 如果恰好出现在最后一轮不会执行。STOP 已做终止动作
  例外。
- 路径卡住、FMM 不可达、cross-track 或单段超时后的自动恢复。目前会安全进入 FAILED，不会
  自动重新拍照请求模型。
- BACKTRACK 当前只使用决策时重新采集的四视图局部地图，尚未融合整个 episode 的累计地图。
- BACKTRACK 执行过程中的动态障碍更新和在线重新规划。
- 多轮 occupancy map 累积、运动中地图更新和在线重规划。
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

## 7. 四路 depth 局部导航地图

增加：

```text
--lavira_local_map_probe
```

后，客户端把本轮四路 depth 合并为当前局部世界栅格。

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

规则：

- 未观测区域不可通行。
- 障碍按照 G1 水平包络膨胀。
- traversable 只保留与机器人当前位置连通的区域。
- bbox 原始目标不可通行时，先沿相机射线向机器人方向回退。
- 回退失败后才搜索有限距离内的最近 traversable 点。
- 没有安全目标时停止规划。

这是当前四相机视野生成的局部地图，不是读取完整 USD 得到的全局地图，也不会跨决策轮累积。

输出：

```text
navigation_map.json
navigation_map.npz
navigation_map.png
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
个普通响应只读。收到 BACKTRACK 时，程序默认读取目标 waypoint 的历史世界坐标，在当前
四视图 traversability map 上重新运行 FMM，并在接受动作时立即把 history 截断到该 waypoint；
decision index 仍保持单调递增。BACKTRACK 路径上限默认为 `6.0m`，可用
`--lavira_backtrack_max_path_m` 调整。旧反向路径策略可用
`--lavira_backtrack_strategy stored_reverse` 显式开启。

STOP 严格复用 NAVIGATE 已有的 bbox 底边中心投影、安全目标回退、局部地图、FMM 和
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
`bounded_episode_status.json`。

## 12. 关键代码

| 文件 | 作用 |
| --- | --- |
| `isaacsim_path_follwing.py` | 程序入口，装配环境、相机和 runner |
| `goal_tracking/config.py` | 默认资源路径和所有 CLI 参数 |
| `goal_tracking/runners.py` | stand、locomotion、switch、一次性 FMM 和 history controller 接入 |
| `goal_tracking/control.py` | 命令写入、速度斜坡、stand/locomotion 状态机 |
| `goal_tracking/path.py` | waypoint、pure-pursuit、动态 FMM 路径和安全中止 |
| `goal_tracking/camera.py` | 四相机固定安装、方向、俯角和调试显示 |
| `goal_tracking/frame_bundle.py` | 同步 RGB-D、内参、外参和位姿 |
| `goal_tracking/lavira_protocol.py` | schema v2 请求、history、响应和严格校验 |
| `goal_tracking/lavira_offline.py` | 可复用任意 decision/history 的 multipart、HTTP、落盘和单轮流程 |
| `goal_tracking/lavira_episode.py` | LaViRA 风格候选 waypoint、到达提交和有限多轮状态机 |
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
Ran 55 tests
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
- FMM 绕障、不可达和路径安全。
- FMM 路径执行准备、起点漂移、超长、cross-track 和倾角中止。

## 14. 已知限制

目前尚未实现：

- 不阻塞仿真主线程的异步网络 worker。
- 跨进程退出/重启后的 episode 和 history 恢复。
- BACKTRACK 默认只使用本轮四视图局部地图重新规划，尚未融合整个 episode 的累计地图。
- 多轮 occupancy map 累积。
- 运动过程中的动态地图更新和重新规划。
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

真实服务器已经验证：

```text
decision_000 -> NAVIGATE，执行并提交 waypoint 0
decision_001 -> history=1, images=6，NAVIGATE 并提交 waypoint 1
decision_002 -> history=2, images=8，NAVIGATE 并提交 waypoint 2
decision_003 -> history=3, images=10，返回 BACKTRACK waypoint=1
```

上述 `decision_003` 当时恰好是有限测试的最后一个普通响应，因此该次 BACKTRACK 只验证和
保存。另一次 `robot_01_backtrack_execute_test_003` 已在旧 `stored_reverse` 策略下真实执行
BACKTRACK waypoint 1，沿 `1.962m` 反向路径在 step 3165 到达。当前默认策略已改为按该
waypoint 的世界坐标在当前地图重新 FMM。

BACKTRACK 和 STOP 都已按原版 LaViRA action 语义接入，STOP 客户端执行链也已通过 mock
实机仿真验证。接下来需要：

```text
STOP：真实 Qwen 主动触发后的最终接近和 stand 验证
BACKTRACK：真实 Qwen 下验证默认 world-goal 重规划，并评估局部地图遮挡/未知区域
EPISODE：支持持续运行直到 STOP，并为卡住/不可达增加重新决策恢复
评测：加入 ground-truth goal region、success、SPL 和碰撞统计
```
