# Isaac Sim 合并模型 + 本机 iPlanner + G1 局部导航

本文档记录截至 2026-08-11 的导航实现、启动方法、实测结果和当前限制。

新代码目录：

```text
/home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_iplanner_g1
```

主要参考和复用来源：

```text
scripts/isaacsim_goal_tracking
/home/yile/projects/uni-lavira-code
```

本目录复用 `isaacsim_goal_tracking` 中已经与服务器连通的 schema-v2 模型协议、四相机配置、
stand/locomotion ONNX policy 和 IsaacLab 环境配置，并在运行时直接加载 Uni-LaViRA G1 原始
iPlanner 客户端及服务端实现。

## 1. 2026-08-11 完成情况

今天完成并验证的代码工作：

- 四方向 `forward / left / behind / right` RGB-D、合并模型一次返回
  `action + direction + bbox`、所选视图目标投影和转身后 forward iPlanner 主链路已经接通。
- 修复 Pure Pursuit 的 `current_idx`：同一路径只向前搜索，新 iPlanner 路径到达时才重置为 0。
- 模型历史长度和任务决策轮数已经分离；默认决策次数无限，直到 STOP、失败、Ctrl+C 或外层退出。
- 每次启动创建独立的 `session/run_时间戳` 输出目录，不再覆盖同 session 的旧实验文件。
- iPlanner 服务保持在本机 `127.0.0.1:8888`；当前客户端直接加载 Uni-LaViRA G1 原始
  `robot/iplanner_client.py`，适配层只负责 RGB→BGR 和米制浮点深度→整数毫米。
- iPlanner 保持 Uni 的固定 5 秒 HTTP timeout、墙钟重规划计时、失败后保留旧路径继续尝试、
  轨迹截短异常时使用原轨迹，以及原始 Pure Pursuit/yaw bias 顺序。
- 无 odometry 支持命令积分和“目标已经越过”停车保护；真机默认经验比例为 `0.7/0.8`，
  Isaac Sim 无 odometry 测试必须显式使用理想比例 `1.0/1.0`。
- odometry 分支按 Uni 行为只持续更新固定世界目标，不变换两次重规划之间的旧局部路径。
- bbox 底边中心没有有效深度时，改为 Uni 的 `[1.5, 0.0] m` 正前方 fallback，继续调用 iPlanner；
  投影 JSON 使用 `used_forward_fallback` 明确记录是否走了该分支。
- 保存 Uni 风格的初始规划和连续重规划投影图到 `images/iplanner/`。
- 真机 runner 已接入参数化 DDS、可选 ROS 2 odometry、启动时一次 `HighStand()`、默认关闭 IMU
  转向、Uni 风格持续发送最近速度、Ctrl+C 停车清理，以及 `StopMove()` 后等待 0.2 秒。
- DDS 命令 TTL 已移除以匹配 Uni；同步 iPlanner 阻塞期间真机后台仍持续发送最近速度命令。

今天完成的验证：

- 自动化测试：`43 passed`。
- odometry 仿真曾连续完成 2 次 NAVIGATE；也暴露了小到达半径下越过目标后绕行的问题。
- 无 odometry 多轮实验验证了 `goal_x < -0.1` 越过保护、iPlanner 断开时跳过动作以及
  `BACKTRACK` 当前会终止任务。
- 简单单轮无 odometry 实验
  `simple_doorway_noodom_20260811_01/run_20260811_182213_724047` 成功结束：
  `state=stopped, history=1, failure=None`。该轮原始路径约 `1.054 m`，安全裁剪后约 `0.558 m`。

尚未完成：

- 真机四方向 RGB-D camera backend、设备标定和真机联调。
- 当前一步组合模型要求所选方向也有深度；真机左右/后向深度方案仍未确定。
- `BACKTRACK` 没有 FMM/回退执行器，收到后仍会停车并进入 `FAILED`。
- Isaac 深度中的 `NaN/Inf` 转换为 Uni `uint16` 协议时仍会打印 cast warning。
- 无 odometry 长距离、多轮累计误差，以及更大且地面连续场景的验证。
- 最终模型 STOP 的完整长任务验证。

## 2. 完整导航流程

```text
四方向 RGB-D 全景
    ↓
现有 schema-v2 合并模型
    ↓
一次返回 action + direction + bbox
    ↓
所选方向转身前 bbox + 同方向 depth + K
    ↓
转身后的机器人局部目标 [x=前方, y=左方]
    ↓
固定 yaw 速度 + 固定持续时间转身
    ↓
等待机器人稳定
    ↓
采集最新 forward RGB-D
    ↓
局部目标 + 最新 forward depth 发送给本机 iPlanner
    ↓
iPlanner 局部轨迹
    ↓
末端安全距离裁剪
    ↓
Pure Pursuit + 周期性最新 forward RGB-D 重规划
    ↓
IsaacLab 速度命令 → locomotion ONNX → G1 关节动作
    ↓
到达目标 → stand → 写入 history → 下一轮模型决策
```

这里没有把“转身前 bbox 像素”直接放到“转身后 forward depth”中查询。

转身前的数据负责确定目标位置：

```text
所选方向 bbox + 所选方向 depth + 所选相机 K
→ 局部二维目标 [forward, left]
```

转身后的数据负责局部避障：

```text
最新 forward RGB-D + 局部二维目标
→ iPlanner 轨迹
```

当前采用理想化假设：模型选择的方向相机光轴，在固定角度转身后等价于新的机器人前方。

## 3. 坐标与反投影

模型全景方向顺序固定为：

```text
forward, left, behind, right
```

相机光学坐标：

```text
X：图像右方
Y：图像下方
Z：相机前方
```

机器人局部坐标：

```text
x：机器人前方
y：机器人左方
```

模型返回 bbox 后，使用 bbox 底边中心像素：

```text
u = (x1 + x2) / 2
v = y2
```

在该像素附近使用 `7×7` 深度窗口，过滤无效值后取 P30。水平反投影与 Uni-LaViRA G1 一致：

```text
camera_x = (u - cx) * depth / fx
goal_forward = depth
goal_left = -camera_x
```

与 Uni-LaViRA G1 一致，bbox 附近没有有效深度时使用转身后正前方 `1.5 m` 的默认目标，并继续调用
iPlanner。投影 JSON 会记录 `depth_m: null`、`valid_depth_count: 0` 和
`used_forward_fallback: true`，用来区分真实深度目标与 fallback。

## 4. 转向方式

方向到相对 yaw 的映射：

```text
forward = 0
left    = +pi/2
right   = -pi/2
behind  = -pi
```

转向使用固定角速度和固定时间：

```text
duration = abs(target_yaw) / rotation_speed * duration_scale
```

默认仿真值：

```text
rotation_speed_rad_s = 0.4
rotation_duration_scale_sim = 1.0
```

Uni-LaViRA 真机开环时间补偿为 `1.4`，仿真和真机系数刻意分开，后续需要分别标定。

## 5. iPlanner

iPlanner 在本机运行：

```text
http://127.0.0.1:8888
```

使用的 checkpoint：

```text
scripts/isaacsim_lavira_iplanner_g1/checkpoints/iplanner.pth
```

该 checkpoint 由官方 RSS 2023 iPlanner 文件转换为 `state_dict + metadata`，没有修改
Uni-LaViRA 源码。转换后 checkpoint 已通过 CPU 和 CUDA `strict=True` 加载以及 HTTP 推理测试。

iPlanner 每次收到：

- 最新 forward RGB PNG。
- 最新 forward depth PNG，编码为 `uint16`、单位 `0.1 mm`。
- 当前机器人局部目标 `goal_x/goal_y`。

运动期间默认每 `0.1` 个墙钟秒检查一次是否需要重新采集 forward RGB-D 并调用 iPlanner，以匹配
Uni-LaViRA 的 `time.time()` 行为。它不是 30 Hz 视频流，而是同步请求；本机一次推理约 1 秒时，
Isaac Sim 会在请求期间暂停推进物理，而真机 DDS 后台会持续发送最近一条速度。当前 iPlanner
神经网络主要使用 depth 和 point-goal，RGB 保留在兼容协议中。

## 6. Pure Pursuit 和 odometry

当前控制参数默认值：

```text
navigation_rate_hz     = 20.0
locomotion_policy_hz   = 50.0
walk_speed_m_s       = 0.3
lookahead_m          = 0.5
max_forward_speed    = 0.4
max_yaw_speed        = 0.5
safe_distance_m      = 0.5
replan_interval_s    = 0.1
```

为复刻 Uni-LaViRA G1 的命令时序，Isaac runner 使用分层频率：PhysX 保持 `200 Hz`，训练得到的
locomotion policy 保持 `50 Hz`，旋转/Pure Pursuit 状态机按平均 `20 Hz` 更新。由于 `50/20` 不是
整数，导航更新会落在交替的约 `60/40 ms` policy 边界上，长期平均严格为 `20 Hz`；两次导航更新
之间，50 Hz policy 始终读取最近一条速度命令，对应 Uni 真机的 50 Hz DDS 重发线程。

该 runner 的导航命令不再经过 `command_ramp_duration` 的额外 Python ramp，只进行 policy 训练范围
限幅后直接写入并保持。进入 stand 时仍立即写零速度，并保留 Isaac 特有的稳定窗口和 policy 动作
混合，避免破坏两套 ONNX policy 的安全接管。启动日志应包含：

```text
navigation_rate=20.0Hz policy_rate=50.0Hz command_filter=direct-hold
```

不要通过修改 `sim.dt` 或 `decimation` 把 locomotion policy 降到 20 Hz；那会改变 policy 训练时的
动作保持周期，而不是复刻 Uni 的高层 Pure Pursuit 频率。可用
`--local_navigation_rate_hz 20` 显式写出默认值。

短距离 Isaac Sim 调试建议使用：

```text
goal_tolerance_m     = 0.4
blind_yaw_radius_m   = 0.2
dead_reckoning       = 1.0 / 1.0
```

Uni-LaViRA 原代码的 `goal_tolerance=1.0 m` 与 `safe_distance=0.5 m` 会让短轨迹第一帧就被判定
到达；原代码的 blind-yaw 注释写 `0.6 m`，实际值却是 `2.0 m`，会让近距离横向目标完全不修正
yaw。`0.4/0.2` 是当前小场景的测试值，不代表真机最终参数。

### 无 odometry（默认）

默认不构造 odometry backend，不读取 Isaac 世界位姿。局部目标根据已经发送的速度命令更新：

```text
位移估计 = command_velocity * dt * dead_reckoning_scale
```

通用配置默认保留 Uni 真机经验比例 `0.7/0.8`。Isaac Sim 的运动没有同样的真实打滑和响应滞后，
无 odometry 仿真命令应显式传入：

```bash
--local_dead_reckoning_linear_scale_sim 1.0 \
--local_dead_reckoning_angular_scale_sim 1.0
```

启动日志应显示：

```text
[LOCAL-VLN] odometry backend: disabled
```

### Isaac odometry（显式诊断模式）

添加以下参数才会读取 Isaac root pose：

```bash
--local_use_isaac_odometry
```

启动日志应显示：

```text
[LOCAL-VLN] odometry backend: Isaac root pose (explicit opt-in)
```

有 odometry 时，固定世界目标根据当前真实 root pose 重新转换到机器人局部坐标；为了与 Uni-LaViRA
代码一致，两次 iPlanner 重规划之间不会再用 odometry 变换旧的局部路径。`goal_x < -0.1` 保护只应用于
无 odometry 航位推算；有 odometry 时使用真实二维目标距离继续跟踪。

## 7. 目录内容

| 文件 | 作用 |
| --- | --- |
| `run_isaacsim.py` | Isaac Sim 完整运行入口 |
| `run_g1_real.py` | 参数化的 G1 真机 runner 框架；等待真实四相机 backend |
| `run_iplanner_local.sh` | 启动本机 Uni-LaViRA iPlanner Flask 服务 |
| `config.yaml` | 当前模型、相机、运动和 planner 参数的参考记录；运行时以 CLI 参数为准 |
| `unified_vln/model_contract.py` | 只读复用现有 schema-v2 模型协议 |
| `unified_vln/model_client.py` | 合并模型 multipart/PNG HTTP 客户端和 history |
| `unified_vln/types.py` | RGB-D frame、全景 bundle 等基础数据结构 |
| `unified_vln/local_projection.py` | bbox 底边中心、7×7 P30 深度和局部反投影 |
| `unified_vln/rotation.py` | 默认固定时间转向；真机可用 `--use-imu-rotation` 显式启用 IMU yaw 闭环 |
| `unified_vln/iplanner_client.py` | 薄适配层：直接加载 Uni-LaViRA G1 原始 iPlanner 客户端，并完成 RGB→BGR、米→整数毫米转换 |
| `unified_vln/local_trajectory.py` | 安全裁剪、Pure Pursuit、重规划和航位推算 |
| `unified_vln/odometry.py` | 可选 Pose2D odometry 接口与坐标转换 |
| `unified_vln/ros2_odometry.py` | ROS 2 `/Odometry` 到 Pose2D 的真机适配器 |
| `unified_vln/isaac_backend.py` | Isaac 四相机和显式 root odometry adapter |
| `unified_vln/episode.py` | 完整导航状态机 |
| `unified_vln/g1_dds_backend.py` | G1 `LocoClient.Move` 50 Hz 与 DDS IMU yaw 后端骨架 |
| `convert_iplanner_checkpoint.py` | 官方完整模型 checkpoint 转换工具 |
| `verify_iplanner_checkpoint.py` | checkpoint CPU/CUDA 严格加载和推理检查 |
| `smoke_test_iplanner_http.py` | iPlanner HTTP 端到端 smoke test |
| `tests/` | 协议、反投影、转向、轨迹和状态机测试 |

`g1_dds_backend.py` 与 Uni-LaViRA G1 一样，只从 `rt/sportmodestate` 读取 IMU yaw，不把
`SportModeState.position` 当作轨迹里程计。位置由 ROS 2 `/Odometry` 提供；没有新鲜 ROS 位姿时，
局部跟随器使用命令积分。`run_g1_real.py` 已把状态机、DDS 和可选 ROS 2 odometry 串起来，但真实
相机仍通过 `--camera-factory module:function` 注入。没有完成相机驱动、标定和真机安全验证前，
不能把这个框架当作已经可以让机器人行走的成品。

模型历史和任务轮数现在相互独立：默认保留所有已完成 waypoint 的文字历史，协议只给最近 4 个
waypoint 附带图片；`--local_history_max_waypoints N` 可以只发送最近 N 条文字历史。任务默认无限轮，
直到模型返回 STOP。Isaac runner 可用 `--local_max_decisions N` 临时增加测试安全上限，`0` 表示无限。
继承自旧目录的 `--lavira_history_max_decisions` 不再控制这个新 runner。

真机框架的全部参数可以先在没有硬件时查看：

```bash
conda activate isaacsim
cd /home/yile/projects/unitree-g1-isaaclab-project
python scripts/isaacsim_lavira_iplanner_g1/run_g1_real.py --help
```

以后完成相机 backend 后，启动形式如下。尖括号内容都必须根据真机和标定结果填写：

```bash
python scripts/isaacsim_lavira_iplanner_g1/run_g1_real.py \
  --instruction "<导航指令>" \
  --model-url "<组合模型地址>" \
  --iplanner-url "http://127.0.0.1:8888" \
  --network-interface "<G1 有线网卡>" \
  --camera-factory "<Python模块>:<创建函数>" \
  --camera-config "<相机序列号和标定配置>" \
  --odometry-topic "<实际 ROS2 Odometry topic>" \
  --rotation-duration-scale "<真机标定值>" \
  --history-max-waypoints 4
```

相机创建函数接收一个 `Path`，返回实现 `capture_panorama()` 和 `capture_forward()` 的对象；返回帧
必须符合 `ViewFrame` 的 RGB、米制对齐深度、内参 K 和递增 frame_id 契约。

## 8. 启动前准备

需要三个终端：

1. 远程合并模型 SSH 隧道。
2. 本机 iPlanner 服务。
3. Isaac Sim 导航程序。

不要给 Isaac Sim 命令添加 `--disable_fabric`，现有项目记录显示它可能导致视觉与物理不同步。

### 终端 1：远程合并模型 SSH 隧道

```bash
ssh -N \
  -L 127.0.0.1:18765:127.0.0.1:8765 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  wangchu@131.159.60.188
```

Isaac 侧模型 URL：

```text
http://127.0.0.1:18765/v1/lavira/decision
```

### 终端 2：本机 iPlanner

```bash
conda activate isaacsim
cd /home/yile/projects/unitree-g1-isaaclab-project
bash scripts/isaacsim_lavira_iplanner_g1/run_iplanner_local.sh
```

健康检查：

```bash
curl http://127.0.0.1:8888/health
```

服务刚启动、还没有收到 `/navigator_reset` 时，下面的结果是正常的：

```json
{"device":"cuda","model_loaded":false,"status":"ok"}
```

第一次导航规划后，`model_loaded` 会变为 `true`。

## 9. 打开 Isaac Sim：odometry 连续导航测试

下面是今天已经成功完成连续两轮 NAVIGATE 的配置。它使用显式 Isaac root odometry，适合先验证
模型、iPlanner 和运动控制闭环。

```bash
conda activate isaacsim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

cd /home/yile/projects/unitree-g1-isaaclab-project

python scripts/isaacsim_lavira_iplanner_g1/run_isaacsim.py \
  --house \
  --spawn 1.15 5.25 0.8 \
  --device cuda:0 \
  --real-time \
  --no-four_rgbd_set_viewport \
  --instruction "Navigate to the doorway. Do not stop until the robot is near the doorway." \
  --lavira_session_id "doorway_odom_test_NEW" \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 90 \
  --iplanner_url "http://127.0.0.1:8888" \
  --iplanner_timeout_s 5 \
  --local_history_max_waypoints 4 \
  --local_goal_tolerance_m 0.2 \
  --local_blind_yaw_radius_m 0.2 \
  --local_safe_distance_m 0.5 \
  --local_replan_interval_s 0.1 \
  --local_use_isaac_odometry
```

`--no-four_rgbd_set_viewport` 使用第三人称视角观察机器人。删除该参数后，GUI viewport 会切到
机器人 forward RGB-D 相机。

每次实验必须使用新的 `--lavira_session_id`，避免远程模型继续使用旧 session 的历史。

上面的命令没有指定 `--local_max_decisions`，因此会持续运行到模型 STOP。第一次验证代码改动时，
建议临时添加 `--local_max_decisions 2`，确认两轮稳定后再删除该参数。

## 10. 打开 Isaac Sim：无 odometry 简单单轮测试

下面的配置完全不读取 Isaac root pose，只靠命令积分执行一次“靠近最近门口”的简单决策。该流程
已经实测得到：

```text
state=stopped history=1 failure=None
```

运行命令：

```bash
conda activate isaacsim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

cd /home/yile/projects/unitree-g1-isaaclab-project

python scripts/isaacsim_lavira_iplanner_g1/run_isaacsim.py \
  --house \
  --spawn 1.15 5.25 0.8 \
  --device cuda:0 \
  --real-time \
  --no-four_rgbd_set_viewport \
  --instruction "Move toward the nearest visible doorway and stop near it." \
  --lavira_session_id "simple_doorway_noodom_NEW" \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 90 \
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

确认日志中没有：

```text
--local_use_isaac_odometry
```

并且启动输出为：

```text
[LOCAL-VLN] odometry backend: disabled
```

## 11. 无界面运行

服务器或批量测试时，把 GUI 命令中的：

```text
--real-time
--no-four_rgbd_set_viewport
```

替换为：

```text
--headless
```

例如：

```bash
python scripts/isaacsim_lavira_iplanner_g1/run_isaacsim.py \
  --headless \
  --house \
  --spawn 1.15 5.25 0.8 \
  --device cuda:0 \
  --instruction "Navigate to the doorway." \
  --lavira_session_id "headless_test_NEW" \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 90 \
  --iplanner_url "http://127.0.0.1:8888" \
  --local_max_decisions 1 \
  --local_goal_tolerance_m 0.2 \
  --local_blind_yaw_radius_m 0.2
```

## 12. 使用其他地图

当前 `--house` 使用：

```text
/home/yile/scene/House/scene_047/mujoco/usd/scene_scene_047.usda
```

当前新 runner 默认出生点：

```text
--spawn 1.15 5.25 0.8
--yaw 3.141592653589793
```

换地图时不使用 `--house`，改为：

```bash
--scene_usd /absolute/path/to/new_scene.usd \
--spawn X Y Z \
--yaw YAW_RAD
```

建议的新场景条件：

- 至少两个房间连通。
- 门后有连续地面和有效碰撞几何。
- 出生点距离目标约 3～6 米。
- 相机 `5 m` 深度范围内能看到地面、墙面或障碍物。
- 不要让门口直接连接到 USD 场景外部。

## 13. 输出文件

每次运行都会在对应 session 下自动创建一个独立目录。即使重复使用同一个
`lavira_session_id`，新实验也不会覆盖或混入旧实验文件：

```text
outputs/isaacsim_lavira_iplanner_g1/<lavira_session_id>/run_<日期>_<时间>_<微秒>/
```

| 文件 | 内容 |
| --- | --- |
| `decision_000_request.json` | 发给合并模型的元数据、instruction、history 和四图字段名 |
| `decision_000_response.json` | 模型原始 `action/direction/target/bbox` 回答 |
| `decision_000_projection.json` | bbox 像素、深度窗口、有效点数和局部目标 |
| `decision_000_plan.json` | iPlanner 原始轨迹、safe trajectory、safe goal 和 fear |
| `decision_000_completed.json` | 机器人完成动作后正式写入 history 的记录 |
| `images/iplanner/step*_plan_*.jpg` | 初始 iPlanner 轨迹投影图，与 Uni 的调试图格式一致 |
| `images/iplanner/replan_*.jpg` | 连续重规划产生的轨迹投影图 |
| `failure.json` | 状态机无法继续运行时记录的原因 |

这些 JSON 和轨迹图仅用于调试和实验审计，运行时不会重新读取它们。当前组合模型协议也不会把
`images/iplanner/` 自动发送给远程模型；Uni 原版把它们交给下一轮 LA，而组合模型是否需要这类输入
必须由组合模型接口另行定义。启动日志中的
`[LOCAL-VLN] output directory:` 会显示本次运行的准确目录。旧 run 文件夹可以删除，但删除后无法
恢复其中的实验记录。远程模型仍然使用 `lavira_session_id` 识别会话；需要全新的远程模型上下文时，
仍应更换 session ID。

## 14. 2026-08-11 实测结果

### 成功：简单无 odometry 单轮

运行目录：

```text
outputs/isaacsim_lavira_iplanner_g1/simple_doorway_noodom_20260811_01/
run_20260811_182213_724047/
```

最终结果：

```text
odometry backend: disabled
dead_reckoning=(1.000,1.000)
Configured decision limit reached; episode stopped.
state=stopped history=1 failure=None
```

该轮模型选择 `right / nearest visible doorway`，真实深度目标约 `[0.845,-0.139] m`，没有触发
fallback。iPlanner 原始路径 51 点、约 `1.054 m`；保留 0.5 m 后的安全路径 11 点、约 `0.558 m`，
safe goal 约 `[0.556,0.047] m`。该结果验证了无 odometry、目标投影、本机 iPlanner、路径裁剪、
Pure Pursuit、stand/locomotion 收尾和单轮正常停止。

### 诊断：odometry 连续运行

odometry 模式曾完成连续两次 NAVIGATE，但 `goal_tolerance=0.2 m` 时机器人以约 `0.235 m` 的最近
距离擦过目标。odometry 分支没有无 odometry 的 `goal_x < -0.1` 越过保护，随后围绕后方目标持续
重规划。这证明 iPlanner 链路工作，但也说明当前小到达半径和 Uni odometry 语义不适合该小场景。

### 诊断：无 odometry 多轮

无 odometry 多轮运行验证了越过目标保护：目标到达 `x=-0.101 m` 时结束当前路径，没有继续绕圈。
一次实验的前 3 次 iPlanner 请求因本机 `127.0.0.1:8888` 未启动而被跳过，服务恢复后完成两段；
最终组合模型返回 `BACKTRACK`，因本地没有 FMM executor 而进入 `FAILED`。正式实验前必须先执行
`curl http://127.0.0.1:8888/health`，并为每次运行使用新的远程 session ID。

### 深度 fallback

旧实验曾因 bbox 底边附近无有效深度而 fail closed。2026-08-11 已改为 Uni-LaViRA 的行为：记录
`used_forward_fallback=true`，使用 `[1.5,0.0] m` 继续规划。自动化测试已覆盖该分支；最近一次简单
实测的深度有效，因此尚未在真实仿真画面中触发该 fallback。

## 15. 与 Uni-LaViRA G1 真机代码的关系

基本一致：

- 局部坐标 `x forward / y left`。
- bbox 底边中心、7×7 深度窗口和 P30。
- `goal_x=Z`、`goal_y=-X_camera`。
- bbox 深度不可用时使用 `[1.5,0.0] m` 正前方 fallback。
- iPlanner `/navigator_reset` 和 `/pointgoal_step` 协议。
- forward depth + point-goal 局部规划。
- 末端 `0.5 m` 安全裁剪。
- Pure Pursuit、lookahead、速度限制和连续重规划结构。
- Pure Pursuit 平均 20 Hz 更新，最近速度由 50 Hz locomotion policy 持续保持；导航阶段没有额外
  Python 速度 ramp，对应 Uni 的 20 Hz 跟踪循环和 50 Hz DDS 最近命令重发。
- 无 odometry 命令积分和可选 odometry 分支。
- DDS `SportModeState` 只提供 IMU yaw，ROS 2 `/Odometry` 提供轨迹位姿。
- 固定角速度转身，IMU 新鲜时闭环累计 yaw，缺失时按时间开环兜底。
- G1 高层速度接口 `LocoClient.Move(vx, vy, wz)` 和 50 Hz 发送方式。

按本项目要求有意不同：

- Uni-LaViRA 原版是 LA 选方向、转身、VA 再给 bbox。
- 本项目保留已经调好的合并模型，一次返回 `action + direction + bbox`。
- 本项目使用转身前所选方向的 bbox+depth，转身后只用最新 forward depth 规划。
- 不执行 FMM 和 backtrack。

真机部署前仍需完成：

- Uni-LaViRA 原始相机代码只有 front 可靠提供 depth，left/right 默认只拍 RGB，behind 是 V4L2 RGB。
  当前一步模型需要所选方向的 depth，因此必须补齐真机多方向深度或设计明确替代方案。
- 为 `run_g1_real.py` 实现并注入真实四方向 RGB-D camera factory，然后进行真机安全联调。
- 验证本机 `/home/yile/projects/unitree_slam_example` 能否连接 G1 真机上的
  `slam_operate` 服务：先用 `keyDemo <真机有线网卡>` 按 `q` 启动建图，并确认建图过程中
  `/unitree/slam_mapping/odom` 持续发布 `nav_msgs/msg/Odometry`。若真机固件使用新版 topic，
  同时检查 `/lio_sam_ros2/mapping/re_location_odometry`。
- 将实际可用的上述 topic 配置给 `Ros2OdometryProvider`，检查 frame、QoS、发布频率、时间戳、
  坐标方向和漂移；再验证按 `w` 保存地图、下次按 `a` 加载地图并重定位后仍能持续获得 odometry。
  这里只使用 Unitree SLAM 的建图/重定位与位姿输出，不启动 `keyDemo` 自带的目标点导航，避免与
  本项目通过 DDS `LocoClient.Move` 发送的速度命令发生控制冲突。
- 增加真机相机序列号、内参、RGB-depth 对齐和新帧检查。
- 在真实 G1 上验证同步 iPlanner 阻塞期间 DDS 持续发送最近速度、Ctrl+C 停车和外部硬件急停。
- 分别标定真机旋转时间系数和无 odometry 线性/角速度系数。

## 16. 测试和诊断命令

运行全部新目录测试：

```bash
conda activate isaacsim
cd /home/yile/projects/unitree-g1-isaaclab-project
python -m pytest -q scripts/isaacsim_lavira_iplanner_g1/tests
```

当前结果：

```text
45 passed
```

验证 iPlanner checkpoint：

```bash
conda activate isaacsim
cd /home/yile/projects/unitree-g1-isaaclab-project

python scripts/isaacsim_lavira_iplanner_g1/verify_iplanner_checkpoint.py \
  --checkpoint scripts/isaacsim_lavira_iplanner_g1/checkpoints/iplanner.pth \
  --config /home/yile/projects/uni-lavira-code/real-world-code/unitree_g1/iplanner/configs/iplanner.yaml \
  --module-dir /home/yile/projects/uni-lavira-code/real-world-code/unitree_g1/iplanner \
  --device cuda
```

iPlanner HTTP smoke test：

```bash
python scripts/isaacsim_lavira_iplanner_g1/smoke_test_iplanner_http.py \
  --url http://127.0.0.1:8888
```

PyTorch 2.7 可能在 Uni-LaViRA 的 `iplanner_agent.py` 中打印 expanded tensor 原地写入的弃用警告。
当前 CPU、CUDA 和 HTTP 推理均已通过，该警告没有改变本次输出，原 Uni-LaViRA 源码未被修改。

## 17. 常见日志解释

成功：

```text
[LOCAL-VLN] state=executing
[LOCAL-VLN] state=wait_action_stand
[LOCAL-VLN] state=capture_and_decide
```

达到人为决策次数限制，属于正常停止：

```text
Configured decision limit reached; episode stopped.
finished: state=stopped ... failure=None
```

所选 bbox 附近没有深度时，不再终止任务，而是打印警告并使用 Uni fallback：

```text
No valid selected-view depth ... using Uni-LaViRA fallback goal [1.5, 0.0]m
```

iPlanner 请求或重规划失败时，行为与 Uni-LaViRA 一致：初始规划无路径会跳过本次动作；
连续重规划失败则保留旧路径并持续重试，不设置失败次数上限：

```text
[iPlanner] Plan request failed: ...
[LOCAL-VLN WARN] iPlanner replan failed: ...
```

本机 iPlanner 没有启动：

```text
Cannot connect to server ... 127.0.0.1:8888 ... Connection refused
```

遇到该错误时先停止实验，在另一个终端启动 `run_iplanner_local.sh`，确认 `/health` 可访问后使用新
session ID 重跑。当前初始规划失败会按 Uni 行为跳过本次动作，因此服务未启动时继续运行会浪费模型决策。

组合模型返回 `BACKTRACK`：

```text
BACKTRACK is preserved by the model schema but intentionally unsupported
```

这不是 iPlanner 故障；当前执行器没有 FMM 回退路径，因此会停车并进入 `FAILED`。

Isaac 浮点深度包含 `NaN/Inf` 时，直接适配 Uni 的整数毫米输入可能出现：

```text
RuntimeWarning: invalid value encountered in cast
```

当前无效值会进入 `uint16` 深度中的无效区域；该 warning 本身不会终止任务，但仍是待处理的仿真
适配问题。

无 odometry 航位推算认为已经越过目标：

```text
local goal was passed without odometry (x=..., y=..., distance=...)
```

Isaac/RTX/Fabric 的 performance warning、DLSS 分辨率 warning 和 URDF fixed-link warning 通常不是本导航
流程失败原因，应优先查看最后的 `[LOCAL-VLN ERROR]` 和 `failure.json`。

## 18. 2026-08-13：scene_200 大场景与多轮模型 STOP 实测

### 场景安装和文件位置

从 SceneSmith 示例数据集下载了 `House/scene_200.tar`。这是一个约
`13.3 m × 13.7 m` 的艺术展馆，包含三个展厅、走廊、办公室、储藏室和卫生间。下载缓存和解压后的
Isaac Sim 主 USD 分别位于：

```text
/home/yile/scene/.downloads/scenesmith-example-scenes/House/scene_200.tar
/home/yile/scene/House/scene_200/mujoco/usd/scene_scene_200.usda
```

本项目无需复制或修改场景 USD，运行时直接通过 `--scene_usd` 使用绝对路径。不要同时使用
`--house`，因为 `--house` 仍是旧 `scene_047` 的快捷参数。

静态检查确认主 USD 的 payload、Geometry、Materials 和 Physics 使用相对引用，资源已经完整解压。
Isaac Sim 可以成功创建物理场景。启动时部分复杂小物体可能打印：

```text
failed to cook GPU-compatible mesh, collision detection will fall back to CPU
```

这表示对应碰撞网格退回 CPU，不是 USD、贴图或碰撞资源缺失。

### 已验证的出生点和相机

第一展厅内部约 `6 m × 5 m`，中央有一张长凳。已验证的安全出生点为：

```text
spawn = (4.6, 3.5, 0.8)
```

两个测试朝向：

```text
yaw = -0.72       # 面向展厅门口，仅用于门口诊断
yaw = pi          # 面向展厅内部，后续室内实验使用
```

`yaw=pi` 的站立探测中，机器人高度从 `0.798 m` 稳定到约 `0.784 m`，水平速度和 yaw rate 收敛到接近
零。四方向 RGB-D 也已实际采集，四个方向深度有效率约 `99.9%–100%`。验证输出：

```text
outputs/scene_200_room_only_probe/run_20260813_123512_528538/
bundle_000000_step_000008/
```

其中 `montage.png` 是 forward/left/behind/right 的四图拼接，`*_depth.npy` 是米制浮点深度，
`metadata.json` 保存内参、帧号和采集时间。

### 实验一：门口目标会投影到门后

运行目录：

```text
outputs/isaacsim_lavira_iplanner_g1/scene200_gallery_door_20260813_01/
run_20260813_122821_277839/
```

模型正确选择前方 open doorway，但 bbox 底边中心的深度为 `4.70 m`，目标实际落在门后的地面上；
0.8 m 之前使用的安全裁剪结果仍要求机器人向门后移动约 `4.01 m`。该轮持续生成 152 张 replan 图，
最终进入门后狭窄区域并卡住。结论：当前 `bbox 底边 + depth` 不能表达“只停在门口”，门口任务不适合
用来验证这个场景中的基础链路。

### 实验二：文字要求不能保证 iPlanner 避开长凳

运行目录：

```text
outputs/isaacsim_lavira_iplanner_g1/scene200_left_then_right_20260813_01/
run_20260813_124040_941836/
```

模型选择长凳后方的彩虹画，初始目标约为 `[3.39,-1.60] m`。初始 iPlanner 绿色轨迹直接穿过长凳。
原因是“避开长凳”的 instruction 只传给组合模型；本机 iPlanner 只收到转身后的 RGB-D 和二维局部
目标，Pure Pursuit 又只负责跟踪 iPlanner 轨迹。当前没有独立的碰撞急停、局部占据栅格或轨迹碰撞
复核层，因此不能保证不碰障碍物。模型 instruction 应尽量选择与机器人之间已有开阔直线的视觉目标，
但这只是降低风险，不是硬保证。

### 实验三：过小 goal tolerance 会导致 odometry 分支追逐身后目标

运行目录：

```text
outputs/isaacsim_lavira_iplanner_g1/scene200_clear_wall_route_20260813_01/
run_20260813_124757_639852/
```

使用 `--local_goal_tolerance_m 0.4` 时，机器人直行后最近一次目标约为 `[0.22,-0.95] m`，二维距离
约 `0.975 m`，仍没有进入 0.4 m 到达范围。机器人随后越过目标，目标变到身后；odometry 分支按照
Uni 语义没有 `goal_x < -0.1` 的越过保护，Pure Pursuit 又保持至少 `0.1 m/s` 的前进速度，因此开始
绕圈追逐身后目标，并生成 334 张 replan 图。

本次没有修改代码，因为 `LocalFollowerConfig.goal_tolerance_m` 的默认值本来就是 Uni 的 `1.0 m`；
问题来自实验命令用 `0.4` 覆盖了默认值。scene_200 后续测试统一使用：

```text
--local_goal_tolerance_m 1.0
```

### 实验四：不限轮数并由模型主动 STOP

成功运行目录：

```text
outputs/isaacsim_lavira_iplanner_g1/scene200_clear_wall_unlimited_20260813_01/
run_20260813_132209_962621/
```

最终状态：

```text
[LOCAL-VLN] decision accepted: index=3 action=STOP direction=forward target='rainbow-colored painting'
[LOCAL-VLN] STOP final approach completed; episode stopped.
[LOCAL-VLN] finished: state=stopped history=3 failure=None
```

这次没有 `--local_max_decisions`，停止原因确实是组合模型返回 `STOP`，不是人为轮数限制。实际过程：

1. decision 0：`NAVIGATE forward`，沿长凳右侧接近深色风景照片，初始目标约
   `[4.54,-0.22] m`，`fear≈0.000043`。
2. decision 1：模型再次选择深色照片，目标约 `[2.01,-0.09] m`；iPlanner
   `fear≈0.999938`，但安全轨迹很短，因此没有明显继续平移。
3. decision 2：模型 reasoning 正确表示下一步转向右侧彩虹画，并执行右转；response 的 `target`
   字段仍错误写成 dark landscape photograph，bbox 和画面实际对应彩虹画。iPlanner 返回的可执行段
   小于安全距离，因此只转向、未继续平移。
4. decision 3：模型在新 forward 图中识别彩虹画，返回 `STOP`。最后投影距离约 `1.76 m`，没有额外
   前进，模型认为该距离已经完成任务。

这次实验证明四图合并模型、history、多轮 NAVIGATE、方向旋转、本机 iPlanner、stand/locomotion
收尾以及模型主动 STOP 的完整链路可以跑通。仍未证明模型每轮的目标文字稳定、fear 能参与安全决策，
或 iPlanner 对任意室内障碍都能可靠避让。`history=3` 表示 STOP 前有三条 completed history，不代表
机器人完成了三段明显的平移。

### 当前 scene_200 推荐测试命令

先启动远程模型 SSH 隧道和本机 `run_iplanner_local.sh`，然后使用新的 session ID 运行：

```bash
conda activate isaacsim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
cd /home/yile/projects/unitree-g1-isaaclab-project

python scripts/isaacsim_lavira_iplanner_g1/run_isaacsim.py \
  --scene_usd /home/yile/scene/House/scene_200/mujoco/usd/scene_scene_200.usda \
  --spawn 4.6 3.5 0.8 \
  --yaw 3.141592653589793 \
  --device cuda:0 \
  --real-time \
  --no-four_rgbd_set_viewport \
  --instruction "Stay inside this exhibition room. First move straight toward the dark landscape photograph directly ahead on the far wall. Then move toward the rainbow-colored painting on the adjacent wall to the right and stop near it. Do not enter any doorway." \
  --lavira_session_id "scene200_clear_wall_NEW_SESSION" \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 90 \
  --iplanner_url "http://127.0.0.1:8888" \
  --iplanner_timeout_s 5 \
  --local_history_max_waypoints 4 \
  --local_goal_tolerance_m 1.0 \
  --local_rotation_speed_rad_s 0.4 \
  --local_safe_distance_m 0.5 \
  --local_blind_yaw_radius_m 0.2 \
  --local_replan_interval_s 0.1 \
  --local_walk_speed_m_s 0.30 \
  --local_max_forward_speed_m_s 0.40 \
  --local_max_yaw_speed_rad_s 0.50 \
  --local_use_isaac_odometry
```

不要添加 `--local_max_decisions` 即表示不限模型轮数，任务持续到模型 `STOP`、失败或用户按
`Ctrl+C`。每次正式实验应更换 `lavira_session_id`。出现路径穿越长凳、持续转圈、贴墙或其它明显
危险行为时立即按 `Ctrl+C`；当前视觉导航链路不能替代碰撞检测或硬件急停。

上面五个运动参数与 Uni-LaViRA G1 保持一致：期望行走速度 `0.30 m/s`、最大前进速度
`0.40 m/s`、最大 Pure Pursuit yaw `0.50 rad/s`、原地转向速度 `0.40 rad/s`、路径末端安全距离
`0.50 m`。它们本来就是 `run_isaacsim.py` 的默认值；这里显式写出是为了避免后续实验命令再次用
临时的低速/`0.8 m` 配置覆盖默认值。无 odometry 线性/角速度比例和开环转向时间比例仍应根据
Isaac Sim 与真实 G1 分别标定，不属于这五项统一参数。
