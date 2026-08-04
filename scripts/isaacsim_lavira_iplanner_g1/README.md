# Isaac Sim 合并模型 + 本机 iPlanner + G1 局部导航

本文档记录 2026-08-04 完成的新导航目录、实现内容、启动方法、已经验证的结果和当前限制。

新代码目录：

```text
/home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_iplanner_g1
```

以下已有代码没有被修改：

```text
scripts/isaacsim_goal_tracking
/home/yile/projects/uni-lavira-code
```

新实现只读复用 `isaacsim_goal_tracking` 中已经与服务器连通的 schema-v2 模型协议、四相机配置、
stand/locomotion ONNX policy 和 IsaacLab 环境配置。

## 1. 当前完成情况

目前已经验证成功：

- 四方向 `forward / left / behind / right` RGB-D 全景采集。
- 保持现有合并模型接口，一次返回 `action + direction + bbox`。
- 使用模型所选方向转身前的 bbox、depth 和相机内参 K 计算局部目标。
- 默认不读取世界坐标，不使用全局地图、FMM 或 Isaac root pose。
- 使用固定角速度和固定时间执行左转、右转和向后转。
- 转身稳定后重新采集最新 forward RGB-D。
- 本机 RTX 5080 运行官方 iPlanner checkpoint。
- 使用 Uni-LaViRA HTTP 协议调用 `/navigator_reset` 和 `/pointgoal_step`。
- 使用安全距离裁剪、Pure Pursuit 和周期性 iPlanner 重规划产生 `vx/vy/wz`。
- 复用现有 Isaac G1 stand/locomotion ONNX policy 执行速度命令。
- 支持默认无 odometry 航位推算和显式启用 Isaac root odometry 两种模式。
- 成功动作写入模型 history；STOP 支持最后一次接近后结束。
- `BACKTRACK` 保留模型协议，但本地执行器不使用 FMM，收到后安全停车并报错退出。

已经完成的实际仿真验证：

- odometry 模式连续完成 2 次 NAVIGATE，最终 `history=2`。
- 无 odometry 模式完成 1 次 NAVIGATE，最终 `history=1, failure=None`。
- 新目录测试共 14 项，全部通过。

尚未完成：

- 完整真机 G1 启动入口和真机四相机 backend。
- 真机左右/后向 depth 方案。
- 无 odometry 长距离、多轮累计误差标定。
- 更大且房间之间地面连续的 Isaac Sim 场景验证。
- 最终 STOP 的长任务完整验证。

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

默认不生成虚假的 `1.5 m` 前方目标。bbox 附近没有有效深度时，机器人保持停止并 fail closed。

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

运动期间默认每 `0.1` 个仿真秒重新采集一次 forward RGB-D 并重新调用 iPlanner。它不是 30 Hz
视频流，而是同步的周期性请求。当前 iPlanner 神经网络主要使用 depth 和 point-goal，RGB 保留在
兼容协议中。

## 6. Pure Pursuit 和 odometry

当前控制参数默认值：

```text
walk_speed_m_s       = 0.3
lookahead_m          = 0.5
max_forward_speed    = 0.4
max_yaw_speed        = 0.5
safe_distance_m      = 0.5
replan_interval_s    = 0.1
```

测试时建议使用：

```text
goal_tolerance_m     = 0.2
blind_yaw_radius_m   = 0.2
```

Uni-LaViRA 原代码的 `goal_tolerance=1.0 m` 与 `safe_distance=0.5 m` 会让短轨迹第一帧就被判定
到达；原代码的 blind-yaw 注释写 `0.6 m`，实际值却是 `2.0 m`，会让近距离横向目标完全不修正
yaw。当前仿真测试使用 `0.2/0.2` 避免这两个问题。

### 无 odometry（默认）

默认不构造 odometry backend，不读取 Isaac 世界位姿。局部目标根据已经发送的速度命令更新：

```text
位移估计 = command_velocity * dt * dead_reckoning_scale
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

有 odometry 时，固定世界目标根据当前真实 root pose 重新转换到机器人局部坐标。`goal_x < -0.1`
保护只应用于无 odometry 航位推算；有 odometry 时使用真实二维目标距离继续跟踪，这一点与
Uni-LaViRA 的分支语义一致。

## 7. 目录内容

| 文件 | 作用 |
| --- | --- |
| `run_isaacsim.py` | Isaac Sim 完整运行入口 |
| `run_iplanner_local.sh` | 启动本机 Uni-LaViRA iPlanner Flask 服务 |
| `config.yaml` | 当前模型、相机、运动和 planner 参数的参考记录；运行时以 CLI 参数为准 |
| `unified_vln/model_contract.py` | 只读复用现有 schema-v2 模型协议 |
| `unified_vln/model_client.py` | 合并模型 multipart/PNG HTTP 客户端和 history |
| `unified_vln/types.py` | RGB-D frame、全景 bundle 等基础数据结构 |
| `unified_vln/local_projection.py` | bbox 底边中心、7×7 P30 深度和局部反投影 |
| `unified_vln/rotation.py` | 固定速度、固定时间相对转向 |
| `unified_vln/iplanner_client.py` | Uni-LaViRA iPlanner HTTP 协议客户端 |
| `unified_vln/local_trajectory.py` | 安全裁剪、Pure Pursuit、重规划和航位推算 |
| `unified_vln/odometry.py` | 可选 Pose2D odometry 接口与坐标转换 |
| `unified_vln/isaac_backend.py` | Isaac 四相机和显式 root odometry adapter |
| `unified_vln/episode.py` | 完整导航状态机 |
| `unified_vln/g1_dds_backend.py` | Unitree G1 `LocoClient.Move` 50 Hz DDS 后端骨架 |
| `convert_iplanner_checkpoint.py` | 官方完整模型 checkpoint 转换工具 |
| `verify_iplanner_checkpoint.py` | checkpoint CPU/CUDA 严格加载和推理检查 |
| `smoke_test_iplanner_http.py` | iPlanner HTTP 端到端 smoke test |
| `tests/` | 协议、反投影、转向、轨迹和状态机测试 |

`g1_dds_backend.py` 已实现惰性导入、`LocoClient.Move(vx, vy, wz)`、50 Hz 命令线程、状态订阅和
watchdog，但当前还没有接入完整真机相机和真机启动入口，因此不能把它当作已经完成的真机 runner。

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
  --lavira_history_max_decisions 3 \
  --local_goal_tolerance_m 0.2 \
  --local_blind_yaw_radius_m 0.2 \
  --local_safe_distance_m 0.5 \
  --local_replan_interval_s 0.1 \
  --local_use_isaac_odometry
```

`--no-four_rgbd_set_viewport` 使用第三人称视角观察机器人。删除该参数后，GUI viewport 会切到
机器人 forward RGB-D 相机。

每次实验必须使用新的 `--lavira_session_id`，避免远程模型继续使用旧 session 的历史。

## 10. 打开 Isaac Sim：默认无 odometry 单轮测试

下面的配置完全不读取 Isaac root pose，并且只执行一次模型决策，避免当前小场景中的机器人继续走到
门外。该流程今天已经得到：

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
  --instruction "Navigate to the doorway. Do not stop until the robot is near the doorway." \
  --lavira_session_id "doorway_noodom_test_NEW" \
  --lavira_server_url "http://127.0.0.1:18765/v1/lavira/decision" \
  --lavira_timeout 90 \
  --iplanner_url "http://127.0.0.1:8888" \
  --iplanner_timeout_s 5 \
  --lavira_history_max_decisions 1 \
  --local_goal_tolerance_m 0.2 \
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
  --lavira_history_max_decisions 1 \
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

每个 session 的输出目录：

```text
outputs/isaacsim_lavira_iplanner_g1/<lavira_session_id>/
```

| 文件 | 内容 |
| --- | --- |
| `decision_000_request.json` | 发给合并模型的元数据、instruction、history 和四图字段名 |
| `decision_000_response.json` | 模型原始 `action/direction/target/bbox` 回答 |
| `decision_000_projection.json` | bbox 像素、深度窗口、有效点数和局部目标 |
| `decision_000_plan.json` | iPlanner 原始轨迹、safe trajectory、safe goal 和 fear |
| `decision_000_completed.json` | 机器人完成动作后正式写入 history 的记录 |
| `failure.json` | fail-closed 原因 |

这些 JSON 仅用于调试和实验审计，运行时不会重新读取它们。旧 session 文件夹可以删除，但删除后无法
恢复其中的实验记录；使用新的 session ID 可以避免覆盖和混淆。

## 14. 今天的实测结果

### odometry：`doorway_odom_test_002`

成功完成：

```text
decision 0: right → rotate → execute → stand → completed
decision 1: forward → execute → stand → completed
history=2
```

第三轮模型继续选择门洞，但 bbox 底边中心 `(252, 336)` 没有有效深度：

```text
No valid selected-view depth around pixel (252, 336)
```

这不是运动控制失败。机器人已经靠近当前小地图出口，门洞后没有连续有效几何，系统按设计停车。

### 无 odometry：`doorway_noodom_test_001`

成功完成一轮：

```text
state=stopped history=1 failure=None
```

该轮数据：

```text
模型目标距离约 0.833 m
iPlanner 原始轨迹长度约 1.036 m
安全裁剪轨迹长度约 0.537 m
safe goal 约 [0.534, 0.054] m
```

它证明无 odometry 单步逻辑已经跑通，但目标较近，尚不能证明 3～5 米长距离航位推算精度。

## 15. 与 Uni-LaViRA G1 真机代码的关系

基本一致：

- 局部坐标 `x forward / y left`。
- bbox 底边中心、7×7 深度窗口和 P30。
- `goal_x=Z`、`goal_y=-X_camera`。
- iPlanner `/navigator_reset` 和 `/pointgoal_step` 协议。
- forward depth + point-goal 局部规划。
- 末端 `0.5 m` 安全裁剪。
- Pure Pursuit、lookahead、速度限制和连续重规划结构。
- 无 odometry 命令积分和可选 odometry 分支。
- 固定角速度/时间开环转身。
- G1 高层速度接口 `LocoClient.Move(vx, vy, wz)` 和 50 Hz 发送方式。

按本项目要求有意不同：

- Uni-LaViRA 原版是 LA 选方向、转身、VA 再给 bbox。
- 本项目保留已经调好的合并模型，一次返回 `action + direction + bbox`。
- 本项目使用转身前所选方向的 bbox+depth，转身后只用最新 forward depth 规划。
- 不执行 FMM 和 backtrack。
- 缺少有效深度时不使用 Uni-LaViRA 的虚假 `1.5 m forward` fallback，而是安全停止。

真机部署前仍需完成：

- Uni-LaViRA 原始相机代码只有 front 可靠提供 depth，left/right 默认只拍 RGB，behind 是 V4L2 RGB。
  当前一步模型需要所选方向的 depth，因此必须补齐真机多方向深度或设计明确替代方案。
- 将 `UnitreeG1DDSBackend` 接入完整 episode runner。
- 增加真机相机序列号、内参、RGB-depth 对齐和新帧检查。
- 处理同步 iPlanner 重规划与 DDS watchdog 的命令持续发送关系。
- 分别标定真机旋转时间系数和无 odometry 线性/角速度系数。

## 16. 测试和诊断命令

运行全部新目录测试：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
python -m pytest -q scripts/isaacsim_lavira_iplanner_g1/tests
```

当前结果：

```text
14 passed
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

所选 bbox 附近没有深度，属于安全停止：

```text
No valid selected-view depth ... no forward fallback generated
```

iPlanner 请求或重规划失败：

```text
iPlanner replan failed
```

无 odometry 航位推算认为已经越过目标：

```text
local goal was passed without odometry (x=..., y=..., distance=...)
```

Isaac/RTX/Fabric 的 performance warning、DLSS 分辨率 warning 和 URDF fixed-link warning 通常不是本导航
流程失败原因，应优先查看最后的 `[LOCAL-VLN ERROR]` 和 `failure.json`。
