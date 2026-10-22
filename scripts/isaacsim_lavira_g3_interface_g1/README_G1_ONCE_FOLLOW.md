# 复用仿真实现进行一次真机局部跟随

完整多轮链路使用[按pelvis假设的正式入口](README_G1_PELVIS_VLN.md)。本页仍是单次诊断。

## 本轮使用：单次功能诊断入口

`g1_once_follow_probe.py` 直接调用下表中的既有目标投影、iPlanner、路径截短与
`LocalTrajectoryFollower`，没有新写跟随控制律。旋转采图与一次模型请求也调用现有
`g1_rotation_model_probe.py`。本入口不创建完整G3 Episode，不发送地图进展/执行上报，
模型会话仍按原采图探针结束，避免把短程诊断写成完整导航任务成功。

机器人相机支架固定后，先保持D415服务、iPlanner、模型隧道、SLAM重定位运行，打开
`g1_iplanner_viewer.py`。笔记本另开终端：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source /home/yile/projects/unitree_ros2/setup.sh
source /home/yile/projects/.venvs/unitree_g1/bin/activate

# 默认只检查相机、模型、iPlanner和SLAM，不创建运动后端。
python scripts/isaacsim_lavira_g3_interface_g1/g1_once_follow_probe.py
```

以下命令**由用户手动执行，会使真机旋转和行走**：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_once_follow_probe.py \
  --network-interface enxf8e43bbea486 \
  --instruction "向前走到一个黑色的行李箱前" \
  --follow-seconds 4 \
  --execute
```

### 完成本次局部跟随

用户手动完成短测后，可以改用下列命令，取消短测的0.5m位移截止及4秒截止。
**该命令会使真机旋转和行走，只由用户手动执行。**

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_once_follow_probe.py \
  --instruction "向前走到一个黑色的行李箱前" \
  --complete-local-follow \
  --execute
```

`--complete-local-follow` 沿用原跟随器的局部到达条件，默认保留正式真机入口原有的
60秒动作超时，相机/低层状态/SLAM过期或定位跳变仍请求停车。没有设置该标志时仍是
原短测模式，默认3秒/0.5m；完整局部跟随模式不接受 `--max-travel-m`。
该模式可用 `--follow-seconds` 指定不超过60秒的动作超时，命令中不要带旧的
`--follow-seconds 4`，否则仍会在4秒触发超时。

它持续获取深度并重规划，跟随原来的局部目标，不请求第二次模型决策，不是完整VLN任务。
仍保留路径末端约0.5m截短、1.0m到达容差及原越过目标检查，因此不会精确走到原始
iPlanner最后一个路径点。正常由原到达条件结束记录为 `LOCAL_GOAL_REACHED`；
`TIME_LIMIT` 是动作超时，不能记作到达。0.6m/s期望速度、原转向减速、DDS发送方式不变。
报告中的 `completion_mode=local_goal`、`follow_limits.slam_displacement_m=null`
标识本模式；`full_g3_execution_reporting` 仍为false。

顺序：四段旋转采图 → 一次模型决策 → 停车显示决策框2秒 → 新前视深度规划 → 原跟随器
短程执行 → 停车退出。运动开始后清除旧模型框；窗口读取 `follow_live.json` 显示本次实际
跟随路径，而不是另外规划的预览路径。停止或超过0.5秒未更新时隐藏执行路径。
停止机器人要在运动终端按 Ctrl+C；监控窗口的Q只关闭监控。

按用户要求，本真机诊断入口传给原Pure Pursuit的期望前进速度与前进上限均设为0.6m/s，
对应 `target_speed_m_s` 和 `max_forward_speed_m_s`；共享配置和仿真的默认值不改。
原转向减速仍生效，不强制每条速度命令都是0.6m/s。最大yaw 0.5rad/s、lookahead 0.5m、
到达容差1.0m、blind-yaw半径1.5m保持原值；不增加单独的起步动作。
0.6m/s尚待真机验证。用户选择用已有 `--follow-seconds 4` 测试，4秒是上限，
原0.5m SLAM位移限制仍有效，因此不保证实际跟随满4秒。
测试外层默认在行走达到3秒或SLAM距起点位移达到0.5m时请求停车；这不是独立实测距离、
不是累计里程，也不保证物理刹停严格小于0.5m。可用 `--follow-seconds` 在0～6秒内设置，
`--max-travel-m` 上限仍为0.5m。`--decision-display-s`设置停车显示时间，默认2秒。

首次只接受NAVIGATE/forward。侧后方、STOP、恢复/SAFE_STOP、缺少有效目标深度或无路径
不进入行走；不执行原无深度1.5m兜底。取图/规划失败、低层状态或SLAM过期会请求停车。
独立watchdog在相机/HTTP同步等待期间也检查限制，限额触发后不允许旧速度恢复。
这是诊断入口的执行保护，不更改共享的跟随算法。

本轮仍使用仿真的名义相机前/左目标假设，不应用安装外参、不补偿首尾视角偏差，
不构成物体到达或避障验证。完整G3的几何配置仍按下文单独处理，不是本次控制功能检查的前置条件。

只回放保存的图像、调用iPlanner并计算第一帧速度，不初始化ROS/DDS：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_once_follow_probe.py \
  --replay-run outputs/d415_rotation_model/run_20261016_162406_8bbae1ab
```

每次自动新建 `outputs/d415_rotation_model/once_.../`。子目录 `rotation_model/` 保存模型和
采图，`initial_plan.json`保存首次计划，`latest_replan.json`保存最新重规划，
`follow_trace.json`保存请求速度和SLAM轨迹，`follow_report.json`记录停止原因。
`STOPPED`只代表诊断停止，不是整个语义任务完成；`robot_motion_sent`在旋转中断时可能为null，
表示是否已发送速度未知，不能把中断报告误记为完全没动。

已完成只读实机连接检查和记录图像回放。新增测试验证默认不创建运动后端、回放不接ROS/DDS、
限额/失效停车、同步重规划期间触发停车后不恢复速度，以及实际路径显示与停止清除。
本轮尚未代用户启动真实行走。

### 2026-10-16 用户手动实测与起步取图修正

- `once_20261016_171431_bb3aa14f`：旋转/模型请求完成。框底边到达720p图像底边，
  取点 `(536,719)` 的28个深度样本全部为0，因此拒绝1.5m无深度兜底，没有进入行走。
- `once_20261016_172007_347ea05a`：选中目标深度3.124m，49个样本有效；
  iPlanner成功返回路径。记录到两条非零速度请求，约为 `(0.294,0,0.073)`，
  随后以 `DATA_FAILURE: camera input stale` 请求停车。这不能记作短程跟随通过，
  两条请求也不能证明机器人实际走了多远。
- 排查发现首次规划之后还要初始化DDS和低层订阅，原代码直接沿用规划图的接收时间，
  起步时可能已接近1秒超时。现在DDS/订阅器就绪后实际重新收一帧，再检查规划位姿
  是否变化，之后才允许进入跟随；记录 `follow_ready_frame_id` 和图像年龄。
  不重置旧帧时间戳，不放宽1秒断流阈值；重规划也使用接收器记录的真实接收时间。
- 只读复查D415连续6帧均为1280×720，取图约31～68ms，未发送运动命令。
  新增离线测试覆盖初始化耗时、旧帧不能伪装成新帧、断连和准备期间位姿改变。
  修正后的真实跟随仍需用户手动执行验证。
- 本次模型响应的 `stage_progress.parse_success=false` 是阶段进度解析失败。
  原 `episode.py` 对此警告并保留有效Navigator动作，本入口沿用该行为；
  不能把这条进度解析错误解释成任务已完成，也不能把它当作相机超时原因。
- `once_20261016_172643_faf20731`：起步新图检查通过，约3秒后以 `TIME_LIMIT`
  请求停车，共记录15次跟随器速度更新，前进请求0.279～0.296m/s。SLAM记录首末位置
  差约0.038m，用户确认没有观察到行走，不记为实际跟随通过。随后仅在本入口把上述
  两个速度参数提至0.6m/s，其余控制算法、DDS发送方式和停车限制保持原样。

## 完整状态机的复用关系

本次准备不新增控制器，不修改 `run_isaacsim.py`、`episode.py` 或 `local_trajectory.py`。
正式真机入口已经是 `run_g1_real.py`，使用 `RealG1Episode`；其导航和跟随继承仿真创建的
`LocalEndToEndEpisode`，真机子类只替换全景旋转采集及只读显示记录。

## 复用关系

| 步骤 | 仿真和真机共同实现 |
| --- | --- |
| 模型框＋选中方向深度投影 | `unified_vln/local_projection.py: project_selected_view_target()` |
| 模型动作、转向、规划、执行、停车状态 | `unified_vln/episode.py: LocalEndToEndEpisode` |
| iPlanner 请求 | `unified_vln/iplanner_client.py: IPlannerClient` |
| 路径末端保留距离 | `unified_vln/local_trajectory.py: truncate_trajectory_for_safety()` |
| 路径转速度、到达和越过目标检查 | `unified_vln/local_trajectory.py: LocalTrajectoryFollower` |
| 真机数据与执行装配 | `run_g1_real.py: run()`；ROS 2 SLAM、D415 TCP、G1 DDS |

沿用当前默认 `pure_pursuit`。`kp` 是项目原本已有的可选分支，不能把切换控制器记作
“与默认仿真完全一样”。第一轮不改跟随规律；速度、到达容差、lookahead、blind-yaw 等参数
要显式记录，不能只保留一个蓝色预览图作为追踪证据。

## 一次的含义

完整 `run_g1_real.py` 使用既有 `--max-decisions 1`：采全景、请求一次高层决策、执行该局部动作、停车收尾，
不进入第二轮全景/决策。一个局部动作期间仍按原状态机重新取前视深度并调用 iPlanner，
并非只规划一次后开环发送速度。STOP、无路径或失败可能导致没有行走，不能强制改成前进。
决策次数上限结束不代表整个导航任务成功。

原参数默认 `safe_distance_m=0.5`、`goal_tolerance_m=1.0`，后者是距截短后局部终点的容差，
不是实际距物体1米。`blind_yaw_radius_m=1.5` 范围内 Pure Pursuit 不继续修正yaw。
这都是既有行为，本轮没有另写一套到达或转向算法。

## 当前真机配置状态（2026-10-16）

- D415仍在之前标定的位置；用户说明支架尚不牢，会有少量偏移。相机随身体运动属于位姿变化，
  支架相对胸部滑动则会改变安装外参。实际行走前需要固定支架。
- RGB光心离地约1.10米，为用户粗测的静止名义高度，不能视为动态测量。
- 已读取 `/unitree/slam_relocation/odom`，消息声明 `map → base_link`。
- 已有相机候选外参的参考是 `torso_link`。URDF根为 `pelvis`，不能仅凭消息里的 `base_link`
  名称就认定它等于torso或pelvis。还需要核对SLAM实际基座定义与相机到该基座的关系。
- 正式G3入口的 `--map-config` 用于RGB-D探索地图证据，要求真实相机/基座几何、
  基座名义高度与地面参考。尚未生成完整G3地图监督配置；不会把仿真默认几何填成真机标定。
- `local_projection.py` 的局部目标沿用仿真的名义前/左轴假设，不应用完整安装外参。
  `--map-config` 也不是对该局部投影做完整六自由度校正，两个用途须区分。

## 完整G3参数模板：不是本轮短程诊断命令

下面是原入口的参数模板，`MAP_CONFIG_REQUIRED.json` 尚未准备好，当前不能直接运行。
命令会使真机旋转和行走，助手不代为启动。

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source /home/yile/projects/unitree_ros2/setup.sh
source /home/yile/projects/.venvs/unitree_g1/bin/activate

python scripts/isaacsim_lavira_g3_interface_g1/run_g1_real.py \
  --instruction "向前走到一个黑色的行李箱前" \
  --model-url http://127.0.0.1:18765/v1/lavira/decision \
  --iplanner-url http://127.0.0.1:8888 \
  --network-interface enxf8e43bbea486 \
  --camera-factory camera_d415.network_camera:create_camera \
  --camera-config scripts/isaacsim_lavira_g3_interface_g1/camera_d415/network_camera_d415_local.json \
  --odometry-topic /unitree/slam_relocation/odom \
  --map-config MAP_CONFIG_REQUIRED.json \
  --rotation-duration-scale 1.4 \
  --tracking-controller pure_pursuit \
  --max-decisions 1 \
  --action-timeout-s 6
```

6秒是拟用于首次短时测试的参数限制，算法原默认60秒；它不限制为准确0.5米，也不保证物理刹停距离。
`--rotation-duration-scale` 不改变真机全景的IMU75°提前停车逻辑；SLAM可用时模型选定方向的转向也使用SLAM反馈。
保留原来的 Ctrl+C 先停车再清理流程，不启动 `g1_rotation_model_probe.py` 与此入口并行运控。

## 已完成的只读/离线检查

使用 `run_20261016_162406_8bbae1ab` 保存图像，调用原选中视图投影、真实本机iPlanner、
原路径截短函数和原Pure Pursuit跟随器计算第一帧速度。没有创建运动后端或发送命令，
不是实际跟随通过。证据：
`outputs/g1_once_prepare/run_20261016_164524/shared_modules_check.json`。
跟随器现有17项单元测试通过；未启动Isaac仿真或真实行走。

实时RGB监控可以保持打开。正式任务建议用 `--model-results` 绑定本轮真实输出目录。
没有活动的单次跟随诊断时，监控窗口的路径是独立规划预览，不是完整正式状态机实际发出速度的追踪证明；实际判断应读取
正式入口的 `decision_XXX_plan.json`、状态转换、SLAM执行轨迹和停车记录。
