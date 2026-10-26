# SLAM基座为pelvis的完整真机G3

2026-10-20用户通过其他方法确认：`/unitree/slam_relocation/odom` 中的
`base_link` 对应URDF的 `pelvis`，正常站立时骨盆原点离地约0.80m（用户更正，先前0.90m不再使用）。
身份确认来源记录为用户报告，不声称本程序已取得或独立验证TF。
不发布假的TF，不修改机器人SDK/相机库，不修改仿真入口或共享Episode、跟随器、稀疏地图算法。

## 入口与文件

| 文件 | 作用 |
| --- | --- |
| `g1_pelvis_vln.py` | 默认只读检查；显式`--execute`才启动原`run_g1_real.py` |
| `camera_d415/pelvis.json` | 基座身份确认来源、相机配置、标定来源及约0.80m骨盆离地高度 |
| `camera_d415/pelvis_camera.py` | 逐帧配对图像与腰关节；URDF/FK合成pelvis外参；原探索地图的坐标适配 |
| `unified_vln/real_input_guard.py` | 真机外层失效停车，阻塞等待HTTP时也检查；不计算导航速度 |
| `run_g1_real.py` | 原正式入口，仅在选用该相机工厂时装配上述适配器 |
| `unified_vln/real_episode.py` | 原真机全景流程，加只读的实际跟随路径输出和运动后隐藏框 |
| `unified_vln/real_navigation_rotation.py` | 导航选定方向复用全景IMU75°停车策略；实际转角核验后才允许规划前进 |
| `unified_vln/real_run_records.py` | 真机执行追加日志、原子JSON保存及HTTP前请求记录 |

## 如何换算

沿用已保存、重处理后的手触碰标定候选：
`docs/real_robot/2026-10-13/hand_candidate.json`。
它明确标注 `CANDIDATE_REQUIRES_PHYSICAL_VALIDATION`，指尖模型偏移仍未实物核实。
确认SLAM基座身份不会把相机外参候选升级成已验证结果。

```text
T_pelvis_camera(frame) = T_pelvis_torso(q_waist(frame)) × T_torso_camera
```

`q_waist` 读取 `rt/lowstate`，使用原 `camera_extrinsics/robot_state.py` 的关节映射和
`kinematics.py` 的正运动学。每次取图按本机接收时间找最近关节样本，接收时间差最多50ms，
记录帧号、腰角、矩阵和时间差；这是接收配对，不声称曝光硬同步。
地图积分使用该图像自己的矩阵缓存，不能拿之后的腰角替代先前图像。

动态换算只应用到G3的RGB-D探索地图坐标变换。仍沿用原SLAM平面x/y/yaw和固定名义基座
高度，不使用SLAM的z/roll/pitch；不是完整动态六自由度世界模型。地面z=0是该简化模型
的竖直参考，不是实测SLAM地图地面。现直接使用用户更正的骨盆离地约0.80m：

```text
nominal_base_height_m = floor_z_world_m + pelvis_height_above_floor_m
camera_height_above_floor_m(frame) = 0.80 + T_pelvis_camera(frame)[2,3]
```

不再用先前约1.10m的相机高度反推骨盆高度。例如相机高于骨盆约0.35m时，
模型推算相机离地约1.15m；这是新骨盆高度与既有外参的合成结果，不是重新实测的相机高度。
若与实物明显不符，需要核对实际坐标原点、安装状态和外参，不能把两份粗测同时强制套用。
骨盆0.80m只作为正常站立的固定近似，不是行走时动态测高；未补偿骨盆整体roll/pitch/z。

2026-10-20该修订通过260项接口离线回归，包含骨盆高度直接配置、非零地面参考下的
高度筛选和默认不运动检查。身份确认和0.80m粗测不作为完整导航成功证据。

**局部导航目标投影仍沿用原名义相机前/左轴，没有改写`local_projection.py`，也没有
把完整外参强行塞进原iPlanner跟随算法。** 地图证据坐标换算和局部目标投影是两个用途。
真实planner入口沿用单轮测试已验证的0.1～5m深度过滤，再交给同一个iPlanner客户端，
避免较远深度在旧uint16毫米协议中回绕。

## 先只读检查

保持现有D415 TCP服务、模型SSH隧道、iPlanner和SLAM重定位运行。

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source /home/yile/projects/unitree_ros2/setup.sh
source /home/yile/projects/.venvs/unitree_g1/bin/activate

python scripts/isaacsim_lavira_g3_interface_g1/g1_pelvis_vln.py
```

此命令不创建运动客户端、不调用HighStand、不启动模型Session，也不旋转。
只读相机、关节和SLAM，检查G3/iPlanner健康并积分一帧地图。
输出到 `outputs/g1_pelvis_vln_prepare/check_.../`，包括：

- `preflight_report.json`：组件状态与`robot_motion_sent=false`。
- `geometry_assumptions.json`：基座确认来源、标定候选状态、来源哈希、URDF、初始矩阵和近似高度。
- `camera_metadata.json`：当前帧对应的腰角、外参及接收配对时间差。
- `map_config.json`：供原正式入口使用的启动参数；运行中该相机工厂继续按帧更新外参。
- `runner_command.json`：复用原正式入口的结构化argv；不会在只读检查中执行。

2026-10-16已完成真实只读检查：720p D415、G3 b92abb3、GPU iPlanner、map→base_link和
腰关节可用；一帧11387个有效深度样本积分得到5939个探索格。
证据为 `outputs/g1_pelvis_vln_prepare/readonly_pelvis_01/`。当时使用尚未确认的基座身份及
旧相机高度推算，历史报告不改写，也不代表当前0.80m骨盆高度已实测验证。
这是连接与坐标计算通过，不是真实多轮导航通过。
最终复查时网卡`enxf8e43bbea486`为DOWN且没有IP，检查在DDS只读初始化前后失败并退出，
未启动运动。记录为 `outputs/g1_pelvis_vln_prepare/readonly_pelvis_final_01/`。
用户随后确认机器人已断电；上述最终复查失败记录不作为SDK/相机代码故障。
新增实现通过258项接口离线回归（含显示等待、逐帧外参、默认不运动和失效停车），
完整多轮真机运行仍由用户下次开机、连好网络后手动验证。

2026-10-20重新连机准备：D415 720p、关节订阅、远端G3健康接口及GPU iPlanner可用，
一次实时深度纯规划返回51个路径点，未发送运动命令。当时更正前配置的名义骨盆高度0.90m，
既有外参和当前腰角推算相机离地约1.235m；这不是实际测高。
用户随后将骨盆高度更正为0.80m，当前配置和离线测试已同步；该旧采样记录不改写。
重定位和建图里程计话题可发现，但只读订阅均未收到实际消息，正式只读检查因此未通过。
需用户启动SLAM重定位并在RViz核对位置，然后重跑只读检查；尚未启动完整运动链路。

## 行走时持续更新 Map Progress

正式入口已经复用仿真的持续融合流程，不是仅在出发前建立一帧地图：

```text
行走跟随 → 到达重规划时机 → 取新前视RGB-D + 当前SLAM位姿
→ 用该帧腰关节FK和相机外参转换深度点 → 累积到同一张探索地图
→ iPlanner重规划 → 继续原Pure Pursuit跟随
```

`Episode._capture_forward_observation()` 每次成功取图都会尝试融合地图；
`PelvisExplorationMap` 只适配真机几何，共享地图和跟随算法不变。
默认重规划间隔0.1秒是尝试间隔，取图、规划、日志和网络耗时会降低实际融合频率，
不是保证10Hz，也不是相机15fps每帧都融合。图像与SLAM使用现有读取流程，尚未做曝光时刻的位姿插值。

默认约每1秒动作时间生成一次 `motion_window`：保存探索格总数、新增探索格、占用格、
更新次数及地图失败计数，并上报原G3 Physical Monitor。统计窗口可能因阻塞延后。
地图是进度证据，不替代iPlanner避障输入；地图融合异常按原状态机累计并打印警告。
连续新图/定位/关节失效的停车由真机输入保护负责，不能把单纯地图投影失败解释为已经触发停车。

真机验证时查看：

- `[LOCAL-VLN MAP] integrated forward RGB-D`：帧号推进，`explored`累计变化。
- `[LOCAL-VLN MAP] motion window`：每个窗口的`new`和累计探索量。
- 正式输出中的 `map_progress_decision_*_window_*.json` 与 `map_progress/` 图片：
  `update_count`持续增加、`map_update_failures`理想为0，并检查墙面/障碍物有无明显重影。

重复看到同一栅格不会重复增加探索量，因此`new=0`不等于地图没有更新；
实物静止时传感器噪声、SLAM抖动仍可能产生少量新格。
该地图累积已探索和障碍占用，不具备动态障碍物移走后的完整清除机制。

2026-10-20补充连续帧离线验证：相同观测不重复计增量、位姿移动后地图累积新区域，
并保留之前格子；接口离线回归259项通过。该验证使用合成深度和位姿，无硬件运动；真机边走边更新仍待现场验证，
随后用户确认`base_link=pelvis`，骨盆离地高度最终更正为约0.80m，当前配置已同步；
旧测试记录保留原状。之前的`g1_once_follow_probe.py`单次前进测试不包含这套地图统计。

## 实时监视窗口

关闭旧观察窗口后，在单独终端运行以下只读命令，让窗口监听正式入口目录：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_iplanner_viewer.py \
  --model-results /home/yile/projects/unitree-g1-isaaclab-project/outputs/g1_real_unified_vln
```

每次新决策更新实时RGB框，导航转向和规划完成后停车显示2秒；开始跟随后隐藏旧框，
不跟踪物体。显示等待是本入口传入的`--decision-display-s 2`，不改目标或速度算法。
正式NAVIGATE跟随时读取`follow_live.json`，显示原跟随器持有的局部路径；蓝线不是
另外一次预览规划。标签为`FORMAL FOLLOWER - VIEWER IS READ ONLY`。
路径数据是跟随器请求依据，不是SDK执行确认；停止或数据超过0.5秒时隐藏。
BACKTRACK及没有正式执行路径时，不把独立预览当作实际执行证据。
窗口的Q只关闭观察窗口，不停止另一终端的任务。

## 每次决策保存目标框和位置

2026-10-20起，正式真机`RealG1Episode`接收决策后自动生成以下辅助文件：

- `targets.html`：直接用浏览器打开，按决策顺序显示中文目标、所选方向、阶段判断、
  原始像素框、实际深度采样点、深度及名义局部目标坐标，方便对照阶段与Navigator是否一致。
- `decision_000_target.png`等：在模型选择方向的原图上绘制绿框；有实际投影时用黄色十字和
  小框标注深度采样位置。上方说明栏不计入原图像素坐标；中文目标完整保存在HTML/JSON中。
- `decision_000_target.json`等：记录源视图、源帧号、原始/裁剪框、实际投影及客户端是否接受导航。

STOP、BACKTRACK、PREEMPT等也保留对应决策记录；没有局部投影时不伪造深度位置。
这些是模型原始选择的可视化，不纠正错误框、不表示目标识别或地图位置已经验证。
原图、模型响应和投影JSON保持原状；不改变仿真入口、目标投影、跟随算法或控制协议。
辅助输出失败时打印`[REAL VLN ARTIFACT WARN]`，不掩盖原导航异常。

已有运行可以仅用保存文件补齐，无需连接机器人、相机或模型：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_g3_interface_g1
/home/yile/projects/.venvs/unitree_g1/bin/python -m unified_vln.real_decision_artifacts \
  --run /绝对路径/到/已保存的run目录
```

离线补齐的记录标记`reconstructed_from_saved_observation`，`navigation_accepted=null`表示
不从图片反推是否实际执行；应查看原执行报告。新功能下次启动正式真机入口即生效。

## 终端实时摘要

正式真机入口默认输出中文字段摘要，无需额外参数；下一次启动程序生效。
实现为`unified_vln/real_console.py`，仅格式化观察数据，不增加语义控制规则或修改模型响应。

- 启动时`[任务阶段]`列出服务器冻结的各个子任务。
- 每次`[模型返回]`显示decision编号、Navigator/Recovery来源、动作、方向、目标和control。
  `[目标框]`显示原始bbox的xyxy坐标、对应方向原图的帧号和尺寸，不套用到别的方向画面。
- `[阶段进度]`显示服务器parse_success、完成数/总数、当前stage_id、阶段文字、完成ID；
  面向用户显示“第2阶段”时协议ID仍为1。解析失败单独显示`[阶段解析错误]`及原错误，
  不把缺失字段猜成第1阶段。解析成功只表示结构有效，不证明物理任务已经完成。
- `[模型进度说明]`和`[模型理由]`保留模型原文；存在时继续显示STOP Gate、Failure Verifier、
  Semantic Audit和Recovery结果。`[执行判定]`区分收到模型消息与客户端接受执行，
  PREEMPT/SAFE_STOP明确显示保持停车；`[目标投影]`显示实际深度、有效点数及是否使用兜底。
- `[地图窗口]`显示本窗口新增探索cell、累计探索、占用、可通行、融合次数/失败数及分辨率。
  这些cell来自RGB-D探索地图，不是iPlanner直接输出；新增0也如实显示，不代表停止融合。
- `[执行窗口]`显示实际窗口时长、SLAM首尾位移、局部目标距离变化、上报耗时和Physical Monitor
  返回的control/候选。CONTINUE不表示目标识别正确或已经证明无碰撞。
- `[iPlanner规划]`约每秒最多显示一条成功规划摘要：调用号、输入帧、局部目标、路径点数、
  fear和调用耗时；无有效路径时立即显示。逐次耗时仍全部写入执行日志，跟随器仍按原频率调用。

例如从本次已保存数据离线格式化可看到：

```text
[模型返回] decision=3 来源=RECOVERY 动作=NAVIGATE 方向=left(左方) 目标="table area" control=CONTINUE
[目标框] decision=3 bbox_xyxy=[320,468,960,706] 所选原图=left 尺寸=1280x720 帧=9159
[阶段进度] decision=3 解析=成功 已完成=1/2 当前=第2阶段(stage_id=1) 内容="向左转走到绿色瓶子前停下" 完成ID=[0]
[地图窗口] decision=3 window=0 新增探索cell=22 累计探索=25109 占用=11380 可通行=2907 融合次数=40 分辨率=0.05m 融合失败=0 来源=RGB-D探索地图
```

该示例是历史记录的只读格式化，不表示重新运行或修正了当时的决策。
从`g1_pelvis_vln.py`启动时，上述输出也写入预检查目录的`runner_console.log`。
终端格式化或输出异常不替换原导航返回值和异常；未修改共享仿真代码或原规划/跟随算法。
新增摘要功能通过298项接口离线测试，本次未启动真机运动。

## 人工停车与记录保留（2026-10-20）

运行任务的终端按 **Ctrl+C**，正式程序先调用原DDS停车，再保存本地退出信息、
关闭资源，最后尝试结束模型会话。父入口等待子进程收尾，避免`subprocess.run()`
在KeyboardInterrupt后杀死尚未保存完的子进程。收尾期间重复Ctrl+C不会再次打断保存。
SIGTERM也进入正式程序的同一收尾流程；人工停止以FAILURE/user_interrupt结束远端会话，
不冒充任务语义成功。停车调用返回只表示软件请求已返回，不是物理急停确认。

随后补充：信号处理器在打印及异常退栈前，先设置不可恢复的关停锁存，禁止原DDS适配器
和输入保护接收新的非零速度，并唤醒独立输入保护线程请求零速度/StopMove。
即使主线程等待规划或上报，保护线程也能执行停车；主线程收尾与该线程共用一次关停调用。
`real_input_guard.json`记录信号接收、目标归零及停车调用的起止时刻。原控制速度、Pure Pursuit
和仿真代码不变，也没有修改机器人原SDK库。已进入SDK/网络的旧请求不能被该标志撤回，
Python调度、DDS/RPC及机器人减速仍有时延；没有实测真机停车延迟上限，Ctrl+C不等同于硬件急停。

在正式运行目录`outputs/g1_real_unified_vln/<session>/run_.../`新增：

| 文件 | 保存内容 |
| --- | --- |
| `execution_events.jsonl` | 逐条追加控制周期、速度请求、SLAM位姿、实际跟随路径、窗口请求/响应和退出事件；每行立即flush，约每秒及收尾时fsync |
| `execution_tail.json` | 尚未完成动作的世界轨迹、最后窗口起止位姿与地图计数；不足1秒也保存，标记partial且不上传为完整窗口/完成动作 |
| `run_exit.json` | 退出原因、异常、停车请求时间与返回情况、最终可用位姿及来源/年龄、资源清理错误、远端结束确认状态 |
| `g3_decision_*_motion_*_request.json` | 调用窗口HTTP之前先保存请求参数；即使等待响应时中断，请求也保留 |
| `g3_decision_*_action_complete_request.json` | 调用动作完成HTTP之前保存的原请求参数；不表示服务器已收到 |
| `g3_session_end_request.json` | 实际尝试发送的会话结束状态和原因 |
| `g3_session_end_attempt.json` | 远端确认结果，或网络异常下的UNCONFIRMED及错误信息 |
| `timing_summary.json` | iPlanner原调用、窗口HTTP及实际窗口跨度的次数、均值、最大值和p95；逐次数据在execution_events.jsonl |

原RGB、深度、决策、目标框、投影和地图文件仍保留。真机JSON使用同目录临时文件再原子替换，
避免中断写入覆盖已有完整JSON。日志不是所有相机帧的录像，记录的是原流程采集/保存的图像
和每轮控制证据。停止时拿不到新SLAM位姿会保留最后记录并明确标记缓存来源，不伪造最终测量。

本地`run_exit.json`先以`LOCAL_RECORDS_SAVED`落盘，再关闭资源和发送结束HTTP；
处理结束后更新为`CLEANUP_FINISHED`。结束HTTP的socket超时上限调整为5秒，网络不可达时
保留本地数据和UNCONFIRMED，不生成假的`g3_session_ended.json`。
从`g1_pelvis_vln.py`启动时，预检查目录额外保存`runner_console.log`，保留正式程序终端输出。

真机Motion Window改为按首尾SLAM位姿的本机接收时间差达到配置间隔后再触发，
修复进入执行前的循环耗时导致首窗口过短的问题。阻塞造成的窗口延后仍按实际时间记录，
不声称严格1Hz，不补发伪造的等长窗口；共享仿真Episode、跟随算法和协议不改。
`iplanner_call_s`测量格式适配和同步规划客户端调用总耗时，不能当作纯GPU推理时间。
`motion_window_span_s`是证据窗口的真实时间跨度，`motion_window_http_s`是上报调用耗时，
二者分开统计。规划与上报本轮仍保持原同步方式，因此实际窗口延后尚未全部消除。

记录收尾修改通过278项接口离线测试；随后新增停车锁存和耗时统计测试，没有启动真机运动。
历史缺失的停车/远端确认记录
无法凭空恢复；这些新增记录在下次启动正式程序后生效。SIGKILL、断电或磁盘故障不能保证
完成退出收尾，已落盘文件保留到最后成功写入的位置。

## 用户手动启动完整链路

**以下命令会启动真机运动，只由用户手动执行。**

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_pelvis_vln.py \
  --instruction "向前走到一个黑色的行李箱前" \
  --execute
```

先重新做只读检查，通过后关闭检查资源，再以独立进程调用原正式入口：

```text
start_session → 原HighStand → 四段转圈采图 → 模型决策 → 选定方向转向
→ iPlanner重规划与原Pure Pursuit跟随 → motion_window / action_complete
→ 下一轮观察和决策 → STOP_CONFIRMED或失败 → 停车、end_session
```

期望前进和上限均0.6m/s，最大跟随yaw仍0.5rad/s。末端截短0.5m、到达容差1.0m、
全景0.8rad/s及IMU75°提前请求停车、每次动作60秒超时保持原实现。
2026-10-20本次实机记录显示旧导航转速0.4rad/s持续约12.75秒、SLAM只转约5.7°，
旧旋转器反馈超时返回done=True后，上层仍进入前进。历史记录保持原状。
按用户要求，当前正式入口显式传入`--rotation-speed-rad-s 0.8`，真机runner默认也改为0.8。
导航左/右转复用四方向采图的顺序：

```text
开始实际转向时冻结IMU/SLAM基线 → ±0.8rad/s → 定向IMU累计75°
→ 请求停车 → 停车调用返回后等待0.5秒 → SLAM本段实际转角70°～110°
→ 整体朝向误差不超过20° → 新前视图 → iPlanner → 原Pure Pursuit
```

前方不转；左方一段左转；右方一段右转；后方按原右转约定拆成两段右转，
每段独立停车和核验。IMU/SLAM丢失、反向转动、每段10秒超时、停稳后角度不符，
均失败停车，不能把超时当完成，也不回退到开环计时。75°提前停车是经验补偿，
不是精确90°保证；70°～110°核验范围沿用全景，整体目标容许误差20°。
`decision_XXX_navigation_rotation.json`保存方向、转速、IMU/SLAM角度、分段结果和失败原因。
终端会先显示`IMU early-stop`，再显示`navigation left/right verified ... planning allowed`。
真机默认转速变更不改变仿真类、iPlanner或Pure Pursuit算法；跟随时最大yaw仍0.5rad/s。
新增实现通过291项接口离线回归，含假IMU/SLAM的提前停车、停稳核验、超时拒绝前进、
反馈丢失、反向及后方两段转向；实现时未启动真机运动。

随后用户手动执行两阶段任务，运行记录为
`outputs/g1_real_unified_vln/g1_pelvis_check_20261020_182930_fb929806/run_20261020_182933_574664/`。
Navigator先选择黑色箱子，后在decision=2主动选择left/绿色瓶子；导航转速0.8rad/s，
停稳后SLAM左转85.54°，相对90°误差-4.46°，记录为SETTLED_VERIFIED。
动作完成上报的SLAM首尾位移：箱子段约1.134m，左转并接近瓶子段约0.564m；
这不是轨迹弧长，也不独立证明停靠距离精度。decision=1曾缺少箱子框底部深度、使用1.5m兜底，
该段报告位移约0.0065m，不能把它解释为可靠的目标测距。
最终decision=4阶段2/2、STOP Gate ALLOW、STOP_CONFIRMED，远端Session SUCCESS/stop_confirmed；
run_exit记录CLEANUP_FINISHED、remote_end_status=CONFIRMED、cleanup_errors为空，最终命令为零。
因此本次验证了前进、导航左转、第二目标接近、语义停止及正常收尾的完整链路，
不作为精确停靠或避障验证。iPlanner调用平均约131ms，窗口HTTP平均48ms，窗口平均1.16秒。
终端COMPLETED的终态stage_id=2不代表第三个子任务，后续显示已修正为“全部阶段完成”；
STOP_CONFIRMED摘要也改为明确的停车确认提示，不改任何决策或控制行为。

默认`--max-decisions 0`沿用原入口：不设决策次数截止，直到模型STOP确认或失败。
可显式给正数做多轮上限测试，但次数到顶不是任务成功。`stage_progress.parse_success=false`
仍按原状态机警告，不能解释成阶段完成。

姿态源过期、定位跳变、取图/关节配对失败会请求停车，失效一旦锁存不允许迟到命令恢复。
行走中超过1秒没有新图也停车；全景转圈有意不连续取图，不套用行走取图超时。
运动终端Ctrl+C沿用原先先停车再关闭资源的处理；它仍是软件停车，不是硬件急停。

正式结果在 `outputs/g1_real_unified_vln/g1_pelvis_.../run_.../`，包含原决策、地图进度、
执行上报、会话结束记录，以及 `real_geometry_assumptions.json`、`real_input_guard.json`。
只有STOP_CONFIRMED并结束Session为SUCCESS才记语义任务成功；退出码、局部到达、
只读PREFLIGHT_PASS都不能替代它。该实验也不证明物理外参或避障精度已经合格。

完整实验分层整理与上传证据见[10月20日两阶段实验报告](../../docs/real_robot/2026-10-20/README.md)，包括Physical窗口、两次Semantic Audit、STOP Gate和已验证范围。

## 2026-10-22：真机目标框局部深度补采样

正式`RealG1Episode`先在目标框底边中心取7×7深度窗口；没有有效深度时，
仅再试15×15，不使用31×31。两个窗口均与裁剪后的目标框、图像边界取交集，
不平移窗口、不补洞；因此框底部的窗口通常不足完整正方形。
保持原0.1～5m过滤、第30百分位测距、同一名义像素中心投影。
两个窗口均无有效深度时仍保留原1.5m兜底，本次没有改变失效策略、到达容差或控制算法。

共享Episode仅新增投影策略调用入口，默认仍使用原7×7、不限制目标框的行为；
仿真不启用15×15，真机通过独立覆盖启用。`depth_sampling`记录两次尝试的窗口范围、
实际像素数、有效点数、是否扩大；原有projection及target JSON自动保存这些信息。
扩窗只沿用原有效值规则，没有新增最小点数、深度聚类或置信度门槛，
不能把“取到了值”当成独立的测距精度或可行走性验证。

以10月20日decision_001、forward帧5187离线重放：原7×7被图像边界裁成7×4，
0/28有效；15×15裁成15×8，16/120有效，深度1.155500m，局部目标约
[1.155500,0.281821]m，`used_forward_fallback=false`。原始实验记录不覆盖；
比较结果另存`outputs/depth_review/decision_001_frame_5187/projection_7_then_15.json`。
16点仅占13.3%，仍有空洞；这不是实际执行结果，也不保证因目标更近而一定产生运动。
新增测试覆盖原窗口优先、边缘补采、框外深度拒绝、深度范围和百分位保持、
仿真原行为不变；完整接口离线回归304项通过，未启动真机运动。

## 2026-10-22：目标外参和实际转向修正（前两阶段）

`unified_vln/real_target_geometry.py`实现以下计算：

```text
Pc = [(u-cx) * Z/fx, (v-cy) * Z/fy, Z]    光学坐标：右、下、前
Pb = R_pelvis_camera @ Pc + t_pelvis_camera  使用所选采图帧的腰关节外参
Pmap_xy = Rz(yaw_capture) @ Pb_xy + position_capture_xy
Pcurrent_xy = Rz(yaw_current).T @ (Pmap_xy - position_current_xy)
```

Z为当前采样策略得到的深度代表值，不是射线长度。保留原底边中心像素，
窗口深度作为该中心射线的局部估计；不新增目标分割或深度聚类。
使用所选帧记录的`T_pelvis_camera`、`pose_after_capture`，校验帧号、sim_step及timestamp，
拒绝借用forward或另一帧的姿态。目标在map的XY中冻结，实际转向后重新表达在当前pelvis系，
同时考虑转角误差和转向期间平移；不再在这个独立几何结果中套用“left必为90°”。
固定1.5m兜底不属于观测深度，不能据此生成三维目标。

真机接受NAVIGATE时计算第一阶段，转向后新前视采集完成、原规划调用之前更新第二阶段，
新增`decision_XXX_target_geometry.json`记录源帧、像素、光学三维点、pelvis三维点、
map目标XY、采图位姿、当前位姿、当前局部目标、与旧名义目标的差异及限制。
缺少外参、采图匹配或位姿时明确标记UNAVAILABLE，不能伪造已修正坐标。

**当前仅完成这两阶段计算和记录，`mode=geometry_comparison_only`、
`applied_to_planner=false`。原projection、规划目标及Pure Pursuit继续按原坐标执行。**
原因是用户将第三阶段iPlanner坐标接口适配留待后续：不能把新的pelvis目标直接交给仍按
名义相机坐标工作的接口。后续完成规划输入/输出坐标统一后，才能启用新目标执行行走。
当前查看新几何请读`target_geometry.json`，旧`projection.json`和预览仍是执行中的旧名义坐标。

这不是完整六自由度地面补偿：安装外参仍是候选，腰角按主机接收时刻配对，采图SLAM位姿
为采集后观测而非曝光插值；map只用平面XY/yaw，pelvis相对于map的roll/pitch没有应用。
Pb的Z是相对pelvis的高度，不是离地高度；新XY也不代表地面接触点、净空或可行走性。

离线重放命令（不访问机器人、相机、模型）：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/replay_target_geometry.py \
  --run-dir outputs/g1_real_unified_vln/g1_pelvis_check_20261020_182930_fb929806/run_20261020_182933_574664 \
  --output outputs/depth_review/target_geometry_20261022/comparison_new.json
```

10月20日历史日志没有与新前视曝光严格配对的规划位姿，重放采用第一次规划调用前最近
控制日志位姿，输出保留该来源和时间间隔；不能称作精确曝光时刻结果。现场新流程直接
记录新前视采集后读取的SLAM位姿，仍不声称曝光硬同步。

该次离线对比（单位米，X前/Y左）：

| decision | 原记录名义目标 | 新几何当前pelvis目标 |
|---|---|---|
| 0 | [2.590,0.336] | [2.592,0.561] |
| 1 | [1.500,0.000]（无深度兜底） | [1.278,0.221]（15×15恢复测距后） |
| 2 | [1.933,0.134] | [1.901,0.464] |
| 3 | [1.304,0.275] | [1.343,0.393] |

这张表包含坐标原点变化、安装旋转和平移、实际位姿变化，decision1还包含采样策略变化，
不能将差值全部归因于俯仰角，更不能把新结果当作独立标定真值。
新增测试覆盖竖直像素反投影、俯仰与安装平移、左视图及85°实际转角/平移补偿、
无深度不生成假观测、非法外参拒绝、真机记录接入和帧匹配；310项接口离线测试通过，
没有启动真机运动，仿真未启用该几何分支。

运行终端的空格键阻尼及导航锁存见[电脑按键DAMPING](README_DAMPING_KEY.md)。
正式入口自动启动独立后台监听进程，直接读取同一运行终端的空格准备、5秒内Enter确认，不需要独立安全窗口；
终端需获得焦点。Ctrl+C先按原方式停车、保存记录并结束导航，但继续保护待命，仍可空格+Enter阻尼；
待命时q+Enter才退出保护。空格+Enter确认后保留导航锁存，不自动恢复站立。

2026-10-22：下次主动启动完整任务时，只读确认FSM=802后自动清除旧阻尼锁存，正常使用不再单独执行clear命令；
仍处于阻尼或查询失败则保留锁存、阻止任务，不自动恢复机器人模式。运行中再次触发的锁存不清除。
