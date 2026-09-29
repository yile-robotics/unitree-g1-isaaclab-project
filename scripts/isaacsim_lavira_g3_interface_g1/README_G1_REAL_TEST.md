# G1 EDU 29DoF 真机高层 DDS 与 SLAM 测试记录

整理日期：2026-09-29。早期高层运控片段未附时间戳；SLAM 双路对照日志包含终端时间。下列结论按现场反馈与日志记录。

本记录覆盖 G1 内置高层运控的 DDS 连通、速度前进和软件停车，以及随后的 Unitree SLAM
建图、重定位与地图位姿读取。运动测试使用 [`g1_real_smoke.py`](g1_real_smoke.py)，
SLAM 测试使用 `unitree_slam_example` 中的 `keyDemo` 和本目录的只读监控脚本。
本记录不代表 LaViRA、iPlanner 或完整 VLN 状态机已在真机端到端运行，也不加载
Isaac Sim 中训练的站立或行走策略。`HighStand` 是高层站立高度请求，
并非机器人跌倒后的起身指令。

## 测试环境与脚本保护

- 机器人：Unitree G1 EDU 29DoF。操作者曾通过遥控器切换到走跑模式，遥控器可以控制机器人行走。
- 本机网络接口：`enxf8e43bbea486`；本机地址 `192.168.123.99`，机器人地址 `192.168.123.164`。此前 ping、SSH 和 DDS 状态订阅均已连通。下列命令在本机终端运行，不在 SSH 会话里运行。
- 虚拟环境：`/home/yile/projects/.venvs/unitree_g1`，使用 `unitree_sdk2_python` 的 G1 高层 `LocoClient`。
- 脚本订阅 `rt/lowstate`，动作前和动作期间检查有效状态是否持续到达；正常完成、Ctrl+C 或动作期间出错时，请求 `StopMove` 并发送显式零速度。
- 普通单轴移动限速为 `0.15 m/s`、最长 1 秒；`--fast-forward-test` 仅允许 `vx=0.5 m/s, vy=wz=0`、最长 1 秒；再加 `--ctrl-c-test` 时才允许设置 2 秒。脚本计划到时请求停车，但 RPC 阻塞、进程异常或网络故障时不能保证严格 2 秒物理停车。

软件零速度请求不是遥控器硬件急停。现场测试应留出机器人前方空间，并让操作者随时能够使用遥控器人工停止。

## 最初的网络连接与接口检查

先在**本机终端**检查连接 G1 的网口与机器人 IP：

```bash
ip -brief addr show dev enxf8e43bbea486
ping -c 4 192.168.123.164
```

本轮本机网口地址为 `192.168.123.99/24`，机器人地址为 `192.168.123.164`。
`ping -c 4` 实际返回 4 发 4 收、丢包率 0%，平均 RTT 为 `1.744 ms`。
这一步只确认 IP 网络可达，不能证明 DDS 控制或机器人运动正常。

随后从本机 SSH 登录机器人：

```bash
ssh unitree@192.168.123.164
```

本轮成功进入机器人上的 Ubuntu 20.04.6（aarch64），终端提示符变为 `unitree@ubuntu:~$`。
**在机器人 SSH 终端**查看 ROS2 话题，完成后退出，回到本机 `yile@yile-ROG` 终端：

```bash
ros2 topic list
exit
```

本轮话题列表中可见 `/lowstate`、`/sportmodestate`、`/api/sport/request`、
`/api/sport/response`、`/frontvideostream` 和 `/utlidar/cloud_livox_mid360` 等。
这只能证明机器人侧相应话题存在；本机 Python SDK 的 DDS 数据接收还需通过下节 `status`
命令单独确认，运动控制则需后续真机观察。

## 命令

在本机终端进入项目并激活环境：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source ../.venvs/unitree_g1/bin/activate
```

只读检查状态，不发送运动命令：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_real_smoke.py \
  --network-interface enxf8e43bbea486 status
```

发送软件停车请求：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_real_smoke.py \
  --network-interface enxf8e43bbea486 stop
```

请求 `HighStand`（机器人应已处于正常、稳定的高层控制状态）：
当前脚本使用与 SDK `HighStand()` 相同的高度参数，并检查 `SetStandHeight` 的 RPC 返回码。
`code=0` 仍只能证明请求未报错；如果原本已经处于高站姿，重复命令可能没有可见变化。

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_real_smoke.py \
  --network-interface enxf8e43bbea486 --execute stand
```

已观察到实际前进的 1 秒测试：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_real_smoke.py \
  --network-interface enxf8e43bbea486 --execute \
  move --vx 0.5 --seconds 1.0 --fast-forward-test
```

Ctrl+C 停车测试：执行下列命令，等机器人开始前进后、2 秒结束前，在同一终端按一次 Ctrl+C。

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_real_smoke.py \
  --network-interface enxf8e43bbea486 --execute \
  move --vx 0.5 --seconds 2.0 --fast-forward-test --ctrl-c-test
```

若输出 `bounded motion complete; software stop sent`，说明动作运行到时，不能计为 Ctrl+C 中断测试；
若输出 `^CG1 smoke test stopped:`，说明程序进入中断退出路径，还需现场确认机器人实际停止。

StopMove 单独观察测试（**尚未真机执行**）：同一进程以 `0.5 m/s` 前进约 0.8 秒，
然后调用一次 SDK `StopMove()`；等待约 0.6 秒供现场观察，最后发送备用零速度。
前进命令单条有效期为 1.5 秒，意在把自然到期与前述观察窗口分开。运行前预留空间并确保
遥控器人工停止手段可用；若机器人在 `StopMove returned` 后仍继续运动，立即准备人工停止。

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_real_smoke.py \
  --network-interface enxf8e43bbea486 --execute stopmove-probe
```

若机器人在 `StopMove returned` 后、`backup zero-velocity RPC code=0` 前停下，
且 StopMove 调用没有耗尽前进命令有效期，可作为 StopMove 请求起作用的现场证据；
若直到备用命令或自然到期才停，结论不能归于首次 StopMove。SDK 的 `StopMove()`
本身就是 `SetVelocity(0, 0, 0)` 的包装，并非独立的硬件急停机制。

复测建议保持相同参数，从侧面用慢动作视频同时拍到机器人和终端，确认发出 `StopMove` 时机器人
确实仍在行走，并逐帧对照 `StopMove returned` 与 `backup zero-velocity RPC code=0` 两条输出。
如果停止时刻仍看不清，本项继续记为无法判定，不通过增大速度来推断。

## 已有真机结果

| 测试 | 终端与现场记录 | 当前结论 |
|---|---|---|
| 连通与状态 | ping 为 4/4、0% 丢包、平均 1.744 ms；SSH 登录成功；机器人侧列出 ROS2 话题；本机脚本收到 `lowstate` 并打印 IMU yaw | IP、SSH 和状态订阅分别通过；yaw 不是位移测量 |
| 软件 `stop` | 输出 `software stop RPC sent` | 停车 RPC 已发送；机器人当时没有运动，不能据此验证运动中停车 |
| `stand`（旧版脚本） | 输出 `HighStand request sent`；一次运行后出现 `[Reader] take sample error`，后续状态与移动测试仍可运行 | 旧版包装函数丢弃 RPC 返回码；未看到明确的姿态变化，姿态效果与 Reader 提示尚未单独确认 |
| `stand`（返回码复测） | `lowstate` 正常到达，`HighStand height RPC code=0`；本轮未提供可量化的机身高度或姿态变化记录 | 高站高度 RPC 未报错；实际抬高效果仍未确认，不能记作物理动作通过 |
| 早期低速前进 | `vx=0.05/0.10/0.15 m/s` 的短时命令输出完成，但未观察到机器人移动 | RPC 无错误不等于有物理运动；不能由此推断最小运动速度 |
| `vx=0.5 m/s`、0.5 秒 | 机器人出现较小向前动作 | 高层速度命令开始得到物理响应；未测实际位移 |
| `vx=0.5 m/s`、1.0 秒 | 输出 `bounded motion complete; software stop sent`；现场观察机器人向前走了一步并停住 | 前进和到时停车已得到现场观察 |
| `vx=0.5 m/s`、最多 2.0 秒，共三次 | 前两次按 Ctrl+C，均输出 `^CG1 smoke test stopped:`；第三次到时输出 `bounded motion complete; software stop sent`；操作者反馈均停下 | 两次中断停车路径和一次正常到时停车路径均已触发；实际停车延迟未测量 |
| `stopmove-probe` 首次真机运行 | `lowstate` 有效；0.5 m/s 前进命令持续约 0.8 秒；`StopMove returned after 0.002 s`，其后输出 `backup zero-velocity RPC code=0`；操作者表示停车时刻看不出来 | StopMove 方法快速返回、备用零速度 RPC 成功；首次 StopMove 是否在备用命令之前使机器人停下，本次**无法判定** |

这些数据来自终端片段和操作者观察，没有测量实际位移、速度跟踪误差或按键到物理停止的延迟。
两次 Ctrl+C 的终端输出表明脚本执行了中断清理路径、零速度 RPC 没有报错；不能仅凭输出证明
物理停止时刻，也不能分辨停止由零速度请求还是单条短时速度命令到期造成。

## 后续软件停车约定

StopMove 单独生效的首次对照测试无法判定，本阶段不再为归因重复该实验。独立冒烟脚本已经验证：
正常到时和 Ctrl+C 退出后，请求 `StopMove()`，随后显式发送 `SetVelocity(0, 0, 0)` 并检查
后者的 RPC 返回码；操作者观察机器人停止。正式 VLN 控制链迁移时应保持同一停车顺序，
并先将持续发送线程的目标速度清零，避免后续非零命令覆盖停车请求。

在当前 Python SDK 中，`StopMove()` 本身就是零速度 `SetVelocity` 的包装；两次请求是同一
高层通道上的重复停车指令，不是两套独立的急停机制。软件停车仍需现场确认物理效果，
遥控器人工停止/硬件急停必须保留。

## SLAM 建图、重定位与地图坐标读取

以下操作在**本机**进行；`keyDemo` 的按键只在运行它的终端生效。用遥控器缓慢带机器人走过待建区域，并保留人工停车手段。先打开 Unitree SLAM 示例：

```bash
cd /home/yile/projects/unitree_slam_example/example/build
./keyDemo enxf8e43bbea486
```

在此终端按一次 `q` 启动建图，本轮返回 `statusCode:0`、`Successfully started mapping.`。机器人上的 `/utlidar/cloud_livox_mid360` 此前约 10 Hz；建图启动后，`/unitree/slam_mapping/odom` 和 `/unitree/slam_mapping/points` 也约 10 Hz。话题有数据说明扫描与里程计在发布，不等于地图质量已经得到验证。走完一圈后按一次 `w`，本轮返回 `Save pcd successfully.`。之后重复按 `w` 曾返回 `errorCode:505`、`Pcd buffer less than 1.`；不要把重复保存的失败当作首次保存失败。

建图时可在另一个**本机终端**查看点云：

```bash
source /home/yile/projects/unitree_ros2/setup.sh
rviz2 -d /home/yile/projects/unitree_slam_example/rviz2/mapping.rviz
```

本轮在 `/unitree/slam_mapping/points` 看到了随机器人移动变化的彩色点云。RViz 的 `Fixed Frame=map` 曾提示缺少 TF，虽然 PointCloud2 显示 `Status: Ok`，不能因此认为 TF 已正确发布。显示的主要是当前或最近扫描；屏幕不保留走过的所有区域，不代表保存的地图为空。按 `a` 后，RViz 显示了白色的已有地图点云。

**PCD 的实际存储位置仍未确认。** `keyDemo.cpp` 在按 `w` 时请求保存到 `/home/unitree/test.pcd`，但在已登录的 `192.168.123.164` 机器人上，`ls` 与限定目录的 `find` 都未找到该文件。机器人重启后，`a` 返回成功，RViz 的 `/unitree/slam_relocation/global_map` 也显示了地图点云，说明重启后确实有地图可供重定位；但地图文件实际在哪台设备、是否就是同一份 PCD，仍需核对。

建图结束后在 `keyDemo` 中按一次 `a` 发起重定位。本轮既出现过 `Successfully started re-location.`，也在匹配较差时出现过 `errorCode:509`、`The current location matching degree is low.`。成功返回只表示请求被接受，还需观察位姿是否稳定且符合真实移动。本轮 `/unitree/slam_relocation/odom` 持续发布，消息为 `map → base_link`；这里的“世界坐标”是当前 SLAM 地图中的机器人基座坐标，不是地理坐标。

在另一个本机终端持续看 ROS 2 地图位姿：

```bash
source /home/yile/projects/unitree_ros2/setup.sh
/home/yile/projects/.venvs/unitree_g1/bin/python \
  /home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_g3_interface_g1/monitor_g1_slam_pose.py
```

该脚本显示 `x/y/yaw` 和距启动监控位置的**直线距离**，检查消息新鲜度与 `map → base_link` 坐标系；所显示距离不是累计路程。若要像 `keyDemo.cpp` 一样直接从 Unitree DDS 的 `rt/slam_info` 读取 `pos_info.data.currentPose`：

```bash
/home/yile/projects/.venvs/unitree_g1/bin/python \
  /home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_g3_interface_g1/monitor_g1_slam_info.py \
  --network-interface enxf8e43bbea486
```

DDS 监控显示原始 `x/y/z`、四元数及换算的 `yaw`，只订阅，不发送运动或导航命令。`keyDemo` 的 `s` 虽会打印当前位姿，但**同时将其加入导航任务列表**；`d` 会执行该列表。纯监控应使用上述脚本，而不是按 `s/d`。

同时对照 DDS 与 ROS 2 位姿：

```bash
source /home/yile/projects/unitree_ros2/setup.sh
/home/yile/projects/.venvs/unitree_g1/bin/python \
  /home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_g3_interface_g1/compare_g1_slam_pose.py \
  --network-interface enxf8e43bbea486
```

本轮静止和遥控器移动期间记录了 99 组双路数据，终端显示精度下每组均为 `xy=0.000 m`、`yaw=0.0°`，本机接收时间差最多 `0.001 s`。起点约 `(0.304, -0.111) m`，终点约 `(-1.145, 0.058) m`，两点直线距离约 1.46 m，而非累计路程。`yaw` 跨越 `±180°` 时的正负切换正常。这证明本次会话中两种接口的平面位姿一致，但它们不是两套独立的真实位置测量，不能仅凭一致性证明绝对定位精度。按 Ctrl+C 结束监控；本轮比较脚本结束后出现一次 `[Reader] take sample error`，发生在读数结束之后，原因尚未单独确认。

真机 VLN 入口已有 ROS 2 位姿适配器，后续可指定 `--odometry-topic /unitree/slam_relocation/odom`，不必为同一位姿改用 DDS JSON。入口本身不启动 SLAM、不加载 PCD。一次 VLN Episode 中若重新按 `a` 或发生坐标跳变，应结束当前任务并重新建立坐标基准；现有入口只能检测 `frame_id` 名称变化，不能识别同名 `map` 中的重定位跳变。

### 重启后的重定位复测

重启前机器人停在已标记位置，位姿约为 `x=-1.147 m、y=+0.055 m、yaw=+145.3°`。机器人重启后稍有移动，主要是朝向变化；第一次按 `a` 返回 `Successfully started re-location.`，且 ROS 2 `map → base_link` 持续发布，但读数约为 `x=+0.94 m、y=-1.59 m、yaw=+55°`。与重启前的平面位置相差约 2.7 m，不能由小幅挪动或仅改变朝向解释；RViz 中操作者也判断机器人落在地图错误区域。因此**API 成功和位姿持续发布不等于重定位正确**。

操作者随后又按了一次 `a`，反馈 RViz 中机器人位置“好像又对了”。本次尚未提供第二次定位后的 `x/y/yaw` 数值、同一物理标记点对照或点云与地图的量化匹配结果，因此记录为**视觉上改善，数值复核未完成**，不写作重定位精度通过。使用旧地图开展 VLN 前，仍应在同一标记点核对位置与朝向，并在运行中检测重定位跳变。

## 尚待验证

- 遥控器硬件急停、网络中断或 DDS 失联时的实际停车行为。
- `HighStand` 的姿态效果，以及一次 `[Reader] take sample error` 的原因。
- 地图 PCD 的实际保存位置与地图身份；重启后虽有全局地图点云，但同一标记点的坐标一致性尚未通过数值复核。
- SLAM 绝对定位误差、失配恢复与重定位跳变监测；当前已验证在线位姿可读和双路接口一致。
- D435 相机/深度、iPlanner、LaViRA 服务器和完整 VLN 状态机的真机端到端联调。

离线控制逻辑测试位于 [`tests/test_g1_real_smoke.py`](tests/test_g1_real_smoke.py)；本轮运行 15 项通过。
离线结果只验证参数边界和代码清理路径，不替代真机观察。
