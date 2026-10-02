# G1 EDU 29DoF 真机测试记录

更新：2026-10-02。已验证高层前进、运动中 Ctrl+C 停车、SLAM 位姿读取、连续四段旋转，以及相机30/60秒预览、参数保存和服务停止后的断流检测。相机命令与数据见[相机说明](camera_d435i/README.md)。完整VLN、长时间稳定性和整机失联停车仍待联调。

以下命令除特别注明外均在笔记本执行。实测数据来自终端日志和操作者观察，具体边界在各节说明。

## 环境和连接

- 机器人：Unitree G1 EDU 29DoF，使用机载高层运控。Python 侧通过 `unitree_sdk2py.g1.loco.g1_loco_client.LocoClient` 发送速度，机器人自己执行步态。这里不部署 Isaac Sim 中训练的策略。
- 本机网卡：`enxf8e43bbea486`，现场本机地址曾为 `192.168.123.99/24`；机器人 SSH 地址为 `192.168.123.164`。网络配置变化时先重新检查实际地址。
- 项目：`/home/yile/projects/unitree-g1-isaaclab-project`；Python 环境：`/home/yile/projects/.venvs/unitree_g1`。
- 软件 `StopMove()` 和零速度命令均属于高层软件停车请求。操作者须保留遥控器人工停止手段；RPC 成功不等于已经测得物理停车时延。

本机先检查网络：

```bash
ip -brief addr show dev enxf8e43bbea486
ping -c 4 192.168.123.164
```

现场 `ping` 为 4/4、0% 丢包、平均 RTT `1.744 ms`。如需查看机器人侧 ROS 2 话题，可在本机运行 `ssh unitree@192.168.123.164`，机器人提示符为 `unitree@ubuntu:~$` 时执行 `ros2 topic list`，完成后输入 `exit` 回到本机。现场能看到 `/lowstate`、`/sportmodestate`、`/api/sport/request`、`/utlidar/cloud_livox_mid360` 等话题；这只证明相应接口存在，不能代替 Python DDS 接收和实际运动测试。

本机 Python 测试先进入项目环境：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source ../.venvs/unitree_g1/bin/activate
```

## 高层运控：已验证的操作

以下 `g1_real_smoke.py` 使用 Python SDK，运动前检查 `rt/lowstate` 是否新鲜。先运行只读状态检查；`stop` 请求软件停车；`stand` 请求高站高度：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_real_smoke.py \
  --network-interface enxf8e43bbea486 status
python scripts/isaacsim_lavira_g3_interface_g1/g1_real_smoke.py \
  --network-interface enxf8e43bbea486 stop
python scripts/isaacsim_lavira_g3_interface_g1/g1_real_smoke.py \
  --network-interface enxf8e43bbea486 --execute stand
```

现场 `status` 收到 `lowstate` 和 IMU yaw；`stop` 输出 `software stop RPC sent`，但当时机器人没有运动，不能凭它证明运动中停车。`stand` 复测得到 `HighStand height RPC code=0`；机器人原本已站立，操作者看不到明确姿态变化，因此只确认请求返回成功，未确认可见的高度变化。`HighStand` 不是跌倒后的起身命令。

在前方有足够空间时，可复现已观察到向前迈步并到时停车的命令：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_real_smoke.py \
  --network-interface enxf8e43bbea486 --execute \
  move --vx 0.5 --seconds 1.0 --fast-forward-test
```

现场 `vx=0.5 m/s`、1 秒时机器人向前走了一步并停住；实际位移未测量。早期 `vx=0.05/0.10/0.15 m/s` 短时命令没有观察到运动，因此不能把“RPC 返回完成”写成低速行走成功，也不能由此推断速度死区的精确数值。

`g1_real_smoke.py` 的 2 秒 Ctrl+C 试验命令如下。应在机器人确实运动、尚未到时的期间按键；若输出 `bounded motion complete`，只算到时停车。现场两次按键后的退出路径均执行，操作者报告停下，但当时没有量化按键到物理停稳的时延。

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_real_smoke.py \
  --network-interface enxf8e43bbea486 --execute \
  move --vx 0.5 --seconds 2.0 --fast-forward-test --ctrl-c-test
```

**VLN DDS 后端的 Ctrl+C 清理**另用 `g1_vln_ctrl_c_probe.py` 验证。前两次旧版 2 秒试验均到时，不能算按键测试。之后运行 3 秒倒数版本，在机器人正在行走时按 Ctrl+C，终端打印 `Ctrl+C received; stopping VLN DDS backend...`、`cleanup returned`；操作者确认随后停稳。这个探针复用正式 VLN 的后台速度发送线程和 `stop()`/`close()` 清理路径，但没有启动完整 VLN 进程。

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_vln_ctrl_c_probe.py \
  --network-interface enxf8e43bbea486 --seconds 3.0 --execute
```

`stopmove-probe` 现场曾先调用 `StopMove()`、观察约 0.6 秒，再发备用零速度。日志为 `StopMove returned after 0.002 s` 和 `backup zero-velocity RPC code=0`，但操作者无法分辨机器人是在备用零速度之前还是之后停下，所以 **StopMove 单独生效的时刻未判定**。当前停车约定是先将持续发送线程的目标速度清零，再调用 `StopMove()`；独立 SDK 脚本另外显式发送零速度。`StopMove()` 在当前 SDK 中本身也是零速度 `SetVelocity` 的包装，并非独立硬件急停。

## SLAM：建图、重定位和地图坐标

### 建图一次

在本机一个终端打开 Unitree SLAM 示例：

```bash
cd /home/yile/projects/unitree_slam_example/example/build
./keyDemo enxf8e43bbea486
```

在 `keyDemo` 终端按 `q` 启动建图，现场返回 `Successfully started mapping.`。用遥控器带机器人走过要建的区域；完成后按一次 `w`，现场返回 `Save pcd successfully.`。重复按 `w` 曾返回 `errorCode:505`、`Pcd buffer less than 1.`，不能把重复保存的报错当成第一次保存失败。机器人侧 `/utlidar/cloud_livox_mid360`、建图时的 `/unitree/slam_mapping/odom` 和 `/unitree/slam_mapping/points` 均约 10 Hz。

建图期间可在本机另一个 ROS 2 终端查看扫描点云：

```bash
source /home/yile/projects/unitree_ros2/setup.sh
rviz2 -d /home/yile/projects/unitree_slam_example/rviz2/mapping.rviz
```

RViz 的 `/unitree/slam_mapping/points` 主要显示当前或近期扫描，不等于已经累计保存的完整地图。现场 `Fixed Frame=map` 曾出现缺少 TF 警告，即使点云显示 `Status: Ok`，也不能据此认定 TF 链正确。

**PCD 实际存储位置尚未查明。** `keyDemo.cpp` 的保存请求指向 `/home/unitree/test.pcd`，但在 SSH 登录的 `192.168.123.164` 上没有找到该文件。机器人重启后仍能显示已有地图，说明存在可用于重定位的地图数据；尚不能确认它的文件路径或是否就是同一份 PCD。

### 下次开机使用已有地图

现场有效顺序是先在本机打开重定位 RViz：

```bash
source /home/yile/projects/unitree_ros2/setup.sh
rviz2 -d /home/yile/projects/unitree_slam_example/rviz2/relocation.rviz
```

再在本机**另一个终端**启动 `keyDemo`，在该终端按一次 `a`：

```bash
cd /home/yile/projects/unitree_slam_example/example/build
./keyDemo enxf8e43bbea486
```

按 `a` 后检查 RViz 中白色的 `/unitree/slam_relocation/global_map`、当前点云和机器人所处区域是否吻合。RViz 配置只设置显示话题，不自行从本地文件加载 PCD。`a` 返回 `Successfully started re-location.` 只表示请求接受，**还必须核对位置**。现场重启后第一次 `a` 虽返回成功，位姿却从重启前约 `(-1.147,+0.055) m` 跳到约 `(+0.94,-1.59) m`，RViz 中也处在错误区域；再次按 `a` 后操作者认为位置视觉上恢复，但没有提供数值复核，不能记作重定位精度通过。匹配较差时也曾收到 `errorCode:509`。

请在同一物理标记点核对地图中的位置和朝向。执行中的 VLN 若重新重定位或地图坐标突变，应结束当前 Episode、重新建立坐标基准。不要在只想读取位置时按 `keyDemo` 的 `s` 或 `d`：`s` 会把位姿加入导航任务列表，`d` 会执行任务列表。

### 只读查看世界坐标

这里的“世界坐标”是 SLAM 地图 `map → base_link` 中的机器人位置和朝向，不是地理坐标。在已经按 `a` 并确认重定位后，另开本机终端持续读取 ROS 2 位姿：

```bash
source /home/yile/projects/unitree_ros2/setup.sh
/home/yile/projects/.venvs/unitree_g1/bin/python \
  /home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_g3_interface_g1/monitor_g1_slam_pose.py
```

该脚本打印 `x/y/yaw` 和距**本次监控起点**的直线距离，不是累计路程。直接读取 Unitree DDS `rt/slam_info` 的 `pos_info.data.currentPose` 可用：

```bash
/home/yile/projects/.venvs/unitree_g1/bin/python \
  /home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_g3_interface_g1/monitor_g1_slam_info.py \
  --network-interface enxf8e43bbea486
```

双路对照脚本使用同一地图坐标来源的 ROS 2 和 DDS 接口：

```bash
source /home/yile/projects/unitree_ros2/setup.sh
/home/yile/projects/.venvs/unitree_g1/bin/python \
  /home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_g3_interface_g1/compare_g1_slam_pose.py \
  --network-interface enxf8e43bbea486
```

现场静止和遥控器移动期间取得 99 组数据；在终端显示精度下，两路每组均为 `xy=0.000 m`、`yaw=0.0°` 差值，本机接收时间差最多 `0.001 s`。这证明本次会话中两个接口读到了相同平面位姿，**不证明绝对定位误差为零**。真机 VLN 入口已有 ROS 2 `/unitree/slam_relocation/odom` 适配器，入口本身不会启动 SLAM 或加载地图。

### 速度命令与地图位移对照

`g1_slam_speed_probe.py`用VLN DDS后端发送固定`vx`（`vy=wz=0`），订阅`rt/slam_info`，按**起始地图yaw**将位移投影为前方和左方。记录请求停车前、`stop()`返回后及再等待0.8秒后的位移。Ctrl+C、位姿过期或定位跳变时请求停车。

各速度允许的最长窗口：0.3m/s为3秒，0.4m/s为2秒，0.5m/s为4秒。已测命令示例：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_slam_speed_probe.py \
  --network-interface enxf8e43bbea486 --vx 0.5 --seconds 4.0 --execute
```

不同次试验从各自起点独立计算。下表距离单位为米，速度为m/s；“最终”指停车请求后执行清理并再等待0.8秒。

| vx / 时长 | 命令积分 | 停车请求前 | stop返回后 | 最终前向 | 命令期平均速度 | 最终/积分 | 停车后增量 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 / 1s | 0.500 | 0.087 | 0.114 | 0.184 | 0.087 | 0.37 | 0.097 |
| 0.5 / 2s | 1.000 | 0.454 | 0.551 | 0.605 | 0.227 | 0.60 | 0.150 |
| 0.5 / 3s | 1.500 | 0.854 | 0.926 | 0.965 | 0.285 | 0.64 | 0.111 |
| 0.5 / 4s | 2.000 | 1.227 | 1.300 | 1.380 | 0.307 | 0.69 | 0.153 |
| 0.4 / 2s | 0.800 | 0.161 | 0.273 | 0.360 | 0.080 | 0.45 | 0.199 |

四次0.5m/s试验总最终前向位移为3.134m，总命令积分为5.000m，合计比例约62.7%。这是含起步和停车过程的距离比例，不是稳态速度跟踪率。

| vx / 时长 | 起点(x,y)，yaw | 终点(x,y) | 最终侧向 | yaw变化 |
|---|---|---|---:|---:|
| 0.5 / 1s | (+0.559,-0.896)，-25.7° | (+0.738,-0.947) | +0.032 | +1.7° |
| 0.5 / 2s | (-0.280,-0.705)，-18.5° | (+0.315,-0.833) | +0.067 | +4.0° |
| 0.5 / 3s | (-0.450,-0.569)，+0.3° | (+0.514,-0.504) | +0.061 | +17.1° |
| 0.5 / 4s | (-0.242,-0.442)，-17.6° | (+1.150,-0.617) | +0.256 | +9.3° |
| 0.4 / 2s | (+0.604,-0.578)，-14.9° | (+0.968,-0.609) | +0.064 | +3.9° |

起步阶段的日志要点：

- 0.5m/s、1秒：t=0.25/0.50/0.75s时前向为0.000/0.002/0.022m。
- 0.5m/s、2秒：t≈1.01s时0.079m，后约1秒增加0.375m。
- 0.5m/s、3秒：t=1.00s时0.056m，t≈2.01s时0.474m，后约2秒平均约0.40m/s。
- 0.5m/s、4秒：t=0.50s基本未前进；最终直线距离1.403m，前向投影1.380m。
- 0.4m/s、2秒：t=1.00s时仅0.035m；最终直线距离0.365m。
- 另一次0.5m/s、4秒试验在t≈2.5s被Ctrl+C中断，按键前最近前向位移0.745m，日志显示清理返回；没有停车后测量或肉眼确认记录，不计入上表。
- 0.3m/s、3秒：操作者未见行走，没有SLAM输出可量化实际位移。0.9m只是命令积分。

平均速度由SLAM位移除以发命令时长得到，包含起步延迟；停车后增量包含运控响应和定位刷新。各次环境与起点不同，尚不能确定精确低速死区或稳态速度。尤其3秒和4秒试验存在朝向、侧向偏移，需要闭环跟踪验证。

正式Pure Pursuit会根据轨迹调整速度并依据地图目标距离结束动作；其默认前进速度0.3m/s尚未通过上述真机测试，不能直接套用0.5m/s固定命令的结果。

## 四方向旋转：当前推荐的真机探针

先按上节在已有地图按 `a` 重定位，核对 RViz 中位置和机器人周围的转身空间。`g1_vln_rotation_probe.py --mode imu75_panorama` 使用正式 VLN 的 `UnitreeG1DDSBackend`：每段由后台线程重复调用 SDK `Move(0,0,+0.8)`，读取 `rt/lowstate` 的 IMU yaw；**该段 IMU 累计达到 75° 才请求停车**。停车后等待稳定，读取 `rt/slam_info` 地图 yaw 并打印结果，再自动进入下一段。每段最多 10 秒，SLAM 位姿过期、反向转动或单段地图转角不在 70°–110° 时停止后续段；Ctrl+C 请求 `stop()`/`close()`。四段之间会短暂停车，对应 VLN 全景中的四个拍摄边界。

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source ../.venvs/unitree_g1/bin/activate
python scripts/isaacsim_lavira_g3_interface_g1/g1_vln_rotation_probe.py \
  --network-interface enxf8e43bbea486 \
  --mode imu75_panorama --quarters 4 --execute
```

**75° 是 IMU 反馈的提前停车阈值，不是 SDK 的角度命令。** 机器人实际只收到 `wz=+0.8 rad/s` 的速度命令；达到阈值后仍可能继续转动一段。SLAM yaw 在此模式只用于观察停稳后的地图结果和决定是否进入下一段，不控制何时停车。这个阈值来自前面的现场试验，不保证每段停在精确 90°。

本次**单进程、VLN DDS 后端**连续四段试验：

| 段与拍摄标记 | 停稳后 IMU 相对转角 | 停稳后 SLAM 相对转角 |
|---|---:|---:|
| 1 → left | +83.9° | +86.1° |
| 2 → behind | +93.2° | +96.3° |
| 3 → right | +90.3° | +89.1° |
| 4 → forward | +88.3° | +88.5° |

四段都触发 `IMU reached +75° limit`；脚本根据未四舍五入值报告 SLAM 合计 `+359.9°`（表中显示值相加为 `360.0°`）。第一段终点 SLAM yaw `+66.5°`，推得起点约 `-19.6°`；末段终点 `-19.7°`，即地图朝向回到起点附近。脚本只打印 `simulated capture` 标记，**没有拍摄或保存图像**。

此前用**直接 SDK `SetVelocity()`** 连续手动执行四次同样的 `wz=0.8`、IMU 75° 停车测试，SLAM 分段为 `98.9°、89.6°、88.1°、88.7°`，合计 `365.3°`；IMU 分段为 `87.8°、88.8°、87.1°、90.1°`。直接 SDK 和 VLN DDS 后端的发送与停车时序不同，因此以各自日志记录，不把这两组角度合并成统一误差。需要单步诊断时，命令为：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/g1_rotation_rpc_probe.py \
  --network-interface enxf8e43bbea486 --wz 0.8 --seconds 10 \
  --target-imu-deg 75 --observe-slam --execute
```

早期排障结论：`wz=0.4` 的定时测试终点仅 IMU `3.3°/9.2°`，现场未见明显旋转；`wz=0.5` 的直接 SDK 1.5 秒和 4 秒测试均约 `3.8°`；`wz=0.6` 有时可转（直接 SDK IMU 45° 请求停车后读到 `53.9°`，VLN DDS 后端读到 `58.2°`），也出现过连续发送 20 秒却停在 IMU 约 `10.5°`、SLAM 约 `12.6°` 的情况。故当前推荐测试用已跑通的 `0.8` 与提前停车阈值，不根据命令角速度和时长推算机器人实际转角。上述早期表现不足以确定低速不转的根因。

**正式 VLN 主程序仍需改造和验证。** `run_g1_real.py` 当前默认旋转速度 `0.4 rad/s`；其 `TimedFixedSpeedRotation` 在配置地图 yaw 反馈时按相对 90° 结束，单相机全景状态在完成段落的控制周期内采图。它尚未采用探针的 `0.8 rad/s + IMU 75° 提前停车 + 停稳后采图` 逻辑。因此这次只能记作**高层控制和四段动作顺序真机通过**，不能记作相机、LaViRA、iPlanner 或完整 VLN 通过。

## 仍需完成

- 把已验证的四段转向参数、IMU 停车与停稳后采图时序接入正式 VLN；再用真实 RGB-D 相机验证四方向图像与深度、相机标定和数据上传。
- 联调服务器、iPlanner 和完整 VLN 状态机；本 README 中的旋转探针没有调用这些组件。
- 测量停车请求到物理停稳的时延、SLAM 转角的外部基准误差；当前读数来自 IMU 和地图估计，不是独立真值。
- 核对 PCD 实际保存位置、重启后同一物理标记点的坐标一致性，以及同名 `map` 内的重定位跳变。
- 确认 `HighStand` 的可见姿态效果、网络/DDS 失联时的停车行为；`[Reader] take sample error` 曾偶发出现，尚未找到原因。

代码的离线测试位于本目录 `tests/`；它们验证控制逻辑和清理分支，不代替真机结果。
