# 真机 G3 Session 接入

配置 `--odometry-topic` 后，机器人平面世界位置 `(x,y)` 和地图 yaw 来自 SLAM：
同一个位姿源供轨迹跟随、目标转向、全景采集位姿、历史轨迹及执行上报使用。保留 SLAM 原始
世界 x/y，不对大坐标重新归零。每轮控制前后检查位姿，失效时终止导航并停车，不继续
采用命令积分。真机四方向全景的提前停车单独使用 `rt/lowstate` 的IMU yaw，
停稳后的转角检查和图像位姿记录仍使用上述SLAM来源。
Unitree 示例中的已有地图定位话题候选为 `/unitree/slam_relocation/odom`，建图期间为
`/unitree/slam_mapping/odom`。实际消息必须表示机器人基座位姿；需在真机确认类型和外参。
RGB-D 探索地图、iPlanner 和 G3 协议维持现有实现，不读取 SLAM PCD 地图。

`run_g1_real.py` 默认启用 G3，装配真机专用 `RealG1Episode`；监督、STOP Gate、
PREEMPT、Recovery、BACKTRACK 和执行上报复用 `episode.py`。仿真仍创建原来的
`LocalEndToEndEpisode`，本次未修改共享状态机或仿真入口。

## 真机全景：已接入实测旋转与采图顺序

`unified_vln/real_episode.py` 只替换真机四段全景采集：先主动停车、等待后采初始前方；
每段持续输出 `wz=0.8 rad/s`，`rt/lowstate` IMU累计到75°时调用 `dds.stop()`。
停车调用返回后再等待0.5秒，期间控制循环保持零速度；用SLAM检查本段转角在70°～110°
范围后采集。前三段依次保存左、后、右，第四段另存回正图，模型仍只收到初始前方、左、
后、右四张图。采图结束后的下一轮才开始下一段，不在拍照期间旋转。

75°是经验提前停车阈值，不保证最终停在精确90°。每段最多10秒；IMU/SLAM失效、转角
检查失败或相机异常均主动停车，状态机进入FAILED。Ctrl+C保留原先先停车再清理的路径，
并关闭新增的IMU订阅器。等待按真实单调时钟计时，不把停车RPC耗时算进0.5秒。

参数默认已生效，无需给真机入口重复添加：`--panorama-speed-rad-s 0.8`、
`--panorama-imu-stop-deg 75`、`--panorama-quarter-timeout-s 10`、`--rotation-settle-s 0.5`。
`--rotation-speed-rad-s`仍用于模型选定方向和BACKTRACK转向；其原有默认0.4及SLAM反馈
逻辑本次未改，不与全景速度混用。

每次决策保存五组 `decision_XXX_方向.png` 与 `decision_XXX_方向_depth_m.npy`；
`decision_XXX_panorama_capture.json`记录状态、四段转角、停车请求/返回时间、
采集请求/返回时间、帧号、K、取图前后位姿和相机元数据。位姿尚未插值到曝光时间。
相机需要固定随机器人转动才具有可靠的方向几何意义。

已通过186项离线回归，含真实状态机的停车等待、相机阻塞/失败、IMU/SLAM失效、
超时、跨±180°、初始图与回正图分离，以及原有仿真流程。此前探针的真机旋转采图已通过，
新正式入口与模型、iPlanner的全链路尚未现场运行，不能把离线回归记成真机导航成功。

启动顺序：校验参数和地图配置 → DDS 零速度与LowState订阅 → 相机 → ROS 2 里程计 → 等待有效位姿与IMU
→ 创建稀疏探索地图 → health/start_session → HighStand → 导航状态机。
instruction 只在 start_session 提交；decision 使用 Session 中的任务。
运动期间发送 motion_window，动作结束发送 action_complete。正常退出、异常和 Ctrl+C
都先停止机器人、关闭本地资源，再尽力 end_session。只有 STOP_CONFIRMED 结束为 SUCCESS；
决策次数上限、人工中断和失败均结束为 FAILURE。远端结束请求失败会打印警告。

## 必需的外部组件

- 已运行的 G3 服务或 SSH 隧道，`--model-url` 指向完整 `/v1/lavira/decision` 地址。
- 已运行的本机 iPlanner 服务，参见 `run_iplanner_local.sh`。
- 可导入的 `MODULE:FUNCTION` 相机工厂，返回对齐 RGB-D 和真实内参的 CameraBackend。
- 已运行的 SLAM，发布机器人基座位姿的 ROS 2 Odometry。此入口不启动 SLAM，也不做 TF
  外参转换。默认最多等待首个有效且 frame_id 非空的位姿 10 秒。
- 可接收新鲜 `rt/lowstate` IMU状态；启动等待最多3秒，旋转中0.5秒未更新即不可用。
- 真实相机和基座几何配置，使用 `--map-config /absolute/path/map.json`。

地图配置是一个 JSON 对象，字段对应 `unified_vln.map_progress.SparseMapConfig`。
以下字段必须显式填写，避免把仿真安装尺寸当作真机标定：

| 字段 | 含义 |
| --- | --- |
| camera_offset_x_m/y_m/z_m | 相机相对机器人基座的平移，米；前/左/上为正 |
| camera_yaw_rad | 相机相对基座的水平朝向，弧度 |
| camera_down_tilt_rad | 相机下俯角，弧度 |
| nominal_base_height_m | 基座名义世界高度，米 |
| floor_z_world_m | 同一世界坐标中的地面高度，米 |

其余参数可以在同一 JSON 中覆盖，例如 resolution_m（默认 0.05）、depth_stride（8）、
depth_min_m（0.1）、depth_max_m（5.0）、robot_radius_m（0.35）。地图使用固定高度、yaw 和
下俯角的简化几何模型，不支持完整动态六自由度外参。地图是探索进展证据，不替代 iPlanner。
配置必须与相机工厂和 SLAM 的基座定义一致。SLAM 重置或重定位改变坐标系时应结束并重新
启动 Episode；入口检测 frame_id 字符串变化并终止，不会自动检测同名坐标系内部跳变。

## 启动模板

下列大写占位符必须替换为本机实际值，地图文件必须填写实测值：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/run_g1_real.py \
  --instruction "Go through the door and stop near the sofa." \
  --model-url http://127.0.0.1:18765/v1/lavira/decision \
  --iplanner-url http://127.0.0.1:8888 \
  --network-interface YOUR_G1_INTERFACE \
  --camera-factory YOUR_CAMERA_MODULE:YOUR_FACTORY \
  --camera-config /absolute/path/camera_config \
  --odometry-topic /YOUR_ODOMETRY_TOPIC \
  --map-config /absolute/path/map.json \
  --rotation-duration-scale YOUR_MEASURED_SCALE
```

可选参数：`--g3-session-timeout-s 180`、`--g3-motion-window-s 1.0`、
`--odometry-startup-timeout-s 10`。不启用完整 G3 的旧接口诊断需显式加
`--no-g3-session`；该模式可省略 map-config，也不会调用会话接口。
新版真机全景仍需要 `--odometry-topic` 来检查转角，不能省略SLAM。

真机Python环境需要现有iPlanner客户端使用的 `requests`。本次已在
`/home/yile/projects/.venvs/unitree_g1` 补齐该依赖。

每次运行的输出目录默认在 `outputs/g1_real_unified_vln/<session_id>/run_.../`。
查看 `g3_health.json`、`g3_session_started.json`、决策/运动/动作报告和
`g3_session_ended.json` 确认实际协议过程。可用 `--output-dir` 覆盖根目录。

离线回归（不会连接机器人）：

```bash
python -m unittest discover -s scripts/isaacsim_lavira_g3_interface_g1/tests
```
