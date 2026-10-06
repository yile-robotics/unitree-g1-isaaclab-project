# 真机四段旋转与RGB-D采集

入口为 `g1_vln_rotation_probe.py --mode imu75_panorama --capture-panorama`。复用已通过真机测试的旋转：每段持续发送 `wz=0.8 rad/s`，IMU相对转角达到75°时调用正式DDS后端的 `stop()`，等待0.5秒，再用SLAM检查本段停稳转角是否在70°～110°。75°是提前停车阈值；实际最终转角受制动影响，不能保证恰好90°。

同一控制与采集顺序已接入正式真机入口 `run_g1_real.py` 的 `RealG1Episode`，说明见
[真机G3接入](README_G1_G3.md)。本文件仍是独立探针的启动步骤，不运行模型或iPlanner。

拍摄顺序为初始前方、第一段后的左方、第二段后的后方、第三段后的右方，第四段转回前方后额外保存一张回正图。四段自动衔接，无需按Enter。本入口不连接模型，不执行前进导航。

## 1. 保持相机服务运行

D415仍通过USB连接笔记本。同一时间只运行一个取图客户端，先退出旧的预览或手动采集程序。相机服务已运行时直接使用，勿重复启动；未运行时在笔记本终端执行：

```bash
/home/yile/projects/g1_camera_sdk/local/build-probe-x86_64/g1_d435i_stream \
  --bind 127.0.0.1 --port 8765 --serial 211222063932 \
  --settings-dir /home/yile/projects/g1_camera_sdk/d415_camera_logs
```

## 2. 先开RViz，再启动重定位

笔记本新终端打开RViz：

```bash
source /home/yile/projects/unitree_ros2/setup.sh
rviz2 -d /home/yile/projects/unitree_slam_example/rviz2/relocation.rviz
```

另一个笔记本终端启动SLAM示例：

```bash
cd /home/yile/projects/unitree_slam_example/example/build
./keyDemo enxf8e43bbea486
```

按一次 `a`，收到成功响应后确认RViz地图中的位置正确。保持这些窗口运行。旋转入口直接订阅DDS的 `rt/slam_info`，其终端不需要source ROS环境。

## 3. 桌面相机先验证功能

机器人转动，相机留在桌上。保存的方向名称表示机器人所在旋转阶段；图像不代表真实前后左右视野。本次只检查旋转、停车、取图、保存的衔接。

在新的笔记本终端运行，便于随时按Ctrl+C：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source /home/yile/projects/.venvs/unitree_g1/bin/activate
python scripts/isaacsim_lavira_g3_interface_g1/g1_vln_rotation_probe.py \
  --network-interface enxf8e43bbea486 \
  --mode imu75_panorama --quarters 4 \
  --capture-panorama --camera-on-desk \
  --output /home/yile/projects/g1_camera_sdk/d415_rotation_capture_01 \
  --execute
```

启动时先接收一组RGB-D并检查IMU与SLAM；倒数3秒后开始流程。Ctrl+C会先请求停车、关闭DDS后端，然后关闭相机连接；相机服务可以继续运行。取图失败会结束流程，不再开始下一段。每段仍保留10秒时限。

输出目录必须是新目录，重复测试请修改末尾编号。相机真正固定到机器人后，去掉 `--camera-on-desk`，换一个新目录；USB和连接地址若变化，使用 `--camera-config` 指定对应配置。

手动让相机跟随机器人旋转时，将 `--camera-on-desk` 换成 `--camera-manual-follow`。采集时序和旋转参数相同，清单标记 `rotation_manual_follow`，不把机器人yaw当作已验证的相机朝向。该数据先用于查看四张图的视角效果，模型入口不会将其当作固定安装全景上传。

## 4. 检查结果

正常完成时终端依次打印 `已保存 forward / left / behind / right / forward_return` 和累计SLAM转角。保存内容：

- `current_forward.png`、`current_left.png`、`current_behind.png`、`current_right.png`：四个阶段的RGB。
- `*_depth_m.npy`：每张RGB对应的米制对齐深度。
- `current_forward_return.png` 与对应深度：第四段结束的回正图，不覆盖初始前方。
- `panorama.json`：完成状态、K、帧号、相机时间与传输信息、各次取图前后的IMU和SLAM位姿，以及四段SLAM转角。

取图前后位姿是观测记录，尚未插值到曝光时刻。桌面测试清单标记 `capture_mode=rotation_bench`，模型重放入口会拒绝将它当作真实四方向全景。相机随机器人转动且完整完成的采集可供 `g1_camera_model_probe.py --panorama-dir` 重放，但世界目标投影仍需要实测安装外参。

离线验证覆盖旋转停车后的采集顺序、相机失败不再旋转、五组RGB-D保存、初始图与回正图分离、资源清理和桌面数据标记。实际整圈旋转与取图是否成功需依据现场输出和动作确认。

## 2026-10-06 实测

结果目录：`/home/yile/projects/g1_camera_sdk/d415_rotation_capture_03`。机器人执行四段左转，D415通过USB连接笔记本且仍在桌上，使用 `--camera-on-desk`。清单状态为 `PASS`，五组RGB与米制深度文件均已检查，帧号分别为624、685、731、774、850，初始前方图未被回正图覆盖。

| 分段 | 停车后IMU相对转角 | 停车后SLAM相对转角 |
| --- | --- | --- |
| 1 | 89.9° | 97.0° |
| 2 | 88.8° | 87.1° |
| 3 | 86.2° | 84.5° |
| 4 | 88.0° | 92.4° |

SLAM四段合计361.038°。RGB均为640×480，深度均为480×640的float32米制矩阵，非零深度比例49.5%～74.4%。本次确认实际旋转、分段停车与真实RGB-D采集保存能自动衔接；相机未随机器人转动，尚未验证真实四方向全景、安装外参或全景采集后的模型导航闭环。

### 第二轮：手动让相机跟随旋转

结果目录：`/home/yile/projects/g1_camera_sdk/d415_rotation_capture_04`。使用
`--camera-manual-follow`，状态 `PASS`，帧号为3992、4070、4118、4160、4212。
五张RGB拼图保存在同目录 `rgb_capture_overview.jpg`，已在本机查看。图像出现不同视角，
存在手持倾斜，回正图右侧被近处的人遮挡；相机朝向没有独立测量，不能将机器人yaw
直接当作相机每次转了90°的证据。

| 分段 | 停车后IMU相对转角 | 停车后SLAM相对转角 |
| --- | --- | --- |
| 1 | 88.8° | 96.3° |
| 2 | 87.8° | 85.0° |
| 3 | 83.0° | 81.1° |
| 4 | 89.3° | 90.9° |

分段SLAM转角合计353.202°。另按每次取图前后记录的SLAM位姿，与初始前方相比：

| 采集阶段 | 理想累计朝向 | 取图时机器人累计朝向 |
| --- | --- | --- |
| forward | 0° | 0° |
| left | 90° | 96.3° |
| behind | 180° | 181.3° |
| right | 270° | 262.3° |
| forward_return | 360° | 352.9° |

分段与取图时的统计使用不同时间点作为基线，不混为同一个累计量。每次请求图像前后的
SLAM yaw没有变化；确认停车调用返回、等待0.5秒后才取图，采图完成后再进入下一段。
这验证了当前脚本的采集时序，没有机器人侧“已停稳可拍照”的确认，也未检测角速度
持续接近零。每段是约90°，不能保证恰好90°或完全没有余动。

### 正式状态机接入

当天已将上述旋转、主动停车和延时采图接入 `run_g1_real.py` 使用的真机专用
`unified_vln/real_episode.py`，并新增专用 `rt/lowstate` IMU订阅与清理。
仿真入口和共享 `episode.py`、`rotation.py` 未修改；186项离线回归通过。
两次本节实测运行的是独立探针，新正式入口仍待相机固定后的全链路现场验证。
