# D415 笔记本 USB 测试

**当前模式（2026-10-16起）：RGB与深度均为1280×720、15fps。** 正式D415配置已同步到720p，下面10月6日的640×480结果和内参属于历史记录，不用于当前模式。修改的只是自己的取图程序；SDK、固件和驱动不变。

2026-10-06：D415 直接连接笔记本。官方 librealsense 2.54.1 源码保持原样，只在用户目录编译；没有升级显卡驱动、内核或相机固件。程序名称沿用 `g1_d435i_*`，本次实际设备是 D415。

## 已验证结果

### 2026-10-16：正式切换720p

自己的C++文件取图和实时服务已重新编译，默认RGB8/Z16均为1280×720、15fps；480p可显式选择。正式D415接收配置和双相机标定的新相机配置已同步。SDK源码、安装库、相机固件、显卡驱动与安装外参均未修改。

| 检查 | 结果 |
|---|---|
| 默认文件取图 | RGB与对齐深度均1280×720；内参来自当前SDK流 |
| 480p切回 | `--resolution 480p`保存640×480成功 |
| 正式720p TCP取图 | 约20秒收到298组，14.89fps；请求加传输P95=67.98ms |
| RGB-D配对 | 时间差绝对值P95=0.601ms；沿用原50ms拒绝阈值，未放宽校验 |
| 非零深度比例 | 中位数82.21%，不代表测距准确率 |
| iPlanner实际请求 | 接收720p RGB/深度和本模式内参，返回51个路径点；本次首次请求约504ms，不代表稳定推理延迟 |
| 离线检查 | 202项接口/状态机测试、32项手触碰、12项双相机测试，共246项通过 |

当前720p RGB内参：

```text
K = [[911.0340,   0,      652.1967],
     [  0,     910.6531,  367.9189],
     [  0,       0,        1     ]]
```

记录目录：`outputs/camera_720p_switch/run_20261016_142455/`，含 `verification_report.json`、`preview.png`、正式请求的 `latest_plan.json`、原图/米制深度/内参，以及720p和480p文件样本。规划目标只是本次像素诊断输入，不是模型识别结果。未连接DDS、未发送运动命令、未验证避障；测试进程均已退出。相机当前接在笔记本USB，本次TCP性能只代表本机回环短测。

### 2026-10-06：历史480p结果

设备序列号 `211222063932`，固件 `5.16.0.1`；系统显示 USB 5000M。RGB8/Z16 均为 640×480、15fps，深度对齐到 RGB。

| 检查 | 结果 |
|---|---|
| 文件取图 | 15帧预热后读取150帧，保存最后一组，耗时13.569秒（含预热和保存） |
| 非零深度比例 | 文件样本67.39%；实时日志打印值49.5%～90.3%，随近距离遮挡及场景变化 |
| 深度尺度 | 约0.001米/原始计数，零值为缺测 |
| RGB-D 时间差 | 文件样本约0.601ms；实时日志打印值均约0.60ms |
| 60秒实时预览 | 收到890组，14.82fps；请求加接收P50=61.6ms，P95=63.7ms |
| 服务停止检测 | 对测试服务发送SIGINT后，Python抛出连接重置异常并关闭连接 |
| VLN输入格式 | 已通过现有`NetworkCamera.capture_forward()`的`ViewFrame`检查；尚未联调完整VLN |

上述实时传输发生在笔记本回环地址，不是机器人到笔记本的网络性能。非零深度比例不等于测距准确率。文件样本中有7个原始值65535的极端深度像素，保留原始数据供后续范围处理检查。

当天后续结果：已完成[手动四方向采集和远端G3静态接口联调](../README_CAMERA_MODEL_PROBE.md)，
以及[机器人四段旋转后的自动RGB-D采集](../README_ROTATION_CAPTURE.md)两轮测试。
真机专用全景状态机已接入正式入口，见[真机G3接入](../README_G1_G3.md)。
相机安装外参、固定安装后的准确视角与正式真机导航全链路仍待验证。

10月6日RGB内参（仅对应上述640×480分辨率）：

```text
K = [[607.3560,   0,      328.1311],
     [  0,     607.1021,  245.2793],
     [  0,       0,        1     ]]
```

对齐深度K与RGB一致；SDK返回畸变系数为零。深度到RGB的内部外参已保存，相机到机器人安装外参尚未测量。

## 目录

- 官方源码：`/home/yile/projects/librealsense-v2.54.1`
- 本机SDK：`/home/yile/projects/g1_camera_sdk/local/install-x86_64`
- 自己的可执行程序：`/home/yile/projects/g1_camera_sdk/local/build-probe-x86_64`
- 文件样本及预览：`/home/yile/projects/g1_camera_sdk/d415_rgbd_test_01`
- 实时样本及统计：`/home/yile/projects/g1_camera_sdk/d415_stream_test_01`
- 设备信息：`/home/yile/projects/g1_camera_sdk/d415_device_info.txt`
- 汇总：`/home/yile/projects/g1_camera_sdk/d415_test_summary.json`
- 断连记录：`/home/yile/projects/g1_camera_sdk/d415_disconnect_report.json`

## 再次测试

本机USB权限规则已经安装，重新插接后生效。取图本身不依赖ROS或Python；Python接收使用已有的`unitree_g1`环境。

终端1，运行取图服务：

```bash
/home/yile/projects/g1_camera_sdk/local/build-probe-x86_64/g1_d435i_stream \
  --bind 127.0.0.1 --port 8765 --serial 211222063932 \
  --resolution 720p \
  --settings-dir /home/yile/projects/g1_camera_sdk/d415_camera_logs
```

看到`RGB-D ready`后，终端2运行预览：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source /home/yile/projects/.venvs/unitree_g1/bin/activate
python scripts/isaacsim_lavira_g3_interface_g1/camera_d415/network_camera.py \
  --config scripts/isaacsim_lavira_g3_interface_g1/camera_d415/network_camera_d415_local.json \
  --seconds 60 --output /home/yile/projects/g1_camera_sdk/d415_stream_test_02
```

输出目录必须不存在。按`q`、Esc或Ctrl+C结束预览；终端1按Ctrl+C停止服务。本机10月16日更新后的服务允许最多4个接收端共享SDK采集，可同时运行预览与VLN接收器；上面的历史测试当时使用单接收端。
若需480p，服务端改为 `--resolution 480p`，接收端 `--config` 改为同目录的 `network_camera_d415_local_480p.json`；只改一端会被尺寸检查拒绝。

单次保存RGB-D时，先停掉实时服务，再运行：

```bash
/home/yile/projects/g1_camera_sdk/local/build-probe-x86_64/g1_d435i_probe \
  --frames 150 --output /home/yile/projects/g1_camera_sdk/d415_rgbd_test_02
/home/yile/projects/.venvs/g1_camera_analysis/bin/python \
  /home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_g3_interface_g1/camera_d415/inspect_capture.py \
  /home/yile/projects/g1_camera_sdk/d415_rgbd_test_02
```

## 之后接入机器人

相机到模型的手动四方向静态测试见[README_CAMERA_MODEL_PROBE.md](../README_CAMERA_MODEL_PROBE.md)，入口为`g1_camera_model_probe.py`。不要求机器人连接，不调用DDS或运控。

固定安装后测量相机到机器人外参；在机器人上编译ARM版并用新序列号启动服务，接收配置的host改为机器人IP。重新验证真实网络传输、图像与SLAM位姿配对，以及转向停稳后的四方向拍摄。还需用已知距离验证测距精度。
