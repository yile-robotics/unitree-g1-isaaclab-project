# D415 笔记本 USB 测试

2026-10-06：D415 直接连接笔记本。官方 librealsense 2.54.1 源码保持原样，只在用户目录编译；没有升级显卡驱动、内核或相机固件。程序名称沿用 `g1_d435i_*`，本次实际设备是 D415。

## 已验证结果

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

当前RGB内参（对应上述分辨率）：

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
  --settings-dir /home/yile/projects/g1_camera_sdk/d415_camera_logs
```

看到`RGB-D ready`后，终端2运行预览：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source /home/yile/projects/.venvs/unitree_g1/bin/activate
python scripts/isaacsim_lavira_g3_interface_g1/camera_d435i/network_camera.py \
  --config scripts/isaacsim_lavira_g3_interface_g1/camera_d435i/network_camera_d415_local.json \
  --seconds 60 --output /home/yile/projects/g1_camera_sdk/d415_stream_test_02
```

输出目录必须不存在。按`q`、Esc或Ctrl+C结束预览；终端1按Ctrl+C停止服务。每次仅允许一个接收端，不能同时运行预览与VLN接收器。

单次保存RGB-D时，先停掉实时服务，再运行：

```bash
/home/yile/projects/g1_camera_sdk/local/build-probe-x86_64/g1_d435i_probe \
  --frames 150 --output /home/yile/projects/g1_camera_sdk/d415_rgbd_test_02
/home/yile/projects/.venvs/g1_camera_analysis/bin/python \
  /home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_g3_interface_g1/camera_d435i/inspect_capture.py \
  /home/yile/projects/g1_camera_sdk/d415_rgbd_test_02
```

## 之后接入机器人

相机到模型的手动四方向静态测试见[README_CAMERA_MODEL_PROBE.md](../README_CAMERA_MODEL_PROBE.md)，入口为`g1_camera_model_probe.py`。不要求机器人连接，不调用DDS或运控。

固定安装后测量相机到机器人外参；在机器人上编译ARM版并用新序列号启动服务，接收配置的host改为机器人IP。重新验证真实网络传输、图像与SLAM位姿配对，以及转向停稳后的四方向拍摄。还需用已知距离验证测距精度。
