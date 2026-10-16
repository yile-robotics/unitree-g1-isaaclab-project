# D415：RGB-D 取图与外参标定

当前设备为 Intel RealSense D415，序列号 `211222063932`。相机固定在机器人胸前，当前 USB 连接笔记本。源码目录已统一为 `camera_d415`，默认接收配置为 `network_camera_d415_local.json`。

## 数据链路

D415 → 官方 librealsense SDK → 自己的 C++ 取图服务 → TCP → Python `NetworkCamera` → VLN `ViewFrame`。

当前请求RGB8/Z16均为640×480、15fps（480p，不是720p）；深度对齐到RGB，接收后转换为米。
标定参数来自当前设备的实际流配置，不使用旧D435i的内参。

| 文件 | 作用 |
|---|---|
| `stream.cpp` | 持续取RGB-D，通过TCP发送最新帧 |
| `capture.cpp` | 保存最后一组RGB-D和标定参数 |
| `network_camera.py` | Python接收、预览和VLN相机工厂 |
| `network_camera_d415_local.json` | 当前D415的地址、序列号及接收阈值 |
| `network_camera.json` | 历史机器人内置相机配置，需显式指定才使用 |
| `sensor_settings.hpp` | 读取和保存实际启动参数 |
| `inspect_capture.py` | 分析保存样本 |
| `CMakeLists.txt` / `build_sdk.sh` | 构建自己的取图程序及首次安装SDK |

## 操作与记录

- [D415启动命令与实测结果](README_D415_LOCAL.md)
- [胸部外参标定](../../camera_extrinsics/README.md)
- [旋转后自动采集](../README_ROTATION_CAPTURE.md)
- [手动采集与模型联调](../README_CAMERA_MODEL_PROBE.md)
- [实时iPlanner观察窗口](../README_IPLANNER_VIEWER.md)
- [10月13/15日实验与结果证据](../../../docs/real_robot/README.md)
- [历史D435i记录](README_D435I_HISTORY.md)：其中序列号、内参与性能数据属于旧相机。

VLN使用的工厂和配置：

```text
--camera-factory camera_d415.network_camera:create_camera
--camera-config scripts/isaacsim_lavira_g3_interface_g1/camera_d415/network_camera_d415_local.json
```

官方SDK源码和安装库保持不变。现有已编译程序名暂沿用 `g1_d435i_stream` / `g1_d435i_probe`，这是旧可执行文件名，实际设备由序列号选择，支持当前D415；程序位置与现有启动命令保持一致。

相机到胸部外参正在标定；尚无完成验证、可直接用于导航的外参。

## 时间与关闭语义

TCP负责可靠有序传输字节，不保证曝光同步。RGB和深度保留各自SDK时间域、时间戳和帧号，
接收端检查配对时间差和两路帧号更新；跨PC2/笔记本的双相机同时请求不等于同时曝光。
双相机标定依靠板和相机在采集窗口静止，尚未解决运动中的图像、关节、SLAM曝光时刻配对。
`ViewFrame.timestamp`是调用方时间轴上的收齐图像时间，不是曝光时间。

Python `NetworkCamera.close()`只关闭TCP连接；C++服务仍可继续取图并等待下一个客户端。
服务端正常退出采集循环时才调用SDK `pipeline.stop()`。USB掉线后旧进程可能无法出图或正常退出，
应确认真正的新鲜RGB-D恢复，不能只检查8765端口。
