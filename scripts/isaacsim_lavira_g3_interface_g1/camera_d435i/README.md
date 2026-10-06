# G1 D435i：取图、实时传输与 VLN 接口

D415直接连接笔记本的测试和命令见[README_D415_LOCAL.md](README_D415_LOCAL.md)。

更新：2026-10-02。已通过机器人端 SDK 取图、30/60 秒网络预览、启动参数保存和服务停止后的断流检测。完整 VLN 尚未联调。

## 代码与数据链路

D435i → 机器人官方 C++ SDK → `g1_d435i_stream` → TCP → 笔记本 `NetworkCamera` → `ViewFrame`。

| 文件 | 用途 |
|---|---|
| `stream.cpp` | 持续采集 RGB-D，逐帧对齐深度，通过 TCP 发送最新观测 |
| `network_camera.py` | Python 接收器、预览入口和 VLN 相机工厂 |
| `network_camera.json` | 地址、序列号、分辨率及时间阈值 |
| `sensor_settings.hpp` | 读取、打印和保存启动参数 |
| `capture.cpp` | 连续读取若干帧，保存最后一组 RGB-D 和标定 |
| `inspect_capture.py` | 分析保存样本并生成深度可视化 |
| `build_sdk.sh` / `CMakeLists.txt` | 编译 SDK 和两个取图程序 |

使用[官方 librealsense v2.54.1](https://github.com/realsenseai/librealsense/tree/v2.54.1)，固定提交 `8ffb17b027e100c2a14fa21f01f97a1921ec1e1b`，源码位于 `/home/yile/projects/librealsense-v2.54.1`。RSUSB 后端运行于用户态；构建不启用 CUDA，不升级显卡驱动、内核或固件。取图不依赖 ROS，笔记本无需安装 `pyrealsense2`。

## 实时取图

配置为 RGB8/Z16、640×480、15fps。深度对齐到 RGB，接收端乘设备实际尺度得到米。一次连接一个客户端，预览和 VLN 顺序使用。

### 1. 更新机器人程序

先在机器人相机服务终端按 Ctrl+C。然后在**笔记本**传文件并登录：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
scp scripts/isaacsim_lavira_g3_interface_g1/camera_d435i/{stream.cpp,capture.cpp,sensor_settings.hpp,CMakeLists.txt} \
  unitree@192.168.123.164:/home/unitree/g1_camera_sdk/camera_d435i/
ssh unitree@192.168.123.164
```

ROS 选择提示输入 `1`。在**机器人 SSH 终端**编译；已有 SDK 无需重编：

```bash
cmake -S /home/unitree/g1_camera_sdk/camera_d435i \
  -B /home/unitree/g1_camera_sdk/local/build-probe-aarch64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/home/unitree/g1_camera_sdk/local/install-aarch64
cmake --build /home/unitree/g1_camera_sdk/local/build-probe-aarch64 \
  --target g1_d435i_stream g1_d435i_probe -j 2
```

### 2. 启动机器人服务

在**机器人 SSH 终端**执行，关闭其他占用 D435i 的程序：

```bash
/home/unitree/g1_camera_sdk/local/build-probe-aarch64/g1_d435i_stream \
  --bind 192.168.123.164 --port 8765 --serial 344422072128 \
  --settings-dir /home/unitree/g1_camera_sdk/camera_logs
```

看到 `RGB-D ready` 后保持运行；Ctrl+C 结束服务。

### 3. 笔记本预览和保存

在**笔记本另一个终端**执行：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source ../.venvs/unitree_g1/bin/activate
python scripts/isaacsim_lavira_g3_interface_g1/camera_d435i/network_camera.py \
  --seconds 60 --output /home/yile/projects/rgbd_stream_settings_test_02
```

输出目录必须不存在。窗口左侧 RGB、右侧深度；深度显示范围 0～5 米，黑色为缺测。按 `q`、Esc 或 Ctrl+C 结束接收；机器人服务可继续运行。加 `--headless` 只在终端输出。

只保存最后一组：`rgb.png`、米制 `depth_m.npy`、`stream_report.json`。报告包含帧率、请求耗时、K、尺度、时间戳、相机内部外参和启动参数，不是整段录像。

断流测试：接收期间在机器人服务终端按 Ctrl+C。笔记本应输出 `Robot camera stream disconnected` 并退出。恢复时重新启动两端。

## 启动参数记录

`pipeline.start()` 后、预热前，通过 `supports()`/`get_option()` 顺序读取：自动曝光、曝光值、深度预设、发射器状态、激光功率。终端打印 `[CAMERA SETTINGS]`，不调用 `set_option()`。

- 实时服务：每次在 `--settings-dir` 下生成独立 `startup-*.json`；默认目录为当前工作目录的 `camera_logs/`。文件名使用单调时钟和 PID。
- 笔记本：预览加 `--output`，报告的 `last_frame.startup_settings` 保留同一份快照。
- 文件取图：样本 `metadata.json` 的 `startup_settings` 保存快照。

不支持的选项标记 `unsupported`；查询失败标记 `read_error`，值均为 null。单项查询失败不阻止取图，文件写入失败会报错。自动曝光开启时，启动读数不代表之后每帧的曝光。

## 单次取图与分析

在**机器人**运行，每次使用新目录：

```bash
# 只保存 RGB
/home/unitree/g1_camera_sdk/local/build-probe-aarch64/g1_d435i_probe \
  --rgb-only --output /home/unitree/g1_camera_sdk/rgb_test_02

# 15帧预热后读取150帧，保存最后一组
/home/unitree/g1_camera_sdk/local/build-probe-aarch64/g1_d435i_probe \
  --frames 150 --output /home/unitree/g1_camera_sdk/rgbd_test_02
```

输出 `rgb.ppm`、16位大端 `depth_aligned_z16.pgm` 和 `metadata.json`。原始深度乘 `depth_scale_m` 得米，零值为缺测。Ctrl+C 退出，等待帧时可能延迟最多5秒响应。

在**笔记本**复制并分析：

```bash
scp -r unitree@192.168.123.164:/home/unitree/g1_camera_sdk/rgbd_test_02 /home/yile/projects/
/home/yile/projects/.venvs/g1_camera_analysis/bin/python \
  /home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_g3_interface_g1/camera_d435i/inspect_capture.py \
  /home/yile/projects/rgbd_test_02
xdg-open /home/yile/projects/rgbd_test_02/rgbd_preview.png
```

分析图用有效深度 p99 截断色轴，不修改原始数据。

## 实测记录（2026-10-02，笔记本日期）

设备为 D435i，序列号 `344422072128`，固件 `5.15.1.55`，USB 5000M，重启后 uvcvideo 已绑定。机器人为 aarch64，SDK 2.54.1。

| 项目 | 文件取图150帧 | 网络预览30秒 | 网络预览60秒 |
|---|---:|---:|---:|
| 接收组数 | 150（另有15帧预热） | 174 | 346 |
| 接收帧率 | 未单独测量 | 5.79fps | 5.76fps |
| 请求加传输 P50 | — | — | 168.0ms |
| 请求加传输 P95 | — | 168.7ms | 169.6ms |
| RGB-D 时间差 | 最后一组+0.044ms | 打印值均+0.04ms | 最后一组+0.044ms |
| 非零深度比例 | 最后一组97.832% | 打印值95.6%～98.1% | 最后一组96.86% |
| 有效深度中位数 | 最后一组1.576m | 打印值1.518～1.625m | — |

标定：`fx=606.8643, fy=606.8256, cx=325.2682, cy=256.8781`，畸变系数均0，对齐后深度 K 与 RGB K 相同；尺度约0.001米/原始计数，尺寸均640×480。

- 文件取图总耗时13.639秒含启动、预热和保存，不能据此计算采集帧率。样本及分析保存在 `/home/yile/projects/rgbd_test_01`。
- 30秒预览打印的RGB帧号4915→5350、Depth帧号4913→5348；请求耗时164.4～170.0ms。Qt字体警告未阻止窗口显示或正常结束。
- 60秒样本位于 `/home/yile/projects/rgbd_stream_settings_test_01`，已检查RGB文件、480×640 float32米制深度和报告中的启动参数。
- 启动参数：深度自动曝光1、曝光读数8500、预设0（Custom）、发射器1（Laser）、激光功率150；RGB自动曝光1、曝光读数166。
- **服务停止断流检测通过**：机器人端Ctrl+C后，笔记本报 `Robot camera stream disconnected` 并退出。未测精确检测时延，也未覆盖拔网线、拔USB或完整VLN失联停车。

设备配置15fps与客户端接收帧率不同：按需取最新帧会跳过中间帧。两路帧号独立计数，以时间戳检查配对。请求耗时含等帧和传输；非零比例不是测距精度，SDK时间差不是独立曝光同步测量。

早期一次RGB冻结：连续8组网络数据中RGB停在第89帧，Depth从1359增至1378，时间差90.959～92.226秒。SDK聚合器可缓存各流最近帧，因此新frameset不保证两路都更新。接收器已分别检查两路帧号及时间差；后续30/60秒测试未重现，根因尚未确定。

## VLN 接入约定

`run_g1_real.py`的相机参数（不是完整启动命令）：

```text
--camera-factory camera_d435i.network_camera:create_camera
--camera-config scripts/isaacsim_lavira_g3_interface_g1/camera_d435i/network_camera.json
```

`capture_forward(sim_step, timestamp)`返回 `ViewFrame`：RGB uint8、同尺寸float32米制深度、设备K、递增帧号及 `forward` 方向。服务收到 `NEXT` 后等待随后采集的一组图；超时、断流、重复帧、全零深度或标定错误会关闭连接并抛异常。

`ViewFrame.timestamp`是调用方时间轴上的**笔记本接收时刻**，不是曝光时间。`last_metadata`保留SDK时间域、原始时间戳、服务端帧年龄和客户端接收单调时间。机器人时钟曾为1970年，不能直接与笔记本epoch相减。当前非零畸变会被拒绝，换相机后需要先做RGB-D校正。

下一步：

1. 测量相机→机器人安装外参、深度精度和物体边缘配准，并将图像与SLAM位姿按时间配对。SDK的depth→color只是相机内部外参，安装外参仍为null。
2. 接入四方向旋转、停稳和真实拍照，记录每张图实际朝向与位移。
3. 用真实图像联调模型、bbox投影及iPlanner，再做SLAM反馈下的局部目标跟踪。
4. 适配正式runner的运动参数：当前默认walk=0.3、max-forward=0.4m/s、rotation=0.4rad/s；探针通过的是0.5m/s前进及0.8rad/s、IMU75°提前停车。正式旋转器使用SLAM反馈，两者尚未统一。
5. 验证完整VLN及Ctrl+C、相机/定位/服务器失效后的停车；继续测试长时间取图及网络无响应超时。

## 构建与离线验证

换机器人时，先传送官方源码和本目录：

```bash
ssh unitree@192.168.123.164 'mkdir -p /home/unitree/g1_camera_sdk'
scp -r /home/yile/projects/librealsense-v2.54.1 unitree@192.168.123.164:/home/unitree/g1_camera_sdk/
scp -r /home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_g3_interface_g1/camera_d435i unitree@192.168.123.164:/home/unitree/g1_camera_sdk/
```

在机器人执行；缺依赖时再安装git、build-essential、cmake、pkg-config、libusb-1.0-0-dev、libudev-dev、libssl-dev：

```bash
bash /home/unitree/g1_camera_sdk/camera_d435i/build_sdk.sh \
  /home/unitree/g1_camera_sdk/librealsense-v2.54.1 \
  /home/unitree/g1_camera_sdk/local
```

SDK安装到 `local/install-aarch64`，程序在 `local/build-probe-aarch64`。参考官方[Ubuntu安装说明](https://github.com/realsenseai/librealsense/blob/v2.54.1/doc/installation.md)及[Jetson说明](https://github.com/realsenseai/librealsense/blob/v2.54.1/doc/installation_jetson.md)。

笔记本已完成C++编译、参数JSON检查及8项离线网络测试。另用保存样本经C++发送函数和Python工厂传输5组完整图像，解码一致；这是回环验证，不是额外真机采集。

```bash
/home/yile/projects/.venvs/unitree_g1/bin/python -m unittest discover \
  -s scripts/isaacsim_lavira_g3_interface_g1/tests -p test_network_camera.py -v
```
