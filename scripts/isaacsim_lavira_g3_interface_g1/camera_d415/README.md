# D415：RGB-D 取图与外参标定

当前设备为 Intel RealSense D415，序列号 `211222063932`。相机固定在机器人胸前，当前 USB 连接笔记本。源码目录已统一为 `camera_d415`，默认接收配置为 `network_camera_d415_local.json`。

## 数据链路

D415 → 官方 librealsense SDK → 自己的 C++ 取图服务 → TCP → Python `NetworkCamera` → VLN `ViewFrame`。

2026-10-16起，当前D415默认请求RGB8/Z16均为1280×720、15fps（720p）；深度对齐到RGB，接收后转换为米。
标定参数来自当前设备的实际流配置，不使用旧D435i的内参。
只改变分辨率且安装不动时，安装外参保持原值；内参必须使用SDK按当前流返回的值。以前的480p数据保留原内参，不改写历史样本。

| 文件 | 作用 |
|---|---|
| `stream.cpp` | 持续取RGB-D，通过TCP发送最新帧 |
| `capture.cpp` | 保存最后一组RGB-D和标定参数 |
| `network_camera.py` | Python接收、预览和VLN相机工厂 |
| `../live_model_target.py` | 监听最近模型决策，替换实时RGB上的显示框 |
| `network_camera_d415_local.json` | 当前D415的地址、序列号及接收阈值 |
| `network_camera_d415_local_480p.json` | 回到旧640×480模式时使用 |
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

## 分辨率切换

文件取图和实时服务默认720p，也支持 `--resolution 480p`。服务器与接收配置必须匹配：

| 模式 | C++参数 | Python相机配置 |
|---|---|---|
| 当前D415 720p | `--resolution 720p`（默认） | `network_camera_d415_local.json` |
| D415 480p | `--resolution 480p` | `network_camera_d415_local_480p.json` |
| 历史内置D435i 480p | `--resolution 480p` | `network_camera.json` |

已有SDK无需重编译，只重编译自己的程序：
如果构建目录来自改名前的 `camera_d435i`，先删除该构建目录内生成的 `CMakeCache.txt` 和 `CMakeFiles/`，再运行下面命令；不要删除SDK源码或安装库。

```bash
cmake -S /home/yile/projects/unitree-g1-isaaclab-project/scripts/isaacsim_lavira_g3_interface_g1/camera_d415 \
  -B /home/yile/projects/g1_camera_sdk/local/build-probe-x86_64 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/home/yile/projects/g1_camera_sdk/local/install-x86_64
cmake --build /home/yile/projects/g1_camera_sdk/local/build-probe-x86_64 --parallel 2
```

当前笔记本启动命令不变；可显式添加 `--resolution 720p`。这些本机编译产物不能直接复制到ARM机器人运行。
iPlanner观察窗口按比例显示720p；回放480p时两侧留黑，黑边不接受目标点击。iPlanner网络仍把深度缩放到640×360，不改变权重和网络结构；高清采集不等于避障已经验证。

## 实时RGB中的模型目标框

纯相机预览和iPlanner实时窗口现在默认监听 `outputs/d415_rotation_model/` 下的模型结果。
相机画面持续更新，最近一次前方NAVIGATE决策的bbox以绿框覆盖在实时RGB上；下一条决策完整写入后自动替换，STOP/BACKTRACK/SAFE_STOP/无框响应清除旧框。中文目标名称、方向和决策编号显示在窗口标题。纯相机预览只做显示，不改变原RGB、深度或运动命令。统一iPlanner窗口默认还将模型框底边中心作为规划目标，详见[统一实时监控](../README_IPLANNER_VIEWER.md)。

默认启动命令不变。观察单次模型任务或正式真机输出时可指定：

```bash
python scripts/isaacsim_lavira_g3_interface_g1/camera_d415/network_camera.py \
  --model-results /绝对路径/本次任务结果目录 --seconds 3600
```

`--model-results`可指向某次运行目录、多个运行的父目录或单个响应JSON。读取探针的 `model/response.json` 与 `request_metadata.json`，或正式真机的 `decision_NNN_response.json` 与对应request。默认多运行目录按响应文件修改时间选最近结果，正式任务建议绑定本次输出目录，避免其他任务覆盖显示。读取未写完的JSON时保留上一框并重试。

这不是目标跟踪：停车决策后框保持在返回的像素位置。窗口只显示打开后写入的新响应；`g1_rotation_model_probe.py` 开始新一轮旋转前写入 `display_motion.json` 清除旧框，失败/停止不恢复旧框。其他入口/遥控运动前在图像窗口按 `H` 手动隐藏；本窗口不检测物理运动。旧480p框不直接拉伸到720p，尺寸不符只提示。左/后/右视图的框先等待转向；正式 `RealG1Episode` 在转向后重新取得前视帧，写入匹配的 `decision_NNN_overlay.json` 后才允许显示。仅旋转采图、不执行模型导航的探针没有这个转向确认，因此非前方目标只显示方向提示。

本机新C++服务最多支持4个接收端，共享同一个SDK采集线程；实时预览可与模型采集同时取图。每个接收端独立请求新帧，断开的线程会回收，不增加相机曝光次数。机器人上未更新的旧服务仍只支持单客户端。

本次验证：预览保持运行，同时第二个720p接收端在12秒内收到180组、14.90fps、请求加传输P95=67.90ms；第二端连续重连5次成功。使用独立目录的明确标记测试决策，验证“框A→框B→STOP清除”，未把它们写入真实模型结果目录，也未调用模型或机器人。257项离线测试通过，包含11项新增的显示更新/转向确认检查。记录在 `outputs/live_model_overlay/run_20261016_144237/`；本次没有实测机器人转向后的框位置精度，静态像素框也不保证运动后的物体对齐。

## 时间与关闭语义

TCP负责可靠有序传输字节，不保证曝光同步。RGB和深度保留各自SDK时间域、时间戳和帧号，
接收端检查配对时间差和两路帧号更新；跨PC2/笔记本的双相机同时请求不等于同时曝光。
双相机标定依靠板和相机在采集窗口静止，尚未解决运动中的图像、关节、SLAM曝光时刻配对。
`ViewFrame.timestamp`是调用方时间轴上的收齐图像时间，不是曝光时间。

Python `NetworkCamera.close()`只关闭TCP连接；C++服务仍可继续取图并等待下一个客户端。
服务端正常退出采集循环时才调用SDK `pipeline.stop()`。USB掉线后旧进程可能无法出图或正常退出，
应确认真正的新鲜RGB-D恢复，不能只检查8765端口。
