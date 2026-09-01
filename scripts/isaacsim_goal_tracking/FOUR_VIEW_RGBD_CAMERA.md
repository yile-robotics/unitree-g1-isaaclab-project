# G1 四方向 RGB-D 相机接口说明

本文记录 `isaacsim_goal_tracking` 中已经实现并验证的四方向 RGB-D 相机系统。
它是 Isaac Sim 本机侧的 VLN 感知入口，负责生成同步的四视图 RGB-D 数据和相机几何信息；
当前阶段不包含远程 LaViRA 模型调用、FMM 路径规划或机器人速度命令生成。


代码：
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch --house --device cuda:0 \
  --four_rgbd_cameras \
  --no-show_path


## 1. 当前实现解决了什么

启用相机功能后，G1 的 `torso_link` 上会创建四台真实的 IsaacLab
`Camera` sensor：

- `forward`：机器人前方
- `left`：机器人左方
- `behind`：机器人后方
- `right`：机器人右方

每次正式抓取会得到同一个仿真 step 下的：

- 四路 RGB 数组
- 四路米制深度数组
- 每台相机的内参矩阵 `K`
- 抓图时的机器人世界位姿 `T_world_base`
- 每台相机的世界位姿 `T_world_camera_ros`
- 每台相机相对机器人根节点的外参 `T_base_camera`
- `bundle_id`、`sim_step`、`timestamp` 和 sensor frame id

相机功能可以用于 `stand`、`locomotion` 和 `switch` 三种运行模式，且不会改变现有
stand/locomotion policy 的动作输出或切换逻辑。当前读取器只采集 `env_0`。

## 2. 代码位置

| 文件 | 职责 |
| --- | --- |
| [`goal_tracking/camera.py`](goal_tracking/camera.py) | 定义四个方向、安装外参、创建 IsaacLab RGB-D sensor，并支持 GUI viewport 切换 |
| [`goal_tracking/frame_bundle.py`](goal_tracking/frame_bundle.py) | 同步相机位姿、渲染、复制 RGB-D、生成 `FrameBundle`、校验和调试落盘 |
| [`goal_tracking/config.py`](goal_tracking/config.py) | 定义相机相关命令行参数和默认值 |
| [`goal_tracking/runners.py`](goal_tracking/runners.py) | 将相机同步与抓取接入三个仿真运行循环 |
| [`isaacsim_path_follwing.py`](isaacsim_path_follwing.py) | 在 Isaac Sim 启动前打开 RTX camera，并在环境创建前注册四台 sensor |
| [`tests/test_camera_geometry.py`](tests/test_camera_geometry.py) | 验证四方向光轴、向下俯角和坐标变换 |

`camera.py` 中还保留了早期的单个 USD 调试相机 `attach_head_camera()`。它只用于
viewport 调试，不是现在的四路 RGB-D 模型输入。启用 `--four_rgbd_cameras` 后不会再创建
这个旧相机，避免出现第五台重复相机。

## 3. 相机安装位置与方向

四台相机都挂在：

```text
{ENV_REGEX_NS}/Robot/torso_link/
```

对应的 USD prim 名称为：

```text
lavira_camera_forward
lavira_camera_left
lavira_camera_behind
lavira_camera_right
```

经过相机刚性跟随修复后的实际行走校准，当前 G1 的导航前方是
`torso_link +X`。因此代码中的固定安装关系为：

| 语义方向 | `torso_link` 中的方向 | 默认光心位置（米） | 安装 yaw |
| --- | --- | --- | --- |
| `forward` | `+X` | `(0.085, 0, 0.56)` | `0°` |
| `left` | `+Y` | `(0, 0.085, 0.56)` | `90°` |
| `behind` | `-X` | `(-0.085, 0, 0.56)` | `180°` |
| `right` | `-Y` | `(0, -0.085, 0.56)` | `-90°` |

四台相机默认统一向下倾斜 `12°`，使画面能够覆盖较近的地面和障碍物。相机高度和半径
描述的是计划安装在 G1 头部小篮子四侧的四个光心，而不是 Habitat point-agent 的
单一相机位置。

这里修改的是每台相机的真实安装位置和旋转，不是仅交换四张图的文字标签。

## 4. 默认成像参数

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--rgbd_camera_width` | `640` | 图像宽度 |
| `--rgbd_camera_height` | `480` | 图像高度 |
| `--rgbd_camera_hfov_deg` | `79` | 水平视场角，与 LaViRA Habitat 配置对齐 |
| `--rgbd_camera_near` | `0.1` m | 近裁剪面 |
| `--rgbd_camera_far` | `5.0` m | 远裁剪面，与 LaViRA Habitat 配置对齐 |
| `--camera_rig_height` | `0.56` m | 相对 `torso_link` 的篮子光心高度 |
| `--camera_rig_radius` | `0.085` m | 光心距篮子中心的水平距离 |
| `--camera_down_tilt_deg` | `12`° | 正数表示光轴向下 |
| `--rgbd_camera_update_period` | `0.0` s | 每次渲染更新；当前同步实现要求保持为 `0` |

启用 `--four_rgbd_cameras` 时，入口脚本会在创建 Isaac Sim application 之前自动设置
`enable_cameras=True`，不需要再额外传 `--enable_cameras`。

## 5. 启动和抓图流程

整体调用顺序如下：

```text
解析命令行参数
  -> 启用 RTX camera
  -> configure_four_rgbd_cameras(env_cfg, args_cli)
  -> gym.make(...) 创建四台 IsaacLab Camera sensor
  -> FourViewCameraRig(raw_env, ..., env_index=0)
  -> 运行 stand / locomotion / switch 主循环
  -> 根据当前 torso_link 位姿同步四台相机
  -> env.step(actions)
  -> 需要模型帧时调用 camera_rig.capture(...)
  -> 得到独立复制的 FrameBundle
```

`capture()` 是提供给后续模型接口使用的权威抓图入口。它执行以下操作：

1. 读取当前 `torso_link` 位姿。
2. 根据四台相机的固定安装外参计算世界位姿。
3. 让 `torso_link` 到 Camera 的固定局部父子关系负责实际相机运动，不写 Camera world pose。
4. 在不推进 physics 的情况下额外 render 一次。
5. 强制更新四台 sensor buffer。
6. 复制 RGB、depth 和内参到独立的 CPU NumPy 数组。
7. 使用当前 torso tensor 和固定局部外参生成对应相机位姿。
8. 验证仿真 step 在抓取期间没有变化。

相机不能在每一步通过 `set_world_poses()` 手工覆盖：默认 IsaacLab 运行使用 Fabric 更新
机器人，而 Camera 的公开 world-pose 写入路径可能落到 USD，二者会产生不同的渲染位姿。
当前实现采用 IsaacLab 官方机器人挂载相机的方式，把 Camera 作为刚体 link 的固定子 prim。

发送给模型的数据必须通过仿真主线程中的 `capture()` 获取，不能直接从网络线程读取普通
per-step sensor buffer。

## 6. FrameBundle 数据契约

`FrameBundle` 表示一次完整的四方向同步快照：

```python
FrameBundle(
    bundle_id: int,
    env_index: int,
    sim_step: int,
    timestamp: float,
    T_world_base: np.ndarray,       # (4, 4), float
    views: dict[str, CameraFrame],  # forward/left/behind/right
)
```

每个 `CameraFrame` 包含：

```python
CameraFrame(
    camera_id: str,
    direction: str,
    sensor_frame_id: int,
    sim_step: int,
    timestamp: float,
    rgb: np.ndarray,                # (H, W, 3), uint8, RGB
    depth_z_m: np.ndarray,          # (H, W), float32, meter
    K: np.ndarray,                  # (3, 3)
    T_world_camera_ros: np.ndarray, # (4, 4)
    T_base_camera: np.ndarray,      # (4, 4)
)
```

四个方向通过字典的语义 key 访问，不应该依靠 list 下标猜测方向。当前本机顺序是：

```text
forward, left, behind, right
```

如果远程 LaViRA wrapper 要求其他顺序，例如 Go1 wrapper 的
`forward, right, behind, left`，应该在未来的服务器适配器序列化阶段显式重排，不能修改
本机相机方向的语义。

数组已经从 IsaacLab tensor/GPU buffer 复制出来，后续网络线程可以安全读取它们；网络线程
不能直接操作 Isaac Sim sensor，也不能调用 `render()`。

## 7. 坐标系和变换含义

对外输出的相机坐标采用 ROS optical convention：

```text
+X：图像右方
+Y：图像下方
+Z：相机光轴前方
```

机器人导航方向在当前 base/torso 坐标中的语义为：

```text
forward = +X
left    = +Y
up      = +Z
```

四元数顺序统一为 `wxyz`。变换矩阵采用：

```text
T_A_B：把 B 坐标系中的齐次点转换到 A 坐标系
```

因此：

```text
p_world = T_world_camera_ros @ p_camera
p_base  = T_base_camera      @ p_camera
```

深度类型是 IsaacLab 的 `distance_to_image_plane`，即 optical Z depth，单位是米。因此将
像素 `(u, v)` 和深度 `z` 反投影到相机坐标时可以使用：

```text
p_camera = z * inverse(K) @ [u, v, 1]^T
```

之后再用 `T_world_camera_ros` 或 `T_base_camera` 转换到需要的坐标系。未来处理模型 bbox
时，建议在 bbox 内取一小块有效深度的中位数，而不是只读取中心的一个像素，以降低边缘和
空洞噪声。

## 8. 调试运行方法

在项目根目录执行一次 headless 四视图检查：

```bash
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch --house --headless --device cuda:0 \
  --four_rgbd_cameras --camera_debug_save_once \
  --no-show_path --max_steps 12 --print_every 5
```

如果机器需要显式指定 NVIDIA Vulkan ICD，可以使用：

```bash
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
python scripts/isaacsim_goal_tracking/isaacsim_path_follwing.py \
  --mode switch --house --headless --device cuda:0 \
  --four_rgbd_cameras --camera_debug_save_once \
  --no-show_path --max_steps 12 --print_every 5
```

`--camera_debug_save_once` 会在默认 warm-up 5 个 step 后保存一个 bundle。输出目录为：

```text
outputs/isaacsim_goal_tracking/camera_probe/
└── run_<timestamp>/
    └── bundle_<id>_step_<step>/
        ├── forward_rgb.png
        ├── left_rgb.png
        ├── behind_rgb.png
        ├── right_rgb.png
        ├── <direction>_depth.npy
        ├── <direction>_depth_preview.png
        ├── montage.png
        └── metadata.json
```

其中 `.npy` 是保留精确米制数值的原始深度；`depth_preview.png` 只用于人眼观察，不能拿来
做几何反投影。`metadata.json` 中保存了内参、外参、机器人位姿、深度统计和坐标约定。

为了防止路径可视化标记进入模型 RGB，验证模型输入时建议使用 `--no-show_path`。

## 9. GUI 中查看相机

去掉 `--headless` 即可运行 GUI。`--four_rgbd_set_viewport` 默认开启，代码会尝试把主
viewport 自动切换到 `env_0` 的 `lavira_camera_forward`。

如果手动查看，可以在 Stage 树中展开：

```text
World/envs/env_0/Robot/torso_link/
```

找到对应的 `lavira_camera_*` prim 后，将 viewport camera 切换到该 prim。GUI 的
`Perspective / Top / Front / Right` 是编辑器通用视角，不是这四台正式 RGB-D sensor。

## 10. 与 LaViRA 及服务器接口的关系

目前已与 LaViRA 侧对齐的部分：

- 四个语义视角
- `640 x 480` 分辨率
- `79°` 水平 FOV
- `0.1–5.0 m` 深度范围
- RGB 数组、米制 depth、内参和拍摄时外参
- 同一个仿真 step 的四视图 bundle

由于平台从 Habitat/Go1 换成 Isaac Sim/G1，物理安装位置和 G1 的身体轴定义不会照抄原
平台，而是采用本项目相机刚性跟随后、结合机器人实际正向运动完成的标定结果。

后续远程模型接口建议保持以下边界：

```text
Isaac Sim 主线程
  -> camera_rig.capture()
  -> 得到已复制的 FrameBundle
  -> 放入有界请求队列

网络线程
  -> 编码四路 RGB
  -> 添加 instruction / history / request_id
  -> 请求远程 LaViRA 模型
  -> 异步返回 direction / bbox / stop / backtrack

Isaac Sim 主线程
  -> 按 request_id 和时间戳拒绝过期结果
  -> bbox + depth + K + 外参得到局部/世界目标点
  -> 后续 occupancy map / FMM / waypoint follower
  -> vx、vy、wz
  -> 现有 locomotion policy
```

网络发送时通常只需要编码 RGB 和结构化元数据；完整 depth 可以保留在本机，并通过
`request_id`/`bundle_id` 找回与模型 bbox 严格对应的那一帧。是否发送 depth 应由远程模型
的真实输入接口决定。

## 11. 当前尚未实现的部分

以下内容不属于本次相机代码，仍需要后续单独实现：

- 远程服务器 API client
- instruction 和 navigation history 的协议
- LaViRA 输入图片的编码、顺序和 prompt 适配
- 模型输出 JSON schema
- bbox 与对应 FrameBundle 的关联缓存
- bbox + depth 的目标点计算
- occupancy map 和 FMM/iPlanner
- VLN 目标点与现有 waypoint follower、速度命令及 locomotion policy 的状态机
- 网络超时、过期响应、重试和安全停车机制

这些模块应消费 `FrameBundle`，不应再次修改或复制一套相机读取逻辑。

## 12. 已验证结果

当前四方向修正后已在 Isaac Sim 中实际运行：

- 四路 RGB shape：`(480, 640, 3)`，dtype：`uint8`
- 四路 depth shape：`(480, 640)`，dtype：`float32`
- 本次场景四路有效深度比例：`1.0`
- 默认 Fabric 模式下，四台相机会随 `torso_link` 的平移和旋转持续运动
- `forward` 已确认与机器人实际正向行走方向一致
- `left / behind / right` 已按照同一机体系在 GUI 中人工确认方向正确
- 相机不会再停留在初始世界位置，也不需要使用 `--disable_fabric`
- 四方向光心位置、光轴、俯角和坐标组合的 4 个单元测试通过

最终标定约定为：

```text
forward = torso_link +X
left    = torso_link +Y
behind  = torso_link -X
right   = torso_link -Y
```

此前曾经通过每步调用 `set_world_poses()` 手工更新 Camera 世界位姿。该方法会让默认
Fabric 中运动的机器人与 USD Camera transform 产生分离，表现为机器人走动而相机停在
原地。现在四台 Camera 都只保存相对 `torso_link` 的固定局部外参，由标准父子层级跟随，
旧的 world-pose 同步代码已经删除。`--disable_fabric` 只曾用于定位问题，不是当前相机运行
所需参数。

运行单元测试：

```bash
python3 -m unittest discover \
  -s scripts/isaacsim_goal_tracking/tests -v
```
