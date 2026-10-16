# D415 → G1 胸部外参标定

目标是求固定的 `T_torso_camera`：将 D415 **RGB 光学坐标系**中的米制点变换到 G1 `torso_link`。骨盆是 `pelvis`；运行时用腰部实际关节角计算 `T_pelvis_torso(q)`，不能把胸部到骨盆当成固定外参。

这是独立的只读工具，不发送运动命令，不改变相机设置，不安装结果到 VLN。沿用现有官方 RealSense SDK/TCP 取图接口和 Unitree Python SDK。URDF 正运动学使用本文件夹的小型 NumPy/XML 实现，只支持当前需要的固定/旋转关节链，未知关节或缺失角度会报错；不需要新装 Pinocchio 或 ROS。

**本次实际完成的是编码硬板＋`live_touch.py` 每次触碰重新读板位姿的流程。**下面的普通棋盘、局部手点棋盘和固定参考图命令保留为旧流程，不是本次最终结果的数据来源。另一个双相机方案在独立目录 [camera_pair_extrinsics](../camera_pair_extrinsics/README.md)，不混用采集记录或求解输入。

## 本次手触碰实验的理论、过程和结果

同一个板上接触点分别拥有相机坐标 `p_camera` 和胸部坐标 `p_torso`。左手中指接触指定方块的左上外角时，假设指尖与这个外角重合，就能建立两套坐标中的点对应。

- 相机侧：RGB 识别 AprilTag 编号，结合 SDK 内参、72mm中心间距和板布局做 PnP，求板 → 相机位姿；再按55mm外方块尺寸计算接触外角的相机坐标。深度图保存为记录，不参与这次求解。
- 机器人侧：订阅实际关节角，根据 URDF 做正运动学，计算 `left_wrist_yaw_link` → `torso_link`；再加上手腕坐标中的固定指尖偏移，得到接触点胸部坐标。不是逆运动学。
- 求解：在多个不同的对应点上最小化 `Σ ||p_torso - (R × p_camera + t)||²`，`kinematics.py` 用 SVD 求刚体旋转和平移，不拟合尺度。最终 `T_torso_camera` 描述安装关系，不是机器人在地图中的位姿。

相机 RGB 光学坐标为 X右/Y下/Z前；胸部使用 X前/Y左/Z上。下面安装角把相机也转换成前/左/上表示，不能与 RGB 光学坐标本身的 RPY 混用。

实际过程（2026-10-09）：

1. 更换为6×6编号硬板，测得大方块55mm、横纵间隔17mm。全部36个ID在不同RGB帧中核对通过；使用 AprilTag 36h11、两格宽编码边框。
2. 第一组 `d415_live_extrinsics_01` 保存9点，但第4点 ID19 的照片显示仍碰在前一个 ID13 的外角，关节状态近乎相同，用户也报告误连按 Enter。原始记录与 `sampling_review.json` 保留，这组没有作为最终外参候选。
3. 第二组 `d415_live_extrinsics_02` 重新采9点：ID14、7、13、19、8、20拟合，ID9、15、21保留检查；每次触碰都用当前图像求板位姿。点与点之间允许胸部相对板改变位置，单次采集保持静止，相机支架相对胸部固定。
4. 按图像窗口 Space → Enter 确认当前接触点，保存图像、关节角、板位姿和时间配对诊断。采集期间检查目标位置波动≤5mm、相关关节角变化≤0.5°；与关节按笔记本接收时间配对，利用静态条件，不声称硬件同步。
5. 第二组6点拟合、3点保留验证。随后第三组 `d415_live_extrinsics_03` 重新采9个接触点，保持02的外参不变验证全部新点，不重新拟合，不为降低误差删除ID20。

| 项目 | 结果 |
|---|---:|
| 02拟合点数 / 保留点数 | 6 / 3 |
| 02拟合 RMS | 6.086 mm |
| 02保留点 RMS / 最大 | 6.758 / 8.599 mm |
| 03新接触验证点数 | 9 |
| 03固定外参验证 RMS / 最大 | 10.908 / 21.871 mm |

02候选的 RGB 光心位置和前/左/上安装角：

| 参数 | 手触碰候选 |
|---|---:|
| torso X，向前 | 7.316 cm |
| torso Y，向左 | 4.552 cm |
| torso Z，向上 | 30.547 cm |
| 安装 Roll | +2.674° |
| 安装 Pitch | −4.741° |
| 安装 Yaw | −1.705° |

候选：`/home/yile/projects/g1_camera_sdk/d415_live_extrinsics_02/extrinsics_candidate.json`；新接触验证报告：`/home/yile/projects/g1_camera_sdk/d415_live_extrinsics_03/validation_report.json`。逐点照片、米制深度和关节数据在各自采集目录中。

本次左手中指偏移使用 mesh 模型候选 `[0.17332935,-0.02819001,0.00668351]` 米（相对于左手腕），保持 `tip_measured=false`。误差是视觉接触点与模型正运动学指尖的一致性，不是外部测量真值；指尖物理偏移、手套厚度、实际接触位置仍需核实。候选未自动应用 VLN。

### 与双相机＋URDF估计的比较

| 参数 | 手触碰 | 双相机＋URDF假设 |
|---|---:|---:|
| X，向前 | 7.316 cm | 10.973 cm |
| Y，向左 | 4.552 cm | 2.133 cm |
| Z，向上 | 30.547 cm | 29.477 cm |
| 安装 Roll | +2.674° | −0.286° |
| 安装 Pitch | −4.741° | −10.828° |
| 安装 Yaw | −1.705° | +1.014° |

两者位置距离 **45.13mm**，相对旋转角 **7.14°**；Roll/Pitch/Yaw分量差约 −2.960°/−6.087°/+2.719°。相对旋转角由旋转矩阵计算，不是三个欧拉角差直接相加。

比较要求两次D415安装未改变。双相机的板中心一致性RMS 0.798mm与手触碰的10.908mm采用不同参考，不能直接用这两个数判断哪个胸部外参更准：前者没有验证原装RGB→胸部的参考，后者依赖未实测指尖模型。双相机过程、独立验证、URDF假设和完整比较文件见[另一方案 README](../camera_pair_extrinsics/README.md#实际采集与求解结果)。

## 文件

| 文件 | 用途 |
|---|---|
| `calibrate.py` | 设备检查、关节监控、棋盘识别、接触点采集、外参求解入口 |
| `collect_points.py` | 在终端依次提示局部3×3触碰点，失败保留原因并支持重试，已保存点可续采 |
| `robot_state.py` | 只订阅 `rt/lowstate`，读取 29 个关节的实际角度/速度、模式和 tick |
| `kinematics.py` | URDF 正运动学、坐标转换、SVD 刚体拟合（不拟合尺度） |
| `inspect_hand.py` | 读取左手 mesh，导出模型最前端候选点和三个投影视图；不会将候选点标为实测值 |
| `partial_board.py` | 棋盘不完整可见时，手动指定局部角点并求棋盘位姿 |
| `verify_marker_layout.py` | 新编码板实时解码及候选编号布局核对；每个ID至少两张不同RGB帧通过检查 |
| `coded_board.py` | 用编码区域中心和72mm间距求当前板位姿，推算棕色大方块外角的位置 |
| `live_touch.py` | 每次触碰重新采当前板位姿，检查视觉/关节稳定性并保存配对记录 |
| `validate_candidate.py` | 用另一组新触碰记录验证已有外参，报告逐点三维误差，不重新拟合或自动应用 |
| `reprocess_saved.py` | 从已保存的代表RGB重新检测板位姿，保留关节角和fit/check分组，另存新session供离线求解 |
| `view_waist.py` / `waist_viewer.html` | 从当前 URDF 和 STL 导出可旋转的三维示意图，显示骨盆、躯干原点及腰部三轴；滑块只改变页面模型 |
| `board_marker_measurements.json` | 新板实测：标记外边55mm、横纵间隔17mm，中心间距72mm |
| `board_6x9_100mm.json` | 用户给定的 6×9 格、100mm 格子尺寸记录 |
| `probe.example.json` | 待填写的探针偏移和安装确认；未测量时不能创建标定 session |
| `test_calibration.py` | 离线几何、棋盘和求解验证 |

## 方案与准备

相机固定在胸部刚性结构上；平整棋盘固定好、测量格子边长。手上安装刚性探针，测量尖端相对于选定 URDF `left_wrist_yaw_link` 或 `right_wrist_yaw_link` 的三维偏移。偏移必须按该 link 的坐标轴表示，不能直接填写“从手掌表面伸出多少厘米”。可动手指上的探针会改变偏移，应固定到相对手腕不动的结构。

相机识别棋盘，按已知格子尺寸及 SDK RGB 内参求棋盘位姿；每个角点因此有相机坐标。机器人探针触碰该角点，记录实际关节角，通过模型计算同一点的胸部坐标。多组对应点用来拟合固定的旋转和平移。

`calibrate.py board/new-session/touch` 的旧路径采用“**先拍棋盘参考图，再触碰**”的工作方式。这样手遮挡棋盘时也能采集。代价是从拍摄参考图到全部触碰结束，**相机与棋盘的相对位置必须保持不变**。机器人躯干晃动或腰部运动都可能破坏参考；稳定关节检测不能证明相机与棋盘未移动，`--reference-fixed` 是操作者对这一条件的确认，不是程序测量结果。发生移动后需新建参考和 session，不能沿用旧点。本次实际实验使用后文 `live_touch.py` 当前图像路径。

使用机器人支持的可拖动状态采集，不要强掰正在保持位置的关节。程序不会切换运控模式。

## 1. 启动相机

相机目前接笔记本 USB，D415 序列号 `211222063932`。终端 1：

```bash
/home/yile/projects/g1_camera_sdk/local/build-probe-x86_64/g1_d435i_stream \
  --bind 127.0.0.1 --port 8765 --serial 211222063932 \
  --settings-dir /home/yile/projects/g1_camera_sdk/d415_camera_logs
```

看到 `RGB-D ready` 后保持运行。若服务已经运行，不要重复启动。相机服务一次处理一个客户端，退出其他预览/采集客户端后再使用本工具。

终端 2，之后的命令均在这里运行：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source /home/yile/projects/.venvs/unitree_g1/bin/activate
```

## 2. 只读检查

```bash
python scripts/camera_extrinsics/calibrate.py check \
  --network-interface enxf8e43bbea486 \
  --output /home/yile/projects/g1_camera_sdk/extrinsics_check_02
```

输出目录须不存在。保存 RGB、米制深度和 `check.json`。检查独立报告相机/关节是否可用；ping 通不代表收到 DDS。

持续看腰部角度和手腕原点（不是探针尖端）：

```bash
python scripts/camera_extrinsics/calibrate.py watch \
  --network-interface enxf8e43bbea486 --seconds 60
```

SDK 电机索引使用官方 G1 29DoF 映射，不使用 URDF XML 出现顺序。按 `mode_machine` 选择本地官方 URDF，拒绝未知版本及非 PR 模式。本工具支持未锁腰的 29DoF 版本 2、5、11、13、15；安装探针前还应核实实际机器人/手部与模型一致。

## 3. 拍棋盘参考图

`cols/rows` 是内部黑白交点数，不是格子数。必须填写实际值；不从纸张外形猜测。把整个内部角点区域放入画面，去掉遮挡，棋盘保持平整。

用户已确认：6×9 个格子、格子边长 100mm。按横向6格、纵向9格摆放，对应 `cols=5, rows=8` 内角点。若旋转90°摆放，行列应对调。完整检测可以执行：

```bash
COLS=5
ROWS=8
SQUARE_MM=100
python scripts/camera_extrinsics/calibrate.py board \
  --cols "$COLS" --rows "$ROWS" --square-mm "$SQUARE_MM" \
  --output /home/yile/projects/g1_camera_sdk/board_reference_01
xdg-open /home/yile/projects/g1_camera_sdk/board_reference_01/corners_numbered.png
```

输出 `rgb.png`、`corners_numbered.png`、`board.json`。图中标签是从 0 开始的 `row,col`；之后按这张图中的标签触碰。普通棋盘可能出现角点排序翻转，不凭记忆认定某个物理角永远是原点。程序使用 SDK 内参，不用深度值求棋盘位置；检测/重投影通过不意味着机器人外参已标定。

### 旧普通棋盘流程：只看得到局部棋盘

边缘破损、机器人需要靠近棋盘时，不要求完整棋盘检测。`partial_board.py` 使用已知格子尺寸、人工明确指定的内角点编号和像素位置做 PnP。至少 4 个分散、构成二维区域的角点，建议 6–9 个；不选被手遮挡、撕裂、弯折的点。其余可推算点仍依赖整块板平整、格子规则的假设。

编号有两种有效方式：使用整块棋盘的固定编号；或者将一块连续可见的局部网格定义为独立参考网格。对当前固定参考图的触碰标定，推荐后一种：选连续3列、3行真实内角点，将选区左上角记作 `(0,0)`，向右列号增加、向下行号增加，设置 `--cols 3 --rows 3`。每两个相邻点仍相距100mm。这里的原点只是局部几何参考，不要求它是整块棋盘的左上角。必须看标号图确认触碰同一批物理角点；换参考图或选区时不可混用旧编号。

当前操作使用连续局部3×3个内角点：

```bash
python scripts/camera_extrinsics/partial_board.py \
  --cols 3 --rows 3 --square-mm 100 \
  --ids 0,0 0,1 0,2 1,0 1,1 1,2 2,0 2,1 2,2 \
  --output /home/yile/projects/g1_camera_sdk/board_partial_01
```

1. 先出现实时图像窗口。调整棋盘，使所选局部角点清晰、分散。点击图像窗口，让它获得键盘焦点，然后按 **空格**冻结图像。
2. 按窗口上方显示的 `row,col` 顺序，逐个点击真实内角点。程序在点击附近做亚像素细化；右键撤销最后一点。
3. 全部点击后，**在图像窗口按 Enter**确认。`Q`/Esc 取消。
4. 输出 `board.json`、原图和标号图。后续 `new-session --board-reference` 使用这个 `board_partial_01/board.json`，其余触碰/求解步骤相同。

超过1像素时保留原图和选点，标记 `REVIEW_REPROJECTION` 待检查，不会因略过阈值丢掉整次操作。即使位姿求解失败，也保存原图、`selected_pixels.json`、`selected_ids.json` 和元数据，便于分析/重选；保存成功不等于参考位姿已经验证。低重投影误差不能证明编号正确，也不能补救不平整的棋盘。刚刚拍到的局部棋盘位姿仍要求相机与棋盘相对位置保持不变。可以先拍、再伸手触碰，不必触碰时仍看到全部选定点。如果身体或相机动了，需要重新获取参考。

离线分析同一张已保存检查图时可添加 `--from-check 检查目录`。脚本不会用新画面配旧像素。没有做本次真实点击/接触时，不会声称得到真实外参。

### 固定硬手指：从已有模型提取接触点

如果左手指相对手腕固定，可把指尖当作刚性接触点，不需要额外手指角度。本地模型包含固定连接的 `left_rubber_hand` 和其 STL。可先读取几何点，结合 URDF 的固定连接换算到手腕：

```bash
python scripts/camera_extrinsics/inspect_hand.py \
  --output /home/yile/projects/g1_camera_sdk/hand_geometry_02
```

输出 `hand_model_views.png`、`model_contact_candidate.json` 和 `probe.mesh_candidate.json`。当前程序自动选择模型表面 X 最大顶点作为候选，图中红点表示这个点；这是几何选择，并没有由实物确认其对应的手指或接触面。只有确认实际手形/安装/尺寸一致、触碰确实使用同一物理点并核对手套厚度后，才能将偏移用于真实标定。

2026-10-08 提取的候选点在 `left_wrist_yaw_link` 中约为 `[0.17332935,-0.02819001,0.00668351]` 米；URDF 固定手腕到手部原点平移为 `[0.0415,0.003,0]` 米。候选图和记录保存在 `/home/yile/projects/g1_camera_sdk/hand_geometry_01`。候选配置保持 `tip_measured=false`，不是已经求出的相机外参。需要试用模型指尖时，用 `tip_source=model` 并记录 `model_candidate_path`，创建 session 时显式增加 `--allow-model-tip`。这会保留模型来源和未实测状态，结果仍只是候选；不能把模型点改标为人工实测。当前试用配置保存在 `hand_geometry_01/probe.left_middle_model_trial.json`。

## 4. 配置探针并创建 session

复制 `probe.example.json` 为自己的配置，例如 `probe.measured.json`：

- `probe_link`：实际使用的手腕坐标系。
- `tip_in_link_m`：实测的 `[x,y,z]`，米。
- `tip_measured`：测量完成后设为 `true`。
- `mount_on_torso_confirmed`：确认相机随胸部刚性结构固定后设为 `true`。

不要把未知偏移写成 `[0,0,0]`。示例中没有假设好的外参或探针位置。

```bash
python scripts/camera_extrinsics/calibrate.py new-session \
  --board-reference /home/yile/projects/g1_camera_sdk/board_reference_01/board.json \
  --probe-config scripts/camera_extrinsics/probe.measured.json \
  --reference-fixed \
  --output /home/yile/projects/g1_camera_sdk/extrinsics_session_01
```

## 5. 逐点触碰

当前局部3×3试标定可用同一个终端依次采集：

```bash
python scripts/camera_extrinsics/collect_points.py \
  --session /home/yile/projects/g1_camera_sdk/extrinsics_session_01 \
  --network-interface enxf8e43bbea486 --reference-fixed
```

按提示用左手中指上的同一固定接触点轻触对应交点，停稳后在该终端按 Enter，继续保持约1秒。显示保存成功后再移动到下一点。前两行6点用于拟合，最后一行3点保留验证；失败可重试当前点，Ctrl+C取消，已保存点可以续采。相机、躯干与棋盘的相对位置必须保持不变。当前指尖是模型候选偏移，尚未实测。

静止检查依据这一秒内手臂/腰部实际关节角的最大变化（不超过0.5°），并检查数据连续更新和模式一致。上报关节速度峰值保留作诊断，不单凭一个速度尖峰拒绝采样。失败原因保存到 `last_contact_error.json`。

以下仅示范参考图上的 `(row=0,col=0)`。每次换一个指定角点，重复执行并改行列编号：

```bash
python scripts/camera_extrinsics/calibrate.py touch \
  --session /home/yile/projects/g1_camera_sdk/extrinsics_session_01 \
  --row 0 --col 0 --role fit --reference-fixed
```

先移动探针轻触指定角点，保持接触和身体稳定，**在终端 2 按 Enter**。程序随后收集约 1 秒的关节数据，检查更新、新鲜度、手臂/腰部静止程度，再保存 `session.json`。手腕原点经实测探针偏移转换为尖端坐标。

建议用 8–12 个分散的角点拟合，每点重复采集。至少保留另 3 个不同角点不参与拟合，采集时使用 `--role check`。避免所有点集中在局部或一条直线上。单张平面棋盘的非共线点可以求刚体变换，但还应在其他位置/距离验证泛化误差。

## 6. 求解与验证

```bash
python scripts/camera_extrinsics/calibrate.py solve \
  --session /home/yile/projects/g1_camera_sdk/extrinsics_session_01
```

至少需要 6 个不同拟合角点和 3 个独立保留角点。保存 `extrinsics_candidate.json`，包含 `T_torso_camera`、逆变换、拟合/保留点误差、URDF 哈希和相机元数据。单位是米，使用列向量：

```text
p_torso = T_torso_camera × p_camera
p_pelvis = T_pelvis_torso(q) × p_torso
```

结果标记为候选值，不因为拟合成功就自动认为适合导航。需要检查独立点误差、重复标定的一致性，以及改变腰部姿态后的坐标变换。接入 VLN 时还需确认 SLAM 位姿对应哪个机体坐标系，并处理图像/关节/SLAM 的时间配对；本工具利用静态采集条件，不声称已经解决动态时间同步。

## 2026-10-08 实测状态

- 机器人 `192.168.123.164` ping 通；初次 DDS 等待未收到，随后实际关节数据已收到。
- `mode_machine=5`、`mode_pr=0`，选择 `g1_29dof_rev_1_0.urdf`。
- D415 `211222063932` RGB-D 640×480 成功接收，2 秒取图测试约 14.9 fps。
- 左右手腕原点的胸部坐标成功计算，持续读取入口成功运行；尚未验证探针接触位置的物理精度。
- 检查样本：`/home/yile/projects/g1_camera_sdk/extrinsics_check_20261008_01`。
- 初次检查图里棋盘只在画面底部露出一部分，尚不满足完整棋盘检测；新增手动局部角点路径。
- 棋盘尺寸已确认6×9格、100mm边长；用户确认相机胸前固定、左手指固定，模型指尖偏移尚未实测。
- 当前参考 `board_partial_05/board.json` 使用局部3×3角点，重投影 RMS 1.015像素，标记待检查；之前身体移动前的参考不再使用。当前 session 为 `extrinsics_session_01`，尚未得到可用于 VLN 的外参。
- 首次触碰窗口退出后没有保存样本，具体原因未保留。后续诊断中关节角1秒变化约0.0055°，但上报速度峰值0.0566rad/s超过旧阈值；已改为按角度变化判断静止并保留速度诊断。
- 13 项离线测试覆盖 FK、刚体拟合、局部棋盘位姿、模型点来源、独立验证点求解、静止角度与速度尖峰，以及触碰顺序续采。

## 2026-10-09 新编码板：先核对布局

当前新板是硬板，已测标记外边长55mm、横纵白色间隔17mm。检测结果支持 AprilTag 36h11；完整画面多尺度检测到25个ID，其位置符合候选6×6排列 `id=6*列+行`。未解码的ID不能仅凭规律标成确认。旧100mm棋盘参数不适用于新板。

随后实时核对完成：0–35全部36个ID均在至少两张不同RGB帧中实际解码，并通过图像上的布局一致性检查。证据及确认结果在 `/home/yile/projects/g1_camera_sdk/marker_layout_verify_01/board_layout_confirmed.json`；尺寸记录中已保存确认布局和证据路径。17项离线测试通过。该结果确认编号位置，不代表米制位姿、指尖位置或安装外参已准确标定。

```bash
python scripts/camera_extrinsics/verify_marker_layout.py \
  --candidate-layout /home/yile/projects/g1_camera_sdk/board_full_check_20261009_150757/layout_review.json \
  --output /home/yile/projects/g1_camera_sdk/marker_layout_verify_01
```

图像窗口绿色为已经在至少两张不同RGB帧中解码、且与候选布局符合的ID；黄色为当前读到但尚未满足该条件的ID。窗口和终端列出全部待确认ID。不需按Enter，调整板或显示区域即可；Q/Esc/Ctrl+C退出，进度和证据图保存到输出目录。没有观察到的ID不会自动补确认。图像放大仅用于辅助检测，不提升相机的实际采样分辨率。

只做图像上的编号布局核对，不调用机器人SDK或运控。全部通过后输出 `board_layout_confirmed.json`，尚不代表相机外参或米制板位姿准确。标记方向与板上角点对应仍需验证。

### 当前编码板：每次触碰重新读位姿

胸部相机在触碰时轻微移动，使用新入口 `live_touch.py`，不用旧 `calibrate.py touch` 固定参考图方式。两个接触点之间允许胸部相对板移动；采每个点时需要停稳，相机支架必须始终相对胸部固定。

该板使用两格宽深色编码边框，检测参数 `markerBorderBits=2`。最初使用默认一格边框时，检测框落在棕色外框内侧且近距离漏读；修正后同一诊断图读到13个标记，中心重投影RMS约0.63像素。程序根据检测框对角线交点求编码中心，以72mm中心间距求板位姿；假设编码中心与棕色大方块中心重合。用实际55mm外边长定义左上外角接触点。该几何假设和实物尺寸仍需独立检查。

```bash
python scripts/camera_extrinsics/live_touch.py \
  --session /home/yile/projects/g1_camera_sdk/d415_live_extrinsics_01 \
  --probe-config /home/yile/projects/g1_camera_sdk/hand_geometry_01/probe.left_middle_model_trial.json \
  --allow-model-tip --network-interface enxf8e43bbea486
```

窗口实时显示目标外角红圈。用同一左手中指接触点触碰红圈对应物理外角，停稳，在**图像窗口先按空格准备，再按Enter采集**，继续保持约1.2秒。程序终端也可输入 `r` 并回车准备，再单独回车采集。每个新点以及每次失败重试都要重新准备；单纯连按Enter不会确认下一个点。程序每次采至少5个当前板位姿，至少6个分散编码；检查重投影误差、目标位置波动≤5mm、手臂/腰部关节角变化≤0.5°和数据更新。板朝向估计波动保留诊断，不在接触点三维位置稳定时单凭略超1°拒绝。失败不保存，调整并重试；成功自动提示下一点。无法开始采样时显示具体原因，保存 `rejected_preview.png` 和错误JSON，不把所有错误笼统说成目标出画面。软件无法证明真实接触点准确，Enter代表操作者确认。Q/Esc/Ctrl+C退出，保存点可续采。

采样时用RANSAC剔除不符合布局的标记，再用至少6个跨行列的有效标记求位姿；编号核对阶段仍严格检查原始布局。偶发无效帧记录到 `last_window_rejections.json`，不因单帧异常立即丢弃整个窗口。有效位姿至少覆盖0.8秒，首尾与中间空缺均不得超过350ms；缺测过久仍拒绝保存。记录保留被剔除的标记编号和无效帧诊断。

相机与关节按**笔记本接收单调时间**最近匹配，间隔≤50ms；保存原始设备时间、接收时间和1.2秒稳定窗口。利用静态接触条件降低传输延迟影响，不能当作已解决曝光时刻硬件同步，不用于连续移动中采样。

顺序：ID14、7、13、19、8、20拟合，ID9、15、21保留检查。均触碰棕色大方块左上外角，旧3×3手点棋盘参考不参与。目标被遮挡时可由其他编码推算；红圈必须在画面内且板位姿有效才允许采样。保存各接触RGB、米制深度、当前板位姿、关节状态、模型来源及配对诊断。左手指尖仍是模型候选，未实测。

采完9点后用 `calibrate.py solve --session /home/yile/projects/g1_camera_sdk/d415_live_extrinsics_01` 离线拟合。结果只标记为待验证候选，不自动应用VLN。

### 固定已有外参，用新接触点验证

2026-10-09 的第二组 `d415_live_extrinsics_02` 已采齐9点；6个拟合点 RMS 6.1mm，3个保留点 RMS 6.8mm、最大8.6mm。这说明该组内部一致性较好，不代表已经独立确认真实安装误差。左手指尖偏移仍由模型给定。

同日第三组 `d415_live_extrinsics_03` 用02外参保持不变验证9个新接触点：RMS 10.9mm，最大21.9mm（ID20）。其余8点误差4.2–13.4mm。ID20采样期间目标位置波动1.83mm、关节角变化0.084°，图像没有显示像第一组那样明确的重复触碰；其残差按“外参预测减FK指尖”定义，XYZ约 `[-6.7,-4.4,-20.3]` mm，主要在竖直方向。保持全部9点，不为降低误差删除ID20；仍需重复接触检查、核对指尖实物偏移及其他板姿态。照片对照为03目录的 `validation_contact_review.jpg`，完整结果为 `validation_report.json`。当前不自动应用VLN。

保持相机与胸部固定，将板稍微平移或倾斜，确保手能触碰且仍能检测至少6个跨行列标记。另采一组新点，每次空格准备、Enter采集；不要移动相机支架。下面的03目录必须是新的，已有其他采样时换新名称：

```bash
python scripts/camera_extrinsics/live_touch.py \
  --session /home/yile/projects/g1_camera_sdk/d415_live_extrinsics_03 \
  --probe-config /home/yile/projects/g1_camera_sdk/hand_geometry_01/probe.left_middle_model_trial.json \
  --allow-model-tip --network-interface enxf8e43bbea486

python scripts/camera_extrinsics/validate_candidate.py \
  --candidate /home/yile/projects/g1_camera_sdk/d415_live_extrinsics_02/extrinsics_candidate.json \
  --session /home/yile/projects/g1_camera_sdk/d415_live_extrinsics_03
```

验证入口保留02的旋转和平移不变，将03全部新点作为验证点，即使采样记录角色标签为fit。比较“相机点经已有外参换算的胸部坐标”和“新关节角经正运动学得到的指尖胸部坐标”，输出逐点误差、RMS、最大误差与XYZ平均残差到03的 `validation_report.json`。拒绝原采样曝光重用、相机序列号/模型/接触点/板配置不一致；不重新拟合、不自动批准或安装到VLN。不要对03运行solve来代替这个验证。多个板位置/距离都应验证；同一模型指尖、板尺寸可能带来共同系统误差，仍需核对实物指尖偏移及手套覆盖。

## 2026-10-13 从原图重算：修正中心筛选和缩放换算

已修正 `verify_marker_layout.py`：布局检查和多尺度一致性检查统一使用对角线交点中心；`coded_board.py` 导入同一中心函数，避免两种定义不一致。放大图角点回到原图改为 `(corner + 0.5) / scale - 0.5`。板原点、55mm方块、17mm间隔、指尖模型及URDF均未改变。

实际重新处理02、03两组共18张已保存接触RGB，没有调用机器人或相机。保留02的6拟合/3检查分组，03全部9点独立验证。用同一URDF和原关节角重新计算FK，并核对与原 `p_torso_m` 一致；新相机点来自原RGB的重新检测。全部点保留，原始记录、原图和旧候选没有覆盖。

新目录：

- 拟合：`/home/yile/projects/g1_camera_sdk/d415_live_extrinsics_02_reprocessed_20261013/`。
- 独立验证：`/home/yile/projects/g1_camera_sdk/d415_live_extrinsics_03_reprocessed_20261013/`。
- 新候选位于拟合目录的 `extrinsics_candidate.json`；独立报告位于验证目录的 `validation_report.json`。

| 参数 | 原候选 | 重算候选 |
|---|---:|---:|
| X，向前 | 7.316 cm | 7.034 cm |
| Y，向左 | 4.552 cm | 4.941 cm |
| Z，向上 | 30.547 cm | 31.141 cm |
| 前/左/上安装 Roll | +2.674° | +2.435° |
| 安装 Pitch | −4.741° | −3.908° |
| 安装 Yaw | −1.705° | −2.302° |
| 6点拟合 RMS | 6.086 mm | 6.026 mm |
| 3点保留检查 RMS | 6.758 mm | 8.092 mm |
| 03新9点验证 RMS | 10.908 mm | 11.286 mm |
| 03新9点验证最大 | 21.871 mm | 22.430 mm |

外参相对原候选的位置变化7.639mm，旋转变化1.062°。修正有可见影响，但没有显著降低独立验证误差；不是根据结果挑选更小误差版本。新指尖偏移仍未实测，候选未应用VLN。与同日双相机＋URDF假设重算结果的位置差51.244mm、旋转差8.011°，原来的较大分歧仍在。

`reprocess_report.json` 保存逐点变化和源图哈希；源session哈希、代码哈希、旧位姿和旧相机点保留。原始静止窗口的所有RGB并未逐张保存，所以只重算保存的代表图；原窗口诊断以 `_original` 字段保留，不冒充重新通过采集静止检查。

复现时输出必须换成不存在的新目录：

```bash
python scripts/camera_extrinsics/reprocess_saved.py \
  --source-session /home/yile/projects/g1_camera_sdk/d415_live_extrinsics_02 \
  --output /home/yile/projects/g1_camera_sdk/d415_hand_reprocess_fit_new
python scripts/camera_extrinsics/reprocess_saved.py \
  --source-session /home/yile/projects/g1_camera_sdk/d415_live_extrinsics_03 \
  --output /home/yile/projects/g1_camera_sdk/d415_hand_reprocess_check_new
python scripts/camera_extrinsics/calibrate.py solve \
  --session /home/yile/projects/g1_camera_sdk/d415_hand_reprocess_fit_new
python scripts/camera_extrinsics/validate_candidate.py \
  --candidate /home/yile/projects/g1_camera_sdk/d415_hand_reprocess_fit_new/extrinsics_candidate.json \
  --session /home/yile/projects/g1_camera_sdk/d415_hand_reprocess_check_new
```

手触碰32项离线测试通过，新增强透视中心筛选和放大图坐标回原图的回归测试。

## 未知指尖偏移方案：仅讨论，尚未实现

现有采集和求解仍使用配置中的固定指尖偏移。后续可以研究把偏移与相机外参共同求解：同一个刚性指尖在手腕坐标系中的位置记为 `d`，第 `i` 次接触时的手腕到 torso 变换为 `(R_i, t_i)`，相机观测到的板上目标位置为 `c_i`，相机到 torso 外参为 `(R, t)`，则有：

```text
R_i d + t_i = R c_i + t
```

未知量是相机旋转3、平移3、指尖偏移3，共9个自由度。“同一个指尖”指手上的接触位置固定，板上可以触碰不同的已知目标。两次位置相减后仍含 `(R_i - R_j)d`；只有两次手腕朝向相同时，这一项才消失。只使用两点间距离约束，也不能直接确定相机外参。

每组相机观测和关节状态应在停稳时对应采集；各组都在当时的 torso 坐标系中建立上述约束。如果改在世界坐标系中比较，还必须乘入每次 torso 到世界的变换，不能假定机器人始终不动。

初步实验可考虑12–20组数据并另留至少5组检查，手腕朝向需要有充分变化；数量本身不保证可辨识，仍须检查退化、残差和独立验证结果。此方案尚未实现、未实测，不能把现有标定结果理解为已经消除了指尖偏移误差。

## 参考实现

- [Unitree SDK](https://github.com/unitreerobotics/unitree_sdk2_python)：状态订阅、关节索引。
- [官方 G1 模型](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/README.md)：版本对应和 URDF。
- [OpenCV](https://docs.opencv.org/4.5.2/d9/d0c/group__calib3d.html)：棋盘检测和 PnP。

离线验证命令：

```bash
python -m unittest discover -s scripts/camera_extrinsics -p 'test_*.py' -v
```
