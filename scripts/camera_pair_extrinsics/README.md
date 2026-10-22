# D415 与原装 D435i 的双相机外参标定

2026-10-16起，`new_camera.json`跟随胸前D415切到1280×720；`old_camera.json`仍为内置D435i的640×480。若用本仓库新版本程序重编译旧相机服务，需加 `--resolution 480p`。已有标定样本保留采集时的尺寸和内参，离线重算不改用当前720p内参。

本目录是新方案，独立于 `scripts/camera_extrinsics/` 的手触碰方案。它不读取手指、关节或正运动学，也不发送运动命令。只复用现有 `camera_d415/network_camera.py` 的 RGB-D TCP 接收接口；不改 SDK、固件或相机参数。

先求 **D415 RGB 光学坐标 → 原装 D435i RGB 光学坐标**。随后必须有可信的 **原装 RGB 光学坐标 → torso_link** 参考，才能合成胸部外参。当前该参考未确认，不能直接拿 URDF 的 `d435_link` 代替 RGB 光学坐标。此前手触碰候选不作为这个方案的参考输入。

## 理论与手触碰方法的区别

两种方法都求旋转 `R` 和平移 `t`，使同一物理点在不同坐标系中的表达满足 `p_A = R × p_B + t`。`T_A_B` 把B坐标转换到A坐标。相机RGB使用X右/Y下/Z前；胸部及本文安装角使用X前/Y左/Z上。

本方案利用同一块静止的编号板建立桥梁：各相机用RGB像素、SDK内参和已知板几何做PnP，分别求 `T_oldRGB_board` 和 `T_newRGB_board`。先从D415回到板，再从板到原装相机，即 `T_oldRGB_newRGB = T_oldRGB_board × inverse(T_newRGB_board)`。这里求逆是坐标变换反向，不是机器人逆运动学。不同编号也属于同一个板坐标系，所以两台相机不需要看到同样的格子。

多个板姿态用于联合优化：固定两台相机的内参，一个相机间刚性变换在所有组中保持相同，每组板位姿分别调整，使预测的编号中心像素尽量接近实测像素。求解使用RGB和已知板几何，深度图只保存作记录。

手触碰方法在[另一个独立目录](../camera_extrinsics/README.md)中，用“视觉板上接触点 ↔ 关节角＋URDF正运动学计算的指尖”建立相机与胸部的直接点对应；不需要原装相机参考，但依赖真实指尖偏移和接触准确性。本方案不使用指尖、关节或手触碰结果；得到相机间外参后，额外需要原装RGB→胸部的参考，才能合成胸部外参。

独立验证检查未拟合图像中的板位置、姿态和投影是否一致，不能验证并未进入图像求解的原装RGB→胸部参考。两种方法的验证误差使用不同参考，不是直接可比较的真实安装误差。

## 文件

| 文件 | 作用 |
| --- | --- |
| `calibrate_pair.py` | 双画面预览、静止采集、离线求解、独立验证、参考外参合成 |
| `board.py` | 解码 AprilTag，检查编号布局，用编号中心求板的位姿 |
| `geometry.py` | 两相机相对变换、固定内参联合重投影优化、验证和变换合成 |
| `reprocess_saved.py` | 从已保存双路RGB重新检测，保留原11/3分组，另存新session |
| `estimate_from_urdf.py` | 显式使用未验证的RGB光心/轴向假设合成URDF胸部估计，与已验证参考的compose入口分开 |
| `board.json` | 这块硬板：6×6 个编号，55 mm 大方块、17 mm 间隔、72 mm 中心距 |
| `old_camera.json` | 原装 D435i：PC2 `192.168.123.164:8766`，序列号 `344422072128` |
| `new_camera.json` | 新 D415：笔记本 `127.0.0.1:8765`，序列号 `211222063932` |
| `old_rgb_to_torso.example.json` | 原装 RGB 到胸部的参考模板，目前明确标记未确认 |
| `tests/test_geometry.py` | 变换方向、部分板、联合求解、独立验证和参考检查的离线测试 |

数据输出到新目录，例如 `/home/yile/projects/g1_camera_sdk/dual_camera_session_01`，不读取旧的触碰采集记录。

## 1. 两台相机取图服务

两台相机必须保持相对于机器人固定；整个采集期间不要重新安装 D415。如果相机仍在桌上，得到的是桌上这次布置的相对外参，安装到机器人后必须重新采集。

机器人终端：

```bash
ssh unitree@192.168.123.164
# 登录选择 1；取图本身不使用 ROS。
/home/unitree/g1_camera_sdk/local/build-probe-aarch64/g1_d435i_stream \
  --bind 192.168.123.164 --port 8766 --serial 344422072128 \
  --settings-dir /home/unitree/g1_camera_sdk/camera_pair_extrinsics_logs
```

笔记本终端 1：

```bash
/home/yile/projects/g1_camera_sdk/local/build-probe-x86_64/g1_d435i_stream \
  --bind 127.0.0.1 --port 8765 --serial 211222063932 \
  --settings-dir /home/yile/projects/g1_camera_sdk/d415_camera_logs
```

如果对应服务已在运行，不要再启动第二份。现有可执行文件名字保留 `d435i`，`--serial` 决定实际打开 D415 还是 D435i。两个服务都看到 `RGB-D ready` 再启动接收端。每个服务同时只允许一个客户端；先退出其他预览、VLN 或采集程序。

笔记本终端 2，先检查双画面：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
source /home/yile/projects/.venvs/unitree_g1/bin/activate
python scripts/camera_pair_extrinsics/calibrate_pair.py check --seconds 600
```

左边是原装相机，右边是 D415。各自至少看到 **6 个可解码编号，跨不同的行和列**。不要求完整板，也不要求两台相机看到同一组编号；两边都使用同一份全局编号布局。只看到一排不能确定板的位姿。调整板的位置，不要为了让板入镜而改变相机的安装角度。

## 2. 采集

在双画面窗口按 Q 退出检查，然后运行：

```bash
python scripts/camera_pair_extrinsics/calibrate_pair.py capture \
  --output /home/yile/projects/g1_camera_sdk/dual_camera_session_01 \
  --seconds 1800
```

操作按键在**图像窗口**，不是终端：

1. 把板摆好，让两边都显示 `Board pose ready`。
2. 按 **Space** 准备，再按 **Enter**。保持板、机器人和两台相机静止，程序检查 5 组新帧后保存最后一组。
3. 改变板的位置、距离和倾角，继续 Space → Enter。建议保存 10–15 组 `role=fit`，最少 8 组。
4. 在图像窗口按键 **v**（小写）切换为 `role=check`，以窗口或终端实际角色为准，再换 3–5 个板姿态采集。再次按v会切回fit；检查样本不参加求解。
5. 按 **Q** 退出。

板法线需要有至少 10° 的倾角变化。保持板在两台相机的共同可观测区域，不用手碰板。采集失败会说明原因，重新按 Space → Enter；不会把连续 Enter 当成下一组。每组保存两张 RGB、两份米制深度、内参、编号中心、各相机板位姿和原始帧时间信息。深度作为原始记录，本方案求解使用 RGB。

两路通过线程同时发出 TCP 请求，**不是硬件同步曝光**。不比较 PC2 与笔记本的系统时间；保存各自 SDK 时间、笔记本接收时间差和静止帧窗口。运动中的板不能用于这套采集方式。

## 本次采集排障记录

- `ready` 表示当前画面具备有效板位姿，不代表已保存。保存成功必须出现 `[SAVED] pair_XXX role=...`，并写入 `session.json` 和两张RGB文件。初期检查到样本数为0，随后在图像窗口 Space → Enter 才开始实际保存。
- 原装相机最初只见板底部一排，调整后D415一度只见一排。程序以 `IDs are collinear` 拒绝一排编号；要求至少6个有效ID分布在不同的行和列，不要求完整板或相同格子。
- 每次Enter检查5组新帧，跨度至少0.8秒；各相机需有至少6个始终识别的共同ID，编号中心最大像素跨度≤2px，板姿态旋转跨度≤2°。日志中的8.7、9.5、4.3、4.7、2.4px等尝试，以及验证采集中的2.1–2.2px或共同ID不足，均显示 `[RETRY]`，没有计入成功样本。显示数值保留一位小数，`2.0px` 仍可能是实际值略大于2px。
- 这些拒绝记录说明图像点或位姿估计发生变化，单凭日志不能区分真实晃动与检测抖动。调整板的可见范围并保持单次采集静止后，实际成功保存14组；没有通过放宽门限或补造样本得到这些结果。
- 图像窗口按v切换验证模式，后来再次按v切回fit，因此第12–14组原始标签也是fit。第12、13组在任何求解输出出现之前已预留验证，用户随后明确采第14组作为第三组验证。下面的分析快照保留原始标签并落实这个预先约定的分组，原始采集数据未改写。

## 3. 离线求解与验证

```bash
python scripts/camera_pair_extrinsics/calibrate_pair.py solve \
  --session /home/yile/projects/g1_camera_sdk/dual_camera_session_01
```

输出 `relative_candidate.json`。固定两台相机的 SDK 内参，用所有 fit 图像联合优化一个刚性相机间变换，以及每组板位姿。允许两幅图看到不同编号。利用未参与拟合的 check 样本，报告板中心三维误差、姿态差和跨相机重投影误差。它们是视觉一致性检查，不是实物测量真值，不自动宣布标定通过。

再次采集到新目录后，可验证已有候选，**不会重新拟合外参**：

```bash
python scripts/camera_pair_extrinsics/calibrate_pair.py validate \
  --candidate /home/yile/projects/g1_camera_sdk/dual_camera_session_01/relative_candidate.json \
  --session /home/yile/projects/g1_camera_sdk/dual_camera_session_02
```

## 4. 合成 torso_link 外参

约定 `T_A_B` 把 B 中的点变换到 A；所有平移单位为米。

```text
T_oldRGB_newRGB = T_oldRGB_board × inverse(T_newRGB_board)
T_torso_newRGB  = T_torso_oldRGB × T_oldRGB_newRGB
```

两台相机的 RGB 光学坐标都是 X 右、Y 下、Z 前。`T_oldRGB_newRGB` 的平移就是 D415 RGB 光学原点在原装 RGB 坐标中的位置。

把经核实的原装 RGB → torso_link 参考、序列号、来源填入模板副本，明确 `verified=true` 后才可以运行：

```bash
python scripts/camera_pair_extrinsics/calibrate_pair.py compose \
  --candidate /home/yile/projects/g1_camera_sdk/dual_camera_session_01/relative_candidate.json \
  --reference /path/to/verified_old_rgb_to_torso.json \
  --output /home/yile/projects/g1_camera_sdk/dual_camera_session_01/torso_candidate.json
```

未确认的模板会被拒绝。结果仍是候选，不自动修改 VLN 配置。原装相机参考的误差也会传递到新相机外参。

## 当前检查记录（2026-10-09）

两台相机 TCP 实际读取成功，证据保存于 `/home/yile/projects/g1_camera_sdk/dual_camera_check_03/`。原装相机先出现 RGB 帧停更、深度继续更新，重启本次取图服务后恢复；D415 正常。最后检查原装 RGB 帧号 744、深度 742，RGB-D 时间差约 0.044 ms；D415 RGB 156495、深度 156493，约 0.601 ms。这不是跨相机曝光时间差。

这次初始检查时原装相机只看到板底部一排编号，随后调整板位置完成了下面的实际采集。没有修改相机库、固件、显卡驱动或手触碰代码。

### 实际采集与求解结果

原始数据：`/home/yile/projects/g1_camera_sdk/dual_camera_session_02/`，共 14 组。采集界面标签都是 `fit`；第 12、13 组因操作时误切回 fit 被写成 fit，但在任何求解结果出现之前已经预留为验证，第 14 组随后明确作为第三组验证采集。

为避免采集进程覆盖标签并保留原始记录，分析使用单独快照 `dual_camera_session_02/analysis_11fit_3check/session.json`。前 11 组用于拟合，第 12–14 组独立验证；快照记录 `original_role`，原图仍引用原始目录。`validation_plan.json` 记录预留和分组原因，没有依据验证误差挑选或丢弃样本。

| 指标 | 结果 |
| --- | --- |
| 拟合组数 / 验证组数 | 11 / 3 |
| 板法线倾角范围 | 71.03° |
| 联合拟合重投影 RMS（像素坐标分量） | 0.425 px |
| 验证板中心三维一致性 RMS / 最大 | 0.798 / 1.021 mm |
| 验证姿态差最大 | 1.576° |
| 验证跨相机重投影 RMS 最大 | 1.986 px |

候选：`analysis_11fit_3check/relative_candidate.json`。D415 RGB 原点在原装 RGB 光学坐标下的位置为 `[-0.003797, 0.052619, 0.134902]` 米。这里的 X/Y/Z 是原装相机的右/下/前，不是机器人的前/左/上。完整旋转矩阵在候选 JSON 中。

这些误差反映采集距离和姿态下的两路 RGB 几何一致性，不是外部测量真值，不能把 0.8 mm 当作最终 torso 外参的绝对精度。原装 RGB → torso_link 参考仍未验证，因此双相机标定本身只给出相对外参；后续按 URDF 假设合成的胸部估计见下一节。两种候选均未应用 VLN。

### 按 URDF 假设合成的胸部估计值

用户随后要求按现有 URDF 计算 D415 → torso_link。估计另存为 `analysis_11fit_3check/torso_urdf_assumed_candidate.json`，不替换双相机候选，不设置参考已验证，也不应用 VLN。

该计算明确假设原装 `d435_link` 为 X 前/Y 左/Z 上，且其原点与原装 RGB 光心重合。这两个假设没有被该 URDF 或实物测量确认。使用的 RGB → d435_link 假设旋转是：

```text
R_d435_RGB = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
T_torso_newRGB = T_torso_d435(URDF) × T_d435_oldRGB(假设) × T_oldRGB_newRGB(标定)
```

估计 D415 RGB 光心在 torso_link 下的位置为 `[0.109731, 0.021327, 0.294770]` 米。将新相机坐标也转换为 X 前/Y 左/Z 上后，安装 Roll/Pitch/Yaw 为 `[-0.285942°, -10.828113°, +1.014154°]`，对应相对 torso 上仰约 10.83°。

RGB 光学坐标本身的 RPY 为 `[-79.171755°, -0.280851°, -89.039565°]`；它与上面的安装角使用不同轴定义，不能混用。JSON 同时保存两种矩阵和角度。此估计的绝对误差仍取决于未验证的原装相机参考，不能引用双相机的 0.8 mm 验证结果来证明它的胸部定位精度。

与此前 `d415_live_extrinsics_02/extrinsics_candidate.json` 手触碰候选比较（两者都为 RGB 光学坐标 → torso_link），位置差向量为 `[+36.57, -24.19, -10.70]` mm，三维距离为 **45.13 mm**，相对旋转角为 **7.14°**。同一前/左/上安装坐标下，手触碰的 RPY 为 `[+2.674°, -4.741°, -1.705°]`，URDF 假设合成结果为 `[-0.286°, -10.828°, +1.014°]`。完整比较另存为 `analysis_11fit_3check/comparison_with_hand_candidate.json`。

这是候选之间的差异，不是某一候选相对于真值的误差。物理比较要求 D415 安装位置在两次采集之间没有变化。手触碰候选的指尖模型偏移未实物核实；URDF 合成候选的原装 RGB 光心参考未确认，不能据此判断任一结果就是最终真值。

复现本次求解：

```bash
OPENBLAS_NUM_THREADS=1 python scripts/camera_pair_extrinsics/calibrate_pair.py solve \
  --session /home/yile/projects/g1_camera_sdk/dual_camera_session_02/analysis_11fit_3check
```

### 2026-10-13 从原图重算

双相机中心筛选原本已经使用对角线交点，本次修正的是多尺度回原图的像素中心映射：`(corner + 0.5) / scale - 0.5`。从原分析快照重读全部14组的28张RGB，重新解码、求板位姿、优化相机间变换；原fit/check分组保持11/3，不丢弃样本。

新目录：`/home/yile/projects/g1_camera_sdk/dual_camera_session_02/reprocessed_20261013/`。

| 指标 | 原结果 | 重算结果 |
|---|---:|---:|
| 联合重投影 RMS（像素坐标分量） | 0.424529 px | 0.424700 px |
| 验证板中心一致性 RMS | 0.798 mm | 0.848 mm |
| 验证板中心一致性最大 | 1.021 mm | 1.102 mm |
| 验证姿态差最大 | 1.576° | 1.580° |
| 验证跨相机重投影 RMS 最大 | 1.986 px | 1.897 px |

新相机间外参相比原结果平移变化0.210mm、旋转变化0.023°，影响很小。重新应用相同URDF假设得到胸部估计位置 `[0.109807,0.021383,0.294582]` 米，前/左/上安装RPY `[-0.281803°,-10.849470°,+1.005847°]`。原装RGB→胸部参考仍未验证，不应用VLN。

与同日重算后的手触碰结果比较，位置差51.244mm、旋转差8.011°；修正没有消除两种胸部候选之间的分歧。完整跨方法及新旧比较为 `/home/yile/projects/g1_camera_sdk/board_geometry_audit_20261013/reprocessed_comparison.json`。

新目录包含 `session.json`、`reprocess_report.json`、`relative_candidate.json`、`torso_urdf_assumed_candidate.json` 及 `comparison_with_hand_candidate.json`。原图仍引用源目录，原来的窗口静止诊断保留为 `_original`，仅代表当时采集检查，未从未保存的全部burst RGB重新计算。

复现时选择不存在的新输出目录：

```bash
python scripts/camera_pair_extrinsics/reprocess_saved.py \
  --source-session /home/yile/projects/g1_camera_sdk/dual_camera_session_02/analysis_11fit_3check \
  --output /home/yile/projects/g1_camera_sdk/dual_camera_session_02/reprocess_new
OPENBLAS_NUM_THREADS=1 python scripts/camera_pair_extrinsics/calibrate_pair.py solve \
  --session /home/yile/projects/g1_camera_sdk/dual_camera_session_02/reprocess_new
python scripts/camera_pair_extrinsics/estimate_from_urdf.py \
  --relative-candidate /home/yile/projects/g1_camera_sdk/dual_camera_session_02/reprocess_new/relative_candidate.json \
  --assume-rgb-at-link-origin \
  --output /home/yile/projects/g1_camera_sdk/dual_camera_session_02/reprocess_new/torso_urdf_assumed_candidate.json
```

双相机12项离线测试通过，包括原图坐标回算、光学前方到URDF前方的轴转换以及未验证状态保留。

离线测试：

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s scripts/camera_pair_extrinsics/tests -v
```
