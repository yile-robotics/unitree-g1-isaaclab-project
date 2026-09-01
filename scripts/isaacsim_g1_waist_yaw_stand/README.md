# Isaac Sim G1 WaistYaw站立控制实验

这个目录是一个独立实验，不修改现有导航、训练任务或policy文件。

## 实际控制结构

当前部署的stand policy不是29维全身policy：

```text
stand ONNX/checkpoint输出：12维腿部动作
locomotion ONNX输出：29维全身动作
```

本实验采用与训练结构一致的组合：

```text
12维lower-body stand policy → 双腿12个q_des
双臂                         → 默认姿态
WaistRoll/WaistPitch         → 默认姿态
WaistYaw                     → 本实验的绝对rad目标
```

训练代码原本用`random_waist`产生腰部平滑扰动。本实验只在本次运行的内存配置中，
把它替换为`CommandableWaistAction`。`unitree_rl_lab`源码和checkpoint都不会变化。

当前控制层按Unitree官方`rt/arm_sdk`示例进行仿真近似：

```text
控制周期                    0.02s（50Hz）
WaistYaw q_des最大变化速度   0.5rad/s
WaistYaw kp/kd              60/1.5
启用时模拟arm_sdk weight     1
结束时weight释放速度         0.2/s（约5秒释放完成）
```

序列结束后，控制器先以0.5rad/s让WaistYaw回到默认角度，再把模拟接管权重从1降到0。
Isaac Sim没有Unitree真机固件内部的控制权仲裁，因此这里的weight是行为近似，不代表
真机固件内部一定采用相同的数学混合公式。

## 改造前后控制逻辑的变化

控制逻辑已经发生变化，但stand policy、checkpoint和腿部控制没有改变。

改造前使用minimum-jerk时间轨迹：

```text
用户角度
  → 按waist_yaw_transition_time生成固定时长的minimum-jerk插值
  → 得到WaistYaw位置目标
  → 使用原环境的执行器参数跟踪目标
```

这种方式中，`transition_time`直接决定角度曲线走完的时间；目标角度越大，实际要求的
平均速度也越大。

当前改成arm_sdk风格的逐周期位置限速：

```text
12维stand policy ───────────────→ 双腿12个joint position target

用户WaistYaw角度
  → policy/override/blend模式计算稳态目标
  → 关节软限位和实验安全限位
  → 每20ms最多改变 max_velocity × 0.02 的arm_sdk reference
  → 模拟arm_sdk接管权重混合
  → WaistYaw最终joint position target

WaistRoll、WaistPitch ──────────→ 默认关节位置
双臂 ──────────────────────────→ 环境默认姿态
```

默认`max_velocity=0.5rad/s`，因此每个50Hz控制周期最多变化：

```text
0.5rad/s × 0.02s = 0.01rad
```

当前的`--waist-yaw-transition-time`不再生成插值曲线，它只是规定一个stage在切换到
下一个目标前至少保留多久。要保证某一stage确实到达目标，应满足：

```text
transition_time >= abs(新目标 - 当前目标) / max_velocity
```

例如从0°到90°、速度为0.5rad/s，理论上至少需要约`1.5708 / 0.5 = 3.14s`；
本次使用5秒，所以能够到达目标后再保持。

WaistYaw执行器参数也只在本次仿真的内存配置中改为`kp=60、kd=1.5`。这不会写回
训练任务配置，也不会改变stand checkpoint。

## 当前每一步实际怎样执行

1. 启动后先进入`disabled`状态。12维stand policy持续控制双腿，WaistYaw保持默认位置。
2. warmup结束后调用`set_command()`。第一次接管会从仿真中的WaistYaw实际角度开始，
   避免q_des突然跳变；状态变为`active`，模拟接管权重设为1。
3. 每个0.02秒控制周期仍先运行一次stand policy，得到12维腿部动作。
4. WaistYaw控制项独立把reference向目标推进，但单步不超过0.01rad。
5. 最终发送的WaistYaw目标为：

   ```text
   q_final = q_baseline + arm_sdk_takeover_weight × (q_reference - q_baseline)
   ```

   在`active`和`returning`阶段，接管权重为1，因此最终目标就是限速后的reference。
6. 下发时，双腿继续使用stand policy的结果；只独立覆盖三个腰部目标，其中WaistYaw
   使用上述最终目标，WaistRoll和WaistPitch保持默认位置。
7. 序列结束后进入`returning`：接管权重仍保持1，WaistYaw先按0.5rad/s回到baseline。
8. 到达baseline后进入`releasing`：reference不再移动，接管权重默认以0.2/s从1降到0，
   约5秒后进入`disabled`并结束。

这里有两个不同的weight：

```text
waist_yaw_weight
    只用于mode=blend，决定用户目标在稳态目标中的比例。

arm_sdk_takeover_weight
    由状态机自动控制，近似表示辅助控制层是否接管最终WaistYaw q_des。
```

因此，本实验并不是把一个29维policy动作中的WaistYaw改掉。当前stand policy只输出
12个腿部动作，WaistYaw本来就不属于policy输出；本实验是在同一个IsaacLab Action
Manager中增加一个独立的腰部位置目标控制项，再与腿部policy输出一起下发给29DoF机器人。

## 三种模式

```text
policy   WaistYaw保持环境默认目标；当前stand policy本身不输出WaistYaw
override WaistYaw完全使用用户目标
blend    默认目标和用户目标按weight混合
```

blend稳态公式：

```text
q_final = (1 - weight) * q_baseline + weight * q_user
```

用户目标先经过模式计算，再使用官方示例风格的逐周期速度限幅更新。目标同时受
运行时关节范围和`--waist-yaw-max-abs-rad`实验安全上限约束。

## 运行

下面是已经实际验证通过的±90°平面往返实验：

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
conda activate isaacsim
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

python scripts/isaacsim_g1_waist_yaw_stand/run_waist_yaw_stand.py \
  --plane \
  --waist-mode override \
  --waist-yaw-sequence-deg "0,30,60,90,60,30,0,-30,-60,-90,-60,-30,0" \
  --waist-yaw-transition-time 5.0 \
  --waist-yaw-hold-time 2.0 \
  --waist-yaw-max-velocity-rad-s 0.5 \
  --arm-sdk-release-weight-rate 0.2 \
  --waist-yaw-kp 60 \
  --waist-yaw-kd 1.5 \
  --waist-yaw-max-abs-rad 1.570796 \
  --real-time
```

删除`--plane`即可使用默认房间：

```text
/home/yile/scene/House/scene_047/mujoco/usd/scene_scene_047.usda
spawn=(2.45, 1.15, 0.8)
yaw=pi
```

单目标模式需要传空的`--waist-yaw-sequence-deg ""`，再通过
`--waist-yaw-target-rad`指定弧度目标。`blend`模式通过`--waist-mode blend`和
`--waist-yaw-weight`设置用户目标与baseline的混合比例；当前baseline通常为0rad，
因此weight为0.5时，最终角度是用户目标的一半。这个参数不是arm_sdk接管权重；
接管权重由控制器状态机自动管理并记录为`arm_sdk_takeover_weight`。

## 输出

每次运行创建新目录：

```text
outputs/isaacsim_g1_waist_yaw_stand/run_<timestamp>/
├── config.json
├── metrics.csv
└── summary.json
```

`metrics.csv`逐步记录：

```text
WaistYaw实际角度、基线目标、arm_sdk限速目标、最终q_des、跟踪误差
控制状态、arm_sdk接管权重、用户blend权重
base roll/pitch/height和速度
左右脚接触与接触时横向滑动速度
fallen和fall_reason
```

默认摔倒判断：

```text
base height < 0.45m
或 abs(roll/pitch) > 0.65rad
```

达到阈值后默认停止实验，但关闭了IsaacLab自动reset，因此不会突然重置机器人。

## 改造前的仿真基线结果（2026-08-17）

以下结果来自改成arm_sdk速度限幅和`kp=60/kd=1.5`之前的minimum-jerk版本，
用于和新控制方式的结果对比，不能当作当前实现已经验证通过的数据。测试时stand
policy同样输出12维腿部动作，腰部控制层单独发送绝对WaistYaw位置目标。

### ±90°往返测试

实际执行序列：

```text
0° → +30° → +60° → +90° → +60° → +30° → 0°
   → -30° → -60° → -90° → -60° → -30° → 0°
```

观测结果：

```text
最大目标角度：约 ±1.5708rad（±90°）
运动过程最大跟踪误差：约 0.0031rad（0.18°）
base height：约 0.783m
base roll：约 -0.011rad
base pitch：约 0.009rad
fallen：始终为 False
```

机器人能够平滑到达正负90°并保持站立。该次日志在最后回到0°阶段由用户按
`Ctrl+C`结束，因此没有生成“序列自然完成”的终端记录，但全部目标阶段均已进入。

### 请求±150°时的实际±135°软限位测试

命令请求了最高`±2.618rad（约±150°）`，但当前Unitree G1 IsaacLab资产设置：

```text
soft_joint_pos_limit_factor = 0.9
```

实验控制层又有意读取`soft_joint_pos_limits`，因此实际限幅为：

```text
±2.618 × 0.9 = ±2.3562rad
±150° × 0.9 = ±135°
```

日志中的实际目标也证明了这个限幅：

```text
请求 +140° / +150° → 实际目标 +2.3562rad（+135°）
请求 -140° / -150° → 实际目标 -2.3562rad（-135°）
```

观测结果：

```text
实际验证范围：约 ±2.3562rad（±135°）
运动过程最大跟踪误差：约 0.0065rad（0.37°）
到达目标后的稳态误差：约 0.0001rad
base height：约 0.783m
base roll/pitch：大多约 0.01rad以内
fallen：已记录过程始终为 False
```

结论：旧版本已经验证到IsaacLab配置的正负软限位，且仿真中保持稳定；尚未真正验证
`±2.618rad（±150°）`机械硬限位。若未来需要测试硬限位，应在这个隔离实验中增加
明确的“仿真硬限位模式”，不要把硬限位实验直接用于真机。

## 当前arm_sdk风格版本实测结果（2026-08-17）

本次实际执行序列：

```text
0° → +90° → 0° → -90° → 0°
```

运行配置：

```text
控制频率：50Hz
WaistYaw最大目标变化速度：0.5rad/s
WaistYaw kp/kd：60/1.5
模拟arm_sdk接管权重：1.0
实验角度上限：±1.570796rad（±90°）
```

日志证明限速按预期工作：每0.5秒，`waist_arm_sdk_reference_rad`大约变化
`0.25rad`，与`0.5rad/s × 0.5s = 0.25rad`一致。机器人完成了正负90°往返，
全过程保持稳定。

观测结果：

```text
实际角度范围：约 ±1.5708rad（±90°）
运动过程最大跟踪误差：约 0.0108rad（0.62°）
base height：约 0.783m
base roll：大约 -0.01rad
base pitch：大约 0.004～0.010rad
fallen：始终为 False
```

本次数据保存在：

```text
outputs/isaacsim_g1_waist_yaw_stand/run_20260817_134339_951005/
```

本次在最终回到0°附近后由用户手动退出，因此运行记录停在约`37.04s`，最后状态仍为
`active`、`arm_sdk_takeover_weight=1.0`，并且没有生成`summary.json`。这是手动退出造成的，
不是控制异常。它表示这次已经验证了腰部接管、50Hz限速、正负90°跟踪以及站立稳定性，
但尚未验证序列自然结束后的模拟控制权释放流程：

```text
state=active, arm_weight=1.000
state=returning
state=releasing, arm_weight逐步下降
Simulated arm_sdk release complete.
```

按照本次参数，动作序列约在38秒完成，随后权重释放约需5秒。下一次应让程序自然运行约
44秒，看到上面的释放完成信息并生成`summary.json`后，再关闭Isaac Sim。

## 纯逻辑单元测试

```bash
cd /home/yile/projects/unitree-g1-isaaclab-project
pytest -q scripts/isaacsim_g1_waist_yaw_stand/tests
```
