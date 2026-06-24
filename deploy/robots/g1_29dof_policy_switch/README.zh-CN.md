# G1 双策略安全切换测试

这套程序用于在 MuJoCo 中独立测试 locomotion policy 和 stand policy 的切换。

## 隔离原则

- 不修改任何训练任务、reward 或 checkpoint。
- 不修改原来的 `deploy/robots/g1_29dof`。
- 不修改 `unitree_mujoco/simulate/config.yaml`。
- 使用独立可执行文件 `g1_policy_switch_ctrl`。
- 使用 DDS domain 10；原测试默认使用 domain 0。

## 切换逻辑

站立切到行走时，两个 policy 会同时推理，并在 0.6 秒内对实际关节位置目标进行
smoothstep 混合。速度命令也会逐渐增加，不会一步跳变。

行走切回站立时，控制器先把 locomotion 的速度命令降到零，然后抓一个短暂的
可接管窗口：身体姿态稳定、腿部关节速度较小、yaw 速度较小。若一直抓不到窗口，
会继续保持 locomotion 零命令等待；确认姿态安全时可以按 `3` 手动强制切到站立
policy。这样避免在摆腿相位无条件超时强切。

每个 policy 拥有独立的 observation history 和 last action，不会把一套 policy 的历史
错误地交给另一套 policy。

此外，最终关节目标还会受到每周期变化限幅。姿态倾斜超过配置阈值时，FSM 会回到
Passive。

## 准备策略

默认使用：

- locomotion: `model_33000_yawrew25_mujoco`
- stand: `model_26999_lower_body_stand`

控制器使用官方 `v0` 的整机 PD。两套策略的腿部 PD 相同；站立策略不控制腰臂，
因此腰部和手臂也沿用官方 `v0` 的 PD，避免切换后官方行走策略的腰臂阻尼被覆盖。

```bash
cd /home/yile/projects/unitree_rl_lab/deploy/robots/g1_29dof_policy_switch
chmod +x scripts/*.sh
./scripts/prepare_policies.sh
```

也可以显式指定两个策略目录：

```bash
./scripts/prepare_policies.sh \
  /absolute/path/to/locomotion_policy \
  /absolute/path/to/stand_policy
```

## 编译

```bash
./scripts/build.sh
```

## 独立 MuJoCo 测试

```bash
./scripts/run_mujoco_policy_switch.sh
```

控制器进入 `PolicySwitch` 后：

- `1`：请求站立 policy
- `2`：请求 locomotion policy
- `3`：强制平滑切换到站立 policy，仅用于调试等待条件
- `w/s`：增加/减少前向速度
- `a/d`：增加/减少横向速度
- `q/e`：增加/减少 yaw 速度
- `x`：速度命令归零

第一次测试建议：

1. 保持弹力绳，确认进入 stand policy 后关节没有跳变。
2. 在控制器终端按 `2` 切到 locomotion，但先保持零速度。
3. 用 `w` 增加到 `0.05 m/s`。
4. 按 `x` 清零，再按 `1` 请求站立。
5. 确认上述流程稳定后，再逐渐提高速度和释放弹力绳。
