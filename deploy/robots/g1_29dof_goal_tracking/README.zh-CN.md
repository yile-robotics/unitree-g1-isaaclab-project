# G1 独立 Goal Tracking 测试

这套程序是从 `g1_29dof_policy_switch` 独立复制出来的 goal tracking 实验链路。

## 隔离原则

- 不修改原来的 `deploy/robots/g1_29dof_policy_switch`。
- 不修改原来的 `deploy/robots/g1_29dof`。
- 不修改训练任务、reward、checkpoint。
- 使用独立可执行文件 `g1_goal_tracking_ctrl`。
- 使用独立 DDS domain `11`。
- 使用独立 MuJoCo 场景 `scene_29dof_goal_tracking.xml`。
- 该场景加载独立模型 `g1_29dof_goal_tracking_camera.xml`，不会修改原始
  `g1_29dof.xml`。头部紫红色测点 `goal_tracking_head_camera_site`
  近似表示实体 G1 头部视觉模组的光心位置，用于 goal tracking 期间的振动测试。
- 启动脚本会为每次测试创建独立目录
  `outputs/goal_tracking/runs/YYYYMMDD_HHMMSS_PID/`，记录轨迹、头部测点
  数据、两张 PNG 和振动统计文本。`outputs/goal_tracking/latest_run.txt`
  只记录最近一次目录路径，不覆盖历史结果。
- CSV 输出到 `unitree_rl_lab/outputs/goal_tracking/latest/`。

## 当前功能

- 复用 locomotion velocity policy。
- 复用 stand policy。
- 复用 stand 和 locomotion 的平滑切换逻辑。
- 从 MuJoCo bridge 发布的 `rt/sportmodestate` 读取世界 `x/y`。
- 从 lowstate IMU quaternion 读取 yaw。
- 用 `PathFollower` 把路径投影点、lookahead 前视点和路径切线 yaw 转成 `[vx, vy, yaw_rate]`。
- 机器人沿整条 waypoint 折线连续跟踪；到达最后一个 waypoint 且 yaw 对齐后速度归零并请求切回 stand。
- 在 MuJoCo 里显示起点、中间 waypoint、期望折线路径和最终目标点。
- 保存 trajectory CSV，并可离线画轨迹图。

## 准备策略

```bash
cd /home/yile/projects/unitree_rl_lab/deploy/robots/g1_29dof_goal_tracking
./scripts/prepare_policies.sh
```

默认复制：

- locomotion: `model_33000_yawrew25_mujoco`
- stand: `model_26999_lower_body_stand`

也可以指定两个策略目录：

```bash
./scripts/prepare_policies.sh /path/to/locomotion_policy /path/to/stand_policy
```

## 编译

```bash
./scripts/build.sh
```

输出在：

```text
build/g1_goal_tracking_ctrl
```

## 运行

```bash
./scripts/run_mujoco_goal_tracking.sh
```

这个脚本会加载独立场景：

```text
/home/yile/projects/unitree_mujoco/unitree_robots/g1/scene_29dof_goal_tracking.xml
```

MuJoCo 中的可视化标记：

- 绿色球：起点 `(0, 0)`
- 青色线：期望 waypoint 折线路径
- 青色小球：中间 waypoint
- 红色球/柱：最终目标点 `(6, 0)`

默认路径在 `config/config.yaml`：

```yaml
goal:
  x: 6.0
  y: 0.0
  yaw: 0.0
waypoints:
  - {x: 0.0, y: 0.0, yaw: 0.00}
  - {x: 1.0, y: 0.0, yaw: 0.00}
  - {x: 2.0, y: 0.6, yaw: 0.55}
  - {x: 3.0, y: -0.6, yaw: -0.75}
  - {x: 4.0, y: 0.7, yaw: 0.75}
  - {x: 5.0, y: -0.4, yaw: -0.55}
  - {x: 6.0, y: 0.0, yaw: 0.00}
path_lookahead_distance: 0.7
```

控制器进入 `GoalTracking` 后默认保持站立，不会自动启动 goal follower。
先在 MuJoCo 窗口按 `8` 放下机器人，站稳后按 `9` 取消弹力绳；
然后回到控制器终端按 `g` 开始 goal tracking。

按键：

- `1`: 请求站立 policy
- `2`: 请求 locomotion policy
- `3`: 强制平滑切换到站立 policy
- `g`: 启动或重新启动 goal tracking，并平滑切到 locomotion
- `x`: 停止 goal tracking，并将速度命令归零
- `w/s/a/d/q/e`: 停止 goal tracking 后手动微调速度命令

## 画图

仿真结束后：

```bash
python3 /home/yile/projects/unitree_rl_lab/scripts/goal_tracking/plot_goal_tracking.py
```

默认读取：

```text
/home/yile/projects/unitree_rl_lab/outputs/goal_tracking/latest/trajectory.csv
```

默认输出：

```text
/home/yile/projects/unitree_rl_lab/outputs/goal_tracking/latest/trajectory.png
```
