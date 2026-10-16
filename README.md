# Unitree G1 IsaacLab Project

Clean external Isaac Lab project for training a Unitree G1 lock-waist standing policy.

The first target is simple: load `g1_29dof_lock_waist_rev_1_0`, stand on flat ground, and keep the code separate from the main IsaacLab repository. Later stages can add velocity walking and VLA/navigation on top of the low-level policy.

## 真机相机、标定与 VLN 测试

当前真机使用胸前 D415。已跑通旋转采集、一轮模型决策及本机 iPlanner 实时路径预览；
避障净空、可靠安装外参和完整真机导航仍待验证，不能把“返回路径”视为“避障成功”。

- [2026-10-13、10-15 工作记录与结果索引](docs/real_robot/README.md)：包含可直接查看的图片、候选外参及原始结果摘要。
- [真机测试总记录](scripts/isaacsim_lavira_g3_interface_g1/README_G1_REAL_TEST.md)：运控、SLAM、相机与当前完成范围。
- [旋转一圈、采图后请求模型](scripts/isaacsim_lavira_g3_interface_g1/README_ROTATION_CAPTURE.md)。
- [实时相机与 iPlanner 规划观察窗口](scripts/isaacsim_lavira_g3_interface_g1/README_IPLANNER_VIEWER.md)：只显示，不发送机器人命令。
- [手触碰外参标定](scripts/camera_extrinsics/README.md)与[独立双相机标定](scripts/camera_pair_extrinsics/README.md)。

以下原有 Isaac Lab 训练说明保留；本次工作没有修改仿真控制流程。

## Layout

```text
unitree-g1-isaaclab-project/
  assets/
    g1_29dof_lock_waist_rev_1_0/
      g1_29dof_lock_waist_rev_1_0.urdf
      meshes/
      usd/
  commands/
    stand_flat.md
  scripts/
    train_stand_flat.py
    play_stand_flat.py
  source/
    unitree_g1_isaaclab/
      config/extension.toml
      unitree_g1_isaaclab/
        assets/g1_lock_waist.py
        tasks/stand_flat/
```

## Environment

- Conda environment: `isaacsim`
- IsaacLab source: `/home/yile/projects/IsaacLab`
- This project: `/home/yile/projects/unitree-g1-isaaclab-project`
- Registered task: `Isaac-Stand-Flat-G1-LockWaist-v0`
- Play task: `Isaac-Stand-Flat-G1-LockWaist-Play-v0`

## Quick Start

Install the local extension once:

```bash
conda activate isaacsim
cd /home/yile/projects/IsaacLab
./isaaclab.sh -p -m pip install -e /home/yile/projects/unitree-g1-isaaclab-project/source/unitree_g1_isaaclab --no-build-isolation
```

Train:

```bash
conda activate isaacsim
cd /home/yile/projects/IsaacLab
./isaaclab.sh -p /home/yile/projects/unitree-g1-isaaclab-project/scripts/train_stand_flat.py \
  --task Isaac-Stand-Flat-G1-LockWaist-v0 \
  --num_envs 1024 \
  --headless
```

Play:

```bash
conda activate isaacsim
cd /home/yile/projects/IsaacLab
./isaaclab.sh -p /home/yile/projects/unitree-g1-isaaclab-project/scripts/play_stand_flat.py \
  --task Isaac-Stand-Flat-G1-LockWaist-Play-v0 \
  --num_envs 16
```

More detailed commands are in `commands/stand_flat.md`.

## Notes

The USD has already been generated from the copied URDF. If the URDF changes, regenerate the USD using the command in `commands/stand_flat.md`.

This project intentionally does not modify IsaacLab's built-in G1 tasks.
