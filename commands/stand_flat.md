# Stand Flat Commands

All commands use the existing `isaacsim` conda environment.

## 1. Install Extension

Run once, and rerun after changing package metadata:

```bash
conda activate isaacsim
cd /home/yile/projects/IsaacLab
./isaaclab.sh -p -m pip install -e /home/yile/projects/unitree-g1-isaaclab-project/source/unitree_g1_isaaclab --no-build-isolation
```

Quick import check:

```bash
conda activate isaacsim
python -c "import unitree_g1_isaaclab.tasks; print('ok')"
```

## 2. Regenerate USD

Only needed after editing the copied URDF or meshes:

```bash
conda activate isaacsim
cd /home/yile/projects/IsaacLab
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/yile/projects/unitree-g1-isaaclab-project/assets/g1_29dof_lock_waist_rev_1_0/g1_29dof_lock_waist_rev_1_0.urdf \
  /home/yile/projects/unitree-g1-isaaclab-project/assets/g1_29dof_lock_waist_rev_1_0/usd/g1_29dof_lock_waist_rev_1_0.usd \
  --joint-stiffness 100.0 \
  --joint-damping 2.0 \
  --joint-target-type position \
  --headless
```

## 3. Train Standing

```bash
conda activate isaacsim
cd /home/yile/projects/IsaacLab
./isaaclab.sh -p /home/yile/projects/unitree-g1-isaaclab-project/scripts/train_stand_flat.py \
  --task Isaac-Stand-Flat-G1-LockWaist-v0 \
  --num_envs 1024 \
  --headless
```

For a small smoke test:

```bash
conda activate isaacsim
cd /home/yile/projects/IsaacLab
./isaaclab.sh -p /home/yile/projects/unitree-g1-isaaclab-project/scripts/train_stand_flat.py \
  --task Isaac-Stand-Flat-G1-LockWaist-v0 \
  --num_envs 16 \
  --max_iterations 2 \
  --headless
```

## 4. Play

```bash
conda activate isaacsim
cd /home/yile/projects/IsaacLab
./isaaclab.sh -p /home/yile/projects/unitree-g1-isaaclab-project/scripts/play_stand_flat.py \
  --task Isaac-Stand-Flat-G1-LockWaist-Play-v0 \
  --num_envs 16
```
