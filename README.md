# Unitree G1 IsaacLab Project

Clean external Isaac Lab project for training a Unitree G1 lock-waist standing policy.

The first target is simple: load `g1_29dof_lock_waist_rev_1_0`, stand on flat ground, and keep the code separate from the main IsaacLab repository. Later stages can add velocity walking and VLA/navigation on top of the low-level policy.

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
