# Run Commands

## Open empty Isaac Sim scene

```bash
cd ~/projects/IsaacLab
conda activate isaacsim
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

```

## Show biped robot demo

```bash
cd ~/projects/IsaacLab
conda activate isaacsim
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
./isaaclab.sh -p scripts/demos/bipeds.py
```

## Try G1 flat play task

This requires a trained checkpoint. If no checkpoint exists, `play.py` exits because `logs/rsl_rl/g1_flat` does not exist.

```bash
cd ~/projects/IsaacLab
conda activate isaacsim
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-G1-Play-v0
```

## Train G1 flat locomotion task with GUI

```bash
cd ~/projects/IsaacLab
conda activate isaacsim
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0
```

## Train G1 flat locomotion task headless

```bash
cd ~/projects/IsaacLab
conda activate isaacsim
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0 --headless
```
