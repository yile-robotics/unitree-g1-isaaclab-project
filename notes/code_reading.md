# Code Reading Notes

## Main IsaacLab files for Unitree G1

### 1. Robot asset configuration

Path:

`~/projects/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py`

Important symbols:

- `G1_MINIMAL_CFG`
- `G1_29DOF_CFG`
- robot USD spawn configuration
- initial joint positions
- actuator stiffness
- actuator damping
- effort and velocity limits

This file answers:

- Where is the G1 model loaded from?
- What is the initial standing pose?
- What are the PD / actuator parameters?

---

### 2. General velocity locomotion task

Path:

`~/projects/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

Important components:

- commands
- actions
- observations
- rewards
- terminations
- scene configuration

This file answers:

- What command does the policy receive?
- What does the policy observe?
- What action does the policy output?
- When does the environment reset?

---

### 3. G1 flat terrain task

Path:

`~/projects/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/flat_env_cfg.py`

This is the recommended first task for understanding G1 standing and flat-ground locomotion.

Important classes:

- `G1FlatEnvCfg`
- `G1FlatEnvCfg_PLAY`

---

### 4. G1 rough terrain and reward configuration

Path:

`~/projects/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py`

Important classes:

- `G1Rewards`
- `G1RoughEnvCfg`
- `G1RoughEnvCfg_PLAY`

Standing-related reward terms to understand:

- `flat_orientation_l2`
- `ang_vel_xy_l2`
- `lin_vel_z_l2`
- `joint_torques_l2`
- `joint_acc_l2`
- `undesired_contacts`

---

### 5. PPO training configuration

Path:

`~/projects/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py`

Important information:

- experiment name
- max iterations
- network architecture
- checkpoint and log directory
