"""Flat-ground velocity walking task registration for the Unitree G1 lock-waist model."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Velocity-Flat-G1-LockWaist-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_flat_env_cfg:G1LockWaistWalkFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LockWaistWalkFlatPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-LockWaist-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_flat_env_cfg:G1LockWaistWalkFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LockWaistWalkFlatPPORunnerCfg",
    },
)
