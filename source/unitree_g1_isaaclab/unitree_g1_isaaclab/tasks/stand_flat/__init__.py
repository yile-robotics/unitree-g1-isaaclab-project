"""Standing task registration for the Unitree G1 lock-waist model."""

import gymnasium as gym

from . import agents
#它注册了任务名：   
#Isaac-Stand-Flat-G1-LockWaist-v0
#Isaac-Stand-Flat-G1-LockWaist-Play-v0
gym.register(
    id="Isaac-Stand-Flat-G1-LockWaist-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_flat_env_cfg:G1LockWaistStandFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LockWaistStandFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Stand-Flat-G1-LockWaist-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_flat_env_cfg:G1LockWaistStandFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LockWaistStandFlatPPORunnerCfg",
    },
)
