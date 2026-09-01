"""Flat-ground velocity walking task registration for the Unitree G1 lock-waist model."""
#这段代码把你自定义的 G1 lock-waist 行走任务注册到 Gymnasium / IsaacLab 里
import gymnasium as gym

from . import agents

#第一个 gym.register：训练用环境
gym.register(
    id="Isaac-Velocity-Flat-G1-LockWaist-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_flat_env_cfg:G1LockWaistWalkFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LockWaistWalkFlatPPORunnerCfg",
    },
)

#第二个 register：Play 测试用环境
gym.register(
    id="Isaac-Velocity-Flat-G1-LockWaist-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_flat_env_cfg:G1LockWaistWalkFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LockWaistWalkFlatPPORunnerCfg",
    },
)
