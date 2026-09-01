"""Unitree-style flat walking task registration for the local G1 lock-waist model."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Velocity-Flat-G1-LockWaist-UnitreeStyle-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_flat_unitree_style_env_cfg:G1LockWaistUnitreeStyleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LockWaistUnitreeStylePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-LockWaist-UnitreeStyle-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_flat_unitree_style_env_cfg:G1LockWaistUnitreeStyleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LockWaistUnitreeStylePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-LockWaist-UnitreeStyle-Robust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walk_flat_unitree_style_env_cfg:G1LockWaistUnitreeStyleRobustEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LockWaistUnitreeStylePPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Velocity-Flat-G1-LockWaist-UnitreeStyle-TrackingTune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.walk_flat_unitree_style_env_cfg:G1LockWaistUnitreeStyleTrackingTuneEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LockWaistUnitreeStylePPORunnerCfg",
    },
)
