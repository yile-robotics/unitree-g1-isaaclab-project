import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-Agile-Reward-Velocity-StandSmooth",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:StandSmoothVelocityEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:StandSmoothVelocityPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_ppo_cfg:StandSmoothVelocityPPORunnerCfg"
        ),
    },
)
