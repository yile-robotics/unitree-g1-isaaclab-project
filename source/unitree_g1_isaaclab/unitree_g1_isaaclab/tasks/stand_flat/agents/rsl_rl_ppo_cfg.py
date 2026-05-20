"""RSL-RL PPO configuration for the Unitree G1 standing task."""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class G1LockWaistStandFlatPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1200
        self.save_interval = 50
        self.experiment_name = "g1_lock_waist_stand_flat"
        self.policy.init_noise_std = 0.5
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
