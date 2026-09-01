"""RSL-RL PPO configuration for the Unitree G1 lock-waist walking task."""

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

#这个 G1 lock-waist walking 任务要训练多久、模型保存到哪里、神经网络多大、初始探索噪声多大
@configclass
class G1LockWaistWalkFlatPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 3000
        self.save_interval = 300
        self.experiment_name = "g1_lock_waist_walk_flat"
        self.policy.init_noise_std = 0.5
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
