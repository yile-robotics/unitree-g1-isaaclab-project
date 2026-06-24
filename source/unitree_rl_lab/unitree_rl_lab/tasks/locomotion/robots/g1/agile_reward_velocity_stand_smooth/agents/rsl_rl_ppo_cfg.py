from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class StandSmoothVelocityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Conservative PPO settings for second-stage reward fine-tuning."""

    seed = 42
    num_steps_per_env = 24
    max_iterations = 2_000
    save_interval = 500
    experiment_name = "unitree_g1_29dof_agile_reward_velocity_stand_smooth"
    run_name = "stand05_contact05_clearance12_dofvel0005_acc025"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.15,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
