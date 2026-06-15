from dataclasses import field

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CoordResidualActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticCoordResidual"
    coord_trunk_hidden_dims: tuple[int, ...] = (512, 256)
    body_head_hidden_dims: tuple[int, ...] = (256, 128)
    hand_head_hidden_dims: tuple[int, ...] = (256, 128)
    body_residual_scale: float = 1.0
    hand_residual_scale: float = 1.0
    fixed_log_std: bool = True
    actor_obs_term_names: tuple[str, ...] = ()
    actor_obs_term_dims: list[int] = field(default_factory=list)


@configclass
class LocomanipCoordResidualPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 1000
    experiment_name = "coordex_locomanip"
    empirical_normalization = True
    policy = CoordResidualActorCriticCfg(
        init_noise_std=0.22,
        noise_std_type="log",
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
