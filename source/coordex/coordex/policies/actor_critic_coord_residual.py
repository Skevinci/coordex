from __future__ import annotations

import numbers
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


def _term_dim_to_int(dim: int | Sequence[int]) -> int:
    if isinstance(dim, numbers.Integral):
        return int(dim)
    size = 1
    for value in dim:
        size *= int(value)
    return size


def _expected_dim_matches(actual_dim: int, expected_dim: int | Sequence[int]) -> bool:
    if isinstance(expected_dim, numbers.Integral):
        return actual_dim == int(expected_dim)
    return actual_dim in {int(value) for value in expected_dim}


def _format_expected_dim(expected_dim: int | Sequence[int]) -> str:
    if isinstance(expected_dim, numbers.Integral):
        return str(int(expected_dim))
    return "one of " + ", ".join(str(int(value)) for value in expected_dim)


def _build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(resolve_nn_activation(activation))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class _CoordResidualActor(nn.Module):
    """Locomanip actor with a shared coordination bottleneck and body/hand heads."""

    _EXPECTED_TERM_DIMS = {
        "projected_gravity": 3,
        "humanoid_prior": (450, 465),
        "right_hand_prior": 66,
        "body_prior_mean": 16,
        "hand_prior_mean": 12,
        "last_latent": 28,
        "fingertip_cube_forces": 15,
        "cube_pose": 9,
        "source_table_pose": 9,
        "cube_pose_in_hand": 9,
    }
    _OPTIONAL_TERM_DIMS = {
        "target_table_pose": 9,
        "target_region_pose": 9,
    }
    _KNOWN_TERM_DIMS = {**_EXPECTED_TERM_DIMS, **_OPTIONAL_TERM_DIMS}

    def __init__(
        self,
        num_actor_obs: int,
        *,
        actor_obs_term_names: Sequence[str],
        actor_obs_term_dims: Sequence[int | Sequence[int]],
        coord_trunk_hidden_dims: Sequence[int],
        body_head_hidden_dims: Sequence[int],
        hand_head_hidden_dims: Sequence[int],
        activation: str,
        body_residual_scale: float,
        hand_residual_scale: float,
    ) -> None:
        super().__init__()

        term_names = [str(name) for name in actor_obs_term_names]
        term_dims = [_term_dim_to_int(dim) for dim in actor_obs_term_dims]
        if len(term_names) != len(term_dims):
            raise ValueError(
                "ActorCriticCoordResidual received mismatched actor term metadata: "
                f"{len(term_names)} names vs {len(term_dims)} dims."
            )

        current = 0
        self._term_slices: dict[str, slice] = {}
        self._term_dims: dict[str, int] = {}
        for name, dim in zip(term_names, term_dims):
            self._term_slices[name] = slice(current, current + dim)
            self._term_dims[name] = dim
            current += dim
        if current != int(num_actor_obs):
            raise ValueError(
                "Actor observation metadata does not match num_actor_obs: "
                f"metadata total={current}, num_actor_obs={num_actor_obs}."
            )

        missing = sorted(set(self._EXPECTED_TERM_DIMS) - set(self._term_slices))
        if missing:
            raise ValueError(
                "ActorCriticCoordResidual is missing required actor observation terms: " + ", ".join(missing)
            )
        for name, expected_dim in self._KNOWN_TERM_DIMS.items():
            if name not in self._term_dims:
                continue
            actual_dim = self._term_dims[name]
            if not _expected_dim_matches(actual_dim, expected_dim):
                raise ValueError(
                    f"Actor observation term '{name}' must have dim {_format_expected_dim(expected_dim)}, got {actual_dim}."
                )

        self.num_prop = int(num_actor_obs)
        self.stage_dim = self._term_dims.get("stage", 0)
        self.body_latent_dim = self._term_dims["body_prior_mean"]
        self.hand_latent_dim = self._term_dims["hand_prior_mean"]
        self.num_actions = self.body_latent_dim + self.hand_latent_dim
        self.body_residual_scale = float(body_residual_scale)
        self.hand_residual_scale = float(hand_residual_scale)

        x_ctx_dim = (
            self.stage_dim
            + self._term_dims["projected_gravity"]
            + self._term_dims["last_latent"]
            + self._term_dims["cube_pose"]
            + self._term_dims["source_table_pose"]
            + self._term_dims["cube_pose_in_hand"]
            + self._term_dims["fingertip_cube_forces"]
            + sum(self._term_dims[name] for name in self._OPTIONAL_TERM_DIMS if name in self._term_dims)
        )
        coord_input_dim = x_ctx_dim + self.body_latent_dim + self.hand_latent_dim
        coord_feature_dim = int(coord_trunk_hidden_dims[-1])
        body_input_dim = coord_feature_dim + self._term_dims["humanoid_prior"] + self.body_latent_dim
        hand_input_dim = (
            coord_feature_dim
            + self._term_dims["right_hand_prior"]
            + self.hand_latent_dim
            + self._term_dims["cube_pose_in_hand"]
            + self._term_dims["fingertip_cube_forces"]
        )

        self.coord_trunk = _build_mlp(coord_input_dim, coord_trunk_hidden_dims[:-1], coord_feature_dim, activation)
        body_feature_dim = int(body_head_hidden_dims[-1])
        hand_feature_dim = int(hand_head_hidden_dims[-1])
        self.body_head_backbone = _build_mlp(body_input_dim, body_head_hidden_dims[:-1], body_feature_dim, activation)
        self.hand_head_backbone = _build_mlp(hand_input_dim, hand_head_hidden_dims[:-1], hand_feature_dim, activation)
        self.body_residual_linear = nn.Linear(body_feature_dim, self.body_latent_dim)
        self.hand_residual_linear = nn.Linear(hand_feature_dim, self.hand_latent_dim)

        self.last_body_residual: torch.Tensor | None = None
        self.last_hand_residual: torch.Tensor | None = None
        # Preserve runner compatibility when old gate logging is still enabled.
        self.last_body_gate: torch.Tensor | None = None
        self.last_hand_gate: torch.Tensor | None = None

    def _slice(self, observations: torch.Tensor, term_name: str) -> torch.Tensor:
        return observations[:, self._term_slices[term_name]]

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2:
            observations = observations.view(observations.shape[0], -1)

        mu_b = self._slice(observations, "body_prior_mean")
        mu_h = self._slice(observations, "hand_prior_mean")
        cube_pose_in_hand = self._slice(observations, "cube_pose_in_hand")
        fingertip_cube_forces = self._slice(observations, "fingertip_cube_forces")

        x_ctx_terms = []
        if self.stage_dim:
            x_ctx_terms.append(self._slice(observations, "stage"))
        x_ctx_terms.extend(
            [
                self._slice(observations, "projected_gravity"),
                self._slice(observations, "last_latent"),
                self._slice(observations, "cube_pose"),
                self._slice(observations, "source_table_pose"),
            ]
        )
        for optional_name in self._OPTIONAL_TERM_DIMS:
            if optional_name in self._term_slices:
                x_ctx_terms.append(self._slice(observations, optional_name))
        x_ctx_terms.extend([cube_pose_in_hand, fingertip_cube_forces])
        x_ctx = torch.cat(x_ctx_terms, dim=-1)
        coordination_input = torch.cat((x_ctx, mu_b, mu_h), dim=-1)
        c_t = self.coord_trunk(coordination_input)

        body_input = torch.cat((c_t, self._slice(observations, "humanoid_prior"), mu_b), dim=-1)
        body_features = self.body_head_backbone(body_input)
        raw_dz_b = self.body_residual_linear(body_features)
        dz_b = self.body_residual_scale * torch.tanh(raw_dz_b)

        hand_input = torch.cat(
            (c_t, self._slice(observations, "right_hand_prior"), mu_h, cube_pose_in_hand, fingertip_cube_forces),
            dim=-1,
        )
        hand_features = self.hand_head_backbone(hand_input)
        raw_dz_h = self.hand_residual_linear(hand_features)
        dz_h = self.hand_residual_scale * torch.tanh(raw_dz_h)

        self.last_body_residual = dz_b.detach()
        self.last_hand_residual = dz_h.detach()
        self.last_body_gate = None
        self.last_hand_gate = None
        return torch.cat((dz_b, dz_h), dim=-1)


class ActorCriticCoordResidual(nn.Module):
    """Locomanip actor-critic with a shared coordination bottleneck for residual latents."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int] | None = None,
        critic_hidden_dims: Sequence[int] = (1024, 512, 256),
        activation: str = "elu",
        init_noise_std: float | Sequence[float] | torch.Tensor = 0.22,
        noise_std_type: str = "log",
        coord_trunk_hidden_dims: Sequence[int] = (512, 256),
        body_head_hidden_dims: Sequence[int] = (256, 128),
        hand_head_hidden_dims: Sequence[int] = (256, 128),
        body_residual_scale: float = 1.0,
        hand_residual_scale: float = 1.0,
        fixed_log_std: bool = True,
        actor_obs_term_names: Sequence[str] = (),
        actor_obs_term_dims: Sequence[int | Sequence[int]] = (),
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__()
        del actor_hidden_dims

        self.actor = _CoordResidualActor(
            num_actor_obs,
            actor_obs_term_names=actor_obs_term_names,
            actor_obs_term_dims=actor_obs_term_dims,
            coord_trunk_hidden_dims=coord_trunk_hidden_dims,
            body_head_hidden_dims=body_head_hidden_dims,
            hand_head_hidden_dims=hand_head_hidden_dims,
            activation=activation,
            body_residual_scale=body_residual_scale,
            hand_residual_scale=hand_residual_scale,
        )
        if self.actor.num_actions != int(num_actions):
            raise ValueError(
                "ActorCriticCoordResidual action dim mismatch: "
                f"actor produces {self.actor.num_actions}, env expects {num_actions}."
            )
        self.critic = _build_mlp(int(num_critic_obs), critic_hidden_dims[:-1], int(critic_hidden_dims[-1]), activation)
        self.critic_output_activation = resolve_nn_activation(activation)
        self.value_head = nn.Linear(int(critic_hidden_dims[-1]), 1)

        if noise_std_type != "log":
            raise ValueError("ActorCriticCoordResidual expects noise_std_type='log'.")
        self.noise_std_type = noise_std_type
        self.fixed_log_std = bool(fixed_log_std)
        self.log_std = nn.Parameter(torch.zeros(int(num_actions)))
        std_tensor = torch.as_tensor(init_noise_std, device=self.log_std.device, dtype=self.log_std.dtype)
        if std_tensor.numel() == 1:
            std_tensor = std_tensor.expand_as(self.log_std)
        elif std_tensor.numel() != self.log_std.numel():
            raise ValueError(
                f"init_noise_std has {std_tensor.numel()} values, but {self.log_std.numel()} actions are expected."
            )
        std_tensor = std_tensor.view_as(self.log_std)
        with torch.no_grad():
            self.log_std.copy_(torch.log(torch.clamp(std_tensor, min=1.0e-6)))
        self.log_std.requires_grad_(not self.fixed_log_std)

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        return None

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        if self.distribution is None:
            raise RuntimeError("Distribution not updated. Call update_distribution first.")
        return self.distribution.mean

    @property
    def action_std(self):
        if self.distribution is None:
            raise RuntimeError("Distribution not updated. Call update_distribution first.")
        return self.distribution.stddev

    @property
    def entropy(self):
        if self.distribution is None:
            raise RuntimeError("Distribution not updated. Call update_distribution first.")
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor(observations)
        std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution not updated. Call update_distribution first.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.value_head(self.critic_output_activation(self.critic(critic_observations)))

    def load_state_dict(self, state_dict, strict: bool = True):
        gate_suffixes = (
            "body_gate_linear.weight",
            "body_gate_linear.bias",
            "hand_gate_linear.weight",
            "hand_gate_linear.bias",
        )
        if any(str(key).endswith(gate_suffixes) for key in state_dict):
            state_dict = {key: value for key, value in state_dict.items() if not str(key).endswith(gate_suffixes)}
        super().load_state_dict(state_dict, strict=strict)
        return True
