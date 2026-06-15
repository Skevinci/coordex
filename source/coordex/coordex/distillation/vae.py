from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.storage import RolloutStorage


def _build_mlp(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int, activation: nn.Module) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation)
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def _gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_log_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_std: torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL divergence between two diagonal Gaussians."""
    posterior_var = torch.exp(2.0 * posterior_log_std)
    prior_var = torch.exp(2.0 * prior_log_std)
    squared_mean_diff = (posterior_mean - prior_mean) ** 2
    kl = prior_log_std - posterior_log_std
    kl += (posterior_var + squared_mean_diff) / (2.0 * prior_var)
    kl -= 0.5
    return kl.sum(dim=-1).mean()


def _gaussian_kl_per_env(
    posterior_mean: torch.Tensor,
    posterior_log_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_log_std: torch.Tensor,
) -> torch.Tensor:
    """Closed-form KL divergence between two diagonal Gaussians for each sample."""
    posterior_var = torch.exp(2.0 * posterior_log_std)
    prior_var = torch.exp(2.0 * prior_log_std)
    squared_mean_diff = (posterior_mean - prior_mean) ** 2
    kl = prior_log_std - posterior_log_std
    kl += (posterior_var + squared_mean_diff) / (2.0 * prior_var)
    kl -= 0.5
    return kl.sum(dim=-1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the leading batch dimension using a boolean mask."""
    if values.ndim == 0:
        return values
    if mask.ndim > 1:
        mask = mask.view(mask.shape[0], -1).all(dim=-1)
    mask = mask.to(dtype=torch.bool, device=values.device)
    if values.shape[0] != mask.shape[0]:
        raise ValueError(
            f"Masked mean expected matching batch dimensions, got values={values.shape}, mask={mask.shape}."
        )
    if not torch.any(mask):
        return values.sum() * 0.0
    return values[mask].mean()


class SafeRolloutStorage(RolloutStorage):
    """Rollout storage that tracks per-env sample validity for distillation."""

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        super().__init__(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs_shape,
            privileged_obs_shape,
            actions_shape,
            rnd_state_shape=rnd_state_shape,
            device=device,
        )
        self.valid_masks = torch.ones(
            num_transitions_per_env,
            num_envs,
            1,
            dtype=torch.bool,
            device=self.device,
        )

    def add_transitions(self, transition: RolloutStorage.Transition):
        valid_mask = getattr(transition, "valid_mask", None)
        if valid_mask is None:
            self.valid_masks[self.step].fill_(True)
        else:
            self.valid_masks[self.step].copy_(
                valid_mask.view(-1, 1).to(device=self.device, dtype=torch.bool))
        super().add_transitions(transition)

    def generator(self):
        if self.training_type != "distillation":
            raise ValueError(
                "This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            if self.privileged_observations is not None:
                privileged_observations = self.privileged_observations[i]
            else:
                privileged_observations = self.observations[i]
            yield (
                self.observations[i],
                privileged_observations,
                self.actions[i],
                self.privileged_actions[i],
                self.dones[i],
                self.valid_masks[i],
            )


class StudentTeacherVAE(nn.Module):
    """Student-teacher policy with a VAE-style latent bottleneck for the student."""

    is_recurrent = False

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        *,
        activation: str,
        init_noise_std: float,
        teacher_hidden_dims: Tuple[int, ...],
        latent_dim: int,
        encoder_hidden_dims: Tuple[int, ...],
        decoder_hidden_dims: Tuple[int, ...],
        prior_hidden_dims: Tuple[int, ...],
        posterior_std_min: float = 1.0e-4,
        posterior_std_max: float = 3,
        prior_std_min: float = 1.0e-4,
        prior_std_max: float = 3,
        latent_dropout: float = 0.0,
        tanh_actions: bool = False,
        proprio_dim: int | None = None,
        **kwargs,
    ):
        super().__init__()

        action_loss_mask = kwargs.pop("action_loss_mask", None)
        wrist_action_dim = int(kwargs.pop("wrist_action_dim", 0) or 0)
        use_teacher_wrist = bool(kwargs.pop("use_teacher_wrist", False))
        mask_wrist_in_loss = bool(kwargs.pop("mask_wrist_in_loss", False))
        proprio_source_dim = kwargs.pop("proprio_source_dim", None)
        action_tail_dim = int(kwargs.pop("action_tail_dim", 0) or 0)
        action_keep_dim = int(kwargs.pop("action_keep_dim", 0) or 0)
        proprio_term_indices = tuple(
            int(value) for value in kwargs.pop("proprio_term_indices", ()))
        proprio_observation_term_dims = tuple(
            int(value) for value in kwargs.pop("proprio_observation_term_dims", ()))
        proprio_term_names = tuple(kwargs.pop("proprio_term_names", ()))
        proprio_flat_suffix_dim = int(
            kwargs.pop("proprio_flat_suffix_dim", 0) or 0)
        proprio_history_length = int(kwargs.pop("proprio_history_length", 1))
        proprio_frame_dim = int(kwargs.pop("proprio_frame_dim", 0) or 0)

        student_dims_alias = tuple(kwargs.pop("student_hidden_dims", ()))
        kwargs.pop("noise_std_type", None)
        if student_dims_alias:
            decoder_hidden_dims = tuple(student_dims_alias)

        self.loaded_teacher = False
        self.latent_dim = latent_dim
        self.posterior_std_min = posterior_std_min
        self.posterior_std_max = posterior_std_max
        self.prior_std_min = prior_std_min
        self.prior_std_max = prior_std_max
        self.proprio_history_length = max(1, proprio_history_length)
        self.proprio_dim = proprio_dim if proprio_dim is not None else num_student_obs
        self.proprio_frame_dim = proprio_frame_dim if proprio_frame_dim > 0 else 0
        self.proprio_source_dim = int(
            proprio_source_dim) if proprio_source_dim is not None else None
        self.tanh_actions = tanh_actions
        self.wrist_action_dim = max(
            0, min(int(wrist_action_dim), int(num_actions)))
        self.use_teacher_wrist = use_teacher_wrist and self.wrist_action_dim > 0
        self._cached_stats: Dict[str, torch.Tensor] | None = None
        self.action_tail_dim = max(0, action_tail_dim)
        self.action_keep_dim = max(0, action_keep_dim)
        self.proprio_flat_suffix_dim = max(0, proprio_flat_suffix_dim)
        self._explicit_proprio_term_slices: tuple[tuple[int, int], ...] = ()
        self._proprio_term_layout_dim: int | None = None
        self._proprio_observation_term_dims: tuple[int, ...] = ()
        self._proprio_term_offsets: tuple[int, ...] = ()
        self._proprio_term_names: tuple[str, ...] = ()
        self._explicit_proprio_block_start = 0
        self._explicit_proprio_block_end = 0
        self._explicit_proprio_block_is_suffix = False

        if self.proprio_frame_dim <= 0 and self.proprio_history_length > 1 and self.proprio_dim > 0:
            if self.proprio_dim % self.proprio_history_length == 0:
                self.proprio_frame_dim = self.proprio_dim // self.proprio_history_length
            else:
                self.proprio_frame_dim = self.proprio_dim

        if proprio_term_indices or proprio_observation_term_dims:
            if not proprio_term_indices or not proprio_observation_term_dims:
                raise ValueError(
                    "proprio_term_indices and proprio_observation_term_dims must be provided together."
                )
            if self.proprio_flat_suffix_dim > 0:
                raise ValueError(
                    "Cannot combine proprio_flat_suffix_dim with explicit proprio term slicing."
                )
            if self.action_tail_dim > 0 or self.action_keep_dim > 0:
                raise ValueError(
                    "Cannot combine explicit proprio term slicing with action_tail_dim/action_keep_dim slicing."
                )
            if self.proprio_history_length <= 0:
                raise ValueError("proprio_history_length must be >= 1.")

            if any(index < 0 for index in proprio_term_indices):
                raise ValueError("proprio_term_indices must be non-negative.")
            if any(dim < 0 for dim in proprio_observation_term_dims):
                raise ValueError(
                    "proprio_observation_term_dims must be non-negative.")
            if proprio_term_names and len(proprio_term_names) != len(proprio_observation_term_dims):
                raise ValueError(
                    "proprio_term_names length must match proprio_observation_term_dims length.")

            max_term_index = max(proprio_term_indices)
            if max_term_index >= len(proprio_observation_term_dims):
                raise ValueError(
                    "proprio_term_indices contain an index outside the range of "
                    f"proprio_observation_term_dims (len={len(proprio_observation_term_dims)})."
                )

            term_offsets: list[int] = [0]
            for dim in proprio_observation_term_dims:
                term_offsets.append(term_offsets[-1] + dim)
            if not term_offsets or term_offsets[-1] <= 0:
                raise ValueError(
                    "proprio_observation_term_dims must contain at least one positive term dimension.")

            self._proprio_observation_term_dims = tuple(
                proprio_observation_term_dims)
            self._proprio_term_offsets = tuple(term_offsets)
            if proprio_term_names:
                self._proprio_term_names = tuple(
                    str(value) for value in proprio_term_names)
            else:
                self._proprio_term_names = tuple(
                    f"term_{index}" for index in range(len(term_offsets) - 1))

            explicit_slices: list[tuple[int, int]] = []
            for term_index in proprio_term_indices:
                start = int(term_offsets[term_index])
                end = int(term_offsets[term_index + 1])
                if end <= start:
                    raise ValueError(
                        f"Term index {term_index} has zero width in proprio_observation_term_dims.")
                explicit_slices.append((start, end))

            selected_frame_dim = sum(
                end - start for start, end in explicit_slices)
            if selected_frame_dim <= 0:
                raise ValueError(
                    "Selected proprioception frame dim must be positive.")

            self._explicit_proprio_term_slices = tuple(explicit_slices)
            explicit_slices_sorted = tuple(
                sorted(explicit_slices, key=lambda item: item[0]))
            if explicit_slices_sorted:
                self._explicit_proprio_block_start = int(
                    explicit_slices_sorted[0][0])
                self._explicit_proprio_block_end = int(
                    explicit_slices_sorted[-1][1])
                self._explicit_proprio_block_is_suffix = False
                contiguous = True
                for index in range(1, len(explicit_slices_sorted)):
                    if explicit_slices_sorted[index][0] != explicit_slices_sorted[index - 1][1]:
                        contiguous = False
                        break
                if (
                    contiguous
                    and self._explicit_proprio_block_end >= self._explicit_proprio_block_start
                    and term_offsets[-1] > 0
                ):
                    # Select by term indices gives the tail block of a per-frame layout.
                    # In that case we can directly take the last proprio_frame_dim values from each frame.
                    self._explicit_proprio_block_is_suffix = (
                        self._explicit_proprio_block_end == term_offsets[-1]
                    )
            self._proprio_term_layout_dim = term_offsets[-1]
            self.proprio_frame_dim = selected_frame_dim
            self.proprio_dim = selected_frame_dim * \
                max(1, self.proprio_history_length)

            if proprio_dim is not None and int(proprio_dim) > 0 and self.proprio_dim != int(proprio_dim):
                raise ValueError(
                    "Explicit proprio term slicing expects proprio_dim to match selected_dim * proprio_history_length = "
                    f"{self.proprio_dim}, but configured proprio_dim={proprio_dim}."
                )
            if proprio_frame_dim > 0 and self.proprio_frame_dim != proprio_frame_dim:
                raise ValueError(
                    "Explicit proprio term slicing expects proprio_frame_dim to match selected frame dim "
                    f"{self.proprio_frame_dim}, but configured proprio_frame_dim={proprio_frame_dim}."
                )

        if self.action_tail_dim > 0 or self.action_keep_dim > 0:
            if self.action_tail_dim <= 0 or self.action_keep_dim <= 0:
                raise ValueError(
                    "Both action_tail_dim and action_keep_dim must be positive when action slicing is enabled."
                )
            if self.proprio_source_dim is None:
                raise ValueError(
                    "proprio_source_dim must be provided when action slicing is enabled.")
            if self.action_keep_dim > self.action_tail_dim:
                raise ValueError("action_keep_dim must be <= action_tail_dim.")
            if self.proprio_source_dim < self.action_tail_dim:
                raise ValueError(
                    "proprio_source_dim must be >= action_tail_dim.")
            expected_dim = self.proprio_source_dim - \
                self.action_tail_dim + self.action_keep_dim
            if expected_dim != int(self.proprio_dim):
                raise ValueError(
                    "Invalid split dims: expected output "
                    f"{expected_dim} from source={self.proprio_source_dim}, "
                    f"tail={self.action_tail_dim}, keep={self.action_keep_dim}, "
                    f"but proprio_dim={self.proprio_dim}."
                )

        if self.proprio_flat_suffix_dim > 0:
            if self.proprio_source_dim is not None and self.proprio_source_dim > 0:
                raise ValueError(
                    "Cannot combine proprio_flat_suffix_dim with proprio_source_dim.")
            if self.proprio_dim <= 0:
                self.proprio_dim = self.proprio_flat_suffix_dim
            elif self.proprio_dim != self.proprio_flat_suffix_dim:
                raise ValueError(
                    "Cannot combine proprio_flat_suffix_dim with different proprio_dim. "
                    f"Configured proprio_dim={self.proprio_dim}, proprio_flat_suffix_dim={self.proprio_flat_suffix_dim}."
                )

        if action_loss_mask is not None:
            self.action_loss_mask = torch.tensor(
                action_loss_mask, dtype=torch.float32)
        elif mask_wrist_in_loss and self.wrist_action_dim > 0:
            mask = torch.ones(num_actions, dtype=torch.float32)
            mask[: self.wrist_action_dim] = 0.0
            self.action_loss_mask = mask
        else:
            self.action_loss_mask = None

        activation_layer = resolve_nn_activation(activation)
        encoder_hidden_dims = tuple(encoder_hidden_dims)
        decoder_hidden_dims = tuple(decoder_hidden_dims)
        prior_hidden_dims = tuple(prior_hidden_dims)

        # Teacher branch (same as the RL actor head).
        teacher_hidden_dims = tuple(teacher_hidden_dims)
        teacher_layers = []
        prev_dim = num_teacher_obs
        teacher_layers.append(nn.Linear(prev_dim, teacher_hidden_dims[0]))
        teacher_layers.append(activation_layer)
        for index in range(len(teacher_hidden_dims)):
            if index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(
                    nn.Linear(teacher_hidden_dims[index], num_actions))
            else:
                teacher_layers.append(
                    nn.Linear(teacher_hidden_dims[index], teacher_hidden_dims[index + 1]))
                teacher_layers.append(activation_layer)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        # Student VAE components.
        self.encoder = _build_mlp(
            num_student_obs, encoder_hidden_dims, 2 * latent_dim, activation_layer)
        self.prior_net = _build_mlp(
            self.proprio_dim, prior_hidden_dims, 2 * latent_dim, activation_layer)
        self.decoder = _build_mlp(
            self.proprio_dim + latent_dim, decoder_hidden_dims, num_actions, activation_layer)

        self.latent_dropout = nn.Dropout(
            p=latent_dropout) if latent_dropout > 0.0 else nn.Identity()
        self.log_std = nn.Parameter(torch.full(
            (num_actions,), math.log(max(init_noise_std, 1.0e-6))))
        self.distribution: Normal | None = None

    def _split_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if self.proprio_dim is None or self.proprio_dim <= 0:
            return observations

        total_dim = observations.shape[-1]
        if self.proprio_flat_suffix_dim > 0:
            if observations.ndim > 2:
                flat_obs = observations.reshape(
                    *observations.shape[:-2], observations.shape[-2] * observations.shape[-1])
            else:
                flat_obs = observations
            if flat_obs.shape[-1] < self.proprio_flat_suffix_dim:
                raise ValueError(
                    "Proprioceptive flat-suffix slicing failed: "
                    f"observation dim={flat_obs.shape[-1]} is smaller than suffix_dim={self.proprio_flat_suffix_dim}."
                )
            return flat_obs[..., flat_obs.shape[-1] - self.proprio_flat_suffix_dim:]

        def _infer_history_sequence(
            data: torch.Tensor,
            *,
            expected_frame_dim: int | None = None,
            min_frame_dim: int | None = None,
        ) -> tuple[torch.Tensor, int]:
            """Reshape proprioception observations into [*, H, frame_dim]."""
            history_length = self.proprio_history_length
            if history_length <= 1:
                raise ValueError(
                    "History length must be >= 2 when inferring temporally stacked observations.")

            if data.ndim > 2:
                if data.shape[-2] != history_length:
                    raise ValueError(
                        "Explicit proprio term slicing failed: history axis does not match configured history length. "
                        f"Expected shape[-2]={history_length}, got {data.shape[-2]}."
                    )
                frame_dim = data.shape[-1]
                if min_frame_dim is not None and frame_dim < min_frame_dim:
                    raise ValueError(
                        "Explicit proprio term slicing failed: "
                        f"inferred frame_dim={frame_dim} is smaller than required minimum={min_frame_dim}."
                    )
                return data, frame_dim

            candidate_dim = data.shape[-1]
            candidates: list[tuple[int, int]] = []

            if candidate_dim > history_length and candidate_dim % history_length == 0:
                candidates.append(
                    (candidate_dim // history_length, candidate_dim))

            if (
                candidate_dim > history_length + 1
                and (candidate_dim - history_length) % history_length == 0
            ):
                candidates.append(((candidate_dim - history_length) //
                                  history_length, candidate_dim - history_length))

            if not candidates:
                raise ValueError(
                    "Explicit proprio term slicing failed: cannot infer frame dimension from observations.")

            if expected_frame_dim is not None and expected_frame_dim > 0:
                for frame_dim, usable_dim in candidates:
                    if frame_dim == expected_frame_dim:
                        trimmed = data[..., :usable_dim]
                        return trimmed.reshape(*trimmed.shape[:-1], history_length, frame_dim), frame_dim

            selected_frame_dim = None
            selected_usable_dim = None
            if min_frame_dim is not None and min_frame_dim > 0:
                filtered = [(frame_dim, usable_dim) for frame_dim,
                            usable_dim in candidates if frame_dim >= min_frame_dim]
                if filtered:
                    filtered.sort(key=lambda item: item[0])
                    selected_frame_dim, selected_usable_dim = filtered[0]
            if selected_frame_dim is None:
                selected_frame_dim, selected_usable_dim = sorted(
                    candidates, key=lambda item: item[0])[0]

            trimmed = data[..., :selected_usable_dim]
            return trimmed.reshape(*trimmed.shape[:-1], history_length, selected_frame_dim), selected_frame_dim

        if self._explicit_proprio_term_slices:
            if self.proprio_history_length <= 1:
                if self._explicit_proprio_block_is_suffix:
                    return observations[..., total_dim - self.proprio_dim:]
                chunks = []
                for start, end in self._explicit_proprio_term_slices:
                    if total_dim < end:
                        raise ValueError(
                            "Explicit proprio term slicing failed: "
                            f"observation dim={total_dim} is smaller than slice end={end}."
                        )
                    chunks.append(observations[..., start:end])
                return torch.cat(chunks, dim=-1)

            sequence, frame_dim = _infer_history_sequence(
                observations,
                expected_frame_dim=self.proprio_frame_dim,
                min_frame_dim=self._proprio_term_layout_dim,
            )
            if self._proprio_term_layout_dim is None or frame_dim < self._proprio_term_layout_dim:
                raise ValueError(
                    "Explicit proprio term slicing failed: "
                    f"inferred frame_dim={frame_dim} is smaller than configured term layout total="
                    f"{self._proprio_term_layout_dim}."
                )

            if self._explicit_proprio_block_is_suffix:
                proprio_sequence = sequence[..., -self.proprio_frame_dim:]
                return proprio_sequence.reshape(
                    *observations.shape[:-1], self.proprio_history_length * self.proprio_frame_dim
                )

            chunks = [sequence[..., start:end]
                      for start, end in self._explicit_proprio_term_slices]
            proprio_sequence = torch.cat(chunks, dim=-1)
            return proprio_sequence.reshape(*observations.shape[:-1], self.proprio_history_length * self.proprio_frame_dim)

        if self.action_tail_dim > 0 and self.action_keep_dim > 0:
            source_dim = int(
                self.proprio_source_dim if self.proprio_source_dim is not None else self.proprio_dim)
            source_dim = min(source_dim, total_dim)
            source = observations[..., total_dim - source_dim:]
            action_tail_dim = min(self.action_tail_dim, source.shape[-1])
            keep_dim = min(self.action_keep_dim, action_tail_dim)
            non_action_dim = source.shape[-1] - action_tail_dim
            return torch.cat((source[..., :non_action_dim], source[..., -keep_dim:]), dim=-1)

        proprio_dim = min(self.proprio_dim, total_dim)
        if self.proprio_history_length > 1 and self.proprio_frame_dim > 0:
            sequence, frame_dim = _infer_history_sequence(
                observations,
                expected_frame_dim=self.proprio_frame_dim,
                min_frame_dim=self.proprio_frame_dim,
            )
            if frame_dim >= self.proprio_frame_dim:
                start = frame_dim - self.proprio_frame_dim
                proprio_sequence = sequence[..., start:frame_dim]
                return proprio_sequence.reshape(
                    *observations.shape[:-1], self.proprio_history_length * self.proprio_frame_dim
                )

        return observations[..., total_dim - proprio_dim:]

    def _encode_from_safe(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.encoder(observations)
        return torch.chunk(stats, 2, dim=-1)

    def _prior_from_safe(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.prior_net(self._split_observations(observations))
        return torch.chunk(stats, 2, dim=-1)

    def _decode_from_safe(self, observations: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        decoder_input = torch.cat(
            [self._split_observations(observations), latent], dim=-1)
        decoded = self.decoder(decoder_input)
        return decoded

    def _mix_actions(self, observations: torch.Tensor, student_actions: torch.Tensor) -> torch.Tensor:
        if not self.use_teacher_wrist:
            return student_actions
        with torch.no_grad():
            teacher_actions = self.evaluate(observations)
        if teacher_actions.shape[-1] != student_actions.shape[-1]:
            return student_actions
        if self.wrist_action_dim >= student_actions.shape[-1]:
            return teacher_actions
        return torch.cat(
            (
                teacher_actions[..., : self.wrist_action_dim],
                student_actions[..., self.wrist_action_dim:],
            ),
            dim=-1,
        )

    def encode(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and log std of q(z | obs)."""
        posterior_mean, posterior_log_std = self._encode_from_safe(
            observations)
        posterior_log_std = torch.clamp(
            posterior_log_std,
            min=math.log(self.posterior_std_min),
            max=math.log(self.posterior_std_max),
        )
        return posterior_mean, posterior_log_std

    def prior(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return prior mean and log std of p(z | proprio)."""
        prior_mean, prior_log_std = self._prior_from_safe(observations)
        prior_log_std = torch.clamp(
            prior_log_std,
            min=math.log(self.prior_std_min),
            max=math.log(self.prior_std_max),
        )
        return prior_mean, prior_log_std

    def decode(self, observations: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        latent = self.latent_dropout(latent)
        return self._decode_from_safe(observations, latent)

    @staticmethod
    def _sample_gaussian(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        std = torch.exp(log_std)
        eps = torch.randn_like(mean)
        return mean + std * eps

    def latent_forward(self, observations: torch.Tensor, sample_latent: bool = True) -> dict[str, torch.Tensor]:
        posterior_mean, posterior_log_std = self.encode(observations)
        if sample_latent:
            latent = self._sample_gaussian(posterior_mean, posterior_log_std)
        else:
            latent = posterior_mean
        prior_mean, prior_log_std = self.prior(observations)
        action_mean = self._decode_from_safe(observations, latent)
        stats = {
            "action_mean": action_mean,
            "posterior_mean": posterior_mean,
            "posterior_log_std": posterior_log_std,
            "prior_mean": prior_mean,
            "prior_log_std": prior_log_std,
        }
        self._cached_stats = stats
        return stats

    def reset(self, dones=None, hidden_states=None):
        del dones, hidden_states

    def update_distribution(self, observations: torch.Tensor):
        stats = self.latent_forward(observations, sample_latent=True)
        action_std = torch.exp(self.log_std).expand_as(stats["action_mean"])
        action_std = torch.clamp(action_std, min=1.0e-6)
        action_mean = stats["action_mean"]
        self.distribution = Normal(action_mean, action_std)

    def act(self, observations: torch.Tensor):
        self.update_distribution(observations)
        assert self.distribution is not None
        actions = self.distribution.mean
        actions = self._mix_actions(observations, actions)

        if self.tanh_actions:
            actions = torch.tanh(actions)
        return actions

    def act_inference(self, observations: torch.Tensor):
        stats = self.latent_forward(observations, sample_latent=False)
        return self._mix_actions(observations, stats["action_mean"])

    def evaluate(self, teacher_observations: torch.Tensor):
        with torch.no_grad():
            return self.teacher(teacher_observations)

    @property
    def action_mean(self):
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call act() or update_distribution() first.")
        return self.distribution.mean

    @property
    def action_std(self):
        if self.distribution is not None:
            return self.distribution.stddev
        return torch.exp(self.log_std)

    @property
    def entropy(self):
        if self.distribution is None:
            raise RuntimeError(
                "Distribution not initialized. Call act() or update_distribution() first.")
        return self.distribution.entropy().sum(dim=-1)

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        del dones

    # type: ignore[override]
    def load_state_dict(self, state_dict, strict: bool = True):
        """Load parameters for teacher and student networks.

        Returns:
            bool: True if this resumes distillation, False if only the teacher weights were loaded.
        """
        if any("actor." in key for key in state_dict.keys()):
            teacher_state_dict = {
                key.replace("actor.", ""): value for key, value in state_dict.items() if "actor." in key
            }
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            return False

        nn.Module.load_state_dict(self, state_dict, strict=strict)
        self.loaded_teacher = True
        self.teacher.eval()
        return True


class DistillationVAE(Distillation):
    """Distillation algorithm with additional VAE-based losses."""

    def __init__(
        self,
        policy,
        *,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        action_loss_coef: float = 1.0,
        regularization_loss_coef: float = 0.1,
        kl_loss_coef: float = 0.01,
        kl_anneal_iters: int = 0,
        adaptive_kl_loss_coef: bool = False,
        adaptive_kl_target: float = 20.0,
        adaptive_kl_eta: float = 1.0e-4,
        adaptive_kl_min_coef: float = 1.0e-5,
        adaptive_kl_max_coef: float = 5.0e-3,
        kl_final_coef: float | None = None,
        kl_anneal_start_iter: int = 0,
        kl_anneal_end_iter: int = 0,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        super().__init__(
            policy,
            num_learning_epochs=num_learning_epochs,
            gradient_length=gradient_length,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            loss_type=loss_type,
            device=device,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.action_loss_coef = action_loss_coef
        self.regularization_loss_coef = regularization_loss_coef
        self.kl_loss_coef = kl_loss_coef
        self.kl_anneal_iters = kl_anneal_iters
        self.adaptive_kl_loss_coef = adaptive_kl_loss_coef
        self.adaptive_kl_target = adaptive_kl_target
        self.adaptive_kl_eta = adaptive_kl_eta
        self.adaptive_kl_min_coef = adaptive_kl_min_coef
        self.adaptive_kl_max_coef = adaptive_kl_max_coef
        self.kl_final_coef = kl_final_coef
        self.kl_anneal_start_iter = kl_anneal_start_iter
        self.kl_anneal_end_iter = kl_anneal_end_iter

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        student_obs_shape,
        teacher_obs_shape,
        actions_shape,
    ):
        self.storage = SafeRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
        )

    def process_env_step(self, rewards, dones, infos):
        valid_mask = getattr(self.transition, "valid_mask", None)
        if valid_mask is None:
            valid_mask = torch.ones_like(dones, dtype=torch.bool)
            self.transition.valid_mask = valid_mask
        else:
            valid_mask = valid_mask.to(device=dones.device, dtype=torch.bool)

        if not torch.all(valid_mask):
            rewards = rewards.clone()
            dones = dones.clone()
            rewards[~valid_mask] = 0.0
            dones[~valid_mask] = 1

        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def _current_kl_coef(self) -> float:
        if self.adaptive_kl_loss_coef:
            return self.kl_loss_coef
        if (
            self.kl_final_coef is not None
            and self.kl_anneal_end_iter > self.kl_anneal_start_iter
            and self.kl_anneal_start_iter >= 0
        ):
            if self.num_updates < self.kl_anneal_start_iter:
                return self.kl_loss_coef
            if self.num_updates >= self.kl_anneal_end_iter:
                return self.kl_final_coef
            span = self.kl_anneal_end_iter - self.kl_anneal_start_iter
            frac = (self.num_updates - self.kl_anneal_start_iter) / float(span)
            return (1.0 - frac) * self.kl_loss_coef + frac * self.kl_final_coef
        if self.kl_anneal_iters <= 0:
            return self.kl_loss_coef
        progress = min(1.0, self.num_updates / float(self.kl_anneal_iters))
        return self.kl_loss_coef * progress

    def _maybe_update_adaptive_kl(self, kl_value: float) -> None:
        if not self.adaptive_kl_loss_coef:
            return
        updated_coef = self.kl_loss_coef * \
            math.exp(self.adaptive_kl_eta *
                     (kl_value - self.adaptive_kl_target))
        if not math.isfinite(updated_coef):
            updated_coef = self.adaptive_kl_min_coef
        updated_coef = min(
            max(updated_coef, self.adaptive_kl_min_coef), self.adaptive_kl_max_coef)
        self.kl_loss_coef = float(updated_coef)

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0.0
        mean_reg_loss = 0.0
        mean_kl_loss = 0.0
        mean_total_loss = 0.0
        total_loss_accumulator: torch.Tensor | None = None
        accumulation_count = 0
        cnt = 0
        post_sum = 0.0
        post_count = 0.0
        post_max = float("-inf")
        post_min = float("inf")
        prior_sum = 0.0
        prior_count = 0.0
        prior_max = float("-inf")
        prior_min = float("inf")

        current_kl = self._current_kl_coef()

        for _ in range(self.num_learning_epochs):
            prev_posterior_mean: torch.Tensor | None = None
            prev_dones: torch.Tensor | None = None
            prev_valid_mask: torch.Tensor | None = None
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for batch in self.storage.generator():
                if len(batch) == 6:
                    obs, _, _, privileged_actions, dones, valid_mask = batch
                else:
                    obs, _, _, privileged_actions, dones = batch
                    valid_mask = torch.ones_like(dones, dtype=torch.bool)
                valid_mask = valid_mask.view(-1).to(
                    device=obs.device, dtype=torch.bool)
                # alive_mask = (dones.view(-1) == 0)
                stats = self.policy.latent_forward(obs, sample_latent=True)
                action_mean = stats["action_mean"]
                posterior_mean = stats["posterior_mean"]
                posterior_log_std = stats["posterior_log_std"]
                prior_mean = stats["prior_mean"]
                prior_log_std = stats["prior_log_std"]
                if posterior_log_std.numel() > 0:
                    valid_posterior_log_std = posterior_log_std[valid_mask]
                    if valid_posterior_log_std.numel() > 0:
                        post_sum += valid_posterior_log_std.sum().item()
                        post_count += valid_posterior_log_std.numel()
                        post_max = max(
                            post_max, valid_posterior_log_std.max().item())
                        post_min = min(
                            post_min, valid_posterior_log_std.min().item())
                if prior_log_std.numel() > 0:
                    valid_prior_log_std = prior_log_std[valid_mask]
                    if valid_prior_log_std.numel() > 0:
                        prior_sum += valid_prior_log_std.sum().item()
                        prior_count += valid_prior_log_std.numel()
                        prior_max = max(
                            prior_max, valid_prior_log_std.max().item())
                        prior_min = min(
                            prior_min, valid_prior_log_std.min().item())

                action_mask = getattr(self.policy, "action_loss_mask", None)
                if action_mask is not None:
                    mask = action_mask.to(action_mean.device)
                    if mask.ndim == 1:
                        mask = mask.unsqueeze(0)
                    if mask.shape[-1] == action_mean.shape[-1]:
                        diff = (action_mean - privileged_actions) * mask
                        denom = mask.sum(dim=-1).clamp(min=1.0)
                        behavior_loss = _masked_mean(
                            diff.pow(2).sum(dim=-1) / denom, valid_mask)
                    else:
                        per_env_behavior = (
                            action_mean - privileged_actions).pow(2).mean(dim=-1)
                        behavior_loss = _masked_mean(
                            per_env_behavior, valid_mask)
                else:
                    per_env_behavior = (
                        action_mean - privileged_actions).pow(2).mean(dim=-1)
                    behavior_loss = _masked_mean(per_env_behavior, valid_mask)

                if prev_posterior_mean is None or prev_dones is None or prev_valid_mask is None:
                    regularization_loss = torch.zeros(
                        (), device=obs.device, dtype=behavior_loss.dtype)
                else:
                    # Only compare consecutive timesteps within the same env (skip resets).
                    pair_mask = valid_mask & prev_valid_mask & (
                        ~prev_dones.view(-1).bool())
                    if pair_mask.shape[0] != posterior_mean.shape[0]:
                        regularization_loss = torch.zeros(
                            (), device=obs.device, dtype=behavior_loss.dtype)
                    else:
                        if torch.any(pair_mask):
                            regularization_loss = (
                                posterior_mean[pair_mask] - prev_posterior_mean[pair_mask]).pow(2).mean()
                        else:
                            regularization_loss = torch.zeros(
                                (), device=obs.device, dtype=behavior_loss.dtype)

                kl_loss = _masked_mean(
                    _gaussian_kl_per_env(
                        posterior_mean, posterior_log_std, prior_mean, prior_log_std),
                    valid_mask,
                )

                total_loss = (
                    self.action_loss_coef * behavior_loss
                    + self.regularization_loss_coef * regularization_loss
                    + current_kl * kl_loss
                )

                if total_loss_accumulator is None:
                    total_loss_accumulator = total_loss
                else:
                    total_loss_accumulator = total_loss_accumulator + total_loss
                accumulation_count += 1
                mean_behavior_loss += behavior_loss.item()
                mean_reg_loss += regularization_loss.item()
                mean_kl_loss += kl_loss.item()
                mean_total_loss += total_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    if total_loss_accumulator is None or accumulation_count <= 0:
                        continue
                    self.optimizer.zero_grad()
                    (total_loss_accumulator / float(accumulation_count)).backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(
                            self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    total_loss_accumulator = None
                    accumulation_count = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

                prev_posterior_mean = posterior_mean.detach()
                prev_dones = dones.detach()
                prev_valid_mask = valid_mask.detach()

        if total_loss_accumulator is not None and accumulation_count > 0:
            self.optimizer.zero_grad()
            (total_loss_accumulator / float(accumulation_count)).backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.policy.detach_hidden_states()
            total_loss_accumulator = None
            accumulation_count = 0

        if cnt > 0:
            mean_behavior_loss /= cnt
            mean_reg_loss /= cnt
            mean_kl_loss /= cnt
            mean_total_loss /= cnt
        self._maybe_update_adaptive_kl(mean_kl_loss)
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()
        if post_count > 0:
            post_mean = post_sum / post_count
        else:
            post_mean = 0.0
            post_max = 0.0
            post_min = 0.0
        if prior_count > 0:
            prior_mean = prior_sum / prior_count
        else:
            prior_mean = 0.0
            prior_max = 0.0
            prior_min = 0.0

        return {
            "behavior": mean_behavior_loss,
            "regularization": mean_reg_loss,
            "kl": mean_kl_loss,
            "total": mean_total_loss,
            "posterior_log_std_max": post_max,
            "posterior_log_std_min": post_min,
            "posterior_log_std_mean": post_mean,
            "prior_log_std_max": prior_max,
            "prior_log_std_min": prior_min,
            "prior_log_std_mean": prior_mean,
            "kl_coef": self.kl_loss_coef,
        }
