from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import torch

from isaaclab.envs.mdp import JointPositionAction, JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from coordex.robots.g1_wuji import WUJI_RH_ACTIVE_JOINT_NAMES
from coordex.distillation.vae import StudentTeacherVAE
from coordex.tasks.locomanip.constants import MODE12_JOINT_ORDER

from .observations import (
    TRACKING_PRIOR_NOISE_CFG,
    clear_prior_obs_step_cache,
    reset_tracking_prior_history,
    right_hand_proprio,
    tracking_prior_history,
)


_BODY_PRIOR_FRAME_DIM_WITH_BASE_LIN_VEL = 93
_BODY_PRIOR_FRAME_DIM_NO_BASE_LIN_VEL = 90
_BODY_PRIOR_FRAME_DIMS = (
    _BODY_PRIOR_FRAME_DIM_WITH_BASE_LIN_VEL, _BODY_PRIOR_FRAME_DIM_NO_BASE_LIN_VEL)


def _distill_cfg(
    *,
    proprio_dim: int,
    latent_dim: int,
    posterior_std_max: float,
    prior_std_max: float,
    proprio_frame_dim: int = 0,
    proprio_history_length: int = 1,
    proprio_source_dim: int = 0,
    action_tail_dim: int = 0,
    action_keep_dim: int = 0,
):
    return SimpleNamespace(
        init_noise_std=0.05,
        noise_std_type="log",
        student_hidden_dims=[256, 256, 128],
        teacher_hidden_dims=(1024, 512, 256),
        activation="elu",
        tanh_actions=False,
        latent_dim=latent_dim,
        encoder_hidden_dims=(512, 256, 128),
        decoder_hidden_dims=(512, 256, 128),
        prior_hidden_dims=(512, 256, 128),
        posterior_std_min=1.0e-4,
        posterior_std_max=posterior_std_max,
        prior_std_min=1.0e-4,
        prior_std_max=prior_std_max,
        latent_dropout=0.0,
        proprio_dim=proprio_dim,
        proprio_frame_dim=proprio_frame_dim,
        proprio_history_length=proprio_history_length,
        proprio_source_dim=proprio_source_dim,
        action_tail_dim=action_tail_dim,
        action_keep_dim=action_keep_dim,
        wrist_action_dim=0,
        use_teacher_wrist=False,
        mask_wrist_in_loss=False,
    )


_BODY_PRIOR_CFG_WITH_BASE_LIN_VEL = _distill_cfg(
    proprio_dim=465,
    latent_dim=16,
    posterior_std_max=5.0,
    prior_std_max=5.0,
    proprio_frame_dim=93,
    proprio_history_length=5,
)
_BODY_PRIOR_CFG_NO_BASE_LIN_VEL = _distill_cfg(
    proprio_dim=450,
    latent_dim=16,
    posterior_std_max=5.0,
    prior_std_max=5.0,
    proprio_frame_dim=90,
    proprio_history_length=5,
)
_WUJI_HAND_ACTIVE_CFG = _distill_cfg(
    proprio_dim=66,
    latent_dim=12,
    posterior_std_max=10.0,
    prior_std_max=10.0,
    proprio_source_dim=72,
    action_tail_dim=26,
    action_keep_dim=20,
)
_WUJI_HAND_KINEMATIC_WRIST_CFG = _distill_cfg(
    proprio_dim=66,
    latent_dim=12,
    posterior_std_max=10.0,
    prior_std_max=10.0,
)


_HAND_PRIOR_VARIANT_SPECS = {
    "wuji_floating_active": {
        "policy_cfg": _WUJI_HAND_ACTIVE_CFG,
        "default_joint_names": tuple(WUJI_RH_ACTIVE_JOINT_NAMES),
        "wrist_body_name": "right_palm_link",
        "proprio_dim": 66,
        "dummy_action_dim": 6,
    },
    "wuji_floating_kinematic_wrist": {
        "policy_cfg": _WUJI_HAND_KINEMATIC_WRIST_CFG,
        "default_joint_names": tuple(WUJI_RH_ACTIVE_JOINT_NAMES),
        "wrist_body_name": "right_palm_link",
        "proprio_dim": 66,
        "dummy_action_dim": 0,
    },
}


def _get_hand_prior_variant_spec(hand_prior_variant: str) -> dict[str, object]:
    try:
        return _HAND_PRIOR_VARIANT_SPECS[hand_prior_variant]
    except KeyError as exc:
        supported = ", ".join(sorted(_HAND_PRIOR_VARIANT_SPECS))
        raise ValueError(
            f"Unsupported hand_prior_variant '{hand_prior_variant}'. Expected one of: {supported}.") from exc


def _normalize_hand_joint_names(
    hand_joint_names: tuple[str, ...] | list[str] | None,
    *,
    variant: str,
) -> tuple[str, ...]:
    if hand_joint_names:
        return tuple(str(name) for name in hand_joint_names)
    spec = _get_hand_prior_variant_spec(variant)
    return tuple(spec["default_joint_names"])


def get_hand_prior_default_wrist_body_name(hand_prior_variant: str) -> str:
    return str(_get_hand_prior_variant_spec(hand_prior_variant)["wrist_body_name"])


def get_hand_prior_expected_dims(
    hand_prior_variant: str,
    hand_joint_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[int, int]:
    spec = _get_hand_prior_variant_spec(hand_prior_variant)
    joint_names = _normalize_hand_joint_names(
        hand_joint_names, variant=hand_prior_variant)
    proprio_dim = int(spec["proprio_dim"])
    action_dim = int(spec["dummy_action_dim"]) + len(joint_names)
    return proprio_dim, action_dim


class _ObsNormalizer:
    def __init__(self, mean: torch.Tensor, var: torch.Tensor, eps: float = 1.0e-8, clip: float | None = None):
        self.mean = mean
        self.var = var
        self.eps = eps
        self.clip = clip

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = (obs - self.mean) / torch.sqrt(self.var + self.eps)
        if self.clip is not None:
            normalized = torch.clamp(normalized, -self.clip, self.clip)
        return normalized


def _infer_prior_input_dim_from_checkpoint(ckpt_path: str | None) -> int | None:
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict, _ = _extract_state_dict(checkpoint)
    state_dict = _strip_module_prefix(state_dict)
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            continue
        if key.endswith("prior_net.0.weight") or ".prior_net.0.weight" in key:
            return int(value.shape[1])
    return None


def _infer_body_prior_frame_dim(proprio_dim: int) -> int:
    for frame_dim in _BODY_PRIOR_FRAME_DIMS:
        if proprio_dim > 0 and proprio_dim % frame_dim == 0:
            return frame_dim
    supported = ", ".join(str(value) for value in _BODY_PRIOR_FRAME_DIMS)
    raise RuntimeError(
        "Body prior proprio dim must be an integer multiple of one supported single-frame proprio size "
        f"({supported}); got {proprio_dim}."
    )


def _get_body_distill_policy_cfg(ckpt_path: str | None):
    input_dim = _infer_prior_input_dim_from_checkpoint(ckpt_path)
    if input_dim is not None:
        frame_dim = _infer_body_prior_frame_dim(input_dim)
        if frame_dim == _BODY_PRIOR_FRAME_DIM_NO_BASE_LIN_VEL:
            return _BODY_PRIOR_CFG_NO_BASE_LIN_VEL
    return _BODY_PRIOR_CFG_WITH_BASE_LIN_VEL


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _extract_state_dict(checkpoint: object) -> tuple[dict[str, torch.Tensor], dict]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "policy_state_dict", "state_dict", "actor_critic_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint, checkpoint
    raise ValueError(
        "Unsupported checkpoint format; expected a state_dict or a checkpoint dict.")


def _find_module_prefix(state_dict: dict[str, torch.Tensor], module_name: str) -> str | None:
    token = f".{module_name}."
    for key in state_dict:
        if token in key:
            return key.split(token)[0]
        if key.startswith(f"{module_name}."):
            return ""
    return None


def _strip_key_prefix(key: str, prefix: str | None) -> str:
    if not prefix:
        return key
    prefix_with_dot = f"{prefix}."
    if key.startswith(prefix_with_dot):
        return key[len(prefix_with_dot):]
    return key


def _coerce_obs_norm_state(state: object | None) -> dict | None:
    if state is None:
        return None
    if isinstance(state, dict):
        if isinstance(state.get("state_dict"), dict):
            return state["state_dict"]
        return state
    if hasattr(state, "state_dict"):
        try:
            extracted = state.state_dict()
        except Exception:
            extracted = None
        if isinstance(extracted, dict):
            return extracted
    attr_pairs = (
        ("mean", "var", "mean", "var"),
        ("running_mean", "running_var", "running_mean", "running_var"),
        ("_mean", "_var", "_mean", "_var"),
        ("mean", "std", "mean", "std"),
        ("running_mean", "running_std", "running_mean", "running_std"),
        ("_mean", "_std", "_mean", "_std"),
    )
    for mean_key, var_key, out_mean, out_var in attr_pairs:
        if hasattr(state, mean_key) and hasattr(state, var_key):
            return {out_mean: getattr(state, mean_key), out_var: getattr(state, var_key)}
    return None


def _infer_obs_norm_dim(state: dict | None) -> int:
    if not state:
        return 0
    for key in (
        "mean",
        "running_mean",
        "_mean",
        "var",
        "running_var",
        "variance",
        "_var",
        "std",
        "running_std",
        "_std",
    ):
        if key in state:
            return int(torch.as_tensor(state[key]).view(-1).numel())
    return 0


def _iter_obs_norm_states(checkpoint: dict, _depth: int = 0):
    if _depth > 2:
        return
    candidate_keys = (
        "obs_normalizer_state_dict",
        "obs_normalizer",
        "normalizer_state_dict",
        "obs_norm_state_dict",
        "normalizer",
        "obs_rms",
        "obs_rms_state_dict",
        "obs_rms_state",
    )
    for key in candidate_keys:
        if key in checkpoint:
            state = _coerce_obs_norm_state(checkpoint[key])
            if state:
                yield key, state
    nested_keys = (
        "runner_state_dict",
        "runner_state",
        "algorithm_state_dict",
        "algo_state",
        "train_state",
    )
    for nested_key in nested_keys:
        nested = checkpoint.get(nested_key)
        if isinstance(nested, dict):
            yield from _iter_obs_norm_states(nested, _depth=_depth + 1)


def _select_obs_norm_state(checkpoint: dict, expected_dim: int) -> dict | None:
    candidates: list[tuple[int, dict]] = []
    for _, state in _iter_obs_norm_states(checkpoint):
        dim = _infer_obs_norm_dim(state)
        if dim > 0:
            candidates.append((dim, state))
    if not candidates:
        return None
    if expected_dim > 0:
        viable = [entry for entry in candidates if entry[0] >= expected_dim]
        if viable:
            candidates = viable
    candidates.sort(key=lambda entry: entry[0], reverse=True)
    return candidates[0][1]


def _resolve_obs_norm_hparams(state: dict) -> tuple[float, float]:
    if "clip" in state:
        clip = state["clip"]
    elif "clip_range" in state:
        clip = state["clip_range"]
    else:
        clip = None
    default_eps = 1.0e-2
    eps = float(state["eps"] if "eps" in state else state.get(
        "epsilon") or default_eps)
    clip_value = float(clip) if clip is not None else None
    return eps, clip_value


def _slice_norm_tensor(
    tensor: torch.Tensor,
    expected_dim: int,
    split_spec: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    if tensor.numel() == expected_dim:
        return tensor
    if tensor.numel() < expected_dim:
        raise ValueError(
            f"Obs normalizer dimension mismatch: expected {expected_dim}, got {tensor.numel()}."
        )
    if split_spec is not None:
        src_dim, tail_dim, keep_dim = split_spec
        if (
            src_dim > 0
            and tail_dim > 0
            and keep_dim > 0
            and keep_dim <= tail_dim
            and tensor.numel() >= src_dim
            and (src_dim - tail_dim + keep_dim) == expected_dim
        ):
            source = tensor[-src_dim:]
            return torch.cat((source[: src_dim - tail_dim], source[-keep_dim:]))
    return tensor[-expected_dim:]


def _prepare_obs_norm_state(
    state: dict,
    expected_dim: int,
    split_spec: tuple[int, int, int] | None = None,
) -> dict:
    prepared: dict = {}
    slice_keys = (
        "mean",
        "running_mean",
        "_mean",
        "var",
        "running_var",
        "variance",
        "_var",
        "std",
        "running_std",
        "_std",
    )
    for key, value in state.items():
        if key in slice_keys:
            tensor = torch.as_tensor(value, dtype=torch.float32).view(-1)
            prepared[key] = _slice_norm_tensor(
                tensor, expected_dim, split_spec)
        elif isinstance(value, (torch.Tensor, list, tuple)):
            prepared[key] = torch.as_tensor(value, dtype=torch.float32)
        else:
            prepared[key] = value
    return prepared


def _build_rsl_obs_normalizer(
    state: dict,
    expected_dim: int,
    device: torch.device,
    split_spec: tuple[int, int, int] | None = None,
) -> object | None:
    try:
        from rsl_rl.modules import EmpiricalNormalization  # type: ignore
    except Exception as exc:
        print(f"[WARN] rsl_rl EmpiricalNormalization not available: {exc}")
        return None

    eps, clip_value = _resolve_obs_norm_hparams(state)
    prepared_state = _prepare_obs_norm_state(
        state, expected_dim, split_spec=split_spec)

    init_kwargs: dict[str, object] = {}
    try:
        sig = inspect.signature(EmpiricalNormalization.__init__)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        params = sig.parameters
        if "shape" in params:
            init_kwargs["shape"] = [expected_dim]
        elif "size" in params:
            init_kwargs["size"] = [expected_dim]
        if "epsilon" in params:
            init_kwargs["epsilon"] = eps
        elif "eps" in params:
            init_kwargs["eps"] = eps
        if "clip" in params and clip_value is not None:
            init_kwargs["clip"] = clip_value
        if "until" in params:
            init_kwargs["until"] = 1.0e8

    try:
        normalizer = EmpiricalNormalization(**init_kwargs)
    except Exception:
        normalizer = EmpiricalNormalization([expected_dim])
    normalizer.to(device)
    try:
        target_state = normalizer.state_dict()

        def _reshape_to_target(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            if value.shape == target.shape:
                return value
            if value.numel() == target.numel():
                return value.view(target.shape)
            if value.ndim == 1 and target.ndim == 2:
                if target.shape[0] == 1 and value.shape[0] == target.shape[1]:
                    return value.unsqueeze(0)
                if target.shape[1] == 1 and value.shape[0] == target.shape[0]:
                    return value.unsqueeze(1)
            if value.ndim == 2 and target.ndim == 1:
                if value.shape[0] == 1 and value.shape[1] == target.shape[0]:
                    return value.squeeze(0)
                if value.shape[1] == 1 and value.shape[0] == target.shape[0]:
                    return value.squeeze(1)
            return value

        adjusted_state: dict[str, object] = {}
        for key, value in prepared_state.items():
            if key in target_state and isinstance(value, torch.Tensor):
                adjusted = _reshape_to_target(value, target_state[key])
                if adjusted.shape != target_state[key].shape:
                    continue
                adjusted_state[key] = adjusted
            else:
                adjusted_state[key] = value

        normalizer.load_state_dict(adjusted_state, strict=False)
    except Exception as exc:
        print(f"[WARN] Failed to load obs normalizer state: {exc}")
    if hasattr(normalizer, "clip"):
        normalizer.clip = clip_value
    if hasattr(normalizer, "epsilon"):
        normalizer.epsilon = eps
    if hasattr(normalizer, "eps"):
        normalizer.eps = eps
    normalizer.eval()
    return normalizer


def _build_obs_normalizer(
    checkpoint: dict,
    expected_dim: int,
    device: torch.device,
    *,
    enable: bool,
    mode: str = "simple",
    split_spec: tuple[int, int, int] | None = None,
) -> _ObsNormalizer | object | None:
    if not enable:
        return None
    state = _coerce_obs_norm_state(
        _select_obs_norm_state(checkpoint, expected_dim))
    if not state:
        print(
            "[WARN] Obs normalizer state not found; prior inputs will be unnormalized.")
        return None
    if mode == "rsl_rl":
        normalizer = _build_rsl_obs_normalizer(
            state, expected_dim, device, split_spec=split_spec
        )
        if normalizer is not None:
            return normalizer
        print("[WARN] Falling back to simple obs normalizer.")

    def _maybe_tensor(value: object) -> torch.Tensor | None:
        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        return tensor.view(-1)

    mean = _maybe_tensor(state.get("mean") or state.get(
        "running_mean") or state.get("_mean"))
    var = _maybe_tensor(state.get("var") or state.get(
        "running_var") or state.get("variance") or state.get("_var"))
    std = _maybe_tensor(state.get("std") or state.get(
        "running_std") or state.get("_std"))
    if var is None and std is not None:
        var = std.pow(2)

    if mean is None or var is None:
        return None

    mean = _slice_norm_tensor(mean, expected_dim, split_spec)
    var = _slice_norm_tensor(var, expected_dim, split_spec)

    eps, clip_value = _resolve_obs_norm_hparams(state)
    return _ObsNormalizer(mean, var, eps=eps, clip=clip_value)


def _load_student_prior(
    ckpt_path: str,
    *,
    policy_cfg: object,
    num_actions: int,
    device: torch.device,
    enable_obs_norm: bool,
    obs_norm_mode: str = "simple",
    activation_override: str | None = None,
    tanh_actions_override: bool | None = None,
) -> tuple[StudentTeacherVAE, _ObsNormalizer | None, int, int, int]:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict, checkpoint_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_module_prefix(state_dict)

    prior_prefix = _find_module_prefix(state_dict, "prior_net")
    decoder_prefix = _find_module_prefix(state_dict, "decoder")
    if prior_prefix is None or decoder_prefix is None:
        raise ValueError(
            f"Checkpoint '{ckpt_path}' does not contain prior/decoder weights.")

    prior_module = f"{prior_prefix}.prior_net" if prior_prefix else "prior_net"
    decoder_module = f"{decoder_prefix}.decoder" if decoder_prefix else "decoder"

    policy_cfg = getattr(policy_cfg, "policy", policy_cfg)
    proprio_dim = int(getattr(policy_cfg, "proprio_dim", 0) or 0)
    latent_dim = int(getattr(policy_cfg, "latent_dim", 0) or 0)
    if proprio_dim <= 0 or latent_dim <= 0:
        raise ValueError(
            "Distillation policy config must define positive proprio_dim and latent_dim.")

    activation = activation_override or getattr(
        policy_cfg, "activation", "elu")
    if tanh_actions_override is None:
        tanh_actions = bool(getattr(policy_cfg, "tanh_actions", False))
    else:
        tanh_actions = tanh_actions_override

    proprio_source_dim = int(getattr(policy_cfg, "proprio_source_dim", 0) or 0)
    action_tail_dim = int(getattr(policy_cfg, "action_tail_dim", 0) or 0)
    action_keep_dim = int(getattr(policy_cfg, "action_keep_dim", 0) or 0)
    split_spec: tuple[int, int, int] | None = None
    if proprio_source_dim > 0 and action_tail_dim > 0 and action_keep_dim > 0:
        expected_split_dim = proprio_source_dim - action_tail_dim + action_keep_dim
        if expected_split_dim != proprio_dim:
            raise ValueError(
                "Policy cfg split spec is inconsistent: "
                f"proprio_source_dim({proprio_source_dim}) - action_tail_dim({action_tail_dim}) + "
                f"action_keep_dim({action_keep_dim}) = {expected_split_dim}, expected proprio_dim={proprio_dim}."
            )
        split_spec = (proprio_source_dim, action_tail_dim, action_keep_dim)

    policy_kwargs: dict[str, object] = {}
    student_hidden_dims = tuple(getattr(policy_cfg, "student_hidden_dims", ()))
    if student_hidden_dims:
        policy_kwargs["student_hidden_dims"] = student_hidden_dims
    noise_std_type = getattr(policy_cfg, "noise_std_type", None)
    if noise_std_type is not None:
        policy_kwargs["noise_std_type"] = noise_std_type
    for optional_key in ("wrist_action_dim", "use_teacher_wrist", "mask_wrist_in_loss"):
        if hasattr(policy_cfg, optional_key):
            policy_kwargs[optional_key] = getattr(policy_cfg, optional_key)

    policy = StudentTeacherVAE(
        num_student_obs=proprio_dim,
        num_teacher_obs=proprio_dim,
        num_actions=num_actions,
        activation=activation,
        init_noise_std=float(getattr(policy_cfg, "init_noise_std", 1.0)),
        teacher_hidden_dims=tuple(
            getattr(policy_cfg, "teacher_hidden_dims", ())),
        latent_dim=latent_dim,
        encoder_hidden_dims=tuple(
            getattr(policy_cfg, "encoder_hidden_dims", ())),
        decoder_hidden_dims=tuple(
            getattr(policy_cfg, "decoder_hidden_dims", ())),
        prior_hidden_dims=tuple(getattr(policy_cfg, "prior_hidden_dims", ())),
        posterior_std_min=float(
            getattr(policy_cfg, "posterior_std_min", 1.0e-4)),
        posterior_std_max=float(getattr(policy_cfg, "posterior_std_max", 3.0)),
        prior_std_min=float(getattr(policy_cfg, "prior_std_min", 1.0e-4)),
        prior_std_max=float(getattr(policy_cfg, "prior_std_max", 3.0)),
        latent_dropout=float(getattr(policy_cfg, "latent_dropout", 0.0)),
        tanh_actions=tanh_actions,
        proprio_dim=proprio_dim,
        **policy_kwargs,
    )

    filtered_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prior_module + "."):
            filtered_state[_strip_key_prefix(key, prior_prefix)] = value
        elif key.startswith(decoder_module + "."):
            filtered_state[_strip_key_prefix(key, decoder_prefix)] = value
        elif key.endswith("log_std"):
            filtered_state[_strip_key_prefix(key, prior_prefix)] = value

    policy.load_state_dict(filtered_state, strict=False)
    policy.to(device)
    policy.eval()
    policy.use_teacher_wrist = False

    obs_normalizer = _build_obs_normalizer(
        checkpoint_dict,
        proprio_dim,
        device,
        enable=enable_obs_norm,
        mode=obs_norm_mode,
        split_spec=split_spec,
    )
    return policy, obs_normalizer, proprio_dim, num_actions, latent_dim


class _PriorPolicy:
    def __init__(
        self,
        ckpt_path: str,
        *,
        policy_cfg: object,
        action_dim: int,
        device: torch.device,
        use_obs_norm: bool,
        obs_norm_mode: str = "simple",
        activation_override: str | None = None,
        tanh_actions_override: bool | None = None,
    ):
        (
            self.policy,
            self.normalizer,
            self.proprio_dim,
            self.action_dim,
            self.latent_dim,
        ) = _load_student_prior(
            ckpt_path,
            policy_cfg=policy_cfg,
            num_actions=action_dim,
            device=device,
            enable_obs_norm=use_obs_norm,
            obs_norm_mode=obs_norm_mode,
            activation_override=activation_override,
            tanh_actions_override=tanh_actions_override,
        )

    @torch.no_grad()
    def infer_with_delta(self, obs: torch.Tensor, latent_delta: torch.Tensor | None) -> torch.Tensor:
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        latent_mean, _ = self.policy.prior(obs)
        if latent_delta is not None:
            if latent_delta.shape[-1] != latent_mean.shape[-1]:
                raise RuntimeError(
                    f"Latent residual dim mismatch: expected {latent_mean.shape[-1]}, got {latent_delta.shape[-1]}."
                )
            latent_mean = latent_mean + latent_delta
        actions = self.policy.decode(obs, latent_mean)
        if self.policy.tanh_actions:
            actions = torch.tanh(actions)
        return actions

    @torch.no_grad()
    def prior_mean(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        latent_mean, _ = self.policy.prior(obs)
        return latent_mean


class ResidualLatentAction(ActionTerm):
    """Residual latent action composed with frozen body/hand priors."""

    cfg: "ResidualLatentActionCfg"

    def __init__(self, cfg: "ResidualLatentActionCfg", env):
        self._action_dim = 0
        super().__init__(cfg, env)
        self._env = env
        self._joint_action = JointPositionAction(cfg, env)
        # Expose joint action buffers for event terms that update offsets.
        self._offset = self._joint_action._offset
        self._joint_ids = self._joint_action._joint_ids
        self._body_obs_cfg = SceneEntityCfg(
            "robot", joint_names=MODE12_JOINT_ORDER, preserve_order=True)
        self._hand_prior_variant = str(
            getattr(cfg, "hand_prior_variant", "wuji_floating_kinematic_wrist"))
        self._hand_joint_names_cfg = _normalize_hand_joint_names(
            getattr(cfg, "hand_joint_names", ()), variant=self._hand_prior_variant)
        self._hand_wrist_body_name = str(
            getattr(cfg, "hand_wrist_body_name", "") or get_hand_prior_default_wrist_body_name(
                self._hand_prior_variant)
        )
        self._hand_obs_cfg = SceneEntityCfg("robot", joint_names=list(
            self._hand_joint_names_cfg), preserve_order=True)

        body_joint_ids, _ = self._asset.find_joints(
            MODE12_JOINT_ORDER, preserve_order=True)
        hand_joint_ids, hand_joint_names = self._asset.find_joints(
            list(self._hand_joint_names_cfg), preserve_order=True)
        if len(hand_joint_ids) != len(self._hand_joint_names_cfg):
            missing = sorted(set(self._hand_joint_names_cfg) -
                             set(hand_joint_names))
            raise RuntimeError(
                f"Missing right-hand action joints on '{self._asset.name}': {missing}")
        self._body_obs_cfg.joint_ids = list(body_joint_ids)
        self._hand_obs_cfg.joint_ids = list(hand_joint_ids)
        self._hand_joint_ids = torch.tensor(
            hand_joint_ids, dtype=torch.long, device=self.device)

        joint_to_action = self._build_joint_to_action_index()
        self._body_action_ids = joint_to_action[torch.as_tensor(
            body_joint_ids, device=self.device)]
        self._hand_action_ids = joint_to_action[self._hand_joint_ids]

        self._body_policy_cfg = _get_body_distill_policy_cfg(
            cfg.body_prior_checkpoint)
        hand_variant_spec = _get_hand_prior_variant_spec(
            self._hand_prior_variant)
        self._hand_policy_cfg = hand_variant_spec["policy_cfg"]
        self._hand_dummy_action_dim = int(
            hand_variant_spec["dummy_action_dim"])
        self._expected_hand_proprio_dim, self._expected_hand_action_dim = get_hand_prior_expected_dims(
            self._hand_prior_variant,
            self._hand_joint_names_cfg,
        )

        self._body_prior = self._maybe_load_prior(
            cfg.body_prior_checkpoint,
            cfg.body_activation,
            cfg.body_tanh_actions,
            cfg.body_use_obs_norm,
            cfg.body_obs_norm_mode,
            policy_cfg=self._body_policy_cfg,
            action_dim=self._body_action_ids.numel(),
        )
        if self._body_prior is not None:
            self._body_prior_frame_dim = _infer_body_prior_frame_dim(
                self._body_prior.proprio_dim)
            self._body_prior_include_base_lin_vel = (
                self._body_prior_frame_dim == _BODY_PRIOR_FRAME_DIM_WITH_BASE_LIN_VEL
            )
        else:
            self._body_prior_frame_dim = 0
            self._body_prior_include_base_lin_vel = True

        self._hand_prior = self._maybe_load_prior(
            cfg.hand_prior_checkpoint,
            cfg.hand_activation,
            cfg.hand_tanh_actions,
            cfg.hand_use_obs_norm,
            cfg.hand_obs_norm_mode,
            policy_cfg=self._hand_policy_cfg,
            action_dim=self._expected_hand_action_dim,
        )
        if self._hand_prior is not None and self._hand_prior.proprio_dim != self._expected_hand_proprio_dim:
            raise RuntimeError(
                f"Floating active hand prior expects proprio dim {self._expected_hand_proprio_dim}, "
                f"got {self._hand_prior.proprio_dim}."
            )

        if self._body_prior is not None and (self._body_action_ids < 0).any():
            raise RuntimeError(
                "Residual action term could not resolve joint indices for base priors.")
        if self._hand_prior is not None and (self._hand_action_ids < 0).any():
            raise RuntimeError(
                "Residual action term could not resolve joint indices for base priors.")
        if self._hand_prior is None:
            self._hand_action_ids = torch.empty(
                0, dtype=torch.long, device=self.device)
            self._hand_joint_ids = torch.empty(
                0, dtype=torch.long, device=self.device)
            self._hand_obs_cfg.joint_ids = []

        self._body_history_length = (
            self._body_prior.proprio_dim // self._body_prior_frame_dim
            if self._body_prior and self._body_prior_frame_dim > 0
            else 0
        )

        self._hand_action_alpha = max(
            0.0, min(1.0, float(getattr(cfg, "hand_action_moving_average", 1.0))))
        self._hand_prev_targets = None
        self._hand_prior_last_action = None
        if self._hand_action_ids.numel() > 0:
            self._hand_prev_targets = torch.zeros(
                (env.num_envs, self._hand_action_ids.numel()), device=self.device
            )
            self._hand_prev_targets[:] = self._asset.data.joint_pos[:,
                                                                    self._hand_joint_ids]
            self._hand_prior_last_action = torch.zeros(
                (env.num_envs, self._hand_joint_ids.numel()), device=self.device
            )

        self._body_latent_dim = self._body_prior.latent_dim if self._body_prior is not None else 0
        self._hand_latent_dim = self._hand_prior.latent_dim if self._hand_prior is not None else 0
        self._action_dim = self._body_latent_dim + self._hand_latent_dim
        if self._action_dim <= 0:
            raise RuntimeError(
                "Residual latent action requires at least one prior checkpoint.")

        self._raw_actions = torch.zeros(
            (env.num_envs, self.action_dim), device=self.device)
        self._combined_action = torch.zeros(
            (env.num_envs, self._joint_action.action_dim), device=self.device)
        self._last_action = torch.zeros_like(self._combined_action)
        self._env._locomanip_last_action = self._last_action

    def _build_joint_to_action_index(self) -> torch.Tensor:
        joint_ids = getattr(self._joint_action, "_joint_ids", slice(None))
        if isinstance(joint_ids, slice):
            joint_ids = torch.arange(
                self._asset.num_joints, device=self.device, dtype=torch.long)
        else:
            joint_ids = torch.as_tensor(
                joint_ids, device=self.device, dtype=torch.long)
        joint_to_action = torch.full(
            (self._asset.num_joints,), -1, device=self.device, dtype=torch.long)
        joint_to_action[joint_ids] = torch.arange(
            joint_ids.numel(), device=self.device, dtype=torch.long)
        return joint_to_action

    def _maybe_load_prior(
        self,
        ckpt_path: str | None,
        activation: str,
        tanh_actions: bool,
        use_obs_norm: bool,
        obs_norm_mode: str,
        policy_cfg: object,
        action_dim: int,
    ) -> _PriorPolicy | None:
        if not ckpt_path:
            return None
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Prior checkpoint not found: {ckpt_path}")
        return _PriorPolicy(
            ckpt_path,
            policy_cfg=policy_cfg,
            action_dim=action_dim,
            device=self.device,
            use_obs_norm=use_obs_norm,
            obs_norm_mode=obs_norm_mode,
            activation_override=activation,
            tanh_actions_override=tanh_actions,
        )

    def _hand_targets_to_actions(
        self,
        hand_targets: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = getattr(self._joint_action, "_scale", None)
        offset = getattr(self._joint_action, "_offset", None)
        if not isinstance(scale, torch.Tensor) or not isinstance(offset, torch.Tensor):
            raise RuntimeError(
                "Joint action scale/offset tensors are required to map hand targets to actions.")
        if env_ids is None:
            hand_scale = scale[:, self._hand_action_ids]
            hand_offset = offset[:, self._hand_action_ids]
        else:
            hand_scale = scale[env_ids][:, self._hand_action_ids]
            hand_offset = offset[env_ids][:, self._hand_action_ids]
        safe_scale = torch.where(
            hand_scale.abs() < 1.0e-8, torch.ones_like(hand_scale), hand_scale)
        return (hand_targets - hand_offset) / safe_scale

    def _hand_actions_to_targets(
        self,
        hand_actions: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = getattr(self._joint_action, "_scale", None)
        offset = getattr(self._joint_action, "_offset", None)
        if not isinstance(scale, torch.Tensor) or not isinstance(offset, torch.Tensor):
            raise RuntimeError(
                "Joint action scale/offset tensors are required to map hand actions to targets.")
        if env_ids is None:
            hand_scale = scale[:, self._hand_action_ids]
            hand_offset = offset[:, self._hand_action_ids]
        else:
            hand_scale = scale[env_ids][:, self._hand_action_ids]
            hand_offset = offset[env_ids][:, self._hand_action_ids]
        return hand_offset + hand_scale * hand_actions

    def _decode_hand_output_to_targets(self, decoded_hand: torch.Tensor) -> torch.Tensor:
        if decoded_hand.shape[-1] != self._expected_hand_action_dim:
            raise RuntimeError(
                "Hand prior decoded action dim mismatch: expected "
                f"{self._expected_hand_action_dim}, "
                f"got {decoded_hand.shape[-1]}."
            )
        hand_actions = decoded_hand[:, self._hand_dummy_action_dim:]
        return self._hand_actions_to_targets(hand_actions)

    def _get_body_prior_obs(self) -> torch.Tensor:
        if self._body_prior is None or self._body_history_length <= 0:
            return torch.zeros((self.num_envs, 0), device=self.device)
        return tracking_prior_history(
            self._env,
            asset_cfg=self._body_obs_cfg,
            noise_cfg=TRACKING_PRIOR_NOISE_CFG,
            history_length=self._body_history_length,
            include_base_lin_vel=self._body_prior_include_base_lin_vel,
        )

    def _get_hand_prior_obs(self) -> torch.Tensor:
        if self._hand_prior is None:
            return torch.zeros((self.num_envs, 0), device=self.device)
        obs = right_hand_proprio(
            self._env,
            asset_cfg=self._hand_obs_cfg,
            wrist_body_name=self._hand_wrist_body_name,
        )
        return obs

    @torch.no_grad()
    def get_body_prior_mean(self) -> torch.Tensor:
        if self._body_prior is None:
            return torch.zeros((self.num_envs, 0), device=self.device)
        return self._body_prior.prior_mean(self._get_body_prior_obs())

    @torch.no_grad()
    def get_hand_prior_mean(self) -> torch.Tensor:
        if self._hand_prior is None:
            return torch.zeros((self.num_envs, 0), device=self.device)
        return self._hand_prior.prior_mean(self._get_hand_prior_obs())

    def _apply_hand_ema(self, hand_targets: torch.Tensor) -> torch.Tensor:
        if self._hand_prev_targets is None:
            return hand_targets
        target = hand_targets
        if self._hand_action_alpha < 1.0:
            target = self._hand_action_alpha * target + \
                (1.0 - self._hand_action_alpha) * self._hand_prev_targets
        limits = self._asset.data.soft_joint_pos_limits[:,
                                                        self._hand_joint_ids]
        target = torch.clamp(target, limits[..., 0], limits[..., 1])
        self._hand_prev_targets[:] = target
        return target

    def _compute_joint_action(self, residual_latent: torch.Tensor) -> torch.Tensor:
        self._combined_action.zero_()
        if self._body_prior is not None:
            body_obs = self._get_body_prior_obs()
            body_delta = residual_latent[:,
                                         : self._body_latent_dim] if self._body_latent_dim > 0 else None
            body_action = self._body_prior.infer_with_delta(
                body_obs, body_delta)
            if body_action.shape[-1] != self._body_action_ids.numel():
                raise RuntimeError(
                    f"Body prior action dim mismatch: expected {self._body_action_ids.numel()}, got {body_action.shape[-1]}."
                )
            self._combined_action[:, self._body_action_ids] = body_action

        if self._hand_prior is not None:
            hand_obs = self._get_hand_prior_obs()
            hand_delta = residual_latent[:,
                                         self._body_latent_dim: self._body_latent_dim + self._hand_latent_dim]
            if self._hand_latent_dim <= 0:
                hand_delta = None
            hand_action = self._hand_prior.infer_with_delta(
                hand_obs, hand_delta)
            hand_targets = self._decode_hand_output_to_targets(hand_action)
            if self._hand_action_alpha < 1.0 or self._hand_prev_targets is not None:
                hand_targets = self._apply_hand_ema(hand_targets)
            hand_locomanip_action = self._hand_targets_to_actions(hand_targets)
            self._combined_action[:,
                                  self._hand_action_ids] = hand_locomanip_action
            if self._hand_prior_last_action is not None:
                self._hand_prior_last_action[:] = hand_locomanip_action

        if self.cfg.base_scale != 1.0:
            self._combined_action.mul_(float(self.cfg.base_scale))
        if self.cfg.clip_actions:
            self._combined_action.clamp_(
                self.cfg.clip_action_min, self.cfg.clip_action_max)
        return self._combined_action

    @property
    def action_dim(self) -> int:
        return getattr(self, "_action_dim", 0)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._joint_action.processed_actions

    def capture_rsi_aux_state(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        env_ids_t = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long)
        state = {
            "raw_actions": self._raw_actions[env_ids_t].detach().clone(),
        }
        if self._hand_prev_targets is not None:
            state["hand_prev_targets"] = self._hand_prev_targets[env_ids_t].detach().clone()
        return state

    def restore_rsi_aux_state(
        self,
        env_ids: torch.Tensor,
        *,
        raw_actions: torch.Tensor | None = None,
        hand_prev_targets: torch.Tensor | None = None,
    ) -> None:
        env_ids_t = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        if raw_actions is not None:
            raw_actions_t = torch.as_tensor(
                raw_actions, device=self.device, dtype=self._raw_actions.dtype)
            if raw_actions_t.ndim != 2 or raw_actions_t.shape[0] != env_ids_t.numel():
                raise ValueError(
                    f"Expected raw_actions with shape [N, D], got {tuple(raw_actions_t.shape)} for "
                    f"{env_ids_t.numel()} envs."
                )
            if raw_actions_t.shape[1] != self._raw_actions.shape[1]:
                raise ValueError(
                    f"Raw action dim mismatch: expected {self._raw_actions.shape[1]}, got {raw_actions_t.shape[1]}."
                )
            self._raw_actions[env_ids_t] = raw_actions_t

        if hand_prev_targets is not None and self._hand_prev_targets is not None:
            hand_prev_targets_t = torch.as_tensor(
                hand_prev_targets,
                device=self.device,
                dtype=self._hand_prev_targets.dtype,
            )
            if hand_prev_targets_t.ndim != 2 or hand_prev_targets_t.shape[0] != env_ids_t.numel():
                raise ValueError(
                    f"Expected hand_prev_targets with shape [N, D], got {tuple(hand_prev_targets_t.shape)} for "
                    f"{env_ids_t.numel()} envs."
                )
            if hand_prev_targets_t.shape[1] != self._hand_prev_targets.shape[1]:
                raise ValueError(
                    "Hand previous target dim mismatch: expected "
                    f"{self._hand_prev_targets.shape[1]}, got {hand_prev_targets_t.shape[1]}."
                )
            self._hand_prev_targets[env_ids_t] = hand_prev_targets_t

        if self._combined_action.shape == self._last_action.shape:
            self._combined_action[env_ids_t] = self._last_action[env_ids_t]

    def process_actions(self, actions: torch.Tensor):
        residual_latent = actions.view(self.num_envs, self.action_dim)
        settle_buf = getattr(self._env, "_no_demo_rsi_settle_steps", None)
        settle_mask = None
        if isinstance(settle_buf, torch.Tensor):
            settle_mask = settle_buf > 0
            if settle_mask.any():
                residual_latent = residual_latent.clone()
                residual_latent[settle_mask] = 0.0
                settle_buf[settle_mask] = torch.clamp(
                    settle_buf[settle_mask] - 1, min=0)

        self._raw_actions.copy_(residual_latent)
        joint_actions = self._compute_joint_action(residual_latent)
        if settle_mask is not None and settle_mask.any():
            joint_actions = joint_actions.clone()
            joint_actions[settle_mask] = 0.0
        self._last_action.copy_(joint_actions.detach())
        self._joint_action.process_actions(joint_actions)

    def apply_actions(self):
        self._joint_action.apply_actions()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._joint_action.reset(env_ids)
        if env_ids is None:
            env_ids_t = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long)
        elif isinstance(env_ids, slice):
            env_ids_t = torch.arange(
                self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        else:
            env_ids_t = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        self._raw_actions[env_ids_t] = 0.0
        self._combined_action[env_ids_t] = 0.0
        self._last_action[env_ids_t] = 0.0
        if hasattr(self._env, "_locomanip_last_action"):
            self._env._locomanip_last_action[env_ids_t] = 0.0
        prev_joint_rate_actions = getattr(
            self._env, "_locomanip_prev_joint_action_rate_action", None)
        if isinstance(prev_joint_rate_actions, torch.Tensor) and prev_joint_rate_actions.shape == self._last_action.shape:
            prev_joint_rate_actions[env_ids_t] = 0.0

        if self._body_prior is not None and self._body_history_length > 0:
            clear_prior_obs_step_cache(self._env)
            reset_tracking_prior_history(
                self._env,
                asset_cfg=self._body_obs_cfg,
                noise_cfg=TRACKING_PRIOR_NOISE_CFG,
                history_length=self._body_history_length,
                env_ids=env_ids_t,
                include_base_lin_vel=self._body_prior_include_base_lin_vel,
            )

        if self._hand_prev_targets is not None:
            hand_targets = self._asset.data.joint_pos[env_ids_t][:,
                                                                 self._hand_joint_ids]
            self._hand_prev_targets[env_ids_t] = hand_targets

            if self._hand_prior_last_action is not None:
                self._hand_prior_last_action[env_ids_t] = 0.0

        if hasattr(self._env, "_locomanip_last_action"):
            self._env._locomanip_last_action[env_ids_t] = self._last_action[env_ids_t]


@configclass
class ResidualLatentActionCfg(JointPositionActionCfg):
    class_type = ResidualLatentAction

    body_prior_checkpoint: str | None = None
    hand_prior_checkpoint: str | None = None
    hand_joint_names: tuple[str, ...] = tuple(WUJI_RH_ACTIVE_JOINT_NAMES)
    hand_wrist_body_name: str = "right_palm_link"
    hand_prior_variant: str = "wuji_floating_kinematic_wrist"
    body_obs_norm_mode: str = "simple"
    hand_obs_norm_mode: str = "simple"
    body_activation: str = "elu"
    hand_activation: str = "elu"
    body_tanh_actions: bool = False
    hand_tanh_actions: bool = False
    body_use_obs_norm: bool = True
    hand_use_obs_norm: bool = True
    hand_action_moving_average: float = 1.0
    base_scale: float = 1.0
    clip_actions: bool = False
    clip_action_min: float = -1.0
    clip_action_max: float = 1.0
