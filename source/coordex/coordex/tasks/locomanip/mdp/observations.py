from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp import base_ang_vel, base_lin_vel, joint_pos_rel, joint_vel_rel
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, subtract_frame_transforms

from coordex.tasks.locomanip.constants import RIGHT_HAND_TIP_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


TRACKING_PRIOR_NOISE_CFG: dict[str, tuple[float, float]] = {
    "lin_vel": (-0.5, 0.5),
    "ang_vel": (-0.2, 0.2),
    "joint_pos": (-0.01, 0.01),
    "joint_vel": (-0.5, 0.5),
}

_TRACKING_PRIOR_LIN_VEL_DIM = 3
_TRACKING_PRIOR_ANG_VEL_DIM = 3


def _current_step_tag(env: ManagerBasedRLEnv) -> int:
    return int(getattr(env, "common_step_counter", -1))


def _prior_obs_step_cache(env: ManagerBasedRLEnv) -> dict:
    cache = getattr(env, "_locomanip_prior_obs_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(env, "_locomanip_prior_obs_cache", cache)
    return cache


def clear_prior_obs_step_cache(env: ManagerBasedRLEnv) -> None:
    setattr(env, "_locomanip_prior_obs_cache", {})


def _get_cached_step_tensor(env: ManagerBasedRLEnv, cache_key: tuple) -> torch.Tensor | None:
    cache = getattr(env, "_locomanip_prior_obs_cache", None)
    if not isinstance(cache, dict):
        return None
    cached = cache.get(cache_key)
    if not isinstance(cached, tuple) or len(cached) != 2:
        return None
    step_tag, tensor = cached
    if step_tag != _current_step_tag(env) or not isinstance(tensor, torch.Tensor):
        return None
    return tensor


def _set_cached_step_tensor(env: ManagerBasedRLEnv, cache_key: tuple, tensor: torch.Tensor) -> torch.Tensor:
    _prior_obs_step_cache(env)[cache_key] = (_current_step_tag(env), tensor)
    return tensor


def _joint_ids_cache_key(joint_ids: list[int] | slice | torch.Tensor | None) -> tuple | None:
    if joint_ids is None:
        return None
    if isinstance(joint_ids, slice):
        return ("slice", joint_ids.start, joint_ids.stop, joint_ids.step)
    if isinstance(joint_ids, torch.Tensor):
        joint_ids = joint_ids.tolist()
    return tuple(int(joint_id) for joint_id in joint_ids)


def _resolve_env_ids_tensor(
    env: ManagerBasedRLEnv,
    env_ids,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    target_device = env.device if device is None else device
    if env_ids is None:
        return torch.arange(env.num_envs, device=target_device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=target_device, dtype=torch.long)[env_ids]
    return torch.as_tensor(env_ids, device=target_device, dtype=torch.long)


def _noise_cfg_cache_key(noise_cfg: dict[str, tuple[float, float]] | None) -> tuple | None:
    if noise_cfg is None:
        return None
    return tuple(
        (str(name), tuple(float(value) for value in values))
        for name, values in sorted(noise_cfg.items(), key=lambda item: item[0])
    )


def _tracking_prior_frame_cache_key(
    asset_cfg: SceneEntityCfg,
    noise_cfg: dict[str, tuple[float, float]] | None,
    include_base_lin_vel: bool = True,
) -> tuple:
    return (
        "tracking_prior_frame",
        asset_cfg.name,
        _joint_ids_cache_key(asset_cfg.joint_ids),
        _noise_cfg_cache_key(noise_cfg),
        bool(include_base_lin_vel),
    )


def _tracking_prior_history_cache_key(
    asset_cfg: SceneEntityCfg,
    noise_cfg: dict[str, tuple[float, float]] | None,
    history_length: int,
    include_base_lin_vel: bool = True,
) -> tuple:
    return (
        "tracking_prior_history",
        asset_cfg.name,
        _joint_ids_cache_key(asset_cfg.joint_ids),
        _noise_cfg_cache_key(noise_cfg),
        int(history_length),
        bool(include_base_lin_vel),
    )


def _right_hand_prior_cache_key(asset_cfg: SceneEntityCfg, wrist_body_name: str) -> tuple:
    return (
        "right_hand_prior",
        asset_cfg.name,
        _joint_ids_cache_key(asset_cfg.joint_ids),
        wrist_body_name,
    )


def _tracking_prior_history_state(
    env: ManagerBasedRLEnv,
    cache_key: tuple,
    history_length: int,
    frame_obs: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    history_state = getattr(
        env, "_locomanip_tracking_prior_history_state", None)
    if not isinstance(history_state, dict):
        history_state = {}
        setattr(env, "_locomanip_tracking_prior_history_state", history_state)

    state = history_state.get(cache_key)
    frame_dim = int(frame_obs.shape[-1])
    if (
        not isinstance(state, dict)
        or int(state.get("history_length", -1)) != history_length
        or int(state.get("frame_dim", -1)) != frame_dim
        or not isinstance(state.get("skip_update_once"), torch.Tensor)
    ):
        state = {
            "history_length": history_length,
            "frame_dim": frame_dim,
            "frame_history": torch.zeros(
                (env.num_envs, history_length,
                 frame_dim), device=frame_obs.device, dtype=frame_obs.dtype
            ),
            "initialized": torch.zeros((env.num_envs,), device=frame_obs.device, dtype=torch.bool),
            "skip_update_once": torch.zeros((env.num_envs,), device=frame_obs.device, dtype=torch.bool),
            "last_step": None,
        }
        history_state[cache_key] = state
    return state


def _add_uniform_noise(tensor: torch.Tensor, noise_range: tuple[float, float] | None) -> torch.Tensor:
    if noise_range is None:
        return tensor
    low, high = noise_range
    if low == 0.0 and high == 0.0:
        return tensor
    noise = (high - low) * torch.rand_like(tensor) + low
    return tensor + noise


def _resolve_body_indices(
    env: ManagerBasedRLEnv,
    asset: Articulation,
    body_names: list[str],
    cache_attr: str,
) -> torch.Tensor:
    cached = getattr(env, cache_attr, None)
    if isinstance(cached, torch.Tensor):
        return cached
    body_ids, body_found = asset.find_bodies(body_names, preserve_order=True)
    if len(body_ids) != len(body_names):
        missing = sorted(set(body_names) - set(body_found))
        raise RuntimeError(f"Missing bodies on '{asset.name}': {missing}")
    ids = torch.as_tensor(
        body_ids, device=asset.data.body_pos_w.device, dtype=torch.long)
    setattr(env, cache_attr, ids)
    return ids


def _resolve_body_index(env: ManagerBasedRLEnv, asset: Articulation, body_name: str, cache_attr: str) -> int:
    cached = getattr(env, cache_attr, None)
    if cached is not None:
        return int(cached)
    body_ids, _ = asset.find_bodies([body_name], preserve_order=True)
    if len(body_ids) == 0:
        raise RuntimeError(
            f"Body '{body_name}' not found on asset '{asset.name}'.")
    body_id = int(body_ids[0])
    setattr(env, cache_attr, body_id)
    return body_id


def _resolve_action_ids(
    env: ManagerBasedRLEnv,
    joint_ids: list[int] | slice | torch.Tensor,
    device: torch.device,
    *,
    action_term_name: str = "joint_pos",
) -> slice | torch.Tensor:
    if isinstance(joint_ids, slice):
        return slice(None)

    if isinstance(joint_ids, torch.Tensor):
        joint_ids_list = joint_ids.tolist()
    else:
        joint_ids_list = list(joint_ids)

    if len(joint_ids_list) == 0:
        return torch.zeros((0,), device=device, dtype=torch.long)

    cache = getattr(env, "_locomanip_action_id_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(env, "_locomanip_action_id_cache", cache)

    cache_key = (action_term_name, tuple(int(j) for j in joint_ids_list))
    cached = cache.get(cache_key)
    if isinstance(cached, torch.Tensor):
        return cached

    term = env.action_manager.get_term(action_term_name)
    if hasattr(term, "_joint_action"):
        joint_ids_all = getattr(term._joint_action, "_joint_ids", slice(None))
    else:
        joint_ids_all = getattr(term, "_joint_ids", slice(None))

    if isinstance(joint_ids_all, slice):
        action_ids = torch.as_tensor(
            joint_ids_list, device=device, dtype=torch.long)
    else:
        joint_ids_all = torch.as_tensor(
            joint_ids_all, device=device, dtype=torch.long)
        max_joint_id = int(joint_ids_all.max().item())
        joint_to_action = torch.full(
            (max_joint_id + 1,), -1, device=device, dtype=torch.long)
        joint_to_action[joint_ids_all] = torch.arange(
            joint_ids_all.numel(), device=device, dtype=torch.long)
        joint_ids_tensor = torch.as_tensor(
            joint_ids_list, device=device, dtype=torch.long)
        if int(joint_ids_tensor.max().item()) >= joint_to_action.numel():
            action_ids = torch.full_like(joint_ids_tensor, -1)
        else:
            action_ids = joint_to_action[joint_ids_tensor]

    cache[cache_key] = action_ids
    return action_ids


def _get_last_joint_actions(
    env: ManagerBasedRLEnv,
    joint_ids: list[int] | slice | torch.Tensor,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    action_term_name: str = "joint_pos",
) -> torch.Tensor:
    actions_full = getattr(env, "_locomanip_last_action", None)
    if actions_full is None:
        return torch.zeros((env.num_envs, action_dim), device=device, dtype=dtype)

    action_ids = _resolve_action_ids(
        env, joint_ids, device, action_term_name=action_term_name)
    if isinstance(action_ids, slice):
        if actions_full.shape[-1] == action_dim:
            return actions_full
        return torch.zeros((env.num_envs, action_dim), device=device, dtype=dtype)

    if action_ids.numel() == 0:
        return torch.zeros((env.num_envs, action_dim), device=device, dtype=dtype)
    if (action_ids < 0).any():
        return torch.zeros((env.num_envs, action_dim), device=device, dtype=dtype)

    max_id = int(action_ids.max().item())
    if actions_full.shape[-1] <= max_id:
        return torch.zeros((env.num_envs, action_dim), device=device, dtype=dtype)
    return actions_full[:, action_ids]


def _tracking_prior_joint_dim(frame_obs_dim: int, include_base_lin_vel: bool = True) -> int:
    velocity_dim = _TRACKING_PRIOR_ANG_VEL_DIM
    if include_base_lin_vel:
        velocity_dim += _TRACKING_PRIOR_LIN_VEL_DIM
    residual = frame_obs_dim - velocity_dim
    if residual <= 0 or residual % 3 != 0:
        layout = (
            "[lin_vel, ang_vel, joint_pos, joint_vel, actions]"
            if include_base_lin_vel
            else "[ang_vel, joint_pos, joint_vel, actions]"
        )
        raise ValueError(
            f"Tracking prior frame dim is incompatible with {layout} layout: "
            f"got {frame_obs_dim}."
        )
    return residual // 3


def tracking_prior_history_flat(frame_history: torch.Tensor, include_base_lin_vel: bool = True) -> torch.Tensor:
    """Flatten [env, history, frame] tracking prior history using tracking-task term-major ordering."""

    if frame_history.ndim != 3:
        raise ValueError(
            f"Expected tracking prior history with shape [N, H, D], got {tuple(frame_history.shape)}.")

    frame_dim = int(frame_history.shape[-1])
    joint_dim = _tracking_prior_joint_dim(
        frame_dim, include_base_lin_vel=include_base_lin_vel)
    lin_end = _TRACKING_PRIOR_LIN_VEL_DIM if include_base_lin_vel else 0
    ang_end = lin_end + _TRACKING_PRIOR_ANG_VEL_DIM
    joint_pos_end = ang_end + joint_dim
    joint_vel_end = joint_pos_end + joint_dim

    parts: list[torch.Tensor] = []
    if include_base_lin_vel:
        parts.append(frame_history[..., :lin_end].reshape(
            frame_history.shape[0], -1))
    parts.extend(
        (
            frame_history[..., lin_end:ang_end].reshape(
                frame_history.shape[0], -1),
            frame_history[..., ang_end:joint_pos_end].reshape(
                frame_history.shape[0], -1),
            frame_history[..., joint_pos_end:joint_vel_end].reshape(
                frame_history.shape[0], -1),
            frame_history[..., joint_vel_end:].reshape(
                frame_history.shape[0], -1),
        )
    )
    return torch.cat(parts, dim=-1)


def _tracking_prior_frame_terms(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    noise_cfg: dict[str, tuple[float, float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lin_vel = base_lin_vel(env, asset_cfg=asset_cfg)
    ang_vel = base_ang_vel(env, asset_cfg=asset_cfg)
    joint_pos = joint_pos_rel(env, asset_cfg=asset_cfg)
    joint_vel = joint_vel_rel(env, asset_cfg=asset_cfg)
    if noise_cfg:
        lin_vel = _add_uniform_noise(lin_vel, noise_cfg.get("lin_vel"))
        ang_vel = _add_uniform_noise(ang_vel, noise_cfg.get("ang_vel"))
        joint_pos = _add_uniform_noise(joint_pos, noise_cfg.get("joint_pos"))
        joint_vel = _add_uniform_noise(joint_vel, noise_cfg.get("joint_vel"))
    action_dim = joint_pos.shape[-1]
    actions = _get_last_joint_actions(
        env,
        asset_cfg.joint_ids,
        action_dim,
        joint_pos.device,
        joint_pos.dtype,
    )
    return lin_vel, ang_vel, joint_pos, joint_vel, actions


def tracking_prior_proprio(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noise_cfg: dict[str, tuple[float, float]] | None = None,
    use_cache: bool = True,
    include_base_lin_vel: bool = True,
) -> torch.Tensor:
    """Proprioception stack expected by the humanoid prior (G1 tracking distill)."""

    cache_key = _tracking_prior_frame_cache_key(
        asset_cfg, noise_cfg, include_base_lin_vel)
    if use_cache:
        cached = _get_cached_step_tensor(env, cache_key)
        if cached is not None:
            return cached

    lin_vel, ang_vel, joint_pos, joint_vel, actions = _tracking_prior_frame_terms(
        env,
        asset_cfg=asset_cfg,
        noise_cfg=noise_cfg,
    )
    if include_base_lin_vel:
        obs = torch.cat((lin_vel, ang_vel, joint_pos,
                        joint_vel, actions), dim=-1)
    else:
        obs = torch.cat((ang_vel, joint_pos, joint_vel, actions), dim=-1)
    if use_cache:
        return _set_cached_step_tensor(env, cache_key, obs)
    return obs


def reset_tracking_prior_history(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noise_cfg: dict[str, tuple[float, float]] | None = None,
    history_length: int = 5,
    env_ids=None,
    include_base_lin_vel: bool = True,
) -> None:
    clear_prior_obs_step_cache(env)
    cache_key = _tracking_prior_history_cache_key(
        asset_cfg, noise_cfg, history_length, include_base_lin_vel)
    history_state = getattr(
        env, "_locomanip_tracking_prior_history_state", None)
    if not isinstance(history_state, dict):
        return
    state = history_state.get(cache_key)
    if not isinstance(state, dict):
        return

    initialized = state.get("initialized")
    frame_history = state.get("frame_history")
    skip_update_once = state.get("skip_update_once")
    if (
        not isinstance(initialized, torch.Tensor)
        or not isinstance(frame_history, torch.Tensor)
        or not isinstance(skip_update_once, torch.Tensor)
    ):
        return

    if env_ids is None:
        env_ids_t = torch.arange(
            env.num_envs, device=initialized.device, dtype=torch.long)
    elif isinstance(env_ids, slice):
        env_ids_t = torch.arange(
            env.num_envs, device=initialized.device, dtype=torch.long)[env_ids]
    else:
        env_ids_t = torch.as_tensor(
            env_ids, device=initialized.device, dtype=torch.long)
    if env_ids_t.numel() == 0:
        return

    initialized[env_ids_t] = False
    frame_history[env_ids_t] = 0.0
    skip_update_once[env_ids_t] = False
    state["last_step"] = None


def capture_tracking_prior_frame_history(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noise_cfg: dict[str, tuple[float, float]] | None = None,
    history_length: int = 5,
    env_ids=None,
    include_base_lin_vel: bool = True,
) -> torch.Tensor:
    """Capture the raw [history, frame] tracking-prior state for the requested environments."""

    history_length = max(1, int(history_length))
    _ = tracking_prior_history(
        env,
        asset_cfg=asset_cfg,
        noise_cfg=noise_cfg,
        history_length=history_length,
        include_base_lin_vel=include_base_lin_vel,
    )
    cache_key = _tracking_prior_history_cache_key(
        asset_cfg, noise_cfg, history_length, include_base_lin_vel)
    history_state = getattr(
        env, "_locomanip_tracking_prior_history_state", None)
    if not isinstance(history_state, dict):
        return torch.zeros((0, history_length, 0), device=env.device)

    state = history_state.get(cache_key)
    if not isinstance(state, dict):
        return torch.zeros((0, history_length, 0), device=env.device)

    frame_history = state.get("frame_history")
    if not isinstance(frame_history, torch.Tensor):
        return torch.zeros((0, history_length, 0), device=env.device)

    env_ids_t = _resolve_env_ids_tensor(
        env, env_ids, device=frame_history.device)
    if env_ids_t.numel() == 0:
        return frame_history[:0].clone()
    return frame_history[env_ids_t].clone()


def restore_tracking_prior_frame_history(
    env: ManagerBasedRLEnv,
    frame_history: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noise_cfg: dict[str, tuple[float, float]] | None = None,
    history_length: int = 5,
    env_ids=None,
    mark_current_step: bool = True,
    include_base_lin_vel: bool = True,
) -> None:
    """Restore raw [history, frame] tracking-prior state for the requested environments."""

    history_length = max(1, int(history_length))
    frame_history_t = torch.as_tensor(
        frame_history, device=env.device, dtype=torch.float32)
    if frame_history_t.ndim != 3:
        raise ValueError(
            f"Expected tracking prior frame history with shape [N, H, D], got {tuple(frame_history_t.shape)}."
        )

    env_ids_t = _resolve_env_ids_tensor(
        env, env_ids, device=frame_history_t.device)
    if env_ids_t.numel() == 0:
        return
    if frame_history_t.shape[0] != env_ids_t.numel():
        raise ValueError(
            f"Frame history batch size ({frame_history_t.shape[0]}) does not match env_ids ({env_ids_t.numel()})."
        )
    if frame_history_t.shape[1] != history_length:
        raise ValueError(
            f"Frame history length ({frame_history_t.shape[1]}) does not match requested history_length ({history_length})."
        )

    clear_prior_obs_step_cache(env)
    cache_key = _tracking_prior_history_cache_key(
        asset_cfg, noise_cfg, history_length, include_base_lin_vel)
    dummy_frame = torch.zeros(
        (env.num_envs, frame_history_t.shape[-1]), device=env.device, dtype=frame_history_t.dtype)
    state = _tracking_prior_history_state(
        env, cache_key, history_length, dummy_frame)
    state["frame_history"][env_ids_t] = frame_history_t
    state["initialized"][env_ids_t] = True
    state["skip_update_once"][env_ids_t] = bool(mark_current_step)
    state["last_step"] = None


def tracking_prior_history(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noise_cfg: dict[str, tuple[float, float]] | None = None,
    history_length: int = 5,
    include_base_lin_vel: bool = True,
) -> torch.Tensor:
    history_length = max(1, int(history_length))
    cache_key = _tracking_prior_history_cache_key(
        asset_cfg, noise_cfg, history_length, include_base_lin_vel)
    cached = _get_cached_step_tensor(env, cache_key)
    if cached is not None:
        return cached

    frame_obs = tracking_prior_proprio(
        env,
        asset_cfg=asset_cfg,
        noise_cfg=noise_cfg,
        include_base_lin_vel=include_base_lin_vel,
    )
    state = _tracking_prior_history_state(
        env, cache_key, history_length, frame_obs)
    frame_history = state["frame_history"]
    initialized = state["initialized"]
    skip_update_once = state["skip_update_once"]
    step_tag = _current_step_tag(env)
    if state.get("last_step") != step_tag:
        needs_init = ~initialized
        skip_mask = skip_update_once.bool()
        if needs_init.any():
            repeated = frame_obs[needs_init].unsqueeze(
                1).repeat(1, history_length, 1)
            frame_history[needs_init] = repeated
            initialized[needs_init] = True
        active = ~(needs_init | skip_mask)
        if active.any():
            frame_history[active, :-1] = frame_history[active, 1:].clone()
            frame_history[active, -1] = frame_obs[active]
        if skip_mask.any():
            skip_update_once[skip_mask] = False
        state["last_step"] = step_tag

    history_obs = tracking_prior_history_flat(
        frame_history, include_base_lin_vel=include_base_lin_vel)
    return _set_cached_step_tensor(env, cache_key, history_obs)


class TrackingPriorHistoryTerm(ManagerTermBase):
    """Stateful 5-step humanoid proprio history with tracking-task term-major flattening."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def reset(self, env_ids=None):
        params = getattr(self.cfg, "params", {}) if getattr(
            self, "cfg", None) is not None else {}
        asset_cfg = params.get("asset_cfg", SceneEntityCfg("robot"))
        noise_cfg = params.get("noise_cfg")
        history_length = params.get("history_length", 5)
        include_base_lin_vel = params.get("include_base_lin_vel", True)
        reset_tracking_prior_history(
            self._env,
            asset_cfg=asset_cfg,
            noise_cfg=noise_cfg,
            history_length=history_length,
            env_ids=env_ids,
            include_base_lin_vel=include_base_lin_vel,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        noise_cfg: dict[str, tuple[float, float]] | None = None,
        history_length: int = 5,
        include_base_lin_vel: bool = True,
    ) -> torch.Tensor:
        return tracking_prior_history(
            env,
            asset_cfg=asset_cfg,
            noise_cfg=noise_cfg,
            history_length=history_length,
            include_base_lin_vel=include_base_lin_vel,
        )


def right_hand_proprio(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wrist_body_name: str = "right_palm_link",
    use_cache: bool = True,
) -> torch.Tensor:
    """Right-hand proprioception stack for floating-hand priors (adds last action slice)."""

    cache_key = _right_hand_prior_cache_key(asset_cfg, wrist_body_name)
    if use_cache:
        cached = _get_cached_step_tensor(env, cache_key)
        if cached is not None:
            return cached

    asset: Articulation = env.scene[asset_cfg.name]

    # Cache the wrist index on the environment to avoid repeated lookups.
    wrist_idx: int
    cache_attr = "_locomanip_right_wrist_id"
    if hasattr(env, cache_attr):
        wrist_idx = int(getattr(env, cache_attr))
    else:
        wrist_ids, _ = asset.find_bodies(
            [wrist_body_name], preserve_order=True)
        if len(wrist_ids) == 0:
            raise RuntimeError(
                f"Wrist body '{wrist_body_name}' not found on asset '{asset_cfg.name}'.")
        wrist_idx = int(wrist_ids[0])
        setattr(env, cache_attr, wrist_idx)

    wrist_quat = asset.data.body_quat_w[:, wrist_idx]
    wrist_lin_vel_b = quat_apply(
        quat_inv(wrist_quat), asset.data.body_lin_vel_w[:, wrist_idx])
    wrist_ang_vel_b = quat_apply(
        quat_inv(wrist_quat), asset.data.body_ang_vel_w[:, wrist_idx])

    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - \
        asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_vel_rel = asset.data.joint_vel[:, asset_cfg.joint_ids] - \
        asset.data.default_joint_vel[:, asset_cfg.joint_ids]

    action_dim = joint_pos_rel.shape[-1]
    actions = _get_last_joint_actions(
        env,
        asset_cfg.joint_ids,
        action_dim,
        joint_pos_rel.device,
        joint_pos_rel.dtype,
    )
    obs = torch.cat((wrist_lin_vel_b, wrist_ang_vel_b,
                    joint_pos_rel, joint_vel_rel, actions), dim=-1)
    if use_cache:
        return _set_cached_step_tensor(env, cache_key, obs)
    return obs


def object_pose_in_robot_frame(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Relative pose of a target rigid object in the robot root frame (pos + first 2 rotation columns)."""

    robot: Articulation = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]

    rel_pos, rel_quat = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target.data.root_pos_w,
        target.data.root_quat_w,
    )
    rel_rot_cols = matrix_from_quat(
        rel_quat)[..., :2].reshape(env.num_envs, -1)
    return torch.cat((rel_pos, rel_rot_cols), dim=-1)


def virtual_pose_in_robot_frame(
    env: ManagerBasedRLEnv,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    target_pos = torch.tensor(pos, device=robot.device,
                              dtype=robot.data.root_pos_w.dtype).unsqueeze(0)
    target_pos = target_pos + env.scene.env_origins
    target_quat = torch.tensor(
        quat, device=robot.device, dtype=robot.data.root_quat_w.dtype).unsqueeze(0)
    target_quat = target_quat.expand(env.num_envs, -1)
    rel_pos, rel_quat = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target_pos,
        target_quat,
    )
    rel_rot_cols = matrix_from_quat(
        rel_quat)[..., :2].reshape(env.num_envs, -1)
    return torch.cat((rel_pos, rel_rot_cols), dim=-1)


def cube_pose_in_hand_frame(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
    hand_body_name: str = "right_palm_link",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Cube pose relative to the right hand frame (pos + first 2 rotation columns)."""

    robot: Articulation = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]

    hand_idx = _resolve_body_index(
        env, robot, hand_body_name, "_locomanip_right_hand_body_idx")
    hand_pos = robot.data.body_pos_w[:, hand_idx]
    hand_quat = robot.data.body_quat_w[:, hand_idx]

    rel_pos, rel_quat = subtract_frame_transforms(
        hand_pos,
        hand_quat,
        target.data.root_pos_w,
        target.data.root_quat_w,
    )
    rel_rot_cols = matrix_from_quat(
        rel_quat)[..., :2].reshape(env.num_envs, -1)
    return torch.cat((rel_pos, rel_rot_cols), dim=-1)


def body_pose_in_robot_frame(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
    body_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Relative pose of a named articulation body in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    target: Articulation = env.scene[target_cfg.name]

    body_idx = _resolve_body_index(
        env, target, body_name, f"_locomanip_body_pose_robot_{target_cfg.name}_{body_name}")
    rel_pos, rel_quat = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target.data.body_pos_w[:, body_idx],
        target.data.body_quat_w[:, body_idx],
    )
    rel_rot_cols = matrix_from_quat(
        rel_quat)[..., :2].reshape(env.num_envs, -1)
    return torch.cat((rel_pos, rel_rot_cols), dim=-1)


def body_offset_pose_in_robot_frame(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
    body_name: str,
    local_pos: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Relative pose of a local point on a named articulation body in the robot root frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    target: Articulation = env.scene[target_cfg.name]

    body_idx = _resolve_body_index(
        env, target, body_name, f"_locomanip_body_offset_pose_robot_{target_cfg.name}_{body_name}")
    body_pos_w = target.data.body_pos_w[:, body_idx]
    body_quat_w = target.data.body_quat_w[:, body_idx]
    local_pos_t = torch.tensor(
        local_pos, device=body_pos_w.device, dtype=body_pos_w.dtype).view(1, 3)
    point_pos_w = body_pos_w + \
        quat_apply(body_quat_w, local_pos_t.expand(env.num_envs, -1))
    rel_pos, rel_quat = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        point_pos_w,
        body_quat_w,
    )
    rel_rot_cols = matrix_from_quat(
        rel_quat)[..., :2].reshape(env.num_envs, -1)
    return torch.cat((rel_pos, rel_rot_cols), dim=-1)


def body_pose_in_hand_frame(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
    body_name: str,
    hand_body_name: str = "right_palm_link",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Relative pose of a named articulation body in the right-hand frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    target: Articulation = env.scene[target_cfg.name]

    hand_idx = _resolve_body_index(
        env, robot, hand_body_name, f"_locomanip_body_pose_hand_{robot_cfg.name}_{hand_body_name}")
    body_idx = _resolve_body_index(
        env, target, body_name, f"_locomanip_body_pose_hand_{target_cfg.name}_{body_name}")
    rel_pos, rel_quat = subtract_frame_transforms(
        robot.data.body_pos_w[:, hand_idx],
        robot.data.body_quat_w[:, hand_idx],
        target.data.body_pos_w[:, body_idx],
        target.data.body_quat_w[:, body_idx],
    )
    rel_rot_cols = matrix_from_quat(
        rel_quat)[..., :2].reshape(env.num_envs, -1)
    return torch.cat((rel_pos, rel_rot_cols), dim=-1)


def body_offset_pose_in_hand_frame(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg,
    body_name: str,
    local_pos: tuple[float, float, float],
    hand_body_name: str = "right_palm_link",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Relative pose of a local point on a named articulation body in the right-hand frame."""

    robot: Articulation = env.scene[robot_cfg.name]
    target: Articulation = env.scene[target_cfg.name]

    hand_idx = _resolve_body_index(
        env, robot, hand_body_name, f"_locomanip_body_offset_pose_hand_{robot_cfg.name}_{hand_body_name}")
    body_idx = _resolve_body_index(
        env, target, body_name, f"_locomanip_body_offset_pose_hand_{target_cfg.name}_{body_name}")
    body_pos_w = target.data.body_pos_w[:, body_idx]
    body_quat_w = target.data.body_quat_w[:, body_idx]
    local_pos_t = torch.tensor(
        local_pos, device=body_pos_w.device, dtype=body_pos_w.dtype).view(1, 3)
    point_pos_w = body_pos_w + \
        quat_apply(body_quat_w, local_pos_t.expand(env.num_envs, -1))
    rel_pos, rel_quat = subtract_frame_transforms(
        robot.data.body_pos_w[:, hand_idx],
        robot.data.body_quat_w[:, hand_idx],
        point_pos_w,
        body_quat_w,
    )
    rel_rot_cols = matrix_from_quat(
        rel_quat)[..., :2].reshape(env.num_envs, -1)
    return torch.cat((rel_pos, rel_rot_cols), dim=-1)


def articulation_joint_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Selected articulation joint position and velocity."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = getattr(asset_cfg, "joint_ids", None)
    if isinstance(joint_ids, slice):
        return torch.cat((asset.data.joint_pos[:, joint_ids], asset.data.joint_vel[:, joint_ids]), dim=-1)
    if joint_ids is None:
        joint_names = list(getattr(asset_cfg, "joint_names", ()) or ())
        cache_attr = f"_locomanip_articulation_joint_state_{asset_cfg.name}_{'_'.join(joint_names)}"
        cached = getattr(env, cache_attr, None)
        if isinstance(cached, torch.Tensor):
            joint_ids_t = cached
        else:
            joint_ids, joint_found = asset.find_joints(
                joint_names, preserve_order=True)
            if len(joint_ids) != len(joint_names):
                missing = sorted(set(joint_names) - set(joint_found))
                raise RuntimeError(
                    f"Missing joints on '{asset_cfg.name}': {missing}")
            joint_ids_t = torch.as_tensor(
                joint_ids, device=asset.data.joint_pos.device, dtype=torch.long)
            setattr(env, cache_attr, joint_ids_t)
    else:
        joint_ids_t = torch.as_tensor(
            joint_ids, device=asset.data.joint_pos.device, dtype=torch.long)

    return torch.cat((asset.data.joint_pos[:, joint_ids_t], asset.data.joint_vel[:, joint_ids_t]), dim=-1)


def fingertip_force_on_cube(
    env: ManagerBasedRLEnv,
    sensor_names: Iterable[str] | str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fingertip_body_names: tuple[str, ...] | list[str] = tuple(
        RIGHT_HAND_TIP_NAMES),
    force_scale: float | None = None,
    force_clip: float | None = None,
) -> torch.Tensor:
    """Net contact forces on right-hand fingertips; sensor is filtered to cube contacts.
    """

    sensors = getattr(env.scene, "sensors", {})
    sensor_list: list = []
    if isinstance(sensor_names, str):
        sensor = sensors.get(sensor_names) if isinstance(
            sensors, dict) else None
        if sensor is not None:
            sensor_list.append(sensor)
    elif sensor_names is not None:
        for name in sensor_names:
            sensor = sensors.get(name) if isinstance(sensors, dict) else None
            if sensor is not None:
                sensor_list.append(sensor)
    if len(sensor_list) == 0:
        return torch.zeros((env.num_envs, len(fingertip_body_names) * 3), device=env.device)

    forces_accum = []
    for sensor in sensor_list:
        forces = getattr(sensor.data, "force_matrix_w", None)
        if forces is None:
            forces = getattr(sensor.data, "force_matrix_w_history", None)
        if forces is None:
            forces = getattr(sensor.data, "net_forces_w", None)
        if isinstance(forces, torch.Tensor):
            forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
        forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
        if force_scale is not None and force_scale > 0.0:
            forces = forces / float(force_scale)
        if force_clip is not None and force_clip > 0.0:
            clip = float(force_clip)
            forces = torch.clamp(forces, -clip, clip)
        if forces.ndim > 2:
            forces = forces.reshape(env.num_envs, -1, 3).sum(dim=1)
        forces_accum.append(forces)
    if len(forces_accum) == 0:
        return torch.zeros((env.num_envs, len(fingertip_body_names) * 3), device=env.device)

    forces_cat = torch.cat(forces_accum, dim=1)
    return forces_cat.view(env.num_envs, -1)


def last_residual_latent(
    env: ManagerBasedRLEnv,
    action_term_name: str = "joint_pos",
) -> torch.Tensor:
    """Last emitted residual latent (raw action) from the action term."""

    try:
        term = env.action_manager.get_term(action_term_name)
    except Exception:
        return torch.zeros((env.num_envs, 0), device=env.device)
    raw = getattr(term, "raw_actions", None)
    if isinstance(raw, torch.Tensor):
        return raw
    return torch.zeros((env.num_envs, 0), device=env.device)


def zero_last_residual_latent(
    env: ManagerBasedRLEnv,
    action_term_name: str = "joint_pos",
) -> torch.Tensor:
    try:
        term = env.action_manager.get_term(action_term_name)
    except Exception:
        return torch.zeros((env.num_envs, 0), device=env.device)
    raw = getattr(term, "raw_actions", None)
    if isinstance(raw, torch.Tensor):
        return torch.zeros_like(raw)
    action_dim = int(getattr(term, "action_dim", 0))
    return torch.zeros((env.num_envs, action_dim), device=env.device)


def zeros_obs(env: ManagerBasedRLEnv, dim: int) -> torch.Tensor:
    """Constant zero observation of the given dim — used as a stub for obs terms
    that an actor expects by name (e.g. CoordResidual's bottle/table terms) but
    that aren't physically present in a minimal env."""
    return torch.zeros((env.num_envs, int(dim)), device=env.device)


def body_prior_mean(
    env: ManagerBasedRLEnv,
    action_term_name: str = "joint_pos",
) -> torch.Tensor:
    """Current frozen body prior mean used by the residual action term."""

    try:
        term = env.action_manager.get_term(action_term_name)
    except Exception:
        return torch.zeros((env.num_envs, 0), device=env.device)
    getter = getattr(term, "get_body_prior_mean", None)
    if callable(getter):
        value = getter()
        if isinstance(value, torch.Tensor):
            return value
    return torch.zeros((env.num_envs, 0), device=env.device)


def hand_prior_mean(
    env: ManagerBasedRLEnv,
    action_term_name: str = "joint_pos",
) -> torch.Tensor:
    """Current frozen hand prior mean used by the residual action term."""

    try:
        term = env.action_manager.get_term(action_term_name)
    except Exception:
        return torch.zeros((env.num_envs, 0), device=env.device)
    getter = getattr(term, "get_hand_prior_mean", None)
    if callable(getter):
        value = getter()
        if isinstance(value, torch.Tensor):
            return value
    return torch.zeros((env.num_envs, 0), device=env.device)


def stage_one_hot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """One-hot stage buffer from the locomanip stage command."""
    command = env.command_manager.get_term(command_name)
    return command.command
