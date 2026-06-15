from __future__ import annotations

from typing import Iterable

import torch
from isaaclab.utils import configclass

from coordex.tasks.locomanip.mdp.observations import (
    capture_tracking_prior_frame_history,
    restore_tracking_prior_frame_history,
)


_RSI_SNAPSHOT_EXTRA_MARKER = -12345.0
_RSI_SNAPSHOT_EXTRA_META_V1_DIM = 5
_RSI_SNAPSHOT_EXTRA_META_V2_DIM = 8
_RSI_SNAPSHOT_EXTRA_VERSION_V2 = 2.0


@configclass
class NoDemoRSICfg:
    enabled: bool = False
    capacity_per_stage: int = 512
    min_samples_to_enable: tuple[int, ...] = (0, 128, 128, 128, 128)
    p_stage0: float = 0.3
    stage_sampling_weight: tuple[float, ...] = (0.0, 1.0, 1.0, 1.0, 1.0)
    alpha: float = 1.0
    adaptive_stage_sampling_enabled: bool = False
    adaptive_stage_max: int = 3
    adaptive_stats_decay: float = 0.995
    adaptive_success_prior: float = 16.0
    adaptive_difficulty_power: float = 1.0
    adaptive_difficulty_floor: float = 0.10
    adaptive_consolidation_enter: float = 0.80
    adaptive_consolidation_exit: float = 0.75
    adaptive_consolidation_stage0_share: float = 0.85
    settle_steps: int = 3
    max_root_lin_vel: float | None = 0.4
    max_root_ang_vel: float | None = 0.75
    max_joint_vel: float | None = 10.0
    max_cube_lin_vel: float | None = 1.0
    max_cube_ang_vel: float | None = 1.0
    min_stage_steps_before_store: tuple[int, ...] = (0, 12, 6, 6, 6)
    warmup_enabled: bool = True
    warmup_mode: str = "reward_metric"
    warmup_success_threshold: float = 0.6
    warmup_ramp_initial_share: float = 0.05
    warmup_ramp_attempts: int = 1000
    warmup_stage1_reward_key: str = "episode_reward/robot_object_distance"
    warmup_stage1_threshold: float = 7.0
    warmup_stage2_reward_key: str = "episode_reward/cube_lift_height"
    warmup_stage2_threshold: float = 1.0
    warmup_stage1_min_resets: int = 1000
    warmup_stage2_min_resets: int = 2000


def _to_env_ids(env, env_ids: Iterable[int] | torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)


def _humanoid_prior_history_cfg(env) -> tuple[object | None, dict | None, int, bool]:
    observations_cfg = getattr(env.cfg, "observations", None)
    policy_cfg = getattr(observations_cfg, "policy", None)
    term_cfg = getattr(policy_cfg, "humanoid_prior", None)
    params = getattr(term_cfg, "params", {}) if term_cfg is not None else {}
    if not isinstance(params, dict):
        return None, None, 0, True

    asset_cfg = params.get("asset_cfg")
    history_length = int(params.get("history_length", 0) or 0)
    if asset_cfg is None or history_length <= 0:
        return None, None, 0, True
    return asset_cfg, params.get("noise_cfg"), history_length, bool(params.get("include_base_lin_vel", True))


def _snapshot_action_term_name(env) -> str:
    observations_cfg = getattr(env.cfg, "observations", None)
    policy_cfg = getattr(observations_cfg, "policy", None)
    term_cfg = getattr(policy_cfg, "last_latent", None)
    params = getattr(term_cfg, "params", {}) if term_cfg is not None else {}
    if isinstance(params, dict):
        return str(params.get("action_term_name", "joint_pos"))
    return "joint_pos"


def _capture_residual_action_aux_state(
    env,
    env_ids_t: torch.Tensor,
    *,
    action_term_name: str,
) -> dict[str, torch.Tensor]:
    if env_ids_t.numel() == 0:
        return {}
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return {}
    try:
        term = action_manager.get_term(action_term_name)
    except Exception:
        return {}

    getter = getattr(term, "capture_rsi_aux_state", None)
    if callable(getter):
        state = getter(env_ids_t)
        return state if isinstance(state, dict) else {}

    state: dict[str, torch.Tensor] = {}
    raw_actions = getattr(term, "raw_actions", None)
    if isinstance(raw_actions, torch.Tensor) and raw_actions.ndim == 2 and raw_actions.shape[0] == env.scene.num_envs:
        state["raw_actions"] = raw_actions[env_ids_t].detach(
        ).clone().to(dtype=torch.float32)
    hand_prev_targets = getattr(term, "_hand_prev_targets", None)
    if (
        isinstance(hand_prev_targets, torch.Tensor)
        and hand_prev_targets.ndim == 2
        and hand_prev_targets.shape[0] == env.scene.num_envs
    ):
        state["hand_prev_targets"] = hand_prev_targets[env_ids_t].detach(
        ).clone().to(dtype=torch.float32)
    return state


def _pending_obs_restore_state(env) -> dict:
    state = getattr(env, "_no_demo_rsi_pending_obs_restore", None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, "_no_demo_rsi_pending_obs_restore", state)
    return state


def _ensure_pending_tensor(
    state: dict,
    key: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = state.get(key)
    if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.device != device or tensor.dtype != dtype:
        tensor = torch.zeros(shape, device=device, dtype=dtype)
        state[key] = tensor
    return tensor


def _stash_locomanip_snapshot_aux_state(
    env,
    env_ids_t: torch.Tensor,
    *,
    last_action: torch.Tensor | None = None,
    prior_frame_history: torch.Tensor | None = None,
    stage1_ready_history: torch.Tensor | None = None,
    raw_actions: torch.Tensor | None = None,
    hand_prev_targets: torch.Tensor | None = None,
) -> None:
    if env_ids_t.numel() == 0:
        return

    state = _pending_obs_restore_state(env)
    num_envs = int(env.scene.num_envs)

    last_action_valid = _ensure_pending_tensor(
        state,
        "last_action_valid",
        (num_envs,),
        device=env.device,
        dtype=torch.bool,
    )
    last_action_valid[env_ids_t] = False
    if isinstance(last_action, torch.Tensor) and last_action.ndim == 2 and last_action.shape[0] == env_ids_t.numel():
        last_action_t = torch.as_tensor(
            last_action, device=env.device, dtype=torch.float32)
        last_action_buf = _ensure_pending_tensor(
            state,
            "last_action",
            (num_envs, last_action_t.shape[1]),
            device=env.device,
            dtype=torch.float32,
        )
        last_action_buf[env_ids_t] = last_action_t
        last_action_valid[env_ids_t] = True

    prior_history_valid = _ensure_pending_tensor(
        state,
        "prior_history_valid",
        (num_envs,),
        device=env.device,
        dtype=torch.bool,
    )
    prior_history_valid[env_ids_t] = False
    if (
        isinstance(prior_frame_history, torch.Tensor)
        and prior_frame_history.ndim == 3
        and prior_frame_history.shape[0] == env_ids_t.numel()
    ):
        prior_frame_history_t = torch.as_tensor(
            prior_frame_history, device=env.device, dtype=torch.float32)
        prior_history_buf = _ensure_pending_tensor(
            state,
            "prior_history",
            (num_envs,
             prior_frame_history_t.shape[1], prior_frame_history_t.shape[2]),
            device=env.device,
            dtype=torch.float32,
        )
        prior_history_buf[env_ids_t] = prior_frame_history_t
        prior_history_valid[env_ids_t] = True

    stage1_ready_valid = _ensure_pending_tensor(
        state,
        "stage1_ready_history_valid",
        (num_envs,),
        device=env.device,
        dtype=torch.bool,
    )
    stage1_ready_valid[env_ids_t] = False
    if (
        isinstance(stage1_ready_history, torch.Tensor)
        and stage1_ready_history.ndim == 2
        and stage1_ready_history.shape[0] == env_ids_t.numel()
    ):
        ready_history_t = torch.as_tensor(
            stage1_ready_history, device=env.device, dtype=torch.bool)
        ready_history_buf = _ensure_pending_tensor(
            state,
            "stage1_ready_history",
            (num_envs, ready_history_t.shape[1]),
            device=env.device,
            dtype=torch.bool,
        )
        ready_history_buf[env_ids_t] = ready_history_t
        stage1_ready_valid[env_ids_t] = True

    raw_actions_valid = _ensure_pending_tensor(
        state,
        "raw_actions_valid",
        (num_envs,),
        device=env.device,
        dtype=torch.bool,
    )
    raw_actions_valid[env_ids_t] = False
    if isinstance(raw_actions, torch.Tensor) and raw_actions.ndim == 2 and raw_actions.shape[0] == env_ids_t.numel():
        raw_actions_t = torch.as_tensor(
            raw_actions, device=env.device, dtype=torch.float32)
        raw_actions_buf = _ensure_pending_tensor(
            state,
            "raw_actions",
            (num_envs, raw_actions_t.shape[1]),
            device=env.device,
            dtype=torch.float32,
        )
        raw_actions_buf[env_ids_t] = raw_actions_t
        raw_actions_valid[env_ids_t] = True

    hand_prev_targets_valid = _ensure_pending_tensor(
        state,
        "hand_prev_targets_valid",
        (num_envs,),
        device=env.device,
        dtype=torch.bool,
    )
    hand_prev_targets_valid[env_ids_t] = False
    if (
        isinstance(hand_prev_targets, torch.Tensor)
        and hand_prev_targets.ndim == 2
        and hand_prev_targets.shape[0] == env_ids_t.numel()
    ):
        hand_prev_targets_t = torch.as_tensor(
            hand_prev_targets, device=env.device, dtype=torch.float32)
        hand_prev_targets_buf = _ensure_pending_tensor(
            state,
            "hand_prev_targets",
            (num_envs, hand_prev_targets_t.shape[1]),
            device=env.device,
            dtype=torch.float32,
        )
        hand_prev_targets_buf[env_ids_t] = hand_prev_targets_t
        hand_prev_targets_valid[env_ids_t] = True


def restore_locomanip_snapshot_aux_state(
    env,
    env_ids: Iterable[int] | torch.Tensor | None,
    *,
    command=None,
    action_term_name: str | None = None,
) -> None:
    env_ids_t = _to_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return

    state = getattr(env, "_no_demo_rsi_pending_obs_restore", None)
    if not isinstance(state, dict):
        return

    last_action_valid = state.get("last_action_valid")
    last_action = state.get("last_action")
    env_last_action = getattr(env, "_locomanip_last_action", None)
    if (
        isinstance(last_action_valid, torch.Tensor)
        and isinstance(last_action, torch.Tensor)
        and isinstance(env_last_action, torch.Tensor)
        and env_last_action.ndim == 2
        and last_action.shape[1] == env_last_action.shape[1]
    ):
        restore_mask = last_action_valid[env_ids_t]
        if restore_mask.any():
            env_restore = env_ids_t[restore_mask]
            env_last_action[env_restore] = last_action[env_restore]
            prev_actions = getattr(
                env, "_locomanip_prev_joint_action_rate_action", None)
            if isinstance(prev_actions, torch.Tensor) and prev_actions.shape == env_last_action.shape:
                prev_actions[env_restore] = last_action[env_restore]
    if isinstance(last_action_valid, torch.Tensor):
        last_action_valid[env_ids_t] = False

    prior_history_valid = state.get("prior_history_valid")
    prior_history = state.get("prior_history")
    asset_cfg, noise_cfg, history_length, include_base_lin_vel = _humanoid_prior_history_cfg(
        env)
    if (
        isinstance(prior_history_valid, torch.Tensor)
        and isinstance(prior_history, torch.Tensor)
        and asset_cfg is not None
        and history_length > 0
        and prior_history.shape[1] == history_length
    ):
        restore_mask = prior_history_valid[env_ids_t]
        if restore_mask.any():
            env_restore = env_ids_t[restore_mask]
            restore_tracking_prior_frame_history(
                env,
                prior_history[env_restore],
                asset_cfg=asset_cfg,
                noise_cfg=noise_cfg,
                history_length=history_length,
                env_ids=env_restore,
                mark_current_step=True,
                include_base_lin_vel=include_base_lin_vel,
            )
    if isinstance(prior_history_valid, torch.Tensor):
        prior_history_valid[env_ids_t] = False

    ready_history_valid = state.get("stage1_ready_history_valid")
    ready_history = state.get("stage1_ready_history")
    if (
        command is not None
        and hasattr(command, "restore_stage1_ready_history")
        and isinstance(ready_history_valid, torch.Tensor)
        and isinstance(ready_history, torch.Tensor)
    ):
        restore_mask = ready_history_valid[env_ids_t]
        if restore_mask.any():
            env_restore = env_ids_t[restore_mask]
            command.restore_stage1_ready_history(
                env_restore, ready_history[env_restore])
    if isinstance(ready_history_valid, torch.Tensor):
        ready_history_valid[env_ids_t] = False

    raw_actions_valid = state.get("raw_actions_valid")
    raw_actions = state.get("raw_actions")
    hand_prev_targets_valid = state.get("hand_prev_targets_valid")
    hand_prev_targets = state.get("hand_prev_targets")
    restore_raw_actions = None
    restore_hand_prev_targets = None
    if (
        isinstance(raw_actions_valid, torch.Tensor)
        and isinstance(raw_actions, torch.Tensor)
    ):
        restore_mask = raw_actions_valid[env_ids_t]
        if restore_mask.any():
            env_restore = env_ids_t[restore_mask]
            restore_raw_actions = raw_actions[env_restore]
    if (
        isinstance(hand_prev_targets_valid, torch.Tensor)
        and isinstance(hand_prev_targets, torch.Tensor)
    ):
        restore_mask = hand_prev_targets_valid[env_ids_t]
        if restore_mask.any():
            env_restore = env_ids_t[restore_mask]
            restore_hand_prev_targets = hand_prev_targets[env_restore]

    if restore_raw_actions is not None or restore_hand_prev_targets is not None:
        action_manager = getattr(env, "action_manager", None)
        term_name = _snapshot_action_term_name(
            env) if action_term_name is None else action_term_name
        if action_manager is not None:
            try:
                term = action_manager.get_term(term_name)
            except Exception:
                term = None
            if term is not None:
                restorer = getattr(term, "restore_rsi_aux_state", None)
                if callable(restorer):
                    restore_mask = None
                    if restore_raw_actions is not None and isinstance(raw_actions_valid, torch.Tensor):
                        restore_mask = raw_actions_valid[env_ids_t]
                    elif restore_hand_prev_targets is not None and isinstance(hand_prev_targets_valid, torch.Tensor):
                        restore_mask = hand_prev_targets_valid[env_ids_t]
                    if isinstance(restore_mask, torch.Tensor):
                        env_restore = env_ids_t[restore_mask]
                        restorer(
                            env_restore,
                            raw_actions=restore_raw_actions,
                            hand_prev_targets=restore_hand_prev_targets,
                        )

    if isinstance(raw_actions_valid, torch.Tensor):
        raw_actions_valid[env_ids_t] = False
    if isinstance(hand_prev_targets_valid, torch.Tensor):
        hand_prev_targets_valid[env_ids_t] = False


def capture_locomanip_snapshot(env, env_ids: Iterable[int] | torch.Tensor | None, command_name: str = "stage") -> torch.Tensor:
    """Capture robot + cube state along with stage metadata for the given environments."""
    env_ids_t = _to_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return torch.zeros((0, 0), device=env.device)

    command = env.command_manager.get_term(command_name)
    robot = command.robot
    cube = command.cube
    env_origins = env.scene.env_origins[env_ids_t]

    robot_root_pos = robot.data.root_pos_w[env_ids_t] - env_origins
    robot_root_quat = robot.data.root_quat_w[env_ids_t]
    robot_lin_vel = robot.data.root_lin_vel_w[env_ids_t]
    robot_ang_vel = robot.data.root_ang_vel_w[env_ids_t]
    robot_root_state = torch.cat(
        [robot_root_pos, robot_root_quat, robot_lin_vel, robot_ang_vel], dim=-1)

    robot_joint_pos = robot.data.joint_pos[env_ids_t]
    robot_joint_vel = robot.data.joint_vel[env_ids_t]

    cube_root_pos = cube.data.root_pos_w[env_ids_t] - env_origins
    cube_root_quat = cube.data.root_quat_w[env_ids_t]
    cube_lin_vel = cube.data.root_lin_vel_w[env_ids_t]
    cube_ang_vel = cube.data.root_ang_vel_w[env_ids_t]
    cube_root_state = torch.cat(
        [cube_root_pos, cube_root_quat, cube_lin_vel, cube_ang_vel], dim=-1)

    stage_id = command.stage_id[env_ids_t].to(
        dtype=torch.float32).unsqueeze(-1)
    cube_init_z = (
        command.cube_init_z[env_ids_t] - env_origins[:, 2]).unsqueeze(-1)

    snapshot_parts = [
        robot_root_state,
        robot_joint_pos,
        robot_joint_vel,
        cube_root_state,
        stage_id,
        cube_init_z,
    ]
    meta = torch.zeros((_RSI_SNAPSHOT_EXTRA_META_V2_DIM,),
                       device=env.device, dtype=torch.float32)
    meta[0] = _RSI_SNAPSHOT_EXTRA_MARKER
    meta[1] = _RSI_SNAPSHOT_EXTRA_VERSION_V2

    last_action = getattr(env, "_locomanip_last_action", None)
    if isinstance(last_action, torch.Tensor) and last_action.ndim == 2 and last_action.shape[0] == env.scene.num_envs:
        last_action_snapshot = last_action[env_ids_t].detach(
        ).clone().to(dtype=torch.float32)
        meta[2] = float(last_action_snapshot.shape[1])
        snapshot_parts.append(last_action_snapshot)

    asset_cfg, noise_cfg, history_length, include_base_lin_vel = _humanoid_prior_history_cfg(
        env)
    if asset_cfg is not None and history_length > 0:
        prior_frame_history = capture_tracking_prior_frame_history(
            env,
            asset_cfg=asset_cfg,
            noise_cfg=noise_cfg,
            history_length=history_length,
            env_ids=env_ids_t,
            include_base_lin_vel=include_base_lin_vel,
        )
        if prior_frame_history.numel() > 0:
            meta[3] = float(prior_frame_history.shape[1])
            meta[4] = float(prior_frame_history.shape[2])
            snapshot_parts.append(prior_frame_history.reshape(
                prior_frame_history.shape[0], -1))

    stage1_ready_history = getattr(
        command, "recent_stage1_ready_history", None)
    if (
        isinstance(stage1_ready_history, torch.Tensor)
        and stage1_ready_history.ndim == 2
        and stage1_ready_history.shape[0] == env.scene.num_envs
    ):
        stage1_ready_snapshot = stage1_ready_history[env_ids_t].to(
            dtype=torch.float32)
        meta[5] = float(stage1_ready_snapshot.shape[1])
        snapshot_parts.append(stage1_ready_snapshot)

    action_aux_state = _capture_residual_action_aux_state(
        env,
        env_ids_t,
        action_term_name=_snapshot_action_term_name(env),
    )
    raw_actions_snapshot = action_aux_state.get("raw_actions")
    if isinstance(raw_actions_snapshot, torch.Tensor) and raw_actions_snapshot.ndim == 2:
        meta[6] = float(raw_actions_snapshot.shape[1])
        snapshot_parts.append(raw_actions_snapshot)
    hand_prev_targets_snapshot = action_aux_state.get("hand_prev_targets")
    if isinstance(hand_prev_targets_snapshot, torch.Tensor) and hand_prev_targets_snapshot.ndim == 2:
        meta[7] = float(hand_prev_targets_snapshot.shape[1])
        snapshot_parts.append(hand_prev_targets_snapshot)

    meta_batch = meta.unsqueeze(0).expand(env_ids_t.shape[0], -1)
    snapshot_parts.append(meta_batch)
    return torch.cat(snapshot_parts, dim=-1)


def apply_locomanip_snapshot(
    env,
    env_ids: Iterable[int] | torch.Tensor | None,
    snapshot: torch.Tensor,
    command_name: str = "stage",
    reset_joint_targets: bool = True,
) -> None:
    """Write a cached snapshot back into the simulator for the provided environments."""
    env_ids_t = _to_env_ids(env, env_ids)
    if snapshot is None:
        return
    snap = torch.as_tensor(snapshot, device=env.device, dtype=torch.float32)
    if snap.ndim == 1:
        snap = snap.unsqueeze(0)
    if snap.shape[0] != env_ids_t.numel():
        raise ValueError(
            f"Snapshot batch size ({snap.shape[0]}) does not match env_ids ({env_ids_t.numel()})."
        )

    command = env.command_manager.get_term(command_name)
    robot = command.robot
    cube = command.cube
    env_origins = env.scene.env_origins[env_ids_t]
    num_joints = robot.num_joints

    idx = 0
    robot_root_state = snap[:, idx: idx + 13].clone()
    idx += 13
    robot_joint_pos = snap[:, idx: idx + num_joints]
    idx += num_joints
    robot_joint_vel = snap[:, idx: idx + num_joints]
    idx += num_joints
    cube_root_state = snap[:, idx: idx + 13].clone()
    idx += 13
    stage_id = snap[:, idx].to(dtype=torch.long)
    idx += 1
    cube_init_z = snap[:, idx]
    idx += 1

    last_action = None
    prior_frame_history = None
    stage1_ready_history = None
    raw_actions = None
    hand_prev_targets = None
    meta_dim = 0
    version = 0.0
    if snap.shape[1] >= idx + _RSI_SNAPSHOT_EXTRA_META_V2_DIM:
        meta_v2 = snap[:, -_RSI_SNAPSHOT_EXTRA_META_V2_DIM:]
        if float(meta_v2[0, 0].item()) == _RSI_SNAPSHOT_EXTRA_MARKER:
            meta = meta_v2
            meta_dim = _RSI_SNAPSHOT_EXTRA_META_V2_DIM
            version = float(meta[0, 1].item())
        else:
            meta = None
    else:
        meta = None

    if meta is None and snap.shape[1] >= idx + _RSI_SNAPSHOT_EXTRA_META_V1_DIM:
        meta_v1 = snap[:, -_RSI_SNAPSHOT_EXTRA_META_V1_DIM:]
        if float(meta_v1[0, 0].item()) == _RSI_SNAPSHOT_EXTRA_MARKER:
            meta = meta_v1
            meta_dim = _RSI_SNAPSHOT_EXTRA_META_V1_DIM
            version = 1.0

    if meta is not None:
        if version >= _RSI_SNAPSHOT_EXTRA_VERSION_V2 and meta_dim == _RSI_SNAPSHOT_EXTRA_META_V2_DIM:
            last_action_dim = max(0, int(round(float(meta[0, 2].item()))))
            history_length = max(0, int(round(float(meta[0, 3].item()))))
            frame_dim = max(0, int(round(float(meta[0, 4].item()))))
            ready_history_length = max(0, int(round(float(meta[0, 5].item()))))
            raw_action_dim = max(0, int(round(float(meta[0, 6].item()))))
            hand_prev_targets_dim = max(
                0, int(round(float(meta[0, 7].item()))))
            payload = snap[:, idx: snap.shape[1] - meta_dim]
            expected_payload_dim = (
                last_action_dim
                + history_length * frame_dim
                + ready_history_length
                + raw_action_dim
                + hand_prev_targets_dim
            )
            if payload.shape[1] != expected_payload_dim:
                raise ValueError(
                    "Malformed RSI snapshot payload: "
                    f"expected {expected_payload_dim} aux values, got {payload.shape[1]}."
                )
            payload_idx = 0
            if last_action_dim > 0:
                last_action = payload[:,
                                      payload_idx: payload_idx + last_action_dim]
                payload_idx += last_action_dim
            if history_length > 0 and frame_dim > 0:
                history_flat_dim = history_length * frame_dim
                prior_frame_history = payload[:,
                                              payload_idx: payload_idx + history_flat_dim]
                prior_frame_history = prior_frame_history.reshape(
                    snap.shape[0], history_length, frame_dim)
                payload_idx += history_flat_dim
            if ready_history_length > 0:
                stage1_ready_history = payload[:,
                                               payload_idx: payload_idx + ready_history_length] > 0.5
                payload_idx += ready_history_length
            if raw_action_dim > 0:
                raw_actions = payload[:,
                                      payload_idx: payload_idx + raw_action_dim]
                payload_idx += raw_action_dim
            if hand_prev_targets_dim > 0:
                hand_prev_targets = payload[:,
                                            payload_idx: payload_idx + hand_prev_targets_dim]
        else:
            last_action_dim = max(0, int(round(float(meta[0, 1].item()))))
            history_length = max(0, int(round(float(meta[0, 2].item()))))
            frame_dim = max(0, int(round(float(meta[0, 3].item()))))
            ready_history_length = max(0, int(round(float(meta[0, 4].item()))))
            payload = snap[:, idx: snap.shape[1] - meta_dim]
            expected_payload_dim = last_action_dim + \
                history_length * frame_dim + ready_history_length
            if payload.shape[1] != expected_payload_dim:
                raise ValueError(
                    "Malformed RSI snapshot payload: "
                    f"expected {expected_payload_dim} aux values, got {payload.shape[1]}."
                )
            payload_idx = 0
            if last_action_dim > 0:
                last_action = payload[:,
                                      payload_idx: payload_idx + last_action_dim]
                payload_idx += last_action_dim
            if history_length > 0 and frame_dim > 0:
                history_flat_dim = history_length * frame_dim
                prior_frame_history = payload[:,
                                              payload_idx: payload_idx + history_flat_dim]
                prior_frame_history = prior_frame_history.reshape(
                    snap.shape[0], history_length, frame_dim)
                payload_idx += history_flat_dim
            if ready_history_length > 0:
                stage1_ready_history = payload[:,
                                               payload_idx: payload_idx + ready_history_length] > 0.5

    robot_root_state[:, 0:3] += env_origins
    cube_root_state[:, 0:3] += env_origins
    cube_init_z_world = cube_init_z + env_origins[:, 2]

    robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids_t)
    robot.write_joint_state_to_sim(
        robot_joint_pos, robot_joint_vel, env_ids=env_ids_t)
    cube.write_root_state_to_sim(cube_root_state, env_ids=env_ids_t)

    if reset_joint_targets:
        robot.set_joint_position_target(robot_joint_pos, env_ids=env_ids_t)

        robot.set_joint_velocity_target(
            torch.zeros_like(robot_joint_vel), env_ids=env_ids_t)

    command.cube_init_z[env_ids_t] = cube_init_z_world
    _stash_locomanip_snapshot_aux_state(
        env,
        env_ids_t,
        last_action=last_action,
        prior_frame_history=prior_frame_history,
        stage1_ready_history=stage1_ready_history,
        raw_actions=raw_actions,
        hand_prev_targets=hand_prev_targets,
    )
