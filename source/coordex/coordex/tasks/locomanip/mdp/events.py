from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import reset_scene_to_default, _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

from coordex.tasks.locomanip.rsi_buffer import NoDemoRSIBuffer
from coordex.tasks.locomanip.mdp.observations import clear_prior_obs_step_cache
from coordex.tasks.locomanip.mdp.rsi import (
    NoDemoRSICfg,
    apply_locomanip_snapshot,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


STAGE0_TO_STAGE1_RATE_SPECS = (
    ("approach_done", "approach_done"),
    ("hand_height_ok", "hand_height_ok"),
    ("hand_xy_ok", "hand_xy_ok"),
    ("hand_flat", "hand_flat"),
    ("hand_prepped", "hand_prepped"),
    ("stage1_ready", "ready"),
)


def _resolve_env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.scene.num_envs, device=env.device)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _sample_root_pose_noise(
    device: torch.device,
    count: int,
    pose_range: dict[str, tuple[float, float]] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not pose_range:
        pos_noise = torch.zeros((count, 3), device=device)
        ori_noise = torch.zeros((count, 3), device=device)
        return pos_noise, ori_noise
    range_list = [pose_range.get(key, (0.0, 0.0))
                  for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=device, dtype=torch.float32)
    rand_samples = sample_uniform(
        ranges[:, 0], ranges[:, 1], (count, 6), device=device)
    pos_noise = rand_samples[:, 0:3]
    ori_noise = rand_samples[:, 3:6]
    return pos_noise, ori_noise


def _ensure_no_demo_rsi_tensor(
    env,
    attr_name: str,
    shape,
    dtype: torch.dtype,
    *,
    device=None,
    fill_value=0,
) -> torch.Tensor:
    """Ensure a tensor attribute exists with the requested shape/dtype/device."""
    if shape == ():
        target_shape = torch.Size(())
    elif isinstance(shape, int):
        target_shape = torch.Size((shape,))
    else:
        target_shape = torch.Size(shape)

    target_device = None if device is None else torch.device(device)
    tensor = getattr(env, attr_name, None)
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.shape != target_shape
        or tensor.dtype != dtype
        or (target_device is not None and tensor.device != target_device)
    ):
        kwargs = {"dtype": dtype}
        if target_device is not None:
            kwargs["device"] = target_device
        tensor = torch.full(target_shape, fill_value, **kwargs)
        setattr(env, attr_name, tensor)
    return tensor


def _prepare_no_demo_rsi_state(env, num_stages: int, capacity: int, settle_steps: int, stage0_rate_count: int):
    """Ensure RSI buffers and adaptive trackers exist and match the environment shape."""
    _ = settle_steps
    buffer = getattr(env, "_no_demo_rsi_buffer", None)
    if not isinstance(buffer, NoDemoRSIBuffer) or buffer.num_stages != num_stages or buffer.capacity_per_stage != capacity:
        buffer = NoDemoRSIBuffer(
            num_stages=num_stages, capacity_per_stage=capacity, device=env.device)
        setattr(env, "_no_demo_rsi_buffer", buffer)

    device = env.device
    cpu_device = torch.device("cpu")

    stage_overrides = _ensure_no_demo_rsi_tensor(
        env, "_no_demo_rsi_stage_on_reset", (env.num_envs,), torch.long, device=device, fill_value=-1
    )
    settle_buf = _ensure_no_demo_rsi_tensor(
        env, "_no_demo_rsi_settle_steps", (env.num_envs,), torch.long, device=device, fill_value=0
    )
    last_reset_stage = _ensure_no_demo_rsi_tensor(
        env, "_no_demo_rsi_last_reset_stage", (env.num_envs,), torch.long, device=device, fill_value=0
    )
    last_reset_valid = _ensure_no_demo_rsi_tensor(
        env, "_no_demo_rsi_last_reset_valid", (env.num_envs,), torch.bool, device=device, fill_value=False
    )

    reset_counts = _ensure_no_demo_rsi_tensor(
        env, "_no_demo_rsi_reset_stage_counts", (num_stages,), torch.long, device=cpu_device, fill_value=0
    )
    attempt_ema = _ensure_no_demo_rsi_tensor(
        env, "_no_demo_rsi_attempt_ema", (num_stages,), torch.float32, device=cpu_device, fill_value=0.0
    )
    success_ema = _ensure_no_demo_rsi_tensor(
        env, "_no_demo_rsi_success_ema", (num_stages,), torch.float32, device=cpu_device, fill_value=0.0
    )
    success_rate = _ensure_no_demo_rsi_tensor(
        env, "_no_demo_rsi_success_rate", (num_stages,), torch.float32, device=cpu_device, fill_value=0.0
    )
    stage0_transition_attempt_ema = _ensure_no_demo_rsi_tensor(
        env,
        "_no_demo_rsi_stage0_to_stage1_attempt_ema",
        (stage0_rate_count,),
        torch.float32,
        device=cpu_device,
        fill_value=0.0,
    )
    stage0_transition_success_ema = _ensure_no_demo_rsi_tensor(
        env,
        "_no_demo_rsi_stage0_to_stage1_success_ema",
        (stage0_rate_count,),
        torch.float32,
        device=cpu_device,
        fill_value=0.0,
    )
    stage0_transition_rate = _ensure_no_demo_rsi_tensor(
        env,
        "_no_demo_rsi_stage0_to_stage1_rate",
        (stage0_rate_count,),
        torch.float32,
        device=cpu_device,
        fill_value=0.0,
    )
    current_sample_prob = _ensure_no_demo_rsi_tensor(
        env, "_no_demo_rsi_current_sample_prob", (num_stages,), torch.float32, device=cpu_device, fill_value=0.0
    )
    consolidation_mode = _ensure_no_demo_rsi_tensor(
        env, "_no_demo_rsi_adaptive_consolidation_mode", (), torch.bool, device=cpu_device, fill_value=False
    )
    unlocked_mask = _ensure_no_demo_rsi_tensor(
        env, "_no_demo_rsi_warmup_unlocked_mask", (num_stages,), torch.bool, device=cpu_device, fill_value=False
    )
    unlocked_mask[0] = True
    ramp_attempt_count = _ensure_no_demo_rsi_tensor(
        env, "_no_demo_rsi_warmup_ramp_attempt_count", (num_stages,), torch.long, device=cpu_device, fill_value=0
    )

    return (
        buffer,
        stage_overrides,
        settle_buf,
        reset_counts,
        last_reset_stage,
        last_reset_valid,
        attempt_ema,
        success_ema,
        success_rate,
        stage0_transition_attempt_ema,
        stage0_transition_success_ema,
        stage0_transition_rate,
        current_sample_prob,
        consolidation_mode,
        unlocked_mask,
        ramp_attempt_count,
    )


def _refresh_no_demo_rsi_success_rates(
    success_rate: torch.Tensor,
    success_ema: torch.Tensor,
    attempt_ema: torch.Tensor,
    success_prior: float,
) -> None:
    prior = max(float(success_prior), 0.0)
    denom = torch.clamp(attempt_ema + prior, min=1.0e-6)
    numer = success_ema + (0.5 * prior)
    success_rate.copy_(torch.clamp(numer / denom, min=0.0, max=1.0))


def _score_no_demo_rsi_previous_episodes(
    env_ids: torch.Tensor,
    command,
    last_reset_stage: torch.Tensor,
    last_reset_valid: torch.Tensor,
    attempt_ema: torch.Tensor,
    success_ema: torch.Tensor,
    success_rate: torch.Tensor,
    tracked_stage_max: int,
    stats_decay: float,
    success_prior: float,
) -> None:
    _refresh_no_demo_rsi_success_rates(
        success_rate, success_ema, attempt_ema, success_prior)
    if tracked_stage_max < 0 or env_ids.numel() == 0:
        return

    prev_stage_subset = last_reset_stage[env_ids]
    valid_mask = last_reset_valid[env_ids] & (prev_stage_subset >= 0) & (
        prev_stage_subset <= tracked_stage_max)
    last_reset_valid[env_ids] = False

    num_stages = int(attempt_ema.numel())
    previous_stage = prev_stage_subset[valid_mask]
    current_stage = command.stage_id[env_ids][valid_mask]
    success_mask = current_stage >= (previous_stage + 1)
    final_stage = int(getattr(command.cfg, "num_stages", 1)) - 1
    if final_stage >= 0:
        final_success = command.is_cube_at_target()[env_ids][valid_mask]
        success_mask = success_mask | (
            (previous_stage == final_stage) & final_success)

    attempts_device = torch.bincount(previous_stage, minlength=num_stages)
    successes_device = torch.bincount(
        previous_stage[success_mask], minlength=num_stages)
    counts_cpu = torch.stack([attempts_device, successes_device]).to(
        device=torch.device("cpu"), dtype=torch.float32
    )
    attempts_f = counts_cpu[0]
    successes_f = counts_cpu[1]
    touched_cpu = attempts_f > 0
    if touched_cpu.any():
        decay = float(stats_decay)
        attempt_ema[touched_cpu] = attempt_ema[touched_cpu] * \
            decay + attempts_f[touched_cpu]
        success_ema[touched_cpu] = success_ema[touched_cpu] * \
            decay + successes_f[touched_cpu]
        _refresh_no_demo_rsi_success_rates(
            success_rate, success_ema, attempt_ema, success_prior)


def _score_no_demo_rsi_stage0_transition_metrics(
    env_ids: torch.Tensor,
    command,
    last_reset_stage: torch.Tensor,
    last_reset_valid: torch.Tensor,
    attempt_ema: torch.Tensor,
    success_ema: torch.Tensor,
    success_rate: torch.Tensor,
    stats_decay: float,
    success_prior: float,
) -> None:
    _refresh_no_demo_rsi_success_rates(
        success_rate, success_ema, attempt_ema, success_prior)
    if env_ids.numel() == 0:
        return

    prev_stage_subset = last_reset_stage[env_ids]
    valid_env_ids = env_ids[last_reset_valid[env_ids]
                            & (prev_stage_subset == 0)]
    if valid_env_ids.numel() == 0:
        return

    flags = getattr(command, "stage0_transition_episode_flags", None)
    if not isinstance(flags, dict):
        return

    success_counts = []
    for key, _ in STAGE0_TO_STAGE1_RATE_SPECS:
        value = flags.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() != command.num_envs:
            success_counts.append(torch.zeros(
                (), device=env_ids.device, dtype=torch.float32))
            continue
        success_counts.append(value[valid_env_ids].float().sum())

    successes_f = torch.stack(success_counts).to(
        device=torch.device("cpu"), dtype=torch.float32)
    attempts_f = torch.full_like(successes_f, float(valid_env_ids.numel()))
    touched_cpu = attempts_f > 0
    if touched_cpu.any():
        decay = float(stats_decay)
        attempt_ema[touched_cpu] = attempt_ema[touched_cpu] * \
            decay + attempts_f[touched_cpu]
        success_ema[touched_cpu] = success_ema[touched_cpu] * \
            decay + successes_f[touched_cpu]
        _refresh_no_demo_rsi_success_rates(
            success_rate, success_ema, attempt_ema, success_prior)


def _compute_no_demo_rsi_availability_weight(
    *,
    stage: int,
    size_k: int,
    min_req: int,
    buffer_capacity: int,
    alpha: float,
) -> float:
    """Compressed availability weight shared by stage-0 and RSI-buffer stages.
    """

    cap = max(int(buffer_capacity), 1)
    if stage == 0:
        effective_size = cap
    else:
        if size_k < min_req or size_k <= 0:
            return 0.0
        effective_size = min(max(int(size_k), 1), cap)

    if alpha == 0.0:
        return 1.0
    return float(math.log1p(float(effective_size)) ** float(alpha))


def _compute_no_demo_rsi_adaptive_probabilities(
    *,
    num_stages: int,
    adaptive_stage_max: int,
    max_stage_allowed: int,
    min_samples: list[int],
    buffer_sizes_list: list[int],
    buffer_capacity: int,
    alpha: float,
    success_rate: torch.Tensor,
    difficulty_power: float,
    difficulty_floor: float,
    consolidation_mode: bool,
    consolidation_stage0_share: float,
) -> torch.Tensor:
    probs = torch.zeros(num_stages, dtype=torch.float32)
    if num_stages <= 0:
        return probs

    floor = max(float(difficulty_floor), 0.0)
    power = float(difficulty_power)
    stage0_difficulty = max(
        1.0 - float(success_rate[0].item()), floor) ** power
    stage0_availability = _compute_no_demo_rsi_availability_weight(
        stage=0,
        size_k=0,
        min_req=0,
        buffer_capacity=buffer_capacity,
        alpha=alpha,
    )

    eligible_stages: list[int] = []
    eligible_weights: list[float] = []
    for stage in range(1, adaptive_stage_max + 1):
        if stage > max_stage_allowed:
            continue
        min_req = min_samples[stage] if stage < len(
            min_samples) else min_samples[-1]
        size_k = buffer_sizes_list[stage] if stage < len(
            buffer_sizes_list) else 0
        difficulty = max(
            1.0 - float(success_rate[stage].item()), floor) ** power
        size_weight = _compute_no_demo_rsi_availability_weight(
            stage=stage,
            size_k=size_k,
            min_req=min_req,
            buffer_capacity=buffer_capacity,
            alpha=alpha,
        )
        raw_weight = difficulty * size_weight
        if raw_weight <= 0.0:
            continue
        eligible_stages.append(stage)
        eligible_weights.append(raw_weight)

    if consolidation_mode and eligible_stages:
        stage0_share = min(max(float(consolidation_stage0_share), 0.0), 1.0)
        probs[0] = stage0_share
        residual_mass = max(0.0, 1.0 - stage0_share)
        total_weight = sum(eligible_weights)
        if total_weight <= 0.0:
            for stage in eligible_stages:
                probs[stage] = residual_mass / len(eligible_stages)
        else:
            for stage, raw_weight in zip(eligible_stages, eligible_weights, strict=True):
                probs[stage] = residual_mass * (raw_weight / total_weight)
        return probs

    raw_weights = [(0, stage0_difficulty * stage0_availability)]
    raw_weights.extend(zip(eligible_stages, eligible_weights, strict=True))
    total_weight = sum(weight for _, weight in raw_weights if weight > 0.0)
    if total_weight <= 0.0:
        probs[0] = 1.0
        return probs

    for stage, raw_weight in raw_weights:
        if raw_weight > 0.0:
            probs[stage] = raw_weight / total_weight
    return probs


def _compute_no_demo_rsi_nonadaptive_probabilities(
    *,
    num_stages: int,
    min_samples: list[int],
    buffer_sizes_list: list[int],
    alpha: float,
    p_stage0: float,
    stage_weights: list[float],
) -> torch.Tensor:
    probs = torch.zeros(num_stages, dtype=torch.float32)
    if num_stages <= 0:
        return probs

    eligible: list[int] = []
    weights: list[float] = []
    for stage in range(1, num_stages):
        min_req = min_samples[stage] if stage < len(
            min_samples) else min_samples[-1]
        size_k = buffer_sizes_list[stage] if stage < len(
            buffer_sizes_list) else 0
        required_samples = max(int(min_req), 1)
        if size_k < required_samples:
            continue
        stage_weight = float(stage_weights[stage]) if stage < len(
            stage_weights) else 1.0
        if stage_weight <= 0.0:
            continue
        weight = (float(size_k**alpha) if alpha != 0.0 else 1.0) * stage_weight
        if weight <= 0.0:
            continue
        eligible.append(stage)
        weights.append(weight)

    if not eligible:
        probs[0] = 1.0
        return probs

    stage0_share = min(max(float(p_stage0), 0.0), 1.0)
    probs[0] = stage0_share
    residual_mass = max(0.0, 1.0 - stage0_share)
    total_weight = sum(weights)
    if total_weight <= 0.0:
        probs[0] = 1.0
        return probs
    for stage, weight in zip(eligible, weights, strict=True):
        probs[stage] = residual_mass * (weight / total_weight)
    return probs


def _compute_no_demo_rsi_reward_metric_unlocked_mask(
    *,
    env,
    rsi_cfg: NoDemoRSICfg,
    num_stages: int,
    reset_counts: torch.Tensor,
) -> torch.Tensor:
    unlocked_mask = torch.zeros((num_stages,), dtype=torch.bool)
    if num_stages <= 0:
        return unlocked_mask

    max_stage_allowed = 0
    total_resets = int(reset_counts.sum().item()) if isinstance(
        reset_counts, torch.Tensor) else 0
    key_s1 = getattr(rsi_cfg, "warmup_stage1_reward_key",
                     "episode_reward/robot_object_distance")
    thr_s1 = float(getattr(rsi_cfg, "warmup_stage1_threshold", 7.0))
    val_s1 = _get_episode_reward_metric(env, key_s1)
    allow_stage1 = val_s1 is not None and val_s1 > thr_s1
    allow_stage1 = allow_stage1 or (total_resets >= int(
        getattr(rsi_cfg, "warmup_stage1_min_resets", 0)))

    key_s2 = getattr(rsi_cfg, "warmup_stage2_reward_key",
                     "episode_reward/cube_lift_height")
    thr_s2 = float(getattr(rsi_cfg, "warmup_stage2_threshold", 1.0))
    val_s2 = _get_episode_reward_metric(env, key_s2)
    allow_stage2_plus = val_s2 is not None and val_s2 > thr_s2
    allow_stage2_plus = allow_stage2_plus or (total_resets >= int(
        getattr(rsi_cfg, "warmup_stage2_min_resets", 0)))

    if allow_stage1:
        max_stage_allowed = max(max_stage_allowed, 1)
    if allow_stage2_plus:
        max_stage_allowed = num_stages - 1

    unlocked_mask[: max_stage_allowed + 1] = True
    return unlocked_mask


def _update_no_demo_rsi_prev_stage_unlocks(
    *,
    unlocked_mask: torch.Tensor,
    success_rate: torch.Tensor,
    buffer_sizes_list: list[int],
    min_samples: list[int],
    success_threshold: float,
) -> None:
    if unlocked_mask.numel() == 0:
        return

    unlocked_mask[0] = True
    for stage in range(1, int(unlocked_mask.numel())):
        if bool(unlocked_mask[stage].item()):
            continue
        if not bool(unlocked_mask[stage - 1].item()):
            break
        min_req = min_samples[stage] if stage < len(
            min_samples) else min_samples[-1]
        size_k = buffer_sizes_list[stage] if stage < len(
            buffer_sizes_list) else 0
        required_samples = max(int(min_req), 1)
        if size_k < required_samples:
            break
        if float(success_rate[stage - 1].item()) < float(success_threshold):
            break
        unlocked_mask[stage] = True


def _normalize_no_demo_rsi_probabilities(probs: torch.Tensor) -> torch.Tensor:
    if probs.numel() == 0:
        return probs
    total = float(probs.sum().item())
    if total <= 0.0:
        probs.zero_()
        probs[0] = 1.0
        return probs
    probs /= total
    return probs


def _apply_no_demo_rsi_stage_sampling_constraints(
    *,
    raw_probs: torch.Tensor,
    unlocked_mask: torch.Tensor,
    apply_ramp: bool,
    ramp_attempt_count: torch.Tensor | None,
    ramp_initial_share: float,
    ramp_attempts: int,
) -> torch.Tensor:
    probs = raw_probs.clone()
    if probs.numel() == 0:
        return probs

    gate_mask = unlocked_mask.to(device=probs.device, dtype=torch.bool)
    if gate_mask.numel() != probs.numel():
        raise ValueError(
            "Warmup unlocked mask shape does not match RSI stage probability shape.")
    gate_mask[0] = True
    probs[~gate_mask] = 0.0
    _normalize_no_demo_rsi_probabilities(probs)
    if not apply_ramp or ramp_attempt_count is None:
        return probs

    initial_share = min(max(float(ramp_initial_share), 0.0), 1.0)
    ramp_span = int(ramp_attempts)
    if ramp_span <= 0:
        return probs

    caps = torch.ones_like(probs)
    counts = ramp_attempt_count.to(device=probs.device, dtype=torch.float32)
    progress = torch.clamp(counts / float(ramp_span), min=0.0, max=1.0)
    caps[1:] = initial_share + (1.0 - initial_share) * progress[1:]
    caps[~gate_mask] = 0.0

    base_weights = probs.clone()
    for _ in range(int(probs.numel())):
        overflow_mask = probs > (caps + 1.0e-6)
        if not bool(overflow_mask.any().item()):
            break

        excess = torch.clamp(probs[overflow_mask] -
                             caps[overflow_mask], min=0.0).sum()
        probs[overflow_mask] = caps[overflow_mask]
        recipient_mask = gate_mask & (~overflow_mask)
        weights = torch.where(recipient_mask, base_weights,
                              torch.zeros_like(base_weights))
        weight_sum = float(weights.sum().item())
        if weight_sum <= 0.0:
            if bool(recipient_mask[0].item()):
                probs[0] += excess
            else:
                probs.zero_()
                probs[0] = 1.0
            break
        probs += excess * (weights / weight_sum)

    return _normalize_no_demo_rsi_probabilities(probs)


def _get_episode_reward_metric(env, key: str) -> float | None:
    """Fetch an episode-level reward metric from env.extras, if present."""
    extras = getattr(env, "extras", None)
    if not isinstance(extras, dict):
        return None
    candidates = [key, key.lower()]
    if "/" in key:
        suffix = key.split("/", 1)[1]
        candidates.extend(
            [
                suffix,
                suffix.lower(),
                f"episode_reward/{suffix}",
                f"Episode_Reward/{suffix}",
            ]
        )
    containers = [extras]
    episode_dict = extras.get("episode") if isinstance(extras, dict) else None
    if isinstance(episode_dict, dict):
        containers.append(episode_dict)
        reward_dict = episode_dict.get("reward")
        if isinstance(reward_dict, dict):
            containers.append(reward_dict)
    for container in containers:
        if not isinstance(container, dict):
            continue
        for name in candidates:
            if name in container:
                val = container[name]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
    return None


def reset_scene_with_no_demo_rsi(
    env,
    env_ids: torch.Tensor | None,
    rsi_cfg: NoDemoRSICfg,
    base_reset_func=reset_scene_to_default,
    reset_joint_targets: bool = True,
    command_name: str = "stage",
    fixed_stage: int | None = None,
    **base_reset_kwargs,
):
    """Reset scene with optional no-demo RSI sampling."""
    env_ids_t = _resolve_env_ids(env, env_ids)
    if env_ids_t.numel() == 0:
        return
    if fixed_stage is not None:
        fixed_stage = int(fixed_stage)
        if fixed_stage < 0:
            fixed_stage = None

    base_kwargs = dict(base_reset_kwargs)
    nested_base_kwargs = base_kwargs.pop("base_reset_kwargs", None)
    if isinstance(nested_base_kwargs, dict):
        base_kwargs.update(nested_base_kwargs)
    base_kwargs.setdefault("reset_joint_targets", reset_joint_targets)

    stage_overrides = getattr(env, "_no_demo_rsi_stage_on_reset", None)
    settle_buf = getattr(env, "_no_demo_rsi_settle_steps", None)
    last_reset_stage = getattr(env, "_no_demo_rsi_last_reset_stage", None)
    last_reset_valid = getattr(env, "_no_demo_rsi_last_reset_valid", None)
    current_sample_prob = getattr(
        env, "_no_demo_rsi_current_sample_prob", None)
    consolidation_mode = getattr(
        env, "_no_demo_rsi_adaptive_consolidation_mode", None)

    if fixed_stage is not None and fixed_stage > 0 and not (rsi_cfg and getattr(rsi_cfg, "enabled", False)):
        raise ValueError("fixed_stage > 0 requires an enabled NoDemoRSICfg.")

    if not (rsi_cfg and getattr(rsi_cfg, "enabled", False)):
        if isinstance(stage_overrides, torch.Tensor):
            stage_overrides[env_ids_t] = 0
        if isinstance(settle_buf, torch.Tensor):
            settle_buf[env_ids_t] = 0
        if isinstance(last_reset_stage, torch.Tensor):
            last_reset_stage[env_ids_t] = 0
        if isinstance(last_reset_valid, torch.Tensor):
            last_reset_valid[env_ids_t] = False
        if isinstance(current_sample_prob, torch.Tensor):
            current_sample_prob.zero_()
        if isinstance(consolidation_mode, torch.Tensor):
            consolidation_mode.fill_(False)
        base_reset_func(env, env_ids_t, **base_kwargs)
        return

    command = env.command_manager.get_term(command_name)
    num_stages = int(getattr(command.cfg, "num_stages", len(
        getattr(rsi_cfg, "min_samples_to_enable", [])) or 0))
    if num_stages <= 0:
        num_stages = max(1, len(getattr(rsi_cfg, "min_samples_to_enable", [])))
    if fixed_stage is not None and fixed_stage >= num_stages:
        raise ValueError(
            f"fixed_stage must be in [0, {num_stages - 1}], got {fixed_stage}.")
    stage0_rate_specs = STAGE0_TO_STAGE1_RATE_SPECS

    (
        buffer,
        stage_overrides,
        settle_buf,
        reset_counts,
        last_reset_stage,
        last_reset_valid,
        attempt_ema,
        success_ema,
        success_rate,
        stage0_transition_attempt_ema,
        stage0_transition_success_ema,
        stage0_transition_rate,
        current_sample_prob,
        consolidation_mode,
        warmup_unlocked_mask,
        ramp_attempt_count,
    ) = _prepare_no_demo_rsi_state(
        env,
        num_stages=num_stages,
        capacity=int(rsi_cfg.capacity_per_stage),
        settle_steps=int(rsi_cfg.settle_steps),
        stage0_rate_count=len(stage0_rate_specs),
    )
    forced_stage_enabled = fixed_stage is not None
    adaptive_enabled = bool(getattr(
        rsi_cfg, "adaptive_stage_sampling_enabled", False)) and (not forced_stage_enabled)
    warmup_enabled = bool(getattr(rsi_cfg, "warmup_enabled", False))
    warmup_mode = str(getattr(rsi_cfg, "warmup_mode", "reward_metric"))
    use_prev_stage_success_warmup = warmup_enabled and (
        warmup_mode == "prev_stage_success") and (not forced_stage_enabled)
    adaptive_stage_cap = int(getattr(rsi_cfg, "adaptive_stage_max", 3))
    adaptive_stage_max = min(num_stages - 1, max(adaptive_stage_cap, 0))
    tracked_stage_max = (
        num_stages - 1) if use_prev_stage_success_warmup else adaptive_stage_max
    success_tracking_enabled = (
        adaptive_enabled or use_prev_stage_success_warmup) and tracked_stage_max >= 0
    success_prior = float(getattr(rsi_cfg, "adaptive_success_prior", 16.0))
    _score_no_demo_rsi_stage0_transition_metrics(
        env_ids_t,
        command,
        last_reset_stage,
        last_reset_valid,
        stage0_transition_attempt_ema,
        stage0_transition_success_ema,
        stage0_transition_rate,
        stats_decay=float(getattr(rsi_cfg, "adaptive_stats_decay", 0.995)),
        success_prior=success_prior,
    )
    if success_tracking_enabled:
        _score_no_demo_rsi_previous_episodes(
            env_ids_t,
            command,
            last_reset_stage,
            last_reset_valid,
            attempt_ema,
            success_ema,
            success_rate,
            tracked_stage_max=tracked_stage_max,
            stats_decay=float(getattr(rsi_cfg, "adaptive_stats_decay", 0.995)),
            success_prior=success_prior,
        )
    else:
        _refresh_no_demo_rsi_success_rates(
            success_rate, success_ema, attempt_ema, success_prior)
        success_rate.zero_()
        last_reset_valid[env_ids_t] = False
    if not adaptive_enabled:
        consolidation_mode.fill_(False)

    min_samples = list(getattr(rsi_cfg, "min_samples_to_enable", []))
    if len(min_samples) < num_stages:
        if min_samples:
            min_samples.extend([min_samples[-1]] *
                               (num_stages - len(min_samples)))
        else:
            min_samples = [0] * num_stages

    alpha = float(getattr(rsi_cfg, "alpha", 1.0))
    buffer_sizes_list = [int(x) for x in buffer.sizes().tolist()]
    unlocked_mask = torch.ones((num_stages,), dtype=torch.bool)
    apply_warmup_ramp = False
    if warmup_enabled and not forced_stage_enabled:
        if use_prev_stage_success_warmup:
            _update_no_demo_rsi_prev_stage_unlocks(
                unlocked_mask=warmup_unlocked_mask,
                success_rate=success_rate,
                buffer_sizes_list=buffer_sizes_list,
                min_samples=min_samples,
                success_threshold=float(
                    getattr(rsi_cfg, "warmup_success_threshold", 0.6)),
            )
            unlocked_mask = warmup_unlocked_mask.clone()
            apply_warmup_ramp = True
        else:
            unlocked_mask = _compute_no_demo_rsi_reward_metric_unlocked_mask(
                env=env,
                rsi_cfg=rsi_cfg,
                num_stages=num_stages,
                reset_counts=reset_counts,
            )

    if forced_stage_enabled:
        stage_choices = torch.full(
            (env_ids_t.numel(),), int(fixed_stage), device=env.device, dtype=torch.long
        )
        current_sample_prob.zero_()
        current_sample_prob[fixed_stage] = 1.0
        last_reset_valid[env_ids_t] = False
        consolidation_mode.fill_(False)
    else:
        if adaptive_enabled:
            tracked_stage_count = min(num_stages, adaptive_stage_max + 1)
            enter_threshold = float(
                getattr(rsi_cfg, "adaptive_consolidation_enter", 0.80))
            exit_threshold = float(
                getattr(rsi_cfg, "adaptive_consolidation_exit", 0.75))
            # success_rate, consolidation_mode are CPU tensors → .item() is sync-free.
            consolidation_active = bool(consolidation_mode.item())
            if tracked_stage_count > 0:
                tracked_rates = success_rate[:tracked_stage_count]
                if consolidation_active:
                    if bool((tracked_rates < exit_threshold).any().item()):
                        consolidation_mode.fill_(False)
                        consolidation_active = False
                else:
                    if bool((tracked_rates >= enter_threshold).all().item()):
                        consolidation_mode.fill_(True)
                        consolidation_active = True

            raw_stage_probs = _compute_no_demo_rsi_adaptive_probabilities(
                num_stages=num_stages,
                adaptive_stage_max=adaptive_stage_max,
                max_stage_allowed=num_stages - 1,
                min_samples=min_samples,
                buffer_sizes_list=buffer_sizes_list,
                buffer_capacity=int(getattr(rsi_cfg, "capacity_per_stage", 1)),
                alpha=alpha,
                success_rate=success_rate,
                difficulty_power=float(
                    getattr(rsi_cfg, "adaptive_difficulty_power", 1.0)),
                difficulty_floor=float(
                    getattr(rsi_cfg, "adaptive_difficulty_floor", 0.10)),
                consolidation_mode=consolidation_active,
                consolidation_stage0_share=float(
                    getattr(rsi_cfg, "adaptive_consolidation_stage0_share", 0.85)),
            )
        else:
            stage_weights = list(getattr(rsi_cfg, "stage_sampling_weight", []))
            if len(stage_weights) < num_stages:
                if stage_weights:
                    stage_weights.extend(
                        [stage_weights[-1]] * (num_stages - len(stage_weights)))
                else:
                    stage_weights = [1.0] * num_stages
            raw_stage_probs = _compute_no_demo_rsi_nonadaptive_probabilities(
                num_stages=num_stages,
                min_samples=min_samples,
                buffer_sizes_list=buffer_sizes_list,
                alpha=alpha,
                p_stage0=float(getattr(rsi_cfg, "p_stage0", 0.0)),
                stage_weights=stage_weights,
            )

        sampled_stage_probs = _apply_no_demo_rsi_stage_sampling_constraints(
            raw_probs=raw_stage_probs,
            unlocked_mask=unlocked_mask,
            apply_ramp=apply_warmup_ramp,
            ramp_attempt_count=ramp_attempt_count if apply_warmup_ramp else None,
            ramp_initial_share=float(
                getattr(rsi_cfg, "warmup_ramp_initial_share", 0.05)),
            ramp_attempts=int(getattr(rsi_cfg, "warmup_ramp_attempts", 1000)),
        )
        current_sample_prob.copy_(sampled_stage_probs)
        sampled_probs_gpu = sampled_stage_probs.to(
            device=env.device, non_blocking=True)
        stage_choices = torch.multinomial(
            sampled_probs_gpu, env_ids_t.numel(), replacement=True
        ).to(dtype=torch.long)

    stage_overrides[env_ids_t] = stage_choices
    last_reset_stage[env_ids_t] = stage_choices
    if forced_stage_enabled:
        last_reset_valid[env_ids_t] = False
    elif success_tracking_enabled:
        last_reset_valid[env_ids_t] = stage_choices <= tracked_stage_max
    else:
        last_reset_valid[env_ids_t] = False
    if hasattr(command, "reset_stage0_transition_episode_flags"):
        command.reset_stage0_transition_episode_flags(env_ids_t)
    settle_steps_val = int(rsi_cfg.settle_steps)
    if settle_steps_val > 0:
        settle_buf[env_ids_t] = torch.where(
            stage_choices > 0, settle_steps_val, 0)
    else:
        settle_buf[env_ids_t] = 0

    counts_cpu = torch.bincount(stage_choices, minlength=num_stages).cpu()
    reset_counts[: counts_cpu.numel()] += counts_cpu
    if apply_warmup_ramp and not forced_stage_enabled:
        ramp_attempt_count += counts_cpu

    stage0_envs_tensor = env_ids_t[stage_choices == 0] if int(
        counts_cpu[0]) > 0 else env_ids_t[:0]
    rsi_stage_map: dict[int, torch.Tensor] = {}
    for stage in range(1, num_stages):
        if int(counts_cpu[stage]) == 0:
            continue
        rsi_stage_map[stage] = env_ids_t[stage_choices == stage]

    if stage0_envs_tensor.numel() > 0:
        env_subset = stage0_envs_tensor
        base_reset_func(env, env_subset, **base_kwargs)

    for stage, env_subset in rsi_stage_map.items():
        if env_subset.numel() == 0:
            continue
        try:
            snapshots = buffer.sample(
                stage, env_subset.numel(), device=env.device)
        except Exception as err:
            if forced_stage_enabled:
                print(
                    f"[WARN] Requested reset_stage={stage}, but its RSI buffer is empty or unavailable "
                    f"({err}). Falling back to stage 0 reset."
                )
            stage_overrides[env_subset] = 0
            settle_buf[env_subset] = 0
            last_reset_stage[env_subset] = 0
            if success_tracking_enabled:
                last_reset_valid[env_subset] = True
            else:
                last_reset_valid[env_subset] = False
            n_fallback = int(env_subset.numel())
            reset_counts[stage] -= n_fallback
            reset_counts[0] += n_fallback
            base_reset_func(env, env_subset, **base_kwargs)
            continue
        apply_locomanip_snapshot(
            env,
            env_subset,
            snapshots,
            command_name=command_name,
            reset_joint_targets=reset_joint_targets,
        )


def reset_fridge_to_pregrasp_demo_frame(
    env,
    env_ids,
    *,
    pose_range: dict[str, tuple[float, float]] | None = None,
    joint_position_range: tuple[float, float] | None = None,
    reset_joint_targets: bool = True,
):
    """Reset the fridge env to the static pre-grasp pose in fridge_reset_states."""
    from coordex.tasks.locomanip.fridge_reset_states import (
        PREGRASP_FRIDGE_JOINT_POS,
        PREGRASP_ROBOT_JOINT_POS,
        PREGRASP_ROOT_POS,
        PREGRASP_ROOT_QUAT_WXYZ,
    )

    env_ids = _resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return

    robot: Articulation = env.scene["robot"]
    fridge: Articulation = env.scene["fridge"]
    device = env.device
    n = int(env_ids.numel())
    env_origins = env.scene.env_origins[env_ids]

    # Robot root pose in world frame.
    root_pos = torch.tensor(PREGRASP_ROOT_POS, device=device,
                            dtype=torch.float32).expand(n, 3).clone()
    root_pos = root_pos + env_origins
    root_quat = torch.tensor(
        PREGRASP_ROOT_QUAT_WXYZ, device=device, dtype=torch.float32).expand(n, 4).clone()
    if pose_range is not None:
        pos_noise, ori_noise = _sample_root_pose_noise(device, n, pose_range)
        root_pos = root_pos + pos_noise
        noise_quat = quat_from_euler_xyz(
            ori_noise[:, 0], ori_noise[:, 1], ori_noise[:, 2])
        root_quat = quat_mul(root_quat, noise_quat)
    root_pose = torch.cat([root_pos, root_quat], dim=-1)
    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(
        (n, 6), device=device), env_ids=env_ids)

    # Robot joint state.
    joint_pos = (
        torch.tensor(PREGRASP_ROBOT_JOINT_POS,
                     device=device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(n, -1)
        .clone()
    )
    if joint_position_range is not None:
        lo, hi = float(joint_position_range[0]), float(joint_position_range[1])
        joint_pos = joint_pos + torch.empty_like(joint_pos).uniform_(lo, hi)
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    if reset_joint_targets:
        robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)
        robot.write_data_to_sim()

    # Fridge joint state.
    fridge_joint_pos = (
        torch.tensor(PREGRASP_FRIDGE_JOINT_POS,
                     device=device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(n, -1)
        .clone()
    )
    fridge_joint_vel = torch.zeros_like(fridge_joint_pos)
    fridge.write_joint_state_to_sim(
        fridge_joint_pos, fridge_joint_vel, env_ids=env_ids)
    if reset_joint_targets:
        fridge.set_joint_position_target(fridge_joint_pos, env_ids=env_ids)
        fridge.set_joint_velocity_target(fridge_joint_vel, env_ids=env_ids)
        fridge.write_data_to_sim()

    clear_prior_obs_step_cache(env)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    offset_sync_mode: Literal["action", "legacy_body"] = "action",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    asset.data.default_joint_pos_nominal = torch.clone(
        asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    env_ids_action = env_ids if isinstance(env_ids, slice) else env_ids.clone()

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(
            asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        action_term = env.action_manager.get_term("joint_pos")

        offset = action_term._offset
        action_joint_ids = getattr(action_term, "_joint_ids", slice(None))
        # normalize joint id containers to tensors when needed

        def _to_tensor_ids(ids, device):
            if isinstance(ids, slice):
                return None
            if isinstance(ids, torch.Tensor):
                return ids.to(device=device, dtype=torch.long)
            return torch.tensor(list(ids), device=device, dtype=torch.long)

        action_env_ids = env_ids_action
        if not isinstance(action_env_ids, slice) and action_env_ids.ndim > 1:
            action_env_ids = action_env_ids.squeeze(-1)

        rand_joint_ids = _to_tensor_ids(joint_ids, offset.device)
        # action index -> joint id mapping for the action term
        if isinstance(action_joint_ids, slice):
            action_joint_ids_tensor = torch.arange(
                offset.shape[1], device=offset.device, dtype=torch.long)
        else:
            action_joint_ids_tensor = _to_tensor_ids(
                action_joint_ids, offset.device)
            if action_joint_ids_tensor is None:
                action_joint_ids_tensor = torch.arange(
                    offset.shape[1], device=offset.device, dtype=torch.long)

        # choose the action indices that correspond to randomized joint ids
        if rand_joint_ids is None:
            action_idx = torch.arange(
                action_joint_ids_tensor.numel(), device=offset.device, dtype=torch.long)
        else:
            mask = torch.isin(action_joint_ids_tensor, rand_joint_ids)
            action_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if action_idx.numel() == 0:
                return

        if offset_sync_mode == "legacy_body":
            # Apply offsets in ascending joint-id order to preserve legacy body-only behavior.
            action_idx_sorted = torch.sort(action_idx).values
            legacy_joint_ids = torch.sort(
                action_joint_ids_tensor[action_idx]).values
            if isinstance(action_env_ids, slice):
                new_offset = asset.data.default_joint_pos[:, legacy_joint_ids].to(
                    offset.device)
                offset[:, action_idx_sorted] = new_offset
            else:
                new_offset = asset.data.default_joint_pos[action_env_ids][:, legacy_joint_ids].to(
                    offset.device)
                offset[action_env_ids[:, None], action_idx_sorted] = new_offset
            return

        joint_ids_update = action_joint_ids_tensor[action_idx]
        if isinstance(action_env_ids, slice):
            new_offset = asset.data.default_joint_pos[:, joint_ids_update].to(
                offset.device)
            offset[:, action_idx] = new_offset
        else:
            new_offset = asset.data.default_joint_pos[action_env_ids][:, joint_ids_update].to(
                offset.device)
            offset[action_env_ids[:, None], action_idx] = new_offset


def randomize_rigid_object_default_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Persistently perturb a rigid object's default root pose per env.

    Designed for startup-mode randomization: each env's bottle (or other rigid
    object) gets its own spawn offset, fixed across all episode resets within
    that env. The function modifies ``asset.data.default_root_state`` directly,
    so subsequent ``reset_scene_to_default`` calls read the perturbed pose
    rather than the original spawn pose. The perturbed pose is also written
    into the live sim so the very first observation already reflects it.

    ``pose_range`` follows the same key convention as IsaacLab's
    ``reset_root_state_uniform`` (``x``, ``y``, ``z``, ``roll``, ``pitch``,
    ``yaw``); values are tuples ``(min, max)`` interpreted in env-local
    coordinates. Velocities are not perturbed.
    """
    from isaaclab.assets import RigidObject

    asset: RigidObject = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids_t = torch.arange(
            env.scene.num_envs, device=asset.device, dtype=torch.long)
    elif isinstance(env_ids, torch.Tensor):
        env_ids_t = env_ids.to(device=asset.device, dtype=torch.long)
    else:
        env_ids_t = torch.as_tensor(
            env_ids, device=asset.device, dtype=torch.long)
    if env_ids_t.numel() == 0:
        return

    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in keys]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(
        ranges[:, 0], ranges[:, 1], (env_ids_t.numel(), 6), device=asset.device
    )

    # Position: in-place add to env-local default position.
    asset.data.default_root_state[env_ids_t, 0:3] += rand_samples[:, 0:3]
    # Orientation: compose only if any rotation range is non-zero (the common
    # bottle-on-table case is position-only, so we skip the quat math then).
    has_rotation = any(abs(low) > 0.0 or abs(
        high) > 0.0 for low, high in range_list[3:])
    if has_rotation:
        delta = quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        asset.data.default_root_state[env_ids_t, 3:7] = quat_mul(
            asset.data.default_root_state[env_ids_t, 3:7], delta
        )

    # Push the perturbed default pose into sim (in world frame).
    pose = asset.data.default_root_state[env_ids_t, :7].clone()
    pose[:, 0:3] = pose[:, 0:3] + env.scene.env_origins[env_ids_t]
    asset.write_root_pose_to_sim(pose, env_ids=env_ids_t)


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(
            asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(
            asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)
