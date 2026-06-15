from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import base_ang_vel, base_lin_vel
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from coordex.tasks.locomanip.constants import RIGHT_HAND_TIP_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _sanitize_non_negative(value: torch.Tensor, clip_fill: float) -> torch.Tensor:
    """Keep penalties finite and non-negative before optional clipping."""
    value = torch.nan_to_num(value, nan=float(clip_fill), posinf=float(clip_fill), neginf=0.0)
    return torch.clamp(value, min=0.0)


def _stage_mask(command, stage_id) -> torch.Tensor:
    if stage_id is None:
        return torch.ones_like(command.stage_id, dtype=torch.float32)
    if isinstance(stage_id, torch.Tensor):
        stage_ids = stage_id.to(device=command.stage_id.device, dtype=command.stage_id.dtype).view(-1)
    elif isinstance(stage_id, (list, tuple, set)):
        if len(stage_id) == 0:
            return torch.zeros_like(command.stage_id, dtype=torch.float32)
        stage_ids = torch.tensor(list(stage_id), device=command.stage_id.device, dtype=command.stage_id.dtype)
    else:
        return (command.stage_id == int(stage_id)).float()

    if stage_ids.numel() == 1:
        return (command.stage_id == int(stage_ids.item())).float()
    return (command.stage_id[:, None] == stage_ids[None, :]).any(dim=-1).float()


def _reduce_joint_metric(values: torch.Tensor, aggregation: str) -> torch.Tensor:
    """Reduce per-joint values into a single scalar per environment."""
    if values.ndim <= 1:
        return values
    if aggregation == "mean":
        return values.mean(dim=-1)
    if aggregation in ("max", "worst_joint"):
        return values.max(dim=-1).values
    if aggregation == "sum":
        return values.sum(dim=-1)
    raise ValueError(f"Unsupported joint aggregation '{aggregation}'. Expected one of: mean, max, sum.")


def _resolve_joint_indices(env: ManagerBasedRLEnv, asset, joint_names: list[str], cache_attr: str) -> torch.Tensor:
    cached = getattr(env, cache_attr, None)
    if isinstance(cached, torch.Tensor):
        return cached
    joint_ids, _ = asset.find_joints(joint_names, preserve_order=True)
    if len(joint_ids) == 0:
        raise RuntimeError(f"No joints found for patterns {joint_names}.")
    joint_ids_tensor = torch.as_tensor(joint_ids, device=asset.data.joint_pos.device, dtype=torch.long)
    setattr(env, cache_attr, joint_ids_tensor)
    return joint_ids_tensor


def _resolve_body_index(env: ManagerBasedRLEnv, asset, body_name: str, cache_attr: str) -> int:
    cached = getattr(env, cache_attr, None)
    if cached is not None:
        return int(cached)
    body_ids, _ = asset.find_bodies([body_name], preserve_order=True)
    if len(body_ids) == 0:
        raise RuntimeError(f"Body '{body_name}' not found on asset '{asset.name}'.")
    body_id = int(body_ids[0])
    setattr(env, cache_attr, body_id)
    return body_id


def _resolve_body_indices(env: ManagerBasedRLEnv, asset, body_names: list[str], cache_attr: str) -> torch.Tensor:
    cached = getattr(env, cache_attr, None)
    if isinstance(cached, torch.Tensor):
        return cached
    body_ids, body_found = asset.find_bodies(body_names, preserve_order=True)
    if len(body_ids) != len(body_names):
        missing = sorted(set(body_names) - set(body_found))
        raise RuntimeError(f"Missing bodies on '{asset.name}': {missing}")
    ids = torch.as_tensor(body_ids, device=asset.data.body_pos_w.device, dtype=torch.long)
    setattr(env, cache_attr, ids)
    return ids


def _get_contact_sensor_list(env: ManagerBasedRLEnv, sensor_names: Iterable[str] | str | None) -> list:
    sensors = getattr(env.scene, "sensors", {})
    sensor_list: list = []
    if sensor_names is None:
        return sensor_list
    if isinstance(sensor_names, str):
        sensor = sensors.get(sensor_names) if isinstance(sensors, dict) else None
        if sensor is not None:
            sensor_list.append(sensor)
        return sensor_list

    for name in sensor_names:
        sensor = sensors.get(name) if isinstance(sensors, dict) else None
        if sensor is not None:
            sensor_list.append(sensor)
    return sensor_list


def _sensor_forces(sensor):
    forces = getattr(sensor.data, "force_matrix_w", None)
    if forces is None:
        forces = getattr(sensor.data, "force_matrix_w_history", None)
    if forces is None:
        forces = getattr(sensor.data, "net_forces_w", None)
    if forces is None:
        forces = getattr(sensor.data, "net_forces_w_history", None)
    if isinstance(forces, torch.Tensor):
        return torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)
    return None


def _cube_lift_from_reference(
    env: ManagerBasedRLEnv,
    command,
    cube_name: str = "cube",
    table_name: str | None = None,
    table_top_offset: float | None = None,
) -> torch.Tensor:
    cube = env.scene[cube_name]
    if table_name is not None:
        table = env.scene[table_name]
        reference = table.data.root_pos_w[:, 2]
        if table_top_offset is not None:
            reference = reference + float(table_top_offset)
    else:
        reference = torch.as_tensor(command.cube_init_z, device=cube.data.root_pos_w.device, dtype=cube.data.root_pos_w.dtype)
        if reference.ndim == 0:
            reference = torch.full_like(cube.data.root_pos_w[:, 2], float(reference.item()))
        else:
            reference = reference.to(device=cube.data.root_pos_w.device, dtype=cube.data.root_pos_w.dtype)

    return torch.clamp(cube.data.root_pos_w[:, 2] - reference, min=0.0)


def _new_episode_mask(env: ManagerBasedRLEnv, command) -> torch.Tensor:
    mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    stage_steps = getattr(command, "_stage_steps", None)
    if isinstance(stage_steps, torch.Tensor) and stage_steps.shape[0] == env.num_envs:
        mask = mask | (stage_steps.to(device=env.device) <= 1)
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if isinstance(episode_length_buf, torch.Tensor) and episode_length_buf.shape[0] == env.num_envs:
        mask = mask | (episode_length_buf.to(device=env.device) <= 1)
    return mask


def stage_transition_bonus(env: ManagerBasedRLEnv, command_name: str, bonus: float = 1.0) -> torch.Tensor:
    """One-time bonus when the stage id increases."""

    command = env.command_manager.get_term(command_name)
    current_stage = command.stage_id
    last_stage = getattr(env, "_locomanip_last_stage_reward_stage_id", None)

    if (not isinstance(last_stage, torch.Tensor)) or last_stage.shape != current_stage.shape:
        env._locomanip_last_stage_reward_stage_id = current_stage.clone()
        return torch.zeros_like(current_stage, dtype=torch.float32)

    progressed = (current_stage > last_stage).float() * bonus
    env._locomanip_last_stage_reward_stage_id = current_stage.clone()
    return progressed


def robot_object_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id=None,
    object_name: str = "cube",
    target_distance: float = 0.45,
    scale: float = 4.0,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    obj = env.scene[object_name]
    delta = robot.data.root_pos_w[:, :2] - obj.data.root_pos_w[:, :2]
    dist = torch.norm(delta, dim=-1)
    reward = torch.exp(-scale * (dist - target_distance).pow(2))
    return reward * _stage_mask(command, stage_id)


def robot_grasp_gated_forward_target_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    post_grasp_target_x: float = 2.0,
    pre_grasp_target_x: float | None = None,
    pre_grasp_x_offset: float = 0.0,
    target_y: float | None = 0.0,
    scale: float = 1.0,
    cube_name: str = "bottle",
) -> torch.Tensor:
    """Reward root x target tracking, switching target after grasp."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    root_xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

    if hasattr(command, "is_grasped"):
        grasped = command.is_grasped().to(root_xy.dtype)
    else:
        grasped = torch.zeros(env.num_envs, device=env.device, dtype=root_xy.dtype)

    post_x = root_xy.new_full((env.num_envs,), float(post_grasp_target_x))
    if pre_grasp_target_x is not None:
        pre_x = root_xy.new_full((env.num_envs,), float(pre_grasp_target_x))
    else:
        cube = env.scene[cube_name]
        pre_x = cube.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0] + float(pre_grasp_x_offset)

    target_x = torch.where(grasped > 0.5, post_x, pre_x)

    x_error = root_xy[:, 0] - target_x
    if target_y is None:
        dist_sq = x_error.pow(2)
    else:
        y_error = root_xy[:, 1] - float(target_y)
        dist_sq = x_error.pow(2) + y_error.pow(2)
    reward = torch.exp(-float(scale) * dist_sq)
    return reward * _stage_mask(command, stage_id)


def base_target_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id=None,
    target_name: str = "target_region",
    speed_scale: float = 1.0,
) -> torch.Tensor:
    """Reward world-frame root velocity projected toward a target object."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    target = env.scene[target_name]
    delta_xy = target.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    direction = delta_xy / (delta_xy.norm(dim=-1, keepdim=True) + 1.0e-6)
    vel_xy = robot.data.root_lin_vel_w[:, :2]
    progress = torch.sum(vel_xy * direction, dim=-1)
    reward = torch.clamp(progress / float(speed_scale), min=0.0, max=1.0)
    return reward * _stage_mask(command, stage_id)


def bottle_forward_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    cube_name: str = "bottle",
    speed_scale: float = 1.0,
    require_grasp: bool = True,
    min_lift: float | None = None,
    table_name: str | None = None,
    table_top_offset: float | None = None,
) -> torch.Tensor:
    """Reward positive bottle velocity along env x, optionally gated by grasp/lift."""

    command = env.command_manager.get_term(command_name)
    cube = env.scene[cube_name]
    root_lin_vel_w = getattr(getattr(cube, "data", None), "root_lin_vel_w", None)
    if not isinstance(root_lin_vel_w, torch.Tensor):
        reward = torch.zeros(env.num_envs, device=env.device)
    else:
        reward = torch.clamp(root_lin_vel_w[:, 0] / float(speed_scale), min=0.0, max=1.0)

    if require_grasp and hasattr(command, "is_grasped"):
        reward = reward * command.is_grasped().to(reward.dtype)

    if min_lift is not None:
        lift_height = _cube_lift_from_reference(
            env,
            command,
            cube_name=cube_name,
            table_name=table_name,
            table_top_offset=table_top_offset,
        )
        reward = reward * (lift_height > float(min_lift)).to(reward.dtype)

    return reward * _stage_mask(command, stage_id)


def bottle_upright_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    scale: float = 4.0,
    gate_until_lift: float | None = None,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    if hasattr(command, "bottle_upright_angle"):
        angle = command.bottle_upright_angle()
    else:
        cube = env.scene[command.cfg.cube_name]
        axis = cube.data.root_pos_w.new_tensor((0.0, 0.0, 1.0)).view(1, 3).expand(cube.data.root_quat_w.shape[0], -1)
        world_axis = quat_apply(cube.data.root_quat_w, axis)
        angle = torch.acos(world_axis[:, 2].clamp(-1.0, 1.0))
    reward = torch.exp(-float(scale) * angle.square())
    if gate_until_lift is not None:
        cube = env.scene[getattr(command.cfg, "cube_name", "bottle")]
        lift_height = cube.data.root_pos_w[:, 2] - command.cube_init_z
        reward = reward * (lift_height <= float(gate_until_lift)).to(reward.dtype)
    return reward * _stage_mask(command, stage_id)


def upright_orientation_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]

    body_ids = None
    if getattr(asset_cfg, "body_names", None):
        body_ids = _resolve_body_indices(
            env,
            asset,
            list(asset_cfg.body_names),
            "_locomanip_upright_orientation_body_ids",
        )
    elif getattr(asset_cfg, "body_ids", None):
        body_ids = asset_cfg.body_ids

    if body_ids is not None:
        body_quat = asset.data.body_quat_w[:, body_ids]
        if body_quat.ndim == 2:
            body_quat = body_quat.unsqueeze(1)
        gravity_dir = asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, body_quat.shape[1], -1)
        projected = quat_apply_inverse(body_quat, gravity_dir)
        xy_squared = torch.sum(projected[..., :2] ** 2, dim=-1)
        if xy_squared.ndim > 1:
            xy_squared = xy_squared.mean(dim=-1)
    else:
        xy_squared = torch.sum(asset.data.projected_gravity_b[:, :2] ** 2, dim=-1)

    reward = torch.exp(-xy_squared / (std * std))
    return reward * _stage_mask(command, stage_id)


def heading_forward_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    std: float = 0.3,
    target_yaw: float = 0.0,
) -> torch.Tensor:
    """Reward the robot root facing the configured yaw."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    forward_axis = torch.tensor((1.0, 0.0, 0.0), device=env.device).view(1, 3)
    forward = quat_apply(robot.data.root_quat_w, forward_axis.expand(env.num_envs, -1))
    yaw = torch.atan2(forward[:, 1], forward[:, 0])
    error = yaw - float(target_yaw)
    error = torch.atan2(torch.sin(error), torch.cos(error))
    reward = torch.exp(-error.pow(2) / (float(std) ** 2))
    return reward * _stage_mask(command, stage_id)


def root_y_below_zero_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.0,
    max_penalty: float | None = None,
) -> torch.Tensor:
    """Penalty for root y drifting below the threshold in env frame."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    root_y = robot.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    excess = (float(threshold) - root_y).clamp(min=0.0)
    if max_penalty is not None and max_penalty > 0.0:
        excess = excess.clamp(max=float(max_penalty))
    return excess


def heading_to_object_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id=None,
    object_name: str = "cube",
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    obj = env.scene[object_name]

    delta = obj.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]
    psi_obj = torch.atan2(delta[:, 1], delta[:, 0])

    quat = robot.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    psi_robot = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    diff = psi_obj - psi_robot
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    penalty = (diff / torch.pi) ** 2
    return penalty * _stage_mask(command, stage_id)


def right_hand_height_direction_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    hand_body: str = "right_palm_link",
    fingertip_body_names: Iterable[str] | None = None,
    target_name: str = "cube",
    height_offset: float = 0.07,
    height_scale: float = 40.0,
    gate_far: float | None = 0.6,
    gate_near: float | None = 0.45,
) -> torch.Tensor:
    """Reward the hand mean position tracking a point above the cube."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    target = env.scene[target_name]

    hand_idx = _resolve_body_index(env, robot, hand_body, "_locomanip_rhand_height_dir_idx")
    tip_ids = getattr(command, "fingertip_ids", None)
    if not isinstance(tip_ids, torch.Tensor):
        tip_names = list(fingertip_body_names) if fingertip_body_names is not None else list(RIGHT_HAND_TIP_NAMES)
        tip_ids = _resolve_body_indices(env, robot, tip_names, "_locomanip_rhand_height_dir_tip_ids")
    body_ids = torch.cat(
        [torch.as_tensor([hand_idx], device=tip_ids.device, dtype=tip_ids.dtype), tip_ids], dim=0
    )
    hand_pos = robot.data.body_pos_w[:, body_ids].mean(dim=1)
    target_pos = target.data.root_pos_w
    target_pos = target_pos.clone()
    target_pos[:, 2] = command.cube_init_z + height_offset

    pos_error_sq = torch.sum((hand_pos - target_pos) ** 2, dim=-1)
    pos_reward = torch.exp(-height_scale * pos_error_sq)
    gate = torch.ones_like(pos_reward)

    if gate_far is not None and gate_far > 0.0:
        delta = robot.data.root_pos_w[:, :2] - target.data.root_pos_w[:, :2]
        dist = torch.norm(delta, dim=-1)
        if gate_near is not None and gate_far > gate_near:
            gate = torch.clamp((float(gate_far) - dist) / (float(gate_far) - float(gate_near)), 0.0, 1.0)
        else:
            gate = (dist <= float(gate_far)).to(pos_reward.dtype)
        pos_reward = pos_reward * gate

    return pos_reward * _stage_mask(command, stage_id)


def right_hand_prep_region_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    hand_body: str = "right_palm_link",
    fingertip_body_names: Iterable[str] | None = None,
    target_name: str = "cube",
    xy_scale: float = 60.0,
    z_scale: float = 20.0,
) -> torch.Tensor:
    """Penalty for the hand mean position being outside the prep region."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    target = env.scene[target_name]

    hand_idx = _resolve_body_index(env, robot, hand_body, "_locomanip_rhand_prep_region_idx")
    tip_ids = getattr(command, "fingertip_ids", None)
    if not isinstance(tip_ids, torch.Tensor):
        tip_names = list(fingertip_body_names) if fingertip_body_names is not None else list(RIGHT_HAND_TIP_NAMES)
        tip_ids = _resolve_body_indices(env, robot, tip_names, "_locomanip_rhand_prep_region_tip_ids")

    body_ids = torch.cat(
        [torch.as_tensor([hand_idx], device=tip_ids.device, dtype=tip_ids.dtype), tip_ids], dim=0
    )
    hand_pos = robot.data.body_pos_w[:, body_ids].mean(dim=1)

    hand_xy = hand_pos[:, :2]
    target_xy = target.data.root_pos_w[:, :2]
    hand_xy_dist = torch.norm(hand_xy - target_xy, dim=-1)

    xy_threshold = max(float(getattr(command, "_hand_prep_xy_threshold", 0.0)), 0.0)
    if xy_threshold > 0.0:
        xy_excess = torch.clamp(hand_xy_dist - xy_threshold, min=0.0)
    else:
        xy_excess = torch.zeros_like(hand_xy_dist)

    cube_ref = command.cube_init_z if hasattr(command, "cube_init_z") else target.data.root_pos_w[:, 2]
    height_upper = cube_ref + float(getattr(command, "_hand_prep_height_offset", 0.125))
    z_excess = torch.clamp(hand_pos[:, 2] - height_upper, min=0.0)

    excess = float(xy_scale) * xy_excess.square() + float(z_scale) * z_excess.square()
    penalty = 1.0 - torch.exp(-excess)
    return penalty * _stage_mask(command, stage_id)


def right_hand_flat_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    hand_body: str = "right_palm_link",
    angle_tolerance: float = 0.35,
    scale: float = 8.0,
    palm_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    target_name: str | None = None,
    height_offset: float | None = None,
    pos_gate_far: float | None = None,
) -> torch.Tensor:
    """Bounded penalty for non-flat palm orientation."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]

    hand_idx = _resolve_body_index(env, robot, hand_body, "_locomanip_rhand_flat_penalty_idx")
    hand_quat = robot.data.body_quat_w[:, hand_idx]

    hand_palm_axis = hand_quat.new_tensor(palm_axis).view(1, 3).expand(hand_quat.shape[0], -1)
    world_down = hand_palm_axis.new_tensor((0.0, 0.0, -1.0)).view(1, 3).expand(hand_quat.shape[0], -1)

    hand_normal = quat_apply(hand_quat, hand_palm_axis)
    cos_angle = torch.clamp((hand_normal * world_down).sum(dim=-1), -1.0, 1.0)
    angle = torch.acos(cos_angle)
    penalty = 1.0 - torch.exp(-(angle / angle_tolerance) ** 2)

    if target_name is not None and pos_gate_far is not None and pos_gate_far > 0.0:
        target = env.scene[target_name]
        pos_dist = torch.norm(robot.data.root_pos_w[:, :2] - target.data.root_pos_w[:, :2], dim=-1)
        pos_gate = (pos_dist <= float(pos_gate_far)).float()
        penalty = penalty * pos_gate

    return penalty * _stage_mask(command, stage_id)


def base_lin_vel_excess(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id=None,
    axis: int = 0,
    threshold: float = 0.5,
    max_penalty: float | None = None,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    vel = base_lin_vel(env)
    excess = torch.clamp(torch.abs(vel[:, axis]) - threshold, min=0.0)
    excess = torch.nan_to_num(excess, nan=0.0, posinf=float(max_penalty) if max_penalty else 0.0)
    if max_penalty is not None and max_penalty > 0.0:
        excess = excess.clamp(max=float(max_penalty))
    return excess * _stage_mask(command, stage_id)


def base_ang_vel_excess(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id=None,
    axis: int = 2,
    threshold: float = 0.5,
    max_penalty: float | None = None,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    ang = base_ang_vel(env)
    excess = torch.clamp(torch.abs(ang[:, axis]) - threshold, min=0.0)
    excess = torch.nan_to_num(excess, nan=0.0, posinf=float(max_penalty) if max_penalty else 0.0)
    if max_penalty is not None and max_penalty > 0.0:
        excess = excess.clamp(max=float(max_penalty))
    return excess * _stage_mask(command, stage_id)


def base_velocity_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    linear_axes: list[int] | tuple[int, ...] = (0, 1),
    angular_axis: int | None = None,
    soft_cap: float | None = None,
    max_penalty: float | None = None,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    vel = base_lin_vel(env)
    if linear_axes:
        lin = torch.sum(torch.abs(vel[:, list(linear_axes)]), dim=-1)
    else:
        lin = torch.zeros(env.num_envs, device=vel.device)
    if angular_axis is not None:
        ang = torch.abs(base_ang_vel(env)[:, angular_axis])
        lin = lin + ang
    lin = torch.nan_to_num(lin, nan=0.0, posinf=float(soft_cap or max_penalty or 0.0), neginf=0.0)
    if soft_cap is not None and soft_cap > 0.0:
        lin = float(soft_cap) * torch.tanh(lin / float(soft_cap))
    if max_penalty is not None and max_penalty > 0.0:
        lin = lin.clamp(max=float(max_penalty))
    return lin * _stage_mask(command, stage_id)


def hand_default_pose_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    asset_cfg: SceneEntityCfg,
    tolerance: float = 0.01,
    scale: float = 6.0,
    progress_scale: float = 1.0,
    progress_clip: float | None = None,
    aggregation: str = "mean",
    gate_pre_grasp: bool = False,
) -> torch.Tensor:
    """Reward staying near and moving toward the default hand pose."""

    command = env.command_manager.get_term(command_name)
    stage_mask = _stage_mask(command, stage_id)
    asset = env.scene[asset_cfg.name]
    joint_ids = _resolve_joint_indices(env, asset, asset_cfg.joint_names, "_locomanip_hand_default_joint_ids")
    joint_pos = asset.data.joint_pos[:, joint_ids]
    default_pos = asset.data.default_joint_pos[:, joint_ids]

    deviation = torch.abs(joint_pos - default_pos) - float(tolerance)
    deviation = torch.clamp(deviation, min=0.0)
    reward_pos = torch.exp(-scale * _reduce_joint_metric(deviation, aggregation))

    err = _reduce_joint_metric(torch.abs(joint_pos - default_pos), aggregation)
    prev_err = getattr(env, "_locomanip_hand_default_prev_err", None)
    if (not isinstance(prev_err, torch.Tensor)) or prev_err.shape != err.shape:
        prev_err = err.detach().clone()

    reset_buf = getattr(env, "reset_buf", None)
    if isinstance(reset_buf, torch.Tensor) and reset_buf.shape == err.shape:
        prev_err = torch.where(reset_buf.bool(), err.detach(), prev_err)

    progress = prev_err - err
    if progress_clip is not None and progress_clip > 0.0:
        progress = progress.clamp(min=-float(progress_clip), max=float(progress_clip))
    reward_progress = progress_scale * progress

    env._locomanip_hand_default_prev_err = err.detach()

    reward = reward_pos + reward_progress
    if gate_pre_grasp and hasattr(command, "is_grasped"):
        reward = reward * (~command.is_grasped()).to(reward.dtype)

    return reward * stage_mask


def undesired_table_contacts(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    sensor_names: Iterable[str],
    threshold: float = 1.0,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    if isinstance(sensor_names, str):
        sensor_names = [sensor_names]

    penalty = torch.zeros(env.num_envs, device=env.device)
    for name in sensor_names:
        sensor = env.scene.sensors[name]
        forces = sensor.data.force_matrix_w_history
        if forces is None:
            forces = sensor.data.net_forces_w_history
            if forces is None:
                continue
            contact = forces.norm(dim=-1) > threshold
            in_contact = contact.any(dim=(1, 2))
        else:
            contact = forces.norm(dim=-1) > threshold
            in_contact = contact.any(dim=(1, 2, 3))
        penalty = penalty + in_contact.float()

    return penalty * _stage_mask(command, stage_id)


def termination_penalty(env: ManagerBasedRLEnv, term_name: str) -> torch.Tensor:
    term_mgr = getattr(env, "termination_manager", None)
    if term_mgr is None:
        return torch.zeros(env.num_envs, device=env.device)
    try:
        term = term_mgr.get_term(term_name)
    except KeyError:
        return torch.zeros(env.num_envs, device=env.device)
    return term.float()


def joint_action_rate_l2(env: ManagerBasedRLEnv, action_term_name: str = "joint_pos") -> torch.Tensor:
    """Penalize the rate of change of decoded joint-space actions."""

    actions = getattr(env, "_locomanip_last_action", None)
    if actions is None:
        actions = env.action_manager.get_term(action_term_name).processed_actions

    prev_actions = getattr(env, "_locomanip_prev_joint_action_rate_action", None)
    if (not isinstance(prev_actions, torch.Tensor)) or prev_actions.shape != actions.shape:
        env._locomanip_prev_joint_action_rate_action = actions.detach().clone()
        return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

    reset_buf = getattr(env, "reset_buf", None)
    if isinstance(reset_buf, torch.Tensor) and reset_buf.shape[0] == actions.shape[0]:
        reset_mask = reset_buf.to(device=actions.device, dtype=torch.bool)
        if reset_mask.any():
            prev_actions = prev_actions.clone()
            prev_actions[reset_mask] = actions[reset_mask].detach()

    rate = torch.sum(torch.square(actions - prev_actions), dim=1)
    env._locomanip_prev_joint_action_rate_action = actions.detach().clone()
    return rate


def action_rate_l2_clipped(
    env: ManagerBasedRLEnv,
    clip_max: float = 100.0,
) -> torch.Tensor:
    """Bound raw latent action-rate penalties to avoid catastrophic critic targets."""

    from isaaclab.envs.mdp.rewards import action_rate_l2 as base_fn

    raw = _sanitize_non_negative(base_fn(env), clip_fill=clip_max)
    return torch.clamp(raw, max=float(clip_max))


def joint_action_rate_l2_clipped(
    env: ManagerBasedRLEnv,
    action_term_name: str = "joint_pos",
    clip_max: float = 50.0,
) -> torch.Tensor:
    """Bound decoded joint-action rate penalties while preserving normal-range gradients."""

    raw = _sanitize_non_negative(joint_action_rate_l2(env, action_term_name=action_term_name), clip_fill=clip_max)
    return torch.clamp(raw, max=float(clip_max))


def joint_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared joint velocity magnitude over the selected joints."""

    command = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name]
    joint_ids = _resolve_joint_indices(
        env,
        asset,
        list(asset_cfg.joint_names),
        f"_locomanip_joint_velocity_l2_{asset_cfg.name}_{'_'.join(str(name) for name in asset_cfg.joint_names)}",
    )
    joint_vel = asset.data.joint_vel[:, joint_ids]
    penalty = torch.sum(torch.square(joint_vel), dim=-1)
    return penalty * _stage_mask(command, stage_id)


def joint_velocity_l2_clipped(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stage_id=None,
    clip_max: float = 200.0,
) -> torch.Tensor:
    """Bound squared joint-velocity penalties to keep rare physics spikes finite."""

    raw = _sanitize_non_negative(
        joint_velocity_l2(env, command_name=command_name, stage_id=stage_id, asset_cfg=asset_cfg),
        clip_fill=clip_max,
    )
    return torch.clamp(raw, max=float(clip_max))


def feet_slip_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalize horizontal foot sliding while the foot is in contact."""

    command = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    in_contact = net_forces.norm(dim=-1) > threshold

    robot = env.scene["robot"]
    feet_vel = robot.data.body_lin_vel_w[:, sensor_cfg.body_ids]
    slip_speed = torch.norm(feet_vel[..., :2], dim=-1)
    penalty = (in_contact * slip_speed).sum(dim=-1)
    return penalty * _stage_mask(command, stage_id)


def feet_slip_penalty_clipped(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    stage_id=None,
    clip_max: float = 2.0,
) -> torch.Tensor:
    """Bound foot-slip penalties so a single bad state cannot dominate returns."""

    raw = _sanitize_non_negative(
        feet_slip_penalty(
            env,
            command_name=command_name,
            stage_id=stage_id,
            sensor_cfg=sensor_cfg,
            threshold=threshold,
        ),
        clip_fill=clip_max,
    )
    return torch.clamp(raw, max=float(clip_max))


def palm_facing_object_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    cube_name: str = "bottle",
    hand_body: str = "right_palm_link",
    palm_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    axis: str = "xy",
    cos_cap: float = 1.0,
    target_cos: float | None = None,
    target_cos_std: float = 0.25,
    gate_pre_grasp: bool = False,
) -> torch.Tensor:
    """Reward the palm normal pointing toward the object."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    cube = env.scene[cube_name]

    hand_idx = _resolve_body_index(env, robot, hand_body, "_locomanip_palm_facing_idx")
    hand_pos = robot.data.body_pos_w[:, hand_idx]
    hand_quat = robot.data.body_quat_w[:, hand_idx]

    palm_axis_t = hand_quat.new_tensor(palm_axis).view(1, 3).expand(hand_quat.shape[0], -1)
    palm_normal = quat_apply(hand_quat, palm_axis_t)

    h2o = cube.data.root_pos_w - hand_pos
    if axis == "xy":
        palm_n = palm_normal[:, :2]
        h2o = h2o[:, :2]
    else:
        palm_n = palm_normal
    palm_n = palm_n / (palm_n.norm(dim=-1, keepdim=True) + 1.0e-6)
    h2o = h2o / (h2o.norm(dim=-1, keepdim=True) + 1.0e-6)

    cos_angle = (palm_n * h2o).sum(dim=-1)
    if target_cos is not None:
        reward = torch.exp(-(((cos_angle - float(target_cos)) / float(target_cos_std)) ** 2))
    else:
        reward = torch.clamp(cos_angle, min=0.0, max=float(cos_cap))
    if gate_pre_grasp and hasattr(command, "is_grasped"):
        reward = reward * (~command.is_grasped()).to(reward.dtype)
    return reward * _stage_mask(command, stage_id)


def hand_cube_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    cube_name: str = "cube",
    fingertip_body_names: Iterable[str] = RIGHT_HAND_TIP_NAMES,
    scale: float = 10.0,
    distance_aggregation: str = "hybrid",
) -> torch.Tensor:
    """Reward for keeping the specified hand bodies close to the cube."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    cube = env.scene[cube_name]

    fingertip_ids = _resolve_body_indices(
        env, robot, list(fingertip_body_names), "_locomanip_fingertip_ids_reward"
    )
    tip_pos = robot.data.body_pos_w[:, fingertip_ids]
    cube_pos = cube.data.root_pos_w[:, None, :]

    dists = torch.norm(tip_pos - cube_pos, dim=-1)
    if distance_aggregation == "mean":
        agg_dist = dists.mean(dim=-1)
    elif distance_aggregation == "softmin":
        beta = 5.0
        agg_dist = -torch.logsumexp(-beta * dists, dim=-1) / beta
    else:
        mean_dist = dists.mean(dim=-1)
        beta = 5.0
        softmin_dist = -torch.logsumexp(-beta * dists, dim=-1) / beta
        agg_dist = 0.5 * mean_dist + 0.5 * softmin_dist
    reward = torch.exp(-scale * agg_dist)
    return reward * _stage_mask(command, stage_id)


def _aggregate_distance(dists: torch.Tensor, distance_aggregation: str) -> torch.Tensor:
    """Aggregate a (N, K) per-body distance tensor to (N,)."""
    if distance_aggregation == "mean":
        return dists.mean(dim=-1)
    if distance_aggregation == "softmin":
        beta = 5.0
        return -torch.logsumexp(-beta * dists, dim=-1) / beta
    mean_dist = dists.mean(dim=-1)
    beta = 5.0
    softmin_dist = -torch.logsumexp(-beta * dists, dim=-1) / beta
    return 0.5 * mean_dist + 0.5 * softmin_dist


def hand_cube_grasp_pose_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    cube_name: str = "cube",
    body_groups: Sequence[Sequence[str]] = (),
    target_distances: Sequence[float] = (),
    scale: float = 10.0,
    distance_aggregation: str = "mean",
    group_aggregations: Sequence[str] | None = None,
    group_modes: Sequence[str] | None = None,
    body_overshoot_margin: float | None = None,
    metric_name: str | None = None,
) -> torch.Tensor:
    """Reward grouped hand bodies reaching configured object standoff distances."""
    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    cube = env.scene[cube_name]

    groups = [list(group) for group in body_groups]
    targets = [float(t) for t in target_distances]
    if len(groups) == 0:
        raise ValueError("hand_cube_grasp_pose_reward requires at least one body group.")
    if len(targets) != len(groups):
        raise ValueError(
            f"target_distances ({len(targets)}) must match body_groups ({len(groups)})."
        )
    if group_modes is None:
        modes = ["target"] * len(groups)
    else:
        modes = [str(m) for m in group_modes]
        if len(modes) != len(groups):
            raise ValueError(
                f"group_modes ({len(modes)}) must match body_groups ({len(groups)})."
            )
        for m in modes:
            if m not in ("target", "monotonic"):
                raise ValueError(
                    f"hand_cube_grasp_pose_reward: unknown group_mode '{m}' "
                    "(expected 'target' or 'monotonic')."
                )

    if group_aggregations is None:
        aggregations = [distance_aggregation] * len(groups)
    else:
        aggregations = [str(a) for a in group_aggregations]
        if len(aggregations) != len(groups):
            raise ValueError(
                f"group_aggregations ({len(aggregations)}) must match body_groups ({len(groups)})."
            )

    cube_pos = cube.data.root_pos_w[:, None, :]
    per_group_reward: list[torch.Tensor] = []
    for group_index, (names, target, mode, agg) in enumerate(
        zip(groups, targets, modes, aggregations)
    ):
        body_ids = _resolve_body_indices(
            env, robot, names, f"_locomanip_grasp_pose_ids_{group_index}"
        )
        body_pos = robot.data.body_pos_w[:, body_ids]
        dists = torch.norm(body_pos - cube_pos, dim=-1)
        agg_dist = _aggregate_distance(dists, agg)
        if mode == "monotonic":
            per_group_reward.append(torch.exp(-float(scale) * agg_dist))
        else:
            per_group_reward.append(torch.exp(-float(scale) * (agg_dist - target).abs()))
        if metric_name and hasattr(command, "metrics"):
            command.metrics[f"{metric_name}_g{group_index}_dist"] = agg_dist.detach()

    reward = torch.stack(per_group_reward, dim=0).mean(dim=0)

    if body_overshoot_margin is not None:
        overshoot = robot.data.root_pos_w[:, 0] - cube.data.root_pos_w[:, 0]
        reward = reward * (overshoot <= float(body_overshoot_margin)).to(reward.dtype)

    return reward * _stage_mask(command, stage_id)


def hand_body_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_cfg: SceneEntityCfg,
    target_body_name: str,
    hand_body_names: Iterable[str],
    stage_id=None,
    target_body_local_pos: tuple[float, float, float] | None = None,
    scale: float = 10.0,
    distance_aggregation: str = "mean",
) -> torch.Tensor:
    """Reward keeping selected hand bodies close to a target articulation body."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    target = env.scene[target_cfg.name]

    hand_body_names = list(hand_body_names)
    hand_ids = _resolve_body_indices(
        env,
        robot,
        hand_body_names,
        f"_locomanip_hand_body_distance_ids_{'_'.join(hand_body_names)}",
    )
    target_id = _resolve_body_index(
        env,
        target,
        target_body_name,
        f"_locomanip_target_body_distance_id_{target_cfg.name}_{target_body_name}",
    )

    hand_pos = robot.data.body_pos_w[:, hand_ids]
    target_pos = target.data.body_pos_w[:, target_id]
    if target_body_local_pos is not None:
        local_pos = torch.tensor(target_body_local_pos, device=target_pos.device, dtype=target_pos.dtype).view(1, 3)
        target_quat = target.data.body_quat_w[:, target_id]
        target_pos = target_pos + quat_apply(target_quat, local_pos.expand(env.num_envs, -1))
    target_pos = target_pos.unsqueeze(1)
    dists = torch.norm(hand_pos - target_pos, dim=-1)
    if distance_aggregation == "mean":
        agg_dist = dists.mean(dim=-1)
    elif distance_aggregation == "softmin":
        beta = 5.0
        agg_dist = -torch.logsumexp(-beta * dists, dim=-1) / beta
    elif distance_aggregation in ("min", "closest"):
        agg_dist = dists.min(dim=-1).values
    else:
        raise ValueError(f"Unsupported distance aggregation '{distance_aggregation}'.")
    reward = torch.exp(-float(scale) * agg_dist)
    return reward * _stage_mask(command, stage_id)


def fridge_door_open_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id=None,
    progress_scale: float = 0.02,
    require_contact: bool = False,
) -> torch.Tensor:
    """Reward positive per-step door opening progress."""

    command = env.command_manager.get_term(command_name)
    if hasattr(command, "door_open_progress"):
        progress = command.door_open_progress()
    else:
        progress = torch.zeros(env.num_envs, device=env.device)
    if progress_scale > 0.0:
        progress = torch.clamp(progress / float(progress_scale), min=0.0, max=1.0)
    if require_contact and hasattr(command, "fingertip_contact"):
        contact = command.fingertip_contact()
        if isinstance(contact, torch.Tensor) and contact.shape[0] == env.num_envs:
            progress = progress * contact.to(progress.dtype)
    return progress * _stage_mask(command, stage_id)


def fridge_door_close_regression_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id=None,
    regression_scale: float = 0.02,
) -> torch.Tensor:
    """Return positive per-step door closing regression."""

    command = env.command_manager.get_term(command_name)
    if hasattr(command, "door_close_regression"):
        regression = command.door_close_regression()
    else:
        regression = torch.zeros(env.num_envs, device=env.device)
    if regression_scale > 0.0:
        regression = torch.clamp(regression / float(regression_scale), min=0.0, max=1.0)
    return regression * _stage_mask(command, stage_id)


def base_backward_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id=None,
    speed_scale: float = 0.25,
    require_contact: bool = False,
) -> torch.Tensor:
    """Reward backward root velocity along the command retreat axis."""

    command = env.command_manager.get_term(command_name)
    if hasattr(command, "backward_velocity"):
        backward_speed = command.backward_velocity()
    else:
        backward_speed = torch.zeros(env.num_envs, device=env.device)
    if speed_scale > 0.0:
        reward = torch.clamp(backward_speed / float(speed_scale), min=0.0, max=1.0)
    else:
        reward = torch.clamp(backward_speed, min=0.0)
    if require_contact and hasattr(command, "fingertip_contact"):
        contact = command.fingertip_contact()
        if isinstance(contact, torch.Tensor) and contact.shape[0] == env.num_envs:
            reward = reward * contact.to(reward.dtype)
    return reward * _stage_mask(command, stage_id)


def fridge_door_held_open_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    hold_angle: float,
    stage_id=None,
    max_seconds: float = 1.0,
    require_contact: bool = True,
) -> torch.Tensor:
    """Reward consecutive time with the door held open."""

    command = env.command_manager.get_term(command_name)
    if hasattr(command, "door_open_angle"):
        angle = command.door_open_angle()
    else:
        angle = torch.zeros(env.num_envs, device=env.device)
    held = angle >= float(hold_angle)
    if require_contact and hasattr(command, "fingertip_contact"):
        contact = command.fingertip_contact()
        if isinstance(contact, torch.Tensor) and contact.shape[0] == env.num_envs:
            held = held & contact.to(torch.bool)

    counter_attr = f"_locomanip_{command_name}_door_held_open_counter"
    counter = getattr(env, counter_attr, None)
    if not isinstance(counter, torch.Tensor) or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    counter = torch.where(held, counter + float(env.step_dt), torch.zeros_like(counter))
    reset_buf = getattr(env, "reset_buf", None)
    if isinstance(reset_buf, torch.Tensor) and reset_buf.shape == counter.shape:
        counter = torch.where(reset_buf.bool(), torch.zeros_like(counter), counter)
    setattr(env, counter_attr, counter.detach())

    max_s = max(float(max_seconds), 1.0e-6)
    reward = (counter / max_s).clamp(min=0.0, max=1.0)
    return reward * _stage_mask(command, stage_id)


def fridge_door_open_with_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_angle: float,
    stage_id=None,
    power: float = 1.0,
    contact_scale: float = 1.0,
) -> torch.Tensor:
    """Reward door opening gated by fingertip contact."""

    command = env.command_manager.get_term(command_name)
    if hasattr(command, "door_open_angle"):
        angle = command.door_open_angle()
    else:
        angle = torch.zeros(env.num_envs, device=env.device)
    if target_angle <= 0.0:
        return torch.zeros_like(angle) * _stage_mask(command, stage_id)
    open_amount = torch.clamp(angle / float(target_angle), min=0.0, max=1.0)
    if power != 1.0:
        open_amount = torch.pow(open_amount, float(power))

    if hasattr(command, "fingertip_contact_count"):
        n_contact = command.fingertip_contact_count().to(open_amount.dtype)
    elif hasattr(command, "fingertip_contact"):
        contact = command.fingertip_contact()
        if isinstance(contact, torch.Tensor) and contact.shape[0] == env.num_envs:
            n_contact = contact.to(open_amount.dtype)
        else:
            n_contact = torch.zeros_like(open_amount)
    else:
        n_contact = torch.zeros_like(open_amount)
    gate = (n_contact / max(float(contact_scale), 1.0e-6)).clamp(min=0.0, max=1.0)
    return open_amount * gate * _stage_mask(command, stage_id)


def grasp_force_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    sensor_names: Iterable[str] | str | None = None,
    fingertip_body_names: Iterable[str] = RIGHT_HAND_TIP_NAMES,
    force_scale: float | None = None,
    force_clip: float | None = None,
    tanh_scale: float | None = 20.0,
    max_force: float | None = 50.0,
    aggregation: str = "sum",
    output_clip: float | None = None,
    gate_min_lift: float | None = None,
    cube_name: str = "cube",
    table_name: str | None = None,
    table_top_offset: float | None = None,
) -> torch.Tensor:
    """Reward fingertip contact force magnitudes."""

    command = env.command_manager.get_term(command_name)
    sensor_list = _get_contact_sensor_list(env, sensor_names)
    if len(sensor_list) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    forces_accum = []
    for sensor in sensor_list:
        forces = _sensor_forces(sensor)
        if isinstance(forces, torch.Tensor):
            magnitudes = forces.norm(dim=-1)
            history_len = getattr(getattr(sensor, "cfg", None), "history_length", None)
            if magnitudes.ndim >= 3 and history_len is not None and magnitudes.shape[1] == int(history_len):
                magnitudes = magnitudes.mean(dim=1)
            magnitudes_flat = magnitudes.view(magnitudes.shape[0], -1)
            forces_accum.append(magnitudes_flat.sum(dim=-1))
    if len(forces_accum) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    stacked_forces = torch.stack(forces_accum, dim=0)
    if aggregation == "sum":
        force_sum = stacked_forces.sum(dim=0)
    elif aggregation == "mean":
        force_sum = stacked_forces.mean(dim=0)
    else:
        raise ValueError("aggregation must be 'sum' or 'mean'.")
    if max_force is not None and max_force > 0.0:
        force_sum = force_sum.clamp(max=float(max_force))
    if force_scale is not None and force_scale > 0.0:
        force_sum = force_sum / float(force_scale)
    if force_clip is not None and force_clip > 0.0:
        clip = float(force_clip)
        force_sum = torch.clamp(force_sum, -clip, clip)
    if tanh_scale is not None and tanh_scale > 0.0:
        force_sum = torch.tanh(force_sum / float(tanh_scale))
    if output_clip is not None and output_clip > 0.0:
        force_sum = torch.clamp(force_sum, min=0.0, max=float(output_clip))
    if gate_min_lift is not None and gate_min_lift > 0.0:
        lift = _cube_lift_from_reference(
            env,
            command,
            cube_name=cube_name,
            table_name=table_name,
            table_top_offset=table_top_offset,
        )
        force_sum = force_sum * (lift >= float(gate_min_lift)).to(force_sum.dtype)
    return force_sum * _stage_mask(command, stage_id)


def grasp_based_on_obj_finger_dir_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    cube_name: str = "bottle",
    thumb_body_name: str = RIGHT_HAND_TIP_NAMES[0],
    index_body_names: Iterable[str] | str = RIGHT_HAND_TIP_NAMES[1],
    normalize_to_unit: bool = False,
    eps: float = 1.0e-6,
    project_plane: str | None = None,
    reward_shape: str = "cos",
    gate_pre_grasp: bool = False,
    gate_fingertip_cube_distance: float | None = None,
    metric_name: str | None = None,
) -> torch.Tensor:
    """Reward opposing thumb and finger directions around the object."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    cube = env.scene[cube_name]

    thumb_ids = getattr(env, "_locomanip_grasp_dir_thumb_ids", None)
    if not isinstance(thumb_ids, torch.Tensor):
        ids, found = robot.find_bodies([thumb_body_name], preserve_order=True)
        if len(ids) != 1:
            raise RuntimeError(f"Missing thumb body '{thumb_body_name}' for grasp direction reward. Found: {found}")
        thumb_ids = torch.as_tensor(ids, device=robot.data.body_pos_w.device, dtype=torch.long)
        setattr(env, "_locomanip_grasp_dir_thumb_ids", thumb_ids)

    if isinstance(index_body_names, str):
        index_names = [index_body_names]
    else:
        index_names = list(index_body_names)
    index_ids = getattr(env, "_locomanip_grasp_dir_index_ids", None)
    if not isinstance(index_ids, torch.Tensor):
        ids, found = robot.find_bodies(index_names, preserve_order=True)
        if len(ids) != len(index_names):
            missing = sorted(set(index_names) - set(found))
            raise RuntimeError(f"Missing index bodies for grasp direction reward: {missing}")
        index_ids = torch.as_tensor(ids, device=robot.data.body_pos_w.device, dtype=torch.long)
        setattr(env, "_locomanip_grasp_dir_index_ids", index_ids)

    obj_pos = cube.data.root_pos_w
    thumb_pos = robot.data.body_pos_w[:, thumb_ids].mean(dim=1)
    index_pos = robot.data.body_pos_w[:, index_ids].mean(dim=1)
    obj_to_thumb = thumb_pos - obj_pos
    obj_to_index = index_pos - obj_pos
    if project_plane == "xy":
        obj_to_thumb[:, 2] = 0.0
        obj_to_index[:, 2] = 0.0
    elif project_plane is not None:
        raise ValueError(
            f"grasp_based_on_obj_finger_dir_reward: unknown project_plane '{project_plane}'."
        )
    obj_to_thumb = obj_to_thumb / (obj_to_thumb.norm(dim=-1, keepdim=True) + float(eps))
    obj_to_index = obj_to_index / (obj_to_index.norm(dim=-1, keepdim=True) + float(eps))
    cos_opp = torch.sum(obj_to_thumb * obj_to_index, dim=-1)
    if reward_shape == "cos":
        reward = -cos_opp
        if normalize_to_unit:
            reward = 0.5 * (reward + 1.0)
    elif reward_shape == "angle":
        theta = torch.acos(cos_opp.clamp(-1.0, 1.0))
        reward = theta / torch.pi
        if not normalize_to_unit:
            reward = 2.0 * reward - 1.0
    else:
        raise ValueError(
            f"grasp_based_on_obj_finger_dir_reward: unknown reward_shape '{reward_shape}'."
        )

    if metric_name and hasattr(command, "metrics"):
        command.metrics[str(metric_name)] = cos_opp.detach()

    if gate_pre_grasp and hasattr(command, "is_grasped"):
        reward = reward * (~command.is_grasped()).to(reward.dtype)

    if (
        gate_fingertip_cube_distance is not None
        and gate_fingertip_cube_distance > 0.0
        and hasattr(command, "fingertip_mean_distance_to_cube")
    ):
        tip_dist = command.fingertip_mean_distance_to_cube()
        reward = reward * (tip_dist >= float(gate_fingertip_cube_distance)).to(reward.dtype)

    return reward * _stage_mask(command, stage_id)


def fingertip_centroid_on_cube_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    cube_name: str = "bottle",
    fingertip_body_names: Iterable[str] = RIGHT_HAND_TIP_NAMES,
    decay_scale: float = 20.0,
    gate_pre_grasp: bool = False,
    gate_fingertip_cube_distance: float | None = None,
    metric_name: str | None = None,
) -> torch.Tensor:
    """Reward the fingertip centroid being close to the object."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    cube = env.scene[cube_name]

    tip_ids = _resolve_body_indices(
        env, robot, list(fingertip_body_names), "_locomanip_fingertip_centroid_tip_ids"
    )
    tip_pos = robot.data.body_pos_w[:, tip_ids]
    centroid = tip_pos.mean(dim=1)
    offset = torch.norm(centroid - cube.data.root_pos_w, dim=-1)

    reward = torch.exp(-float(decay_scale) * offset)

    if metric_name and hasattr(command, "metrics"):
        command.metrics[str(metric_name)] = offset.detach()

    if gate_pre_grasp and hasattr(command, "is_grasped"):
        reward = reward * (~command.is_grasped()).to(reward.dtype)

    if (
        gate_fingertip_cube_distance is not None
        and gate_fingertip_cube_distance > 0.0
        and hasattr(command, "fingertip_mean_distance_to_cube")
    ):
        tip_dist = command.fingertip_mean_distance_to_cube()
        reward = reward * (tip_dist >= float(gate_fingertip_cube_distance)).to(reward.dtype)

    return reward * _stage_mask(command, stage_id)


def fingertip_contact_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id=None,
    sensor_names: Iterable[str] | str | None = None,
    threshold: float = 1.0,
    bonus_start_contacts: int | None = None,
    bonus_scale: float = 0.0,
    bonus_power: float = 2.0,
    gate_min_lift: float | None = None,
    cube_name: str = "cube",
    table_name: str | None = None,
    table_top_offset: float | None = None,
) -> torch.Tensor:
    """Reward fingertip contact ratio with optional dexhand-style bonus and lift gate."""

    command = env.command_manager.get_term(command_name)
    if sensor_names is None:
        contact = command.fingertip_contact()
        if contact is None:
            return torch.zeros(env.num_envs, device=env.device)
        return contact.float() * _stage_mask(command, stage_id)

    sensor_list = _get_contact_sensor_list(env, sensor_names)
    if len(sensor_list) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    contacts = []
    for sensor in sensor_list:
        forces = _sensor_forces(sensor)
        if not isinstance(forces, torch.Tensor):
            continue
        contact = forces.norm(dim=-1) > float(threshold)
        in_contact = contact.view(contact.shape[0], -1).any(dim=-1)
        contacts.append(in_contact.float())

    if len(contacts) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    stacked = torch.stack(contacts, dim=0)
    reward = stacked.mean(dim=0)

    if bonus_start_contacts is not None and bonus_scale > 0.0:
        num_sensors = stacked.shape[0]
        start_contacts = int(max(0, min(int(bonus_start_contacts), int(num_sensors))))
        if start_contacts < num_sensors:
            contact_count = stacked.sum(dim=0)
            surplus_contacts = torch.clamp(contact_count - float(start_contacts), min=0.0)
            normalizer = float(max(1, num_sensors - start_contacts))
            reward = reward + float(bonus_scale) * torch.pow(surplus_contacts / normalizer, float(bonus_power))

    if gate_min_lift is not None and gate_min_lift > 0.0:
        lift = _cube_lift_from_reference(
            env,
            command,
            cube_name=cube_name,
            table_name=table_name,
            table_top_offset=table_top_offset,
        )
        reward = reward * (lift >= float(gate_min_lift)).to(reward.dtype)

    return reward * _stage_mask(command, stage_id)


def cube_lift_height_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    cube_name: str = "cube",
    table_name: str | None = None,
    table_top_offset: float | None = None,
    cap: float = 0.15,
    require_grasp: bool = True,
    yank_penalty_beta: float | None = None,
) -> torch.Tensor:
    """Reward lifting the object above its reference height."""

    command = env.command_manager.get_term(command_name)
    lift_height = _cube_lift_from_reference(
        env,
        command,
        cube_name=cube_name,
        table_name=table_name,
        table_top_offset=table_top_offset,
    )
    cap_t = torch.as_tensor(cap, device=lift_height.device, dtype=lift_height.dtype)
    lift_height = torch.minimum(lift_height, cap_t)
    if cap > 0.0:
        reward = lift_height / cap_t
    else:
        reward = torch.zeros_like(lift_height)
    if require_grasp:
        reward = reward * command.is_grasped().float()
    if yank_penalty_beta is not None and yank_penalty_beta > 0.0:
        cube = env.scene[cube_name]
        v_z = cube.data.root_lin_vel_w[:, 2]
        yank_factor = torch.exp(-float(yank_penalty_beta) * v_z * v_z)
        reward = reward * yank_factor
    return reward * _stage_mask(command, stage_id)


def sustained_grasp_bonus_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    cube_name: str = "cube",
    table_name: str | None = None,
    table_top_offset: float | None = None,
    max_seconds: float = 1.5,
    min_lift: float | None = None,
    min_forward_speed: float | None = None,
) -> torch.Tensor:
    """Reward consecutive grasp duration, optionally gated by lift or forward motion."""

    command = env.command_manager.get_term(command_name)
    is_grasped = command.is_grasped() if hasattr(command, "is_grasped") else torch.zeros(
        env.num_envs, device=env.device, dtype=torch.bool
    )
    if min_lift is not None and min_lift > 0.0:
        lift_height = _cube_lift_from_reference(
            env,
            command,
            cube_name=cube_name,
            table_name=table_name,
            table_top_offset=table_top_offset,
        )
        is_grasped = is_grasped & (lift_height > float(min_lift))
    if min_forward_speed is not None:
        cube = env.scene[cube_name]
        cube_vel = getattr(getattr(cube, "data", None), "root_lin_vel_w", None)
        if isinstance(cube_vel, torch.Tensor):
            is_grasped = is_grasped & (cube_vel[:, 0] > float(min_forward_speed))

    step_dt = float(env.step_dt)
    counter = getattr(env, "_locomanip_sustained_grasp_counter", None)
    if not isinstance(counter, torch.Tensor) or counter.shape[0] != env.num_envs:
        counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    counter = torch.where(is_grasped, counter + step_dt, torch.zeros_like(counter))

    reset_buf = getattr(env, "reset_buf", None)
    if isinstance(reset_buf, torch.Tensor) and reset_buf.shape == counter.shape:
        counter = torch.where(reset_buf.bool(), torch.zeros_like(counter), counter)

    env._locomanip_sustained_grasp_counter = counter.detach()

    max_s = max(float(max_seconds), 1.0e-6)
    reward = (counter / max_s).clamp(min=0.0, max=1.0)
    return reward * _stage_mask(command, stage_id)


def object_hold_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    cube_name: str | None = None,
    table_name: str | None = None,
    table_top_offset: float | None = None,
    min_lift: float | None = None,
) -> torch.Tensor:
    """Reward the current grasp predicate, optionally gated by object lift."""

    command = env.command_manager.get_term(command_name)
    if cube_name is None and table_name is None and table_top_offset is None and min_lift is None:
        return command.is_grasped().float() * _stage_mask(command, stage_id)
    if cube_name is None:
        cube_name = "cube"
    lift_height = _cube_lift_from_reference(
        env,
        command,
        cube_name=cube_name,
        table_name=table_name,
        table_top_offset=table_top_offset,
    )
    if min_lift is None:
        min_lift = float(command.cfg.lift_height)
    return command.is_grasped().float() * (lift_height > float(min_lift)).float() * _stage_mask(command, stage_id)


def cube_not_lifted_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    cube_name: str = "cube",
    table_name: str | None = None,
    table_top_offset: float | None = None,
    target_lift: float = 0.1,
    gate_fingertip_cube_distance: float | None = None,
    gate_use_hand_root: bool = False,
) -> torch.Tensor:
    """Penalty while the object remains close to its resting height."""

    command = env.command_manager.get_term(command_name)
    lift_height = _cube_lift_from_reference(
        env,
        command,
        cube_name=cube_name,
        table_name=table_name,
        table_top_offset=table_top_offset,
    )
    if target_lift <= 0.0:
        penalty = torch.ones(env.num_envs, device=env.device)
    else:
        penalty = 1.0 - torch.clamp(lift_height / float(target_lift), min=0.0, max=1.0)

    if gate_fingertip_cube_distance is not None and gate_fingertip_cube_distance > 0.0:
        gate_dist = None
        if gate_use_hand_root and hasattr(command, "hand_root_mean_distance_to_cube"):
            gate_dist = command.hand_root_mean_distance_to_cube()
        if gate_dist is None and hasattr(command, "fingertip_mean_distance_to_cube"):
            gate_dist = command.fingertip_mean_distance_to_cube()
        if gate_dist is not None:
            penalty = penalty * (gate_dist < float(gate_fingertip_cube_distance)).to(penalty.dtype)

    return penalty * _stage_mask(command, stage_id)


def cube_dropped_phase_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    cube_name: str = "bottle",
    drop_height: float = 0.05,
    min_lift_before_large_penalty: float = 0.02,
    pre_lift_scale: float = 1.0,
    post_lift_scale: float = 5.0,
    sensor_names: Iterable[str] | str | None = None,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalty for dropping the object, with a larger value after it has been lifted once."""

    command = env.command_manager.get_term(command_name)
    cube = env.scene[cube_name]
    cube_z = cube.data.root_pos_w[:, 2]
    dropped = cube_z < (command.cube_init_z - float(drop_height))

    if sensor_names is not None:
        sensor_list = _get_contact_sensor_list(env, sensor_names)
        if len(sensor_list) > 0:
            in_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            for sensor in sensor_list:
                forces = _sensor_forces(sensor)
                if not isinstance(forces, torch.Tensor):
                    continue
                contact = forces.norm(dim=-1) > float(contact_threshold)
                in_contact = in_contact | contact.view(contact.shape[0], -1).any(dim=-1)
            dropped = dropped & ~in_contact

    attr = f"_locomanip_{command_name}_{cube_name}_lifted_before_drop"
    lifted_before = getattr(env, attr, None)
    if not isinstance(lifted_before, torch.Tensor) or lifted_before.shape[0] != env.num_envs:
        lifted_before = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    else:
        lifted_before = lifted_before.to(device=env.device, dtype=torch.bool)
        lifted_before = torch.where(_new_episode_mask(env, command), torch.zeros_like(lifted_before), lifted_before)

    lifted_before = lifted_before | ((cube_z - command.cube_init_z) > float(min_lift_before_large_penalty))
    setattr(env, attr, lifted_before.detach())

    penalty_scale = torch.where(
        lifted_before,
        torch.full_like(cube_z, float(post_lift_scale)),
        torch.full_like(cube_z, float(pre_lift_scale)),
    )
    return dropped.to(cube_z.dtype) * penalty_scale * _stage_mask(command, stage_id)


def cube_table_contact_move_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    sensor_name: str = "cube_table_contact",
    cube_name: str = "cube",
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize cube sliding on the table (xy speed when in contact)."""

    command = env.command_manager.get_term(command_name)
    sensors = getattr(env.scene, "sensors", {})
    sensor = sensors.get(sensor_name) if isinstance(sensors, dict) else None
    if sensor is None:
        contact_mask = torch.zeros(env.num_envs, device=env.device)
    else:
        forces = getattr(sensor.data, "force_matrix_w_history", None)
        if forces is None:
            forces = getattr(sensor.data, "net_forces_w_history", None)
        if forces is None:
            forces = getattr(sensor.data, "force_matrix_w", None)
        if forces is None:
            forces = getattr(sensor.data, "net_forces_w", None)
        if isinstance(forces, torch.Tensor):
            contact = forces.norm(dim=-1) > threshold
            if contact.ndim > 2:
                contact = contact.any(dim=tuple(range(1, contact.ndim)))
            contact_mask = contact.float()
        else:
            contact_mask = torch.zeros(env.num_envs, device=env.device)

    cube = env.scene[cube_name]
    speed_xy = torch.norm(cube.data.root_lin_vel_w[:, :2], dim=-1)
    penalty = speed_xy * contact_mask
    return penalty * _stage_mask(command, stage_id)


def cube_hand_rel_move_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    stage_id,
    cube_name: str = "cube",
    hand_body: str = "right_palm_link",
) -> torch.Tensor:
    """Penalize mismatch between cube and hand vertical velocity once the cube is grasped and lifted."""

    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    cube = env.scene[cube_name]

    hand_idx = _resolve_body_index(env, robot, hand_body, "_locomanip_hand_body_idx")
    hand_vel_z = robot.data.body_lin_vel_w[:, hand_idx, 2]
    cube_vel_z = cube.data.root_lin_vel_w[:, 2]

    grasp_mask = (command.is_grasped() & command.is_lifted()).float()
    penalty = torch.abs(cube_vel_z - hand_vel_z) * grasp_mask
    return penalty * _stage_mask(command, stage_id)
