from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from coordex.tasks.locomanip.mdp.commands import LocomanipStageCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_command(env: ManagerBasedRLEnv, command_name: str) -> LocomanipStageCommand:
    return env.command_manager.get_term(command_name)


def _get_contact_sensor_list(env: ManagerBasedRLEnv, sensor_names: Iterable[str] | str | None) -> list:
    sensors = getattr(env.scene, "sensors", {})
    sensor_list: list = []
    if sensor_names is None:
        return sensor_list
    if isinstance(sensor_names, str):
        sensor = sensors.get(sensor_names) if isinstance(
            sensors, dict) else None
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


def _new_episode_mask(env: ManagerBasedRLEnv, command: LocomanipStageCommand) -> torch.Tensor:
    mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    stage_steps = getattr(command, "_stage_steps", None)
    if isinstance(stage_steps, torch.Tensor) and stage_steps.shape[0] == env.num_envs:
        mask = mask | (stage_steps.to(device=env.device) <= 1)
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if isinstance(episode_length_buf, torch.Tensor) and episode_length_buf.shape[0] == env.num_envs:
        mask = mask | (episode_length_buf.to(device=env.device) <= 1)
    return mask


def stage_success(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = _get_command(env, command_name)
    if hasattr(command, "is_success"):
        return command.is_success()
    final_stage = int(command.cfg.num_stages) - 1
    success = command.is_cube_at_target()
    return (command.stage_id == final_stage) & success


def cube_dropped(
    env: ManagerBasedRLEnv,
    command_name: str,
    drop_height: float,
    min_stage: int = 0,
    sensor_names: Iterable[str] | str | None = None,
    contact_threshold: float = 1.0,
    min_lift_before_drop: float | None = None,
    max_tilt_angle: float | None = None,
) -> torch.Tensor:
    command = _get_command(env, command_name)
    cube_z = command.cube.data.root_pos_w[:, 2]
    threshold = command.cube_init_z - float(drop_height)
    dropped = cube_z < threshold

    if sensor_names is not None:
        sensor_list = _get_contact_sensor_list(env, sensor_names)
        if len(sensor_list) > 0:
            in_contact = torch.zeros(
                env.num_envs, device=env.device, dtype=torch.bool)
            for sensor in sensor_list:
                forces = _sensor_forces(sensor)
                if not isinstance(forces, torch.Tensor):
                    continue
                contact = forces.norm(dim=-1) > float(contact_threshold)
                in_contact = in_contact | contact.view(
                    contact.shape[0], -1).any(dim=-1)
            dropped = dropped & ~in_contact

    if min_stage > 0:
        dropped = dropped & (command.stage_id >= int(min_stage))
    if min_lift_before_drop is not None:
        attr = f"_locomanip_{command_name}_{getattr(command.cfg, 'cube_name', 'cube')}_lifted_before_drop"
        lifted_before = getattr(env, attr, None)
        if not isinstance(lifted_before, torch.Tensor) or lifted_before.shape[0] != env.num_envs:
            lifted_before = torch.zeros(
                env.num_envs, device=env.device, dtype=torch.bool)
        else:
            lifted_before = lifted_before.to(
                device=env.device, dtype=torch.bool)
            lifted_before = torch.where(_new_episode_mask(
                env, command), torch.zeros_like(lifted_before), lifted_before)
        lifted_before = lifted_before | (
            (cube_z - command.cube_init_z) > float(min_lift_before_drop))
        setattr(env, attr, lifted_before.detach())
        dropped = dropped & lifted_before

    if max_tilt_angle is not None:
        if hasattr(command, "bottle_upright_angle"):
            tilt = command.bottle_upright_angle()
        else:
            cube = command.cube
            up = cube.data.root_quat_w.new_tensor((0.0, 0.0, 1.0)).view(
                1, 3).expand(cube.data.root_quat_w.shape[0], -1)
            world_up = quat_apply(cube.data.root_quat_w, up)
            tilt = torch.acos(world_up[:, 2].clamp(-1.0, 1.0))
        dropped = dropped | (tilt > float(max_tilt_angle))

    return dropped.to(torch.bool)


def hand_far_from_handle(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_distance: float = 0.5,
    use_fingertip: bool = False,
) -> torch.Tensor:
    """Terminate when the hand drifts too far from the manipulation target.
    """

    command = _get_command(env, command_name)
    if use_fingertip and hasattr(command, "fingertip_handle_distance"):
        dist = command.fingertip_handle_distance()
    elif hasattr(command, "hand_handle_distance"):
        dist = command.hand_handle_distance()
    elif hasattr(command, "fingertip_handle_distance"):
        dist = command.fingertip_handle_distance()
    else:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return (dist > float(max_distance)).to(torch.bool)


def robot_passed_object_without_grasp(
    env: ManagerBasedRLEnv,
    command_name: str,
    cube_name: str | None = None,
    x_margin: float = 0.5,
) -> torch.Tensor:
    """Terminate if the robot walks past the object before grasping it."""

    command = _get_command(env, command_name)
    robot = env.scene[command.cfg.asset_name]
    object_name = cube_name or getattr(command.cfg, "cube_name", "bottle")
    cube = env.scene[object_name]

    passed_object = robot.data.root_pos_w[:, 0] > (
        cube.data.root_pos_w[:, 0] + float(x_margin))
    if not hasattr(command, "is_grasped"):
        raise RuntimeError(
            f"Command term '{command_name}' does not expose an is_grasped() predicate.")
    object_grasped = command.is_grasped().to(torch.bool)
    return (passed_object & ~object_grasped).to(torch.bool)


def robot_fall(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height: float,
    body_name: str | None = None,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    if body_name:
        body_ids, _ = asset.find_bodies([body_name], preserve_order=True)
        if len(body_ids) == 0:
            raise RuntimeError(
                f"Body '{body_name}' not found on asset '{asset_cfg.name}'.")
        height = asset.data.body_pos_w[:, body_ids[0], 2]
    elif getattr(asset_cfg, "body_names", None):
        body_ids, _ = asset.find_bodies(
            list(asset_cfg.body_names), preserve_order=True)
        if len(body_ids) == 0:
            raise RuntimeError(
                f"No bodies found for {asset_cfg.body_names} on '{asset_cfg.name}'.")
        height = asset.data.body_pos_w[:, body_ids[0], 2]
    else:
        height = asset.data.root_pos_w[:, 2]
    return (height < float(min_height)).to(torch.bool)


def _get_root_state_tensor(env: ManagerBasedRLEnv, asset_name: str, attr_name: str) -> torch.Tensor:
    asset = env.scene[asset_name]
    data = getattr(asset, "data", None)
    value = getattr(data, attr_name, None) if data is not None else None
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(
            f"Asset '{asset_name}' does not expose root state tensor '{attr_name}'.")
    return value


def root_state_blowup(
    env: ManagerBasedRLEnv,
    asset_name: str,
    max_xy_radius: float | None = None,
    min_height: float | None = None,
    max_height: float | None = None,
    max_lin_vel: float | None = None,
    max_ang_vel: float | None = None,
) -> torch.Tensor:
    """Terminate when an asset root pose or velocity becomes non-finite or clearly unreasonable."""

    root_pos_w = _get_root_state_tensor(env, asset_name, "root_pos_w")
    env_origins = getattr(env.scene, "env_origins", None)
    if isinstance(env_origins, torch.Tensor) and env_origins.shape == root_pos_w.shape:
        rel_pos = root_pos_w - env_origins
    else:
        rel_pos = root_pos_w

    blowup = ~torch.isfinite(rel_pos).all(dim=-1)

    if max_xy_radius is not None:
        xy_radius = torch.linalg.norm(torch.nan_to_num(
            rel_pos[:, :2], nan=0.0, posinf=0.0, neginf=0.0), dim=-1)
        blowup = blowup | (xy_radius > float(max_xy_radius))

    z = rel_pos[:, 2]
    blowup = blowup | ~torch.isfinite(z)
    if min_height is not None:
        blowup = blowup | (z < float(min_height))
    if max_height is not None:
        blowup = blowup | (z > float(max_height))

    if max_lin_vel is not None:
        root_lin_vel_w = _get_root_state_tensor(
            env, asset_name, "root_lin_vel_w")
        blowup = blowup | ~torch.isfinite(root_lin_vel_w).all(dim=-1)
        lin_speed = torch.linalg.norm(
            torch.nan_to_num(root_lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0),
            dim=-1,
        )
        blowup = blowup | (lin_speed > float(max_lin_vel))

    if max_ang_vel is not None:
        root_ang_vel_w = _get_root_state_tensor(
            env, asset_name, "root_ang_vel_w")
        blowup = blowup | ~torch.isfinite(root_ang_vel_w).all(dim=-1)
        ang_speed = torch.linalg.norm(
            torch.nan_to_num(root_ang_vel_w, nan=0.0, posinf=0.0, neginf=0.0),
            dim=-1,
        )
        blowup = blowup | (ang_speed > float(max_ang_vel))

    return blowup.to(torch.bool)


def root_y_deviation(
    env: ManagerBasedRLEnv,
    asset_name: str = "robot",
    max_abs_y: float = 1.5,
) -> torch.Tensor:
    """Terminate when an asset root y position leaves the env-local corridor."""

    root_pos_w = _get_root_state_tensor(env, asset_name, "root_pos_w")
    env_origins = getattr(env.scene, "env_origins", None)
    if isinstance(env_origins, torch.Tensor) and env_origins.shape == root_pos_w.shape:
        rel_pos = root_pos_w - env_origins
    else:
        rel_pos = root_pos_w

    y = rel_pos[:, 1]
    invalid = ~torch.isfinite(y)
    abs_y = torch.abs(torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0))
    return (invalid | (abs_y > float(max_abs_y))).to(torch.bool)


def joint_velocity_blowup(
    env: ManagerBasedRLEnv,
    asset_name: str = "robot",
    max_abs_joint_vel: float = 40.0,
) -> torch.Tensor:
    """Terminate when any joint velocity becomes non-finite or unrealistically large."""

    asset = env.scene[asset_name]
    joint_vel = getattr(getattr(asset, "data", None), "joint_vel", None)
    if not isinstance(joint_vel, torch.Tensor):
        raise RuntimeError(
            f"Asset '{asset_name}' does not expose joint velocities.")

    blowup = ~torch.isfinite(joint_vel).all(dim=-1)
    max_joint_speed = torch.abs(torch.nan_to_num(
        joint_vel, nan=0.0, posinf=0.0, neginf=0.0)).max(dim=-1).values
    blowup = blowup | (max_joint_speed > float(max_abs_joint_vel))
    return blowup.to(torch.bool)
