from __future__ import annotations

from dataclasses import MISSING
import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_error_magnitude

from coordex.tasks.locomanip.constants import RIGHT_HAND_JOINT_PATTERNS, RIGHT_HAND_TIP_NAMES
from coordex.tasks.locomanip.mdp.rsi import (
    capture_locomanip_snapshot,
    restore_locomanip_snapshot_aux_state,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class LocomanipStageCommand(CommandTerm):
    """Command term that tracks a 4-stage locomotion-manipulation script."""

    cfg: "LocomanipStageCommandCfg"

    def __init__(self, cfg: "LocomanipStageCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._env = env

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.cube: RigidObject = env.scene[cfg.cube_name]
        self.source_table: RigidObject = env.scene[cfg.source_table_name]
        self.target_table: RigidObject | None = env.scene[
            cfg.target_table_name] if cfg.target_table_name else None
        self.target_region: RigidObject | None = env.scene[
            cfg.target_region_name] if cfg.target_region_name else None

        fingertip_names = list(
            cfg.fingertip_body_names or RIGHT_HAND_TIP_NAMES)
        fingertip_ids, fingertip_found = self.robot.find_bodies(
            fingertip_names, preserve_order=True)
        if len(fingertip_ids) != len(fingertip_names):
            missing = sorted(set(fingertip_names) - set(fingertip_found))
            raise RuntimeError(
                f"Missing fingertip bodies on '{cfg.asset_name}': {missing}")
        self.fingertip_ids = torch.tensor(
            fingertip_ids, dtype=torch.long, device=self.device)

        hand_base_name = str(
            getattr(cfg, "hand_base_body_name", "right_palm_link"))
        hand_base_ids, hand_base_found = self.robot.find_bodies(
            [hand_base_name], preserve_order=True)
        if len(hand_base_ids) == 0:
            raise RuntimeError(
                f"Missing hand base body on '{cfg.asset_name}': {set([hand_base_name]) - set(hand_base_found)}")
        self.hand_base_id = int(hand_base_ids[0])
        self._hand_base_id_tensor = torch.tensor(
            [self.hand_base_id], device=self.device, dtype=torch.long)

        hand_root_names = tuple(getattr(cfg, "hand_root_body_names", ()) or ())
        if hand_root_names:
            root_ids, root_found = self.robot.find_bodies(
                list(hand_root_names), preserve_order=True)
            if len(root_ids) != len(hand_root_names):
                missing = sorted(set(hand_root_names) - set(root_found))
                raise RuntimeError(
                    f"Missing hand-root bodies on '{cfg.asset_name}': {missing}")
            self.hand_root_ids = torch.tensor(
                root_ids, dtype=torch.long, device=self.device)
        else:
            self.hand_root_ids = None

        hand_joint_names = list(cfg.hand_joint_names) if getattr(
            cfg, "hand_joint_names", ()) else list(RIGHT_HAND_JOINT_PATTERNS)
        joint_ids, joint_found = self.robot.find_joints(
            hand_joint_names, preserve_order=True)
        if len(joint_ids) != len(hand_joint_names):
            missing = sorted(set(hand_joint_names) - set(joint_found))
            raise RuntimeError(
                f"Missing right-hand joints on '{cfg.asset_name}': {missing}")
        self._right_hand_joint_ids = torch.tensor(
            joint_ids, dtype=torch.long, device=self.device)

        self._contact_sensor = None
        self._contact_sensors: list = []
        self._contact_sensor_names: list[str] = []
        if hasattr(env.scene, "sensors"):
            sensors = env.scene.sensors
            names = cfg.contact_sensor_name
            if isinstance(names, (list, tuple, set)):
                for name in names:
                    if name in sensors:
                        self._contact_sensors.append(sensors[name])
                        self._contact_sensor_names.append(name)
            else:
                if names in sensors:
                    self._contact_sensor = sensors[names]
                    self._contact_sensors.append(sensors[names])
                    self._contact_sensor_names.append(names)

        self._table_contact_sensors: list = []
        if hasattr(env.scene, "sensors"):
            for name in ("source_table_contact", "target_table_contact"):
                sensor = env.scene.sensors.get(name) if isinstance(
                    env.scene.sensors, dict) else None
                if sensor is not None:
                    self._table_contact_sensors.append(sensor)

        self.stage_id = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.stage_one_hot = torch.zeros(
            (self.num_envs, cfg.num_stages), device=self.device)
        self.stage_one_hot[:, 0] = 1.0
        self._stage_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.actual_time_in_stage_buf = self._stage_steps
        self.time_in_stage_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self._max_stage_time = self._resolve_max_stage_time()
        self._pending_stage_snapshots = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.cube_init_z = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device)
        self._forward_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self._hand_local_normal = torch.tensor(
            getattr(cfg, "hand_palm_axis", (-1.0, 0.0, 0.0)), device=self.device)

        self._hand_prep_height_margin = 0.015
        self._hand_prep_table_top_offset = 0.025  # cube half-height
        self._hand_prep_height_offset = float(
            getattr(cfg, "hand_prep_height_offset", 0.125))
        self._hand_prep_xy_threshold = float(
            getattr(cfg, "hand_prep_xy_threshold", 0.0))
        self._hand_open_tolerance = 0.5
        self._hand_open_fraction = 0.5
        self._hand_flat_angle_tolerance = float(
            getattr(cfg, "hand_flat_angle_tolerance", 0.35))
        self._require_stage1_feet_stance = bool(
            getattr(cfg, "require_stage1_feet_stance", False))
        self._foot_stagger_x_threshold = float(
            getattr(cfg, "foot_stagger_x_threshold", 0.08))
        policy_obs_cfg = getattr(
            getattr(env.cfg, "observations", None), "policy", None)
        humanoid_prior_cfg = getattr(policy_obs_cfg, "humanoid_prior", None)
        humanoid_prior_params = getattr(
            humanoid_prior_cfg, "params", {}) if humanoid_prior_cfg is not None else {}
        self._snapshot_history_length = max(
            1, int(humanoid_prior_params.get("history_length", 1)))
        self.left_foot_id: int | None = None
        self.right_foot_id: int | None = None
        if self._require_stage1_feet_stance:
            left_foot_name = str(
                getattr(cfg, "left_foot_body_name", "left_ankle_roll_link"))
            left_foot_ids, left_foot_found = self.robot.find_bodies(
                [left_foot_name], preserve_order=True)
            if len(left_foot_ids) == 0:
                raise RuntimeError(
                    f"Missing left foot body on '{cfg.asset_name}': {set([left_foot_name]) - set(left_foot_found)}"
                )
            right_foot_name = str(
                getattr(cfg, "right_foot_body_name", "right_ankle_roll_link"))
            right_foot_ids, right_foot_found = self.robot.find_bodies(
                [right_foot_name], preserve_order=True)
            if len(right_foot_ids) == 0:
                raise RuntimeError(
                    f"Missing right foot body on '{cfg.asset_name}': {set([right_foot_name]) - set(right_foot_found)}"
                )
            self.left_foot_id = int(left_foot_ids[0])
            self.right_foot_id = int(right_foot_ids[0])

        self.metrics["stage_id"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["stage_time"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["actual_stage_time"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["stage_time_limit"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["distance_base_source_xy"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["distance_base_target_xy"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["distance_hand_cube"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["cube_lift_height"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["error_place_pos"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["error_place_ori"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["fingertip_contact"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["foot_stagger_x"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["stage1_ready_history_frac"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["stage1_ready_history_stable"] = torch.zeros(
            self.num_envs, device=self.device)
        self._stage0_transition_episode_keys = (
            "approach_done",
            "hand_height_ok",
            "hand_xy_ok",
            "hand_flat",
            "hand_prepped",
            "stage1_ready",
        )
        self._stage0_transition_episode_flags = {
            key: torch.zeros(self.num_envs, dtype=torch.bool,
                             device=self.device)
            for key in self._stage0_transition_episode_keys
        }
        self._stage1_ready_history = torch.zeros(
            (self.num_envs, self._snapshot_history_length),
            dtype=torch.bool,
            device=self.device,
        )

    @property
    def command(self) -> torch.Tensor:
        return self.stage_one_hot

    @property
    def stage0_transition_episode_flags(self) -> dict[str, torch.Tensor]:
        return self._stage0_transition_episode_flags

    @property
    def recent_stage1_ready_history(self) -> torch.Tensor:
        return self._stage1_ready_history

    @torch.no_grad()
    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        if isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids
        else:
            env_ids_t = torch.tensor(
                env_ids, device=self.device, dtype=torch.long)

        self.reset_stage0_transition_episode_flags(env_ids_t)
        self.reset_stage1_ready_history(env_ids_t)

        stage_overrides = None
        override_vals = None
        for attr_name in ("_demo_rsi_stage_on_reset", "_no_demo_rsi_stage_on_reset"):
            candidate = getattr(self._env, attr_name, None)
            if not isinstance(candidate, torch.Tensor) or candidate.numel() < env_ids_t.max().item() + 1:
                continue
            candidate_vals = candidate[env_ids_t]
            if (candidate_vals >= 0).any():
                stage_overrides = candidate
                override_vals = candidate_vals
                break

        cube_height = self.cube.data.root_pos_w[env_ids_t, 2]

        if override_vals is None:
            self.cube_init_z[env_ids_t] = cube_height
            self._set_stage(env_ids_t, 0, force_reentry=True,
                            enqueue_pending=False)
            return

        default_mask = override_vals <= 0
        if default_mask.any():
            env_subset = env_ids_t[default_mask]
            self.cube_init_z[env_subset] = cube_height[default_mask]
            self._set_stage(env_subset, 0, force_reentry=True,
                            enqueue_pending=False)

        target_mask = override_vals > 0
        if target_mask.any():
            env_subset = env_ids_t[target_mask]
            stages = override_vals[target_mask]
            for stage_val in torch.unique(stages):
                env_stage = env_subset[stages == stage_val]
                if env_stage.numel() == 0:
                    continue
                self._set_stage(
                    env_stage,
                    int(stage_val.item()),
                    force_reentry=True,
                    enqueue_pending=False,
                )

        stage_overrides[env_ids_t] = -1
        restore_locomanip_snapshot_aux_state(
            self._env, env_ids_t, command=self)

    @torch.no_grad()
    def _update_command(self):
        stage1_diag = self.stage1_transition_diagnostics()
        self._accumulate_stage0_transition_episode_flags(stage1_diag)
        self._update_stage1_ready_history(stage1_diag["stage1_ready"])
        grasp_ready = self.is_grasped()
        lift_ready = self.is_lifted()
        hand_prepped = stage1_diag["hand_prepped"]
        foot_stance_ready = stage1_diag["foot_stance_ready"]

        heading_error = self.heading_error_to_target()
        turn_done = heading_error < self.cfg.turn_yaw_threshold
        if self.cfg.turn_distance_threshold > 0.0:
            turn_done = turn_done & (
                self.base_distance_to_target_xy() < self.cfg.turn_distance_threshold)

        approach_target_done = self.base_distance_to_target_xy(
        ) < self.cfg.approach_target_distance_threshold
        place_done = self.is_cube_at_target()

        stage0 = self.stage_id == 0
        stage1 = self.stage_id == 1
        stage2 = self.stage_id == 2
        stage3 = self.stage_id == 3
        stage4 = self.stage_id == 4

        table_collision = stage1_diag["table_collision"]

        to_stage1 = torch.where(stage0 & hand_prepped &
                                foot_stance_ready & (~table_collision))[0]
        to_stage2 = torch.where(stage1 & grasp_ready & lift_ready)[0]
        to_stage3 = torch.where(stage2 & turn_done)[0]
        to_stage4 = torch.where(stage3 & approach_target_done & grasp_ready)[0]
        hold_stage4 = torch.where(stage4 & place_done)[0]

        if to_stage1.numel() > 0:
            self._set_stage(to_stage1, 1)
        if to_stage2.numel() > 0:
            self._set_stage(to_stage2, 2)
        if to_stage3.numel() > 0:
            self._set_stage(to_stage3, 3)
        if to_stage4.numel() > 0:
            self._set_stage(to_stage4, 4)
        if hold_stage4.numel() > 0:
            self._set_stage(hold_stage4, 4)

        advanced_ids = [ids for ids in (
            to_stage1, to_stage2, to_stage3, to_stage4) if ids.numel() > 0]
        self._tick_stage_clocks(torch.cat(advanced_ids)
                                if len(advanced_ids) > 0 else None)
        self._flush_pending_stage_snapshots()

    def reset_stage0_transition_episode_flags(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        for key in self._stage0_transition_episode_keys:
            self._stage0_transition_episode_flags[key][env_ids] = False

    def reset_stage1_ready_history(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._stage1_ready_history[env_ids] = False

    def restore_stage1_ready_history(self, env_ids: torch.Tensor, history: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        history_t = torch.as_tensor(
            history, device=self.device, dtype=torch.bool)
        if history_t.ndim != 2:
            raise ValueError(
                f"Expected stage1-ready history with shape [N, H], got {tuple(history_t.shape)}.")
        if history_t.shape[0] != env_ids.numel():
            raise ValueError(
                f"Stage1-ready history batch size ({history_t.shape[0]}) does not match env_ids ({env_ids.numel()})."
            )
        if history_t.shape[1] != self._stage1_ready_history.shape[1]:
            raise ValueError(
                f"Stage1-ready history length ({history_t.shape[1]}) does not match configured length "
                f"({self._stage1_ready_history.shape[1]})."
            )
        self._stage1_ready_history[env_ids] = history_t

    def _update_stage1_ready_history(self, stage1_ready: torch.Tensor) -> None:
        if self._stage1_ready_history.shape[1] <= 1:
            self._stage1_ready_history[:, 0] = stage1_ready.bool()
            return
        self._stage1_ready_history[:, :-
                                   1] = self._stage1_ready_history[:, 1:].clone()
        self._stage1_ready_history[:, -1] = stage1_ready.bool()

    def _stage1_ready_history_stable_mask(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        stable = self._stage1_ready_history.all(dim=-1)
        if env_ids is None:
            return stable
        return stable[env_ids]

    def _accumulate_stage0_transition_episode_flags(self, diag: dict[str, torch.Tensor]) -> None:
        stage0_mask = self.stage_id == 0
        for key in self._stage0_transition_episode_keys:
            value = diag.get(key)
            if isinstance(value, torch.Tensor) and value.numel() == self.num_envs:
                self._stage0_transition_episode_flags[key][stage0_mask] |= value[stage0_mask].bool(
                )

    def hand_above_cube(self, height_offset: float | None = None) -> torch.Tensor:
        """Require the hand to stay below the prep-height ceiling over the cube.

        Height and XY both use the mean of base + fingertips to avoid fingertip sagging tricks
        and palm-only alignment shortcuts.
        """
        height_offset = self._hand_prep_height_offset if height_offset is None else height_offset
        hand_z = self._hand_mean_height()
        cube_ref = self.cube_init_z if hasattr(
            self, "cube_init_z") else self.cube.data.root_pos_w[:, 2]
        upper = cube_ref + float(height_offset)
        height_ok = hand_z <= upper

        if self._hand_prep_xy_threshold > 0.0:
            hand_xy = self._hand_mean_xy()
            cube_xy = self.cube.data.root_pos_w[:, :2]
            xy_dist = torch.norm(hand_xy - cube_xy, dim=-1)
            height_ok = height_ok & (xy_dist <= self._hand_prep_xy_threshold)

        return height_ok

    def is_right_hand_open(self, tolerance: float | None = None, required_fraction: float | None = None) -> torch.Tensor:
        """Hand is considered open if most joints stay near their default (open) pose."""
        tolerance = self._hand_open_tolerance if tolerance is None else tolerance
        required_fraction = self._hand_open_fraction if required_fraction is None else required_fraction
        joint_pos = self.robot.data.joint_pos[:, self._right_hand_joint_ids]
        default_pos = self.robot.data.default_joint_pos[:,
                                                        self._right_hand_joint_ids]
        deviation = torch.abs(joint_pos - default_pos)
        open_mask = deviation < float(tolerance)
        open_fraction = open_mask.float().mean(dim=-1)
        return open_fraction >= float(required_fraction)

    def hand_base_above_cube(self, margin: float = 0.0) -> torch.Tensor:
        """Require hand base link to hover above the cube top."""
        hand_z = self.robot.data.body_pos_w[:, self.hand_base_id, 2]
        cube_top = self.cube.data.root_pos_w[:, 2] + 0.025  # cube half-height
        return hand_z > (cube_top + float(margin))

    def is_hand_flat(self, angle_tolerance: float | None = None) -> torch.Tensor:
        """Hand base link palm normal roughly aligns with world -Z (palm down)."""
        angle_tolerance = self._hand_flat_angle_tolerance if angle_tolerance is None else angle_tolerance
        angle = self.hand_flat_angle()
        return angle < float(angle_tolerance)

    def hand_flat_angle(self) -> torch.Tensor:
        """Angle between the configured palm normal and world -Z in radians."""
        hand_quat = self.robot.data.body_quat_w[:, self.hand_base_id]
        world_down = torch.tensor(
            [0.0, 0.0, -1.0], device=self.device).view(1, 3).expand(hand_quat.shape[0], -1)
        hand_normal = quat_apply(hand_quat, self._hand_local_normal.view(
            1, 3).expand(hand_quat.shape[0], -1))
        cos_angle = torch.clamp(
            (hand_normal * world_down).sum(dim=-1), -1.0, 1.0)
        return torch.acos(cos_angle)

    def stage1_transition_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return the live stage-0 -> 1 gate terms and supporting diagnostics."""
        base_distance = self.base_distance_to_source_xy()
        approach_done = base_distance < self.cfg.approach_distance_threshold

        hand_mean_z = self._hand_mean_height()
        cube_ref = self.cube_init_z if hasattr(
            self, "cube_init_z") else self.cube.data.root_pos_w[:, 2]
        hand_height_upper = cube_ref + float(self._hand_prep_height_offset)
        hand_height_ok = hand_mean_z <= hand_height_upper

        hand_xy = self._hand_mean_xy()
        cube_xy = self.cube.data.root_pos_w[:, :2]
        hand_xy_dist = torch.norm(hand_xy - cube_xy, dim=-1)
        hand_xy_ok = torch.ones_like(hand_xy_dist, dtype=torch.bool)
        if self._hand_prep_xy_threshold > 0.0:
            hand_xy_ok = hand_xy_dist <= self._hand_prep_xy_threshold

        hand_above_cube = hand_height_ok & hand_xy_ok
        hand_flat_angle = self.hand_flat_angle()
        hand_flat = hand_flat_angle < float(self._hand_flat_angle_tolerance)

        foot_stance_ready, foot_stagger_x = self._stage1_foot_stance_state()
        table_collision = self._table_collision_mask()

        hand_prepped = (
            hand_above_cube
            & self.is_right_hand_open()
            # & self.is_hand_flat()
        )
        stage1_ready = hand_prepped & foot_stance_ready & (~table_collision)

        return {
            "approach_done": approach_done,
            "base_distance_source_xy": base_distance,
            "hand_above_cube": hand_above_cube,
            "hand_height_ok": hand_height_ok,
            "hand_prepped": hand_prepped,
            "hand_mean_z": hand_mean_z,
            "hand_height_upper": hand_height_upper,
            "hand_xy_dist": hand_xy_dist,
            "hand_xy_ok": hand_xy_ok,
            "hand_flat": hand_flat,
            "hand_flat_angle": hand_flat_angle,
            "foot_stance_ready": foot_stance_ready,
            "foot_stagger_x": foot_stagger_x,
            "table_collision": table_collision,
            "stage1_ready": stage1_ready,
        }

    def _stage1_foot_stance_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        if (not self._require_stage1_feet_stance) or (self.left_foot_id is None) or (self.right_foot_id is None):
            zeros = torch.zeros(self.num_envs, device=self.device)
            ready = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device)
            return ready, zeros

        base_pos = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w
        left_rel_b = quat_apply_inverse(
            base_quat, self.robot.data.body_pos_w[:, self.left_foot_id] - base_pos)
        right_rel_b = quat_apply_inverse(
            base_quat, self.robot.data.body_pos_w[:, self.right_foot_id] - base_pos)
        stagger_x = torch.abs(left_rel_b[:, 0] - right_rel_b[:, 0])
        ready = stagger_x < self._foot_stagger_x_threshold

        return ready, stagger_x

    def is_stage1_foot_stance_ready(self) -> torch.Tensor:
        """Require the feet to avoid a large fore-aft stagger in the robot base frame."""
        ready, _ = self._stage1_foot_stance_state()
        return ready

    def _hand_mean_height(self) -> torch.Tensor:
        """Mean height of hand base and fingertips."""
        all_ids = torch.cat(
            [self._hand_base_id_tensor, self.fingertip_ids.to(self.device)], dim=0)
        z_vals = self.robot.data.body_pos_w[:, all_ids, 2]
        return z_vals.mean(dim=1)

    def _hand_mean_xy(self) -> torch.Tensor:
        """Mean XY position of hand base and fingertips."""
        all_ids = torch.cat(
            [self._hand_base_id_tensor, self.fingertip_ids.to(self.device)], dim=0)
        xy_vals = self.robot.data.body_pos_w[:, all_ids, :2]
        return xy_vals.mean(dim=1)

    def _flush_pending_stage_snapshots(self) -> None:
        pending = getattr(self, "_pending_stage_snapshots", None)
        if not isinstance(pending, torch.Tensor) or pending.numel() != self.num_envs:
            return

        pending_mask = pending >= 1
        if not pending_mask.any():
            return

        # Drop stale requests when the stage changes before we capture.
        same_stage = self.stage_id == pending
        stale_mask = pending_mask & (~same_stage)
        if stale_mask.any():
            pending[stale_mask] = -1
            pending_mask = pending >= 1
            if not pending_mask.any():
                return

        # Iterate over the fixed set of possible stage values instead of using
        # torch.unique() + .item(), which forces GPU→CPU syncs on every call.
        num_stages = int(getattr(self.cfg, "num_stages", 5))
        for stage_val in range(1, num_stages):
            env_mask = pending_mask & (pending == stage_val)
            if not env_mask.any():
                continue
            env_ids = torch.where(env_mask)[0]
            stored = self._maybe_store_stage_snapshots(env_ids, stage_val)
            if isinstance(stored, torch.Tensor) and stored.numel() > 0:
                pending[stored] = -1

    def _update_metrics(self):
        self.metrics["stage_id"] = self.stage_id.float()
        self.metrics["stage_time"] = self.time_in_stage_buf.float()
        self.metrics["actual_stage_time"] = self.actual_time_in_stage_buf.float()
        self.metrics["stage_time_limit"] = self.max_stage_time_for_current_stage(
        ).float()
        self.metrics["distance_base_source_xy"] = self.base_distance_to_source_xy()
        self.metrics["distance_base_target_xy"] = self.base_distance_to_target_xy()
        self.metrics["distance_hand_cube"] = self.hand_cube_distance()
        self.metrics["cube_lift_height"] = self.cube.data.root_pos_w[:,
                                                                     2] - self.cube_init_z
        self.metrics["error_heading"] = self.heading_error_to_target()

        if self.target_region is not None:
            cube_pos = self.cube.data.root_pos_w
            target_pos = self.target_region.data.root_pos_w
            self.metrics["error_place_pos"] = torch.norm(
                cube_pos - target_pos, dim=-1)

            cube_quat = self.cube.data.root_quat_w
            target_quat = self.target_region.data.root_quat_w
            self.metrics["error_place_ori"] = quat_error_magnitude(
                cube_quat, target_quat)
        else:
            self.metrics["error_place_pos"] = torch.zeros(
                self.num_envs, device=self.device)
            self.metrics["error_place_ori"] = torch.zeros(
                self.num_envs, device=self.device)

        contact = self.fingertip_contact()
        if contact is None:
            self.metrics["fingertip_contact"] = torch.zeros(
                self.num_envs, device=self.device)
        else:
            self.metrics["fingertip_contact"] = contact.float()

        _, foot_stagger_x = self._stage1_foot_stance_state()
        self.metrics["foot_stagger_x"] = foot_stagger_x
        self.metrics["stage1_ready_history_frac"] = self._stage1_ready_history.float(
        ).mean(dim=-1)
        self.metrics["stage1_ready_history_stable"] = self._stage1_ready_history.all(
            dim=-1).float()

    def _table_collision_mask(self) -> torch.Tensor:
        """Detect significant contact on source/target tables to gate stage transition."""
        if len(self._table_contact_sensors) == 0:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        collision = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)
        for sensor in self._table_contact_sensors:
            forces = getattr(sensor.data, "force_matrix_w_history", None)
            if forces is None:
                forces = getattr(sensor.data, "net_forces_w_history", None)
            if forces is None:
                forces = getattr(sensor.data, "net_forces_w", None)
            if isinstance(forces, torch.Tensor):
                threshold = float(
                    getattr(getattr(sensor, "cfg", None), "force_threshold", 10.0))
                contact = forces.norm(dim=-1) > threshold
                contact = contact.view(contact.shape[0], -1).any(dim=-1)
                collision = collision | contact
        return collision

    def _maybe_store_stage_snapshots(self, env_ids: torch.Tensor, stage: int) -> torch.Tensor:
        if stage <= 0:
            return torch.zeros((0,), device=self.device, dtype=torch.long)
        env = getattr(self, "_env", None)
        if env is None:
            return torch.zeros((0,), device=self.device, dtype=torch.long)
        reset_cfg = getattr(env.cfg, "events", None)
        params_cfg = getattr(reset_cfg, "reset_scene", None)
        params_dict = getattr(params_cfg, "params", {}) if params_cfg else {}
        rsi_cfg = params_dict.get("rsi_cfg")
        if not (rsi_cfg and getattr(rsi_cfg, "enabled", False)):
            return torch.zeros((0,), device=self.device, dtype=torch.long)
        buffer = getattr(env, "_no_demo_rsi_buffer", None)
        if buffer is None:
            return torch.zeros((0,), device=self.device, dtype=torch.long)

        env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long)
        env_ids_t = env_ids_t.to(device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return env_ids_t

        term_cfg = getattr(env.cfg, "terminations", None)
        params_rf = getattr(getattr(term_cfg, "robot_fall",
                            None), "params", {}) if term_cfg else {}
        min_height = float(params_rf.get("min_height", 0.0))
        heights = self.robot.data.root_pos_w[env_ids_t, 2]
        keep = heights >= min_height

        stage_steps = getattr(self, "_stage_steps", None)
        min_stage_steps_cfg = getattr(
            rsi_cfg, "min_stage_steps_before_store", 0)
        min_stage_steps = 0
        if isinstance(min_stage_steps_cfg, (list, tuple)) and len(min_stage_steps_cfg) > 0:
            min_stage_steps = int(
                min_stage_steps_cfg[min(stage, len(min_stage_steps_cfg) - 1)])
        elif isinstance(min_stage_steps_cfg, torch.Tensor) and min_stage_steps_cfg.numel() > 0:
            min_stage_steps = int(min_stage_steps_cfg[min(
                stage, min_stage_steps_cfg.numel() - 1)].item())
        else:
            try:
                min_stage_steps = int(min_stage_steps_cfg)
            except (TypeError, ValueError):
                min_stage_steps = 0
        if stage_steps is not None and min_stage_steps > 0:
            steps_here = stage_steps[env_ids_t]
            keep = keep & (steps_here >= min_stage_steps)

        if stage == 1 and self._stage1_ready_history.shape[1] > 0:
            keep = keep & self._stage1_ready_history_stable_mask(env_ids_t)

        # velocity-based filtering using RSI config
        max_root_lin = float(getattr(rsi_cfg, "max_root_lin_vel", 0.0)) if getattr(
            rsi_cfg, "max_root_lin_vel", None) is not None else None
        max_root_ang = float(getattr(rsi_cfg, "max_root_ang_vel", 0.0)) if getattr(
            rsi_cfg, "max_root_ang_vel", None) is not None else None
        max_joint_vel = float(getattr(rsi_cfg, "max_joint_vel", 0.0)) if getattr(
            rsi_cfg, "max_joint_vel", None) is not None else None
        max_cube_lin = float(getattr(rsi_cfg, "max_cube_lin_vel", 0.0)) if getattr(
            rsi_cfg, "max_cube_lin_vel", None) is not None else None
        max_cube_ang = float(getattr(rsi_cfg, "max_cube_ang_vel", 0.0)) if getattr(
            rsi_cfg, "max_cube_ang_vel", None) is not None else None

        if max_root_lin is not None:
            root_lin = torch.norm(
                self.robot.data.root_lin_vel_w[env_ids_t], dim=-1)
            keep = keep & (root_lin <= max_root_lin)
        if max_root_ang is not None:
            root_ang = torch.norm(
                self.robot.data.root_ang_vel_w[env_ids_t], dim=-1)
            keep = keep & (root_ang <= max_root_ang)
        if max_joint_vel is not None:
            joint_vel = torch.norm(
                self.robot.data.joint_vel[env_ids_t], dim=-1)
            keep = keep & (joint_vel <= max_joint_vel)
        if max_cube_lin is not None:
            cube_lin = torch.norm(
                self.cube.data.root_lin_vel_w[env_ids_t], dim=-1)
            keep = keep & (cube_lin <= max_cube_lin)
        if max_cube_ang is not None:
            cube_ang = torch.norm(
                self.cube.data.root_ang_vel_w[env_ids_t], dim=-1)
            keep = keep & (cube_ang <= max_cube_ang)

        if stage >= 2:
            params_drop = getattr(
                getattr(term_cfg, "cube_dropped", None), "params", {}) if term_cfg else {}
            drop_height = float(params_drop.get("drop_height", 0.0))
            cube_z = self.cube.data.root_pos_w[env_ids_t, 2]
            init_z = self.cube_init_z[env_ids_t]
            keep = keep & (cube_z >= (init_z - drop_height))

        if not keep.any():
            return torch.zeros((0,), device=self.device, dtype=torch.long)

        env_valid = env_ids_t[keep]
        snapshot = capture_locomanip_snapshot(
            env, env_valid, command_name=getattr(self, "name", "stage"))
        if snapshot is None or snapshot.numel() == 0:
            return torch.zeros((0,), device=self.device, dtype=torch.long)
        buffer.add(stage, env_valid, snapshot)
        return env_valid

    def _resolve_max_stage_time(self) -> torch.Tensor | None:
        max_stage_time = tuple(getattr(self.cfg, "max_stage_time", ()) or ())
        if len(max_stage_time) == 0:
            return None
        if len(max_stage_time) != int(self.cfg.num_stages):
            raise ValueError(
                f"max_stage_time length ({len(max_stage_time)}) must match num_stages ({self.cfg.num_stages})."
            )
        max_stage_time_t = torch.as_tensor(
            max_stage_time, dtype=torch.long, device=self.device)
        if torch.any(max_stage_time_t <= 0):
            raise ValueError(
                "max_stage_time entries must all be greater than 0 env steps.")
        return max_stage_time_t

    def _tick_stage_clocks(self, advanced_env_ids: torch.Tensor | None = None) -> None:
        tick_mask = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device)
        if isinstance(advanced_env_ids, torch.Tensor) and advanced_env_ids.numel() > 0:
            tick_mask[advanced_env_ids.to(
                device=self.device, dtype=torch.long)] = False
        if tick_mask.any():
            self._stage_steps[tick_mask] += 1
            self.time_in_stage_buf[tick_mask] += 1

    def max_stage_time_for_current_stage(self) -> torch.Tensor:
        if self._max_stage_time is None:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        stage_idx = self.stage_id.clamp(
            min=0, max=self._max_stage_time.numel() - 1)
        return self._max_stage_time[stage_idx]

    def _set_stage(
        self,
        env_ids: torch.Tensor,
        stage: int,
        force_reentry: bool | torch.Tensor = False,
        enqueue_pending: bool = True,
    ) -> None:
        if env_ids.numel() == 0:
            return
        stage = int(stage)
        current_stage = self.stage_id[env_ids]
        stage_steps = getattr(self, "_stage_steps", None)
        pending = getattr(self, "_pending_stage_snapshots", None)

        force_reentry_mask = torch.zeros(
            env_ids.shape[0], device=self.device, dtype=torch.bool)
        if isinstance(force_reentry, torch.Tensor):
            force_reentry_mask = force_reentry.to(
                device=self.device, dtype=torch.bool)
            if force_reentry_mask.ndim == 0:
                force_reentry_mask = force_reentry_mask.expand(
                    env_ids.shape[0])
            elif force_reentry_mask.shape[0] != env_ids.shape[0]:
                raise ValueError(
                    "force_reentry tensor must be scalar or match env_ids batch size."
                )
            change_mask = (current_stage != stage) | force_reentry_mask
        elif force_reentry:
            # Scalar True: every env in the batch counts as a reentry.
            force_reentry_mask = torch.ones(
                env_ids.shape[0], device=self.device, dtype=torch.bool)
            change_mask = torch.ones(
                env_ids.shape[0], device=self.device, dtype=torch.bool)
        else:
            # Scalar False (the common path from _update_command): no allocation.
            change_mask = current_stage != stage
        if not change_mask.any():
            return

        env_changed = env_ids[change_mask]
        previous_stage = current_stage[change_mask]
        previous_time = self.time_in_stage_buf[env_changed].clone()
        changed_force_reentry = force_reentry_mask[change_mask]
        self.stage_id[env_changed] = stage
        self.stage_one_hot[env_changed] = 0.0
        self.stage_one_hot[env_changed, stage] = 1.0
        if stage_steps is not None:
            stage_steps[env_changed] = 0
        new_time = torch.zeros(
            env_changed.shape[0], dtype=torch.long, device=self.device)
        carry_time = (
            self._max_stage_time is not None
            and bool(getattr(self.cfg, "award_remaining_time_on_advance", False))
        )
        if carry_time:
            advance_mask = (previous_stage >= 0) & (
                stage > previous_stage) & (~changed_force_reentry)
            if advance_mask.any():
                previous_stage_idx = previous_stage[advance_mask].clamp(
                    min=0, max=self._max_stage_time.numel() - 1)
                new_time[advance_mask] = previous_time[advance_mask] - \
                    self._max_stage_time[previous_stage_idx]
        self.time_in_stage_buf[env_changed] = new_time
        if isinstance(pending, torch.Tensor):
            if stage <= 0 or not enqueue_pending:
                pending[env_changed] = -1
            else:
                pending[env_changed] = stage

    def _resolve_goal_pos(self, name: str) -> torch.Tensor:
        name = name.lower()
        if name in ("cube", "bottle"):
            return self.cube.data.root_pos_w
        if name == "table":
            return self.source_table.data.root_pos_w
        if name == "target_table":
            if self.target_table is None:
                raise ValueError(
                    "No target_table asset is configured for this command.")
            return self.target_table.data.root_pos_w
        if name == "target_region":
            if self.target_region is None:
                raise ValueError(
                    "No target_region asset is configured for this command.")
            return self.target_region.data.root_pos_w
        raise ValueError(f"Unknown goal name '{name}'.")

    def base_distance_to_source_xy(self) -> torch.Tensor:
        target_pos = self._resolve_goal_pos(self.cfg.source_goal_name)
        base_pos = self.robot.data.root_pos_w
        delta = base_pos[:, :2] - target_pos[:, :2]
        return torch.norm(delta, dim=-1)

    def base_distance_to_target_xy(self) -> torch.Tensor:
        target_pos = self._resolve_goal_pos(self.cfg.target_goal_name)
        base_pos = self.robot.data.root_pos_w
        delta = base_pos[:, :2] - target_pos[:, :2]
        return torch.norm(delta, dim=-1)

    def heading_error_to_goal(self, goal_name: str) -> torch.Tensor:
        target_pos = self._resolve_goal_pos(goal_name)
        base_pos = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w
        if target_pos.dim() == 1:
            target_pos = target_pos.unsqueeze(0)
        if base_pos.dim() == 1:
            base_pos = base_pos.unsqueeze(0)
        if base_quat.dim() == 1:
            base_quat = base_quat.unsqueeze(0)
        forward_axis = self._forward_axis
        if forward_axis.dim() == 1:
            forward_axis = forward_axis.view(1, 3)
        forward_axis = forward_axis.expand(base_quat.shape[0], -1)
        forward = quat_apply(base_quat, forward_axis)
        if forward.dim() == 1:
            forward = forward.unsqueeze(0)
        forward_xy = forward[:, :2]
        target_dir = target_pos[:, :2] - base_pos[:, :2]
        forward_xy = forward_xy / \
            (forward_xy.norm(dim=-1, keepdim=True) + 1.0e-6)
        target_dir = target_dir / \
            (target_dir.norm(dim=-1, keepdim=True) + 1.0e-6)
        cos_heading = torch.sum(forward_xy * target_dir,
                                dim=-1).clamp(-1.0, 1.0)
        return torch.acos(cos_heading)

    def heading_error_to_target(self) -> torch.Tensor:
        return self.heading_error_to_goal(self.cfg.target_goal_name)

    def hand_cube_distance(self) -> torch.Tensor:
        cube_pos = self.cube.data.root_pos_w[:, None, :]
        tip_pos = self.robot.data.body_pos_w[:, self.fingertip_ids]
        dist = torch.norm(tip_pos - cube_pos, dim=-1)
        return dist.min(dim=-1).values

    def fingertip_contact(self) -> torch.Tensor | None:
        if len(self._contact_sensors) == 0:
            return None
        contacts = []
        for sensor in self._contact_sensors:
            forces = getattr(sensor.data, "force_matrix_w", None)
            if forces is None:
                forces = getattr(sensor.data, "force_matrix_w_history", None)
            if forces is None:
                forces = getattr(sensor.data, "net_forces_w", None)
            if isinstance(forces, torch.Tensor):
                forces = torch.nan_to_num(
                    forces, nan=0.0, posinf=0.0, neginf=0.0)
                magnitudes = forces.norm(dim=-1)
                # Collapse any body/history dimensions: one bool per env per sensor.
                magnitudes_flat = magnitudes.view(magnitudes.shape[0], -1)
                contact = (magnitudes_flat >
                           self.cfg.contact_force_threshold).any(dim=-1)
                contacts.append(contact)
        if len(contacts) == 0:
            return None
        contact_count = contacts[0].int()
        for c in contacts[1:]:
            contact_count = contact_count + c.int()
        min_required = int(getattr(self.cfg, "min_fingertip_contacts", 1))
        passed_count = contact_count >= min_required

        required_names = getattr(self.cfg, "required_contact_sensors", ())
        for required_name in required_names:
            if required_name in self._contact_sensor_names:
                idx = self._contact_sensor_names.index(required_name)
                passed_count = passed_count & contacts[idx]
        return passed_count

    def is_grasped(self) -> torch.Tensor:
        dist_ok = self.hand_cube_distance() < self.cfg.grasp_distance_threshold
        if not self.cfg.require_contact:
            return dist_ok
        contact = self.fingertip_contact()
        if contact is None:
            return dist_ok
        return dist_ok & contact

    def is_lifted(self) -> torch.Tensor:
        cube_z = self.cube.data.root_pos_w[:, 2]
        return cube_z > (self.cube_init_z + self.cfg.lift_height)

    def is_cube_at_target(self) -> torch.Tensor:
        cube_pos = self.cube.data.root_pos_w
        target_pos = self.target_region.data.root_pos_w
        pos_err = torch.norm(cube_pos - target_pos, dim=-1)
        cube_quat = self.cube.data.root_quat_w
        target_quat = self.target_region.data.root_quat_w
        ori_err = quat_error_magnitude(cube_quat, target_quat)
        return (pos_err < self.cfg.place_pos_threshold) & (ori_err < self.cfg.place_ori_threshold)


@configclass
class LocomanipStageCommandCfg(CommandTermCfg):
    class_type: type = LocomanipStageCommand

    asset_name: str = MISSING
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    cube_name: str = "cube"
    source_table_name: str = "table"
    target_table_name: str = "target_table"
    target_region_name: str = "target_region"

    contact_sensor_name: str | tuple[str, ...] = ()
    fingertip_body_names: tuple[str, ...] = tuple(RIGHT_HAND_TIP_NAMES)

    hand_root_body_names: tuple[str, ...] = ()
    hand_base_body_name: str = "right_palm_link"
    hand_joint_names: tuple[str, ...] = ()
    hand_palm_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    hand_prep_xy_threshold: float = 0.0
    hand_flat_angle_tolerance: float = 0.35
    require_stage1_feet_stance: bool = False
    left_foot_body_name: str = "left_ankle_roll_link"
    right_foot_body_name: str = "right_ankle_roll_link"
    foot_stagger_x_threshold: float = 0.08
    contact_force_threshold: float = 1.0
    min_fingertip_contacts: int = 1
    required_contact_sensors: tuple[str, ...] = ()
    require_contact: bool = False

    num_stages: int = 5
    max_stage_time: tuple[int, ...] = ()
    award_remaining_time_on_advance: bool = False
    reset_on_overtime: bool = True
    source_goal_name: str = "cube"
    target_goal_name: str = "target_table"

    approach_distance_threshold: float = 0.5
    approach_target_distance_threshold: float = 0.5
    grasp_distance_threshold: float = 0.08
    grasp_root_distance_threshold: float | None = None
    lift_height: float = 0.1
    hand_prep_height_offset: float = 0.125
    turn_yaw_threshold: float = 0.4
    turn_distance_threshold: float = 0.0
    place_pos_threshold: float = 0.06
    place_ori_threshold: float = 0.35


class WalkPickTurnStageCommand(LocomanipStageCommand):
    cfg: "WalkPickTurnStageCommandCfg"

    def __init__(self, cfg: "WalkPickTurnStageCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._turn_entry_yaw = self._current_yaw()

    def _current_yaw(self) -> torch.Tensor:
        forward = quat_apply(self.robot.data.root_quat_w,
                             self._forward_axis.expand(self.num_envs, -1))
        return torch.atan2(forward[:, 1], forward[:, 0])

    def _turn_delta(self) -> torch.Tensor:
        delta = self._current_yaw() - self._turn_entry_yaw
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    def _turn_done(self) -> torch.Tensor:
        return torch.abs(self._turn_delta()) >= (math.pi - float(self.cfg.turn_yaw_threshold))

    def _set_stage(
        self,
        env_ids: torch.Tensor,
        stage: int,
        force_reentry: bool | torch.Tensor = False,
        enqueue_pending: bool = True,
    ) -> None:
        super()._set_stage(env_ids, stage, force_reentry=force_reentry,
                           enqueue_pending=enqueue_pending)
        if int(stage) == 2 and env_ids.numel() > 0:
            self._turn_entry_yaw[env_ids.to(device=self.device, dtype=torch.long)] = self._current_yaw()[
                env_ids.to(device=self.device, dtype=torch.long)
            ]

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        env_ids_t = torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() > 0:
            self._turn_entry_yaw[env_ids_t] = self._current_yaw()[env_ids_t]

    def _update_command(self):
        stage1_diag = self.stage1_transition_diagnostics()
        self._accumulate_stage0_transition_episode_flags(stage1_diag)
        self._update_stage1_ready_history(stage1_diag["stage1_ready"])

        stage0 = self.stage_id == 0
        stage1 = self.stage_id == 1
        table_collision = stage1_diag["table_collision"]
        to_stage1 = torch.where(
            stage0 & stage1_diag["hand_prepped"] & stage1_diag["foot_stance_ready"] & (
                ~table_collision)
        )[0]
        to_stage2 = torch.where(
            stage1 & self.is_grasped() & self.is_lifted())[0]

        if to_stage1.numel() > 0:
            self._set_stage(to_stage1, 1)
        if to_stage2.numel() > 0:
            self._set_stage(to_stage2, 2)

        advanced_ids = [ids for ids in (
            to_stage1, to_stage2) if ids.numel() > 0]
        self._tick_stage_clocks(torch.cat(advanced_ids)
                                if len(advanced_ids) > 0 else None)
        self._flush_pending_stage_snapshots()

    def is_success(self) -> torch.Tensor:
        return (self.stage_id == 2) & self.is_grasped() & self.is_lifted() & self._turn_done()

    def _update_metrics(self):
        self.metrics["stage_id"] = self.stage_id.float()
        self.metrics["stage_time"] = self.time_in_stage_buf.float()
        self.metrics["actual_stage_time"] = self.actual_time_in_stage_buf.float()
        self.metrics["stage_time_limit"] = self.max_stage_time_for_current_stage(
        ).float()
        self.metrics["distance_base_source_xy"] = self.base_distance_to_source_xy()
        self.metrics["distance_hand_cube"] = self.hand_cube_distance()
        self.metrics["cube_lift_height"] = self.cube.data.root_pos_w[:,
                                                                     2] - self.cube_init_z
        self.metrics["turn_delta_abs"] = torch.abs(self._turn_delta())
        self.metrics["walkpickturn_success"] = self.is_success().float()
        contact = self.fingertip_contact()
        self.metrics["fingertip_contact"] = (
            torch.zeros(
                self.num_envs, device=self.device) if contact is None else contact.float()
        )
        _, foot_stagger_x = self._stage1_foot_stance_state()
        self.metrics["foot_stagger_x"] = foot_stagger_x
        self.metrics["stage1_ready_history_frac"] = self._stage1_ready_history.float(
        ).mean(dim=-1)
        self.metrics["stage1_ready_history_stable"] = self._stage1_ready_history.all(
            dim=-1).float()


@configclass
class WalkPickTurnStageCommandCfg(LocomanipStageCommandCfg):
    class_type: type = WalkPickTurnStageCommand

    num_stages: int = 5
    target_table_name: str | None = None
    target_region_name: str | None = None
    target_goal_name: str = "table"
    turn_yaw_threshold: float = 0.4


class WalkgrabStageCommand(LocomanipStageCommand):
    """Single-stage walk-grab-carry command."""

    cfg: "WalkgrabStageCommandCfg"

    def __init__(self, cfg: "WalkgrabStageCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._success_root_x_threshold = float(
            getattr(cfg, "success_root_x_threshold", 2.0))
        self._success_lift_threshold = float(
            getattr(cfg, "success_lift_threshold", 0.02))

        self.metrics["walkgrab_success"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["walkgrab_root_x"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["stage_fingertip_bottle_mean_distance"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["stage_bottle_lift_height"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["right_arm_qpos_tracking_error"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["right_hand_qpos_tracking_error"] = torch.zeros(
            self.num_envs, device=self.device)

    @torch.no_grad()
    def _update_command(self):
        self._tick_stage_clocks()
        self._flush_pending_stage_snapshots()

    def fingertip_mean_distance_to_cube(self) -> torch.Tensor:
        cube_pos = self.cube.data.root_pos_w[:, None, :]
        tip_pos = self.robot.data.body_pos_w[:, self.fingertip_ids]
        dist = torch.norm(tip_pos - cube_pos, dim=-1)
        return dist.mean(dim=-1)

    def hand_root_mean_distance_to_cube(self) -> torch.Tensor | None:
        """Mean distance to cube over the finger-proximal + palm bodies.

        Returns None when ``hand_root_body_names`` was not configured. Used by
        ``is_grasped`` as a curl-invariant cross-check (and by the
        ``right_hand_others_open`` gate) so the policy can't fake a grasp by
        curling its fingertips into the bottle while keeping the palm far
        away.
        """
        if self.hand_root_ids is None:
            return None
        cube_pos = self.cube.data.root_pos_w[:, None, :]
        root_pos = self.robot.data.body_pos_w[:, self.hand_root_ids]
        dist = torch.norm(root_pos - cube_pos, dim=-1)
        return dist.mean(dim=-1)

    def is_grasped(self) -> torch.Tensor:
        dist_ok = self.fingertip_mean_distance_to_cube() < self.cfg.grasp_distance_threshold

        root_thr = getattr(self.cfg, "grasp_root_distance_threshold", None)
        root_dist = self.hand_root_mean_distance_to_cube() if root_thr is not None else None
        if root_dist is not None:
            dist_ok = dist_ok & (root_dist < float(root_thr))

        if not self.cfg.require_contact:
            return dist_ok
        contact = self.fingertip_contact()
        if contact is None:
            return dist_ok
        return dist_ok & contact

    def root_x_from_initial(self) -> torch.Tensor:
        root_x = self.robot.data.root_pos_w[:,
                                            0] - self._env.scene.env_origins[:, 0]
        return root_x - self.robot.data.default_root_state[:, 0]

    def is_success_condition(self) -> torch.Tensor:
        bottle_lift = self.cube.data.root_pos_w[:, 2] - self.cube_init_z
        return (
            (self.root_x_from_initial() > self._success_root_x_threshold)
            & self.is_grasped()
            & (bottle_lift > self._success_lift_threshold)
        )

    def bottle_upright_angle(self) -> torch.Tensor:
        bottle_axis = torch.tensor(
            [0.0, 0.0, 1.0], device=self.device).view(1, 3)
        world_axis = quat_apply(
            self.cube.data.root_quat_w, bottle_axis.expand(self.num_envs, -1))
        cos_angle = world_axis[:, 2].clamp(-1.0, 1.0)
        return torch.acos(cos_angle)

    def is_cube_at_target(self) -> torch.Tensor:
        return self.is_success_condition()

    def is_success(self) -> torch.Tensor:
        return self.is_success_condition()

    def _update_metrics(self):
        super()._update_metrics()
        bottle_lift = self.cube.data.root_pos_w[:, 2] - self.cube_init_z
        self.metrics["walkgrab_success"] = self.is_success().float()
        self.metrics["walkgrab_root_x"] = self.root_x_from_initial()
        self.metrics["stage_fingertip_bottle_mean_distance"] = self.fingertip_mean_distance_to_cube()
        self.metrics["stage_bottle_lift_height"] = bottle_lift


@configclass
class WalkgrabStageCommandCfg(LocomanipStageCommandCfg):
    class_type: type = WalkgrabStageCommand

    cube_name: str = "bottle"
    target_table_name: str | None = None
    target_region_name: str | None = None
    num_stages: int = 1
    max_stage_time: tuple[int, ...] = ()
    award_remaining_time_on_advance: bool = False
    reset_on_overtime: bool = False
    source_goal_name: str = "bottle"
    target_goal_name: str = "bottle"
    approach_target_distance_threshold: float = 0.45
    grasp_distance_threshold: float = 0.4
    lift_height: float = 0.08
    require_contact: bool = True
    success_root_x_threshold: float = 2.0
    success_lift_threshold: float = 0.02


class FridgeStageCommand(CommandTerm):
    """Single-stage command for opening a fridge door."""

    cfg: "FridgeStageCommandCfg"

    def __init__(self, cfg: "FridgeStageCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._env = env

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.fridge: Articulation = env.scene[cfg.fridge_name]

        fingertip_names = list(
            cfg.fingertip_body_names or RIGHT_HAND_TIP_NAMES)
        fingertip_ids, fingertip_found = self.robot.find_bodies(
            fingertip_names, preserve_order=True)
        if len(fingertip_ids) != len(fingertip_names):
            missing = sorted(set(fingertip_names) - set(fingertip_found))
            raise RuntimeError(
                f"Missing fingertip bodies on '{cfg.asset_name}': {missing}")
        self.fingertip_ids = torch.as_tensor(
            fingertip_ids, dtype=torch.long, device=self.device)

        hand_base_ids, hand_base_found = self.robot.find_bodies(
            [cfg.hand_base_body_name], preserve_order=True)
        if len(hand_base_ids) == 0:
            raise RuntimeError(
                f"Missing hand base body on '{cfg.asset_name}': {set([cfg.hand_base_body_name]) - set(hand_base_found)}"
            )
        self.hand_base_id = int(hand_base_ids[0])

        door_ids, door_found = self.fridge.find_bodies(
            [cfg.door_body_name], preserve_order=True)
        if len(door_ids) == 0:
            raise RuntimeError(
                f"Missing door body on '{cfg.fridge_name}': {set([cfg.door_body_name]) - set(door_found)}"
            )
        self.door_body_id = int(door_ids[0])

        handle_reference_body_name = getattr(
            cfg, "handle_reference_body_name", None) or cfg.door_body_name
        handle_reference_ids, handle_reference_found = self.fridge.find_bodies(
            [handle_reference_body_name], preserve_order=True
        )
        if len(handle_reference_ids) == 0:
            raise RuntimeError(
                f"Missing handle reference body on '{cfg.fridge_name}': "
                f"{set([handle_reference_body_name]) - set(handle_reference_found)}"
            )
        self.handle_reference_body_id = int(handle_reference_ids[0])
        self._handle_local_pos = torch.tensor(
            cfg.handle_local_pos, dtype=torch.float, device=self.device).view(1, 3)

        joint_ids, joint_found = self.fridge.find_joints(
            [cfg.door_joint_name], preserve_order=True)
        if len(joint_ids) == 0:
            raise RuntimeError(
                f"Missing door joint on '{cfg.fridge_name}': {set([cfg.door_joint_name]) - set(joint_found)}"
            )
        self.door_joint_id = int(joint_ids[0])

        self._contact_sensors: list = []
        self._contact_sensor_names: list[str] = []
        if hasattr(env.scene, "sensors"):
            sensors = env.scene.sensors
            names = cfg.contact_sensor_name
            if isinstance(names, (list, tuple, set)):
                for name in names:
                    if name in sensors:
                        self._contact_sensors.append(sensors[name])
                        self._contact_sensor_names.append(name)
            elif names in sensors:
                self._contact_sensors.append(sensors[names])
                self._contact_sensor_names.append(names)

        self.stage_id = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.stage_one_hot = torch.zeros(
            (self.num_envs, 1), device=self.device)
        self.stage_one_hot[:, 0] = 1.0
        self._stage_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self._door_initial_pos = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device)
        self._prev_door_angle = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device)
        self._door_angle_progress = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device)
        self._door_angle_regression = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device)
        self._stage3_start_pos_w = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device)
        self._stage3_forward_axis_w = torch.zeros(
            (self.num_envs, 3), dtype=torch.float, device=self.device)
        self._forward_axis = torch.tensor(
            (1.0, 0.0, 0.0), dtype=torch.float, device=self.device)

        self.metrics["stage_id"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["distance_base_fridge_xy"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["distance_hand_handle"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["door_open_angle"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["door_open_progress"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["backward_distance"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["backward_velocity"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["fingertip_contact"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["fridge_success"] = torch.zeros(
            self.num_envs, device=self.device)
        self._held_open_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["door_held_open_steps"] = torch.zeros(
            self.num_envs, device=self.device)
        self.metrics["door_held_open_seconds"] = torch.zeros(
            self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.stage_one_hot

    @torch.no_grad()
    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids_t = env_ids if isinstance(
            env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device)
        env_ids_t = env_ids_t.to(device=self.device, dtype=torch.long)

        door_pos = self.fridge.data.joint_pos[env_ids_t, self.door_joint_id]
        self._door_initial_pos[env_ids_t] = door_pos
        self._prev_door_angle[env_ids_t] = torch.zeros_like(door_pos)
        self._door_angle_progress[env_ids_t] = 0.0
        self._door_angle_regression[env_ids_t] = 0.0
        self._stage3_start_pos_w[env_ids_t] = self.robot.data.root_pos_w[env_ids_t]
        self._stage3_forward_axis_w[env_ids_t] = self.robot_forward_axis_w()[
            env_ids_t]
        self.stage_id[env_ids_t] = 0
        self.stage_one_hot[env_ids_t] = 0.0
        self.stage_one_hot[env_ids_t, 0] = 1.0
        self._stage_steps[env_ids_t] = 0
        self._held_open_steps[env_ids_t] = 0

    @torch.no_grad()
    def _update_command(self):
        door_angle = self.door_open_angle()
        delta = door_angle - self._prev_door_angle
        self._door_angle_progress = torch.clamp(delta, min=0.0)
        self._door_angle_regression = torch.clamp(-delta, min=0.0)
        self._prev_door_angle = door_angle.clone()
        self._stage_steps.add_(1)

        held = door_angle >= float(self.cfg.held_open_angle)
        if self.cfg.held_open_require_contact:
            contact = self.fingertip_contact()
            if isinstance(contact, torch.Tensor) and contact.shape[0] == self.num_envs:
                held = held & contact.to(torch.bool)
        self._held_open_steps = torch.where(
            held,
            self._held_open_steps + 1,
            torch.zeros_like(self._held_open_steps),
        )

    def _update_metrics(self):
        self.metrics["stage_id"] = self.stage_id.float()
        self.metrics["distance_base_fridge_xy"] = self.base_distance_to_fridge_xy()
        self.metrics["distance_hand_handle"] = self.hand_handle_distance()
        self.metrics["door_open_angle"] = self.door_open_angle()
        self.metrics["door_open_progress"] = self.door_open_progress()
        self.metrics["backward_distance"] = self.backward_distance()
        self.metrics["backward_velocity"] = self.backward_velocity()
        contact = self.fingertip_contact()
        self.metrics["fingertip_contact"] = torch.zeros(
            self.num_envs, device=self.device) if contact is None else contact.float()
        self.metrics["fridge_success"] = self.is_success().float()
        self.metrics["door_held_open_steps"] = self._held_open_steps.float()
        self.metrics["door_held_open_seconds"] = self.held_open_seconds()

    def robot_forward_axis_w(self) -> torch.Tensor:
        axis = self._forward_axis.view(1, 3).expand(self.num_envs, -1)
        return quat_apply(self.robot.data.root_quat_w, axis)

    def base_distance_to_fridge_xy(self) -> torch.Tensor:
        delta = self.robot.data.root_pos_w[:,
                                           :2] - self.fridge.data.root_pos_w[:, :2]
        return torch.norm(delta, dim=-1)

    def hand_handle_distance(self) -> torch.Tensor:
        hand_pos = self.robot.data.body_pos_w[:, self.hand_base_id]
        handle_pos = self.handle_pos_w()
        return torch.norm(hand_pos - handle_pos, dim=-1)

    def fingertip_handle_distance(self) -> torch.Tensor:
        handle_pos = self.handle_pos_w().unsqueeze(1)
        tip_pos = self.robot.data.body_pos_w[:, self.fingertip_ids]
        return torch.norm(tip_pos - handle_pos, dim=-1).min(dim=-1).values

    def handle_pos_w(self) -> torch.Tensor:
        door_pos_w = self.fridge.data.body_pos_w[:,
                                                 self.handle_reference_body_id]
        door_quat_w = self.fridge.data.body_quat_w[:,
                                                   self.handle_reference_body_id]
        handle_local_pos = self._handle_local_pos.expand(self.num_envs, -1)
        return door_pos_w + quat_apply(door_quat_w, handle_local_pos)

    def door_joint_pos(self) -> torch.Tensor:
        return self.fridge.data.joint_pos[:, self.door_joint_id]

    def door_open_angle(self) -> torch.Tensor:
        return torch.abs(self.door_joint_pos() - self._door_initial_pos)

    def door_open_progress(self) -> torch.Tensor:
        return self._door_angle_progress

    def door_close_regression(self) -> torch.Tensor:
        return self._door_angle_regression

    def _retreat_forward_axis_xy(self) -> torch.Tensor:
        forward_xy = self._stage3_forward_axis_w[:, :2]
        forward_xy = forward_xy / \
            (forward_xy.norm(dim=-1, keepdim=True) + 1.0e-6)
        yaw_off = float(getattr(self.cfg, "backward_axis_yaw_offset", 0.0))
        if yaw_off == 0.0:
            return forward_xy
        cos_t = math.cos(yaw_off)
        sin_t = math.sin(yaw_off)
        fx = forward_xy[:, 0]
        fy = forward_xy[:, 1]
        rot_x = cos_t * fx + sin_t * fy
        rot_y = -sin_t * fx + cos_t * fy
        return torch.stack([rot_x, rot_y], dim=-1)

    def backward_distance(self) -> torch.Tensor:
        delta = self._stage3_start_pos_w - self.robot.data.root_pos_w
        forward_xy = self._retreat_forward_axis_xy()
        return torch.sum(delta[:, :2] * forward_xy, dim=-1)

    def backward_velocity(self) -> torch.Tensor:
        forward_xy = self._retreat_forward_axis_xy()
        vel_xy = self.robot.data.root_lin_vel_w[:, :2]
        return -torch.sum(vel_xy * forward_xy, dim=-1)

    def fingertip_contact(self) -> torch.Tensor | None:
        if len(self._contact_sensors) == 0:
            return None
        contacts: list[torch.Tensor] = []
        for sensor in self._contact_sensors:
            forces = getattr(sensor.data, "force_matrix_w", None)
            if forces is None:
                forces = getattr(sensor.data, "force_matrix_w_history", None)
            if forces is None:
                forces = getattr(sensor.data, "net_forces_w", None)
            if forces is None:
                forces = getattr(sensor.data, "net_forces_w_history", None)
            if isinstance(forces, torch.Tensor):
                forces = torch.nan_to_num(
                    forces, nan=0.0, posinf=0.0, neginf=0.0)
                contact = forces.norm(
                    dim=-1) > self.cfg.contact_force_threshold
                contacts.append(contact.view(contact.shape[0], -1).any(dim=-1))
        if len(contacts) == 0:
            return None

        stacked = torch.stack(contacts, dim=0)
        contact_count = stacked.sum(dim=0)
        min_required = int(getattr(self.cfg, "min_fingertip_contacts", 1))
        result = contact_count >= int(max(1, min_required))

        required_names = tuple(
            getattr(self.cfg, "required_contact_sensors", ()) or ())
        if required_names:
            for req_name in required_names:
                if req_name in self._contact_sensor_names:
                    idx = self._contact_sensor_names.index(req_name)
                    result = result & contacts[idx]
        return result

    def held_open_seconds(self) -> torch.Tensor:
        return self._held_open_steps.float() * float(self._env.step_dt)

    def is_success(self) -> torch.Tensor:
        door_ok = self.door_open_angle() >= float(self.cfg.success_door_angle)
        hold_ok = self._held_open_steps >= int(self.cfg.success_hold_steps)
        return door_ok & hold_ok


@configclass
class FridgeStageCommandCfg(CommandTermCfg):
    class_type: type = FridgeStageCommand

    asset_name: str = MISSING
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    fridge_name: str = "fridge"
    handle_reference_body_name: str | None = None
    handle_local_pos: tuple[float, float, float] = (
        -0.5832248999999999, -0.09136289999999997, -0.5385152000000001)
    door_body_name: str = "E_door_1"
    door_joint_name: str = "RevoluteJoint"

    contact_sensor_name: str | tuple[str, ...] = ()
    fingertip_body_names: tuple[str, ...] = tuple(RIGHT_HAND_TIP_NAMES)
    hand_base_body_name: str = "right_palm_link"
    contact_force_threshold: float = 1.0

    held_open_angle: float = math.radians(60.0)
    held_open_require_contact: bool = True
    success_door_angle: float = math.radians(70.0)
    success_hold_steps: int = 30
    min_fingertip_contacts: int = 1
    required_contact_sensors: tuple[str, ...] = ()
    backward_axis_yaw_offset: float = 0.0
