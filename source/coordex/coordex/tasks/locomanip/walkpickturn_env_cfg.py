from __future__ import annotations

from dataclasses import MISSING
import re
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import coordex.tasks.locomanip.mdp as mdp
from coordex.robots.g1_wuji import G1_WUJI_ACTION_SCALE, G1_WUJI_CFG, WUJI_RH_ACTIVE_JOINT_NAMES
from coordex.tasks.locomanip.constants import (
    MODE12_JOINT_ORDER,
    RIGHT_HAND_TIP_NAMES,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
FOOT_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
LEFT_ARM_JOINT_PATTERNS = [
    "left_shoulder_.*",
    "left_elbow_joint",
    "left_wrist_.*",
]

HAND_PREP_HEIGHT_OFFSET = 0.16


@configclass
class WalkPickTurnSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    robot: ArticulationCfg = MISSING

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/source_table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.8, 0.04),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, 0.0, 0.635),
            # pos=(0.65, 0.0, 0.535),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    target_table: RigidObjectCfg | None = None

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.3, 0.0, 0.675),
            # pos=(0.45, 0.0, 0.58),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    target_region: RigidObjectCfg | None = None

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )
    contact_force_thumb = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger1_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_index = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger2_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_middle = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger3_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_ring = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger4_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_pinky = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger5_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    cube_table_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/source_table"],
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )
    source_table_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/source_table",
        # No filter to avoid GPU contact filter warnings; cube contact is negligible
        # versus the threshold, so only significant robot impacts trigger this.
        history_length=3,
        track_air_time=True,
        force_threshold=5.0,
        debug_vis=False,
    )
    target_table_contact: ContactSensorCfg | None = None


@configclass
class CommandsCfg:
    stage = mdp.WalkPickTurnStageCommandCfg(
        asset_name="robot",
        cube_name="cube",
        source_table_name="table",
        target_table_name=None,
        target_region_name=None,
        contact_sensor_name=[
            "contact_force_thumb",
            "contact_force_index",
            "contact_force_middle",
            "contact_force_ring",
            "contact_force_pinky",
        ],
        fingertip_body_names=RIGHT_HAND_TIP_NAMES,
        hand_base_body_name="right_palm_link",
        hand_joint_names=tuple(WUJI_RH_ACTIVE_JOINT_NAMES),
        hand_palm_axis=(1.0, 0.0, 0.0),
        require_stage1_feet_stance=True,
        left_foot_body_name="left_ankle_roll_link",
        right_foot_body_name="right_ankle_roll_link",
        foot_stagger_x_threshold=0.4,
        require_contact=True,
        num_stages=5,
        source_goal_name="cube",
        target_goal_name="table",
        approach_distance_threshold=0.6,
        approach_target_distance_threshold=0.5,
        grasp_distance_threshold=0.08,
        lift_height=0.1,
        hand_prep_height_offset=HAND_PREP_HEIGHT_OFFSET,
        turn_yaw_threshold=0.4,
        turn_distance_threshold=0.0,
        place_pos_threshold=0.02,
        place_ori_threshold=0.35,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.ResidualLatentActionCfg(asset_name="robot", joint_names=[
                                            ".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        stage = ObsTerm(func=mdp.stage_one_hot, params={
                        "command_name": "stage"})

        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        humanoid_prior = ObsTerm(
            func=mdp.TrackingPriorHistoryTerm,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=MODE12_JOINT_ORDER, preserve_order=True),
                "noise_cfg": mdp.TRACKING_PRIOR_NOISE_CFG,
                "history_length": 5,
            },
        )

        right_hand_prior = ObsTerm(
            func=mdp.right_hand_proprio,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=WUJI_RH_ACTIVE_JOINT_NAMES, preserve_order=True),
                "wrist_body_name": "right_palm_link",
            },
        )
        last_latent = ObsTerm(
            func=mdp.last_residual_latent,
            params={"action_term_name": "joint_pos"},
        )

        fingertip_cube_forces = ObsTerm(
            func=mdp.fingertip_force_on_cube,
            params={
                "sensor_names": [
                    "contact_force_thumb",
                    "contact_force_index",
                    "contact_force_middle",
                    "contact_force_ring",
                    "contact_force_pinky",
                ],
                "asset_cfg": SceneEntityCfg("robot"),
                "fingertip_body_names": RIGHT_HAND_TIP_NAMES,
                "force_scale": 50.0,
                "force_clip": 1.0,
            },
        )

        cube_pose = ObsTerm(func=mdp.object_pose_in_robot_frame, params={
                            "target_cfg": SceneEntityCfg("cube")})
        source_table_pose = ObsTerm(
            func=mdp.object_pose_in_robot_frame, params={
                "target_cfg": SceneEntityCfg("table")}
        )
        target_table_pose = ObsTerm(
            func=mdp.virtual_pose_in_robot_frame,
            params={"pos": (-1.5, 0.0, 0.635)},
        )
        target_region_pose = ObsTerm(
            func=mdp.virtual_pose_in_robot_frame,
            params={"pos": (-1.3, 0.0, 0.65)},
        )
        cube_pose_in_hand = ObsTerm(
            func=mdp.cube_pose_in_hand_frame,
            params={"target_cfg": SceneEntityCfg(
                "cube"), "hand_body_name": "right_palm_link"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(
        func=mdp.reset_scene_with_no_demo_rsi,
        mode="reset",
        params={
            "rsi_cfg": mdp.NoDemoRSICfg(),
            "base_reset_func": mdp.reset_scene_to_default,
            "base_reset_kwargs": {"reset_joint_targets": True},
            "reset_joint_targets": True,
            "command_name": "stage",
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )


@configclass
class RewardsCfg:
    # All stages' rewards.
    robot_object_distance = RewTerm(
        func=mdp.robot_object_distance_reward,
        weight=15.0,
        params={
            "command_name": "stage",
            "stage_id": [0, 1, 2, 3],
            "object_name": "cube",
            "target_distance": 0.5,
            "scale": 4.0,
        },
    )

    # Stage progression bonus.
    stage_progress_bonus = RewTerm(
        func=mdp.stage_transition_bonus,
        weight=50.0,
        params={"command_name": "stage"},
    )

    # Stage 0 (approach) specific rewards.
    heading_to_object = RewTerm(
        func=mdp.heading_to_object_penalty,
        weight=-10.0,
        params={"command_name": "stage", "stage_id": 0, "object_name": "cube"},
    )

    # Keep the right hand pre-shaped and above the cube during the approach.
    right_hand_target_pose = RewTerm(
        func=mdp.right_hand_height_direction_reward,
        weight=15.0,
        params={
            "command_name": "stage",
            "stage_id": 0,
            "hand_body": "right_palm_link",
            "target_name": "cube",
            "height_offset": 0.16,
            "height_scale": 25.0,
            "gate_far": None,
            "gate_near": None,
        },
    )

    right_hand_prep_region_penalty = RewTerm(
        func=mdp.right_hand_prep_region_penalty,
        weight=-5.0,
        params={
            "command_name": "stage",
            "stage_id": 0,
            "hand_body": "right_palm_link",
            "target_name": "cube",
            "xy_scale": 60.0,
            "z_scale": 20.0,
        },
    )

    right_hand_flat_stage0 = RewTerm(
        func=mdp.right_hand_flat_penalty,
        weight=-5.0,
        params={
            "command_name": "stage",
            "stage_id": 0,
            "hand_body": "right_palm_link",
            "angle_tolerance": 0.35,
            "palm_axis": (1.0, 0.0, 0.0),
        },
    )

    keep_hand_origin = RewTerm(
        func=mdp.hand_default_pose_reward,
        weight=5.0,
        params={
            "command_name": "stage",
            "stage_id": 0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=WUJI_RH_ACTIVE_JOINT_NAMES),
            "tolerance": 0.05,
            "scale": 3.0,
        },
    )

    # grasp related specific rewards.
    hand_cube_distance = RewTerm(
        func=mdp.hand_cube_distance_reward,
        weight=20.0,
        params={
            "command_name": "stage",
            "stage_id": [1, 2, 3],
            "cube_name": "cube",
            "fingertip_body_names": [*RIGHT_HAND_TIP_NAMES, "right_palm_link"],
            "scale": 10.0,
            "distance_aggregation": "mean",
        },
    )
    fingertip_contact = RewTerm(
        func=mdp.fingertip_contact_reward,
        weight=10.0,
        params={
            "command_name": "stage",
            "stage_id": [1, 2, 3],
            "sensor_names": [
                "contact_force_thumb",
                "contact_force_index",
                "contact_force_middle",
                "contact_force_ring",
                "contact_force_pinky",
            ],
            "threshold": 1.0,
            "bonus_start_contacts": 3,
            "bonus_scale": 0.25,
            "bonus_power": 2.0,
            "gate_min_lift": 0.01,
            "cube_name": "cube",
            "table_name": "table",
            "table_top_offset": 0.04,
        },
    )
    grasp_force = RewTerm(
        func=mdp.grasp_force_reward,
        weight=10.0,
        params={
            "command_name": "stage",
            "stage_id": [1, 2, 3],
            "sensor_names": [
                "contact_force_thumb",
                "contact_force_index",
                "contact_force_middle",
                "contact_force_ring",
                "contact_force_pinky",
            ],
            "fingertip_body_names": RIGHT_HAND_TIP_NAMES,
            "force_scale": 10.0,
            "force_clip": 20.0,
            "tanh_scale": 2.0,
            "max_force": 75.0,
        },
    )
    cube_lift_height = RewTerm(
        func=mdp.cube_lift_height_reward,
        weight=50.0,
        params={
            "command_name": "stage",
            "stage_id": [1, 2, 3],
            "cube_name": "cube",
            "table_name": "table",
            "table_top_offset": 0.04,
            "cap": 0.12,
            "require_grasp": False,
        },
    )
    hold_object = RewTerm(
        func=mdp.object_hold_reward,
        weight=20.0,
        params={
            "command_name": "stage",
            "stage_id": [1, 2, 3],
        },
    )
    cube_not_lifted = RewTerm(
        func=mdp.cube_not_lifted_penalty,
        weight=-10.0,
        params={
            "command_name": "stage",
            "stage_id": [1, 2, 3],
            "cube_name": "cube",
            "table_name": "table",
            "table_top_offset": 0.04,
            "target_lift": 0.1,
        },
    )
    cube_table_contact_move = RewTerm(
        func=mdp.cube_table_contact_move_penalty,
        weight=-5.0,
        params={
            "command_name": "stage",
            "stage_id": [0, 1, 2],
            "sensor_name": "cube_table_contact",
            "cube_name": "cube",
            "threshold": 1.0,
        },
    )
    cube_hand_rel_move = RewTerm(
        func=mdp.cube_hand_rel_move_penalty,
        weight=-5.0,
        params={
            "command_name": "stage",
            "stage_id": [1, 2, 3],
            "cube_name": "cube",
            "hand_body": "right_palm_link",
        },
    )

    upright_orientation = RewTerm(
        func=mdp.upright_orientation_reward,
        weight=5.0,
        params={
            "command_name": "stage",
            "stage_id": [0, 1, 2, 3],
            "std": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )

    success_bonus = RewTerm(
        func=mdp.termination_penalty,
        weight=1000.0,
        params={"term_name": "success"},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)
    joint_action_rate_l2 = RewTerm(
        func=mdp.joint_action_rate_l2, weight=-1.0e-2)

    left_arm_joint_velocity_l2 = RewTerm(
        func=mdp.joint_velocity_l2,
        weight=-1e-2,
        params={
            "command_name": "stage",
            "stage_id": [0, 1, 2, 3],
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINT_PATTERNS),
        },
    )
    dof_velocity_l2 = RewTerm(
        func=mdp.joint_velocity_l2,
        weight=-1e-2,
        params={
            "command_name": "stage",
            "stage_id": [0, 1, 2, 3],
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )

    large_lin_vx = RewTerm(
        func=mdp.base_lin_vel_excess,
        weight=-10.0,
        params={
            "command_name": "stage",
            "stage_id": [0, 1, 2, 3],
            "axis": 0,
            "threshold": 0.5,
            "max_penalty": 5.0,
        },
    )
    large_lin_vy = RewTerm(
        func=mdp.base_lin_vel_excess,
        weight=-10.0,
        params={
            "command_name": "stage",
            "stage_id": [0, 1, 2, 3],
            "axis": 1,
            "threshold": 0.5,
            "max_penalty": 5.0,
        },
    )
    large_ang_vel = RewTerm(
        func=mdp.base_ang_vel_excess,
        weight=-5.0,
        params={
            "command_name": "stage",
            "stage_id": [0, 1, 2, 3],
            "axis": 2,
            "threshold": 0.5,
            "max_penalty": 5.0,
        },
    )
    feet_slip = RewTerm(
        func=mdp.feet_slip_penalty,
        weight=-5.0,
        params={
            "command_name": "stage",
            "stage_id": [0, 1, 2, 3],
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 5.0,
            # "clip_max": 2.0,
        },
    )
    undesired_table_contacts = RewTerm(
        func=mdp.undesired_table_contacts,
        weight=-10.0,
        params={
            "command_name": "stage",
            "stage_id": [0, 1, 2, 3],
            "sensor_names": ["source_table_contact"],
            "threshold": 5.0,
        },
    )
    robot_fall_penalty = RewTerm(
        func=mdp.termination_penalty,
        weight=-10.0,
        params={"term_name": "robot_fall"},
    )

    # When grabing and placing, no angular velocity drift
    zero_vel_stage_1_4 = RewTerm(
        func=mdp.base_velocity_l1,
        weight=-10.0,
        params={
            "command_name": "stage",
            "stage_id": [1],
            "linear_axes": [0, 1],
            "angular_axis": 2,
            "soft_cap": 3.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.stage_success, params={
                       "command_name": "stage"})
    cube_dropped = DoneTerm(
        func=mdp.cube_dropped,
        params={
            "command_name": "stage",
            "drop_height": 0.3,
            "sensor_names": [
                "contact_force_thumb",
                "contact_force_index",
                "contact_force_middle",
                "contact_force_ring",
                "contact_force_pinky",
            ],
            "contact_threshold": 1.0,
        },
    )
    # Contain rare physics explosions before they poison PPO rollouts.
    robot_root_state_blowup = DoneTerm(
        func=mdp.root_state_blowup,
        params={
            "asset_name": "robot",
            "max_xy_radius": 6.0,
            "max_height": 3.0,
            "max_lin_vel": 10.0,
            "max_ang_vel": 20.0,
        },
    )
    robot_joint_velocity_blowup = DoneTerm(
        func=mdp.joint_velocity_blowup,
        params={
            "asset_name": "robot",
            "max_abs_joint_vel": 40.0,
        },
    )
    robot_fall = DoneTerm(
        func=mdp.robot_fall,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 0.5},
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class WalkPickTurnEnvCfg(ManagerBasedRLEnvCfg):
    scene: WalkPickTurnSceneCfg = WalkPickTurnSceneCfg(
        num_envs=4096, env_spacing=5.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 60 Hz
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / (60.0 * 4)
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.scene.robot = G1_WUJI_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot")

        action_joint_names = MODE12_JOINT_ORDER + \
            list(WUJI_RH_ACTIVE_JOINT_NAMES)
        action_scale = dict(G1_WUJI_ACTION_SCALE)
        for joint_name in WUJI_RH_ACTIVE_JOINT_NAMES:
            action_scale.setdefault(joint_name, 0.1)

        action = self.actions.joint_pos
        action.scale = {
            pattern: value
            for pattern, value in action_scale.items()
            if any(re.fullmatch(pattern, joint_name) for joint_name in action_joint_names)
        }
        action.joint_names = action_joint_names
        action.preserve_order = True
        action.clip_actions = False
        action.body_prior_checkpoint = str(
            REPO_ROOT / "ckpts/body_prior/walkpickturn_30k.pt")
        action.hand_prior_checkpoint = str(
            REPO_ROOT / "ckpts/hand_prior/active_wrist_20k.pt")
        action.body_obs_norm_mode = "rsl_rl"
        action.hand_action_moving_average = 0.4
        action.hand_joint_names = tuple(WUJI_RH_ACTIVE_JOINT_NAMES)
        action.hand_wrist_body_name = "right_palm_link"
        action.hand_prior_variant = "wuji_floating_active"

        for obs_group in (self.observations.policy, self.observations.critic):
            obs_group.humanoid_prior.params["include_base_lin_vel"] = True

        self.observations.policy.body_prior_mean = ObsTerm(
            func=mdp.body_prior_mean,
            params={"action_term_name": "joint_pos"},
        )
        self.observations.policy.hand_prior_mean = ObsTerm(
            func=mdp.hand_prior_mean,
            params={"action_term_name": "joint_pos"},
        )
