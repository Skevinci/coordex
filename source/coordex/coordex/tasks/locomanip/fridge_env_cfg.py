"""Open Fridge environment for G1 Wuji."""

from __future__ import annotations

from dataclasses import MISSING
import math
import re
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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

import coordex.tasks.locomanip.mdp as mdp
from coordex.assets import ASSET_DIR
from coordex.robots.g1_wuji import G1_WUJI_ACTION_SCALE, G1_WUJI_CFG, WUJI_RH_ACTIVE_JOINT_NAMES
from coordex.tasks.locomanip.constants import MODE12_JOINT_ORDER, RIGHT_HAND_TIP_NAMES


REPO_ROOT = Path(__file__).resolve().parents[5]
FRIDGE_DOOR_BODY = "E_door_1"
FRIDGE_DOOR_JOINT = "RevoluteJoint_fridge_1_up"
FRIDGE_HANDLE_LOCAL_POS = (-0.5832248999999999, -
                           0.09136289999999997, -0.5385152000000001)
FRIDGE_HELD_OPEN_ANGLE = math.radians(35.0)
FRIDGE_SUCCESS_ANGLE = math.radians(60.0)
FRIDGE_SUCCESS_HOLD_STEPS = 30
FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


@configclass
class FridgeSceneCfg(InteractiveSceneCfg):
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

    fridge: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Fridge",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_DIR}/objects/fridge/model_fridge_1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                fix_root_link=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.6, 0.0, 0.0),
            rot=(0.70710678, 0.0, 0.0, -0.70710678),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "passive_doors": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=10.0,
                stiffness=0.0,
                damping=2.0,
                armature=0.001,
            ),
        },
    )

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
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Fridge/E_door_1",
            "{ENV_REGEX_NS}/Fridge/E_handle_1_7",
        ],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_index = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger2_tip_link",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Fridge/E_door_1",
            "{ENV_REGEX_NS}/Fridge/E_handle_1_7",
        ],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_middle = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger3_tip_link",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Fridge/E_door_1",
            "{ENV_REGEX_NS}/Fridge/E_handle_1_7",
        ],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_ring = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger4_tip_link",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Fridge/E_door_1",
            "{ENV_REGEX_NS}/Fridge/E_handle_1_7",
        ],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_pinky = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger5_tip_link",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Fridge/E_door_1",
            "{ENV_REGEX_NS}/Fridge/E_handle_1_7",
        ],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
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

        source_table_pose = ObsTerm(
            func=mdp.object_pose_in_robot_frame,
            params={"target_cfg": SceneEntityCfg("fridge")},
        )
        cube_pose = ObsTerm(
            func=mdp.body_offset_pose_in_robot_frame,
            params={
                "target_cfg": SceneEntityCfg("fridge"),
                "body_name": FRIDGE_DOOR_BODY,
                "local_pos": FRIDGE_HANDLE_LOCAL_POS,
            },
        )
        target_table_pose = ObsTerm(
            func=mdp.body_pose_in_robot_frame,
            params={"target_cfg": SceneEntityCfg(
                "fridge"), "body_name": FRIDGE_DOOR_BODY},
        )
        cube_pose_in_hand = ObsTerm(
            func=mdp.body_offset_pose_in_hand_frame,
            params={
                "target_cfg": SceneEntityCfg("fridge"),
                "body_name": FRIDGE_DOOR_BODY,
                "local_pos": FRIDGE_HANDLE_LOCAL_POS,
                "hand_body_name": "right_palm_link",
            },
        )
        door_joint_state = ObsTerm(
            func=mdp.articulation_joint_state,
            params={"asset_cfg": SceneEntityCfg(
                "fridge", joint_names=[FRIDGE_DOOR_JOINT], preserve_order=True)},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CommandsCfg:
    stage = mdp.FridgeStageCommandCfg(
        asset_name="robot",
        fridge_name="fridge",
        handle_local_pos=FRIDGE_HANDLE_LOCAL_POS,
        door_body_name=FRIDGE_DOOR_BODY,
        door_joint_name=FRIDGE_DOOR_JOINT,
        contact_sensor_name=[
            "contact_force_thumb",
            "contact_force_index",
            "contact_force_middle",
            "contact_force_ring",
            "contact_force_pinky",
        ],
        fingertip_body_names=RIGHT_HAND_TIP_NAMES,
        hand_base_body_name="right_palm_link",
        contact_force_threshold=1.0,
        held_open_angle=FRIDGE_HELD_OPEN_ANGLE,
        held_open_require_contact=True,
        success_door_angle=FRIDGE_SUCCESS_ANGLE,
        success_hold_steps=FRIDGE_SUCCESS_HOLD_STEPS,
        min_fingertip_contacts=1,
        required_contact_sensors=(),
        backward_axis_yaw_offset=math.radians(25.0),
    )


@configclass
class RewardsCfg:
    robot_fridge_distance = RewTerm(
        func=mdp.robot_object_distance_reward,
        weight=0.0,
        params={
            "command_name": "stage",
            "object_name": "fridge",
            "target_distance": 0.70,
            "scale": 4.0,
        },
    )
    forward_progress = RewTerm(
        func=mdp.base_target_progress_reward,
        weight=0.0,
        params={
            "command_name": "stage",
            "target_name": "fridge",
            "speed_scale": 0.6,
        },
    )
    heading_to_fridge = RewTerm(
        func=mdp.heading_to_object_penalty,
        weight=0.0,
        params={"command_name": "stage", "object_name": "fridge"},
    )

    hand_handle_distance = RewTerm(
        func=mdp.hand_body_distance_reward,
        weight=30.0,
        params={
            "command_name": "stage",
            "target_cfg": SceneEntityCfg("fridge"),
            "target_body_name": FRIDGE_DOOR_BODY,
            "hand_body_names": [*RIGHT_HAND_TIP_NAMES, "right_palm_link"],
            "target_body_local_pos": FRIDGE_HANDLE_LOCAL_POS,
            "scale": 10.0,
            "distance_aggregation": "mean",
        },
    )

    fingertip_contact = RewTerm(
        func=mdp.fingertip_contact_reward,
        weight=15.0,
        params={
            "command_name": "stage",
            "sensor_names": [
                "contact_force_thumb",
                "contact_force_index",
                "contact_force_middle",
                "contact_force_ring",
                "contact_force_pinky",
            ],
            "threshold": 1.0,
            "bonus_start_contacts": 2,
            "bonus_scale": 0.25,
            "bonus_power": 2.0,
        },
    )

    thumb_contact = RewTerm(
        func=mdp.fingertip_contact_reward,
        weight=10.0,
        params={
            "command_name": "stage",
            "sensor_names": ["contact_force_thumb"],
            "threshold": 1.0,
        },
    )

    door_open_amount = RewTerm(
        func=mdp.fridge_door_open_with_contact_reward,
        weight=30.0,
        params={
            "command_name": "stage",
            "target_angle": FRIDGE_SUCCESS_ANGLE,
            "power": 1.0,
            "contact_scale": 1.0,
        },
    )
    door_open_progress = RewTerm(
        func=mdp.fridge_door_open_progress_reward,
        weight=20.0,
        params={
            "command_name": "stage",
            "progress_scale": 0.02,
            "require_contact": True,
        },
    )
    door_close_regression = RewTerm(
        func=mdp.fridge_door_close_regression_reward,
        weight=-20.0,
        params={
            "command_name": "stage",
            "regression_scale": 0.02,
        },
    )

    door_held_open = RewTerm(
        func=mdp.fridge_door_held_open_reward,
        weight=100.0,
        params={
            "command_name": "stage",
            "hold_angle": FRIDGE_HELD_OPEN_ANGLE,
            "max_seconds": 1.0,
            "require_contact": True,
        },
    )

    backward_progress = RewTerm(
        func=mdp.base_backward_progress_reward,
        weight=8.0,
        params={
            "command_name": "stage",
            "speed_scale": 0.25,
            "require_contact": True,
        },
    )

    success_bonus = RewTerm(
        func=mdp.termination_penalty,
        weight=1000.0,
        params={"term_name": "success"},
    )
    robot_fall_penalty = RewTerm(
        func=mdp.termination_penalty,
        weight=-10.0,
        params={"term_name": "robot_fall"},
    )
    hand_far_penalty = RewTerm(
        func=mdp.termination_penalty,
        weight=-20.0,
        params={"term_name": "hand_far_from_handle"},
    )
    time_out_penalty = RewTerm(
        func=mdp.termination_penalty,
        weight=-20.0,
        params={"term_name": "time_out"},
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_clipped,
        weight=-1.0e-2,
        params={"clip_max": 100.0},
    )
    joint_action_rate_l2 = RewTerm(
        func=mdp.joint_action_rate_l2_clipped,
        weight=-1.0e-2,
        params={"clip_max": 50.0},
    )
    dof_velocity_l2 = RewTerm(
        func=mdp.joint_velocity_l2_clipped,
        weight=-1.0e-2,
        params={
            "command_name": "stage",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "clip_max": 200.0,
        },
    )
    feet_slip = RewTerm(
        func=mdp.feet_slip_penalty_clipped,
        weight=-5.0,
        params={
            "command_name": "stage",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 5.0,
            "clip_max": 2.0,
        },
    )
    large_lin_vx = RewTerm(
        func=mdp.base_lin_vel_excess,
        weight=-8.0,
        params={"command_name": "stage", "axis": 0, "threshold": 0.8, "max_penalty": 2.0},
    )
    large_lin_vy = RewTerm(
        func=mdp.base_lin_vel_excess,
        weight=-10.0,
        params={"command_name": "stage", "axis": 1, "threshold": 0.35, "max_penalty": 2.0},
    )
    large_ang_vel = RewTerm(
        func=mdp.base_ang_vel_excess,
        weight=-10.0,
        params={"command_name": "stage", "axis": 2, "threshold": 0.6, "max_penalty": 5.0},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.stage_success, params={
                       "command_name": "stage"})
    hand_far_from_handle = DoneTerm(
        func=mdp.hand_far_from_handle,
        params={"command_name": "stage", "max_distance": 0.5},
    )
    robot_fall = DoneTerm(
        func=mdp.robot_fall,
        params={"asset_cfg": SceneEntityCfg(
            "robot"), "body_name": "torso_link", "min_height": 0.35},
    )
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
    fridge_root_state_blowup = DoneTerm(
        func=mdp.root_state_blowup,
        params={
            "asset_name": "fridge",
            "max_xy_radius": 6.0,
            "min_height": -0.5,
            "max_height": 3.0,
            "max_lin_vel": 5.0,
            "max_ang_vel": 10.0,
        },
    )
    robot_joint_velocity_blowup = DoneTerm(
        func=mdp.joint_velocity_blowup,
        params={"asset_name": "robot", "max_abs_joint_vel": 40.0},
    )
    fridge_joint_velocity_blowup = DoneTerm(
        func=mdp.joint_velocity_blowup,
        params={"asset_name": "fridge", "max_abs_joint_vel": 20.0},
    )


@configclass
class CurriculumCfg:
    pass


@configclass
class EventCfg:
    reset_scene = EventTerm(
        func=mdp.reset_fridge_to_pregrasp_demo_frame,
        mode="reset",
        params={
            "pose_range": None,
            "joint_position_range": None,
            "reset_joint_targets": True,
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
class FridgeEnvCfg(ManagerBasedRLEnvCfg):
    scene: FridgeSceneCfg = FridgeSceneCfg(num_envs=4096, env_spacing=5.0)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):

        self.decimation = 4
        self.episode_length_s = 7.0
        self.sim.dt = 1.0 / (60.0 * 4)
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (-1.2, 2.1, 2.1)
        self.viewer.lookat = (1.0, 0.0, 0.85)
        self.viewer.origin_type = "world"
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
            REPO_ROOT / "ckpts/body_prior/fridge_2k.pt")
        action.hand_prior_checkpoint = str(
            REPO_ROOT / "ckpts/hand_prior/kinematic_wrist_16k.pt")
        action.body_obs_norm_mode = "rsl_rl"
        action.hand_action_moving_average = 0.4
        action.hand_joint_names = tuple(WUJI_RH_ACTIVE_JOINT_NAMES)
        action.hand_wrist_body_name = "right_palm_link"
        action.hand_prior_variant = "wuji_floating_kinematic_wrist"

        for obs_group in (self.observations.policy, self.observations.critic):
            obs_group.humanoid_prior.params["include_base_lin_vel"] = False

        self.observations.policy.body_prior_mean = ObsTerm(
            func=mdp.body_prior_mean,
            params={"action_term_name": "joint_pos"},
        )
        self.observations.policy.hand_prior_mean = ObsTerm(
            func=mdp.hand_prior_mean,
            params={"action_term_name": "joint_pos"},
        )
