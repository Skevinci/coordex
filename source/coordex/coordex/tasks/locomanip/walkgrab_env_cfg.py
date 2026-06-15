"""WalkGrab environment for G1 Wuji."""

from __future__ import annotations

from dataclasses import MISSING
import re
from pathlib import Path

import isaaclab.sim as sim_utils
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

import coordex.tasks.locomanip.mdp as mdp
from coordex.robots.g1_wuji import G1_WUJI_ACTION_SCALE, G1_WUJI_CFG, WUJI_RH_ACTIVE_JOINT_NAMES
from coordex.tasks.locomanip.constants import MODE12_JOINT_ORDER, RIGHT_HAND_TIP_NAMES


REPO_ROOT = Path(__file__).resolve().parents[5]
FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


@configclass
class WalkgrabSceneCfg(InteractiveSceneCfg):
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
            size=(0.25, 0.5, 0.04),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Oak_Planks.mdl",
                project_uvw=True,
            ),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, -0.55, 0.635),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    bottle: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bottle",
        spawn=sim_utils.CylinderCfg(
            radius=0.035,
            height=0.3,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.5, -0.35, 0.730),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
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
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Bottle"],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_index = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger2_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Bottle"],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_middle = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger3_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Bottle"],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_ring = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger4_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Bottle"],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_pinky = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger5_tip_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Bottle"],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    contact_force_palm = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_palm_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Bottle"],
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )
    source_table_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/source_table",
        history_length=3,
        track_air_time=True,
        force_threshold=5.0,
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

        cube_pose = ObsTerm(func=mdp.object_pose_in_robot_frame, params={
                            "target_cfg": SceneEntityCfg("bottle")})
        source_table_pose = ObsTerm(
            func=mdp.object_pose_in_robot_frame, params={
                "target_cfg": SceneEntityCfg("table")}
        )
        cube_pose_in_hand = ObsTerm(
            func=mdp.cube_pose_in_hand_frame,
            params={"target_cfg": SceneEntityCfg(
                "bottle"), "hand_body_name": "right_palm_link"},
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
class CurriculumCfg:
    pass


@configclass
class CommandsCfg:
    stage = mdp.WalkgrabStageCommandCfg(
        asset_name="robot",
        cube_name="bottle",
        source_table_name="table",
        contact_sensor_name=[
            "contact_force_thumb",
            "contact_force_index",
            "contact_force_middle",
            "contact_force_ring",
            "contact_force_pinky",
            "contact_force_palm",
        ],
        fingertip_body_names=RIGHT_HAND_TIP_NAMES,
        hand_base_body_name="right_palm_link",
        hand_joint_names=tuple(WUJI_RH_ACTIVE_JOINT_NAMES),
        require_contact=True,
        source_goal_name="bottle",
        target_goal_name="bottle",
        min_fingertip_contacts=2,
        required_contact_sensors=("contact_force_thumb",),
        grasp_distance_threshold=0.10,
        lift_height=0.08,
        success_root_x_threshold=2.0,
    )


@configclass
class RewardsCfg:
    forward_target = RewTerm(
        func=mdp.robot_grasp_gated_forward_target_reward,
        weight=25.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "post_grasp_target_x": 2.0,
            "pre_grasp_target_x": 2.0,
            "target_y": 0.0,
            "scale": 1.0,
            "cube_name": "bottle",
        },
    )

    right_hand_pre_grasp_pos = None

    hand_bottle_distance = RewTerm(
        func=mdp.hand_cube_grasp_pose_reward,
        weight=50.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "cube_name": "bottle",
            "body_groups": (
                [*RIGHT_HAND_TIP_NAMES, "right_palm_link"],
                list(RIGHT_HAND_TIP_NAMES),
            ),
            "target_distances": (0.09, 0.046),
            "scale": 8.5,
            "distance_aggregation": "mean",
            "group_aggregations": ("mean", "max"),
            "group_modes": ("monotonic", "monotonic"),
            "metric_name": "hand_bottle",
        },
    )

    palm_facing_object = RewTerm(
        func=mdp.palm_facing_object_reward,
        weight=15.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "cube_name": "bottle",
            "hand_body": "right_palm_link",
            "palm_axis": (1.0, 0.0, 0.0),
            "axis": "xy",
            "target_cos": 0.47,
            "target_cos_std": 0.25,
            "gate_pre_grasp": True,
        },
    )

    fingertip_contact = RewTerm(
        func=mdp.fingertip_contact_reward,
        weight=15.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "sensor_names": [
                "contact_force_thumb",
                "contact_force_index",
                "contact_force_middle",
                "contact_force_ring",
                "contact_force_pinky",
            ],
            "threshold": 1.0,
            "cube_name": "bottle",
        },
    )

    fingertip_centroid_on_cube = RewTerm(
        func=mdp.fingertip_centroid_on_cube_reward,
        weight=0.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "cube_name": "bottle",
            "fingertip_body_names": RIGHT_HAND_TIP_NAMES,
            "decay_scale": 20.0,
            "gate_pre_grasp": True,
            "metric_name": "fingertip_centroid_to_bottle",
        },
    )

    grasp_force = RewTerm(
        func=mdp.grasp_force_reward,
        weight=15.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "sensor_names": [
                "contact_force_thumb",
                "contact_force_index",
                "contact_force_middle",
                "contact_force_ring",
                "contact_force_pinky",
            ],
            "fingertip_body_names": RIGHT_HAND_TIP_NAMES,
            "aggregation": "mean",
            "tanh_scale": None,
            "max_force": None,
            "output_clip": 2.0,
        },
    )

    grasp_based_on_obj_finger_dir = RewTerm(
        func=mdp.grasp_based_on_obj_finger_dir_reward,
        weight=40.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "cube_name": "bottle",
            "thumb_body_name": RIGHT_HAND_TIP_NAMES[0],
            "index_body_names": list(RIGHT_HAND_TIP_NAMES[1:]),
            "normalize_to_unit": True,
            "project_plane": "xy",
            "reward_shape": "angle",
            "gate_pre_grasp": False,
            "metric_name": "thumb_oppose_cos",
        },
    )

    hand_aperture_preshape = None

    right_hand_qpos_tracking_front_object = None
    right_hand_thumb_oppose = None
    right_hand_others_open = None

    bottle_lift_height = RewTerm(
        func=mdp.cube_lift_height_reward,
        weight=100.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "cube_name": "bottle",
            "cap": 0.05,
            "require_grasp": True,
            "yank_penalty_beta": 2.0,
        },
    )

    hold_bottle = RewTerm(
        func=mdp.object_hold_reward,
        weight=100.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
        },
    )

    sustained_grasp_bonus = RewTerm(
        func=mdp.sustained_grasp_bonus_reward,
        weight=100.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "cube_name": "bottle",
            "max_seconds": 1.5,
            "min_lift": 0.01,
        },
    )

    bottle_forward_progress = RewTerm(
        func=mdp.bottle_forward_progress_reward,
        weight=100.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "cube_name": "bottle",
            "speed_scale": 1.0,
            "require_grasp": True,
            "min_lift": 0.01,
        },
    )

    bottle_upright_pre_lift = RewTerm(
        func=mdp.bottle_upright_reward,
        weight=0.0,
        params={"command_name": "stage", "stage_id": [
            0], "scale": 4.0, "gate_until_lift": 0.01},
    )

    upright_orientation = RewTerm(
        func=mdp.upright_orientation_reward,
        weight=5.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "std": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )

    heading_forward = RewTerm(
        func=mdp.heading_forward_reward,
        weight=10.0,
        params={"command_name": "stage", "stage_id": [
            0], "std": 0.3, "target_yaw": 0.0},
    )

    root_y_below_zero_penalty = RewTerm(
        func=mdp.root_y_below_zero_penalty,
        weight=-10.0,
        params={"command_name": "stage", "threshold": 0.0, "max_penalty": 0.5},
    )

    success_bonus = RewTerm(
        func=mdp.termination_penalty,
        weight=500.0,
        params={"term_name": "success"},
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-2)
    undesired_table_contacts = RewTerm(
        func=mdp.undesired_table_contacts,
        weight=-20.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "sensor_names": ["source_table_contact"],
            "threshold": 5.0,
        },
    )
    robot_fall_penalty = RewTerm(
        func=mdp.termination_penalty,
        weight=-10.0,
        params={"term_name": "robot_fall"},
    )
    time_out_penalty = RewTerm(
        func=mdp.termination_penalty,
        weight=-20.0,
        params={"term_name": "time_out"},
    )
    bottle_dropped_penalty = RewTerm(
        func=mdp.cube_dropped_phase_penalty,
        weight=-1.0,
        params={
            "command_name": "stage",
            "stage_id": [0],
            "cube_name": "bottle",
            "drop_height": 0.05,
            "min_lift_before_large_penalty": 0.02,
            "pre_lift_scale": 0.0,
            "post_lift_scale": 5.0,
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
    passed_without_grasp_penalty = RewTerm(
        func=mdp.termination_penalty,
        weight=-20.0,
        params={"term_name": "robot_passed_bottle_without_grasp"},
    )


@configclass
class EventCfg:
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
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
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.stage_success, params={
                       "command_name": "stage"})
    bottle_dropped = DoneTerm(
        func=mdp.cube_dropped,
        params={
            "command_name": "stage",
            "drop_height": 0.05,
            "min_stage": 0,
            "min_lift_before_drop": 0.02,
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
    robot_fall = DoneTerm(
        func=mdp.robot_fall,
        params={"asset_cfg": SceneEntityCfg(
            "robot"), "body_name": "torso_link", "min_height": 0.35},
    )
    robot_y_deviation = DoneTerm(
        func=mdp.root_y_deviation,
        params={"asset_name": "robot", "max_abs_y": 0.5},
    )
    robot_passed_bottle_without_grasp = DoneTerm(
        func=mdp.robot_passed_object_without_grasp,
        params={"command_name": "stage",
                "cube_name": "bottle", "x_margin": 0.5},
    )
    robot_root_state_blowup = DoneTerm(
        func=mdp.root_state_blowup,
        params={"asset_name": "robot", "max_xy_radius": 6.0,
                "max_height": 3.0, "max_lin_vel": 10.0, "max_ang_vel": 20.0},
    )


@configclass
class WalkgrabEnvCfg(ManagerBasedRLEnvCfg):
    scene: WalkgrabSceneCfg = WalkgrabSceneCfg(num_envs=4096, env_spacing=5.0)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):

        self.decimation = 4
        self.episode_length_s = 15.0
        self.sim.dt = 1.0 / (60.0 * 4)
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
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
            REPO_ROOT / "ckpts/body_prior/walkgrab_10k.pt")
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
