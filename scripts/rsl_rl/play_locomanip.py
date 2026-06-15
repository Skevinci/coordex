from __future__ import annotations

import argparse
import os
import pathlib
import sys
from math import prod

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "coordex"
for path in (REPO_ROOT, SOURCE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from isaaclab.app import AppLauncher


DEFAULTS = {
    "CoorDex-WalkGrab-Wuji-v0": {
        "checkpoint": "ckpts/locomanip/walkgrab_8k.pt",
        "body_prior": "ckpts/body_prior/walkgrab_10k.pt",
        "hand_prior": "ckpts/hand_prior/kinematic_wrist_16k.pt",
    },
    "CoorDex-Fridge-Wuji-v0": {
        "checkpoint": "ckpts/locomanip/fridge_10k.pt",
        "body_prior": "ckpts/body_prior/fridge_2k.pt",
        "hand_prior": "ckpts/hand_prior/kinematic_wrist_16k.pt",
    },
    "CoorDex-WalkPickTurn-Wuji-v0": {
        "checkpoint": "ckpts/locomanip/walkpickturn_30k.pt",
        "body_prior": "ckpts/body_prior/walkpickturn_30k.pt",
        "hand_prior": "ckpts/hand_prior/active_wrist_20k.pt",
    },
}


parser = argparse.ArgumentParser(
    description="Roll out a CoorDex locomanip checkpoint.")
parser.add_argument(
    "--task", default="CoorDex-WalkGrab-Wuji-v0", choices=tuple(DEFAULTS))
parser.add_argument("--checkpoint", default=None,
                    help="Policy checkpoint. Defaults are selected from --task.")
parser.add_argument("--body-prior-checkpoint", default=None,
                    help="Frozen body prior checkpoint override.")
parser.add_argument("--hand-prior-checkpoint", default=None,
                    help="Frozen hand prior checkpoint override.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--max_steps", type=int, default=None,
                    help="Optional rollout step limit.")
parser.add_argument("--video", action="store_true",
                    help="Record a video from the rollout.")
parser.add_argument("--video_length", type=int, default=500,
                    help="Number of sim steps to record.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from coordex.policies import ActorCriticCoordResidual
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

import coordex.tasks.locomanip  # noqa: F401


def _resolve_path(path: str | os.PathLike[str]) -> str:
    candidate = pathlib.Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return str(candidate)


def _term_dim_to_int(dim) -> int:
    if isinstance(dim, int):
        return int(dim)
    if isinstance(dim, tuple):
        return int(prod(dim))
    return int(prod(int(value) for value in dim))


def _inject_coord_residual_actor_metadata(env, agent_cfg) -> None:
    policy_cfg = getattr(agent_cfg, "policy", None)
    if getattr(policy_cfg, "class_name", "") != "ActorCriticCoordResidual":
        return
    obs_mgr = getattr(env.unwrapped, "observation_manager", None)
    if obs_mgr is None:
        raise RuntimeError(
            "CoordResidual policy requires a ManagerBased env observation manager.")
    policy_cfg.actor_obs_term_names = tuple(
        obs_mgr._group_obs_term_names["policy"])
    policy_cfg.actor_obs_term_dims = [_term_dim_to_int(
        dim) for dim in obs_mgr._group_obs_term_dim["policy"]]


def _register_policy_classes() -> None:
    from rsl_rl import modules as rsl_modules
    from rsl_rl.runners import on_policy_runner as runner_module

    runner_module.ActorCriticCoordResidual = ActorCriticCoordResidual
    rsl_modules.ActorCriticCoordResidual = ActorCriticCoordResidual


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    defaults = DEFAULTS[args_cli.task]
    checkpoint = _resolve_path(args_cli.checkpoint or defaults["checkpoint"])
    body_prior = _resolve_path(
        args_cli.body_prior_checkpoint or defaults["body_prior"])
    hand_prior = _resolve_path(
        args_cli.hand_prior_checkpoint or defaults["hand_prior"])

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    if hasattr(args_cli, "device") and args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    joint_action_cfg = getattr(
        getattr(env_cfg, "actions", None), "joint_pos", None)
    if joint_action_cfg is None:
        raise RuntimeError("Locomanip env is missing actions.joint_pos.")
    joint_action_cfg.body_prior_checkpoint = body_prior
    joint_action_cfg.hand_prior_checkpoint = hand_prior

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Policy checkpoint: {checkpoint}")
    print(f"[INFO] Body prior: {body_prior}")
    print(f"[INFO] Hand prior: {hand_prior}")

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)
    _inject_coord_residual_actor_metadata(env, agent_cfg)

    if args_cli.video:
        video_kwargs = {
            "video_folder": str(REPO_ROOT / "videos" / args_cli.task),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)
    _register_policy_classes()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(),
                            log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    rollout_steps = args_cli.max_steps
    if args_cli.video and rollout_steps is None:
        rollout_steps = args_cli.video_length

    obs, _ = env.get_observations()
    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        step += 1
        if rollout_steps is not None and step >= rollout_steps:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
