from __future__ import annotations

import gymnasium as gym

from coordex.tasks.locomanip.config.g1_wuji import agents


gym.register(
    id="CoorDex-WalkGrab-Wuji-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "coordex.tasks.locomanip.walkgrab_env_cfg:WalkgrabEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomanipCoordResidualPPORunnerCfg",
    },
)

gym.register(
    id="CoorDex-Fridge-Wuji-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "coordex.tasks.locomanip.fridge_env_cfg:FridgeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomanipCoordResidualPPORunnerCfg",
    },
)

gym.register(
    id="CoorDex-WalkPickTurn-Wuji-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "coordex.tasks.locomanip.walkpickturn_env_cfg:WalkPickTurnEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LocomanipCoordResidualPPORunnerCfg",
    },
)
