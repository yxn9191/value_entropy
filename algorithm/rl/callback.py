from typing import Dict, Sequence

from algorithm.rl.env_warpper import RLlibEnvWrapper

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

import pandas as pd


class MyCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        envs: Sequence[RLlibEnvWrapper] = base_env.get_sub_environments()
        social_metrics = pd.DataFrame([
            e.env.scenario_metrics()
            for e in envs
        ]).mean().to_dict()

        for k, v in social_metrics.items():
            episode.custom_metrics[k] = v
