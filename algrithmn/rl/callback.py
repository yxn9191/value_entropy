from typing import Dict, Optional, Sequence

import pandas as pd
from ray.rllib.env import BaseEnv
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy, PolicyID

from env_wrapper import RLlibEnvWrapper


class MyCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        envs: Sequence[RLlibEnvWrapper] = base_env.get_unwrapped()


