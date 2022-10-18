import warnings

import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from base.environment import BaseEnvironment

_BIG_NUMBER = 1e20

default_env_config = {
    "width": 200,
    "height": 200,
    "episode_length": 100
}


def recursive_list_to_np_array(d):
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                new_d[k] = np.array(v)
            elif isinstance(v, dict):
                new_d[k] = recursive_list_to_np_array(v)
            elif isinstance(v, (float, int, np.floating, np.integer)):
                new_d[k] = np.array([v])
            elif isinstance(v, np.ndarray):
                new_d[k] = v
            else:
                raise AssertionError
        return new_d
    raise AssertionError


# 为了适应RLlib库进行学习，设计的环境容器
class RLlibEnvWrapper(MultiAgentEnv):
    """
    """

    def __init__(self, env_config=default_env_config, mesaEnv=BaseEnvironment):
        super(RLlibEnvWrapper, self).__init__
        self.env_config = env_config
        self.env = mesaEnv(**self.env_config)

        obs = self.env.reset()

        # 定义动作空间, 定义多少动作该智能体可以选择,即多少订单可以选
        if self.env.all_agents[0].multi_action_mode:
            self.action_space = spaces.MultiDiscrete(
                self.env.self.env.all_agents[0].action_spaces
            )
            self.action_space.dtype = np.int64
            self.action_space.nvec = self.action_space.nvec.astype(np.int64)

        else:
            self.action_space = spaces.Discrete(
                self.env.all_agents[0].action_spaces
            )
            self.action_space.dtype = np.int64

        # 定义观察空间, 定义智能体观察的范围，具体形式为键值对， {name: value}
        sample_agent = list(obs.keys())[0]
        self.observation_space = self._dict_to_spaces_dict(obs[sample_agent])
        self._agent_ids = self.env._agent_lookup.keys()

    def get_seed(self):
        return int(self._seed)

    def _dict_to_spaces_dict(self, obs):
        dict_of_spaces = {}

        for k, v in obs.items():
            _v = v
            if isinstance(v, list):
                _v = np.array(v)
            elif isinstance(v, tuple):
                _v = np.array(v)
            elif isinstance(v, (int, float, np.floating, np.integer)):
                _v = np.array([v])

            if isinstance(_v, np.ndarray):
                x = float(_BIG_NUMBER)
                if np.max(_v) > x:
                    warnings.warn("Input is too large!")
                if np.min(_v) < -x:
                    warnings.warn("Input is too small!")
                box = spaces.Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                while not low_high_valid:
                    x = x // 2
                    box = spaces.Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                    low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                dict_of_spaces[k] = box

            elif isinstance(_v, dict):
                dict_of_spaces[k] = self._dict_to_spaces_dict(_v)
            else:
                raise TypeError
        return spaces.Dict(dict_of_spaces)

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return recursive_list_to_np_array(obs)

    def step(self, action_dict):
        # 只优化寻找订单步骤
        self.env.action_parse(action_dict)
        obs, rew, done, info = self.env.step()
        return recursive_list_to_np_array(obs), rew, done, info
