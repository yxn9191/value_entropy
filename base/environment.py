import sys
import os
current_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(current_path)

import mesa
from mesa import DataCollector

from base.agent import Agent
from base.resource import Resource
from base.geoagent import GeoAgent
from base.georesource import GeoResource
from utils.env_reward import *


class BaseEnvironment(mesa.Model):
    """
    """
    name = ""

    def __init__(self,
                 episode_length=None,
                 schedule=None,
                 grid=None,
                 ratio_low=0,
                 ratio_medium=0
                 ):
        super().__init__()

        self.schedule = schedule
        self.grid = grid
        self.episode_length = episode_length  # 一次演化的时长

        # self.timestep = 0  # 环境当前处于的时间点=》mesa有默认的，我先去掉了
        # 在model中，用self.schedule.steps可以获取当前时间步

        self.ratio_low = ratio_low  # 低智能agent的比例
        self.ratio_medium = ratio_medium  # 中智能agent的比例
        self.ratio_high = 1 - self.ratio_low - self.ratio_medium  # 高智能agent的比例
        assert self.ratio_low + self.ratio_medium <= 1
        # 环境中每个智能体的当前奖赏值
        self.curr_optimization_metric = dict()

        # 环境中所有的可运动agent
        self.all_agents = []
        # 环境中所有Resource
        self.all_resources = []
        self._agent_lookup = dict()
        self._resource_lookup = dict()

        self.actions = None
        # 数据收集器
        self.datacollector = DataCollector()
        self.set_all_agents_list()

    # 重置整个环境
    def reset(self):
        self.set_all_agents_list()
        for agent in self.all_agents:
            agent.reset()
        obs = self.generate_observations()

        return obs

    # 生成观察值（强化学习的输入，待补充）
    def generate_observations(self):
        pass

    # 生成即时奖赏值（强化学习的输入，待补充）
    def generate_rewards(self):
        pass

    def action_parse(self, action_dict):
        self.actions = action_dict

    def set_all_agents_list(self):
        if self.schedule:
            for agent in self.schedule.agents:
                if isinstance(agent, Resource) or isinstance(agent, GeoResource):
                    self.all_resources.append(agent)
                elif isinstance(agent, Agent) or isinstance(agent, GeoAgent):
                    self.all_agents.append(agent)
                else:
                    pass

            self._agent_lookup = {str(agent.unique_id): agent for agent in self.all_agents}
            self._resource_lookup = {str(order.unique_id): order for order in self.all_resources}

    def scenario_metrics(self):
        metrics = dict()
        energy = np.array([agent.energy for agent in self._agent_lookup.values()])
        metrics["social/productivity"] = get_productivity(energy)
        metrics["social/equality"] = get_equality(energy)

        metrics[
            "social_welfare/eq_times_productivity"
        ] = metrics["social/productivity"] * metrics["social/equality"]

        return metrics




    def step(self):
        pass
