import math
import os
from copy import deepcopy,copy

import mesa
from mesa_geo import GeoSpace, AgentCreator
import numpy as np
import pandas as pd

from analysis.utils.write_to_csv import write_csv_hearders, write_csv_rows, write_csv_rows_cover
from base.environment import BaseEnvironment
from base.geoagent import GeoAgent
from base.georesource import GeoResource
from base.region import Region
from example.cloudManufacturing.orderAgent import OrderAgent
from example.cloudManufacturing.serviceAgent import ServiceAgent
from ray.tune.registry import register_env

from algorithm.rl.env_warpper import RLlibEnvWrapper, recursive_list_to_np_array
from example.cloudManufacturing.generateOrders import *
from shapely.geometry import Point
import sys

current_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(current_path)


def social_rewards(model):
    return model.total_rewards / model.schedule.get_type_count(ServiceAgent)


class CloudManufacturing(BaseEnvironment):
    name = "CloudManufacturing"
    MAP_COORDS = [117.222503, 39.117489]

    def __init__(self, num_order=10, num_service=5, num_organization=2, episode_length=200,
                 ratio_low=0, ratio_medium=0, tax_rate=0, is_training=True, trainer=None, reset_random=True):
        super().__init__()
        # 初始时的order和agent
        # print("__________", num_order, num_service)

        self.order_num = num_order
        self.service_num = num_service
        self.num_a, self.num_b, self.num_c = 0, 0, 0
        self.num_organization = num_organization  # 组织的数目
        self.episode_length = episode_length  # 一次演化的时长
        self.new_orders = []  # 当前时刻产生的订单的数目
        self.new_services = []  # 当前时刻产生的新企业
        self.finish_orders = 0  # 当前预期可以完成的order的数目（step中可以算到）
        self.actions = {}
        self.total_rewards = 0
        self.reset_random = reset_random #重置环境类型 True则为随机新生成，False 则使用固定随机树种子生成

        # 算法1中的M（订单）和N（企业）
        self.M = 20
        self.N = 10

        self.schedule = mesa.time.RandomActivationByType(self)
        # self.grid = mesa.space.MultiGrid(width, height, True)  # True一个关于网格是否为环形的布尔值

        self.grid = GeoSpace()

        ac = AgentCreator(Region, {"model": self})
        geo_path = os.path.join(current_path, "data/120116.geoJson")
        self.region = ac.from_file(
            geo_path, unique_id="name"
        )
        self.grid.add_agents(self.region)
        self.all_orders_list = all_orders_list(self.region[0], self.reset_random)

        self.ratio_low = ratio_low
        self.ratio_medium = ratio_medium
        self.ratio_high = 1 - self.ratio_low - self.ratio_medium
        self.match_agent = []
        self.match_order = []
        self.tax_rate = tax_rate
        self.pos_matrix = None

        self.init_services(self.service_num)
        self.generate_orders()
        self.set_intelligence(self.new_services)
        self.is_training = is_training


        # # 环境中所有的可运动agent
        # self.init_all_agents = deepcopy(self.new_services)
        # # 环境中所有Resource
        # self.init_all_resources = deepcopy(self.new_orders)


        # self.set_all_agents_list()  # 注意！！有了这个函数，self.all_agents和look_up系列会直接同schedule变动，
        # 而不需要额外操作向其中手动添加agent了，此函数每次在环境的step开始时就调用，同步系统中所有存储agent的list

        if self.is_training == False :
            self.obs = self.reset()
            self.trainer = trainer
            self.obs = {
                k: {
                    k1: v1 if type(v1) is np.ndarray else np.array([v1])
                    for k1, v1 in v.items()
                }
                for k, v in self.obs.items()
            }

    # 修改agent的智能等级
    # 设定为同一智能体的智能等级，一经初始化设定后无法再修改，防止出现在执行任务时突然变更，影响系统效果。
    # 同时我们无需保证系统内智能体的比例不变，而是在初始设置后实行优胜略汰的机制:如果智能体因为智商低死去了，那就死去了不管他。
    def set_intelligence(self, new_agents):
        low_agents = []
        medium_agents = []
        high_agents = []
        temp_agents = set(new_agents)
        low_agents.extend(
            [temp_agents.pop() for _ in range(int(len(new_agents) * self.ratio_low - len(low_agents)))])
        medium_agents.extend(
            [temp_agents.pop() for _ in range(int(len(new_agents) * self.ratio_medium - len(medium_agents)))])
        high_agents.extend(list(temp_agents))
        for agent in low_agents:
            agent.set_intelligence(0)
        for agent in medium_agents:
            agent.set_intelligence(1)
        for agent in high_agents:
            agent.set_intelligence(2)

    # def random_place_agent(self, agent):
    #     x = self.random.randrange(self.grid.width)
    #     y = self.random.randrange(self.grid.height)
    #     self.grid.place_agent(agent, (x, y))
    # a.location = (x, y) 这行不需要，place_agent就自动将该属性添加到agent中，属性值为pos

    def init_services(self, new_service_num):
        self.new_services = []
        # 默认每轮新增5企业
        if not self.reset_random:
            random.seed(0)
        energy = [random.uniform(1e3, 5e3) for i in range(new_service_num)]
        consumption = [random.uniform(1, 10) for i in range(new_service_num)]

        for j in range(new_service_num):
            shape = self.region[0].random_point
            ac_population = AgentCreator(
                ServiceAgent,
                {"model": self, "service_type": generate_service_type(), "difficulty": generate_difficulty(),
                "energy": energy[j], "consumption": consumption[j]}
            )

            this_person = ac_population.create_agent(
                shape, self.next_id()
            )
            this_person.pos = (shape.x, shape.y)
        
            self.schedule.add(this_person)
            self.grid.add_agents(this_person)
            self.new_services.append(this_person)

    # 企业繁衍
    def generate_services(self):
        self.new_services = []

        for agent in self._agent_lookup.values():
            if agent.energy >= 5e3:
                print("企业繁衍",agent.unique_id)
                while 1:
                    random_point = Point(agent.shape.x + random.uniform(-10, 10),
                                         agent.shape.y + random.uniform(-10, 10))
                    if self.region[0].shape.contains(random_point):
                        break
                ac_population = AgentCreator(
                    ServiceAgent,
                    {"model": self, "service_type": agent.service_type, "difficulty": agent.difficulty,
                     "intelligence_level": agent.intelligence_level, "energy": agent.energy / 2}
                )

                agent.energy = agent.energy / 2

                this_person = ac_population.create_agent(
                    random_point, self.next_id()
                )
                this_person.pos = (random_point.x, random_point.y)

                self.schedule.add(this_person)
                self.grid.add_agents(this_person)
                self.new_services.append(this_person)

    def generate_orders(self):
        self.new_orders = []
        try:
            self.all_orders_list[self.schedule.steps]
        except IndexError:
            raise IndexError(self.schedule.steps)
        for ord in self.all_orders_list[self.schedule.steps]:
            shape = Point(ord[-1][0], ord[-1][1])
            a = OrderAgent(self.next_id(), self, shape, ord[3], ord[0], bonus=ord[1], cost=ord[2])
            a.pos = ord[-1]
            self.schedule.add(a)
            self.grid.add_agents(a)
            # self.random_place_agent(a)
            self.new_orders.append(a)

                # self.init_parm[str(a.unique_id)] = {"energy": a.bonus, "pos": a.pos }
            # self.all_resources.append(a)
            # self._resource_lookup[a.unique_id] = a

    # 比较两个agent是否彼此符合约束条件,即判断该service是否是order的潜在工人（充分条件）
    # 返回1代表是潜在工人
    def sufficient_constraint(self, order, service):
        # 空间约束
        # 时间约束
        # 预算约束
        # 技能约束
        if service.service_type not in list(order.order_type):
            return 0
        # print(distance(order.pos, service.pos) <= order.vision,
        # distance(order.pos, service.pos) / service.speed , (order.left_duration - order.handling_time),
        # distance(order.pos, service.pos) * service.move_cost , (order.bonus - order.cost),
        # skill_constraint(order, service).count(1))
        # 如果订单或企业二者有一个是禁止合作的，则技能约束部分必须满足所有技能条件
        if order.cooperation == 0 | service.cooperation == 0:
            if distance(order.pos, service.pos) <= order.vision and \
                    distance(order.pos, service.pos) / service.speed <= (order.left_duration - order.handling_time) and \
                    distance(order.pos, service.pos) * service.move_cost <= (order.bonus - order.cost) and \
                    skill_constraint(order, service).count(1) == 2:
                return 1
            else:
                return 0
        else:
            # 在其他情况下，可以合作，则技能约束部分只需满足一个及以上的技能条件即可
            if distance(order.pos, service.pos) <= order.vision and \
                    distance(order.pos, service.pos) / service.speed <= (order.left_duration - order.handling_time) and \
                    distance(order.pos, service.pos) * service.move_cost <= (order.bonus - order.cost) and \
                    skill_constraint(order, service).count(1) > 0:

                return 1
            else:
                return 0

    # 判断潜在工人组是否最终可获得该订单（必要条件）,可获得返回1，否则返回0
    def necessary_constraint(self, order, services):
        total_move_cost = 0
        total_skill = [0, 0]

        for service in services:
            service = self._agent_lookup[str(service)]
            # 合作组织中出现单个企业的收益小于0，则该合作无法成立
            if distance(order.pos, service.pos) * service.move_cost + order.cost / len(services) > order.bonus / len(
                    services):
                return 0
            total_move_cost += distance(order.pos, service.pos) * service.move_cost
            total_skill = [int(a or b) for (a, b) in zip(skill_constraint(order, service), total_skill)]
        # 整体预算约束：小组成员的行程代价之和加上处理订单的消耗小于订单给予的价值（等于也不行，那不是白干了）&& 整体技能须满足该订单所有的技能要求
        if order.bonus >= total_move_cost + order.cost and sum(total_skill) == 2:
            return 1
        else:
            return 0

    def generate_action_mask(self):
        masks = {str(agent.unique_id): [0 for _ in range(self.M)] for agent in self.match_agent if
                 agent.intelligence_level == 2}
        position = 0
        for order in self.match_order:
            for agent in self.match_agent:
                if agent.intelligence_level == 2:
                    masks[str(agent.unique_id)][position] = self.sufficient_constraint(order, agent) \
                                                            * int(agent.service_type in order.order_type)
            position += 1

        return masks

    # 低、中智能情况，生成每个agent可能选择的order,存入agent的selected_order_id
    # 同一agent在每一step中，会同时存有低、中、高智能的动作，但是根据不同情况选择执行
    # def get_actions(self):
    #     temp_actions = {str(agent.unique_id): [] for agent in self.match_agent}
    #     for order in self.match_order:
    #         for agent in self.match_agent:
    #             if self.sufficient_constraint(order, agent) == 1:
    #                 temp_actions[str(agent.unique_id)].append(order.unique_id)
    #                 agent.temp_actions = temp_actions[str(agent.unique_id)]
    def get_actions(self):
        temp_actions = {str(agent.unique_id): [] for agent in self.match_agent}
        for order in self.match_order:
            for agent in self.match_agent:
                if self.sufficient_constraint(order, agent) == 1:
                    temp_actions[str(agent.unique_id)].append(order.unique_id)
                    agent.temp_actions = temp_actions[str(agent.unique_id)]

        for agent in self.match_agent:
            if agent.intelligence_level == 0:
                # 随机选择一个满足充分约束的订单
                agent.selected_order_id = random.choice(agent.temp_actions)
                agent.order = agent.selected_order_id
                self.actions.update(
                    {str(agent.unique_id): self.match_order.index(self._resource_lookup[str(agent.selected_order_id)])})
            if agent.intelligence_level == 1:
                # print("medium")
                # print("temp_actions:", self.temp_actions)
                # print("_resource_lookup", self.model._resource_lookup)
                order_reward = {str(order_id): [] for order_id in agent.temp_actions}
                for order_id in agent.temp_actions:
                    order = self._resource_lookup[str(order_id)]
                    reward = order.bonus - sum(
                        [abs(a - b) for (a, b) in zip(order.pos, agent.pos)]) * agent.move_cost - order.cost
                    order_reward[str(order.unique_id)].append(reward)
                # 只选择自己计算出的代价最小的order，不考虑合作分配和社会整体
                agent.selected_order_id = sorted(order_reward.items(), key=lambda o: o[1])[0][0]
                agent.order = agent.selected_order_id
                self.actions.update(
                    {str(agent.unique_id): self.match_order.index(self._resource_lookup[str(agent.selected_order_id)])})
                # order = None
                # for temp in self.model.all_resources:
                #     if str(temp.unique_id) == self.selected_order_id:
                #         order = temp

    # 计算局部观察值,待修改
    def compute_order(self, agent):
        orders = []

        # 这里的self.all_orders要替换从匹配1中返回的orders
        for neighborhood in self.match_order:
            orders.extend([neighborhood.cooperation, neighborhood.cost,
                           neighborhood.bonus,
                           distance(neighborhood.pos, agent.pos) * agent.move_cost,
                           self.sufficient_constraint(neighborhood, agent)
                           ])

            orders.extend(skill_constraint(neighborhood, agent))
        # 生成虚拟订单

        num_orders = len(self.match_order)
        while num_orders < self.M:
            orders.extend([0, 0, 0, 0, 0])
            orders.extend([0 for i in range(len(self.all_resources[0].skills))])
            num_orders += 1
        return orders

    # 获得其他agent的观察
    def get_other_agent_obs(self, obs):
        _obs = dict()
        for k in obs.keys():
            _obs[k] = [obs[k]["cooperation"]]
            for key in obs[k].keys():
                if key != "cooperation":
                    _obs[k] += obs[k][key]

        num_agent = len(obs)
        while num_agent < self.N:
            _obs[str(-num_agent)] = [0 for _ in range(7 * self.M + 1)]
            num_agent += 1
        return _obs

    def reset(self):
        #重新生成企业
        self.schedule.steps = 0
        if self.reset_random :
            for agent in self.schedule.agents:
                if isinstance(agent, OrderAgent) or isinstance(agent, ServiceAgent):
                    self.schedule.remove(agent)
                    self.grid.remove_agent(agent)
            self.generate_orders()
            self.init_services(self.service_num)
        self.set_all_agents_list()
        self.match_order, self.match_agent = self.matching_service_order()
        self.obs = self.generate_observations()
        return self.obs

            

    # 生成观察值（强化学习的输入,待修改）
    def generate_observations(self):
        obs = {}
        # self.match_order, self.match_agent = self.matching_service_order()
        # 影响选择订单规则的内在属性,#这里的 self._agent_lookup要替换从匹配1中返回的agent
        for agent in self.match_agent:
            if agent.intelligence_level == 2:
                obs[str(agent.unique_id)] = {"cooperation": agent.cooperation}
                obs[str(agent.unique_id)].update({"orders": self.compute_order(agent)})

        _obs = self.get_other_agent_obs(obs)

        for agent in self.match_agent:
            if agent.intelligence_level == 2:
                obs_ = deepcopy(_obs)
                del obs_[str(agent.unique_id)]
                obs[str(agent.unique_id)].update({"others": list(obs_.values())})

        # 生成虚拟企业
        num_agents = len(obs)
        while num_agents < self.N:
            obs[str(-num_agents)] = {"cooperation": 0}
            # other = {str(-i): [0, 0, 0, 0, 0, 0] for i in range(1000, 1200)}
            orders = {"orders": [0 for i in range(7 * self.M)]}
            action_mask = {"action_mask": [0 for i in range(self.M)]}
            obs[str(-num_agents)].update(orders)
            obs[str(-num_agents)].update(
                {"others": [np.array([0 for i in range(self.M * 7 + 1)]) for _ in range(self.N - 1)]})
            obs[str(-num_agents)].update(action_mask)
            num_agents += 1

        # Get each agent's action masks and incorporate them into the observations
        for aidx, amask in self.generate_action_mask().items():
            obs[str(aidx)]["action_mask"] = amask

        return obs

    def compute_agent_reward(self, cost, value, alpha=0.1):
        if len(self.new_orders) != 0:
            bonus = 0
            for order in self.new_orders:
                bonus += order.bonus
            rew = alpha * self.finish_orders / len(self.new_orders) + (1 - alpha) * value / cost
        else:
            rew = (1 - alpha) * value / cost
        return rew

    # 算法1：企业和订单的匹配算法
    def matching_service_order(self):
        k = 0
        # 参与本轮匹配的订单集合T和企业集合W
        T = set()
        W = set()
        for order in self._resource_lookup.values():
            if order.occupied == 0:

                for agent in self._agent_lookup.values():
                    if agent.state != 0:
                        continue
                    G = set()
                    if self.sufficient_constraint(order, agent):
                        G.add(agent)
                        if k + 1 <= self.M and len(W.union(G)) <= self.N:
                            k += 1
                            W.update(G)
                            T.add(order)
                        else:
                            break
        return list(T), list(W)

    # 平台反选
    def order_select(self):
        # print("start order select")
        self.finish_orders = 0
        # 当前没有进行匹配的序列
        if len(self.match_order) == 0 or len(self.match_agent) == 0:
            return 0
        order_action = dict()
        # 排除没有匹配的agent的影响
        for agent in self._agent_lookup.values():
            if agent not in self.match_agent:
                agent.action_parse(-1)
                # print(agent.action, self._agent_lookup.get(str(agent.unique_id), None).action)
        for a_id, o_id in self.actions.items():

            if int(a_id) <= 0:
                continue
            if int(o_id) >= len(self.match_order):
                agent = self._agent_lookup.get(str(a_id), None)
                agent.action_parse(-1)
                # print(agent.action, self._agent_lookup.get(str(a_id), None).action)
                continue
            self._agent_lookup.get(str(a_id), None).order_select=1
            self.match_order[o_id].order_select=1
            agent = self._agent_lookup.get(str(a_id), None)
            order = self.match_order[o_id]
            if agent.service_type not in list(order.order_type):
                agent.action_parse(-1)
                continue
            if order.unique_id not in order_action:
                order_action[order.unique_id] = dict()
                order_action[order.unique_id][agent.service_type] = {a_id: distance(agent.pos, order.pos)}
            elif agent.service_type not in order_action[order.unique_id]:
                order_action[order.unique_id][agent.service_type] = {a_id: distance(agent.pos, order.pos)}
            else:
                order_action[order.unique_id][agent.service_type].update({a_id: distance(agent.pos, order.pos)})

        for o_id in order_action.keys():
            order_ = self._resource_lookup.get(str(o_id), None)
            order_.services = []
            order_action_ = deepcopy(order_action)
            if len(order_.order_type) == 1:
                for service in order_action_[o_id][order_.order_type].keys():
                    if not self.necessary_constraint(order_, [service]):
                        del order_action[o_id][order_.order_type][service]
                        agent = self._agent_lookup.get(str(service), None)
                        self._agent_lookup.get(str(service), None).action_parse(-1)
                        # agent.action_parse(-1)
                if len(order_action[o_id][order_.order_type]) > 0:
                    self._resource_lookup.get(str(o_id), None).services.extend([min(order_action[o_id][order_.order_type].items(), key=lambda x: x[1])[0]])
                    self._agent_lookup.get(str(order_.services[0]), None).order = order_.unique_id
                    self._resource_lookup.get(str(o_id), None).occupied = 1
                    self.finish_orders += 1
                    self._resource_lookup.get(str(o_id), None).occupied = 1
                    for a_id in order_action[o_id][order_.order_type].keys():
                        if a_id not in order_.services:
                            self._agent_lookup.get(str(a_id), None).action_parse(-1)
                        else:
                            self._agent_lookup.get(str(a_id), None).order = order_.unique_id


            else:
                if len(order_action[o_id]) < len(order_.order_type):
                    for service_type in order_action[o_id].keys():
                        for a_id in order_action[o_id][service_type].keys():
                            self._agent_lookup.get(str(a_id), None).action_parse(-1)
                    continue
                list_service = {}
                types = list(order_.order_type)
                if len(order_.order_type) == 2:  
                    for agent1 in order_action[o_id][types[0]].keys():
                        for agent2 in order_action[o_id][types[1]].keys():
                            if self.necessary_constraint(order_,[agent1, agent2]):
                                list_service[str(agent1.unique_id)+"+"+str(agent2.unique_id)] = \
                                order_action[o_id][service_type][agent1.unique_id] + order_action[o_id][service_type][agent2.unique_id] 
                
                if len(order_.order_type) == 3:
                    for agent1 in order_action[o_id]["A"].keys():
                        for agent2 in order_action[o_id]["B"].keys():
                            for agent3 in order_action[o_id]["C"].keys():
                                if self.necessary_constraint(order_,[agent1, agent2, agent3]):
                                    list_service[str(agent1.unique_id)+"+"+str(agent2.unique_id)+"+"+str(agent3.unique_id)]= \
                                    order_action[o_id][service_type][agent1.unique_id] + \
                                    order_action[o_id][service_type][agent2.unique_id] + \
                                    order_action[o_id][service_type][agent3.unique_id] 

                if len(list_service) > 0:
                    x = min(list_service.items(), key=lambda x: x[1])[0]
                    order_.services.extend(x.split("+"))
                    self._agent_lookup.get(str(order_.services[0]), None).order = order_.unique_id
                    self.finish_orders += 1
                    self._resource_lookup.get(str(o_id), None).occupied = 1
                    self._resource_lookup.get(str(o_id), None).services.extend(x.split("+"))
                    for service_type in order_action[o_id].keys():
                        for a_id in order_action[o_id][service_type].keys():
                            if str(a_id) not in order_.services:
                                self._agent_lookup.get(str(a_id), None).action_parse(-1)
                            else:
                                self._agent_lookup.get(str(a_id), None).order = order_.unique_id
                else:
                    for service_type in order_action[o_id].keys():
                        for a_id in order_action[o_id][service_type].keys():
                            self._agent_lookup.get(str(a_id), None).action_parse(-1)


        # if len(order_action)> 0:
        #     raise TypeError(order_action)

    # 纳税然后进行重分配
    def pay_taxex(self):
        total_tax = 0
        num_agent = 0
        for agent in self._agent_lookup.values():
            total_tax += agent.energy * self.tax_rate
            agent.energy -= agent.energy * self.tax_rate
            num_agent += 1
        for agent in self._agent_lookup.values():
            agent.energy += total_tax / num_agent

    #
    # def init_rl(self, trainer):
    #
    #     trainer = build_Trainer(run_configuration)
    #     trainer.restore(str(ckpt))
    #     starting_weights_path_agents = run_configuration["general"].get(
    #         "restore_torch_weights_agents", ""
    #     )
    #
    #     load_torch_model_weights(trainer, starting_weights_path_agents)
    #
    #     obs = self.reset()
    #
    #     self.obs = obs
    #     return self.trainer, self.obs

    # 将每轮ABC企业的数目写入csv
    def collect_agent_num(self):
        if self.schedule.steps == 1:
            write_csv_hearders("data/agent_num.csv", ["time_step", "nums", "service_type"])
        for agent in self.schedule.agents:
            if isinstance(agent, GeoAgent):
                if agent.service_type == "A":
                    self.num_a += 1
                elif agent.service_type == "B":
                    self.num_b += 1
                elif agent.service_type == "C":
                    self.num_c += 1
        write_csv_rows("data/agent_num.csv", [[self.schedule.steps, self.num_a, 'A'],
                                              [self.schedule.steps, self.num_b, 'B'],
                                              [self.schedule.steps, self.num_c, 'C']])

    # 将每step的平均效能写入csv
    def collect_avg_reward(self, avg_reward, file_name):
        file_path = os.path.join("data", file_name)
        if self.schedule.steps == 1:
            write_csv_hearders(file_path, ["time_step", "avg_reward"])
        write_csv_rows(file_path, [[self.schedule.steps, avg_reward]])

    # 将每step的平均效能写入csv
    def collect_reward_with_tax(self, avg_reward, title, file_name):
        file_path = os.path.join("data", file_name)
        if self.schedule.steps == 1:
            write_csv_hearders(file_path, ["time_step", title, "tax_rate"])
        write_csv_rows(file_path, [[self.schedule.steps, avg_reward, self.tax_rate]])

    # 将企业的位置写入csv
    # pos:{agentID:agent.pos}
    def collect_agent_pos(self, pos):
        x = []
        y = []
        for v in pos.values():
            x.append(round(v[0]))
            y.append(round(v[1]))
        if self.schedule.steps == 1:

            self.pos_matrix = np.zeros((max(x), max(y)), dtype=int)
            for m, n in zip(x, y):
                # print(self.pos_matrix.shape)
                self.pos_matrix[m - 1][n - 1] += 1
                # print(self.pos_matrix)
        else:
            o_x, o_y = self.pos_matrix.shape
            new_matrix = np.zeros((max(max(x), o_x), max(max(y), o_y)), dtype=int)
            for m, n in zip(x, y):
                new_matrix[m - 1][n - 1] += 1
            # print(new_matrix)
            for i in range(self.pos_matrix.shape[0]):
                for j in range(self.pos_matrix.shape[1]):
                    new_matrix[i][j] += self.pos_matrix[i][j]
            self.pos_matrix = new_matrix
            # print(self.pos_matrix)
        # 注意写入csv时，转置后才是横着x，纵向y
        write_csv_rows_cover("data/high_reward_heatmap.csv", self.pos_matrix.T.tolist())

    def compute_rl_step(self):
        results = {}
        actions = {}
        for agent in self.match_agent:
            if agent.intelligence_level == 2:
                results[str(agent.unique_id)] = self.trainer.compute_single_action(
                    recursive_list_to_np_array(self.obs[str(agent.unique_id)]),
                    policy_id="a",
                    full_fetch=False)
                actions[str(agent.unique_id)] = int(results[str(agent.unique_id)])
        # print("action", actions)
        self.actions.update(actions)

    def collect_with_rate(self, metrics):
        if self.ratio_low == 1:
            if self.tax_rate <= 0.1:
                filename1 = "low_level_low_rate_prod.csv"
                filename2 = "low_level_low_rate_eq.csv"
                filename3 = "low_level_low_rate_eqprod.csv"
            elif self.tax_rate <= 0.2:
                filename1 = "low_level_mid_rate_prod.csv"
                filename2 = "low_level_mid_rate_eq.csv"
                filename3 = "low_level_mid_rate_eqprod.csv"
            elif self.tax_rate <= 0.3:
                filename1 = "low_level_high_rate_prod.csv"
                filename2 = "low_level_high_rate_eq.csv"
                filename3 = "low_level_high_rate_eqprod.csv"
        elif self.ratio_medium == 1:
            if self.tax_rate <= 0.1:
                filename1 = "mid_level_low_rate_prod.csv"
                filename2 = "mid_level_low_rate_eq.csv"
                filename3 = "mid_level_low_rate_eqprod.csv"
            elif self.tax_rate <= 0.2:
                filename1 = "mid_level_mid_rate_prod.csv"
                filename2 = "mid_level_mid_rate_eq.csv"
                filename3 = "mid_level_mid_rate_eqprod.csv"
            elif self.tax_rate <= 0.3:
                filename1 = "mid_level_high_rate_prod.csv"
                filename2 = "mid_level_high_rate_eq.csv"
                filename3 = "mid_level_high_rate_eqprod.csv"
        elif self.ratio_high == 1:
            if self.tax_rate <= 0.1:
                filename1 = "high_level_low_rate_prod.csv"
                filename2 = "high_level_low_rate_eq.csv"
                filename3 = "high_level_low_rate_eqprod.csv"
            elif self.tax_rate <= 0.2:
                filename1 = "high_level_mid_rate_prod.csv"
                filename2 = "high_level_mid_rate_eq.csv"
                filename3 = "high_level_mid_rate_eqprod.csv"
            elif self.tax_rate <= 0.3:
                filename1 = "high_level_high_rate_prod.csv"
                filename2 = "high_level_high_rate_eq.csv"
                filename3 = "high_level_high_rate_eqprod.csv"

        self.collect_reward_with_tax(metrics['social/productivity'],  "productivity",filename1)
        self.collect_reward_with_tax(metrics['social/equality'], "equality", filename2)
        self.collect_reward_with_tax(metrics['social_welfare/eq_times_productivity'], "eq_times_productivity", filename3)        

    def step(self):
        """Advance the model by one step."""

        # 首先检查本轮所有死去的订单和企业，从agent队列中移除
        for agent in self.schedule.agents:
            if agent.done:
                self.schedule.remove(agent)
                self.grid.remove_agent(agent)
                if isinstance(agent, OrderAgent):
                    del self._resource_lookup[str(agent.unique_id)]
                elif isinstance(agent, ServiceAgent):
                    del self._agent_lookup[str(agent.unique_id)]

        # 假设纳税期为10个周期
        if self.schedule.steps % 10 == 0:
            self.pay_taxex()

        self.set_all_agents_list()
        self.set_intelligence(self.new_services)

        alpha = 0.1
        reward = dict()

        # 低智能也是先确定匹配的大小，所以我抽出来了
        self.match_order, self.match_agent = self.matching_service_order()

        # 低智能和中智能的可能动作，存入agent
        self.get_actions()

        self.obs = self.generate_observations()
        if self.is_training == False:
            self.compute_rl_step()

        # 存入高智能的动作
        if self.actions is not None:
            # 这里的self._agent_lookup也要换成算法1获得的企业集合
            for agent_idx, agent_actions in self.actions.items():
                if int(agent_idx) > 0:
                    self._agent_lookup.get(str(agent_idx), None).action_parse(agent_actions)

            # 平台反选
            self.order_select()

            self.schedule.step()
            for agent in self.match_agent:
                if agent.intelligence_level == 2:
                    value, cost = agent.process_order()
                    reward[str(agent.unique_id)] = self.compute_agent_reward(cost, value, alpha)

            # 生成虚拟企业
            num_agents = len(reward)

            while num_agents < self.N:
                reward[str(-num_agents)] = 0
                num_agents += 1

        self.total_rewards = 0

        # 演化结束的判断，待修改
        self.done = {"__all__": self.schedule.steps >= self.episode_length}
        info = {k: {} for k in self.obs.keys()}

        if self.is_training == False:
            metrics = self.scenario_metrics()
            self.collect_agent_num()
            # print(reward.values())
            if len(self.match_agent) > 0:
                avg_reward = sum(reward.values()) / len(self.match_agent)
            else:
                avg_reward = 0
            self.collect_avg_reward(metrics["social_welfare/eq_times_productivity"], "avg_reward.csv")
            self.collect_with_rate(metrics)

            agent_pos = {}
            for agent in self.all_agents:
                flag = 0
                if agent.intelligence_level == 2:
                    # print(agent.pos)
                    # 对agent_pos进行标准化
                    flag = 1
                    min_x, min_y, max_x, max_y = self.region[0].shape.bounds
                    x = (agent.pos[0] - min_x) * 0.08
                    y = (agent.pos[1] - min_y) * 0.08
                    agent_pos.update({str(agent.unique_id): (x, y)})

            # 输入的形式类似：{1: (2, 3), 2: (4, 1), 3: (3, 3), 4: (2, 7)} {agentID:agent.pos} 
            # 只有在环境中有高智能的agent才画这个高智能的热力图
            if flag :
                self.collect_agent_pos(agent_pos)

        # 生成本轮新的企业和订单
        self.generate_orders()
        self.generate_services()

        return self.obs, reward, self.done, info


# 计算两位置的直线距离
def distance(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


# 计算移动到新位置的路线长度
def move_len(A, B):
    return sum([abs(a - b) for (a, b) in zip(A, B)])  # 要取绝对值，不然可能出现负距离


# 返回技能向量的满足列表
# 如满足第一个不满足第二个，返回[1,0]
def skill_constraint(order, service):
    i = 0
    order_diff = 0
    j = 0
    service_diff = 0
    # 是否满足技能点的列表
    list = [0, 0]

    # 判断订单的类型是否为企业可以处理的类型
    if all([(b - a) >= 0 for (a, b) in zip(order.skills[0], service.skills[0])]):
        list[0] = 1

    # 判断订单的难度是否为企业可以处理的难度
    # for k in reversed(order.skills[1]):
    #     order_diff = order_diff + k * 2 ^ i
    #     i += 1

    # for k in reversed(service.skills[1]):
    #     service_diff = service_diff + k * 2 ^ j
    #     j += 1

    # if order_diff <= service_diff:
    #     list[1] = 1
    if order.order_difficulty <= service.difficulty:
        list[1] = 1

    return list


def generate_order_type():
    weight = {"A": 0.3, "B": 0.3, "C": 0.3, "AB": 0.03, "AC": 0.03, "BC": 0.03, "ABC": 0.01}
    return random.choices(list(weight.keys()), weights=list(weight.values()), k=1)[0]


def generate_service_type():
    weight = {"A": 0.3, "B": 0.3, "C": 0.4}
    return random.choices(list(weight.keys()), weights=list(weight.values()), k=1)[0]


def generate_difficulty():
    return random.randint(1, 3)


# 注册强化学习环境
def env_creator(env_config):  # 此处的 env_config对应 我们在建立trainer时传入的dict env_config
    return RLlibEnvWrapper(env_config, CloudManufacturing)


register_env(CloudManufacturing.name, env_creator)

# def build_Trainer(run_configuration):
#     trainer_config = run_configuration.get("trainer")
#     env_config = run_configuration.get("env")["env_config"]
#
#     # === Multiagent Policies ===
#     dummy_env = RLlibEnvWrapper(env_config, CloudManufacturing)
#
#     # Policy tuples for agent/planner policy types
#     agent_policy_tuple = (
#         None,
#         dummy_env.observation_space,
#         dummy_env.action_space,
#         run_configuration.get("agent_policy"),
#     )
#
#     policies = {"a": agent_policy_tuple}
#
#     def policy_mapping_fun(i):
#         return "a"
#
#     trainer_config.update({
#         "env_config": env_config,
#         'framework': 'torch',
#         "multiagent": {
#             "policies": policies,
#             "policies_to_train": ["a"],
#             "policy_mapping_fn": policy_mapping_fun,
#         },
#         "num_workers": trainer_config.get("num_workers")
#     })
#
#     trainer = A2CTrainer(env=run_configuration.get("env")["env_name"], config=trainer_config)
#
#     return trainer
#
