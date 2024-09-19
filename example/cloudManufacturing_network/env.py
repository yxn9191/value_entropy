from copy import deepcopy
from random import random

import networkx as nx
import mesa
import random

import numpy as np
from ray.tune import register_env

from algorithm.rl.env_warpper import RLlibEnvWrapper, recursive_list_to_np_array
from base.utils.env_reward import get_self_utility, H_t, get_vge, system_utility
from example.cloudManufacturing_network.generateOrders import all_orders_list, get_n
from example.cloudManufacturing_network.generateService import generate_cooperation, generate_service_type, generate_difficulty, \
    generate_energy, generate_imitate_pro
from example.cloudManufacturing_network.orderAgent import OrderAgent
from example.cloudManufacturing_network.serviceAgent import ServiceAgent
from example.cloudManufacturing_network.wirte_csv import agent_utility_type_csv, diff_csv



class CloudManufacturing_network(mesa.Model):
    name = "CloudManufacturing_network"

    def __init__(
            self,
            num_nodes=20,
            avg_node_degree=3,
            ratio_low=0,
            ratio_medium=0,
            ratio_imitate=0,
            is_training=True,
            trainer=None,
            reset_random=True,
            episode_length=200,
            network_type='ER',
            BA_add_edges=2,
            RG_init_degree=3,
            WS_init_neighbors=3
    ):
        super().__init__()
        self.done = None
        # 算法1中的M（订单）和N（企业）
        self.M = 20
        self.N = 10
        # 环境中所有的可运动agent
        self.all_agents = []
        # 环境中所有Resource
        self.all_resources = []
        self._agent_lookup = dict()
        self._resource_lookup = dict()
        self.ratio_low = ratio_low  # 随机选择的agent的比例
        self.ratio_medium = ratio_medium  # 固定规则的agent的比例
        self.ratio_imitate = ratio_imitate  # 模仿学习的agent的比例
        self.ratio_high = 1 - self.ratio_low - self.ratio_medium - self.ratio_imitate  # 高智能agent的比例
        assert self.ratio_low + self.ratio_medium <= 1
        self.num_nodes = num_nodes  # node数目=企业agent数目
        self.avg_node_degree = avg_node_degree
        prob = avg_node_degree / self.num_nodes
        self.network_type = network_type
        # WS:小世界网络 BA：无标度网络 RG:规则网络 ER：随机图
        if network_type == 'ER':  # ER
            # 初始网络状态是由num_nodes个节点组成的随机网络，平均节点度数由参数avg_node_degree调控
            self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=prob)
        elif network_type == 'BA':  # BA
            # 初始网络状态是由num_nodes个节点组成的无标度网络，每次加入的边数由参数type2_add_edges调控
            self.G = nx.barabasi_albert_graph(n=self.num_nodes, m=BA_add_edges)
        elif network_type == 'WS':  # WS
            # 生成一个含有num_nodes个节点、每个节点有k个邻居、以概率prob随机化重连边的WS小世界网络
            self.G = nx.watts_strogatz_graph(n=self.num_nodes, k=WS_init_neighbors, p=prob)
        elif network_type == "None":  # 无连边
            self.G = nx.random_regular_graph(n=self.num_nodes, d=0)
        else:  # RG
            # 初始网络状态是由num_nodes个节点组成的规则网络，每次加入的边数由参数type0_init_degree调控
            self.G = nx.random_regular_graph(n=self.num_nodes, d=RG_init_degree)
        self.grid = mesa.space.NetworkGrid(self.G)
        self.schedule = mesa.time.RandomActivation(self)
        self.reset_random = reset_random  # 重置环境类型 True则为随机新生成，False 则使用固定随机树种子生成
        self.all_orders_list = all_orders_list(self.G, self.reset_random)
        self.new_orders = []  # 当前时刻产生的订单的数目
        self.finish_orders = 0  # 当前预期可以完成的order的数目（step中可以算到）
        self.match_agent = []
        self.match_order = []
        self.new_services = []  # 本轮新增的企业，记录这个为了设置新增每轮企业的智能值
        self.actions = {}  # 目前是中低智能体的id和对应的动作，动作就是order在match_order中的位置
        self.obs = {}  # 局部观测值，是强化学习的输入
        self.is_training = is_training  # model是处于训练状态还是推理状态，True是训练
        self.episode_length = episode_length  # 一次演化的时长
        # 设定一个social_matrix，用于存系统每个时间上新增的order和以及完成的order
        # 设定一个agent_matrix,用于存每个agent每个时间上的reward和cost
        self.social_matrix = dict()  # {时间t:[new_order,finish_order],}
        self.agent_matrix = dict()  # {时间t:{agent_id:[reward,cost],},}
        self.agent_utility_matrix = dict()  # 当前时刻系统中agent的个体效能值矩阵,不在match-agent的agent也会被存入，只是他们效能为0
        self.total_energy = []  # 系统的历史总energy
        self.vge = []  # 系统的历史价值生成效率
        self.agent_zhineng_matrix = dict()
        self.niches = []

        # 环境中每个智能体的当前奖赏值
        self.curr_optimization_metric = dict()
        self.init_services()
        self.set_intelligence(self.new_services)

        if self.is_training == False:
            self.obs = self.reset()
            self.trainer = trainer
            self.obs = {
                k: {
                    k1: v1 if type(v1) is np.ndarray else np.array([v1])
                    for k1, v1 in v.items()
                }
                for k, v in self.obs.items()
            }

        self.datacollector = mesa.DataCollector(
            {
                "avg_utility": lambda a: self.scenario_metrics()["agent/avg_utility"],
                "value_entropy": lambda a: self.scenario_metrics()["social/value_entropy"],
                # "productivity": lambda a: self.scenario_metrics()["social/productivity"],
                # "equality": lambda a: self.scenario_metrics()["social/equality"],
                "system_utility": lambda a: self.scenario_metrics()["social/utility"],
                "entropy": lambda a: self.scenario_metrics()["social/entropy"],
                # "DUtility": lambda a: self.scenario_metrics()["DUtility"],
                # "Effectiveness": lambda a: self.scenario_metrics()["Effectiveness"],
                # "new_order": lambda a: self.scenario_metrics()["Order"],
                # "finish_order": lambda a: self.finish_orders

            }
        )

    def init_services(self):
        self.new_services = []
        if not self.reset_random:
            random.seed(1)
        for node in self.G.nodes():
            a = ServiceAgent(
                unique_id=self.next_id(),
                model=self,
                service_type=generate_service_type(),
                difficulty=generate_difficulty(),
                energy=generate_energy(),
                imitate_pro=generate_imitate_pro(),
                cooperation=generate_cooperation()
            )
            # Add the agent to the node
            self.grid.place_agent(a, node)
            a.pos = node
            self.schedule.add(a)
            self.new_services.append(a)
            self.set_all_agents_list()

    def get_nums_of_orders(self):
        return len(self.all_resources)

    def get_nums_of_agents(self):
        return len(self.all_agents)

    def generate_orders(self):
        self.new_orders = []
        try:
            self.all_orders_list[self.schedule.steps]
        except IndexError:
            raise IndexError(self.schedule.steps)
        for ord in self.all_orders_list[self.schedule.steps]:
            a = OrderAgent(self.next_id(), model=self, order_difficulty=ord[3], order_type=ord[0], bonus=ord[1],
                           cost=ord[2])
            a.pos = ord[-1]
            self.schedule.add(a)
            self.grid.place_agent(a, a.pos)
            self.new_orders.append(a)
            self.set_all_agents_list()
        print("本轮订单已经生成,step:", self.schedule.steps)

    def set_all_agents_list(self):
        self.all_agents = []
        self.all_resources = []
        if self.schedule:
            for agent in self.schedule.agents:
                if isinstance(agent, OrderAgent):
                    self.all_resources.append(agent)
                elif isinstance(agent, ServiceAgent):
                    self.all_agents.append(agent)
                else:
                    pass

            self._agent_lookup = {str(agent.unique_id): agent for agent in self.all_agents}
            self._resource_lookup = {str(order.unique_id): order for order in self.all_resources}

    # 修改agent的智能等级
    def set_intelligence(self, new_agents):
        low_agents = []
        medium_agents = []
        high_agents = []
        imitate_agents = []
        temp_agents = set(new_agents)
        low_agents.extend(
            [temp_agents.pop() for _ in range(int(len(new_agents) * self.ratio_low - len(low_agents)))])
        medium_agents.extend(
            [temp_agents.pop() for _ in range(int(len(new_agents) * self.ratio_medium - len(medium_agents)))])
        imitate_agents.extend(
            [temp_agents.pop() for _ in range(int(len(new_agents) * self.ratio_imitate - len(imitate_agents)))])
        high_agents.extend(list(temp_agents))
        for agent in low_agents:
            agent.set_intelligence(0)
        for agent in medium_agents:
            agent.set_intelligence(1)
        for agent in imitate_agents:
            agent.set_intelligence(3)
        for agent in high_agents:
            agent.set_intelligence(2)

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

    # 比较两个agent是否彼此符合约束条件,即判断该service是否是order的潜在工人（充分条件）
    # 返回1代表是潜在工人
    def sufficient_constraint(self, order, service):
        # 空间约束
        # 时间约束
        # 预算约束
        # 技能约束
        if service.service_type not in list(order.order_type):
            return 0
        # 如果订单或企业二者有一个是禁止合作的，则技能约束部分必须满足所有技能条件
        if order.cooperation == 0 | service.cooperation == 0:
            if 0 < self.distance(order.pos, service.pos) <= order.vision and \
                    self.distance(order.pos, service.pos) / service.speed <= (
                    order.left_duration - order.handling_time) and \
                    self.distance(order.pos, service.pos) * service.move_cost <= (order.bonus - order.cost):
                # and \
                # self.skill_constraint(order, service).count(1) == 2:
                return 1
            else:
                return 0
        else:
            # 在其他情况下，可以合作，则技能约束部分只需满足一个及以上的技能条件即可
            if 0 < self.distance(order.pos, service.pos) <= order.vision and \
                    self.distance(order.pos, service.pos) / service.speed <= (
                    order.left_duration - order.handling_time) and \
                    self.distance(order.pos, service.pos) * service.move_cost <= (order.bonus - order.cost):
                # and \
                # self.skill_constraint(order, service).count(1) > 0:

                return 1
            else:
                return 0

    # 返回技能向量的满足列表
    # 如满足第一个不满足第二个，返回[1,0]
    def skill_constraint(self, order, service):
        # 是否满足技能点的列表
        list = [0, 0]

        # 判断订单的类型是否为企业可以处理的类型
        if all([(b - a) >= 0 for (a, b) in zip(order.skills[0], service.skills[0])]):
            list[0] = 1

        # 判断订单的难度是否为企业可以处理的难度
        if order.order_difficulty <= service.difficulty:
            list[1] = 1

        return list

    # 判断潜在工人组是否最终可获得该订单（必要条件）,可获得返回1，否则返回0
    def necessary_constraint(self, order, services):
        total_move_cost = 0
        total_skill = [0, 0]
        # # 如果订单不允许合作，那么潜在工人组的内工人的数量不能大于1 
        # if order.cooperation == 0 and len(services) > 1:
        #     return 0

        # 所有企业之间有通路，合作才能成立
        for i in range(0, len(services)):
            for j in range(i + 1, len(services)):
                if self.distance(self._agent_lookup[str(services[i])].pos,
                                 self._agent_lookup[str(services[j])].pos) < 0:
                    return 0
        for service in services:
            service = self._agent_lookup[str(service)]
            # 首先判断service与order之间有通路
            if self.network_type != "None":  # 如果为None，我们设定企业和订单没有通路时，企业也能执行订单
                if self.distance(order.pos, service.pos) < 0:
                    return 0
            # 合作组织中出现单个企业的收益小于0，则该合作无法成立
            if self.distance(order.pos, service.pos) * service.move_cost + order.cost / len(
                    services) > order.bonus / len(services):
                return 0
            total_move_cost += self.distance(order.pos, service.pos) * service.move_cost
            total_skill = [int(a or b) for (a, b) in zip(self.skill_constraint(order, service), total_skill)]
        # 整体预算约束：小组成员的行程代价之和加上处理订单的消耗小于订单给予的价值（等于也不行，那不是白干了）&& 整体技能须满足该订单所有的技能要求
        # if order.bonus >= total_move_cost + order.cost and sum(total_skill) == 2:
        if order.bonus >= total_move_cost + order.cost:
            return 1
        else:
            return 0

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
                continue
            self._agent_lookup.get(str(a_id), None).order_select = 1
            self.match_order[o_id].order_select = 1
            agent = self._agent_lookup.get(str(a_id), None)
            order = self.match_order[o_id]
            if agent.service_type not in list(order.order_type):
                agent.action_parse(-1)
                continue
            if self.distance(agent.pos, order.pos) < 0:
                agent.action_parse(-1)
                continue
            if order.unique_id not in order_action:
                order_action[order.unique_id] = dict()
                order_action[order.unique_id][agent.service_type] = {a_id: self.distance(agent.pos, order.pos)}
            elif agent.service_type not in order_action[order.unique_id]:
                order_action[order.unique_id][agent.service_type] = {a_id: self.distance(agent.pos, order.pos)}
            else:
                order_action[order.unique_id][agent.service_type].update({a_id: self.distance(agent.pos, order.pos)})

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
                    self._resource_lookup.get(str(o_id), None).services.extend(
                        [min(order_action[o_id][order_.order_type].items(), key=lambda x: x[1])[0]])
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
                            if self.necessary_constraint(order_, [agent1, agent2]):
                                list_service[str(agent1) + "+" + str(agent2)] = \
                                    order_action[o_id][types[0]][agent1] + \
                                    order_action[o_id][types[1]][agent2]

                if len(order_.order_type) == 3:
                    for agent1 in order_action[o_id]["A"].keys():
                        for agent2 in order_action[o_id]["B"].keys():
                            for agent3 in order_action[o_id]["C"].keys():
                                if self.necessary_constraint(order_, [agent1, agent2, agent3]):
                                    list_service[str(agent1) + "+" + str(agent2) + "+" + str(agent3)] = \
                                        order_action[o_id]["A"][agent1] + \
                                        order_action[o_id]["B"][agent2] + \
                                        order_action[o_id]["C"][agent3]

                if len(list_service) > 0:
                    x = min(list_service.items(), key=lambda x: x[1])[0]
                    self._resource_lookup.get(str(o_id), None).services.extend(x.split("+"))
                    self._agent_lookup.get(str(order_.services[0]), None).order = order_.unique_id
                    self.finish_orders += 1
                    self._resource_lookup.get(str(o_id), None).occupied = 1

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

    # 计算graph中两点的最短距离
    def distance(self, agent_pos, order_pos):
        if nx.has_path(self.grid.G, source=agent_pos, target=order_pos):
            return nx.shortest_path_length(self.grid.G, source=agent_pos, target=order_pos, weight=None)
        else:
            return -1

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

    def compute_order(self, agent):
        orders = []

        # 这里的self.all_orders要替换从匹配1中返回的orders
        for neighborhood in self.match_order:
            orders.extend([neighborhood.cooperation, neighborhood.cost,
                           neighborhood.bonus,
                           self.distance(neighborhood.pos, agent.pos) * agent.move_cost,
                           self.sufficient_constraint(neighborhood, agent)
                           ])

            orders.extend(self.skill_constraint(neighborhood, agent))

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

    def generate_observations(self):
        obs = {}
        # 影响选择订单规则的内在属性
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

    def reset(self, *args, **kwargs):
        # 重新生成企业
        self.schedule.steps = 0
        if self.reset_random:
            for agent in self.schedule.agents:
                if isinstance(agent, OrderAgent) or isinstance(agent, ServiceAgent):
                    self.schedule.remove(agent)
                    self.grid.remove_agent(agent)
            self.actions = {}
            # 重新初始化地图
            prob = self.avg_node_degree / self.num_nodes
            # 初始网络状态是由num_nodes个节点组成的随机网络，平均节点度数由参数avg_node_degree调控
            self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=prob)
            self.grid = mesa.space.NetworkGrid(self.G)
            # 重新初始化企业
            self.init_services()
            self.set_intelligence(self.new_services)
            # 重新生成订单
            self.generate_orders()
        self.match_order, self.match_agent = self.matching_service_order()
        self.obs = self.generate_observations()
        self.set_all_agents_list()
        return self.obs

    def action_parse(self, action_dict):
        self.actions = action_dict

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
                order_reward = {str(order_id): [] for order_id in agent.temp_actions}
                for order_id in agent.temp_actions:
                    order = self._resource_lookup[str(order_id)]
                    reward = order.bonus - self.distance(order.pos, agent.pos) * agent.move_cost - order.cost
                    order_reward[str(order.unique_id)].append(reward)
                # 只选择自己计算出的代价最小的order，不考虑合作分配和社会整体
                agent.selected_order_id = sorted(order_reward.items(), key=lambda o: o[1])[0][0]
                agent.order = agent.selected_order_id
                self.actions.update(
                    {str(agent.unique_id): self.match_order.index(self._resource_lookup[str(agent.selected_order_id)])})
            # 模仿学习的决策规则
            if agent.intelligence_level == 3:
                # 每个agent对应一个固定概率，通过模仿学习进化时，学习进化的就是这个概率值
                # 会以这个概率随机选择利益最大或者成本最小的订单。
                order_bonus = {str(order_id): [] for order_id in agent.temp_actions}
                order_total_cost = {str(order_id): [] for order_id in agent.temp_actions}
                for order_id in agent.temp_actions:
                    order = self._resource_lookup[str(order_id)]
                    bonus = order.bonus
                    total_cost = self.distance(order.pos, agent.pos) * agent.move_cost + order.cost
                    order_bonus[str(order.unique_id)].append(bonus)
                    order_total_cost[str(order.unique_id)].append(total_cost)
                pro = random.random()
                if pro < agent.imitate_pro:  # 产生的随机数小于imitate_pro，则选择花费最小的订单
                    agent.selected_order_id = sorted(order_total_cost.items(), key=lambda o: o[1], reverse=True)[0][0]
                else:  # 否则选择利益最大的订单
                    agent.selected_order_id = sorted(order_bonus.items(), key=lambda o: o[1])[0][0]
                agent.order = agent.selected_order_id
                self.actions.update(
                    {str(agent.unique_id): self.match_order.index(
                        self._resource_lookup[str(agent.selected_order_id)])})

    def scenario_metrics(self):
        metrics = dict()
        energy = np.array([agent.energy for agent in self._agent_lookup.values()])
        self.total_energy.append(sum(energy))
        # 当前系统中生态位数m
        m, niches = self.get_niche()
        # self.vge.append(get_vge(niches))
        # mul = list(np.multiply(np.array(self.vge), np.array(self.total_energy)))
        # metrics["social/productivity"] = get_productivity_network(mul)
        # metrics["social/equality"] = get_equality(energy)
        metrics["social/utility"] = system_utility(self)
        # metrics["DUtility"] = get_dutility(self.social_matrix, self.schedule.steps, gamma=0.8)
        # metrics["Effectiveness"] = get_effectiveness(energy, mul, self, gamma=0.8)
        # metrics["Order"] = len(self.all_orders_list[self.schedule.steps])  # 系统在当前时刻新生成的订单数
        # metrics["social/productivity"] = get_productivity(energy)
        # metrics["social/equality"] = get_equality(energy)
        metrics["social/value_entropy"] = get_vge(niches)
        metrics["social/entropy"] = H_t(niches)
        metrics["agent/avg_utility"] = self.get_avg_utility()
        # metrics[
        #     "SUtility"
        # ] = metrics["social/productivity"] * metrics["social/equality"]

        return metrics

    def get_avg_utility(self):
        self.agent_utility_matrix = dict()
        # ids = np.array([agentId for agentId in self.agent_matrix.get(str(self.schedule.steps)).keys()])
        # 这里用self.all_agents，而不是match_agent
        ids = np.array([agent.unique_id for agent in self.all_agents])
        for idx in ids:
            self_utility = get_self_utility(self.agent_matrix, idx, self.schedule.steps, gamma=1)
            # print("计算的该Agent的效能值", self_utility)
            self.agent_utility_matrix.update({str(idx): self_utility})
        # 计算所有个体效能平均值
        # print("match_agent",self.match_agent)
        N = len(self.agent_utility_matrix.keys())
        # print("N", N)
        V = sum(self.agent_utility_matrix.values())
        # print("V", V)
        return V / N

    # 统计每个学习机制的平均个体效能，用于鲇鱼效应
    def get_avg_utility_by_type(self, intelligence_type):
        agent_utility_matrix_type = dict()
        ids = np.array([agent.unique_id for agent in self.all_agents if agent.intelligence_level == intelligence_type])
        for idx in ids:
            self_utility = get_self_utility(self.agent_matrix, idx, self.schedule.steps, gamma=1)
            agent_utility_matrix_type.update({str(idx): self_utility})
        # print("self_utility", self.agent_utility_matrix)  # 得到当前时间段的个体效能总矩阵
        # 计算该学习机制下的个体效能平均值
        N = len(agent_utility_matrix_type.keys())
        V = sum(agent_utility_matrix_type.values())
        if N and V:
            return V / N
        else:
            return None

    # 计算所有个体的个体智能，存入self.agent_zhineng_matrix
    def get_zhineng(self):
        self.agent_zhineng_matrix = dict()
        avg_utility = self.get_avg_utility()
        # 将个体智能值存入矩阵
        for idx in self.agent_utility_matrix.keys():
            if avg_utility != 0:
                self.agent_zhineng_matrix.update({str(idx): (self.agent_utility_matrix[str(idx)] / avg_utility)})
        return self.agent_zhineng_matrix

    # 计算时，首先根据指定的生态位划分标准，利用常用的排序算法，对所有个体智能进行排序，将个体纳入合适的生态位，从而得到生态位数m
    def get_niche(self):
        # 生态位置的划分标准
        # 计算每个生态位上的数目
        niches = [0, 0, 0, 0, 0, 0]
        print('个体智能:', self.get_zhineng().values())
        for v in self.agent_zhineng_matrix.values():
            if v < -1:
                niches[0] += 1
            elif -1 <= v < 0:
                niches[1] += 1
            elif 0 <= v < 1:
                niches[2] += 1
            elif 1 <= v < 2:
                niches[3] += 1
            elif 2 <= v < 3:
                niches[4] += 1
            elif 3 <= v:
                niches[5] += 1
        # 得到系统当前生态位数,当前系统生态位的分布情况
        self.niches = niches
        print('生态位', niches)
        return sum(i > 0 for i in niches), niches

    def step(self):
        self.new_services = []
        # 检查当前energy最高的企业
        energy = [agent.energy for agent in self.all_agents]
        max_energy_agent = self.all_agents[np.argmax(energy)]

        # 首先检查本轮所有死去的订单和企业，从agent队列中移除
        for agent in self.schedule.agents:
            if agent.done:
                node = agent.pos
                self.schedule.remove(agent)
                self.grid.remove_agent(agent)
                self.set_all_agents_list()
                if isinstance(agent, ServiceAgent):
                    # 并且在self.actions中去除该agent的动作，避免下文传递时出错
                    self.actions.pop(str(agent.unique_id), None)
                    # 判断某节点上的企业如果死去，则在该节点上模仿当前energy最高的企业，新增一个随机节点
                    a = ServiceAgent(
                        unique_id=self.next_id(),
                        model=self,
                        service_type=generate_service_type(),  # 如果类型也模仿，会逐渐变成全部地图为同一类型的企业。这个需要思考
                        difficulty=max_energy_agent.difficulty,
                        energy=generate_energy(),
                        imitate_pro=generate_imitate_pro()
                    )
                    # Add the agent to the node
                    a.pos = node
                    self.grid.place_agent(a, node)
                    self.schedule.add(a)
                    self.new_services.append(a)
                    print("新加入的节点位置", a.pos)

        self.set_all_agents_list()
        self.set_intelligence(self.new_services)
        # 如果网络结构设置为None，则所有企业都不接受合作
        if self.network_type == "None":
            for agent in self.all_agents:
                agent.cooperation = 0

        # 进行本轮订单繁殖
        self.generate_orders()

        # 获取本轮匹配的agent和order
        self.match_order, self.match_agent = self.matching_service_order()
        # print('match_agent',len(self.match_agent))
        # print('all_agents',len(self.all_agents))

        # 随机、固定规则、模仿学习的可能动作，存入agent，并填入model的self.actions
        self.get_actions()

        # 生成强化学习的观测值
        self.obs = self.generate_observations()

        if not self.is_training:
            self.compute_rl_step()

        if not self.agent_matrix.get(str(self.schedule.steps)):
            self.agent_matrix.update({str(self.schedule.steps): {}})

        # 初始化的social_matrix
        if not self.social_matrix.get(str(self.schedule.steps)):
            self.social_matrix.update({str(self.schedule.steps): [len(self.new_orders), self.finish_orders]})

        alpha = 0.1
        reward = dict()

        # 存入强化学习的动作
        if self.actions is not None:
            for agent_idx, agent_actions in self.actions.items():
                if int(agent_idx) > 0:
                    # 当企业本轮没有破产，且目前状态是空闲时，才能传递给它新的动作
                    # if self._agent_lookup.get(str(agent_idx), None) and self._agent_lookup.get(str(agent_idx),
                    #                                                                            None).state == 0:
                    self._agent_lookup.get(str(agent_idx), None).action_parse(agent_actions)

            # 平台反选
            self.order_select()
            # self.schedule.steps从0开始
            self.schedule.step()  # 执行这一步骤,step+1

            if not self.agent_matrix.get(str(self.schedule.steps)):
                self.agent_matrix.update({str(self.schedule.steps): {}})
                # for agent in self.all_agents:
                #     self.agent_matrix.get(str(self.schedule.steps)).update({str(agent.unique_id): [0, 0]})

            for agent in self.match_agent:
                if agent.intelligence_level == 2:
                    # value是获得的利润，cost是消耗
                    value, cost = agent.process_order()
                    reward[str(agent.unique_id)] = self.compute_agent_reward(cost, value, alpha)

            # 生成虚拟企业
            num_agents = len(reward)

            # 更新本轮的social_matrix
            self.social_matrix.update({str(self.schedule.steps): [len(self.new_orders), self.finish_orders]})
            # print("self.social_matrix", self.social_matrix)
            # print("self.agent_matrix", self.agent_matrix)

            while num_agents < self.N:
                reward[str(-num_agents)] = 0
                num_agents += 1

        self.set_all_agents_list()

        # print('match_agent',len(self.match_agent))
        # print('all_agents',len(self.all_agents))

        # collect data
        self.datacollector.collect(self)
        # 将数据存入csv
        # utility_csv(self, self.get_avg_utility(), H_t(self.niches), 'utility.csv')

        scale = self.num_nodes  # 规模=节点数
        # # 个体智能箱线图=>按个体写入数据
        # for k in self.agent_zhineng_matrix.keys():
        #     agent_intelligence = self._agent_lookup.get(str(k)).intelligence_level
        #     agent_zhineng_csv(self, agent_intelligence, self.network_type,
        #                       self.agent_zhineng_matrix[k], "agent_zhineng.csv")

        # 统计每种学习机制内部的平均个体效能
        arr = []
        for i in range(4):
            v = self.get_avg_utility_by_type(i)
            if v:
                arr.append(v)
            else:
                arr.append(0)

        agent_utility_type_csv(self, self.get_avg_utility(), "catfish.csv", arr[0], arr[1], arr[2], arr[3],self.ratio_imitate, H_t(self.niches),get_vge(self.niches),system_utility(self))

        # 按系统整体写入数据
        # 不同学习机制下
        if self.ratio_low == 1:
            diff_csv(self, 'random', self.get_avg_utility(), H_t(self.niches), get_vge(self.niches),  system_utility(self),
                     'matthew.csv')
        elif self.ratio_medium == 1:
            diff_csv(self, 'fix_rules', self.get_avg_utility(),  H_t(self.niches),get_vge(self.niches),  system_utility(self),
                     'matthew.csv')
        elif self.ratio_imitate == 1:
            diff_csv(self, 'il', self.get_avg_utility(), H_t(self.niches), get_vge(self.niches),  system_utility(self),
                     'matthew.csv')
        elif self.ratio_high == 1:
            diff_csv(self, 'rl', self.get_avg_utility(), H_t(self.niches), get_vge(self.niches), system_utility(self),
                     'matthew.csv')

        # # 不同拓扑结构下：平均个体效能、价值熵、系统效能
        # diff_csv(self, self.network_type, self.get_avg_utility(), get_vge(self.niches),  system_utility(self),
        #          'diff_topology.csv')
        
        # 不同订单倍数下：平均个体效能、价值熵、系统效能
        diff_csv(self, get_n(), self.get_avg_utility(),  H_t(self.niches), get_vge(self.niches),  system_utility(self),
                 'invotion.csv')
        
        # # 不同scale（即节点数）下：平均个体效能、价值熵、系统效能
        # diff_csv(self, scale, self.get_avg_utility(), get_vge(self.niches), system_utility(self),
        #          'diff_scale.csv')
        
        # # 不同平均节点度数下：平均个体效能、价值熵、系统效能
        # diff_csv(self, self.avg_node_degree, self.get_avg_utility(), get_vge(self.niches), system_utility(self),
        #          'diff_degree.csv')
        #这里用self.all_agents，而不是match_agent
        ids = np.array([agent.unique_id for agent in self.all_agents])
        for idx in ids:
            self_utility = get_self_utility(self.agent_matrix, idx, self.schedule.steps, gamma=1)
            # print("计算的该Agent的效能值", self_utility)
            self.agent_utility_matrix.update({str(idx): self_utility})

        # # 因果推断
        # # 分别选取100、150、200、250四个时间点，50个serviceAgent
        # # "agent_id","time_step", "agent_utility", "avg_utility","system_utility","act_rules", "cooperation","topology", "agent_skill"
        # if self.schedule.steps == 100:
        #     # write_csv_hearders("casual.csv", ["agent_id","time_step", "agent_utility", "avg_utility","system_utility","act_rules", "cooperation","topology", "agent_skill"])
        #     for agent in self.all_agents:
        #          write_csv_rows("casual.csv",
        #             [[agent.unique_id, self.schedule.steps,self.agent_utility_matrix.get(str(agent.unique_id)),self.get_avg_utility(), system_utility(self), agent.intelligence_level,agent.cooperation, self.network_type,agent.difficulty]])
           
        # if self.schedule.steps == 150 or  self.schedule.steps == 200 or  self.schedule.steps == 250:
        #     for agent in self.all_agents:
        #          write_csv_rows("casual.csv",
        #             [[agent.unique_id, self.schedule.steps,self.agent_utility_matrix.get(str(agent.unique_id)),self.get_avg_utility(), system_utility(self), agent.intelligence_level,agent.cooperation, self.network_type,agent.difficulty]])
    
        # 演化结束的判断和信息打印
        self.done = {"__all__": self.schedule.steps >= self.episode_length}
        info = {k: {} for k in self.obs.keys()}

        # 新版本的gym :obs, rewards, terminateds, truncateds, infos = env.step(agent_dict)
        return self.obs, reward, self.done, info

    def run_model(self):

        for i in range(300):
            # if i == 140:
            #     self.reset()
            self.step()
            # print(self.G.nodes)


# 注册强化学习环境
def env_creator(env_config):  # 此处的 env_config对应 我们在建立trainer时传入的dict env_config
    return RLlibEnvWrapper(env_config, CloudManufacturing_network)


register_env(CloudManufacturing_network.name, env_creator)
