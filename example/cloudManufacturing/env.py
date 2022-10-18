import math
from copy import deepcopy
import random

import mesa
from mesa import DataCollector

from base.environment import BaseEnvironment
from example.cloudManufacturing.orderAgent import OrderAgent
from example.cloudManufacturing.organization import Organization
from example.cloudManufacturing.serviceAgent import ServiceAgent
from ray.tune.registry import register_env
from algorithm.rl.env_warpper import RLlibEnvWrapper


class CloudManufacturing(BaseEnvironment):
    name = "CloudManufacturing"

    def __init__(self, num_order=200, num_service=100, width=20, height=20, num_organization=2, episode_length=200,
                 ratio_low=0, ratio_medium=0):
        super().__init__()
        self.num_organization = num_organization  # 组织的数目
        self.episode_length = episode_length  # 一次演化的时长
        self.new_orders = []  # 当前时刻产生的订单的数目
        self.finish_orders = 0  # 当前预期可以完成的order的数目（step中可以算到）
        self.actions = None
        # 算法1中的M（订单）和N（企业）
        self.M = 200
        self.N = 100

        self.schedule = mesa.time.RandomActivationByType(self)
        self.grid = mesa.space.MultiGrid(width, height, True)  # True一个关于网格是否为环形的布尔值
        self.ratio_low = ratio_low
        self.ratio_medium = ratio_medium
        self.ratio_high = 1 - self.ratio_low - self.ratio_medium
        self.match_agent = []
        self.match_order = []

        # Create agents（包括企业和订单）
        self.generate_orders(num_order)
        self.generate_services(num_service)
        self.set_all_agents_list()  # 注意！！有了这个函数，self.all_agents和look_up系列会直接同schedule变动，
        # 而不需要额外操作向其中手动添加agent了，此函数每次在环境的step开始时就调用，同步系统中所有存储agent的list

        # # 数据收集器
        # self.collector = DataCollector(
        #     model_reporters={"Social Reward": },  # 计算社会整体收益
        #     agent_reporters={"Service Num": lambda m: m.schedule.get_type_count(ServiceAgent)}
        # )

    # 修改agent的智能等级
    def set_intelligence(self):
        temp_agents = set(self.all_agents)
        low_agents = [temp_agents.pop() for _ in range(int(len(self.all_agents) * self.ratio_low))]
        medium_agents = [temp_agents.pop() for _ in range(int(len(self.all_agents) * self.ratio_medium))]
        hight_agents = temp_agents
        for agent in low_agents:
            agent.set_intelligence(0)
        for agent in medium_agents:
            agent.set_intelligence(1)
        for agent in hight_agents:
            agent.set_intelligence(2)

    def random_place_agent(self, agent):
        x = self.random.randrange(self.grid.width)
        y = self.random.randrange(self.grid.height)
        self.grid.place_agent(agent, (x, y))
        # a.location = (x, y) 这行不需要，place_agent就自动将该属性添加到agent中，属性值为pos

    def generate_services(self, new_service_num=5):
        # 默认每轮新增5企业
        organization = Organization(random.randint(1, 2), self, [])

        for j in range(new_service_num):
            s = ServiceAgent(self.next_id(), self, generate_service_type(), generate_difficulty()
                             , organization)

            self.schedule.add(s)
            self.random_place_agent(s)

    def generate_orders(self, new_orders_num=25):
        self.new_orders = []
        for i in range(new_orders_num):
            a = OrderAgent(self.next_id(), self, generate_difficulty(), generate_order_type())

            self.schedule.add(a)
            self.random_place_agent(a)
            self.new_orders.append(a)

    # 比较两个agent是否彼此符合约束条件,即判断该service是否是order的潜在工人（充分条件）
    # 返回1代表是潜在工人
    def sufficient_constraint(self, order, service):
        # 空间约束
        # 时间约束
        # 预算约束
        # 技能约束

        # 如果订单或企业二者有一个是禁止合作的，则技能约束部分必须满足所有技能条件
        if order.cooperation == 0 | service.cooperation == 0:
            if distance(order.pos, service.pos) <= order.vision and \
                    move_len(order.pos, service.pos) / service.speed <= (order.left_duration - order.handling_time) and \
                    move_len(order.pos, service.pos) * service.move_cost <= (order.energy - order.consumption) and \
                    skill_constraint(order, service).count(1) == len(order.skills) == len(service.skills):
                return 1
            else:
                return 0
        else:
            # 在其他情况下，可以合作，则技能约束部分只需满足一个及以上的技能条件即可
            if distance(order.pos, service.pos) <= order.vision and \
                    move_len(order.pos, service.pos) / service.speed <= (order.left_duration - order.handling_time) and \
                    move_len(order.pos, service.pos) * service.move_cost <= (order.bonus - order.cost) and \
                    skill_constraint(order, service).count(1) > 0:
                return 1
            else:
                return 0

    # 判断潜在工人组是否最终可获得该订单（必要条件）,可获得返回1，否则返回0
    def necessary_constraint(self, order, services):
        total_move_cost = 0
        total_skill = [0 for i in range(len(order.skills))]

        for service in services:
            # 合作组织中出现单个企业的收益小于0，则该合作无法成立
            if distance(order, service) * service.move_cost + order.cost / len(services) < order.bonus / len(services):
                return 0
            total_move_cost += distance(order, service) * service.move_cost
            total_skill = [int(a or b) for (a, b) in zip(skill_constraint(order, service), total_skill)]

        # 整体预算约束：小组成员的行程代价之和加上处理订单的消耗小于订单给予的价值（等于也不行，那不是白干了）&& 整体技能须满足该订单所有的技能要求
        if order.bonus > total_move_cost + order.cost and total_skill.count(1) == len(order.skills):
            return 1
        else:
            return 0

    def generate_action_mask(self):
        masks = {str(agent.unique_id): [0 for i in range(self.M)] for agent in self.match_agent}
        position = 0
        for order in self.match_order:
            for agent in self.match_agent:
                masks[str(agent.unique_id)][position] = self.sufficient_constraint(order, agent)
            position += 1

        return masks

    # 低、中智能情况，生成每个agent可能选择的order,存入agent的selected_order_id
    # 同一agent在每一step中，会同时存有低、中、高智能的动作，但是根据不同情况选择执行
    def get_actions(self):
        temp_actions = {str(agent.unique_id): [] for agent in self.all_agents}
        for order in self.all_resources:
            for agent in self.all_agents:
                if self.sufficient_constraint(order, agent) == 1:
                    temp_actions[str(agent.unique_id)].append(order.unique_id)
                    agent.temp_actions = temp_actions[str(agent.unique_id)]
                    if agent.intelligence_level == 0:
                        # 随机选择一个满足充分约束的订单
                        agent.selected_order_id = random.choice(agent.temp_actions)
                    if agent.intelligence_level == 1:
                        order_reward = {str(order_id): [] for order_id in agent.temp_actions}
                        for order_id in agent.temp_actions:
                            order = self._resource_lookup(order_id)
                            reward = order.bonus - distance(agent.pos, order.pos) * agent.move_cost - order.cost
                            order_reward[str(order.unique_id)].append(reward)
                        # 只选择自己计算出的代价最小的order，不考虑合作分配和社会整体
                        agent.selected_order_id = sorted(order_reward.items(), key=lambda o: o[1])[0][0]

    # 计算局部观察值,待修改
    def compute_order(self, agent):
        orders = []

        # 这里的self.all_orders要替换从匹配1中返回的orders
        for neighborhood in self.match_order:
            orders.extend([neighborhood.cooperation, neighborhood.cost,
                           neighborhood.bonus,
                           move_len(neighborhood.pos, agent.pos) * agent.move_cost,
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

        num_orders = len(obs)
        while num_orders < self.M:
            _obs[str(-num_orders)] = [0, 0, 0, 0, 0]
            _obs[str(-num_orders)].extend([0 for i in range(len(self.all_resources[0].skills))])

        return _obs

    # 生成观察值（强化学习的输入,待修改）
    def generate_observations(self):
        obs = {}
        self.match_order, self.match_agent = self.matching_service_order()
        # 影响选择订单规则的内在属性,#这里的 self._agent_lookup要替换从匹配1中返回的agent
        # match_agent = list(self.match_agent)
        for agent in self.match_agent:
            obs[str(agent.unique_id)] = {"cooperation": agent.cooperation}
            obs[str(agent.unique_id)].update({"orders": self.compute_order(agent)})

        _obs = self.get_other_agent_obs(obs)

        for agent in self.match_agent:
            obs_ = deepcopy(_obs)
            del obs_[str(agent.unique_id)]
            obs[str(agent.unique_id)].update({"others": list(obs_.values())})

        # 生成虚拟企业
        num_agents = len(self.match_agent)
        while num_agents < self.N:
            obs[str(-num_agents)] = {"cooperation": 0}
            # other = {str(-i): [0, 0, 0, 0, 0, 0] for i in range(1000, 1200)}
            other = {"orders": [0 for i in range(1400)]}
            action_mask = {"action_mask": [0 for i in range(self.M)]}
            obs[str(-num_agents)].update(other)
            obs[str(-num_agents)].update({"others": [[0 for i in range(self.M * 7 + 1)]] for _ in range(self.N - 1)})
            obs[str(-num_agents)].update(action_mask)
            num_agents += 1

        # Get each agent's action masks and incorporate them into the observations
        for aidx, amask in self.generate_action_mask().items():
            obs[aidx]["action_mask"] = amask
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
        for order in self.all_resources:
            if order.occupied == 0:
                for agent in self.all_agents:
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
        orders = self.actions.values()
        self.finish_orders = 0
        order_action = dict()

        for a_id, o_id in self.actions.items():
            if int(a_id) < 0:
                continue
            if int(o_id) < 0:
                agent = self._agent_lookup[a_id]
                agent.action = -1
            agent = self._agent_lookup[a_id]
            order = self._resource_lookup[o_id]
            if o_id not in order_action:
                order_action[o_id] = dict()
                order_action[o_id][agent.service_type] = {a_id: move_len(agent.pos, order.pos)}
            elif agent.service_type not in order_action[o_id]:
                order_action[o_id][agent.service_type] = {a_id: move_len(agent.pos, order.pos)}
            else:
                order_action[o_id][agent.service_type].update({a_id: move_len(agent.pos, order.pos)})

        for o_id in order_action.keys():
            order_ = self._resource_lookup[order]
            if len(order_.ordertype) == 1:
                order_.services.extend(min(order_action[o_id][order_.ordertype].items(), key=lambda x: x[1])[0])
                self.finish_orders += 1
                order_.occupied = 1
                for a_id in order_action[o_id][order_.ordertype].keys():
                    if a_id not in order_.services:
                        self._agent_lookup[a_id].action = -1
            else:
                if len(order_action[o_id]) < len(order_.ordertype):
                    for service_type in order_action[o_id].keys():
                        for a_id in order_action[o_id][service_type].keys():
                            self._agent_lookup[a_id].action = -1
                else:
                    self.finish_orders += 1
                    order_.occupied = 1
                    for service_type in order_action[o_id].keys():
                        order_.services.extend(min(order_action[o_id][service_type].items(), key=lambda x: x[1])[0])
                    for service_type in order_action[o_id].keys():
                        for a_id in order_action[o_id][service_type].keys():
                            if a_id not in order_.services:
                                self._agent_lookup[a_id].action = -1

    def step(self):

        """Advance the model by one step."""
        # 首先检查本轮所有死去的订单和企业，从agent队列中移除
        for agent in self.schedule.agents:
            if agent.done:
                self.schedule.remove(agent)

        # 生成本轮新的企业和订单
        self.generate_orders()
        self.generate_services()

        self.set_all_agents_list()

        alpha = 0.1
        reward = dict()

        # 低智能和中智能的可能动作，存入agent
        self.get_actions()

        if self.actions is not None:

            # 这里的self._agent_lookup也要换成算法1获得的企业集合
            for agent_idx, agent_actions in self.actions.items():
                agent = self._agent_lookup.get(str(agent_idx), None)
                agent.action_parse(agent_actions)

            # 平台反选
            self.order_select()

            for agent_idx, agent_actions in self.actions.items():
                agent = self._agent_lookup.get(str(agent_idx), None)
                agent.action_parse(agent_actions)
                value, cost = agent.process_order()
                reward[str(agent.unique_id)] = self.compute_agent_reward(cost, value, alpha)

        obs = self.generate_observations()

        # 演化结束的判断，待修改
        done = {"__all__": self.schedule.steps >= self.episode_length}
        info = {k: {} for k in obs.keys()}

        # self.collector.collect(self)
        # 激活agent，每个agent执行自己全部动作
        # self.schedule.step()

        return obs, reward, done, info


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
    list = [0 for i in range(len(order.skills))]

    # 判断订单的类型是否为企业可以处理的类型
    if all([(b - a) >= 0 for (a, b) in zip(order.skills[0], service.skills[0])]):
        list[0] = 1

    # 判断订单的难度是否为企业可以处理的难度
    for k in reversed(order.skills[1]):
        order_diff = order_diff + k * 2 ^ i
        i += 1

    for k in reversed(service.skills[1]):
        service_diff = service_diff + k * 2 ^ j
        j += 1

    if order_diff <= service_diff:
        list[1] = 1

    return list


def generate_order_type():
    weight = {"A": 0.2, "B": 0.2, "C": 0.2, "AB": 0.1, "AC": 0.1, "BC": 0.1, "ABC": 0.1}
    return random.choices(list(weight.keys()), weights=list(weight.values()), k=1)[0]


def generate_service_type():
    weight = {"A": 0.3, "B": 0.3, "C": 0.4}
    return random.choices(list(weight.keys()), weights=list(weight.values()), k=1)[0]


def generate_difficulty():
    return random.randint(1, 3)


#     #
# def social_reward(model):
#     return model.finish_orders / len(model.new_orders)

# 注册强化学习环境
def env_creator(env_config):  # 此处的 env_config对应 我们在建立trainer时传入的dict env_config
    return RLlibEnvWrapper(env_config, mesaEnv=CloudManufacturing)


register_env(CloudManufacturing.name, env_creator)
