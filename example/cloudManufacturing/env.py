import math
from copy import deepcopy

import mesa
import numpy as np

from base.environment import BaseEnvironment
from example.cloudManufacturing.orderAgent import OrderAgent
from example.cloudManufacturing.organization import Organization
from example.cloudManufacturing.serviceAgent import ServiceAgent


class CloudManufacturing(BaseEnvironment):

    def __init__(self, num_order=200, num_service=100, width=20, height=20, num_organization=2, episode_length=200):
        super().__init__()
        self.order_num = num_order  # 不同类型订单的数目
        self.service_num = num_service  # 不同企业的数目
        self.num_organization = num_organization  # 组织的数目
        self.episode_length = episode_length  # 一次演化的时长
        self.timestep = 0  # 环境当前处于的时间点
        self.new_orders = []  # 当前时刻产生的订单的数目
        self.finish_orders = 0  # 当前预期可以完成的order的数目（step中可以算到）
        self.actions = None

        self.schedule = mesa.time.RandomActivationByType(self)
        self.grid = mesa.space.MultiGrid(width, height, True)  # True一个关于网格是否为环形的布尔值

        # Create agents（包括企业和订单）
        a_A, a_B, a_C, a_AB, a_BC, a_AC, a_ABC = self.generate_order(self.order_num)
        self.schedule.add(a_A)
        self.schedule.add(a_B)
        self.schedule.add(a_C)
        self.schedule.add(a_AB)
        self.schedule.add(a_BC)
        self.schedule.add(a_AC)
        self.schedule.add(a_ABC)

        self.random_placeAgent(a_A)
        self.random_placeAgent(a_B)
        self.random_placeAgent(a_C)
        self.random_placeAgent(a_AB)
        self.random_placeAgent(a_BC)
        self.random_placeAgent(a_AC)
        self.random_placeAgent(a_ABC)

        organization1 = Organization(1, self, [])
        organization2 = Organization(2, self, [])

        for j in range(math.floor(self.service_num / 3 / 3)):
            s_A_1 = ServiceAgent(self.next_id(), self, "A", 1, organization1)
            s_B_1 = ServiceAgent(self.next_id(), self, "B", 1, organization1)
            s_C_1 = ServiceAgent(self.next_id(), self, "C", 1, organization1)

            self.random_placeAgent_left_down(s_A_1)
            self.random_placeAgent_left_down(s_B_1)
            self.random_placeAgent_left_down(s_C_1)

            self.all_agent.append(s_A_1)
            self.all_agent.append(s_B_1)
            self.all_agent.append(s_C_1)

        for j in range(math.floor(self.service_num / 3 / 3)):
            s_A_2 = ServiceAgent(self.next_id(), self, "A", 1, organization2)
            s_B_2 = ServiceAgent(self.next_id(), self, "B", 2, organization2)
            s_C_2 = ServiceAgent(self.next_id(), self, "C", 3, organization2)

            self.random_placeAgent_right_up(s_A_2)
            self.random_placeAgent_right_up(s_B_2)
            self.random_placeAgent_right_up(s_C_2)

            self.all_agent.append(s_A_2)
            self.all_agent.append(s_B_2)
            self.all_agent.append(s_C_2)

        for j in range(math.floor(self.service_num / 3 / 36)):
            s_A_3 = ServiceAgent(self.next_id(), self, "A", 1, None)
            s_B_3 = ServiceAgent(self.next_id(), self, "B", 2, None)
            s_C_3 = ServiceAgent(self.next_id(), self, "C", 3, None)
            self.random_placeAgent(s_A_3)
            self.random_placeAgent(s_B_3)
            self.random_placeAgent(s_C_3)

            self.all_agent.append(s_A_3)
            self.all_agent.append(s_B_3)
            self.all_agent.append(s_C_3)

        self._agent_lookup = {str(agent.unique_id): agent for agent in self.all_agent}
        self.all_orders = deepcopy(self.new_orders)  # 当前环境中的所有order
        self._order_lookup = {str(order.unique_id): order for order in self.all_orders}

    def random_placeAgent(self, agent):
        x = self.random.randrange(self.grid.width)
        y = self.random.randrange(self.grid.height)
        self.grid.place_agent(agent, (x, y))
        # a.location = (x, y) 这行不需要，place_agent就自动将该属性添加到agent中，属性值为pos

    def random_placeAgent_left_down(self, agent):
        x = self.random.randrange(math.ceil(self.grid.width / 3))
        y = self.random.randrange(math.ceil(self.grid.height / 3))
        self.grid.place_agent(agent, (x, y))
        # a.location = (x, y) 这行不需要，place_agent就自动将该属性添加到agent中，属性值为pos

    def random_placeAgent_right_up(self, agent):
        x = self.random.randrange(math.ceil(self.grid.width / 3 * 2), self.grid.width)
        y = self.random.randrange(math.ceil(self.grid.height / 3 * 2), self.grid.height)
        self.grid.place_agent(agent, (x, y))

    def generate_order(self, num_order):

        # 波动公式(200为初始客户数,25为波动值)
        self.order_num = num_order + 25 * math.sin(self.schedule.steps)

        for i in range(math.floor(self.order_num / 7)):
            a_A = OrderAgent(self.next_id(), self, 1, "A")
            a_B = OrderAgent(self.next_id(), self, 2, "B")
            a_C = OrderAgent(self.next_id(), self, 3, "C")
            a_AB = OrderAgent(self.next_id(), self, 1, "AB")
            a_BC = OrderAgent(self.next_id(), self, 2, "BC")
            a_AC = OrderAgent(self.next_id(), self, 3, "AC")
            a_ABC = OrderAgent(self.next_id(), self, 1, "ABC")

            self.new_orders.extend([a_A, a_B, a_C, a_AB, a_BC, a_AC, a_ABC])

        return a_A, a_B, a_C, a_AB, a_BC, a_AC, a_ABC

    # 比较两个agent是否彼此符合约束条件
    def constraint(self, order, service):
        # 空间约束
        # 时间约束
        # 预算约束
        # 技能约束
        return distance(order.pos, service.pos) <= order.vision and \
               move_len(order.pos, service.pos) / service.speed <= (order.left_duration - order.handling_time) and \
               move_len(order.pos, service.pos) * service.move_cost <= (order.energy - order.consumption) and \
               self.skill_constraint()

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
                    self.skill_constraint().count(1) == len(order.skills) == len(service.skills):
                return 1
            else:
                return 0
        else:
            # 在其他情况下，可以合作，则技能约束部分只需满足一个及以上的技能条件即可
            if distance(order.pos, service.pos) <= order.vision and \
                    move_len(order.pos, service.pos) / service.speed <= (order.left_duration - order.handling_time) and \
                    move_len(order.pos, service.pos) * service.move_cost <= (order.energy - order.consumption) and \
                    self.skill_constraint().count(1) > 0:
                return 1
            else:
                return 0

    # 判断潜在工人组是否最终可获得该订单（必要条件）,可获得返回1，否则返回0
    def necessary_constraint(self, order, services):
        total_move_cost = 0
        total_skill = [0 for i in range(len(order.skills))]

        for service in services:
            total_move_cost += distance(order, service) * service.move_cost
            total_skill = [int(a or b) for (a, b) in zip(self.skill_constraint(order, service), total_skill)]

        # 整体预算约束：小组成员的行程代价之和加上处理订单的消耗小于订单给予的价值（等于也不行，那不是白干了）&& 整体技能须满足该订单所有的技能要求
        if order.bonus > total_move_cost + order.cost and total_skill.count(1) == len(order.skills):
            return 1
        else:
            return 0

    # 返回技能向量的满足列表
    # 如满足第一个不满足第二个，返回[1,0]
    def skill_constraint(self, order, service):
        i = 2
        order_diff = 0
        j = 2
        service_diff = 0
        # 是否满足技能点的列表
        list = [0 for i in range(len(order.skills))]

        # 判断订单的类型是否为企业可以处理的类型，如果满足此要求count+1
        if all([(b - a) >= 0 for (a, b) in zip(order.skills[0], service.skills[0])]):
            list[0] = 1

        # 判断订单的难度是否为企业可以处理的难度，如果满足此要求count+1
        for i in order.skills[1][i]:
            order_diff = order_diff + order.skills[1][i] * 2 ^ (2 - i)
            i -= 1

        for j in service.skills[1][j]:
            service_diff = service_diff + service_diff[1][j] * 2 ^ (2 - j)
            j -= 1

        if order_diff <= service_diff:
            list[1] = 1

        return list

    # 计算局部观察值,待修改
    def compute_order(self, agent):
        orders = dict()
        # 这里的self.all_orders要替换从匹配1中返回的orders
        for neighborhood in self.all_orders:
            orders[str(neighborhood.unque_id)] = np.array([neighborhood.neighborhood, neighborhood.cost,
                                                           neighborhood.bonus,
                                                           distance(neighborhood.pos, agent.pos),
                                                           self.sufficient_constraint(neighborhood, agent),
                                                           neighborhood.match_vector(neighborhood.order_type,
                                                                                     neighborhood.order_difficulty)]
                                                          )
        # 生成虚拟订单
        num_orders = len(self.all_orders)
        while (num_orders < self.order_num):
            orders[str(-num_orders)] = np.array([0, 0, 0, 0, 0, 0])
            num_orders += 1

        return orders

    # 获得其他agent的观察
    def get_other_agent_obs(self, obs):
        _obs = dict()
        for k in obs.keys():
            _obs[k] = []
            for key in obs[k].keys():
                _obs[k] += obs[k][key]
        return _obs

    # 生成观察值（强化学习的输入,待修改）
    def generate_observations(self):
        obs = {}
        # 影响选择订单规则的内在属性,#这里的 self._agent_lookup要替换从匹配1中返回的agent
        for agent in self._agent_lookup.values():
            obs[str(agent.unique_id)] = {"cooperation": agent.cooperation}
            obs[str(agent.unique_id)].update(self.compute_order(agent))

        _obs = self.get_other_agent_obs(obs)

        for agent in self._agent_lookup.values():
            obs_ = deepcopy(_obs)
            del obs_[str(agent.unique_id)]
            obs[str(agent.unique_id)].update({"others": obs_.values})

        # 生成虚拟企业
        num_agents = len(self._agent_lookup)
        while (num_agents < self.service_num):
            obs[str(-num_agents)] = {"cooperation": 0}
            other = {str(-i): np.array([0, 0, 0, 0, 0, 0]) for i in range(1000, 1200)}
            obs[str(-num_agents)].update(other)
            num_agents += 1

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

    def step(self):
        """Advance the model by one step."""
        # self.schedule.step()
        self.timestep += 1
        alpha = 0.1
        reward = dict()

        if self.actions is not None:
            cost, value = 0
            #这里的self._agent_lookup也要换成算法1获得的企业集合
            for agent_idx, agent_actions in self.actions.items():
                agent = self._agent_lookup.get(str(agent_idx), None)
                agent.action_parse(agent_actions)
                cost, value = agent.step()
                reward[str(agent.unique_id)] = self.compute_agent_reward(cost, value, alpha)

        obs = self.generate_observations()
        # 演化结束的判断，待修改
        done = {"__all__": self.timestep >= self.episode_length}
        info = {k: {} for k in obs.keys()}

        return obs, reward, done, info


# 计算两位置的直线距离
def distance(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


# 计算移动到新位置的路线长度
def move_len(A, B):
    return sum([(a - b) for (a, b) in zip(A, B)])
