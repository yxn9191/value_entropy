import math

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

    # 判断是否满足技能向量的约束
    def skill_constraint(self, order, service):
        i = 2
        order_diff = 0
        j = 2
        service_diff = 0

        for i in order.skills[1][i]:
            order_diff = order_diff + order.skills[1][i] * 2 ^ (2 - i)
            i = i - 1

        for j in service.skills[1][j]:
            service_diff = service_diff + service_diff[1][j] * 2 ^ (2 - j)
            j = j - 1

        if all([(b - a) >= 0 for (a, b) in zip(order.skills[0], service.skills[0])]) and \
                order_diff <= service_diff:
            return 1
        else:
            return 0

    #计算agent 周围订单的(待补充）
    def compute_order(self, agent):
        orders = dict()
        neighborhoods = self.model.grid.get_cell_list_contents(agent.pos)
        i = 0
        for neighborhood in neighborhoods:
            i+=1
            if neighborhood.name == "order":
                orders[str(neighborhood.unque_id)]=np.array([neighborhood.cost, neighborhood.bonus,
                distance(neighborhood.pos,agent.pos),self.skill_constraint(neighborhood,agent),
                neighborhood.match_vector(neighborhood.order_type, neighborhood.order_difficulty)]
                )
            else: 
                orders["virtual_agent"+str(i)]  = np.array([0,0,0,0,0])
        return orders

    #生成观察值（强化学习的输入）
    def generate_observations(self):
        obs = {}
        #影响选择订单规则的内在属性
        
        for agent in self.all_agent:

            obs[str(agent.unique_id)] = {"time": self.timestep}
            obs[str(agent.unique_id)].update(self.compute_order(agent))

        return obs

    # 生成即时奖赏值（强化学习的输入，待补充）
    def generate_rewards(self):
        #具体计算公式待修改，形式如下
        #上一个时间的奖赏
        utility_at_last_time_step = deepcopy(self.curr_optimization_metric)
        #当前奖赏
        self.curr_optimization_metric = {str(agent.unique_id): agent.compute_reward() for agent in self.all_agent}
        #即时奖商
        reward = {
            k: float(v - utility_at_last_time_step[k])
            for k, v in self.curr_optimization_metric.items()
        }
        reward = {str(agent.unique_id): 1 for agent in self.all_service}
        return reward

    def step(self, actions=None):
        """Advance the model by one step."""
        # self.schedule.step()
        pass


# 计算两位置的直线距离
def distance(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


# 计算移动到新位置的路线长度
def move_len(A, B):
    return sum([(a - b) for (a, b) in zip(A, B)])
