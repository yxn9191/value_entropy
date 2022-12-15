# 服务节点【提供服务者-企业】
from random import randint

from base.geoagent import GeoAgent
import random
import math
from shapely.geometry import Point

class ServiceAgent(GeoAgent):
    name = "Service"

    def __init__(self, unique_id,
                 model,
                 shape,
                 service_type,
                 difficulty,
                 organization = None,
                 speed=25,
                 energy=random.uniform(1e3, 5e3),
                 consumption=random.uniform(1, 10),
                 failure_prob=0.1,
                 cooperation=1,
                 move_cost= 0.5,
                 intelligence_level=2
                 ):
        super().__init__(unique_id, model, shape)
        self.energy = energy  # 企业的能量
        self.service_type = service_type  # 企业可以处理的订单类型：A,B,C
        self.difficulty = difficulty  # 可处理的订单的最大难度等级
        self.cooperation = cooperation  # 是否接受合作,接受为1，禁止为0
        self.speed = speed  # 移动速度为1
        self.move_cost = move_cost  # 移动单位距离的开销
        self.consumption = consumption  # 订单的成本
        self.failure_prob = failure_prob  # 企业处理失败订单的概率
        self.create_time = self.model.schedule.steps  # 企业agent被创建时间
        self.service_satisfaction = 0  # 企业的满意度
        self.cooperation = cooperation  # 是否合作
        self.cooperation_service = []
        self.organization = organization  # 组织中的企业（组织中的企业协作成本低，假设最初没有在任何组织中）
        self.match_vector(self.service_type, self.difficulty)
        self.done = False
        self.temp_actions = None  # 用来保存agent在满足充分约束下的所有可以选择的订单，值是order的unique_id的数组
        self.selected_order_id = None  # 中低智能下agent最终选择的订单

        self.intelligence_level = intelligence_level
        self.order_end_time = 0  # 企业选择的订单，结束处理的时间
        self.order_select = None
        # self.now_cost = 0
        # self.delta_x = 0  # 记录要想新位置移动多少x
        # self.delta_y = 0  # 记录要往新位置移动多少y
        self.order = None  # 企业正在处理的order

    def match_vector(self, service_type, difficulty):
        if service_type == "A":
            self.skills = [[1, 0, 0]]
        elif service_type == "B":
            self.skills = [[0, 1, 0]]
        elif service_type == "C":
            self.skills = [[0, 0, 1]]
        else:
            self.skills = [[0, 0, 0]]  # 出错，000无法与任何订单匹配

        if difficulty == 1:
            self.skills.append([0, 0, 1])
        elif difficulty == 2:
            self.skills.append([0, 1, 0])
        elif difficulty == 3:
            self.skills.append([1, 0, 0])
        else:
            self.skills.append([0, 0, 0])  # 出错，000无法与任何订单匹配

    @property
    def action_spaces(self):
        return self.model.M

    def set_intelligence(self, level):
        self.intelligence_level = level

    # 执行订单（是接受了必要条件的检查后，确定要执行的订单）
    def process_order(self):
        value = 0
        cost = 1
        # 零智力

        if self.intelligence_level == 0 or self.intelligence_level == 1:
            if self.action == -1:
                self.order = None

        #    if self.intelligence_level == 0:
        #        # 随机选择一个满足充分约束的订单
        #        self.selected_order_id = random.choice(self.temp_actions)
        #    if self.intelligence_level == 1:
        #        print("medium")
        #        print("temp_actions:", self.temp_actions)
        #        print("_resource_lookup", self.model._resource_lookup)
        #        order_reward = {str(order_id): [] for order_id in self.temp_actions}
        #        for order_id in self.temp_actions:
        #            self.order = self.model._resource_lookup[str(order_id)]
        #            reward = self.order.bonus - sum(
        #                [abs(a - b) for (a, b) in zip(self.order.pos, self.pos)]) * self.move_cost - self.order.cost
        #            order_reward[str(self.order.unique_id)].append(reward)
        #        # 只选择自己计算出的代价最小的order，不考虑合作分配和社会整体
        #        self.selected_order_id = sorted(order_reward.items(), key=lambda o: o[1])[0][0]
        #
        #    # order = None
        #    # for temp in self.model.all_resources:
        #    #     if str(temp.unique_id) == self.selected_order_id:
        #    #         order = temp
        #
        #    self.model.actions.update({str(self.unique_id): self.model.match_order.index(self.order)})
        #
        # 高智力
        if self.intelligence_level == 2:
            if self.action is  None or self.action >= len(self.model.match_order) or self.action ==-1:
                # 这里也要换成算法1的订单集合
                self.order = None

        prob = random.uniform(0, 1)
        # 失败
        if prob >= self.failure_prob and self.order:
            order = self.model._resource_lookup[str(self.order)]
            try:
                value = order.bonus / len(order.services)
            except ZeroDivisionError:
                raise TypeError(self.model.necessary_constraint(order,[self.unique_id]),
                order.occupied, order.order_type,order.order_difficulty,
                self.difficulty, self.service_type, len(self.model.match_order),self.action, self.order,order.order_select, self.order_select)
            cost = order.cost / len(order.services) + \
                  math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(order.pos, self.pos)]))* self.move_cost
            self.state = 1  # 状态改变，开始移动
            # self.order.occupied = 1  # 任务被占用

            # 在order中保存它被处理结束的时间 !!!没有考虑除以speed无法取整的情况，speed我先设置为1了
            #order.done_time = self.model.schedule.steps + order.handling_time + sum([abs(a - b) for (a, b) in zip(order.pos, self.pos)]) / self.speed

            # 记录企业的订单处理完成时间
            self.order_end_time = self.model.schedule.steps + order.handling_time + sum(
                [abs(a - b) for (a, b) in zip(order.pos, self.pos)]) / self.speed

            # 计算移动企业位置
            # self.delta_x = self.order.pos[0] - self.pos[0]
            # self.delta_y = self.order.pos[1] - self.pos[1]

            # 由于订单处理的消耗，企业的能量值变更（企业的成本消耗发生在开始处理订单时刻）
            self.energy -= order.cost / len(order.services)
            print("选择订单转为移动", value, cost)
        else:
            self.state = 0
            self.order = None

        return value, cost

    def step(self):
        if self.energy < 0:
            print("企业破产了")
            self.done = True
        self.energy -= 20 # 假定每个step，企业的能量自动减少20

        # 如果当前时刻，订单完成
        if self.state == 2 and self.order_end_time <= self.model.schedule.steps:
            self.state = 0
            flag = 0
            #所有合作的企业都处理完了，订单才算处理完了
            order =  self.model._resource_lookup[str(self.order)]
            for a_id in order.services:
                agent = self.model._agent_lookup[a_id]
                if agent.state != 0:
                    flag = 1
                    break
            if flag == 0:
                for a_id in order.services:
                    agent = self.model._agent_lookup[a_id]
                    # 企业不是立刻获得收益，而是处理结束订单的同时获得收益
                    agent.energy += order.bonus / len(order.services)
                    print("获得收益", order.bonus ,  len(order.services))
                order.done = True

            print("_______订单处理完成_________", order.unique_id, order.pos)

        if self.state == 0:
            if self.intelligence_level !=2:
                self.process_order()

            # self.model.total_rewards += value - cost
        elif self.state == 1:
            print("开始移动")
            self.move()
            # 到达地点，转为处理订单状态
            if self.pos == self.model._resource_lookup[str(self.order)].pos:
                self.state = 2
                print(self.order_end_time , self.model.schedule.steps)

    # def move(self):
    #     print("企业移动了")
    #     if self.delta_x > 0:
    #         self.model.grid.move_agent(self, (self.pos[0] + self.speed, self.pos[1]))
    #         self.delta_x -= self.speed
    #     elif self.delta_x < 0:
    #         self.model.grid.move_agent(self, (self.pos[0] - self.speed, self.pos[1]))
    #         self.delta_x += self.speed
    #     elif self.delta_x == 0:
    #         if self.delta_y > 0:
    #             self.model.grid.move_agent(self, (self.pos[0], self.pos[1] + self.speed))
    #             self.delta_y -= self.speed
    #         elif self.delta_y < 0:
    #             self.model.grid.move_agent(self, (self.pos[0], self.pos[1] - self.speed))
    #             self.delta_y += self.speed
    def move(self):
        try:
            order =  self.model._resource_lookup[str(self.order)]
        except KeyError:
            raise KeyError(self.action, self.state, self.order)
        if math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(self.pos, order.pos)])) > self.speed:
            dx = self.pos[0] - order.pos[0]
            dy = self.pos[1] - order.pos[1]
            angle = math.atan2(dy, dx)
            # angle = int(angle * 180 / math.pi)
            move_x = self.speed * math.cos(angle)#待修改
            move_y = self.speed * math.sin(angle)
            self.shape = self.move_point(-move_x, -move_y)  # Reassign shape
            self.pos = (self.shape.x, self.shape.y)
            self.energy -= self.move_cost * self.speed
        else:
            self.shape = Point(order.pos[0], order.pos[1])
            self.pos = order.pos
            self.energy -= self.move_cost *  math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(self.pos, order.pos)]))
            # 移动每一步都有消耗
        
