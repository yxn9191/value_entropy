# 服务节点【提供服务者-企业】
from random import randint

from base.agent import Agent
import random
import math


class ServiceAgent(Agent):
    name = "Service"

    def __init__(self, unique_id,
                 model,
                 service_type,
                 difficulty,
                 organization,
                 speed=3,
                 energy=randint(100, 200),
                 consumption=randint(10, 30),
                 failure_prob=0.2,
                 cooperation=1,
                 move_cost=2,
                 intelligence_level=2
                 ):
        super().__init__(unique_id, model)
        self.energy = energy  # 企业的能量
        self.service_type = service_type  # 企业可以处理的订单类型：A,B,C,AB,AC,BC,ABC
        self.difficulty = difficulty  # 可处理的订单的最大难度等级
        self.cooperation = cooperation  # 是否接受合作,接受为1，禁止为0
        self.speed = speed  # 移动速度为3
        self.move_cost = move_cost  # 移动单位距离的开销
        self.consumption = consumption  # 订单的成本
        self.failure_prob = failure_prob  # 企业处理失败订单的概率
        self.create_time = self.model.timestep  # 企业agent被创建时间
        self.service_satisfaction = 0  # 企业的满意度
        self.cooperation = cooperation  # 是否合作
        self.cooperation_service = []
        self.organization = organization  # 组织中的企业（组织中的企业协作成本低，假设最初没有在任何组织中）
        self.match_vector(self.service_type, self.difficulty)

        self.intelligence_level = intelligence_level

    def match_vector(self, service_type, difficulty):
        if service_type == "A":
            self.skills = [[1, 0, 0]]
        elif service_type == "B":
            self.skills = [[0, 1, 0]]
        elif service_type == "C":
            self.skills = [[0, 0, 1]]
        elif service_type == "AB":
            self.skills = [[1, 1, 0]]
        elif service_type == "AC":
            self.skills = [[1, 0, 1]]
        elif service_type == "BC":
            self.skills = [[0, 1, 1]]
        elif service_type == "ABC":
            self.skills = [[1, 1, 1]]
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
    def action_space(self):
        return self.model.num_order

    # 选择订单
    def select_order(self):
        # 零智力
        if self.intelligence_level == 0:
            pass
        # 低智力
        elif self.intelligence_level == 1:
            pass
        # 高智力
        elif self.intelligence_level == 2:
            value = 0
            cost = 1
            if self.action != -1:
                #这里也要换成算法1的订单集合
                order = self.model.all_order[self.action]
                prob = random.uniform(0, 1)
                #失败
                if prob >= self.failure_prob:
                    order_len = len(order.order_type)
                    value = order.bonus/order_len
                    cost = order.cost/order_len + sum([abs(a - b) for (a, b) in zip(order.pos, self.pos)])
                    self.state = 1 #状态改变，开始移动

            return value, cost






    # 返回当前选择的任务后的收益和消耗，如果没有选择则返回0，1
    def step(self):
        self.energy -= 10  # 假定每个step，企业的能量自动减少10
        if self.state == 0:
            self.select_order()
        elif self.state == 1:
            self.move()


    # 判断企业是否破产
    def judge_destroy(self):
        if self.energy < 0:
            return 1
        else:
            return 0
