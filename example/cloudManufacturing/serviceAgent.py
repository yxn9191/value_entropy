# 服务节点【提供服务者-企业】
from random import randint

from base.agent import Agent


class ServiceAgent(Agent):
    name = "Service"

    def __init__(self, unique_id,
                 model,
                 service_type,
                 difficulty,
                 organization,
                 speed=3,
                 vision=10,
                 energy=randint(100, 200),
                 consumption=randint(10, 30),
                 failure_prob=0.2,
                 cooperation=1,
                 move_cost=2,
                 ):
        super().__init__(unique_id, model)
        self.energy = energy  # 企业的初始能量
        self.service_type = service_type  # 企业类型：A,B,C
        self.difficulty = difficulty  # 可处理的订单的最大难度等级
        self.cooperation = cooperation  # 是否接受合作,接受为1，禁止为0
        self.speed = speed  # 移动速度为3
        self.vision = vision  # 企业的视野半径
        self.move_cost = move_cost  # 移动单位距离的开销
        self.consumption = consumption  # 订单的成本
        self.failure_prob = failure_prob  # 企业处理失败订单的概率
        self.create_time = self.model.timestep  # 企业agent被创建时间
        self.service_satisfaction = 0  # 企业的满意度
        self.cooperation = []  # 合作处理订单的企业（初始时没有合作的企业）
        self.organization = organization  # 组织中的企业（组织中的企业协作成本低，假设最初没有在任何组织中）

    def match_vector(self, service_type, difficulty):
        self.skills[service_type] = service_type
        self.skills[difficulty] = difficulty

    @property
    def action_space(self):
        return self.vision**2


    def step(self, actions=None):
        pass
