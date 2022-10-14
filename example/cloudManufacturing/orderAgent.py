from random import randint

from base.agent import Agent
from base.resource import Resource


class OrderAgent(Resource):
    name = "Order"
    collectible = True  # 是否可收集

    def __init__(self, unique_id,
                 model,
                 order_difficulty,
                 order_type,
                 speed=0,
                 vision=6,
                 energy=randint(30, 80),
                 consumption=randint(10, 30),
                 failure_prob=0,
                 cooperation=1,
                 handling_time=4,
                 max_duration=10
                 ):
        super().__init__(unique_id, model)
        self.speed = speed  # 移动速度为0  代表不可移动
        self.vision = vision  # 订单的可看半径
        self.consumption = consumption  # 订单的成本
        self.failure_prob = failure_prob  # 订单不涉及会使任务失败，我们场景中企业可能会使任务失败，所以这里订单=0，企业设置为某个值
        self.order_type = order_type  # 订单类型 A\B\C
        self.order_difficulty = int(order_difficulty)  # 订单难度 三个正整数 1\2\3 1为最简单
        self.cooperation = cooperation  # 订单是否支持合作（0禁止，1支持）
        self.energy = energy  # 订单的价值
        self.handling_time = handling_time  # 订单预计处理时间 4-step
        self.max_duration = max_duration  # 订单的生命周期（超出后未接单会消失），10-step
        self.left_duration = self.max_duration  # 订单的剩余时间
        self.create_time = self.model.timestep  # 订单被创建时间
        self.order_satisfaction = 0  # 订单方的满意度

    def match_vector(self, order_type, order_difficulty):
        if order_type == "A":
            self.skills = [[1, 0, 0]]
        elif order_type == "B":
            self.skills = [[0, 1, 0]]
        elif order_type == "C":
            self.skills = [[0, 0, 1]]
        else:
            self.skills = [[0, 0, 0]]  # 出错，000无法与任何企业匹配

        if order_difficulty == 1:
            self.skills.append([0, 0, 1])
        elif order_difficulty == 2:
            self.skills.append([0, 1, 0])
        elif order_difficulty == 3:
            self.skills.append([1, 0, 0])
        else:
            self.skills.append([0, 0, 0])  # 出错，000无法与任何企业匹配

    def step(self, actions=None):
        pass
