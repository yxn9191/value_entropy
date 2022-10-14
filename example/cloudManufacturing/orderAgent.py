from random import randint

from base.agent import Agent


class OrderAgent(Agent):
    name = "Order"

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
        self.energy = energy  # 订单的利润
        self.handling_time = handling_time  # 订单预计处理时间 4-step
        self.max_duration = max_duration  # 订单的生命周期（超出后未接单会消失），10-step
        self.left_duration = self.max_duration  # 订单的剩余时间
        self.create_time = self.model.timestep  # 订单被创建时间
        self.order_satisfaction = 0  # 订单方的满意度

    def action_space(self):
        pass

    def reset(self):
        self.state = 0

    def move(self):
        self.energy = self.energy - self.consumption

    def step(self, actions=None):
        pass
