from random import randint

from base.georesource import GeoResource


class OrderAgent(GeoResource):
    name = "Order"
    collectible = True  # 是否可收集

    def __init__(self, unique_id,
                 model,
                 shape,
                 order_difficulty,
                 order_type,
                 vision=80,
                 bonus=randint(5e3, 1e4),
                 cost=randint(1e2, 2e2),
                 cooperation=1,
                 handling_time=2,
                 max_duration=10
                 ):
        super().__init__(unique_id, model, shape)
        self.vision = vision  # 订单的可看半径
        self.cost = cost  # 订单的成本
        self.order_type = order_type  # 订单类型 A\B\C的随机组合
        self.order_difficulty = int(order_difficulty)  # 订单难度 三个正整数 1\2\3 1为最简单
        self.cooperation = cooperation  # 订单是否支持合作（0禁止，1支持）
        self.bonus = bonus  # 订单的利润
        self.handling_time = handling_time  # 订单预计处理时间 4-step
        self.max_duration = max_duration  # 订单的生命周期（超出后未接单会消失），10-step
        self.time_start = self.model.schedule.steps  # 订单被创建时间
        self.left_duration = self.max_duration - (self.model.schedule.steps - self.time_start)  # 订单的剩余时间
        self.time_end = self.time_start + self.max_duration
        self.order_satisfaction = 0  # 订单方的满意度
        self.skills = [[]]  # 技能向量基本形式
        self.services = []  # 参加完成该订单的企业
        self.match_vector(self.order_type, self.order_difficulty)
        self.done = False  # 订单是否被完成
        self.done_time = None # 订单被处理结束的时间

    # 构建order的技能需求向量
    def match_vector(self, order_type, order_difficulty):
        if "A" in order_type:
            self.skills[0].append(1)
        else:
            self.skills[0].append(0)
        if "B" in order_type:
            self.skills[0].append(1)
        else:
            self.skills[0].append(0)
        if "C" in order_type:
            self.skills[0].append(1)
        else:
            self.skills[0].append(0)

        if order_difficulty == 1:
            self.skills.append([0, 0, 1])
        elif order_difficulty == 2:
            self.skills.append([0, 1, 0])
        elif order_difficulty == 3:
            self.skills.append([1, 0, 0])
        else:
            self.skills.append([0, 0, 0])  # 出错，000无法与任何企业匹配

    def destroy(self):
        self.done = True

    def step(self):
        self.left_duration -= 1
        # 判断生命周期结束还没被处理，则销毁自身
        if self.left_duration == 0:
            self.destroy()
        # 如果它被处理完成，销毁自身
        if self.done_time == self.model.schedule.steps:
            self.destroy()
