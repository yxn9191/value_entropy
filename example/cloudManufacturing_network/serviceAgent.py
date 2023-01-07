# 服务节点【提供服务者-企业】

import random

import mesa
import networkx as nx


class ServiceAgent(mesa.Agent):
    name = "Service"

    def __init__(self, unique_id,
                 model,
                 service_type,
                 difficulty,
                 speed=1,
                 energy=random.uniform(100, 200),
                 failure_prob=0.1,
                 cooperation=1,
                 move_cost=3,
                 intelligence_level=2,
                 multi_action_mode=False
                 ):
        super().__init__(unique_id, model)
        if unique_id is None:
            unique_id = 0
        self.energy = energy  # 企业的能量
        self.state = 0  # define the state of the agent
        # 0 空闲 1移动 2处理订单
        self.action = None  # 高智能的动作存在这里，但是如果agent是中低智能，对应的动作也会存在这里，因为统一存了一遍
        self.service_type = service_type  # 企业可以处理的订单类型：A,B,C
        self.difficulty = difficulty  # 可处理的订单的最大难度等级
        self.cooperation = cooperation  # 是否接受合作,接受为1，禁止为0
        self.speed = speed  # 移动速度为1
        self.move_cost = move_cost  # 移动单位距离的开销
        self.failure_prob = failure_prob  # 企业处理失败订单的概率
        self.create_time = self.model.schedule.steps  # 企业agent被创建时间
        self.cooperation_service = {}  # 企业的协作计数，{str(协作agentid):协作次数}
        self.multi_action_mode = bool(multi_action_mode)
        self.match_vector(self.service_type, self.difficulty)
        self.done = False  # 是否可以被移除
        self.temp_actions = None  # 用来保存agent在满足充分约束下的所有可以选择的订单，值是order的unique_id的数组
        self.selected_order_id = None  # 中低智能下agent最终选择的订单，存的是存的是order的unique_id

        self.intelligence_level = intelligence_level
        self.order_end_time = 0  # 企业选择的订单，结束处理的时间
        self.order_select = None  #
        self.order = None  # 企业正在处理的order，存的是order的unique_id，中低智能时是通过env的get_action()存入的，高智能的是在平台反选里存入的
        self.arrive_pos_time = 0  # 企业到达处理订单的地点的时间
        self.is_cooperating = 0 # 0不是在协作，1正在协作

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

    def action_parse(self, action):
        self.action = action

    def set_intelligence(self, level):
        self.intelligence_level = level

    def destroy(self):
        self.done = True

    def reset(self):
        self.state = 0

    # 执行订单（是接受了必要条件的检查后，确定要执行的订单），执行的结果将用于反馈给强化学习
    def process_order(self):
        value = 0
        cost = 1
        # 低和中智力
        if self.intelligence_level == 0 or self.intelligence_level == 1:
            if self.action == -1:
                self.order = None
        # 高智力
        if self.intelligence_level == 2:
            if self.action is None or self.action >= len(self.model.match_order) or self.action == -1:
                self.order = None

        prob = random.uniform(0, 1)
        # 企业有一定几率处理订单失败，如果prob >= self.failure_prob则没有失败，接着处理
        if prob >= self.failure_prob and self.order:
            order = self.model._resource_lookup[str(self.order)]
            try:
                # value是企业获得的收益，合作企业是均分收益的
                value = order.bonus / len(order.services)
                # 将企业的合作企业们，填入self.cooperation_service
                if len(order.services) > 1:
                    print("企业发生了协作！！！")
                    self.is_cooperating = 1
                    services_remove_self = order.services.remove(str(self.unique_id))

                    for aid in services_remove_self:
                        # if aid != self.unique_id:
                        if str(aid) in self.cooperation_service.keys():
                            # print("合作伙伴是否连通",
                            # nx.has_path(self.model.G,source=self.model._agent_lookup[str(aid)].pos,target=self.model._agent_lookup[str(self.unique_id)].pos))
                            self.cooperation_service[str(aid)] += 1
                            node_list = nx.shortest_path(G=self.model.G, source=self.model._agent_lookup[str(aid)].pos,
                                                         target=self.model._agent_lookup[str(self.unique_id)].pos)
                            print("source", self.model._agent_lookup[str(aid)].pos)
                            print("target", self.model._agent_lookup[str(self.unique_id)].pos)
                            print(str(aid), str(self.unique_id))
                            print("node_list", node_list)
                        else:
                            self.cooperation_service.update({str(aid): 1})
                    print("该企业的协作企业和次数", self.cooperation_service)
            except ZeroDivisionError:
                raise TypeError(self.model.necessary_constraint(order, [self.unique_id]),
                                order.occupied, order.order_type, order.order_difficulty,
                                self.difficulty, self.service_type, len(self.model.match_order), self.action,
                                self.order, order.order_select, self.order_select)
            # cost是企业为了处理订单的总消耗（处理订单本身的消耗和移动消耗）
            cost = order.cost / len(order.services) + self.distance(order) * self.move_cost
            self.state = 1  # 状态改变，开始移动

            # 记录企业的订单处理完成时间
            self.order_end_time = self.model.schedule.steps + order.handling_time + self.distance(order) / self.speed
            # 记录企业预计到达处理订单的地点的时间
            self.arrive_pos_time = int(self.model.schedule.steps + self.distance(order) / self.speed)
            print("企业预计到达处理订单的地点的时间:", self.arrive_pos_time)

            # 由于订单处理的消耗，企业的能量值变更（企业的成本消耗发生在开始处理订单时刻）
            self.energy -= order.cost / len(order.services)
            print("企业选择了订单，接下来开始移动，本次利润将为(考虑了所有消耗):", value - cost,
                  "选择的订单能提供的总收益(不考虑合作和移动消耗)为:",
                  order.bonus)
        else:
            self.state = 0
            self.order = None

        return value, cost

    def step(self):
        if self.energy < 0:
            print("企业破产了，他的id:", self.unique_id)
            self.done = True

        # 假定每个step，企业的能量自动减少5
        self.energy -= 5

        # 如果当前时刻，订单完成
        if self.state == 2 and self.order_end_time <= self.model.schedule.steps:
            self.state = 0
            self.is_cooperating = 0
            flag = 0
            # 所有合作的企业都处理完了，订单才算处理完了
            try:
                order = self.model._resource_lookup[str(self.order)]
            except KeyError:
                raise KeyError(self.order)
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
                    print("企业处理结束订单，获得收益", order.bonus, len(order.services), order.order_type)
                order.done = True

        # 中低智能的process_order()在这里调用，而高智能的在env的step里调用，手动计算了reward，要在env的step最后返回
        if self.state == 0:
            if self.intelligence_level != 2:
                self.process_order()

        elif self.state == 1:
            self.move()
            # 现在企业的pos是不变的了，不能修改，企业到达地点按照时间去推算，如果时间到了，则等于它到达了处理订单的地点
            # 移动消耗依然作数，只是所谓的移动是虚拟的，没有真实改变agent的位置
            # 计算时间，到达了地点，转为处理订单状态
            if int(self.model.schedule.steps) == int(self.arrive_pos_time):
                print("企业转为处理订单状态")
                self.state = 2

    def move(self):
        try:
            order = self.model._resource_lookup[str(self.order)]
        except KeyError:
            raise KeyError(self.action, self.state, self.order)

        # 移动每一步都有消耗
        # 只计算消耗，企业实际不移动了，pos不改变
        self.energy -= self.move_cost * self.distance(order)

    def distance(self, order):
        if nx.has_path(self.model.grid.G, source=self.pos, target=order.pos):
            return nx.shortest_path_length(self.model.grid.G, source=self.pos, target=order.pos, weight=None)
        else:
            return -1
