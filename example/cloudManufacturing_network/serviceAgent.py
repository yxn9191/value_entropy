# 服务节点【提供服务者-企业】

import random

import mesa
import networkx as nx

from copy import deepcopy


class ServiceAgent(mesa.Agent):
    name = "Service"

    def __init__(self, unique_id,
                 model,
                 service_type,
                 difficulty,
                 speed=1,
                 energy=random.uniform(100, 200),
                 failure_prob=0.1,
                 imitate_pro=0.5,
                 cooperation=1,
                 move_cost=30,
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
        self.is_cooperating = 0  # 0不是在协作，1正在协作
        self.last_orders = []  # 企业曾经处理过的订单
        self.imitate_pro = imitate_pro  # 模仿学习发生进化的概率值[0-1]，初始每个企业不一样
        self.daily_cost = 10  # 每天自动减少的能量
        self.daily_cost_total = 0  # 记录企业移动过程中，没有减的daily_cost，放到最后一起减

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
        # 每个step，有self.daily_cost,如果执行订单失败，这里应该返回daily_cost
        cost = self.daily_cost  # 为了防止计算强化学习奖励函数（rew = alpha * self.finish_orders / len(self.new_orders) + (1 -
        # alpha) * value / cost）的时候，cost为0

        # 随机、固定规则、模仿学习的Agent
        if self.intelligence_level == 0 or self.intelligence_level == 1 or self.intelligence_level == 3:
            if self.action == -1:
                self.order = None
        # 强化学习的Agent
        if self.intelligence_level == 2:
            if self.action is None or self.action >= len(self.model.match_order) or self.action == -1:
                self.order = None

        # prob = random.uniform(0, 1)
        prob = 1  # 先不考虑会失败
        # 企业有一定几率处理订单失败，如果prob >= self.failure_prob则没有失败，接着处理
        if prob >= self.failure_prob and self.order:
            order = self.model._resource_lookup[str(self.order)]
            order.occupied = 1
            try:
                # value是企业获得的收益，合作企业是均分收益的
                value = order.bonus / len(order.services)
                # 将企业的合作企业们，填入self.cooperation_service
                if len(order.services) > 1:
                    self.is_cooperating = 1
                    # print("order.services", order.services)
                    services_remove_self = deepcopy(order.services)
                    services_remove_self.remove(str(self.unique_id))
                    print("企业{}发生了协作，伙伴为".format(self.unique_id), services_remove_self)
                    for aid in services_remove_self:
                        # if aid != self.unique_id:
                        if str(aid) in self.cooperation_service.keys():
                            # print("合作伙伴是否连通",
                            # nx.has_path(self.model.G,source=self.model._agent_lookup[str(aid)].pos,target=self.model._agent_lookup[str(self.unique_id)].pos))
                            self.cooperation_service[str(aid)] += 1
                            node_list = nx.shortest_path(G=self.model.G, source=self.model._agent_lookup[str(aid)].pos,
                                                         target=self.model._agent_lookup[str(self.unique_id)].pos)
                            # print("source", self.model._agent_lookup[str(aid)].pos)
                            # print("target", self.model._agent_lookup[str(self.unique_id)].pos)
                            # print(str(aid), str(self.unique_id))
                            # print("node_list", node_list)
                        else:
                            self.cooperation_service.update({str(aid): 1})
                    # print("该企业的协作企业和次数", self.cooperation_service)
            except ZeroDivisionError:
                raise TypeError(self.model.necessary_constraint(order, [self.unique_id]),
                                order.occupied, order.order_type, order.order_difficulty,
                                self.difficulty, self.service_type, len(self.model.match_order), self.action,
                                self.order, order.order_select, self.order_select)
            # cost是企业为了处理订单的总消耗（处理订单本身的消耗和移动消耗）
            cost_order = order.cost / len(order.services) + self.distance(order) * self.move_cost
            self.state = 1  # 状态改变，开始移动

            # 记录企业预计到达处理订单的地点的时间
            self.arrive_pos_time = int(self.model.schedule.steps + self.distance(order) / self.speed)
            # 记录企业的订单处理完成时间
            self.order_end_time = int(self.arrive_pos_time + order.handling_time)
            # print("企业预计到达处理订单的地点的时间:", self.arrive_pos_time)

            # 计算这期间，随着时间流逝，企业产生的cost(日常运营cost)，假设每个step能量自动减少daily_cost
            self.daily_cost_total = int(self.order_end_time - self.model.schedule.steps) * self.daily_cost
            cost = cost_order + self.daily_cost_total
            # 由于订单处理的消耗，企业的能量值变更（企业的成本消耗发生在开始处理订单时刻）。其他的移动消耗和随时间流逝的日常消耗，在别的函数减
            self.energy -= order.cost / len(order.services)
            print("企业{}选择了订单{}，接下来开始移动，本次利润将为{},其中订单引起的消耗为{},日常消耗为{}".format(
                self.unique_id, self.order,
                value - cost, cost_order,
                self.daily_cost_total))
        else:
            self.state = 0
            self.order = None

        # 中低智能，即使当前时刻没有成功处理订单，也会在这个里面记录value=0和cost=5，然鹅，高智能只有在matchagent里，才会进入该process_order函数，才会被记录
        # 所以我要在agent 的step里，判断它如果是高智能，且没有在match agent里，也应该有cost[!!我又没有考虑这个了，没有被match相当于不在系统中！！]
        # 更新矩阵，这个矩阵存了agent在该时刻执行订单的cost和value（其实这样获得value在统计的时候记录在执行订单时刻了，而真正减是在执行订单结束后）
        self.model.agent_matrix.get(str(self.model.schedule.steps)).update({str(self.unique_id): [value, cost]})
        # 现在计算个体效能时，伽马=1，平等看待所有时刻收益，其实也无所谓。
        return value, cost

    def step(self):
        # 企业可能发生，在还没移动到执行订单时，就快破产了，对此的处理为，一旦他有需要处理的订单，能量就不再减少
        # 于此同时，移动消耗也发生在获得利益时刻
        if self.energy < 0:
            print("企业破产了，他的id:", self.unique_id)
            self.done = True

        # 假定每个step，企业的能量自动减少daily_cost。一旦他有需要处理的订单，能量就不再减少；和移动消耗一起放到最后再减
        if not self.order:
            self.energy -= self.daily_cost

        # 采用模仿学习的企业，在每个step发生进化
        if self.intelligence_level == 3:
            self.evolve_imitate()

        # 中低智能的process_order()在这里调用，而高智能的在env的step里调用，手动计算了reward，要在env的step最后返回
        if self.state == 0:
            if self.intelligence_level != 2:
                value, cost = self.process_order()
                # 更新本轮中低智能agent的agent_matrix
                self.model.agent_matrix.get(str(self.model.schedule.steps)).update({str(self.unique_id): [value, cost]})

        # 先不考虑没有被匹配的订单，没有被匹配，相当于不在系统中；参与匹配了但没中的才应该被考虑
        # if self.intelligence_level == 2 and self not in self.model.match_agent:
        #     self.model.agent_matrix.get(str(self.model.schedule.steps)).update(
        #         {str(self.unique_id): [0, self.daliy_cost]})

        elif self.state == 1:
            self.move()
            # 现在企业的pos是不变的了，不能修改，企业到达地点按照时间去推算，如果时间到了，则等于它到达了处理订单的地点
            # 移动消耗依然作数，只是所谓的移动是虚拟的，没有真实改变agent的位置
            # 计算时间，到达了地点，转为处理订单状态
            if int(self.model.schedule.steps) == int(self.arrive_pos_time):
                print("企业{}转为处理订单{}状态".format(self.unique_id, self.order))
                self.state = 2

        # 如果当前时刻，订单完成
        if self.state == 2 and (int(self.order_end_time) == int(self.model.schedule.steps)):
            try:
                order = self.model._resource_lookup[str(self.order)]
            except KeyError:
                raise KeyError(self.order)
            self.state = 0
            self.is_cooperating = 0
            self.last_orders.append(self.order)
            # print(self.last_orders)
            self.order = None
            # flag = 0
            # # 所有合作的企业都处理完了，订单才能被销毁，所有企业才能一起获得收益
            # for a_id in order.services:
            #     agent = self.model._agent_lookup[str(a_id)]
            #     # print("agent.state", agent.state, type(agent.state))
            #     # 因为内部调度不是并行的，是randomactive，可能存在一个agent都执行完成订单，而另一个订单还处于没开始执行的情况，这样会导致仅仅通过state
            #     # 来判断所有企业都执行完成了订单这件事，是错误的，必须加其他的约束
            #     # 应该记录协作企业曾经处理过的订单序列，如果不含当前企业正在处理的订单号，则没有全部处理完成
            #     if self.order not in agent.last_orders:
            #         print("agent.last_orders", agent.last_orders)
            #         flag = 1
            #         break
            # if flag == 0:
            #     for a_id in order.services:
            #         agent = self.model._agent_lookup[str(a_id)]
            # 企业不是立刻获得收益，而是处理结束订单的同时获得收益
            self.energy += order.bonus / len(order.services)
            # 为了防止企业提前死去，移动消耗最后再减
            self.energy -= self.move_cost * self.distance(order)
            # 这期间的日常消耗也是在这里减
            self.energy -= self.daily_cost_total
            print("企业{}处理结束订单{}，已经分得订单利益{},过程中移动消耗为{},处理订单消耗为{}".format(
                self.unique_id, order.unique_id, order.bonus / len(order.services),
                                                 self.move_cost * self.distance(order),
                                                 order.cost / len(order.services)))
            order.process_times += 1
            #     order.done = True
            # else:
            #     print("企业{}等待其他企业处理订单".format(self.unique_id))

    def move(self):
        try:
            order = self.model._resource_lookup[str(self.order)]
            print("企业{}正在向订单{}移动".format(self.unique_id, self.order))
        except KeyError:
            raise KeyError(self.action, self.state, self.order, self.unique_id)
        # =》为了防止企业提前死去，移动消耗最后再减，在step里

    # 模仿学习的进化行为
    def evolve_imitate(self):
        neighbors = []
        for neighbor in list(self.model.grid.G[self.pos]):
            for agent in self.model.all_agents:
                if agent.pos == neighbor:
                    neighbors.append(agent)
        print("将要发生进化的企业的邻居是：", neighbors)
        max_value = 0
        max_value_imitate_pro = 0
        # 遍历周围邻居的energy
        for neighbor in neighbors:
            if neighbor.energy >= max_value and neighbor.imitate_pro:
                max_value = neighbor.energy
                max_value_imitate_pro = neighbor.imitate_pro
        if max_value - self.energy >= 2 * self.energy:
            self.imitate_pro = max_value_imitate_pro
            print("企业{}通过模仿学习发生了进化，将概率值改为{}".format(self.unique_id, self.imitate_pro))

    def distance(self, order):
        if nx.has_path(self.model.grid.G, source=self.pos, target=order.pos):
            return nx.shortest_path_length(self.model.grid.G, source=self.pos, target=order.pos, weight=None)
        else:
            return -1
