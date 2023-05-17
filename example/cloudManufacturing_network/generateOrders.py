import numpy as np
import random


# 订单产生的函数
def fitting_dist(x):
    a = [314.2, 188.3, 95.56, 22.9, 48.67]
    b = [172.5, 281.5, 315.5, 228.9, 267.1]
    c = [4.645, 1.559, 10.69, 167.7, 13.1]
    fitting_model = 0
    for i in range(0, 5):
        fitting_model = fitting_model + a[i] * np.exp(-((x - b[i]) / c[i]) ** 2)
    return fitting_model


# 订单类型函数，订单的类型有A, B, C, A+B, A+C, B+C, A+B+C七种可能
def get_order_type():
    weight = {"A": 0.2, "B": 0.2, "C": 0.2, "AB": 0.1, "AC": 0.1, "BC": 0.1, "ABC": 0.1}
    return random.choices(list(weight.keys()), weights=list(weight.values()), k=1)[0]


# 订单难度，1/2/3
def order_difficulty():
    return random.randint(1, 3)


# 订单是否支持合作,0/1
def order_cooperation():
    return random.randint(0, 1)


# 订单金额
def order_money(order_type):
    if len(order_type) == 1:
        return random.uniform(100, 300)
    elif len(order_type) == 2:
        return random.uniform(400, 800)
    elif len(order_type) == 3:
        return random.uniform(1600, 3200)


# 订单处理成本
def order_process():
    return random.randint(20, 50)


# 订单位置-随机选择
def order_position(G):
    random_node = random.choice(list(G.nodes))
    return random_node


# 生成单日订单序列,单日订单序列是由列表组成，列表的每一个元素为一个订单list，
# 例：[['AC', 76, 17, 2, pos], ['C', 30, 17, 1,pos], ['AB', 48, 17, 2, pos]]
def orders_list(order_num, G):
    daily_order = []
    for i in range(0, order_num):
        daily_order.append([get_order_type(),
                            order_money(get_order_type()),
                            order_process(),
                            order_difficulty(),
                            order_position(G)])

    return daily_order


# 生成365天订单，生成的为订单列表，列表中的每一个元素为一个单日订单列表
def all_orders_list(G, rand=True):
    # 目前生成的订单数量是针对两家企业产生的，如果订单的数量觉得不够的话可以修改n的大小，让订单成倍增长
    if not rand:
        random.seed(0)
    n = 1
    all_list = []
    for i in range(1, 500):
        i = (i % 150) * 3  # 调节波峰出现速率 | 取余，每150循环一遍
        order_num = n * int(fitting_dist(i))
        all_list.append(orders_list(order_num, G))

    return all_list
