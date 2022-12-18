import numpy as np
import random

 

# 订单产生的函数
def fitting_dist(x):
    a = [314.2, 188.3, 95.56, 22.9, 48.67]
    b = [172.5, 281.5, 315.5, 228.9, 267.1]
    c = [4.645, 1.559, 10.69, 167.7, 13.1]
    fitting_model = 0
    for i in range(0, 5):
        # print(i)
        fitting_model = fitting_model + a[i]*np.exp(-((x - b[i])/c[i])**2)
    return fitting_model


# 订单类型函数，订单的类型有A, B, C, A+B, A+C, B+C, A+B+C七种可能
def get_order_type():
    # n = random.randint(1, 3)
    # type_table = ['A', 'B', 'C']
    # # print("现在选择组合数量：", n)
    # order_type = ''
    # if n == 1:
    #     order_type = random.choice(type_table)
    # elif n == 2:
    #     extra_type = random.choice(type_table)
    #     # print("extra_type:", extra_type)
    #     order_type = 'ABC'.replace(extra_type, '')
    # else:
    #     order_type = 'ABC'
    # print(order_type)
    #return order_type
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
    if len(list(order_type))== 1:
        return random.uniform(5e2, 1e3)
    elif len(list(order_type)) == 2:
        return random.uniform(1e3, 2e3)
    elif len(list(order_type)) == 3:
        return random.uniform(2e3, 3e3)


# 订单处理成本
def order_process():
    return random.uniform(50, 1e2)


# 订单位置
def order_position(region):
    shape = region.random_point
    return (shape.x, shape.y)


# 生成单日订单序列,单日订单序列是由列表组成，列表的每一个元素为一个订单list，
# 例：[['AC', 76, 17, 2, (0, 0)], ['C', 30, 17, 1, (0, 0)], ['AB', 48, 17, 2, (0, 0)]]
def orders_list(order_num, region):
    daily_order = []
    for i in range(0, order_num):
        order_type = get_order_type()
        daily_order.append([order_type,
                            order_money(order_type),
                            order_process(),
                            order_difficulty(),
                            order_position(region)])

    return daily_order


# 生成365天订单，生成的为订单列表，列表中的每一个元素为一个单日订单列表
def all_orders_list(region, rand = True):
    # 目前生成的订单数量是针对两家企业产生的，如果订单的数量觉得不够的话可以修改n的大小，让订单成倍增长
    if not rand :
        random.seed(0)
    n = 1
    all_list = []
    for i in range(1, 366):
        order_num = n * int(fitting_dist(i))
        all_list.append(orders_list(order_num, region))

    return all_list


