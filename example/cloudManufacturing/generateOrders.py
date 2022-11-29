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


# 订单类型函数，订单的类型有A, B, C, A+B, A+C, B+C, A+B+B七种可能
def get_order_type():
    n = random.randint(1, 3)
    type_table = ['A', 'B', 'C']
    # print("现在选择组合数量：", n)
    order_type = ''
    if n == 1:
        order_type = random.choice(type_table)
    elif n == 2:
        extra_type = random.choice(type_table)
        # print("extra_type:", extra_type)
        order_type = 'ABC'.replace(extra_type, '')
    else:
        order_type = 'ABC'
    # print(order_type)
    return order_type


# 订单难度，1/2/3
def order_difficulty():
    return random.randint(1, 3)


# 订单是否支持合作,0/1
def order_cooperation():
    return random.randint(0, 1)


# 订单金额
def order_money():
    return random.randint(30, 80)


# 订单处理成本
def order_process():
    return random.randint(15, 30)


# 订单位置
def order_position():
    random_point = (0, 0)

    return random_point


# 生成单日订单序列,单日订单序列是由列表组成，列表的每一个元素为一个订单list，
# 例：[['AC', 76, 17, 2, (0, 0)], ['C', 30, 17, 1, (0, 0)], ['AB', 48, 17, 2, (0, 0)]]
def orders_list(order_num):
    daily_order = []
    for i in range(0, order_num):
        daily_order.append([get_order_type(),
                            order_money(),
                            order_process(),
                            order_difficulty(),
                            order_position()])

    print(daily_order)
    return daily_order


# 生成365天订单，生成的为订单列表，列表中的每一个元素为一个单日订单列表
def all_orders_list():
    all_list = []
    for i in range(1, 366):
        order_num = int(fitting_dist(i))
        all_list.append(orders_list(order_num))
    # print(all_list)
    return all_list

