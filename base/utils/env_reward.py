# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
import math

import numpy as np


# 边际递减函数,eta 可调节
def crra(x, eta=0.5):
    return (x ** (1 - eta) - 1) / (1 - eta)


# 基尼系数
def get_gini(endowments):
    """Returns the normalized Gini index describing the distribution of endowments.

    https://en.wikipedia.org/wiki/Gini_coefficient

    Args:
        endowments (ndarray): The array of endowments for each of the agents in the
            simulated economy.

    Returns:
        Normalized Gini index for the distribution of endowments (float). A value of 1
            indicates everything belongs to 1 agent (perfect inequality), whereas a
            value of 0 indicates all agents have equal endowments (perfect equality).

    Note:
        Uses a slightly different method depending on the number of agents. For fewer
        agents (<30), uses an exact but slow method. Switches to using a much faster
        method for more agents, where both methods produce approximately equivalent
        results.
    """
    n_agents = len(endowments)

    if n_agents < 30:  # Slower. Accurate for all n.
        diff_ij = np.abs(
            endowments.reshape((n_agents, 1)) - endowments.reshape((1, n_agents))
        )
        diff = np.sum(diff_ij)
        norm = 2 * n_agents * endowments.sum(axis=0)
        unscaled_gini = diff / (norm + 1e-10)
        gini = unscaled_gini / ((n_agents - 1) / n_agents)
        return gini

    # Much faster. Slightly overestimated for low n.
    s_endows = np.sort(endowments)
    return 1 - (2 / (n_agents + 1)) * np.sum(
        np.cumsum(s_endows) / (np.sum(s_endows) + 1e-10)
    )


# 公平性
def get_equality(endowments):
    """Returns the complement of the normalized Gini index (equality = 1 - Gini).

    Args:
        endowments (ndarray): The array of endowments for each of the agents in the
            simulated economy.

    Returns:
        Normalized equality index for the distribution of endowments (float). A value
            of 0 indicates everything belongs to 1 agent (perfect inequality),
            whereas a value of 1 indicates all agents have equal endowments (perfect
            equality).
    """
    return 1 - get_gini(endowments)


# 生产力
def get_productivity(coin_endowments):
    """Returns the total coin inside the simulated economy.

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.

    Returns:
        Total coin endowment (float).
    """
    return np.sum(coin_endowments)


# 获取个体效能
# 为每个个体存一个矩阵，记录当前时间t的reward和cost
def get_self_utility(agent_matrix, agent_id, t, gamma=0.1):
    self_utility = 0
    for i in range(t + 1):
        if agent_matrix.get(str(i)).get(str(agent_id)):  # 如果历史时间段有这个agent的数据，才加
            self_utility += (gamma ** (t - i)) * (agent_matrix.get(str(i)).get(str(agent_id))[0] -
                                                  agent_matrix.get(str(i)).get(str(agent_id))[1])
    return self_utility


# 计算系统的联合效能（供应侧效能）,平等性*生产力
def get_sutility(energy, niches):
    return get_equality(energy) * get_productivity_network(niches)


# 计算系统的生产力
def get_productivity_network(niches):
    return abs((H_t(niches) - H_b(sum(niches))) / H_b(sum(niches)))


# 计算需求侧效能
# 设定一个social_matrix，用于存系统每个时间上新增的order和以及完成的order
# 设定一个agent_matrix,用于存每个agent每个时间上的reward和cost
def get_dutility(social_matrix, t, gamma=0.1):
    dutility = 0
    for i in range(int(t + 1)):
        if social_matrix.get(str(i))[0] != 0:
            dutility += (gamma ** (t - i)) * (social_matrix.get(str(i))[1] / social_matrix.get(str(i))[0])
    return dutility


# 计算服务效能（考虑供需两侧）
def get_effectiveness(energy, niches, social_matrix, t, gamma=0.1, alpha=0.5):
    return alpha * get_sutility(energy, niches) + (1 - alpha) * get_dutility(social_matrix, t, gamma)


# 当前时刻的最优价值熵Hb
def H_b(N):
    return math.log2(math.sqrt(N))


# 当前时刻的价值熵Ht
# niches:每个生态位上的数目组成的数组
def H_t(niches):
    H_t = 0
    for n_i in niches:
        p_i = n_i / sum(niches)
        if p_i > 0:
            H_t += -p_i * math.log2(p_i)
    return H_t

# 快速排序
# def quick_sort(arr):
#     if len(arr) <= 1:
#         return arr
#     pivot = arr[0]
#     left = [x for x in arr[1:] if x < pivot]
#     right = [x for x in arr[1:] if x >= pivot]
#     return quick_sort(left) + [pivot] + quick_sort(right)
