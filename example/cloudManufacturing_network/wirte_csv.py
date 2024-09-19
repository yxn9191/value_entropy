import os

from analysis.utils.write_to_csv import write_csv_hearders, write_csv_rows


# def utility_csv(model, avg_agent_utility,system_utility, file_name):
#     file_path = os.path.join("example","cloudManufacturing_network","data", file_name)
#     if model.schedule.steps == 1:
#         write_csv_hearders(file_path, ["time_step", "avg_agent_utility","system_utility"])
#     write_csv_rows(file_path, [[model.schedule.steps, avg_agent_utility,system_utility]])

# 写入平均个体效能、价值熵、系统效能的csv
def diff_csv(model, variable, avg_agent_utility, ht,value_entropy, system_utility, file_name):
    # file_path = os.path.join("example", "cloudManufacturing_network", "data", file_name)
    write_csv_rows(file_name,
                   [[model.schedule.steps, variable, avg_agent_utility,ht,value_entropy, system_utility]])


# def zhineng_box_csv(model, variable, agent_zhineng, file_name):
#     file_path = os.path.join("example", "cloudManufacturing_network", "data", file_name)
#     write_csv_rows(file_path, [[model.schedule.steps, variable, agent_zhineng]])

# # 写入个体智能箱线图的csv
# def agent_zhineng_csv(model, act_rules, topology, agent_zhineng, file_name):
#     # file_path = os.path.join("example", "cloudManufacturing_network", "data", file_name)
#     # if model.schedule.steps == 1:
#     #     write_csv_hearders(file_name, ["time_step", "act_rules", "topology",
#     #                                    "agent_zhineng"])
#     write_csv_rows(file_name,
#                    [[model.schedule.steps, act_rules, topology, agent_zhineng]])


# variable 是目前系统中拥有的智能等级的平均个体效能的数组
def agent_utility_type_csv(model, avg_agent_utility, file_name, type0, type1, type2, type3,ratio_imitate,ht,value_entropy,sys_utility):
    write_csv_rows(file_name,
                   [[model.schedule.steps, avg_agent_utility, type0, type1, type2, type3,ratio_imitate,ht,value_entropy,sys_utility]])


def total_csv(model, act_rules, topology, avg_agent_utility,ht,value_entropy, system_utility, agent_zhineng, file_name):
    # file_path = os.path.join("example", "cloudManufacturing_network", "data", file_name)
    if model.schedule.steps == 1:
        write_csv_hearders(file_name, ["time_step", "act_rules", "topology", "avg_agent_utility", "ht", "value_entropy","system_utility",
                                       "agent_zhineng"])
    write_csv_rows(file_name,
                   [[model.schedule.steps, act_rules, topology, avg_agent_utility, ht,value_entropy,system_utility, agent_zhineng]])
