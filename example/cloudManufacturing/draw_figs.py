import sys
import os
current_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append("/home/bertrand/Desktop/group-intelligence-system")
import matplotlib.pyplot as plt

from analysis.utils.draw_charts import *

def df_conact(df1, df2, df3):
    df = df1.append(df2, ignore_index = True)
    return  df.append(df3, ignore_index = True)

def draw_rate_lines(name1, name2, name3, filename):
    df1 = read_data(name1, 0)
    df2 = read_data(name2, 0)
    df3 = read_data(name3, 0)
    df4 = df_conact(df1, df2, df3)
    files_line_chart(df4, ['tax_rate'], filename)

if __name__ == "__main__":
    # line_chart("agent_num", ['service_type'], 'agent_num')
    # line_chart("avg_reward", [], "avg_reward")
    # draw_rate_lines()
    # heat_map("high_reward_heatmap", 'high_reward_heatmap')
    draw_rate_lines("low_level_low_rate_eq", "low_level_mid_rate_eq", "low_level_high_rate_eq","low_eq")
    draw_rate_lines("low_level_low_rate_prod", "low_level_mid_rate_prod", "low_level_high_rate_prod","low_prod")
    draw_rate_lines("low_level_low_rate_eqprod", "low_level_mid_rate_eqprod", "low_level_high_rate_eqprod","low_eqprod")
    draw_rate_lines("mid_level_low_rate_eq", "mid_level_mid_rate_eq", "mid_level_high_rate_eq","mid_eq")
    draw_rate_lines("mid_level_low_rate_prod", "mid_level_mid_rate_prod", "mid_level_high_rate_prod","mid_prod")
    draw_rate_lines("mid_level_low_rate_eqprod", "mid_level_mid_rate_eqprod", "mid_level_high_rate_eqprod","mid_eqprod")
