from analysis.utils.draw_charts import *

def draw_rate_lines():
    df1 = read_data("low_rate", 0)
    df2 = read_data("mid_rate", 0)
    df3 = read_data("high_rate", 0)
    df = df1.append(df2, ignore_index = True)
    df4 = df.append(df3, ignore_index = True)
    files_line_chart(df4, ['tax_rate'], 'tax')

if __name__ == "__main__":
    line_chart("agent_num", ['service_type'], 'agent_num')
    line_chart("avg_reward", [], "avg_reward")
    draw_rate_lines()
    # heat_map("example2", 'fig2')
