from analysis.utils.draw_charts import line_chart, heat_map

if __name__ == "__main__":
    line_chart("agent_num", ['service_type'], 'agent_num')
    line_chart("avg_reward", [], "avg_reward")
    # heat_map("example2", 'fig2')
