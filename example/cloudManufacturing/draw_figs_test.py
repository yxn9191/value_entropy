from analysis.utils.draw_charts import line_chart, heat_map

if __name__ == "__main__":
    line_chart("example", ['agent_type', 'agent_type'], 'fig1')
    heat_map("example2", 'fig2')