from analysis.utils.draw_charts import line_chart, heat_map
from analysis.utils.write_to_csv import write_csv_hearders

if __name__ == "__main__":
    line_chart("agent_num", ['service_type'], 'agent_num')
    heat_map("example2", 'fig2')


