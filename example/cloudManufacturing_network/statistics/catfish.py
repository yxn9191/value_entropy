import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 准备数据
    data = pd.read_csv("catfish.csv")
    legend_map = {0.3: '0.3:0.7',0.4:"0.4:0.6",0.6:"0.6:0.4",
              0.8: '0.8:0.2',
              1: '1:0',
              0.5:'0.5:0.5'}
    # 清除之前的曲线
    plt.clf()
    # 绘制il平均个体效能曲线
    sns.lineplot(x=data.iloc[:, 0], y=data.iloc[:, 5], hue=data.iloc[:, 6].map(legend_map), data=data, markers="o")
   #  # 调节亮度和饱和度
   #  sns.palplot(sns.hls_palette(l=.7,s=.9))
    # 添加图例
    plt.legend(title = "IL:RL")
    # 添加标题和轴标签
    plt.title('Simulation Progress')
    plt.xlabel('Time step')
    plt.ylabel('IL Average Agent Utility')
    # 显示绘制的曲线
    plt.savefig('catfish_agent_utility.png')

 # 清除之前的曲线
    plt.clf()
    # 绘制系统效能曲线
    sns.lineplot(x=data.iloc[:, 0], y=data.iloc[:, 9], hue=data.iloc[:, 6].map(legend_map), data=data, markers="o")
    # 添加图例
    plt.legend(title = "IL:RL")
    # 添加标题和轴标签
    plt.title('Simulation Progress')
    plt.xlabel('Time step')
    plt.ylabel('System Utility')
    # 显示绘制的曲线
    plt.savefig('catfish_sys_utility.png')

# 清除之前的曲线
    plt.clf()
    # 绘制价值熵曲线
    sns.lineplot(x=data.iloc[:, 0], y=data.iloc[:, 8], hue=data.iloc[:, 6].map(legend_map), data=data, markers="o")
    # 添加图例
    plt.legend(title = "IL:RL")
    # 添加标题和轴标签
    plt.title('Simulation Progress')
    plt.xlabel('Time step')
    plt.ylabel('Value Entropy')
    # 显示绘制的曲线
    plt.savefig('catfish_ve.png')

# 清除之前的曲线
    plt.clf()
    # 绘制熵曲线
    sns.lineplot(x=data.iloc[:, 0], y=data.iloc[:, 7], hue=data.iloc[:, 6].map(legend_map), data=data, markers="o")
    # 添加图例
    plt.legend(title = "IL:RL")
    # 添加标题和轴标签
    plt.title('Simulation Progress')
    plt.xlabel('Time step')
    plt.ylabel('Ht')
    # 显示绘制的曲线
    plt.savefig('catfish_ht.png')