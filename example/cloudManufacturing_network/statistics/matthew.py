import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 准备数据
    data = pd.read_csv("matthew.csv", header=None)
    print(data)

    # 清除之前的曲线
    plt.clf()
    # 绘制平均个体效能曲线
    sns.lineplot(x=data.iloc[:, 0], y=data.iloc[:, 2], hue=data.iloc[:, 1], data=data, markers="o")
    # 添加图例
    plt.legend()
    # 添加标题和轴标签
    plt.title('Simulation Progress')
    plt.xlabel('Time step')
    plt.ylabel('Average Agent Utility')
    # 显示绘制的曲线
    plt.savefig('matthew_agent_utility.png')

    # 清除之前的曲线
    plt.clf()
    # 绘制价值熵曲线
    sns.lineplot(x=data.iloc[:, 0], y=data.iloc[:, 3], hue=data.iloc[:, 1], data=data, markers="o")
    # 添加图例
    plt.legend()
    # 添加标题和轴标签
    plt.title('Simulation Progress')
    plt.xlabel('Time step')
    plt.ylabel('Ht')
    # 显示绘制的曲线
    plt.savefig('matthew_ht.png')

    # 清除之前的曲线
    plt.clf()
    # 绘制价值熵曲线
    sns.lineplot(x=data.iloc[:, 0], y=data.iloc[:, 4], hue=data.iloc[:, 1], data=data, markers="o")
    # 添加图例
    plt.legend()
    # 添加标题和轴标签
    plt.title('Simulation Progress')
    plt.xlabel('Time step')
    plt.ylabel('Value Entropy')
    # 显示绘制的曲线
    plt.savefig('matthew_ve.png')

    # 清除之前的曲线
    plt.clf()
    # 绘制系统效能曲线
    sns.lineplot(x=data.iloc[:, 0], y=data.iloc[:, 5], hue=data.iloc[:, 1], data=data, markers="o")
    # 添加图例
    plt.legend()
    # 添加标题和轴标签
    plt.title('Simulation Progress')
    plt.xlabel('Time step')
    plt.ylabel('System Utility')
    # 显示绘制的曲线
    plt.savefig('matthew_sys_utility.png')
