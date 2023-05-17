import matplotlib
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

# 设置
plt.style.use('ggplot')
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False

DATA_PREFIX = 'data/'
IMG_PREFIX = 'figs/'


def read_data(name, header):
    df = pd.read_csv(DATA_PREFIX + name + '.csv', header=header)
    return df


# 长型数据折线图
# 注意绘制折线图时我们保存的csv的统一数据样式为 x轴数据|y轴数据|分类数据，见example.csv
# setting = [,] 第一个参数表示对哪列数据设置hue，第二个表示对哪列数据设置style，传入的是列名
def line_chart(filename, setting, save_name):
    plt.figure()
    data = read_data(filename, 0)
    # x:横坐标名称，y：纵坐标名称，hue：对分类进行颜色区分,style:对分类进行形状区分
    if len(setting) > 0:
        if len(setting) > 1:
            sns.lineplot(data=data, x=list(data)[0], y=list(data)[1], hue=setting[0], style=setting[1])
        else:
            sns.lineplot(data=data, x=list(data)[0], y=list(data)[1], hue=setting[0])
    else:
        sns.lineplot(data=data, x=list(data)[0], y=list(data)[1])
    plt.savefig(IMG_PREFIX + save_name + '.png', dpi=600)
    # plt.show()


# 长型数据折线图
# 注意绘制折线图时我们保存的csv的统一数据样式为 x轴数据|y轴数据|分类数据，见example.csv
# setting = [,] 第一个参数表示对哪列数据设置hue，第二个表示对哪列数据设置style，传入的是列名
def files_line_chart(data, setting, save_name):
    plt.figure(figsize=(7,4))
    palet = sns.color_palette("cubehelix",3)
    if len(setting) > 0:
        if len(setting) > 1:
            sns.lineplot(data=data, x=data.keys()[0], y=data.keys()[1], hue=setting[0], style=setting[1], palette=["steelblue","darkmagenta","k"])
        else:
            sns.lineplot(data=data, x=data.keys()[0], y=data.keys()[1], hue=setting[0],palette=["steelblue","darkmagenta","k"] )
    else:
        sns.lineplot(data=data, x=data.keys()[0], y=data.keys()[1],palette=["steelblue","darkmagenta","k"])
    plt.savefig(IMG_PREFIX + save_name + '.png', dpi=1200)


# 注意绘制热力图时我们保存的csv的统一数据样式为 数据不含有x、y的行名和列名，见example2.csv
def heat_map(filename, save_name):
    plt.figure()
    data = read_data(filename, None)
    sns.heatmap(data=data,
                annot=False,  # 不显示数据
                center=0.5,  # 居中
                # xticklabels=True, yticklabels=True,  # 显示x轴和y轴
                square=True,  # 每个方格都是正方形
                cbar=True,  # 绘制颜色条
                robust=True
                )
    plt.savefig(IMG_PREFIX + save_name + '.png', dpi=600)
    # plt.show()



