# -*- coding: utf-8 -*-
# @Time    : 2024/1/15 11:13
# @Function:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
def plot_line_bond(line_df,x_name,y_name, hua_name,fig_path,order):
    '''
     round =  [list(range(10)) for _ in range(6)]
        round = [num for group in  round for num in group]
        data = {
            'Round':  round,
            'Accuracy': [random.random() for _ in range(60)],
            'Methods': ['pfgnas']*30 + ['pfgnas-o']*30
        }
        line_df = pd.DataFrame(data)
        plot_line_bond(line_df,x_name='Round',y_name='Accuracy', hua_name='Methods',fig_path=None,order=['pfgnas','pfgnas-o'])
    '''
    sns.set_theme(style="darkgrid")
    line_colors = ['#2878B5', '#9AC9DB',  '#FF8884', '#F8AC8C','#C82423'] #  ['#FFD0E9', '#B9191A', '#DBE7CA','#99BADF', '#99CDCE', '#999ACD']
    # color_palette = dict(zip(order, line_colors))
    ax = sns.lineplot(x=x_name, y=y_name,err_style = "band", hue=hua_name, hue_order=order, data=line_df, palette=line_colors)
    ax.legend(title="")
    ax.set(xlabel=x_name, ylabel=y_name)
    plt.xlabel(x_name, fontsize=20)
    plt.ylabel(y_name, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    sns.despine(left=False, right=True, top=True, bottom=False)
    plt.tight_layout()
    # plt.savefig(fig_path)
    # plt.show()

def plot_Grouped_barplots(df,beta,metric,hua_name,fig_path=None):
    '''
    Args:
        df:
        beta: str, [0.2, 10]
        metric: str, 'acc','roc_auc'
        hua_name: str, ['FedPUB', 'pfgnas-evo','pfgnas-random', 'pfgnas-one', 'pfgnas']

    Example:
    data = {
    'beta':  [random.choice(['0.2', '10']) for _ in range(100)],
    'acc': [random.random() for _ in range(100)],
    'method': [random.choice(['FedPUB', 'pfgnas-evo','pfgnas-random', 'pfgnas-one', 'pfgnas']) for _ in range(100)]
    }
    df = pd.DataFrame(data)
    plot_Grouped_barplots(df,beta,metric,method,fig_path)
    '''
    fontsize = 12
    colors = ['#2878B5', '#9AC9DB', '#FF8884', '#F8AC8C','#C82423']
    ax = sns.catplot(x=beta, y=metric, hue=hua_name, errorbar="sd", kind='bar', data=df, palette=colors, height=4,legend_out=False)

    ax.set_xlabels(r'$\beta$', fontsize=fontsize)
    ax.set_ylabels(metric, fontsize=fontsize)
    ax.set(ylim=(0.4, None))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    # plt.savefig(fig_path)
    plt.show()
    plt.clf()
    # return ax

def plot_radar_chart(data):
    '''
    Args:
        data = pd.DataFrame({
        'Indicator': ['Client 1', 'Client 2', 'Client 3', 'Global'],
        'Value Series 1': [0.22, 0.25, 0.56, 0.58],
        'Value Series 2': [0.52, 0.75, 0.23, 0.77],
})

    Returns:

    '''
    plt.figure(figsize=(8, 8))
    sns.set_style("whitegrid")
    colors = [ '#C82423','#2878B5', '#9AC9DB', '#FF8884', '#F8AC8C',]

    categories = list(data['Indicator'])
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize=20)
    ax.set_ylim([0.5, 1])
    ax.tick_params(axis='y', labelsize=18)

    for i, (column, color) in enumerate(zip(data.columns[1:], colors)):
        values_series = list(data[column])
        values_series += values_series[:1]
        ax.fill(angles, values_series, color=color, alpha=0.5, label=column)
        # ax.fill(angles, values_series, color='none', edgecolor=color, linewidth=2, linestyle='solid', label=column)

    # plt.title("Medical Indicators Radar Chart", fontsize=20)
    plt.legend(loc='upper right', title="Accuracy")

    plt.show()


if __name__ == '__main__':
    # order = ['PFGNAS', 'FL+GNAS', 'PFGNAS-Random', 'PFGNAS-evo', 'PFGNAS-One']
    # # data = {
    # #     'Round': round,
    # #     'Accuracy': [random.random() for _ in range(60)],
    # #     'Methods': ['pfgnas'] * 30 + ['pfgnas-o'] * 30
    # # }
    # round = [list(range(10)) for _ in range(6)]
    # round = [num for group in round for num in group]
    # data = {
    #     'Round': round,
    #     'Accuracy': [random.random() for _ in range(60)],
    #     'Methods': ['pfgnas'] * 30 + ['pfgnas-o'] * 30
    # }
    # line_df = pd.DataFrame(data)
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # plt.sca(axes[0])
    # plot_line_bond(line_df, x_name='Round', y_name='Accuracy', hua_name='Methods', fig_path=None,
    #                               order=['pfgnas', 'pfgnas-o'])
    # plt.title("First Subplot")
    # plt.legend().set_visible(False)
    # # Call plot_line_bond function for the second subplot
    # plt.sca(axes[1])
    # plot_line_bond(line_df, x_name='Round', y_name='Accuracy', hua_name='Methods', fig_path=None,
    #                order=['pfgnas', 'pfgnas-o'])
    # plt.title("Second Subplot")
    # plt.tight_layout()
    # plt.show()

    # data examle
    data = {
        'beta': [random.choice(['0.2', '10']) for _ in range(100)],
        'acc': [random.random() for _ in range(100)],
        'method': [random.choice(['FedPUB', 'pfgnas-evo', 'pfgnas-random', 'pfgnas-one', 'pfgnas']) for _ in range(100)]
    }
    df = pd.DataFrame(data)
    # build subfigure
    # fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    #
    # #
    # plt.sca(axes[0])
    # plot_Grouped_barplots(df1, 'beta', 'acc', 'method', fig_path='fig1.png')
    # plt.title("First Subplot")
    # plt.legend().set_visible(False)
    # plt.sca(axes[1])
    # plot_Grouped_barplots(df2, 'beta', 'acc', 'method', fig_path='fig2.png')
    # plt.title("Second Subplot")

    #
    # merge two fig
    # 创建两个子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 调用函数两次，分别在两个子图中绘制
    # plt.sca(axes[0])
    plot_Grouped_barplots(df, 'beta', 'acc', 'method', axes[0])
    # plt.sca(axes[1])
    plot_Grouped_barplots(df, 'beta', 'acc', 'method', axes[1])
    plt.close(2)
    plt.close(3)

    # 合并图例
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1, 0.9))

    # plt.show()

    #######################################################
    # data = pd.DataFrame({
    #     'Indicator': ['Client 1', 'Client 2', 'Client 3', 'Global'],
    #     'PFGNAS': [0.22, 0.25, 0.56, 0.58],
    #     'PFGNAS-O': [0.52, 0.75, 0.23, 0.77],
    # })
    data = {
        'PFGNAS': {'1': 1.0, '2': 0.7486, '3': 0.8426999999999999, 'Global': 0.8481333333333333},
        'FedPUB': {'1': 0.8387096774193549, '2': 0.8337680403408103, '3': 0.7212485394758804,
                   'Global': 0.8124207858048162},
        'FL+GraphNAS': {'3': 0.6726666666666666, '2': 0.47659999999999997, '1': 0.0, 'Global': 0.7690666666666667},
        'PFGNAS-E': {'1': 1.0, '2': 0.8667666666666666, '3': 0.8088000000000001, 'Global': 0.7812333333333333},
        'PFGNAS-O': {'1': 1.0, '2': 0.8475666666666667, '3': 0.8048333333333334, 'Global': 0.7272666666666666},

    }

    # df = pd.DataFrame(data)
    # df = df.transpose()  # 转置，使得字典的键变为DataFrame的行索引
    # df = df.rename_axis('Indicator')  # 为行索引起一个名字，例如'Method'
    df = pd.DataFrame(data).reset_index()
    df = df.rename(columns={'index': 'Indicator'})
    plot_radar_chart(df)
    ########################################################
    pass

