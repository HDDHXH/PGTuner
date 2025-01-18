import os
import sys
sys.path.append('../')
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



current_directory = os.getcwd()  # 返回当前工作目录的路径
parent_directory = os.path.dirname(current_directory)

def darw_only_grid_drl():
    metrics1 = ['ct_dec', 'st_dec', 'qps_inc']
    metrics2 = ['cdc_dec', 'sdc_dec']
    colors = ['blue', 'green', 'red']

    #mode = 'train'
    mode = 'test_main'
    mode_grid = 'grid'

    para_path = os.path.join(parent_directory, 'Data/experiments_results/{}/index_performance_verify_{}_compare.csv'.format(mode, mode))
    mode_compare_lis = ['grid', 'test_8000_128_alone']

    for mode_compare in tqdm(mode_compare_lis, total = len(mode_compare_lis)):
        root_path = os.path.join(parent_directory, 'Data/experiments_results/{}/{}'.format(mode, mode_compare))

        if not os.path.exists(root_path):
            os.makedirs(root_path)

        df = pd.read_csv(para_path, sep=',', header=0)
        df = df[df['data_source'].isin([mode_grid , mode_compare])]
        df['qps'] = 1/  df['search_time']

        groups = df.groupby('FileName')

        for filename, group in groups:
            group_train = group[group['data_source'] == mode_grid ]
            group_recommend = group[group['data_source'] == mode_compare]

            group_recommend.columns = ['FileName','efConstruction_r','M_r','pr_efSearch_r','real_efSearch_r','construction_time_r','memory_r','target_recall','real_recall_r',
                                        'search_time_r','construct_dc_counts_r','search_dc_counts_r','paras_search_time_r','data_source_r', 'qps_r']

            all_group = pd.merge(group_train, group_recommend, on=['FileName', 'target_recall'], how='right')

            all_group['ct_dec'] = (all_group['construction_time'] - all_group['construction_time_r']) / all_group['construction_time'] * 100
            all_group['st_dec'] = (all_group['search_time'] - all_group['search_time_r']) / all_group['search_time'] * 100
            all_group['cdc_dec'] = (all_group['construct_dc_counts'] - all_group['construct_dc_counts_r']) / all_group['construct_dc_counts'] * 100
            all_group['sdc_dec'] = (all_group['search_dc_counts'] - all_group['search_dc_counts_r']) / all_group['search_dc_counts'] * 100

            all_group['qps_inc'] = (all_group['qps_r'] - all_group['qps']) / all_group['qps'] * 100

            # all_group['ot_dec'] = all_group['ct_dec'] * lamb +  all_group['st_dec']  #另外一种计算总的时间下降百分比的方式
            # all_group['odc_dec'] = all_group['cdc_dec'] * lamb +  all_group['sdc_dec']

            file_path = os.path.join(root_path, 'performance_improvment_{}_{}.csv'.format(filename, mode_grid))
            all_group.to_csv(file_path, index=False)

            save_path = os.path.join(root_path, 'performance_improvment_{}_{}.png'.format(filename, mode_grid))

            # 创建子图
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))

            for i in range(len(metrics1)):
                metric = metrics1[i]
                color = colors[i]

                axs[0].plot(all_group['target_recall'], all_group[metric], marker='o', linestyle='-', color=color, label=metric)
            axs[0].set_xlabel('target_recall')
            axs[0].set_ylabel('performance_improvement')
            axs[0].legend()
            axs[0].grid(True)

            for i in range(len(metrics2)):
                metric = metrics2[i]
                color = colors[i]
                axs[1].plot(all_group['target_recall'], all_group[metric], marker='o', linestyle='-', color=color, label=metric)
            axs[1].set_xlabel('target_recall')
            axs[1].set_ylabel('performance_improvement')
            axs[1].legend()
            axs[1].grid(True)

            # 确保刻度值显示完整
            for ax in axs:
                ax.set_xticks(ax.get_xticks())  # 设置 x 轴刻度
                ax.set_yticks(ax.get_yticks())  # 设置 y 轴刻度
                
                # 强制显示刻度标签
                ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

                # 防止刻度标签被截断
                # ax.set_xticklabels(ax.get_xticks(), rotation=0)
                # ax.set_yticklabels(ax.get_yticks(), rotation=0)

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()


        for filename, group in groups:
            # 在每个 FileName 分组内再按 data_source 分组
            save_path = os.path.join(root_path, 'performance_{}_{}.png'.format(filename, mode_grid))

            data_source_grouped = group.groupby('data_source')
            
            # 创建一个图形对象和4个子图
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'Analysis for {filename}', fontsize=16)

            # 定义四个要绘制的列名和对应的子图位置
            columns = ['construction_time', 'search_time', 'qps', 'construct_dc_counts', 'search_dc_counts']
            positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

            # 遍历每个列和子图位置
            for col, pos in zip(columns, positions):
                ax = axs[pos[0], pos[1]]
                
                # 遍历每个 data_source 分组
                for data_source, data in data_source_grouped:
                    ax.plot(data['target_recall'], data[col], marker='o', label=data_source)
                
                ax.set_title(col)
                ax.set_xlabel('Target Recall')
                ax.set_ylabel(col)
                ax.legend(title='Data Source')
                ax.grid(True)

                ax.set_xticks(ax.get_xticks())  # 设置 x 轴刻度
                ax.set_yticks(ax.get_yticks())  # 设置 y 轴刻度
                
                # 强制显示刻度标签
                ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

                # 防止刻度标签被截断
                # ax.set_xticklabels(ax.get_xticks(), rotation=0)
                # ax.set_yticklabels(ax.get_yticks(), rotation=0)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # 保存图片
            plt.savefig(save_path)
            plt.close()

darw_only_grid_drl()

