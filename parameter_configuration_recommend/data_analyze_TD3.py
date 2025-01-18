# -*- coding: utf-8 -*-
"""
Train the model
"""

import os
import sys
sys.path.append('../')
import torch
import pickle
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from index_performance_predict.utils import read_data, get_dataset, df2np, np2ts, Scaler_minmax_new_gpu
from Args import args as args_r

random.seed(args_r.seed)
np.random.seed(args_r.seed)



def compute_percentage(default, current):
    """ compute metrics percentage versus default settings
    """
    # delta_rec = 100*(current[0] - default[0]) / default[0]
    # delta_ct = 100*(-current[1] + default[1]) / default[1]
    # delta_qps = 100*(current[2] - default[2]) / default[2]

    delta_rec = current[0] - default[0]
    delta_ct = -current[1] + default[1]
    delta_qps = current[2] - default[2]
    return delta_rec, delta_ct, delta_qps

# 将所有参数以及对应的性能保存成csv文件，一个参数文件与性能文件对应一个csv文件
def get_all_paras_index_performance(paras_index_performance_file, best_paras_file, best_performance_file, paras_index_performance_floder):
    current_directory = os.getcwd()  # 返回当前工作目录的路径

    with open(paras_index_performance_file, 'rb') as f:
        all_paras_index_performance = pickle.load(f)

    with open(best_paras_file, 'rb') as f:
        best_paras = pickle.load(f)

    with open(best_performance_file, 'rb') as f:
        best_performance = pickle.load(f)

    all_paras_index_performance.append({"paras": best_paras, "index_performance": best_performance})

    for i, item in enumerate(all_paras_index_performance):
        # 创建 DataFrame
        paras_df = pd.DataFrame(item['paras'], columns=['efConstruction', 'M'])
        index_perf_df = pd.DataFrame(item['index_performance'], columns=['rec', 'ct', 'qps'])

        # 合并两个 DataFrame
        full_df = pd.concat([paras_df, index_perf_df], axis=1)

        # 文件名逻辑
        if i == 0:
            filename = 'paras_index_performance_default.csv'
        elif i == len(all_paras_index_performance) - 1:
            filename = 'paras_index_performance_best.csv'
        else:
            filename = f'paras_index_performance_{i-1}.csv'

        # 保存为 CSV
        file_path = os.path.join(current_directory + '/' + paras_index_performance_floder, filename)
        full_df.to_csv(file_path, index=False)
        print(f"Saved: {filename}")

#把默认参数和性能与最佳参数和性能写入到一个csv文件中，用于后续的分析比较
def index_performance_compare(all_paras_index_performance_file, best_paras_file, best_performance_file, index_performance_improved_file):
    with open(all_paras_index_performance_file, 'rb') as f:
        all_paras_index_performance = pickle.load(f)

    with open(best_paras_file, 'rb') as f:
        best_paras = pickle.load(f)

    with open(best_performance_file, 'rb') as f:
        best_performance = pickle.load(f)

    default_paras_index_performance = all_paras_index_performance[0]
    best_paras_index_performance = {"paras": best_paras, "index_performance": best_performance}

    default_paras_df = pd.DataFrame(default_paras_index_performance['paras'], columns=['efConstruction_d', 'M_d'])
    default_index_perf_df = pd.DataFrame(default_paras_index_performance['index_performance'], columns=['rec_d', 'ct_d', 'qps_d'])
    default_full_df = pd.concat([default_paras_df, default_index_perf_df], axis=1)

    best_paras_df = pd.DataFrame(best_paras_index_performance['paras'], columns=['efConstruction_b', 'M_b'])
    best_index_perf_df = pd.DataFrame(best_paras_index_performance['index_performance'], columns=['rec_b', 'ct_b', 'qps_b'])
    best_full_df = pd.concat([best_paras_df, best_index_perf_df], axis=1)

    # df_default = pd.read_csv(paras_index_performance_default_file, sep=',', header=0)
    # df_best = pd.read_csv(paras_index_performance_best_file, sep=',', header=0)

    # df_default.columns = ['efConstruction_d', 'M_d', 'rec_d', 'ct_d', 'qps_d']
    # df_best.columns = ['efConstruction_b', 'M_b', 'rec_b', 'ct_b', 'qps_b']

    df_all = pd.concat([default_full_df, best_full_df], axis=1)

    # df_all['rec_inc'] = df_all['rec_b'] - df_all['rec_d']
    # df_all['ct_dec'] = df_all['ct_d'] - df_all['ct_b']
    # df_all['qps_inc'] = df_all['qps_b'] - df_all['qps_d']
    # df_all['performance_inc'] = df_all['rec_inc'] * 1 + df_all['ct_dec'] * 0.1 + df_all['qps_inc'] * 1

    df_all['rec_inc'] = (df_all['rec_b'] - df_all['rec_d']) / df_all['rec_d'] * 100
    df_all['ct_dec'] = (df_all['ct_d'] - df_all['ct_b']) / df_all['ct_d'] * 100
    df_all['qps_inc'] = (df_all['qps_b'] - df_all['qps_d']) / df_all['qps_d'] * 100
    # df_all['performance_inc'] = df_all['rec_inc'] * args_r.pec_rec + df_all['ct_dec'] * args_r.pec_ct + df_all['qps_inc'] * args_r.pec_qps
    df_all['performance_inc'] = df_all['rec_inc'] * 1 + df_all['ct_dec'] * 0.01 + df_all['qps_inc'] * 1

    df_all.to_csv(index_performance_improved_file, index=False)

def get_performance_increasement(index_performance_improved_file):
    df = pd.read_csv(index_performance_improved_file, sep=',', header=0)
    mean_rec_inc = df['rec_inc'].mean()
    mean_ct_dec = df['ct_dec'].mean()
    mean_qps_inc = df['qps_inc'].mean()
    mean_performance_inc = df['performance_inc'].mean()

    return mean_rec_inc, mean_ct_dec, mean_qps_inc, mean_performance_inc, df[['rec_inc', 'ct_dec', 'qps_inc', 'performance_inc']]

#计算每个episode的总奖励，然后绘制所有episode的总奖励，观察总奖励是否成上升趋势并最终趋于稳定；同时也计算每10个episode的平均总奖励，这个是为了平滑随机性导致的波动
def draw_reward(episode_score_file, sava_path):
    with open(episode_score_file, 'rb') as f:
        episode_score = pickle.load(f)

    # 计算每10个 episode 的平均总奖励
    avg_rewards = []
    num_episodes = len(episode_score)
    group_size = 10

    episodes = list(episode_score.keys())
    rewards = list(episode_score.values())
    # episodes = []
    # rewards = []
    # num1 = 0
    # num2 = 0
    # num3 = 0
    # for key in episode_score.keys():
    #     score = episode_score[key]
    #     if score > -1500:
    #         # episodes.append(key)
    #         # rewards.append(score)
    #         num1 = num1 + 1
    #     elif score > -3000:
    #         num2= num2 + 1
    #     elif score > -36600:
    #         num3 = num3 + 1
    # print(num1, num2, num3)
    
    num = 0
    for i in range(0, num_episodes, group_size):
        # 计算每组中的平均奖励
        avg_reward = np.mean(rewards[i:i + group_size])
        if avg_reward > -10000:
            avg_rewards.append(avg_reward)
            num = num + 1
    # print(num)
    # print(len(avg_rewards))
    # 绘制每个 episode 的总奖励
    plt.figure(figsize=(24, 12))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, linestyle='-', color='b')
    # for i, txt in enumerate(rewards):
    #     plt.annotate(f'{txt:.3f}', (episodes[i], rewards[i]), textcoords='offset points', xytext=(0, 10), ha='center')
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    # # 绘制每10个 episodes 的平均总奖励
    plt.subplot(1, 2, 2)
    plt.plot(range(0, num*group_size, group_size), avg_rewards, linestyle='-', color='r')
    # for i, txt in enumerate(avg_rewards):
    #     episode_mark = i * group_size + group_size / 2
    #     plt.annotate(f'{txt:.3f}', (episode_mark, rewards[i]), textcoords='offset points', xytext=(0, 10), ha='center')
    plt.title('Average Total Rewards per 10 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.grid(True)

    # 保存图形到文件
    plt.savefig(sava_path)

def draw_steps(episode_steps_file, sava_path):
    with open(episode_steps_file, 'rb') as f:
        episode_steps = pickle.load(f)

    episodes = list(episode_steps.keys())
    steps = list(episode_steps.values())

    plt.figure(figsize=(24, 12))
   
    plt.plot(episodes, steps, linestyle='-', color='b')
    # for i, txt in enumerate(rewards):
    #     plt.annotate(f'{txt:.3f}', (episodes[i], rewards[i]), textcoords='offset points', xytext=(0, 10), ha='center')
    plt.title('Total Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Stpeps')
    plt.grid(True)

    # 保存图形到文件
    plt.savefig(sava_path)

def draw_loss(episode_closs_file, episode_aloss_file, sava_path):
    with open(episode_closs_file, 'rb') as f:
        episode_closs = pickle.load(f)

    with open(episode_aloss_file, 'rb') as f:
        episode_aloss = pickle.load(f)

    episodes = list(episode_closs.keys())
    closs = list(episode_closs.values())
    aloss = list(episode_aloss.values())
    
    plt.figure(figsize=(24, 12))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, closs, linestyle='-', color='b')
    # for i, txt in enumerate(rewards):
    #     plt.annotate(f'{txt:.3f}', (episodes[i], rewards[i]), textcoords='offset points', xytext=(0, 10), ha='center')
    plt.title('Critic Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss')
    plt.grid(True)

    # # 绘制每10个 episodes 的平均总奖励
    plt.subplot(1, 2, 2)
    plt.plot(episodes, aloss, linestyle='-', color='r')
    # for i, txt in enumerate(rewards):
    #     plt.annotate(f'{txt:.3f}', (episodes[i], rewards[i]), textcoords='offset points', xytext=(0, 10), ha='center')
    plt.title('Actor Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Actor Loss')
    plt.grid(True)

    # 保存图形到文件
    plt.savefig(sava_path)


def get_grid_search_results(data_path, result_path):
    recall_thresholds = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    result_dict = {}
    result_paras_dict = {}

    df = pd.read_csv(data_path, sep=',', header=0)
    df = df[['FileName', 'efConstruction', 'M', 'efSearch', 'construction_time', 'memory', 'recall', 'search_time', 'construct_dc_counts',
             'search_dc_counts', 'average_construct_dc_counts', 'average_search_dc_counts']]

    grouped = df.groupby('FileName')

    # 遍历每个分组
    for filename, group in grouped:
        result_dict[filename] = {}
        result_paras_dict[filename] = []

        # 遍历每个 recall 阈值
        for recall_threshold in recall_thresholds:
            # 根据 recall 阈值筛选数据
            filtered_data = group[group['recall'] >= recall_threshold]

            # 如果筛选后的数据非空，获取 search_time 最小的行
            if not filtered_data.empty:
                min_search_time_row = filtered_data.loc[filtered_data['search_time'].idxmin()]

                # 提取所需的列数据
                result_data = [
                    min_search_time_row['efConstruction'],
                    min_search_time_row['M'],
                    min_search_time_row['efSearch'],
                    min_search_time_row['recall'],
                    min_search_time_row['construction_time'],
                    min_search_time_row['memory'],
                    min_search_time_row['search_time'],
                    min_search_time_row['construct_dc_counts'],
                    min_search_time_row['search_dc_counts'],
                ]

                # 存储到结果字典中
                result_dict[filename][recall_threshold] = result_data
                result_paras_dict[filename].append([recall_threshold, min_search_time_row['efConstruction'], min_search_time_row['M'], min_search_time_row['efSearch']])

    print(result_paras_dict)

    data = []

    # 遍历 result_dict，将其转换为列表形式
    for filename, recall_data in result_dict.items():
        for recall_threshold, result_data in recall_data.items():
            # 将数据添加到 data 列表
            data.append([filename, recall_threshold] + result_data)

    # 将列表转换为 DataFrame
    df = pd.DataFrame(data, columns=[
        'FileName', 'target_recall', 'efConstruction', 'M', 
        'efSearch', 'recall', 'construction_time', 'memory', 
        'search_time', 'construct_dc_counts', 'search_dc_counts'
    ])
    df.to_csv(result_path, mode='w', index=False)

    return result_dict

def get_partial_grid_search_results(data_path, result_path):
    recall_thresholds = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    result_dict = {}
    result_paras_dict = {}

    df = pd.read_csv(data_path, sep=',', header=0)
    df = df[['FileName', 'efConstruction', 'M', 'efSearch', 'construction_time', 'memory', 'recall', 'search_time', 'construct_dc_counts',
             'search_dc_counts', 'average_construct_dc_counts', 'average_search_dc_counts']]

    patial_config = [(20, 4), (20, 8), (20, 16), (40, 4), (40, 8), (40, 16), (40, 24), (40, 32), (60, 4), (60, 8), (60, 16), (60, 24), (60, 32), (60, 48), (80, 4), (80, 8), (80, 16), (80, 24), (80, 32), (80, 48), (80, 64), (80, 80), (100, 4), (100, 8), (100, 16), (100, 24), (100, 32), (100, 48), (100, 64), (100, 80), (100, 100), (140, 4), (140, 8), (140, 16), (140, 24), (140, 32), (140, 48), (140, 64), (140, 80), (140, 100), (180, 4), (180, 8), (180, 16), (180, 24), (180, 32), (180, 48), (180, 64), (180, 80), (180, 100), (220, 4), (220, 8), (220, 16), (220, 24), (220, 32), (220, 48), (220, 64), (220, 80), (220, 100), (260, 4), (260, 8), (260, 16), (260, 24), (260, 32), (260, 48), (260, 64), (260, 80), (260, 100), (300, 4), (300, 8), (300, 16), (300, 24), (300, 32), (300, 48), (300, 64), (300, 80), (300, 100), (340, 4), (340, 8), (340, 16), (340, 24), (340, 32), (340, 48), (340, 64), (340, 80), (340, 100), (380, 4), (380, 8), (380, 16), (380, 24), (380, 32), (380, 48), (380, 64), (380, 80), (380, 100), (420, 4), (420, 8), (420, 16), (420, 24), (420, 32), (420, 48), (420, 64), (420, 80), (420, 100), (460, 4), (460, 8), (460, 16), (460, 24), (460, 32), (460, 48), (460, 64), (460, 80), (460, 100), (500, 4), (500, 8), (500, 16), (500, 24), (500, 32), (500, 48), (500, 64), (500, 80), (500, 100), (560, 4), (560, 8), (560, 16), (560, 24), (560, 32), (560, 48), (560, 64), (560, 80), (560, 100), (620, 4), (620, 8), (620, 16), (620, 24), (620, 32), (620, 48), (620, 64), (620, 80), (620, 100), (680, 4), (680, 8), (680, 16), (680, 24), (680, 32), (680, 48), (680, 64), (680, 80), (680, 100), (740, 4), (740, 8), (740, 16), (740, 24), (740, 32), (740, 48), (740, 64), (740, 80), (740, 100), (800, 4), (800, 8), (800, 16), (800, 24), (800, 32), (800, 48), (800, 64), (800, 80), (800, 100)]

    patial_config_df = pd.DataFrame(patial_config, columns=['efConstruction', 'M'])

    # 通过 merge 方法保留 (efConstruction, M) 在 target_list 中的行
    patial_df = df.merge(patial_config_df, on=['efConstruction', 'M'], how='inner')

    grouped = patial_df.groupby('FileName')

    # 遍历每个分组
    for filename, group in grouped:
        result_dict[filename] = {}
        result_paras_dict[filename] = []

        # 遍历每个 recall 阈值
        for recall_threshold in recall_thresholds:
            # 根据 recall 阈值筛选数据
            filtered_data = group[group['recall'] >= recall_threshold]

            # 如果筛选后的数据非空，获取 search_time 最小的行
            if not filtered_data.empty:
                min_search_time_row = filtered_data.loc[filtered_data['search_time'].idxmin()]

                # 提取所需的列数据
                result_data = [
                    min_search_time_row['efConstruction'],
                    min_search_time_row['M'],
                    min_search_time_row['efSearch'],
                    min_search_time_row['recall'],
                    min_search_time_row['construction_time'],
                    min_search_time_row['memory'],
                    min_search_time_row['search_time'],
                    min_search_time_row['construct_dc_counts'],
                    min_search_time_row['search_dc_counts'],
                ]

                # 存储到结果字典中
                result_dict[filename][recall_threshold] = result_data
                result_paras_dict[filename].append([recall_threshold, min_search_time_row['efConstruction'], min_search_time_row['M'], min_search_time_row['efSearch']])

    print(result_paras_dict)

    data = []

    # 遍历 result_dict，将其转换为列表形式
    for filename, recall_data in result_dict.items():
        for recall_threshold, result_data in recall_data.items():
            # 将数据添加到 data 列表
            data.append([filename, recall_threshold] + result_data)

    # 将列表转换为 DataFrame
    df = pd.DataFrame(data, columns=[
        'FileName', 'target_recall', 'efConstruction', 'M', 
        'efSearch', 'recall', 'construction_time', 'memory',
        'search_time', 'construct_dc_counts', 'search_dc_counts'
    ])
    df.to_csv(result_path, mode='w', index=False)

    return result_dict


def get_drl_pr_results(best_performance_file, best_paras_file, performance_scaler, device, result_path, num_dataset, target_rec_lis, flag): #flag为0表示不是serial，为1表示是
    num_target = len(target_rec_lis)

    dataset_name = ['msong'] * num_target + ['paper'] * num_target + ['crawl'] * num_target + ['deep1'] * num_target + ['sift1'] * num_target
    target_recall = target_rec_lis * num_dataset
    data_source  = ['global'] * (num_dataset * num_target)
    
    # data_source = ['general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general', 'general']

    columns = ['efConstruction', 'M', 'efSearch', 'recall', 'average_ct_dc_counts', 'average_st_dc_counts']

    with open(best_paras_file, 'rb') as f:
        best_paras = pickle.load(f)

    with open(best_performance_file, 'rb') as f:
        best_performance = pickle.load(f)

    if not flag:
        result = np.concatenate((best_paras, best_performance), axis=1)
    else:
        result_lis = []
        
        for key in best_paras.keys():
            para = best_paras[key]
            performance = best_performance[key]

            result_lis.append(np.concatenate((para, performance), axis=1))

        result = np.vstack(result_lis)

    df = pd.DataFrame(result, columns=columns)

    df['FileName'] = dataset_name 
    df['target_recall'] = target_recall
    df['data_source'] = data_source

    df.to_csv(result_path, mode='w', index=False)

    para_dic = {}
    para_df = df[['FileName', 'target_recall', 'efConstruction', 'M', 'efSearch']]
    groups = para_df.groupby('FileName')
    for filename, group in groups:
        para_dic[filename] = group[['target_recall', 'efConstruction', 'M', 'efSearch']].values.tolist()

    print(para_dic)
    # 输出 DataFrame
    print(df)

    return result

if __name__ == '__main__':
    print('----------------执行准备工作----------------')
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

    # data_path1 = os.path.join(parent_directory, 'Data/train_data.csv')
    # data_path2 = os.path.join(parent_directory, 'Data/test_data_main.csv')
    # data_path3 = os.path.join(parent_directory, 'Data/test_data_ds_change.csv')
    data_path4 = os.path.join(parent_directory, 'Data/test_data_qw_change.csv')
    #
    # result_path1 = os.path.join(parent_directory, 'Data/experiments_results/train/index_performance_verify_train.csv')
    # result_path2 = os.path.join(parent_directory, 'Data/experiments_results/test_main/index_performance_verify_test_main.csv')
    # result_path3 = os.path.join(parent_directory, 'Data/experiments_results/test_ds_change/index_performance_verify_test_ds_change.csv')
    # result_path4 = os.path.join(parent_directory,  'Data/experiments_results/test_qw_change/index_performance_verify_test_qw_change.csv')
    #
    # result_dic1 = get_grid_search_results(data_path1, result_path1)
    # print('\n')
    # result_dic2 = get_grid_search_results(data_path2, result_path2)
    # print('\n')
    # result_dic3 = get_grid_search_results(data_path3, result_path3)
    # result_dic4 = get_grid_search_results(data_path4, result_path4)

    '''
    performance_scaler = Scaler_minmax_new_gpu(0, device)

    stor_subdir = '{}_{}_TD3'.format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)

    index_performance_improved_floder = stor_subdir + '/' + 'index_performance_improved'
    if not os.path.exists(index_performance_improved_floder):
        os.mkdir(index_performance_improved_floder)

    episode_reward_fig_floder = stor_subdir + '/' + 'episode_reward_fig'
    if not os.path.exists(episode_reward_fig_floder):
        os.mkdir(episode_reward_fig_floder)

    episode_steps_fig_floder = stor_subdir + '/' + 'episode_steps_fig'
    if not os.path.exists(episode_steps_fig_floder):
        os.mkdir(episode_steps_fig_floder)

    episode_loss_fig_floder = stor_subdir + '/' + 'episode_loss_fig'
    if not os.path.exists(episode_loss_fig_floder):
        os.mkdir(episode_loss_fig_floder)

    improved_performance_lis = []

    num_dataset = 5
    target_rec_lis = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]

    cfg_lis = []

    for ep in [4800]:  # 如果使用局部最优，那么max_step就设置为400；使用全局最优就200吧
        for mt in [5000]:
            for lr in [0.0001]:  # [0.0001, 0.00001]
                for bt in [128]:
                    for sigma in [0.2]:
                        for pec_reward in [1]:  # general方式下不能对正奖励扩大
                            for delay_time in [2]:
                                for ncst in [200]:  # general方式下这个阈值要增加
                                    cfg_lis.append((ep, mt, lr, bt, sigma, pec_reward, delay_time, ncst))

    for cfg in tqdm(cfg_lis, total=len(cfg_lis)):
        ep, mt, lr, bt, sigma, pec_reward, delay_time, ncst = cfg

        args_r.epoches = ep
        args_r.max_steps = mt
        args_r.clr = lr
        args_r.alr = lr / 10
        args_r.tau = 0.0001
        args_r.batch_size = bt
        args_r.sigma = sigma
        args_r.pec_reward = pec_reward
        args_r.delay_time = delay_time
        args_r.nochange_steps = ncst


        expr_name = 'onlys_{}_{}_{}_{}_{}_{}_{}_reward3_global_{}_{}'.format(args_r.epoches, args_r.max_steps,
                                                                                             args_r.batch_size, args_r.alr, args_r.tau,
                                                                                             args_r.sigma, args_r.delay_time, args_r.pec_reward, args_r.nochange_steps
                                                                                             )

        best_performance_file = os.path.join(current_directory + '/' + stor_subdir, 'train_best_performance_{}.pkl'.format(expr_name))
        best_paras_file = os.path.join(current_directory + '/' + stor_subdir, 'train_best_paras_{}.pkl'.format(expr_name))

        result_path = os.path.join(current_directory + '/' + stor_subdir+'/recommend_results', '{}.csv'.format(expr_name))

        result = get_drl_pr_results(best_performance_file, best_paras_file, performance_scaler, device, result_path, num_dataset, target_rec_lis, 0)

    
        episode_score_file = os.path.join(current_directory + '/' + stor_subdir, 'train_episode_score_{}.pkl'.format(expr_name))

        episode_steps_file = os.path.join(current_directory + '/' + stor_subdir,
                                        'train_episode_steps_{}.pkl'.format(expr_name))

        episode_closs_file = os.path.join(current_directory + '/' + stor_subdir,
                                        'train_episode_closs_{}.pkl'.format(expr_name))

        episode_aloss_file = os.path.join(current_directory + '/' + stor_subdir,
                                        'train_episode_aloss_{}.pkl'.format(expr_name))
  
        # print('绘制episode奖励并保存')
        episode_reawrd_fig_path = os.path.join(current_directory + '/' + episode_reward_fig_floder, 'train_episode_reward_fig_{}.png'.format(expr_name))
        draw_reward(episode_score_file, episode_reawrd_fig_path)

        # print('绘制episode结束步数并保存')
        episode_steps_fig_path = os.path.join(current_directory + '/' + episode_steps_fig_floder, 'train_episode_steps_fig_{}.png'.format(expr_name))
        draw_steps(episode_steps_file, episode_steps_fig_path)

        # print('绘制episode loss并保存')
        episode_loss_fig_path = os.path.join(current_directory + '/' + episode_loss_fig_floder, 'train_episode_loss_fig_{}.png'.format(expr_name))
        draw_loss(episode_closs_file, episode_aloss_file, episode_loss_fig_path)
        '''







