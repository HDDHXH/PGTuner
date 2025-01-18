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
def get_all_paras_index_performance(paras_index_performance_file, best_paras_file, best_performance_file,
                                    paras_index_performance_floder):
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
            filename = f'paras_index_performance_{i - 1}.csv'

        # 保存为 CSV
        file_path = os.path.join(current_directory + '/' + paras_index_performance_floder, filename)
        full_df.to_csv(file_path, index=False)
        print(f"Saved: {filename}")


# 把默认参数和性能与最佳参数和性能写入到一个csv文件中，用于后续的分析比较
def index_performance_compare(all_paras_index_performance_file, best_paras_file, best_performance_file,
                              index_performance_improved_file):
    with open(all_paras_index_performance_file, 'rb') as f:
        all_paras_index_performance = pickle.load(f)

    with open(best_paras_file, 'rb') as f:
        best_paras = pickle.load(f)

    with open(best_performance_file, 'rb') as f:
        best_performance = pickle.load(f)

    default_paras_index_performance = all_paras_index_performance[0]
    best_paras_index_performance = {"paras": best_paras, "index_performance": best_performance}

    default_paras_df = pd.DataFrame(default_paras_index_performance['paras'], columns=['efConstruction_d', 'M_d'])
    default_index_perf_df = pd.DataFrame(default_paras_index_performance['index_performance'],
                                         columns=['rec_d', 'ct_d', 'qps_d'])
    default_full_df = pd.concat([default_paras_df, default_index_perf_df], axis=1)

    best_paras_df = pd.DataFrame(best_paras_index_performance['paras'], columns=['efConstruction_b', 'M_b'])
    best_index_perf_df = pd.DataFrame(best_paras_index_performance['index_performance'],
                                      columns=['rec_b', 'ct_b', 'qps_b'])
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

    return mean_rec_inc, mean_ct_dec, mean_qps_inc, mean_performance_inc, df[
        ['rec_inc', 'ct_dec', 'qps_inc', 'performance_inc']]


# 计算每个episode的总奖励，然后绘制所有episode的总奖励，观察总奖励是否成上升趋势并最终趋于稳定；同时也计算每10个episode的平均总奖励，这个是为了平滑随机性导致的波动
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
    plt.plot(range(0, num * group_size, group_size), avg_rewards, linestyle='-', color='r')
    # for i, txt in enumerate(avg_rewards):
    #     episode_mark = i * group_size + group_size / 2
    #     plt.annotate(f'{txt:.3f}', (episode_mark, rewards[i]), textcoords='offset points', xytext=(0, 10), ha='center')
    plt.title('Average Total Rewards per 10 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.grid(True)

    # 保存图形到文件
    plt.savefig(sava_path)
    plt.close()


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
    plt.close()


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
    plt.close()


def get_grid_search_results(data_path):
    recall_thresholds = [0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    result_dict = {}

    df = pd.read_csv(data_path, sep=',', header=0)
    df = df[['FileName', 'efConstruction', 'M', 'efSearch', 'construction_time', 'recall', 'search_time',
             'construct_dc_counts',
             'search_dc_counts', 'average_construct_dc_counts', 'average_search_dc_counts']]

    grouped = df.groupby('FileName')

    # 遍历每个分组
    for filename, group in grouped:
        result_dict[filename] = {}

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
                    min_search_time_row['search_time'],
                    min_search_time_row['construct_dc_counts'],
                    min_search_time_row['search_dc_counts'],
                    min_search_time_row['average_construct_dc_counts'],
                    min_search_time_row['average_search_dc_counts']
                ]

                # 存储到结果字典中
                result_dict[filename][recall_threshold] = result_data

    for filename in result_dict.keys():
        print(filename)

        for rec in result_dict[filename].keys():
            print(f'{rec}: {result_dict[filename][rec]}')

    return result_dict


def get_drl_pr_results(best_performance_file, best_paras_file, performance_scaler, device, result_path, dataset_name, target_rec_lis, flag):  # flag为0表示不是serial，为1表示是
    data_source = 'global'

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

    df['FileName'] = [dataset_name] * len(target_rec_lis)
    df['target_recall'] = target_rec_lis
    df['data_source'] = [data_source] * len(target_rec_lis)

    df.to_csv(result_path, mode='w', index=False)

    # 输出 DataFrame
    print(df)
    para_dic = {}
    para_df = df[['FileName', 'target_recall', 'efConstruction', 'M', 'efSearch']]
    groups = para_df.groupby('FileName')
    for filename, group in groups:
        para_dic[filename] = group[['target_recall', 'efConstruction', 'M', 'efSearch']].values.tolist()

    print(para_dic)

    return result


if __name__ == '__main__':
    print('----------------执行准备工作----------------')
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

    # data_path1 = os.path.join(parent_directory, 'Data/train_data.csv')
    # data_path2 = os.path.join(parent_directory, 'Data/real_data_test.csv')
    # data_path3 = os.path.join(parent_directory, 'Data/real_data_test_generalization.csv')
    # result_dic1 = get_grid_search_results(data_path1)
    # print('\n')
    # result_dic2 = get_grid_search_results(data_path2)
    # print('\n')
    # result_dic3 = get_grid_search_results(data_path3)

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

    #target_rec_lis = [0.9, 0.95, 0.99] #sift50
    target_rec_lis = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    #target_rec_lis = [0.9, 0.92, 0.95, 0.98, 0.99]

    result_para_dic = {}

    # 'sift50', 'deep10', 'glove', 'gist'
    for dataset_name in ['gist']:
        cfg_lis = []

        global_tep = 2000


        cfg_lis = []

        for ep in [8000]:  # 如果使用局部最优，那么max_step就设置为400；使用全局最优就200吧
            for tep in [global_tep]:
                for mt in [5000]:
                    for lr in [0.0001]:  # [0.0001, 0.00001]
                        for bt in [128]:
                            for sigma in [0.2]:
                                for pec_reward in [1]:  # general方式下不能对正奖励扩大
                                    for delay_time in [2]:
                                        for ncst in [200]:  # general方式下这个阈值要增加
                                            for ncep in [200]:
                                                cfg_lis.append(
                                                    (ep, tep, mt, lr, bt, sigma, pec_reward, delay_time, ncst, ncep))

        for cfg in tqdm(cfg_lis, total=len(cfg_lis)):
            ep, tep, mt, lr, bt, sigma, pec_reward, delay_time, ncst, ncep = cfg

            args_r.epoches = ep
            args_r.test_epoches = tep
            args_r.max_steps = mt
            args_r.clr = lr
            args_r.alr = lr / 10
            args_r.tau = 0.0001
            args_r.batch_size = bt
            args_r.sigma = sigma
            args_r.pec_reward = pec_reward
            args_r.delay_time = delay_time
            args_r.nochange_steps = ncst
            args_r.nochange_episodes = ncep

            expr_name = 'onlys_{}_{}_{}_{}_{}_{}_{}_reward3_global_{}_{}'.format(args_r.epoches, args_r.max_steps, args_r.batch_size, args_r.alr, args_r.tau, args_r.sigma,
                                                                                 args_r.delay_time, args_r.pec_reward, args_r.nochange_steps)

            result_path = os.path.join(current_directory + '/' + stor_subdir + '/recommend_results',
                                       'dast_change/eval_{}_{}_{}_{}.csv'.format(expr_name, dataset_name, args_r.test_epoches, args_r.nochange_episodes,
                                                                        args_r.test_epoches))

            best_performance_file = os.path.join(current_directory + '/' + stor_subdir,
                                                 'dast_change/eval_best_performance_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
            best_paras_file = os.path.join(current_directory + '/' + stor_subdir,
                                           'dast_change/eval_best_paras_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
            print(best_performance_file)
            print(best_paras_file)
            result = get_drl_pr_results(best_performance_file, best_paras_file, performance_scaler, device, result_path, dataset_name, target_rec_lis, 0)

            episode_score_file = os.path.join(current_directory + '/' + stor_subdir,
                                              'dast_change/eval_episode_score_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))

            episode_steps_file = os.path.join(current_directory + '/' + stor_subdir,
                                              'dast_change/eval_episode_steps_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))

            episode_closs_file = os.path.join(current_directory + '/' + stor_subdir,
                                              'dast_change/eval_episode_closs_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))

            episode_aloss_file = os.path.join(current_directory + '/' + stor_subdir,
                                              'dast_change/eval_episode_aloss_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))

            # print('绘制episode奖励并保存')
            episode_reawrd_fig_path = os.path.join(current_directory + '/' + episode_reward_fig_floder,
                                                   'dast_change/eval_episode_reward_fig_{}_{}_{}_{}.png'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
            draw_reward(episode_score_file, episode_reawrd_fig_path)

            # print('绘制episode结束步数并保存')
            episode_steps_fig_path = os.path.join(current_directory + '/' + episode_steps_fig_floder,
                                                  'dast_change/eval_episode_steps_fig_{}_{}_{}_{}.png'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
            draw_steps(episode_steps_file, episode_steps_fig_path)

            # print('绘制episode loss并保存')
            episode_loss_fig_path = os.path.join(current_directory + '/' + episode_loss_fig_floder,
                                                 'dast_change/eval_episode_loss_fig_{}_{}_{}_{}.png'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
            draw_loss(episode_closs_file, episode_aloss_file, episode_loss_fig_path)

            df = pd.read_csv(result_path)
            result_para_dic[dataset_name] = df[['target_recall', 'efConstruction', 'M', 'efSearch']].values.tolist()

    print(result_para_dic)








