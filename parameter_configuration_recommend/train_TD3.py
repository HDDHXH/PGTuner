# -*- coding: utf-8 -*-
"""
Train the model
"""

import os
import sys

sys.path.append('../')

import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.backends import cudnn
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

from query_performance_predict.Args import args as args_p
from query_performance_predict.utils import read_data, df2np, np2ts, Scaler_minmax_new_gpu, \
    Scaler_minmax_new_partial_gpu

from TD3 import *
from index_env import IndexEnv, IndexEnv_new_state, IndexEnv_new_state_new, IndexEnv_new_state_onlys
from Args import args as args_r
from utils import Logger, time_start, time_end, get_timestamp, time_to_str

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    random.seed(args_r.seed)
    np.random.seed(args_r.seed)
    torch.manual_seed(args_r.seed)
    torch.cuda.manual_seed(args_r.seed)
    torch.cuda.manual_seed_all(args_r.seed)
    cudnn.enabled = False
    cudnn.benchmark = False
    cudnn.deterministic = True

    print('----------------执行准备工作----------------')
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

    data_path = os.path.join(parent_directory, 'Data/train_data.csv')  # 这个只包含训练数据集的特征和性能数据

    predict_model_save_path = os.path.join(parent_directory,
                                               'query_performance_predict/model_checkpoints/DiPredict/{}_{}_{}_{}_checkpoint.pth'.format(
                                                   args_p.dipredict_layer_sizes, args_p.dipredict_n_epochs,
                                                   args_p.dipredict_batch_size, args_p.dipredict_lr))

    standard_path = os.path.join(parent_directory, 'query_performance_predict/scaler_paras/feature_standard.npz')

    df_ini = pd.read_csv(data_path, sep=',', header=0)

    def_df = df_ini[(df_ini['efConstruction'] == 20) & (df_ini['M'] == 4) & (df_ini['efSearch'] == 10)]

    dataset_names = def_df['FileName']
    print(dataset_names)

    feature_df = def_df[
        ['SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio',
         'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]
    def_performance_df = def_df[['recall', 'average_construct_dc_counts', 'average_search_dc_counts']]

    target_rec_lis=[0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    num_target_rec = len(target_rec_lis)

    data_feature = df2np(feature_df)
    default_performance = df2np(def_performance_df)

    data_feature[:, 0:2] = np.log10(data_feature[:, 0:2])

    final_data_feature = np.repeat(data_feature, num_target_rec, axis=0)
    final_default_performance = np.repeat(default_performance, num_target_rec, axis=0)

    # print(final_data_feature)
    # print(final_default_performance)

    num_dataset = data_feature.shape[0]
    print(num_dataset)
    num_data = num_dataset * num_target_rec

    stor_subdir = '{}_{}_TD3'.format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)
    if not os.path.exists(stor_subdir):
        os.mkdir(stor_subdir)

    if not os.path.exists(stor_subdir + '/' + 'log'):
        os.mkdir(stor_subdir + '/' + 'log')

    if not os.path.exists(stor_subdir + '/' + 'runs'):
        os.mkdir(stor_subdir + '/' + 'runs')

    if not os.path.exists(stor_subdir + '/' + 'save_memory'):
        os.mkdir(stor_subdir + '/' + 'save_memory')

    if not os.path.exists(stor_subdir + '/' + 'model_params'):
        os.mkdir(stor_subdir + '/' + 'model_params')

    # cfg_lis = []

    # for ep in [8000]:  # 如果使用局部最优，那么max_step就设置为400；使用全局最优就200吧
    #     for mt in [5000]:
    #         for lr in [0.0001]:  # [0.0001, 0.00001]
    #             for bt in [128]:
    #                 for sigma in [0.2]:
    #                     for pec_reward in [1]:
    #                         for delay_time in [2]:
    #                             for ncst in [200]:
    #                                 cfg_lis.append((ep, mt, lr, bt, sigma, pec_reward, delay_time, ncst))

    cfg_lis = [(4800, 5000, 0.0001, 256, 0.2, 10, 2, 200)]

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
        best_performance_file = os.path.join(current_directory + '/' + stor_subdir,
                                             'train_best_performance_{}.pkl'.format(expr_name))
        best_paras_file = os.path.join(current_directory + '/' + stor_subdir,
                                       'train_best_paras_{}.pkl'.format(expr_name))

        episode_score_file = os.path.join(current_directory + '/' + stor_subdir,
                                          'train_episode_score_{}.pkl'.format(expr_name))

        episode_steps_file = os.path.join(current_directory + '/' + stor_subdir,
                                        'train_episode_steps_{}.pkl'.format(expr_name))

        episode_closs_file = os.path.join(current_directory + '/' + stor_subdir,
                                        'train_episode_closs_{}.pkl'.format(expr_name))

        episode_aloss_file = os.path.join(current_directory + '/' + stor_subdir,
                                        'train_episode_aloss_{}.pkl'.format(expr_name))

        logger = Logger(name=args_r.method, log_file=stor_subdir + '/' + 'log/train_{}.log'.format(expr_name))

        memory_file = os.path.join(current_directory + '/' + stor_subdir, 'save_memory/train_{}.pkl'.format(expr_name))


        writer_dir = os.path.join(current_directory + '/' + stor_subdir + '/' + 'runs', 'train_{}'.format(expr_name))
        if not os.path.exists(writer_dir):
            os.mkdir(writer_dir)
        # writer = SummaryWriter(writer_dir)

        # 加载并初始化环境
        print('----------------初始化环境----------------')
        env = IndexEnv_new_state_onlys(num_dataset, final_default_performance, target_rec_lis, args_r, args_p, predict_model_save_path, standard_path, device)
        
        # 加载并构建DDPG模型
        print('----------------加载DDPG模型----------------')
        ddpg_opt = dict()
        ddpg_opt['tau'] = args_r.tau  # 0.00001
        ddpg_opt['alr'] = args_r.alr  # 0.00001
        ddpg_opt['clr'] = args_r.clr  # 0.00001

        gamma = 0.9

        ddpg_opt['gamma'] = gamma
        ddpg_opt['max_steps'] = args_r.max_steps
        ddpg_opt['batch_size'] = args_r.batch_size
        ddpg_opt['memory_size'] = args_r.memory_size

        ddpg_opt['sigma_decay_rate'] = args_r.sigma_decay_rate
        ddpg_opt['sigma'] = args_r.sigma
        # ddpg_opt['sigma_value'] = args_r.sigma_value

        # ddpg_opt['policy_noise_clip'] = args_r.policy_noise_clip
        ddpg_opt['delay_time'] = args_r.delay_time

        ddpg_opt['actor_layer_sizes'] = eval(args_r.actor_layer_sizes)
        ddpg_opt['critic_layer_sizes'] = eval(args_r.critic_layer_sizes)

        ddpg_opt['actor_path'] = os.path.join(current_directory + '/' + stor_subdir,
                                              'model_params/actor_{}.pth'.format(expr_name))
        ddpg_opt['critic1_path'] = os.path.join(current_directory + '/' + stor_subdir,
                                               'model_params/critic1_{}.pth'.format(expr_name))
        ddpg_opt['critic2_path'] = os.path.join(current_directory + '/' + stor_subdir,
                                                'model_params/critic2_{}.pth'.format(expr_name))

        n_states = args_r.n_states
        n_actions = args_r.n_actions

        model = TD3(n_states=n_states, n_actions=n_actions, num_data=num_data, opt=ddpg_opt, dv=device)

        # decay rate
        step_counter = 0
        episode_score = {}
        # episode_dec = {}
        episode_steps = {} #记录每一个episode跑了多少步
        episode_closs = {}
        episode_aloss = {}

        if os.path.exists(memory_file):
            model.replay_memory.load_memory(memory_file)
            print("Load Memory: {}".format(len(model.replay_memory)))

        # time for every step 完成了个整个一步的时间，包含env_step_time和action_step_time以及train_step_time（如果训练了的话）
        step_times = []
        # time for training
        train_step_times = []
        # 环境执行一步的时间
        env_step_times = []
        # 添加经验的时间
        add_step_times = []
        # 推荐动作的时间
        action_step_times = []

        total_scores = []  # 收集每个episode的总奖励

        print('----------------开始收集经验与训练----------------')
        start_time = time.time()
        for episode in tqdm(range(args_r.epoches), total=args_r.epoches):  # 在这个训练代码中的数据全是在cpu上，状态等数据全是二维数组
            current_states = env._initialize()
            logger.info("\nEnv initialized")

            model.reset(args_r.sigma)

            train_step = 0
            accumulate_loss = [0, 0]

            # episode_dec[episode] = []

            for st in tqdm(range(args_r.max_steps), total=args_r.max_steps):
                # for st in range(args_r.max_steps):
                step_time = time_start()
                states = current_states

                action_step_time = time_start()
                actions = model.choose_action(states, True)
                # print('init_para:', actions)
                action_step_time = time_end(action_step_time)

                # logger.info("[ddpg] Action: {}")

                env_step_time = time_start()
                rewards, states_, dones, _, _, _, _ = env._step(actions, final_data_feature, best_performance_file, best_paras_file)  # filename是指最佳性能存储文件

                env_step_time = time_end(env_step_time)
                # print(f'当前步的总得分为: {total_score}')

                next_states = states_

                add_step_time = time_start()
                model.add_sample(states, actions, rewards, next_states, dones)
                add_step_time = time_end(add_step_time)

                current_states = next_states
                train_step_time = 0.0

                if len(model.replay_memory) > args_r.batch_size:  # replay_memory的大小和batch_size的大小决定了何时开始训练
                    losses = []
                    train_step_time = time_start()
                    for i in range(2):
                        loss = model.update()
                        if (model.update_time % model.delay_time) == 0:
                            losses.append(loss)
                            train_step += 1

                    train_step_time = time_end(train_step_time) / 2

                    accumulate_loss[0] += sum([x[0] for x in losses])
                    accumulate_loss[1] += sum([x[1] for x in losses])
                    # logger.info('[Episode: {}][Step: {}] Critic: {} Actor: {}'.format(episode, st, accumulate_loss[0] / train_step, accumulate_loss[1] / train_step))

                # all_step time
                step_time = time_end(step_time)
                step_times.append(step_time)
                # env_step_time
                env_step_times.append(env_step_time)
                # add_step_time
                add_step_times.append(add_step_time)
                # training step time
                train_step_times.append(train_step_time)
                # action step times
                action_step_times.append(action_step_time)

                # step_counter += 1

                if env.nochange_steps == args_r.nochange_steps or env.score < -2000:  # 这个score比较的数值还要根据实际情况确定
                    break
                # if env.score < -2000:  # 这个score比较的数值还要根据实际情况确定
                #     break

            model.actor_scheduler.step()
            model.critic1_scheduler.step()
            model.critic2_scheduler.step()

            episode_score[episode] = env.score
            episode_steps[episode] = env.steps
            episode_closs[episode] = accumulate_loss[0] / train_step
            episode_aloss[episode] = accumulate_loss[1] / train_step

            print(f'当前episode得分为: {env.score}, 步数为：{env.steps}')
            print(f'****************************第{episode + 1}个episode结束****************************')

            logger.info("[Episode: {}] TotalScore {} TotalSteps {}".format(episode, env.score, env.steps))
            # logger.info('[Episode: {}]Critic: {} Actor: {}'.format(episode, accumulate_loss[0] / train_step,
            #                                                        accumulate_loss[1] / train_step))
            if (episode + 1) % 100 == 0:
                logger.info(
                    "[Episode: {}][Average] step: {}s env step: {}s add step: {}s train step: {}s action time: {}s".format(
                        episode, np.mean(step_times),
                        np.mean(env_step_times), np.mean(add_step_times), np.mean(train_step_times),
                        np.mean(action_step_times)))

            if (episode + 1) % 100 == 0:
                model.replay_memory.save(memory_file)
                model.save_model(episode)

            if (episode + 1) % 100 == 0:
                with open(episode_score_file, 'wb') as f:
                    pickle.dump(episode_score, f)
            if (episode + 1) % 100 == 0:
                with open(episode_steps_file, 'wb') as f:
                    pickle.dump(episode_steps, f)
            if (episode + 1) % 100 == 0:
                with open(episode_closs_file, 'wb') as f:
                    pickle.dump(episode_closs, f)
            if (episode + 1) % 100 == 0:
                with open(episode_aloss_file, 'wb') as f:
                    pickle.dump(episode_aloss, f)

        end_time = time.time()
        reccomend_time = end_time - start_time
        logger.info("Training time: {}s".format(reccomend_time))





