import re
import os
import sys
import time
import math
import torch
import torch.optim as optim
import numpy as np
import pickle

# from index_performance_predict.Args import args as args_p
from index_performance_predict.models import Direct_Predict_MLP, Direct_Predict_MLP_nsg
from index_performance_predict.utils import np2ts, Scaler_minmax_new_gpu, Scaler_minmax_new_gpu_nsg, load_model

# from Args import args as args_r
from utils import Scaler_para, Scaler_para_nsg, Scaler_state_0, Scaler_state_new, Scaler_state_data, Scaler_state_new_new, Scaler_state_new_onlys, Scaler_state_new_onlys_nsg

class IndexEnv(object):  # 创建环境
    def __init__(self, num_dataset, default_performance, args_r, args_p, predict_model_save_path, standard_path, dv):
        self.device = dv

        # print('-------------加载索引性能预测模型-------------')
        self.args_p = args_p
        self.predict_model = Direct_Predict_MLP(eval(args_p.dipredict_layer_sizes)).to(self.device)
        self._get_predict_model(predict_model_save_path)  # 加载索引性能预测模型

        self.feature_scaler = Scaler_minmax_new_gpu(6, dv)
        self.performance_scaler = Scaler_minmax_new_gpu(0, dv)
        self.para_scaler = Scaler_para()  # 这个操作的对象是numpy
        #self.state_scaler = Scaler_state(10)  ##这个操作的对象也是numpy
        self.state_scaler = Scaler_state_0(7)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.args_r = args_r

        self.target_rec = np.tile(np.array([[0.9], [0.92], [0.94], [0.95], [0.96], [0.98], [0.99]]), (num_dataset, 1))  # (7*num_dataset) * 1

        self.score = 0.0
        # self.score_array = np.tile(np.array([0]), self.target_rec.shape[0])
        self.steps = 0
        self.max_steps = args_r.max_steps
        self.nochange_steps = 0

        # print(self.target_rec)
        self.default_paras = np.tile(np.array([[20, 4, 10]]), (self.target_rec.shape[0], 1))  # (7*num_dataset) * 1
        self.last_index_performance = None  # (num_dataset*7) * 1
        self.default_index_performance = default_performance.copy()  # (num_dataset*7) * 1

    def _get_predict_model(self, model_save_path):
        optimizer = optim.Adam(self.predict_model.parameters(), lr=self.args_p.dml_lr, weight_decay=self.args_p.weight_decay)
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path):
        self.feature_scaler.load_parameters(None, standard_path, self.device)
        # self.state_scaler.load_parameters(standard_path)

    def _get_action(self, actions):  # 输入的actions是一个N*3的数组，在传入到这里之前actions已经是一个在cpu上的数组
        paras = self.para_scaler.inverse_transform(actions)  # N*3的数组
        paras[:, 0] = np.power(10, paras[:, 0])
        paras[:, 2] = np.power(10, paras[:, 2])
        paras = np.floor(paras + 0.5)  # 要4舍5入转成整数

        paras[:, 0] = np.where(paras[:, 0] < paras[:, 1], paras[:, 1], paras[:, 0])

        return paras

    def _get_index_performance(self, feature_input):
        # 预测索引性能，输入feature_input是一个N*14的张量，没有经过归一化
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        # batch_size = 20480
        # num = feature_input_scaled.size()[0]
        #
        # n_batch = (num + batch_size - 1) // batch_size
        #
        # index_performance_lis = []
        #
        # with torch.no_grad():
        #     for idx in range(n_batch):
        #         batch_start = idx * batch_size
        #         batch_end = min(num, (idx + 1) * batch_size)
        #
        #         temp_feature_input_scaled = feature_input_scaled[batch_start: batch_end, :]
        #         temp_index_performance = self.predict_model(temp_feature_input_scaled)
        #
        #         index_performance_lis.append(temp_index_performance)
        #
        # index_performance = torch.cat(index_performance_lis, dim=0)
        # index_performance = index_performance.cpu().numpy()  # 要用逆归一化后的数据

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1:] = torch.pow(10, real_index_performance[:, 1:])

        real_index_performance = real_index_performance.cpu().numpy()

        return real_index_performance  # 同时返回归一化后的性能和真实性能，一个用于组成状态向量，一个用于更新最优性能和计算奖励

    @staticmethod
    def _get_performance_improvement(last_index_performance, current_index_performance):  # 这里其实要考虑的是用真实性能计算还是归一化后的性能计算
        ct_counts_dec = (last_index_performance[:, 1] - current_index_performance[:, 1]) / last_index_performance[:, 1]
        st_counts_dec = (last_index_performance[:, 2] - current_index_performance[:, 2]) / last_index_performance[:, 2]

        return ct_counts_dec, st_counts_dec

    def _get_best_now(self, filename):
        with open(filename, 'rb') as f:
            best_index_performance = pickle.load(f)
        return best_index_performance

    def _get_best_paras_now(self, filename):
        with open(filename, 'rb') as f:
            best_paras = pickle.load(f)
        return best_paras

    def _record_best(self, cur_index_performance, cur_paras, performance_filename, paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            # print(target_rec.shape)
            best_rec = best_index_performance[:, 0]
            # print(best_rec.shape)
            cur_rec = cur_index_performance[:, 0]
            # print(cur_rec.shape)

            ct_counts_dec, st_counts_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
            # print(ct_counts_dec.shape)
            # print(st_counts_dec.shape)
            
            target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec
            
            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)
            
            # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
            cond_a = (cond1 & cond4)
             # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
            cond_b = (cond1 & cond3 & cond5)
            # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
            cond_c = (cond2 & cond4 & cond6)

            if (cond_a.any() or cond_b.any() or cond_c.any()):
                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)

        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

    def _record_best2(self, cur_index_performance, cur_paras, performance_filename, paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            # print(target_rec.shape)
            best_rec = best_index_performance[:, 0]
            # print(best_rec.shape)
            cur_rec = cur_index_performance[:, 0]
            # print(cur_rec.shape)

            ct_counts_dec, st_counts_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
            # print(ct_counts_dec.shape)
            # print(st_counts_dec.shape)
            
            target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec
            
            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)

            # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
            cond_a = (cond1 & cond4)
             # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
            cond_b = (cond1 & cond3 & cond5)
            # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
            cond_c = (cond2 & cond4 & cond6)

            if not (cond_a.any() or cond_b.any() or cond_c.any()):
                self.nochange_steps += 1
            else:
                self.nochange_steps = 0

                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)

        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

    def _update_episode_best(self, cur_index_performance):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算                                                  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance, cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        cond1 = (best_rec < target_rec)
        cond2 = (best_rec >= target_rec)
        cond3 = (cur_rec < target_rec)
        cond4 = (cur_rec >= target_rec)
        cond5 = (cur_rec > best_rec)
        cond6 = (target_dec > 0)
        
        # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
        cond_a = (cond1 & cond4)
            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
        cond_b = (cond1 & cond3 & cond5)
        # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
        cond_c = (cond2 & cond4 & cond6)

        if (cond_a.any() or cond_b.any() or cond_c.any()):
            self.last_index_performance[cond_a] = cur_index_performance[cond_a]
            self.last_index_performance[cond_b] = cur_index_performance[cond_b]
            self.last_index_performance[cond_c] = cur_index_performance[cond_c]

    def _update_episode_best2(self, cur_index_performance):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算                                                  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance, cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        cond1 = (best_rec < target_rec)
        cond2 = (best_rec >= target_rec)
        cond3 = (cur_rec < target_rec)
        cond4 = (cur_rec >= target_rec)
        cond5 = (cur_rec > best_rec)
        cond6 = (target_dec > 0)

        # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
        cond_a = (cond1 & cond4)
            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
        cond_b = (cond1 & cond3 & cond5)
        # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
        cond_c = (cond2 & cond4 & cond6)

        if not (cond_a.any() or cond_b.any() or cond_c.any()):
            self.nochange_steps += 1
        else:
            self.nochange_steps = 0

            self.last_index_performance[cond_a] = cur_index_performance[cond_a]
            self.last_index_performance[cond_b] = cur_index_performance[cond_b]
            self.last_index_performance[cond_c] = cur_index_performance[cond_c]

    def _initialize(self):
        self.steps = 0
        self.score = 0.0
        self.nochange_steps = 0

        # self.score_array = np.tile(np.array([0]), self.target_rec.shape[0])

        self.last_index_performance = self.default_index_performance.copy()

        cur_index_performance = self.default_index_performance.copy()
        # best_index_performance = self.default_index_performance.copy()

        cur_index_performance[:, 1:] = np.log10(cur_index_performance[:, 1:])
        # best_index_performance[:, 1:] = np.log10(best_index_performance[:, 1:])

        # 状态向量：当前参数、目标召回率、当前参数对应的索引性能, 当前最好性能（暂时先不考虑数据集特征，共计10个特征）
        #init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.target_rec), cur_index_performance, best_index_performance), axis=1)
        init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.target_rec), cur_index_performance), axis=1) #状态向量只用当前性能，不用当前最佳性能

        init_state[:, 0] = np.log10(init_state[:, 0])
        init_state[:, 2] = np.log10(init_state[:, 2])

        init_state_ = self.state_scaler.transform(init_state)  # 状态向量要归一化再存储

        return init_state_

    def _step(self, actions, data_feature, performance_filename, paras_filename):  # paras是推荐的参数（没有经过归一化处理）， data_feature是数据集输入特征（也没有经过归一化）
        self.steps += 1  # 执行一次就加一次

        paras = self._get_action(actions)
        # print('paras: ', paras)
        feature_input = np.concatenate((np.copy(paras), np.copy(data_feature)), axis=1)
        feature_input[:, 0] = np.log10(feature_input[:, 0])
        feature_input[:, 2] = np.log10(feature_input[:, 2])

        index_performance = self._get_index_performance(feature_input)
        c_index_performance = index_performance.copy()

        reward, average_reward, min_dec = self._get_reward3(index_performance)

        #global方式
        # self._record_best2(index_performance, paras, performance_filename, paras_filename)

        # best_now_performance = self._get_best_now(performance_filename)
        # self.last_index_performance = best_now_performance  # 这里更新self.last_index_performance适用于下一步计算奖励

        #local方式
        # self._record_best(index_performance, paras, performance_filename, paras_filename)
        # self._update_episode_best2(index_performance)

        #general方式
        self._record_best2(index_performance, paras, performance_filename, paras_filename)
        self.last_index_performance = c_index_performance.copy()

        # b_index_performance = best_now_performance.copy()

        c_index_performance[:, 1:] = np.log10(c_index_performance[:, 1:])
        # b_index_performance[:, 1:] = np.log10(b_index_performance[:, 1:])

        #next_state = np.concatenate((np.copy(paras), np.copy(self.target_rec), c_index_performance, b_index_performance), axis=1)
        next_state = np.concatenate((np.copy(paras), np.copy(self.target_rec), c_index_performance), axis=1) #状态向量只用当前性能，不用当前最佳性能
        next_state[:, 0] = np.log10(next_state[:, 0])
        next_state[:, 2] = np.log10(next_state[:, 2])

        next_state_ = self.state_scaler.transform(next_state)  # 状态向量要归一化再存储

        num = feature_input.shape[0]

        # if self.steps < self.max_steps and self.score >= -10000:
        #     terminate = np.zeros((num, 1), dtype=bool)
        # else:
        #     terminate = np.ones((num, 1), dtype=bool)
        # print(average_reward)
        # if average_reward >= -50:
        #     terminate = np.zeros((num, 1), dtype=bool)
        # else:
        #     terminate = np.ones((num, 1), dtype=bool)

        # if self.nochange_steps == self.args_r.nochange_steps:   #用性能连续没有改进的步数是否达到设定阈值来判断是否样稿终止当前episode，达到了则将terminate设置为True，终止当前episode
        #     terminate = np.ones((num, 1), dtype=bool)
        # else:
        #     terminate = np.zeros((num, 1), dtype=bool)
        terminate = np.zeros((num, 1), dtype=bool)

        # reward_1d = reward.reshape(-1)
        # print(reward_1d)

        # terminate = np.zeros((num), dtype=bool)           #这个方式是根据单个组合的得分是否低于-10000来控制，对应train_new中用np.any()
        # conditions = (reward_1d < -500)
        # terminate[conditions] = True
        # terminate = terminate.reshape((num, 1))

        return reward, next_state_, terminate, self.score, average_reward, index_performance, paras, min_dec

    @staticmethod
    def _calculate_reward(delta):
        reward_positive = (1 + delta) ** 2 - 1   #后面再测试一下reward_positive = (1 + delta) ** 2 - 1 ，奖励不同确实影响很大。
        reward_negative = -(1 - delta) ** 2 + 1

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta >= 0, reward_positive, reward_negative)
        # _reward = np.where(delta >= 0, reward_positive, reward_negative)

        return _reward
 
    @staticmethod
    def _calculate_reward_rec(delta0, deltat):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        reward_positive = ((1 + delta0) ** 2 - 1) * np.abs(1 + deltat)
        reward_negative = -((1 - delta0) ** 2 - 1) * np.abs(1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta0 >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec2(delta0, deltat, deltat20):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值 
        _reward = np.zeros_like(delta0)

        cond1 = (deltat20 < 0 ) & (delta0 < 0)
        cond2 = (deltat20 < 0 ) & (delta0 >= 0)
        cond3 = (deltat20 > 0 ) & (delta0 < 0)

        reward_cond1 = -(1 - delta0) ** 2 + 1
        reward_cond2 = ((1 + delta0) ** 2) * (1 + deltat) 
        reward_cond3 = (-(1 - delta0) ** 2) * (1 - deltat) 

 
        # 根据 delta0 的值选择正负奖励
        _reward[cond1] = reward_cond1[cond1]
        _reward[cond2] = reward_cond2[cond2]
        _reward[cond3] = reward_cond3[cond3]
        
        return _reward

    def _get_reward(self, cur_index_performance):  #优先使用reward1，reward1是分别计算ct和st的奖励，再加权求和
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance, cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec
        min_dec = np.min(target_dec)

        delta0 = (cur_rec - target_rec) / target_rec
        deltat = (cur_rec - best_rec) / best_rec

        # print(delta0)
        # print(deltat)
        # print(ct_counts_dec)
        # print(st_counts_dec)

        reward = self._calculate_reward_rec(delta0, deltat)
        # print(reward)

        # 条件: 当前和最佳召回率都满足目标召回率
        # target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        ct_counts_reward = self._calculate_reward(ct_counts_dec)
        st_counts_reward = self._calculate_reward(st_counts_dec)

        counts_reward = self.args_r.lamb * ct_counts_reward + st_counts_reward

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        # condition1 = condition & (target_dec  > 0)
        # condition2 = condition & (target_dec  == 0)
        # condition3 = condition & (target_dec  < 0)

        # reward[condition1] = counts_reward[condition1]
        # reward[condition2] = counts_reward[condition2]
        # reward[condition3] = counts_reward[condition3]

        average_reward = np.mean(reward)
        self.score += average_reward
        
        # self.score_array = self.score_array + reward 
        # print(reward)
        # print(self.score_array)
        # print(average_reward)
        # print('\n')

        reward = np.where(reward > 0, reward * self.args_r.pec_reward, reward)  # 注释是方式1,不注释是方式2
        # print('reward: ', reward)

        reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward, min_dec 

    def _get_reward2(self, cur_index_performance):
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance, cur_index_performance)

        delta0 = (cur_rec - target_rec) / target_rec
        deltat = (cur_rec - best_rec) / best_rec

        reward = self._calculate_reward_rec(delta0, deltat)

        # 条件: 当前和最佳召回率都满足目标召回率
        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        # condition1 = condition & (target_dec  > 0)
        # condition2 = condition & (target_dec  == 0)
        # condition3 = condition & (target_dec  < 0)

        # reward[condition1] = counts_reward[condition1]
        # reward[condition2] = counts_reward[condition2]
        # reward[condition3] = counts_reward[condition3]

        average_reward = np.mean(reward)
        self.score += average_reward

        reward = np.where(reward > 0, reward * self.args_r.pec_reward, reward)  # 注释是方式1,不注释是方式2
        # print('reward: ', reward)

        reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward

    def _get_reward3(self, cur_index_performance):  #优先使用reward1，reward1是分别计算ct和st的奖励，再加权求和
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance, cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec
        min_dec = np.min(target_dec)

        delta0 = cur_rec - target_rec
        deltat = cur_rec - best_rec
        deltat20 = best_rec - target_rec

        # print(delta0)
        # print(deltat)
        # print(ct_counts_dec)
        # print(st_counts_dec)

        reward = self._calculate_reward_rec2(delta0, deltat, deltat20)
        # print(reward)

        # 条件: 当前和最佳召回率都满足目标召回率
        # target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        ct_counts_reward = self._calculate_reward(ct_counts_dec)
        st_counts_reward = self._calculate_reward(st_counts_dec)

        counts_reward = self.args_r.lamb * ct_counts_reward + st_counts_reward

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        # condition1 = condition & (target_dec  > 0)
        # condition2 = condition & (target_dec  == 0)
        # condition3 = condition & (target_dec  < 0)

        # reward[condition1] = counts_reward[condition1]
        # reward[condition2] = counts_reward[condition2]
        # reward[condition3] = counts_reward[condition3]

        average_reward = np.mean(reward)
        self.score += average_reward
        
        # self.score_array = self.score_array + reward 
        # print(reward)
        # print(self.score_array)
        # print(average_reward)
        # print('\n')

        reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward)  # 注释是方式1,不注释是方式2
        # print('reward: ', reward)

        reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward, min_dec 

class IndexEnv_data(object):  # 创建环境
    def __init__(self, num_dataset, default_performance, args_r, args_p, predict_model_save_path, standard_path, dv):
        self.device = dv

        # print('-------------加载索引性能预测模型-------------')
        self.args_p = args_p
        self.predict_model = Direct_Predict_MLP(eval(args_p.dipredict_layer_sizes)).to(self.device)
        self._get_predict_model(predict_model_save_path)  # 加载索引性能预测模型

        self.feature_scaler = Scaler_minmax_new_gpu(6, dv)
        self.performance_scaler = Scaler_minmax_new_gpu(0, dv)
        self.para_scaler = Scaler_para()  # 这个操作的对象是numpy
        self.state_scaler = Scaler_state_data(21)  ##这个操作的对象也是numpy
        # self.state_scaler = Scaler_state_data_0(18)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.args_r = args_r
        self.score = 0.0
        self.steps = 0
        self.max_steps = args_r.max_steps

        self.target_rec = np.tile(np.array([[0.9], [0.92], [0.94], [0.95], [0.96], [0.98], [0.99]]), (num_dataset, 1))  # (7*num_dataset) * 1
        # print(self.target_rec)
        self.default_paras = np.tile(np.array([[20, 4, 10]]), (self.target_rec.shape[0], 1))  # (7*num_dataset) * 1
        self.last_index_performance = None  # (num_dataset*7) * 1
        self.default_index_performance = default_performance.copy()  # (num_dataset*7) * 1

    def _get_predict_model(self, model_save_path):
        optimizer = optim.Adam(self.predict_model.parameters(), lr=self.args_p.dml_lr,
                               weight_decay=self.args_p.weight_decay)
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path):
        self.feature_scaler.load_parameters(None, standard_path, self.device)
        # self.state_scaler.load_parameters(standard_path)

    def _get_action(self, actions):  # 输入的actions是一个N*3的数组，在传入到这里之前actions已经是一个在cpu上的数组
        paras = self.para_scaler.inverse_transform(actions)  # N*3的数组
        paras[:, 0] = np.power(10, paras[:, 0])
        paras[:, 2] = np.power(10, paras[:, 2])
        paras = np.floor(paras + 0.5)  # 要4舍5入转成整数

        paras[:, 0] = np.where(paras[:, 0] < paras[:, 1], paras[:, 1], paras[:, 0])

        return paras

    def _get_index_performance(self, feature_input):
        # 预测索引性能，输入feature_input是一个N*14的张量，没有经过归一化
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        # batch_size = 20480
        # num = feature_input_scaled.size()[0]
        #
        # n_batch = (num + batch_size - 1) // batch_size
        #
        # index_performance_lis = []
        #
        # with torch.no_grad():
        #     for idx in range(n_batch):
        #         batch_start = idx * batch_size
        #         batch_end = min(num, (idx + 1) * batch_size)
        #
        #         temp_feature_input_scaled = feature_input_scaled[batch_start: batch_end, :]
        #         temp_index_performance = self.predict_model(temp_feature_input_scaled)
        #
        #         index_performance_lis.append(temp_index_performance)
        #
        # index_performance = torch.cat(index_performance_lis, dim=0)
        # index_performance = index_performance.cpu().numpy()  # 要用逆归一化后的数据

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1:] = torch.pow(10, real_index_performance[:, 1:])

        real_index_performance = real_index_performance.cpu().numpy()

        return real_index_performance  # 同时返回归一化后的性能和真实性能，一个用于组成状态向量，一个用于更新最优性能和计算奖励

    @staticmethod
    def _get_performance_improvement(last_index_performance, current_index_performance):  # 这里其实要考虑的是用真实性能计算还是归一化后的性能计算
        ct_counts_dec = (last_index_performance[:, 1] - current_index_performance[:, 1]) / last_index_performance[:, 1]
        st_counts_dec = (last_index_performance[:, 2] - current_index_performance[:, 2]) / last_index_performance[:, 2]

        return ct_counts_dec, st_counts_dec

    def _get_best_now(self, filename):
        with open(filename, 'rb') as f:
            best_index_performance = pickle.load(f)
        return best_index_performance

    def _get_best_paras_now(self, filename):
        with open(filename, 'rb') as f:
            best_paras = pickle.load(f)
        return best_paras

    def _record_best(self, cur_index_performance, cur_paras, performance_filename, paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            best_rec = best_index_performance[:, 0]
            cur_rec = cur_index_performance[:, 0]

            ct_counts_dec, st_counts_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)

            target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)
            # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
            cond_a = (cond1 & cond4)
            best_index_performance[cond_a] = cur_index_performance[cond_a]
            best_paras[cond_a] = cur_paras[cond_a]

            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
            cond_b = (cond1 & cond3 & cond5)
            best_index_performance[cond_b] = cur_index_performance[cond_b]
            best_paras[cond_b] = cur_paras[cond_b]

            # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
            cond_c = (cond2 & cond4 & cond6)
            best_index_performance[cond_c] = cur_index_performance[cond_c]
            best_paras[cond_c] = cur_paras[cond_c]

            with open(performance_filename, 'wb') as f:
                pickle.dump(best_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(best_paras, f)

        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

    def _update_episode_best(self, cur_index_performance):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算                                                  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance, cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        cond1 = (best_rec < target_rec)
        cond2 = (best_rec >= target_rec)
        cond3 = (cur_rec < target_rec)
        cond4 = (cur_rec >= target_rec)
        cond5 = (cur_rec > best_rec)
        cond6 = (target_dec > 0)
        # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
        cond_a = (cond1 & cond4)
        self.last_index_performance[cond_a] = cur_index_performance[cond_a]

        # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
        cond_b = (cond1 & cond3 & cond5)
        self.last_index_performance[cond_b] = cur_index_performance[cond_b]

        # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
        cond_c = (cond2 & cond4 & cond6)
        self.last_index_performance[cond_c] = cur_index_performance[cond_c]

    def _initialize(self, scaled_data_feature):  # 状态向量加入归一化后的数据特征
        self.steps = 0
        self.score = 0.0

        self.last_index_performance = self.default_index_performance.copy()

        cur_index_performance = self.default_index_performance.copy()
        best_index_performance = self.default_index_performance.copy()

        cur_index_performance[:, 1:] = np.log10(cur_index_performance[:, 1:])
        best_index_performance[:, 1:] = np.log10(best_index_performance[:, 1:])

        # 状态向量：当前参数、目标召回率、当前参数对应的索引性能, 当前最好性能（暂时先不考虑数据集特征，共计10个特征）
        init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.target_rec), cur_index_performance, best_index_performance, np.copy(scaled_data_feature)), axis=1)
        # init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.target_rec), cur_index_performance, np.copy(scaled_data_feature)), axis=1) #状态向量只用当前性能，不用当前最佳性能

        init_state[:, 0] = np.log10(init_state[:, 0])
        init_state[:, 2] = np.log10(init_state[:, 2])

        init_state_ = self.state_scaler.transform(init_state)  # 状态向量要归一化再存储

        return init_state_

    def _step(self, actions, data_feature, scaled_data_feature, performance_filename, paras_filename):  # paras是推荐的参数（没有经过归一化处理）， data_feature是数据集输入特征（也没有经过归一化），状态向量加入归一化后的数据特征
        self.steps += 1  # 执行一次就加一次

        paras = self._get_action(actions)
        # print('paras: ', paras)
        feature_input = np.concatenate((np.copy(paras), np.copy(data_feature)), axis=1)
        feature_input[:, 0] = np.log10(feature_input[:, 0])
        feature_input[:, 2] = np.log10(feature_input[:, 2])

        index_performance = self._get_index_performance(feature_input)
        c_index_performance = index_performance.copy()

        reward, average_reward = self._get_reward(index_performance)

        self._record_best(index_performance, paras, performance_filename, paras_filename)

        best_now_performance = self._get_best_now(performance_filename)
        self.last_index_performance = best_now_performance  # 这里更新self.last_index_performance适用于下一步计算奖励
        # self._update_episode_best(index_performance)

        b_index_performance = best_now_performance.copy()

        c_index_performance[:, 1:] = np.log10(c_index_performance[:, 1:])
        b_index_performance[:, 1:] = np.log10(b_index_performance[:, 1:])

        next_state = np.concatenate((np.copy(paras), np.copy(self.target_rec), c_index_performance, b_index_performance, np.copy(scaled_data_feature)), axis=1)
        # next_state = np.concatenate((np.copy(paras), np.copy(self.target_rec), c_index_performance, np.copy(scaled_data_feature)), axis=1) #状态向量只用当前性能，不用当前最佳性能
        next_state[:, 0] = np.log10(next_state[:, 0])
        next_state[:, 2] = np.log10(next_state[:, 2])

        next_state_ = self.state_scaler.transform(next_state)  # 状态向量要归一化再存储

        num = feature_input.shape[0]

        if self.steps < self.max_steps:
            terminate = np.zeros((num, 1), dtype=bool)
        else:
            terminate = np.ones((num, 1), dtype=bool)

        return reward, next_state_, terminate, self.score, average_reward, index_performance, paras

    @staticmethod
    def _calculate_reward(delta):
        reward_positive = (1 + delta) ** 2 + 1
        reward_negative = -(1 - delta) ** 2 - 1

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta > 0, reward_positive, np.where(delta < 0, reward_negative, 0.5))

        return _reward

    @staticmethod
    def _calculate_reward_rec(delta0, deltat):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        reward_positive = ((1 + delta0) ** 2 - 1) * np.abs(1 + deltat)
        reward_negative = -((1 - delta0) ** 2 - 1) * np.abs(1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta0 >= 0, reward_positive, reward_negative)

        return _reward

    def _get_reward(self, cur_index_performance):  # 优先使用reward
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance, cur_index_performance)

        delta0 = (cur_rec - target_rec) / target_rec
        deltat = (cur_rec - best_rec) / best_rec

        reward = self._calculate_reward_rec(delta0, deltat)

        # 条件: 当前和最佳召回率都满足目标召回率
        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        ct_counts_reward = self._calculate_reward(ct_counts_dec)
        st_counts_reward = self._calculate_reward(st_counts_dec)

        counts_reward = self.args_r.lamb * ct_counts_reward + st_counts_reward

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        condition1 = condition & (target_dec > 0)
        condition2 = condition & (target_dec == 0)
        condition3 = condition & (target_dec < 0)

        reward[condition1] = counts_reward[condition1]
        reward[condition2] = counts_reward[condition2]
        reward[condition3] = counts_reward[condition3]

        average_reward = np.mean(reward)
        self.score += average_reward

        reward = np.where(reward > 0, reward * 10, reward)  # 注释是方式1,不注释是方式2
        # print('reward: ', reward)

        reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward

    def _get_reward2(self, cur_index_performance):
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance, cur_index_performance)

        delta0 = (cur_rec - target_rec) / target_rec
        deltat = (cur_rec - best_rec) / best_rec

        reward = self._calculate_reward_rec(delta0, deltat)

        # 条件: 当前和最佳召回率都满足目标召回率
        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        condition1 = condition & (target_dec > 0)
        condition2 = condition & (target_dec == 0)
        condition3 = condition & (target_dec < 0)

        reward[condition1] = counts_reward[condition1]
        reward[condition2] = counts_reward[condition2]
        reward[condition3] = counts_reward[condition3]

        average_reward = np.mean(reward)
        self.score += average_reward

        reward = np.where(reward > 0, reward * 10, reward)  # 注释是方式1,不注释是方式2
        # print('reward: ', reward)

        reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward


class IndexEnv_new_state(object):  # 创建环境
    def __init__(self, num_dataset, default_performance, target_rec_lis, args_r, args_p, predict_model_save_path, standard_path, dv):
        self.device = dv

        # print('-------------加载索引性能预测模型-------------')
        self.args_p = args_p
        self.predict_model = Direct_Predict_MLP(eval(args_p.dipredict_layer_sizes)).to(self.device)
        self._get_predict_model(predict_model_save_path)  # 加载索引性能预测模型

        self.feature_scaler = Scaler_minmax_new_gpu(6, dv)
        self.performance_scaler = Scaler_minmax_new_gpu(0, dv)
        self.para_scaler = Scaler_para()  # 这个操作的对象是numpy
        # self.state_scaler = Scaler_state(10)  ##这个操作的对象也是numpy
        self.state_scaler = Scaler_state_new(15)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.args_r = args_r

        self.target_rec = np.tile(np.array(target_rec_lis).reshape(-1,1), (num_dataset, 1))  # (7*num_dataset) * 1

        self.score = 0.0
        # self.score_array = np.tile(np.array([0]), self.target_rec.shape[0])
        self.steps = 0
        self.max_steps = args_r.max_steps
        self.nochange_steps = 0

        # print(self.target_rec)
        self.default_paras = np.tile(np.array([[20, 4, 10]]), (self.target_rec.shape[0], 1))  # (7*num_dataset) * 1
        self.last_index_performance = default_performance.copy()  # (num_dataset*7) * 1
        self.default_index_performance = default_performance.copy()  # (num_dataset*7) * 1

    def _get_predict_model(self, model_save_path):
        optimizer = optim.Adam(self.predict_model.parameters(), lr=self.args_p.dml_lr,
                               weight_decay=self.args_p.weight_decay)
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path):
        self.feature_scaler.load_parameters(None, standard_path, self.device)
        # self.state_scaler.load_parameters(standard_path)

    def _get_action(self, actions):  # 输入的actions是一个N*3的数组，在传入到这里之前actions已经是一个在cpu上的数组
        paras = self.para_scaler.inverse_transform(actions)  # N*3的数组
        paras[:, 0] = np.power(10, paras[:, 0])
        paras[:, 2] = np.power(10, paras[:, 2])
        paras = np.floor(paras + 0.5)  # 要4舍5入转成整数

        paras[:, 0] = np.where(paras[:, 0] < paras[:, 1], paras[:, 1], paras[:, 0])

        return paras

    def _get_index_performance(self, feature_input):
        # 预测索引性能，输入feature_input是一个N*14的张量，没有经过归一化
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        # index_performance = torch.cat(index_performance_lis, dim=0)
        # index_performance = index_performance.cpu().numpy()  # 要用逆归一化后的数据

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1:] = torch.pow(10, real_index_performance[:, 1:])

        real_index_performance = real_index_performance.cpu().numpy()

        return real_index_performance  # 同时返回归一化后的性能和真实性能，一个用于组成状态向量，一个用于更新最优性能和计算奖励

    @staticmethod
    def _get_performance_improvement(last_index_performance, current_index_performance):  # 这里其实要考虑的是用真实性能计算还是归一化后的性能计算
        ct_counts_dec = (last_index_performance[:, 1] - current_index_performance[:, 1]) / last_index_performance[:, 1]
        st_counts_dec = (last_index_performance[:, 2] - current_index_performance[:, 2]) / last_index_performance[:, 2]

        return ct_counts_dec, st_counts_dec

    def _get_best_now(self, filename):
        with open(filename, 'rb') as f:
            best_index_performance = pickle.load(f)
        return best_index_performance

    def _get_best_paras_now(self, filename):
        with open(filename, 'rb') as f:
            best_paras = pickle.load(f)
        return best_paras

    def _record_best(self, cur_index_performance, cur_paras, performance_filename,
                     paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            # print(target_rec.shape)
            best_rec = best_index_performance[:, 0]
            # print(best_rec.shape)
            cur_rec = cur_index_performance[:, 0]
            # print(cur_rec.shape)

            ct_counts_dec, st_counts_dec = self._get_performance_improvement(best_index_performance,
                                                                             cur_index_performance)
            # print(ct_counts_dec.shape)
            # print(st_counts_dec.shape)

            target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)

            # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
            cond_a = (cond1 & cond4)
            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
            cond_b = (cond1 & cond3 & cond5)
            # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
            cond_c = (cond2 & cond4 & cond6)

            if (cond_a.any() or cond_b.any() or cond_c.any()):
                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)

        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

    def _record_best2(self, cur_index_performance, cur_paras, performance_filename,
                      paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            # print(target_rec.shape)
            best_rec = best_index_performance[:, 0]
            # print(best_rec.shape)
            cur_rec = cur_index_performance[:, 0]
            # print(cur_rec.shape)

            ct_counts_dec, st_counts_dec = self._get_performance_improvement(best_index_performance,
                                                                             cur_index_performance)

            target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)

            # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
            cond_a = (cond1 & cond4)
            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
            cond_b = (cond1 & cond3 & cond5)
            # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
            cond_c = (cond2 & cond4 & cond6)

            if cond_a.any() or cond_b.any() or cond_c.any():
                self.nochange_steps = 0

                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)

            else:
                self.nochange_steps += 1

            return cond_c

        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

            return np.zeros(cur_index_performance.shape[0], dtype=bool)

    def _update_episode_best(self,
                             cur_index_performance):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算                                                  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance,
                                                                         cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        cond1 = (best_rec < target_rec)
        cond2 = (best_rec >= target_rec)
        cond3 = (cur_rec < target_rec)
        cond4 = (cur_rec >= target_rec)
        cond5 = (cur_rec > best_rec)
        cond6 = (target_dec > 0)

        # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
        cond_a = (cond1 & cond4)
        # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
        cond_b = (cond1 & cond3 & cond5)
        # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
        cond_c = (cond2 & cond4 & cond6)

        if (cond_a.any() or cond_b.any() or cond_c.any()):
            self.last_index_performance[cond_a] = cur_index_performance[cond_a]
            self.last_index_performance[cond_b] = cur_index_performance[cond_b]
            self.last_index_performance[cond_c] = cur_index_performance[cond_c]

    def _update_episode_best2(self,
                              cur_index_performance):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算                                                  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance,
                                                                         cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        cond1 = (best_rec < target_rec)
        cond2 = (best_rec >= target_rec)
        cond3 = (cur_rec < target_rec)
        cond4 = (cur_rec >= target_rec)
        cond5 = (cur_rec > best_rec)
        cond6 = (target_dec > 0)

        # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
        cond_a = (cond1 & cond4)
        # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
        cond_b = (cond1 & cond3 & cond5)
        # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
        cond_c = (cond2 & cond4 & cond6)

        if not (cond_a.any() or cond_b.any() or cond_c.any()):
            self.nochange_steps += 1
        else:
            self.nochange_steps = 0

            self.last_index_performance[cond_a] = cur_index_performance[cond_a]
            self.last_index_performance[cond_b] = cur_index_performance[cond_b]
            self.last_index_performance[cond_c] = cur_index_performance[cond_c]

    def _initialize(self):
        self.steps = 0
        self.score = 0.0
        self.nochange_steps = 0

        num = self.default_index_performance.shape[0]

        self.last_index_performance = self.default_index_performance.copy()

        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        # cur_rec = cur_index_performance[:, 0]

        deltat20 = (best_rec - target_rec).reshape((num, 1))
        delta0 = deltat20.copy()
        deltat = np.zeros((num, 1))

        dec = np.zeros((num, 3))
        # flag = np.zeros((num, 1))

        # 状态向量：共计16个特征）
        init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.default_paras), np.copy(self.default_index_performance), delta0, deltat20, deltat, dec), axis=1)  # 状态向量只用当前性能，不用当前最佳性能

        init_state[:, 0] = np.log10(init_state[:, 0])
        init_state[:, 2:4] = np.log10(init_state[:, 2:4])
        init_state[:, 5] = np.log10(init_state[:, 5])
        init_state[:, 7:9] = np.log10(init_state[:, 7:9])

        init_state_ = self.state_scaler.transform(init_state)  # 状态向量要归一化再存储

        return init_state_

    def _get_next_state(self, cur_index_performance, best_index_performance, cur_paras, best_paras, target_rec, num):
        best_rec = best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = (cur_rec - target_rec).reshape((num, 1))
        deltat20 = (best_rec - target_rec).reshape((num, 1))
        deltat = (cur_rec - best_rec).reshape((num, 1))

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        ct_counts_dec = ct_counts_dec.reshape((num, 1))
        st_counts_dec = st_counts_dec.reshape((num, 1))
        target_dec = target_dec.reshape((num, 1))

        next_state = np.concatenate((cur_paras, best_paras, np.copy(cur_index_performance),
                                     delta0, deltat20, deltat, ct_counts_dec, st_counts_dec, target_dec), axis=1)
        return next_state

    def _step(self, actions, data_feature, performance_filename, paras_filename):  # paras是推荐的参数（没有经过归一化处理）， data_feature是数据集输入特征（也没有经过归一化）
        self.steps += 1  # 执行一次就加一次

        num = self.target_rec.shape[0]
        target_rec = self.target_rec.reshape(-1)

        cur_paras = self._get_action(actions)

        feature_input = np.concatenate((np.copy(cur_paras), np.copy(data_feature)), axis=1)
        feature_input[:, 0] = np.log10(feature_input[:, 0])
        feature_input[:, 2] = np.log10(feature_input[:, 2])

        cur_index_performance = self._get_index_performance(feature_input)

        reward, average_reward = self._get_reward3(cur_index_performance)

        # global方式
        _ = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        best_now_performance = self._get_best_now(performance_filename)
        best_now_paras = self._get_best_paras_now(paras_filename)

        self.last_index_performance = best_now_performance.copy()  # 更新最优性能

        # local方式
        # self._record_best(index_performance, paras, performance_filename, paras_filename)
        # self._update_episode_best2(index_performance)

        # general方式
        # self.last_index_performance = cur_index_performance.copy()
        # _ = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        # best_now_performance = self._get_best_now(performance_filename)
        # best_now_paras = self._get_best_paras_now(paras_filename)

        # condition = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        # reward = reward.reshape(-1)  #在general方式下，如果当前性能超过了当前最优性能，则同样扩大
        # reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward)
        # reward = reward.reshape((reward.shape[0], 1))

        # best_now_performance = self._get_best_now(performance_filename)
        # best_now_paras = self._get_best_paras_now(paras_filename)

        next_state = self._get_next_state(cur_index_performance, best_now_performance, cur_paras, best_now_paras, target_rec, num)

        next_state[:, 0] = np.log10(next_state[:, 0])
        next_state[:, 2:4] = np.log10(next_state[:, 2:4])
        next_state[:, 5] = np.log10(next_state[:, 5])
        next_state[:, 7:9] = np.log10(next_state[:, 7:9])

        next_state_ = self.state_scaler.transform(next_state)  # 状态向量要归一化再存储

        # if self.steps < self.max_steps and self.score >= -10000:
        #     terminate = np.zeros((num, 1), dtype=bool)
        # else:
        #     terminate = np.ones((num, 1), dtype=bool)
        # print(average_reward)
        # if average_reward >= -50:
        #     terminate = np.zeros((num, 1), dtype=bool)
        # else:
        #     terminate = np.ones((num, 1), dtype=bool)

        # if self.nochange_steps == self.args_r.nochange_steps:   #用性能连续没有改进的步数是否达到设定阈值来判断是否样稿终止当前episode，达到了则将terminate设置为True，终止当前episode
        #     terminate = np.ones((num, 1), dtype=bool)
        # else:
        #     terminate = np.zeros((num, 1), dtype=bool)
        terminate = np.zeros((num, 1), dtype=bool)

        return reward, next_state_, terminate, self.score, average_reward, cur_index_performance, cur_paras

    @staticmethod
    def _calculate_reward(delta):
        reward_positive = (1 + delta) ** 2 - 1  # 后面再测试一下reward_positive = (1 + delta) ** 2 - 1 ，奖励不同确实影响很大。
        reward_negative = -(1 - delta) ** 2 + 1

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta >= 0, reward_positive, reward_negative)
        # _reward = np.where(delta >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec(delta0, deltat):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        reward_positive = ((1 + delta0) ** 2 - 1) * np.abs(1 + deltat)
        reward_negative = -((1 - delta0) ** 2 - 1) * np.abs(1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta0 >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec2(delta0, deltat, deltat20):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        _reward = np.zeros_like(delta0)

        cond1 = (deltat20 < 0) & (delta0 < 0)
        cond2 = (deltat20 < 0) & (delta0 >= 0)
        cond3 = (deltat20 > 0) & (delta0 < 0)

        reward_cond1 = -(1 - delta0) ** 2 + 1
        reward_cond2 = ((1 + delta0) ** 2) * (1 + deltat)
        reward_cond3 = (-(1 - delta0) ** 2) * (1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward[cond1] = reward_cond1[cond1]
        _reward[cond2] = reward_cond2[cond2]
        _reward[cond3] = reward_cond3[cond3]

        return _reward

    def _get_reward3(self, cur_index_performance):  # 优先使用reward1，reward1是分别计算ct和st的奖励，再加权求和
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = cur_rec - target_rec
        deltat = cur_rec - best_rec
        deltat20 = best_rec - target_rec

        reward = self._calculate_reward_rec2(delta0, deltat, deltat20)

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance,
                                                                         cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec
        # min_dec = np.min(target_dec)

        # 分开计算奖励再综合
        # ct_counts_reward = self._calculate_reward(ct_counts_dec)
        # st_counts_reward = self._calculate_reward(st_counts_dec)
        # counts_reward = self.args_r.lamb * ct_counts_reward + st_counts_reward

        # 综合计算奖励
        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        average_reward = np.mean(reward)
        self.score += average_reward

        # reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward) #不注释是global，注释是general

        reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward

class IndexEnv_serial(object):  # 创建环境
    def __init__(self, num_dataset, default_performance, args_r, args_p, predict_model_save_path, standard_path, dv):
        self.device = dv

        # print('-------------加载索引性能预测模型-------------')
        self.args_p = args_p
        self.predict_model = Direct_Predict_MLP(eval(args_p.dipredict_layer_sizes)).to(self.device)
        self._get_predict_model(predict_model_save_path)  # 加载索引性能预测模型

        self.feature_scaler = Scaler_minmax_new_gpu(6, dv)
        self.performance_scaler = Scaler_minmax_new_gpu(0, dv)
        self.para_scaler = Scaler_para()  # 这个操作的对象是numpy
        #self.state_scaler = Scaler_state(10)  ##这个操作的对象也是numpy
        self.state_scaler = Scaler_state_new(15)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.args_r = args_r

        self.target_rec = np.array([[0.9], [0.92], [0.94], [0.95], [0.96], [0.98], [0.99]])

        self.score = 0.0
        # self.score_array = np.tile(np.array([0]), self.target_rec.shape[0])
        self.steps = 0
        self.max_steps = args_r.max_steps
        self.nochange_steps = 0
        self.index = 0

        # print(self.target_rec)
        self.default_paras = np.tile(np.array([[20, 4, 10]]), (7, 1))  # (7*num_dataset) * 1
        self.last_index_performance = None  # (num_dataset*7) * 1
        self.whole_default_index_performance = default_performance.copy()  # (num_dataset*7) * 1
        self.default_index_performance = None

    def _get_predict_model(self, model_save_path):
        optimizer = optim.Adam(self.predict_model.parameters(), lr=self.args_p.dml_lr, weight_decay=self.args_p.weight_decay)
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path):
        self.feature_scaler.load_parameters(None, standard_path, self.device)
        # self.state_scaler.load_parameters(standard_path)

    def _get_action(self, actions):  # 输入的actions是一个N*3的数组，在传入到这里之前actions已经是一个在cpu上的数组
        paras = self.para_scaler.inverse_transform(actions)  # N*3的数组
        paras[:, 0] = np.power(10, paras[:, 0])
        paras[:, 2] = np.power(10, paras[:, 2])
        paras = np.floor(paras + 0.5)  # 要4舍5入转成整数

        paras[:, 0] = np.where(paras[:, 0] < paras[:, 1], paras[:, 1], paras[:, 0])

        return paras

    def _get_index_performance(self, feature_input):
        # 预测索引性能，输入feature_input是一个N*14的张量，没有经过归一化
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        # batch_size = 20480
        # num = feature_input_scaled.size()[0]
        #
        # n_batch = (num + batch_size - 1) // batch_size
        #
        # index_performance_lis = []
        #
        # with torch.no_grad():
        #     for idx in range(n_batch):
        #         batch_start = idx * batch_size
        #         batch_end = min(num, (idx + 1) * batch_size)
        #
        #         temp_feature_input_scaled = feature_input_scaled[batch_start: batch_end, :]
        #         temp_index_performance = self.predict_model(temp_feature_input_scaled)
        #
        #         index_performance_lis.append(temp_index_performance)
        #
        # index_performance = torch.cat(index_performance_lis, dim=0)
        # index_performance = index_performance.cpu().numpy()  # 要用逆归一化后的数据

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1:] = torch.pow(10, real_index_performance[:, 1:])

        real_index_performance = real_index_performance.cpu().numpy()

        return real_index_performance  # 同时返回归一化后的性能和真实性能，一个用于组成状态向量，一个用于更新最优性能和计算奖励

    @staticmethod
    def _get_performance_improvement(last_index_performance, current_index_performance):  # 这里其实要考虑的是用真实性能计算还是归一化后的性能计算
        ct_counts_dec = (last_index_performance[:, 1] - current_index_performance[:, 1]) / last_index_performance[:, 1]
        st_counts_dec = (last_index_performance[:, 2] - current_index_performance[:, 2]) / last_index_performance[:, 2]

        return ct_counts_dec, st_counts_dec

    def _get_best_now(self, filename):
        with open(filename, 'rb') as f:
            whole_best_index_performance = pickle.load(f)
        return whole_best_index_performance

    def _get_best_paras_now(self, filename):
        with open(filename, 'rb') as f:
            whole_best_paras = pickle.load(f)
        return whole_best_paras

    def _record_best2(self, cur_index_performance, cur_paras, performance_filename, paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            whole_best_index_performance = self._get_best_now(performance_filename)
            whole_best_paras = self._get_best_paras_now(paras_filename)

            if self.index not in whole_best_index_performance.keys():
                whole_best_index_performance[self.index] = cur_index_performance
                whole_best_paras[self.index] = cur_paras

                with open(performance_filename, 'wb') as f:
                    pickle.dump(whole_best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(whole_best_paras, f)
            else:
                best_index_performance = whole_best_index_performance[self.index]
                best_paras = whole_best_paras[self.index]

                target_rec = self.target_rec.reshape(-1)
                # print(target_rec.shape)
                best_rec = best_index_performance[:, 0]
                # print(best_rec.shape)
                cur_rec = cur_index_performance[:, 0]
                # print(cur_rec.shape)

                ct_counts_dec, st_counts_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
                # print(ct_counts_dec.shape)
                # print(st_counts_dec.shape)
                
                target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec
                
                cond1 = (best_rec < target_rec)
                cond2 = (best_rec >= target_rec)
                cond3 = (cur_rec < target_rec)
                cond4 = (cur_rec >= target_rec)
                cond5 = (cur_rec > best_rec)
                cond6 = (target_dec > 0)

                # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
                cond_a = (cond1 & cond4)
                # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
                cond_b = (cond1 & cond3 & cond5)
                # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
                cond_c = (cond2 & cond4 & cond6)

                if not (cond_a.any() or cond_b.any() or cond_c.any()):
                    self.nochange_steps += 1
                else:
                    self.nochange_steps = 0

                    best_index_performance[cond_a] = cur_index_performance[cond_a]
                    best_paras[cond_a] = cur_paras[cond_a]

                    best_index_performance[cond_b] = cur_index_performance[cond_b]
                    best_paras[cond_b] = cur_paras[cond_b]

                    best_index_performance[cond_c] = cur_index_performance[cond_c]
                    best_paras[cond_c] = cur_paras[cond_c]

                    whole_best_index_performance[self.index] = best_index_performance
                    whole_best_paras[self.index] = best_paras

                    with open(performance_filename, 'wb') as f:
                        pickle.dump(whole_best_index_performance, f)

                    with open(paras_filename, 'wb') as f:
                        pickle.dump(whole_best_paras, f)

        else:
            whole_best_index_performance = {}
            whole_best_paras = {}

            whole_best_index_performance[self.index] = cur_index_performance
            whole_best_paras[self.index] = cur_paras

            with open(performance_filename, 'wb') as f:
                pickle.dump(whole_best_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(whole_best_paras, f)

    def _update_episode_best2(self, cur_index_performance):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算                                                  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance, cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        cond1 = (best_rec < target_rec)
        cond2 = (best_rec >= target_rec)
        cond3 = (cur_rec < target_rec)
        cond4 = (cur_rec >= target_rec)
        cond5 = (cur_rec > best_rec)
        cond6 = (target_dec > 0)

        # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
        cond_a = (cond1 & cond4)
            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
        cond_b = (cond1 & cond3 & cond5)
        # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
        cond_c = (cond2 & cond4 & cond6)

        if not (cond_a.any() or cond_b.any() or cond_c.any()):
            self.nochange_steps += 1
        else:
            self.nochange_steps = 0

            self.last_index_performance[cond_a] = cur_index_performance[cond_a]
            self.last_index_performance[cond_b] = cur_index_performance[cond_b]
            self.last_index_performance[cond_c] = cur_index_performance[cond_c]

    def _initialize(self):
        self.steps = 0
        self.nochange_steps = 0

        base_index_performance = self.whole_default_index_performance[self.index * 7 : (self.index+1) * 7, :]

        self.defaule_index_performance = base_index_performance.copy()
        self.last_index_performance = base_index_performance.copy()

        num = base_index_performance.shape[0]

        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        # cur_rec = cur_index_performance[:, 0]

        deltat20 = (best_rec - target_rec).reshape((num, 1))
        delta0 = deltat20.copy()
        deltat = np.zeros((num, 1))

        dec = np.zeros((num, 3))
        # flag = np.zeros((num, 1))

        # 状态向量：共计15个特征）
        init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.default_paras), base_index_performance, delta0, deltat20, deltat, dec), axis=1)  # 状态向量只用当前性能，不用当前最佳性能

        init_state[:, 0] = np.log10(init_state[:, 0])
        init_state[:, 2:4] = np.log10(init_state[:, 2:4])
        init_state[:, 5] = np.log10(init_state[:, 5])
        init_state[:, 7:9] = np.log10(init_state[:, 7:9])

        init_state_ = self.state_scaler.transform(init_state)

        return init_state_

    def _get_next_state(self, cur_index_performance, best_index_performance, cur_paras, best_paras, target_rec, num):
        best_rec = best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = (cur_rec - target_rec).reshape((num, 1))
        deltat20 = (best_rec - target_rec).reshape((num, 1))
        deltat = (cur_rec - best_rec).reshape((num, 1))

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        ct_counts_dec = ct_counts_dec.reshape((num, 1))
        st_counts_dec = st_counts_dec.reshape((num, 1))
        target_dec = target_dec.reshape((num, 1))

        next_state = np.concatenate((cur_paras, best_paras, np.copy(cur_index_performance),
                                     delta0, deltat20, deltat, ct_counts_dec, st_counts_dec, target_dec), axis=1)
        return next_state

    def _step(self, actions, data_feature, performance_filename, paras_filename):  # paras是推荐的参数（没有经过归一化处理）， data_feature是数据集输入特征（也没有经过归一化）
        self.steps += 1  # 执行一次就加一次

        num = self.target_rec.shape[0]
        target_rec = self.target_rec.reshape(-1)

        cur_paras = self._get_action(actions)

        feature_input = np.concatenate((np.copy(cur_paras), np.copy(data_feature)), axis=1)
        feature_input[:, 0] = np.log10(feature_input[:, 0])
        feature_input[:, 2] = np.log10(feature_input[:, 2])

        cur_index_performance = self._get_index_performance(feature_input)

        reward, average_reward = self._get_reward3(cur_index_performance)

        # global方式
        self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        whole_best_index_performance = self._get_best_now(performance_filename)
        whole_best_paras = self._get_best_paras_now(paras_filename)

        best_now_performance = whole_best_index_performance[self.index]
        best_now_paras = whole_best_paras[self.index]

        self.last_index_performance = best_now_performance .copy()

        # # general方式
        # self.last_index_performance = cur_index_performance.copy()

        # self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        # whole_best_index_performance = self._get_best_now(performance_filename)
        # whole_best_paras = self._get_best_paras_now(paras_filename)

        # best_now_performance = whole_best_index_performance[self.index]
        # best_now_paras = whole_best_paras[self.index]

        next_state = self._get_next_state(cur_index_performance, best_now_performance, cur_paras, best_now_paras, target_rec, num)

        next_state[:, 0] = np.log10(next_state[:, 0])
        next_state[:, 2:4] = np.log10(next_state[:, 2:4])
        next_state[:, 5] = np.log10(next_state[:, 5])
        next_state[:, 7:9] = np.log10(next_state[:, 7:9])

        next_state_ = self.state_scaler.transform(next_state)

        terminate = np.zeros((num, 1), dtype=bool)

        return reward, next_state_, terminate, self.score, average_reward, cur_index_performance, cur_paras

    @staticmethod
    def _calculate_reward(delta):
        reward_positive = (1 + delta) ** 2 - 1   #后面再测试一下reward_positive = (1 + delta) ** 2 - 1 ，奖励不同确实影响很大。
        reward_negative = -(1 - delta) ** 2 + 1

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta >= 0, reward_positive, reward_negative)
        # _reward = np.where(delta >= 0, reward_positive, reward_negative)

        return _reward
 
    @staticmethod
    def _calculate_reward_rec(delta0, deltat):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        reward_positive = ((1 + delta0) ** 2 - 1) * np.abs(1 + deltat)
        reward_negative = -((1 - delta0) ** 2 - 1) * np.abs(1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta0 >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec2(delta0, deltat, deltat20):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值 
        _reward = np.zeros_like(delta0)

        cond1 = (deltat20 < 0 ) & (delta0 < 0)
        cond2 = (deltat20 < 0 ) & (delta0 >= 0)
        cond3 = (deltat20 > 0 ) & (delta0 < 0)

        reward_cond1 = -(1 - delta0) ** 2 + 1
        reward_cond2 = ((1 + delta0) ** 2) * (1 + deltat) 
        reward_cond3 = (-(1 - delta0) ** 2) * (1 - deltat) 

 
        # 根据 delta0 的值选择正负奖励
        _reward[cond1] = reward_cond1[cond1]
        _reward[cond2] = reward_cond2[cond2]
        _reward[cond3] = reward_cond3[cond3]
        
        return _reward

    def _get_reward3(self, cur_index_performance):  # 优先使用reward1，reward1是分别计算ct和st的奖励，再加权求和
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = cur_rec - target_rec
        deltat = cur_rec - best_rec
        deltat20 = best_rec - target_rec

        reward = self._calculate_reward_rec2(delta0, deltat, deltat20)

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance, cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec
        # min_dec = np.min(target_dec)

        # 分开计算奖励再综合
        # ct_counts_reward = self._calculate_reward(ct_counts_dec)
        # st_counts_reward = self._calculate_reward(st_counts_dec)
        # counts_reward = self.args_r.lamb * ct_counts_reward + st_counts_reward

        # 综合计算奖励
        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        average_reward = np.mean(reward)
        self.score += average_reward

        #reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward) #不注释是global，注释是general

        reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward


class IndexEnv_new_state_new(object):  # 最新的环境，见2024.09.01的日志记录
    def __init__(self, num_dataset, default_performance, target_rec_lis, args_r, args_p, predict_model_save_path, standard_path, dv):
        self.device = dv

        # print('-------------加载索引性能预测模型-------------')
        self.args_p = args_p
        self.predict_model = Direct_Predict_MLP(eval(args_p.dipredict_layer_sizes)).to(self.device)
        self._get_predict_model(predict_model_save_path)  # 加载索引性能预测模型

        self.feature_scaler = Scaler_minmax_new_gpu(6, dv)
        self.performance_scaler = Scaler_minmax_new_gpu(0, dv)
        self.para_scaler = Scaler_para()  # 这个操作的对象是numpy
        # self.state_scaler = Scaler_state_new(15)  ##这个操作的对象也是numpy
        self.state_scaler = Scaler_state_new_new(13)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.args_r = args_r

        self.target_rec = np.tile(np.array(target_rec_lis).reshape(-1, 1), (num_dataset, 1))  #target_rec_lis=[0.85, 0.875, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]

        self.score = 0.0
        # self.score_array = np.tile(np.array([0]), self.target_rec.shape[0])
        self.steps = 0
        self.max_steps = args_r.max_steps
        self.nochange_steps = 0

        # print(self.target_rec)
        self.default_paras = np.tile(np.array([[20, 4, 10]]), (self.target_rec.shape[0], 1))  
        self.last_index_performance = default_performance.copy()  
        self.default_index_performance = default_performance.copy()  
        self.best_index_performance = default_performance.copy() 

    def _get_predict_model(self, model_save_path):
        optimizer = optim.Adam(self.predict_model.parameters(), lr=self.args_p.dml_lr,
                               weight_decay=self.args_p.weight_decay)
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path):
        self.feature_scaler.load_parameters(None, standard_path, self.device)
        # self.state_scaler.load_parameters(standard_path)

    def _get_action(self, actions):  # 输入的actions是一个N*3的数组，在传入到这里之前actions已经是一个在cpu上的数组
        paras = self.para_scaler.inverse_transform(actions)  # N*3的数组
        paras[:, 0] = np.power(10, paras[:, 0])
        paras[:, 2] = np.power(10, paras[:, 2])
        paras = np.floor(paras + 0.5)  # 要4舍5入转成整数

        paras[:, 0] = np.where(paras[:, 0] < paras[:, 1], paras[:, 1], paras[:, 0])

        return paras

    def _get_index_performance(self, feature_input):
        # 预测索引性能，输入feature_input是一个N*14的张量，没有经过归一化
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        # index_performance = torch.cat(index_performance_lis, dim=0)
        # index_performance = index_performance.cpu().numpy()  # 要用逆归一化后的数据

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1:] = torch.pow(10, real_index_performance[:, 1:])

        real_index_performance = real_index_performance.cpu().numpy()

        return real_index_performance  # 同时返回归一化后的性能和真实性能，一个用于组成状态向量，一个用于更新最优性能和计算奖励

    @staticmethod
    def _get_performance_improvement(last_index_performance, current_index_performance):  # 这里其实要考虑的是用真实性能计算还是归一化后的性能计算
        st_counts_dec = (last_index_performance[:, 2] - current_index_performance[:, 2]) / last_index_performance[:, 2]

        return st_counts_dec

    def _get_best_now(self, filename):
        with open(filename, 'rb') as f:
            best_index_performance = pickle.load(f)
        return best_index_performance

    def _get_best_paras_now(self, filename):
        with open(filename, 'rb') as f:
            best_paras = pickle.load(f)
        return best_paras

    def _record_best2(self, cur_index_performance, cur_paras, performance_filename, paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            best_rec = best_index_performance[:, 0]
            cur_rec = cur_index_performance[:, 0]

            target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)

            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)

            # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
            cond_a = (cond1 & cond4)
            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
            cond_b = (cond1 & cond3 & cond5)
            # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
            cond_c = (cond2 & cond4 & cond6)

            if cond_a.any() or cond_b.any() or cond_c.any():
                self.nochange_steps = 0

                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)

            else:
                self.nochange_steps += 1

            return cond_c

        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

            return np.zeros(cur_index_performance.shape[0], dtype=bool)

    def _initialize(self):
        self.steps = 0
        self.score = 0.0
        self.nochange_steps = 0

        num = self.default_index_performance.shape[0]

        self.last_index_performance = self.default_index_performance.copy()  #self.last_index_performance 存储上一次的性能
        self.best_index_performance = self.default_index_performance.copy()  #self.best_index_performance 存储当前最优性能

        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]

        deltat20 = (best_rec - target_rec).reshape((num, 1))
        delta0 = deltat20.copy()
        deltat = np.zeros((num, 1))

        best_dec = np.zeros((num, 1))  #当前性能相对于当前最好性能的改进
        last_dec = np.zeros((num, 1))  #当前性能相对于上一次性能的改进

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = self.default_index_performance[:, 0] #只需要召回率和查询执行的平均距离计算次数
        cur_state_index_performance[:, 1] = self.default_index_performance[:, 2]


        # 状态向量：共计13个特征）
        init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.default_paras), cur_state_index_performance, delta0, deltat20, deltat, last_dec, best_dec), axis=1) 

        init_state[:, 0] = np.log10(init_state[:, 0])
        init_state[:, 2:4] = np.log10(init_state[:, 2:4])
        init_state[:, 5] = np.log10(init_state[:, 5])
        init_state[:, 7] = np.log10(init_state[:, 7])

        init_state_ = self.state_scaler.transform(init_state)  # 状态向量要归一化再存储

        return init_state_

    def _get_next_state(self, cur_index_performance, last_index_performance, best_index_performance, cur_paras, best_paras, target_rec, num):
        best_rec = best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = (cur_rec - target_rec).reshape((num, 1))
        deltat20 = (best_rec - target_rec).reshape((num, 1))
        deltat = (cur_rec - best_rec).reshape((num, 1))

        last_st_counts_dec = self._get_performance_improvement(last_index_performance, cur_index_performance)
        best_st_counts_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)

        last_st_counts_dec= last_st_counts_dec.reshape((num, 1))
        best_st_counts_dec= best_st_counts_dec.reshape((num, 1))

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0] #只需要召回率和查询执行的平均距离计算次数
        cur_state_index_performance[:, 1] = cur_index_performance[:, 2]

        next_state = np.concatenate((cur_paras, best_paras, cur_state_index_performance, delta0, deltat20, deltat, last_st_counts_dec, best_st_counts_dec), axis=1)
        return next_state

    def _step(self, actions, data_feature, performance_filename, paras_filename):  # paras是推荐的参数（没有经过归一化处理）， data_feature是数据集输入特征（也没有经过归一化）
        self.steps += 1  # 执行一次就加一次

        num = self.target_rec.shape[0]
        target_rec = self.target_rec.reshape(-1)

        cur_paras = self._get_action(actions)

        feature_input = np.concatenate((np.copy(cur_paras), np.copy(data_feature)), axis=1)
        feature_input[:, 0] = np.log10(feature_input[:, 0])
        feature_input[:, 2] = np.log10(feature_input[:, 2])

        cur_index_performance = self._get_index_performance(feature_input)

        reward, average_reward = self._get_reward3(cur_index_performance)

        # global方式
        _ = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        best_now_performance = self._get_best_now(performance_filename)
        best_now_paras = self._get_best_paras_now(paras_filename)

        self.best_index_performance = best_now_performance.copy()  # 更新最优性能

        # general方式
        # _ = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        # best_now_performance = self._get_best_now(performance_filename)
        # best_now_paras = self._get_best_paras_now(paras_filename)

        # self.best_index_performance = best_now_performance.copy()  # 更新最优性能

        # condition = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        # reward = reward.reshape(-1)  #在general方式下，如果当前性能超过了当前最优性能，则同样扩大
        # reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward)
        # reward = reward.reshape((reward.shape[0], 1))

        # best_now_performance = self._get_best_now(performance_filename)
        # best_now_paras = self._get_best_paras_now(paras_filename)

        next_state = self._get_next_state(cur_index_performance, np.copy(self.last_index_performance), best_now_performance, cur_paras, best_now_paras, target_rec, num)

        next_state[:, 0] = np.log10(next_state[:, 0])
        next_state[:, 2:4] = np.log10(next_state[:, 2:4])
        next_state[:, 5] = np.log10(next_state[:, 5])
        next_state[:, 7] = np.log10(next_state[:, 7])

        next_state_ = self.state_scaler.transform(next_state)  # 状态向量要归一化再存储

        self.last_index_performance = cur_index_performance.copy() #要更新完状态后才能更新上一次的性能

        # if self.steps < self.max_steps and self.score >= -10000:
        #     terminate = np.zeros((num, 1), dtype=bool)
        # else:
        #     terminate = np.ones((num, 1), dtype=bool)
        # print(average_reward)
        # if average_reward >= -50:
        #     terminate = np.zeros((num, 1), dtype=bool)
        # else:
        #     terminate = np.ones((num, 1), dtype=bool)

        # if self.nochange_steps == self.args_r.nochange_steps:   #用性能连续没有改进的步数是否达到设定阈值来判断是否样稿终止当前episode，达到了则将terminate设置为True，终止当前episode
        #     terminate = np.ones((num, 1), dtype=bool)
        # else:
        #     terminate = np.zeros((num, 1), dtype=bool)
        terminate = np.zeros((num, 1), dtype=bool)

        return reward, next_state_, terminate, self.score, average_reward, cur_index_performance, cur_paras

    @staticmethod
    def _calculate_reward(delta):
        reward_positive = (1 + delta) ** 2 - 1  # 后面再测试一下reward_positive = (1 + delta) ** 2 - 1 ，奖励不同确实影响很大。
        reward_negative = -(1 - delta) ** 2 + 1

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta >= 0, reward_positive, reward_negative)
        # _reward = np.where(delta >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec(delta0, deltat):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        reward_positive = ((1 + delta0) ** 2 - 1) * np.abs(1 + deltat)
        reward_negative = -((1 - delta0) ** 2 - 1) * np.abs(1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta0 >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec2(delta0, deltat, deltat20):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        _reward = np.zeros_like(delta0)

        cond1 = (deltat20 < 0) & (delta0 < 0)
        cond2 = (deltat20 < 0) & (delta0 >= 0)
        cond3 = (deltat20 > 0) & (delta0 < 0)

        reward_cond1 = -(1 - delta0) ** 2 + 1
        reward_cond2 = ((1 + delta0) ** 2) * (1 + deltat)
        reward_cond3 = (-(1 - delta0) ** 2) * (1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward[cond1] = reward_cond1[cond1]
        _reward[cond2] = reward_cond2[cond2]
        _reward[cond3] = reward_cond3[cond3]

        return _reward

    def _get_reward3(self, cur_index_performance):  # 优先使用reward1，reward1是分别计算ct和st的奖励，再加权求和
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0] #global方式用self.best_index_performance，general用self.best_index_performance
        cur_rec = cur_index_performance[:, 0]

        delta0 = cur_rec - target_rec
        deltat = cur_rec - best_rec
        deltat20 = best_rec - target_rec

        reward = self._calculate_reward_rec2(delta0, deltat, deltat20)

        st_counts_dec = self._get_performance_improvement(self.best_index_performance, cur_index_performance)  #global方式用self.best_index_performance，general用self.best_index_performance

        target_dec = st_counts_dec
        # min_dec = np.min(target_dec)

        # 分开计算奖励再综合
        # ct_counts_reward = self._calculate_reward(ct_counts_dec)
        # st_counts_reward = self._calculate_reward(st_counts_dec)
        # counts_reward = self.args_r.lamb * ct_counts_reward + st_counts_reward

        # 综合计算奖励
        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        average_reward = np.mean(reward)
        self.score += average_reward

        #reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward) #不注释是global，注释是general

        reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward

class IndexEnv_new_state_onlys(object):  # 见2024.09.02的日志记录
    def __init__(self, num_dataset, default_performance, target_rec_lis, args_r, args_p, predict_model_save_path, standard_path, dv):
        self.device = dv

        # print('-------------加载索引性能预测模型-------------')
        self.args_p = args_p
        self.predict_model = Direct_Predict_MLP(eval(args_p.dipredict_layer_sizes)).to(self.device)
        self._get_predict_model(predict_model_save_path)  # 加载索引性能预测模型

        self.feature_scaler = Scaler_minmax_new_gpu_nsg(9, dv)
        self.performance_scaler = Scaler_minmax_new_gpu_nsg(0, dv)
        self.para_scaler = Scaler_para_nsg()  # 这个操作的对象是numpy
        self.state_scaler = Scaler_state_new_onlys_nsg(18)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.args_r = args_r

        self.target_rec = np.tile(np.array(target_rec_lis).reshape(-1, 1), (num_dataset, 1))  #target_rec_lis=[0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]

        self.score = 0.0
        # self.score_array = np.tile(np.array([0]), self.target_rec.shape[0])
        self.steps = 0
        self.max_steps = args_r.max_steps
        self.nochange_steps = 0

        # print(self.target_rec)
        self.default_paras = np.tile(np.array([[20, 4, 10]]), (self.target_rec.shape[0], 1))   
        self.default_index_performance = default_performance.copy()  
        self.best_index_performance = default_performance.copy() 

    def _get_predict_model(self, model_save_path):
        optimizer = optim.Adam(self.predict_model.parameters(), lr=self.args_p.dml_lr,
                               weight_decay=self.args_p.weight_decay)
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path):
        self.feature_scaler.load_parameters(None, standard_path, self.device)
        # self.state_scaler.load_parameters(standard_path)

    def _get_action(self, actions):  # 输入的actions是一个N*3的数组，在传入到这里之前actions已经是一个在cpu上的数组
        paras = self.para_scaler.inverse_transform(actions)  # N*3的数组
        paras[:, 0] = np.power(10, paras[:, 0])
        paras[:, 2] = np.power(10, paras[:, 2])
        paras = np.floor(paras + 0.5)  # 要4舍5入转成整数

        paras[:, 0] = np.where(paras[:, 0] < paras[:, 1], paras[:, 1], paras[:, 0])

        return paras

    def _get_index_performance(self, feature_input):
        # 预测索引性能，输入feature_input是一个N*14的张量，没有经过归一化
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        # index_performance = torch.cat(index_performance_lis, dim=0)
        # index_performance = index_performance.cpu().numpy()  # 要用逆归一化后的数据

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1:] = torch.pow(10, real_index_performance[:, 1:])

        real_index_performance = real_index_performance.cpu().numpy()

        return real_index_performance  # 同时返回归一化后的性能和真实性能，一个用于组成状态向量，一个用于更新最优性能和计算奖励

    @staticmethod
    def _get_performance_improvement(last_index_performance, current_index_performance):  # 这里其实要考虑的是用真实性能计算还是归一化后的性能计算
        st_counts_dec = (last_index_performance[:, 2] - current_index_performance[:, 2]) / last_index_performance[:, 2]

        return st_counts_dec

    def _get_best_now(self, filename):
        with open(filename, 'rb') as f:
            best_index_performance = pickle.load(f)
        return best_index_performance

    def _get_best_paras_now(self, filename):
        with open(filename, 'rb') as f:
            best_paras = pickle.load(f)
        return best_paras

    def _record_best2(self, cur_index_performance, cur_paras, performance_filename, paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            best_rec = best_index_performance[:, 0]
            cur_rec = cur_index_performance[:, 0]

            target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)

            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)

            # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
            cond_a = (cond1 & cond4)
            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
            cond_b = (cond1 & cond3 & cond5)
            # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
            cond_c = (cond2 & cond4 & cond6)

            if cond_a.any() or cond_b.any() or cond_c.any():
                self.nochange_steps = 0

                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)

            else:
                self.nochange_steps += 1

            return cond_c

        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

            return np.zeros(cur_index_performance.shape[0], dtype=bool)

    def _initialize(self):
        self.steps = 0
        self.score = 0.0
        self.nochange_steps = 0

        num = self.default_index_performance.shape[0]

        self.best_index_performance = self.default_index_performance.copy()  #self.best_index_performance 存储当前最优性能

        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]

        deltat20 = (best_rec - target_rec).reshape((num, 1))
        delta0 = deltat20.copy()
        deltat = np.zeros((num, 1))

        target_dec = np.zeros((num, 1))  #当前性能相对于当前最好性能的改进

        cur_index_performance = self.default_index_performance.copy()
        
        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0] #只需要召回率和查询执行的平均距离计算次数
        cur_state_index_performance[:, 1] = cur_index_performance[:, 2]


        # 状态向量：共计13个特征）
        init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.default_paras), cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1) 

        init_state[:, 0] = np.log10(init_state[:, 0])
        init_state[:, 2:4] = np.log10(init_state[:, 2:4])
        init_state[:, 5] = np.log10(init_state[:, 5])
        init_state[:, 7] = np.log10(init_state[:, 7])

        init_state_ = self.state_scaler.transform(init_state)  # 状态向量要归一化再存储

        return init_state_

    def _get_next_state(self, cur_index_performance, best_index_performance, cur_paras, best_paras, target_rec, num):
        best_rec = best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = (cur_rec - target_rec).reshape((num, 1))
        deltat20 = (best_rec - target_rec).reshape((num, 1))
        deltat = (cur_rec - best_rec).reshape((num, 1))

        target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
        target_dec= target_dec.reshape((num, 1))

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0] #只需要召回率和查询执行的平均距离计算次数
        cur_state_index_performance[:, 1] = cur_index_performance[:, 2]

        next_state = np.concatenate((cur_paras, best_paras, cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1)
        return next_state

    def _step(self, actions, data_feature, performance_filename, paras_filename):  # paras是推荐的参数（没有经过归一化处理）， data_feature是数据集输入特征（也没有经过归一化）
        self.steps += 1  # 执行一次就加一次

        num = self.target_rec.shape[0]
        target_rec = self.target_rec.reshape(-1)

        cur_paras = self._get_action(actions)

        feature_input = np.concatenate((np.copy(cur_paras), np.copy(data_feature)), axis=1)
        feature_input[:, 0] = np.log10(feature_input[:, 0])
        feature_input[:, 2] = np.log10(feature_input[:, 2])

        cur_index_performance = self._get_index_performance(feature_input)

        reward, average_reward = self._get_reward3(cur_index_performance)

        # global方式
        _ = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        best_now_performance = self._get_best_now(performance_filename)
        best_now_paras = self._get_best_paras_now(paras_filename)

        self.best_index_performance = best_now_performance.copy()  # 更新最优性能

        # general方式
        # self.best_index_performance = cur_index_performance.copy()

        # _ = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        # best_now_performance = self._get_best_now(performance_filename)
        # best_now_paras = self._get_best_paras_now(paras_filename)

        # self.best_index_performance = best_now_performance.copy()  # 更新最优性能

        # condition = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        # reward = reward.reshape(-1)  #在general方式下，如果当前性能超过了当前最优性能，则同样扩大
        # reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward)
        # reward = reward.reshape((reward.shape[0], 1))

        # best_now_performance = self._get_best_now(performance_filename)
        # best_now_paras = self._get_best_paras_now(paras_filename)

        next_state = self._get_next_state(cur_index_performance, best_now_performance, cur_paras, best_now_paras, target_rec, num)

        next_state[:, 0] = np.log10(next_state[:, 0])
        next_state[:, 2:4] = np.log10(next_state[:, 2:4])
        next_state[:, 5] = np.log10(next_state[:, 5])
        next_state[:, 7] = np.log10(next_state[:, 7])

        next_state_ = self.state_scaler.transform(next_state)  # 状态向量要归一化再存储

        # if self.steps < self.max_steps and self.score >= -10000:
        #     terminate = np.zeros((num, 1), dtype=bool)
        # else:
        #     terminate = np.ones((num, 1), dtype=bool)
        # print(average_reward)
        # if average_reward >= -50:
        #     terminate = np.zeros((num, 1), dtype=bool)
        # else:
        #     terminate = np.ones((num, 1), dtype=bool)

        # if self.nochange_steps == self.args_r.nochange_steps:   #用性能连续没有改进的步数是否达到设定阈值来判断是否样稿终止当前episode，达到了则将terminate设置为True，终止当前episode
        #     terminate = np.ones((num, 1), dtype=bool)
        # else:
        #     terminate = np.zeros((num, 1), dtype=bool)
        terminate = np.zeros((num, 1), dtype=bool)

        return reward, next_state_, terminate, self.score, average_reward, cur_index_performance, cur_paras

    @staticmethod
    def _calculate_reward(delta):
        reward_positive = (1 + delta) ** 2 - 1  # 后面再测试一下reward_positive = (1 + delta) ** 2 - 1 ，奖励不同确实影响很大。
        reward_negative = -(1 - delta) ** 2 + 1

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta >= 0, reward_positive, reward_negative)
        # _reward = np.where(delta >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec(delta0, deltat):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        reward_positive = ((1 + delta0) ** 2 - 1) * np.abs(1 + deltat)
        reward_negative = -((1 - delta0) ** 2 - 1) * np.abs(1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta0 >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec2(delta0, deltat, deltat20):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        _reward = np.zeros_like(delta0)

        cond1 = (deltat20 < 0) & (delta0 < 0)
        cond2 = (deltat20 < 0) & (delta0 >= 0)
        cond3 = (deltat20 > 0) & (delta0 < 0)

        reward_cond1 = -(1 - delta0) ** 2 + 1
        reward_cond2 = ((1 + delta0) ** 2) * (1 + deltat)
        reward_cond3 = (-(1 - delta0) ** 2) * (1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward[cond1] = reward_cond1[cond1]
        _reward[cond2] = reward_cond2[cond2]
        _reward[cond3] = reward_cond3[cond3]

        return _reward

    def _get_reward3(self, cur_index_performance):  # 优先使用reward1，reward1是分别计算ct和st的奖励，再加权求和
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0] 
        cur_rec = cur_index_performance[:, 0]

        delta0 = cur_rec - target_rec
        deltat = cur_rec - best_rec
        deltat20 = best_rec - target_rec

        reward = self._calculate_reward_rec2(delta0, deltat, deltat20)

        st_counts_dec = self._get_performance_improvement(self.best_index_performance, cur_index_performance)  

        target_dec = st_counts_dec
        # min_dec = np.min(target_dec)

        # 分开计算奖励再综合
        # ct_counts_reward = self._calculate_reward(ct_counts_dec)
        # st_counts_reward = self._calculate_reward(st_counts_dec)
        # counts_reward = self.args_r.lamb * ct_counts_reward + st_counts_reward

        # 综合计算奖励
        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        average_reward = np.mean(reward)
        self.score += average_reward

        reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward) #不注释是global，注释是general

        # reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward


class IndexEnv_new_state_onlys_eval(object):  # 见2024.09.02的日志记录
    def __init__(self, num_dataset, default_performance, target_rec_lis, args_r, args_p, predict_model_save_path,
                 standard_path, dv):
        self.device = dv

        # print('-------------加载索引性能预测模型-------------')
        self.args_p = args_p
        self.predict_model = Direct_Predict_MLP(eval(args_p.dipredict_layer_sizes)).to(self.device)
        self._get_predict_model(predict_model_save_path)  # 加载索引性能预测模型

        self.feature_scaler = Scaler_minmax_new_gpu(6, dv)
        self.performance_scaler = Scaler_minmax_new_gpu(0, dv)
        self.para_scaler = Scaler_para()  # 这个操作的对象是numpy
        # self.state_scaler = Scaler_state_new(15)  ##这个操作的对象也是numpy
        self.state_scaler = Scaler_state_new_onlys(12)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.args_r = args_r

        self.target_rec = np.tile(np.array(target_rec_lis).reshape(-1, 1),
                                  (num_dataset, 1))  # target_rec_lis=[0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
        print(self.target_rec)

        self.score = 0.0
        # self.score_array = np.tile(np.array([0]), self.target_rec.shape[0])
        self.steps = 0
        self.max_steps = args_r.max_steps
        self.nochange_steps = 0
        self.nochange_episodes = 0

        # print(self.target_rec)
        self.default_paras = np.tile(np.array([[20, 4, 10]]), (self.target_rec.shape[0], 1))
        self.default_index_performance = default_performance.copy()
        self.best_index_performance = default_performance.copy()

    def _get_predict_model(self, model_save_path):
        optimizer = optim.Adam(self.predict_model.parameters(), lr=self.args_p.dml_lr,
                               weight_decay=self.args_p.weight_decay)
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path):
        self.feature_scaler.load_parameters(None, standard_path, self.device)
        # self.state_scaler.load_parameters(standard_path)

    def _get_action(self, actions):  # 输入的actions是一个N*3的数组，在传入到这里之前actions已经是一个在cpu上的数组
        paras = self.para_scaler.inverse_transform(actions)  # N*3的数组
        paras[:, 0] = np.power(10, paras[:, 0])
        paras[:, 2] = np.power(10, paras[:, 2])
        paras = np.floor(paras + 0.5)  # 要4舍5入转成整数

        paras[:, 0] = np.where(paras[:, 0] < paras[:, 1], paras[:, 1], paras[:, 0])

        return paras

    def _get_index_performance(self, feature_input):
        # 预测索引性能，输入feature_input是一个N*14的张量，没有经过归一化
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        # index_performance = torch.cat(index_performance_lis, dim=0)
        # index_performance = index_performance.cpu().numpy()  # 要用逆归一化后的数据

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1:] = torch.pow(10, real_index_performance[:, 1:])

        real_index_performance = real_index_performance.cpu().numpy()

        return real_index_performance  # 同时返回归一化后的性能和真实性能，一个用于组成状态向量，一个用于更新最优性能和计算奖励

    @staticmethod
    def _get_performance_improvement(last_index_performance, current_index_performance):  # 这里其实要考虑的是用真实性能计算还是归一化后的性能计算
        st_counts_dec = (last_index_performance[:, 2] - current_index_performance[:, 2]) / last_index_performance[:, 2]

        return st_counts_dec

    def _get_best_now(self, filename):
        with open(filename, 'rb') as f:
            best_index_performance = pickle.load(f)
        return best_index_performance

    def _get_best_paras_now(self, filename):
        with open(filename, 'rb') as f:
            best_paras = pickle.load(f)
        return best_paras

    def _record_best2(self, cur_index_performance, cur_paras, performance_filename, paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            best_rec = best_index_performance[:, 0]
            cur_rec = cur_index_performance[:, 0]

            target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)

            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)

            # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
            cond_a = (cond1 & cond4)
            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
            cond_b = (cond1 & cond3 & cond5)
            # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
            cond_c = (cond2 & cond4 & cond6)

            if cond_a.any() or cond_b.any() or cond_c.any():
                self.nochange_steps = 0

                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)

            else:
                self.nochange_steps += 1

            return cond_c

        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

            return np.zeros(cur_index_performance.shape[0], dtype=bool)

    def _initialize(self):
        self.steps = 0
        self.score = 0.0
        self.nochange_steps = 0

        num = self.default_index_performance.shape[0]

        self.best_index_performance = self.default_index_performance.copy()  # self.best_index_performance 存储当前最优性能

        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]

        deltat20 = (best_rec - target_rec).reshape((num, 1))
        delta0 = deltat20.copy()
        deltat = np.zeros((num, 1))

        target_dec = np.zeros((num, 1))  # 当前性能相对于当前最好性能的改进

        cur_index_performance = self.default_index_performance.copy()

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0]  # 只需要召回率和查询执行的平均距离计算次数
        cur_state_index_performance[:, 1] = cur_index_performance[:, 2]

        # 状态向量：共计13个特征）
        init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.default_paras),
                                     cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1)

        init_state[:, 0] = np.log10(init_state[:, 0])
        init_state[:, 2:4] = np.log10(init_state[:, 2:4])
        init_state[:, 5] = np.log10(init_state[:, 5])
        init_state[:, 7] = np.log10(init_state[:, 7])

        init_state_ = self.state_scaler.transform(init_state)  # 状态向量要归一化再存储

        return init_state_

    def _get_next_state(self, cur_index_performance, best_index_performance, cur_paras, best_paras, target_rec, num):
        best_rec = best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = (cur_rec - target_rec).reshape((num, 1))
        deltat20 = (best_rec - target_rec).reshape((num, 1))
        deltat = (cur_rec - best_rec).reshape((num, 1))

        target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
        target_dec = target_dec.reshape((num, 1))

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0]  # 只需要召回率和查询执行的平均距离计算次数
        cur_state_index_performance[:, 1] = cur_index_performance[:, 2]

        next_state = np.concatenate((cur_paras, best_paras, cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1)
        return next_state

    def _step(self, actions, data_feature, performance_filename,
              paras_filename):  # paras是推荐的参数（没有经过归一化处理）， data_feature是数据集输入特征（也没有经过归一化）
        self.steps += 1  # 执行一次就加一次

        num = self.target_rec.shape[0]
        target_rec = self.target_rec.reshape(-1)

        cur_paras = self._get_action(actions)

        feature_input = np.concatenate((np.copy(cur_paras), np.copy(data_feature)), axis=1)
        feature_input[:, 0] = np.log10(feature_input[:, 0])
        feature_input[:, 2] = np.log10(feature_input[:, 2])

        cur_index_performance = self._get_index_performance(feature_input)

        reward, average_reward = self._get_reward3(cur_index_performance)

        # global方式
        _ = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        best_now_performance = self._get_best_now(performance_filename)
        best_now_paras = self._get_best_paras_now(paras_filename)

        self.best_index_performance = best_now_performance.copy()  # 更新最优性能

        # general方式
        # self.best_index_performance = cur_index_performance.copy()

        # _ = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        # best_now_performance = self._get_best_now(performance_filename)
        # best_now_paras = self._get_best_paras_now(paras_filename)

        # self.best_index_performance = best_now_performance.copy()  # 更新最优性能

        # condition = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        # reward = reward.reshape(-1)  #在general方式下，如果当前性能超过了当前最优性能，则同样扩大
        # reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward)
        # reward = reward.reshape((reward.shape[0], 1))

        # best_now_performance = self._get_best_now(performance_filename)
        # best_now_paras = self._get_best_paras_now(paras_filename)

        next_state = self._get_next_state(cur_index_performance, best_now_performance, cur_paras, best_now_paras, target_rec, num)

        next_state[:, 0] = np.log10(next_state[:, 0])
        next_state[:, 2:4] = np.log10(next_state[:, 2:4])
        next_state[:, 5] = np.log10(next_state[:, 5])
        next_state[:, 7] = np.log10(next_state[:, 7])

        next_state_ = self.state_scaler.transform(next_state)  # 状态向量要归一化再存储

        # if self.steps < self.max_steps and self.score >= -10000:
        #     terminate = np.zeros((num, 1), dtype=bool)
        # else:
        #     terminate = np.ones((num, 1), dtype=bool)
        # print(average_reward)
        # if average_reward >= -50:
        #     terminate = np.zeros((num, 1), dtype=bool)
        # else:
        #     terminate = np.ones((num, 1), dtype=bool)

        # if self.nochange_steps == self.args_r.nochange_steps:   #用性能连续没有改进的步数是否达到设定阈值来判断是否样稿终止当前episode，达到了则将terminate设置为True，终止当前episode
        #     terminate = np.ones((num, 1), dtype=bool)
        # else:
        #     terminate = np.zeros((num, 1), dtype=bool)
        terminate = np.zeros((num, 1), dtype=bool)

        return reward, next_state_, terminate, self.score, average_reward, cur_index_performance, cur_paras

    @staticmethod
    def _calculate_reward(delta):
        reward_positive = (1 + delta) ** 2 - 1  # 后面再测试一下reward_positive = (1 + delta) ** 2 - 1 ，奖励不同确实影响很大。
        reward_negative = -(1 - delta) ** 2 + 1

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta >= 0, reward_positive, reward_negative)
        # _reward = np.where(delta >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec(delta0, deltat):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        reward_positive = ((1 + delta0) ** 2 - 1) * np.abs(1 + deltat)
        reward_negative = -((1 - delta0) ** 2 - 1) * np.abs(1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta0 >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec2(delta0, deltat, deltat20):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        _reward = np.zeros_like(delta0)

        cond1 = (deltat20 < 0) & (delta0 < 0)
        cond2 = (deltat20 < 0) & (delta0 >= 0)
        cond3 = (deltat20 > 0) & (delta0 < 0)

        reward_cond1 = -(1 - delta0) ** 2 + 1
        reward_cond2 = ((1 + delta0) ** 2) * (1 + deltat)
        reward_cond3 = (-(1 - delta0) ** 2) * (1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward[cond1] = reward_cond1[cond1]
        _reward[cond2] = reward_cond2[cond2]
        _reward[cond3] = reward_cond3[cond3]

        return _reward

    def _get_reward3(self, cur_index_performance):  # 优先使用reward1，reward1是分别计算ct和st的奖励，再加权求和
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = cur_rec - target_rec
        deltat = cur_rec - best_rec
        deltat20 = best_rec - target_rec

        reward = self._calculate_reward_rec2(delta0, deltat, deltat20)

        st_counts_dec = self._get_performance_improvement(self.best_index_performance, cur_index_performance)

        target_dec = st_counts_dec
        # min_dec = np.min(target_dec)

        # 分开计算奖励再综合
        # ct_counts_reward = self._calculate_reward(ct_counts_dec)
        # st_counts_reward = self._calculate_reward(st_counts_dec)
        # counts_reward = self.args_r.lamb * ct_counts_reward + st_counts_reward

        # 综合计算奖励
        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        average_reward = np.mean(reward)
        self.score += average_reward

        # reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward) #不注释是global，注释是general

        reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward

class IndexEnv_new_state_eval(object):  # 创建环境
    def __init__(self, num_dataset, default_performance, target_rec_lis, args_r, args_p, predict_model_save_path, standard_path, dv):
        self.device = dv

        # print('-------------加载索引性能预测模型-------------')
        self.args_p = args_p
        self.predict_model = Direct_Predict_MLP(eval(args_p.dipredict_layer_sizes)).to(self.device)
        self._get_predict_model(predict_model_save_path)  # 加载索引性能预测模型

        self.feature_scaler = Scaler_minmax_new_gpu(6, dv)
        self.performance_scaler = Scaler_minmax_new_gpu(0, dv)
        self.para_scaler = Scaler_para()  # 这个操作的对象是numpy
        # self.state_scaler = Scaler_state(10)  ##这个操作的对象也是numpy
        self.state_scaler = Scaler_state_new(15)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.args_r = args_r

        self.target_rec = np.array(target_rec_lis).reshape(-1, 1)
                                  

        self.score = 0.0
        # self.score_array = np.tile(np.array([0]), self.target_rec.shape[0])
        self.steps = 0
        self.max_steps = args_r.max_steps
        self.nochange_steps = 0
        self.nochange_episodes = 0

        # print(self.target_rec)
        self.default_paras = np.tile(np.array([[20, 4, 10]]), (self.target_rec.shape[0], 1))
        self.last_index_performance = default_performance.copy()  
        self.default_index_performance = default_performance.copy() 

    def _get_predict_model(self, model_save_path):
        optimizer = optim.Adam(self.predict_model.parameters(), lr=self.args_p.dml_lr,
                               weight_decay=self.args_p.weight_decay)
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path):
        self.feature_scaler.load_parameters(None, standard_path, self.device)
        # self.state_scaler.load_parameters(standard_path)

    def _get_action(self, actions):  # 输入的actions是一个N*3的数组，在传入到这里之前actions已经是一个在cpu上的数组
        paras = self.para_scaler.inverse_transform(actions)  # N*3的数组
        paras[:, 0] = np.power(10, paras[:, 0])
        paras[:, 2] = np.power(10, paras[:, 2])
        paras = np.floor(paras + 0.5)  # 要4舍5入转成整数

        paras[:, 0] = np.where(paras[:, 0] < paras[:, 1], paras[:, 1], paras[:, 0])

        return paras

    def _get_index_performance(self, feature_input):
        # 预测索引性能，输入feature_input是一个N*14的张量，没有经过归一化
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        # index_performance = torch.cat(index_performance_lis, dim=0)
        # index_performance = index_performance.cpu().numpy()  # 要用逆归一化后的数据

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1:] = torch.pow(10, real_index_performance[:, 1:])

        real_index_performance = real_index_performance.cpu().numpy()

        return real_index_performance  # 同时返回归一化后的性能和真实性能，一个用于组成状态向量，一个用于更新最优性能和计算奖励

    @staticmethod
    def _get_performance_improvement(last_index_performance, current_index_performance):  # 这里其实要考虑的是用真实性能计算还是归一化后的性能计算
        ct_counts_dec = (last_index_performance[:, 1] - current_index_performance[:, 1]) / last_index_performance[:, 1]
        st_counts_dec = (last_index_performance[:, 2] - current_index_performance[:, 2]) / last_index_performance[:, 2]

        return ct_counts_dec, st_counts_dec

    def _get_best_now(self, filename):
        with open(filename, 'rb') as f:
            best_index_performance = pickle.load(f)
        return best_index_performance

    def _get_best_paras_now(self, filename):
        with open(filename, 'rb') as f:
            best_paras = pickle.load(f)
        return best_paras

    def _record_best(self, cur_index_performance, cur_paras, performance_filename,
                     paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            # print(target_rec.shape)
            best_rec = best_index_performance[:, 0]
            # print(best_rec.shape)
            cur_rec = cur_index_performance[:, 0]
            # print(cur_rec.shape)

            ct_counts_dec, st_counts_dec = self._get_performance_improvement(best_index_performance,
                                                                             cur_index_performance)
            # print(ct_counts_dec.shape)
            # print(st_counts_dec.shape)

            target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)

            # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
            cond_a = (cond1 & cond4)
            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
            cond_b = (cond1 & cond3 & cond5)
            # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
            cond_c = (cond2 & cond4 & cond6)

            if (cond_a.any() or cond_b.any() or cond_c.any()):
                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)

        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

    def _record_best2(self, cur_index_performance, cur_paras, performance_filename,
                      paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            # print(target_rec.shape)
            best_rec = best_index_performance[:, 0]
            # print(best_rec.shape)
            cur_rec = cur_index_performance[:, 0]
            # print(cur_rec.shape)

            ct_counts_dec, st_counts_dec = self._get_performance_improvement(best_index_performance,
                                                                             cur_index_performance)

            target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)

            # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
            cond_a = (cond1 & cond4)
            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
            cond_b = (cond1 & cond3 & cond5)
            # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
            cond_c = (cond2 & cond4 & cond6)

            if cond_a.any() or cond_b.any() or cond_c.any():
                self.nochange_steps = 0

                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)

            else:
                self.nochange_steps += 1

            return cond_c

        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

            return np.zeros(cur_index_performance.shape[0], dtype=bool)

    def _update_episode_best(self,
                             cur_index_performance):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算                                                  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance,
                                                                         cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        cond1 = (best_rec < target_rec)
        cond2 = (best_rec >= target_rec)
        cond3 = (cur_rec < target_rec)
        cond4 = (cur_rec >= target_rec)
        cond5 = (cur_rec > best_rec)
        cond6 = (target_dec > 0)

        # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
        cond_a = (cond1 & cond4)
        # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
        cond_b = (cond1 & cond3 & cond5)
        # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
        cond_c = (cond2 & cond4 & cond6)

        if (cond_a.any() or cond_b.any() or cond_c.any()):
            self.last_index_performance[cond_a] = cur_index_performance[cond_a]
            self.last_index_performance[cond_b] = cur_index_performance[cond_b]
            self.last_index_performance[cond_c] = cur_index_performance[cond_c]

    def _update_episode_best2(self,
                              cur_index_performance):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算                                                  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance,
                                                                         cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        cond1 = (best_rec < target_rec)
        cond2 = (best_rec >= target_rec)
        cond3 = (cur_rec < target_rec)
        cond4 = (cur_rec >= target_rec)
        cond5 = (cur_rec > best_rec)
        cond6 = (target_dec > 0)

        # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
        cond_a = (cond1 & cond4)
        # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
        cond_b = (cond1 & cond3 & cond5)
        # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
        cond_c = (cond2 & cond4 & cond6)

        if not (cond_a.any() or cond_b.any() or cond_c.any()):
            self.nochange_steps += 1
        else:
            self.nochange_steps = 0

            self.last_index_performance[cond_a] = cur_index_performance[cond_a]
            self.last_index_performance[cond_b] = cur_index_performance[cond_b]
            self.last_index_performance[cond_c] = cur_index_performance[cond_c]

    def _initialize(self):
        self.steps = 0
        self.score = 0.0
        self.nochange_steps = 0

        num = self.default_index_performance.shape[0]

        self.last_index_performance = self.default_index_performance.copy()

        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        # cur_rec = cur_index_performance[:, 0]

        deltat20 = (best_rec - target_rec).reshape((num, 1))
        delta0 = deltat20.copy()
        deltat = np.zeros((num, 1))

        dec = np.zeros((num, 3))
        # flag = np.zeros((num, 1))

        # 状态向量：共计16个特征）
        init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.default_paras), np.copy(self.default_index_performance), delta0, deltat20, deltat, dec), axis=1)  # 状态向量只用当前性能，不用当前最佳性能

        init_state[:, 0] = np.log10(init_state[:, 0])
        init_state[:, 2:4] = np.log10(init_state[:, 2:4])
        init_state[:, 5] = np.log10(init_state[:, 5])
        init_state[:, 7:9] = np.log10(init_state[:, 7:9])

        init_state_ = self.state_scaler.transform(init_state)  # 状态向量要归一化再存储

        return init_state_

    def _get_next_state(self, cur_index_performance, best_index_performance, cur_paras, best_paras, target_rec, num):
        best_rec = best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = (cur_rec - target_rec).reshape((num, 1))
        deltat20 = (best_rec - target_rec).reshape((num, 1))
        deltat = (cur_rec - best_rec).reshape((num, 1))

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec

        ct_counts_dec = ct_counts_dec.reshape((num, 1))
        st_counts_dec = st_counts_dec.reshape((num, 1))
        target_dec = target_dec.reshape((num, 1))

        next_state = np.concatenate((cur_paras, best_paras, np.copy(cur_index_performance),
                                     delta0, deltat20, deltat, ct_counts_dec, st_counts_dec, target_dec), axis=1)
        return next_state

    def _step(self, actions, data_feature, performance_filename, paras_filename):  # paras是推荐的参数（没有经过归一化处理）， data_feature是数据集输入特征（也没有经过归一化）
        self.steps += 1  # 执行一次就加一次

        num = self.target_rec.shape[0]
        target_rec = self.target_rec.reshape(-1)

        cur_paras = self._get_action(actions)

        feature_input = np.concatenate((np.copy(cur_paras), np.copy(data_feature)), axis=1)
        feature_input[:, 0] = np.log10(feature_input[:, 0])
        feature_input[:, 2] = np.log10(feature_input[:, 2])

        cur_index_performance = self._get_index_performance(feature_input)

        reward, average_reward = self._get_reward3(cur_index_performance)

        # global方式
        _ = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        best_now_performance = self._get_best_now(performance_filename)
        best_now_paras = self._get_best_paras_now(paras_filename)

        self.last_index_performance = best_now_performance.copy()  # 更新最优性能

        # local方式
        # self._record_best(index_performance, paras, performance_filename, paras_filename)
        # self._update_episode_best2(index_performance)

        # general方式
        # self.last_index_performance = cur_index_performance.copy()
        # _ = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        # best_now_performance = self._get_best_now(performance_filename)
        # best_now_paras = self._get_best_paras_now(paras_filename)

        # condition = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        # reward = reward.reshape(-1)  #在general方式下，如果当前性能超过了当前最优性能，则同样扩大
        # reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward)
        # reward = reward.reshape((reward.shape[0], 1))

        # best_now_performance = self._get_best_now(performance_filename)
        # best_now_paras = self._get_best_paras_now(paras_filename)

        next_state = self._get_next_state(cur_index_performance, best_now_performance, cur_paras, best_now_paras, target_rec, num)

        next_state[:, 0] = np.log10(next_state[:, 0])
        next_state[:, 2:4] = np.log10(next_state[:, 2:4])
        next_state[:, 5] = np.log10(next_state[:, 5])
        next_state[:, 7:9] = np.log10(next_state[:, 7:9])

        next_state_ = self.state_scaler.transform(next_state)  # 状态向量要归一化再存储

        # if self.steps < self.max_steps and self.score >= -10000:
        #     terminate = np.zeros((num, 1), dtype=bool)
        # else:
        #     terminate = np.ones((num, 1), dtype=bool)
        # print(average_reward)
        # if average_reward >= -50:
        #     terminate = np.zeros((num, 1), dtype=bool)
        # else:
        #     terminate = np.ones((num, 1), dtype=bool)

        # if self.nochange_steps == self.args_r.nochange_steps:   #用性能连续没有改进的步数是否达到设定阈值来判断是否样稿终止当前episode，达到了则将terminate设置为True，终止当前episode
        #     terminate = np.ones((num, 1), dtype=bool)
        # else:
        #     terminate = np.zeros((num, 1), dtype=bool)
        terminate = np.zeros((num, 1), dtype=bool)

        return reward, next_state_, terminate, self.score, average_reward, cur_index_performance, cur_paras

    @staticmethod
    def _calculate_reward(delta):
        reward_positive = (1 + delta) ** 2 - 1  # 后面再测试一下reward_positive = (1 + delta) ** 2 - 1 ，奖励不同确实影响很大。
        reward_negative = -(1 - delta) ** 2 + 1

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta >= 0, reward_positive, reward_negative)
        # _reward = np.where(delta >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec(delta0, deltat):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        reward_positive = ((1 + delta0) ** 2 - 1) * np.abs(1 + deltat)
        reward_negative = -((1 - delta0) ** 2 - 1) * np.abs(1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta0 >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec2(delta0, deltat, deltat20):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        _reward = np.zeros_like(delta0)

        cond1 = (deltat20 < 0) & (delta0 < 0)
        cond2 = (deltat20 < 0) & (delta0 >= 0)
        cond3 = (deltat20 > 0) & (delta0 < 0)

        reward_cond1 = -(1 - delta0) ** 2 + 1
        reward_cond2 = ((1 + delta0) ** 2) * (1 + deltat)
        reward_cond3 = (-(1 - delta0) ** 2) * (1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward[cond1] = reward_cond1[cond1]
        _reward[cond2] = reward_cond2[cond2]
        _reward[cond3] = reward_cond3[cond3]

        return _reward

    def _get_reward3(self, cur_index_performance):  # 优先使用reward1，reward1是分别计算ct和st的奖励，再加权求和
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.last_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = cur_rec - target_rec
        deltat = cur_rec - best_rec
        deltat20 = best_rec - target_rec

        reward = self._calculate_reward_rec2(delta0, deltat, deltat20)

        ct_counts_dec, st_counts_dec = self._get_performance_improvement(self.last_index_performance,
                                                                         cur_index_performance)

        target_dec = self.args_r.lamb * ct_counts_dec + st_counts_dec
        # min_dec = np.min(target_dec)

        # 分开计算奖励再综合
        # ct_counts_reward = self._calculate_reward(ct_counts_dec)
        # st_counts_reward = self._calculate_reward(st_counts_dec)
        # counts_reward = self.args_r.lamb * ct_counts_reward + st_counts_reward

        # 综合计算奖励
        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        average_reward = np.mean(reward)
        self.score += average_reward

        # reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward) #不注释是global，注释是general

        reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward


class IndexEnv_new_state_onlys_nsg(object):  # 见2024.09.02的日志记录
    def __init__(self, num_dataset, default_performance, target_rec_lis, args_r, args_p, predict_model_save_path,
                 standard_path, dv):
        self.device = dv

        # print('-------------加载索引性能预测模型-------------')
        self.args_p = args_p
        self.predict_model = Direct_Predict_MLP_nsg(eval(args_p.dipredict_layer_sizes_nsg)).to(self.device)
        self._get_predict_model(predict_model_save_path)  # 加载索引性能预测模型

        self.feature_scaler = Scaler_minmax_new_gpu_nsg(9, dv)
        self.performance_scaler = Scaler_minmax_new_gpu_nsg(0, dv)
        self.para_scaler = Scaler_para_nsg()  # 这个操作的对象是numpy
        self.state_scaler = Scaler_state_new_onlys_nsg(18)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.args_r = args_r

        self.target_rec = np.tile(np.array(target_rec_lis).reshape(-1, 1), (num_dataset, 1))  # target_rec_lis=[0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]

        self.score = 0.0
        # self.score_array = np.tile(np.array([0]), self.target_rec.shape[0])
        self.steps = 0
        self.max_steps = args_r.max_steps
        self.nochange_steps = 0

        # print(self.target_rec)
        self.default_paras = np.tile(np.array([[100, 100, 150, 5, 300, 10]]), (self.target_rec.shape[0], 1))
        self.default_index_performance = default_performance.copy()
        self.best_index_performance = default_performance.copy()

    def _get_predict_model(self, model_save_path):
        optimizer = optim.Adam(self.predict_model.parameters(), lr=self.args_p.dml_lr,
                               weight_decay=self.args_p.weight_decay)
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path):
        self.feature_scaler.load_parameters(None, standard_path, self.device)
        # self.state_scaler.load_parameters(standard_path)

    def _get_action(self, actions):  # 输入的actions是一个N*6的数组，在传入到这里之前actions已经是一个在cpu上的数组
        paras = self.para_scaler.inverse_transform(actions)  # N*6的数组
        paras[:, 5] = np.power(10, paras[:, 5])
        paras = np.floor(paras + 0.5)  # 要4舍5入转成整数

        paras[:, 1] = np.where(paras[:, 1] < paras[:, 0], paras[:, 0], paras[:, 1])

        return paras

    def _get_index_performance(self, feature_input):
        # 预测索引性能，输入feature_input是一个N*14的张量，没有经过归一化
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        # index_performance = torch.cat(index_performance_lis, dim=0)
        # index_performance = index_performance.cpu().numpy()  # 要用逆归一化后的数据

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1] = torch.pow(10, real_index_performance[:, 1])

        real_index_performance = real_index_performance.cpu().numpy()

        return real_index_performance  # 同时返回归一化后的性能和真实性能，一个用于组成状态向量，一个用于更新最优性能和计算奖励

    @staticmethod
    def _get_performance_improvement(last_index_performance, current_index_performance):  # 这里其实要考虑的是用真实性能计算还是归一化后的性能计算
        st_counts_dec = (last_index_performance[:, 1] - current_index_performance[:, 1]) / last_index_performance[:, 1]

        return st_counts_dec

    def _get_best_now(self, filename):
        with open(filename, 'rb') as f:
            best_index_performance = pickle.load(f)
        return best_index_performance

    def _get_best_paras_now(self, filename):
        with open(filename, 'rb') as f:
            best_paras = pickle.load(f)
        return best_paras

    def _record_best2(self, cur_index_performance, cur_paras, performance_filename,
                      paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            best_rec = best_index_performance[:, 0]
            cur_rec = cur_index_performance[:, 0]

            target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)

            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)

            # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
            cond_a = (cond1 & cond4)
            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
            cond_b = (cond1 & cond3 & cond5)
            # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
            cond_c = (cond2 & cond4 & cond6)

            if cond_a.any() or cond_b.any() or cond_c.any():
                self.nochange_steps = 0

                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)

            else:
                self.nochange_steps += 1

            return cond_c

        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

            return np.zeros(cur_index_performance.shape[0], dtype=bool)

    def _initialize(self):
        self.steps = 0
        self.score = 0.0
        self.nochange_steps = 0

        num = self.default_index_performance.shape[0]

        self.best_index_performance = self.default_index_performance.copy()  # self.best_index_performance 存储当前最优性能

        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]

        deltat20 = (best_rec - target_rec).reshape((num, 1))
        delta0 = deltat20.copy()
        deltat = np.zeros((num, 1))

        target_dec = np.zeros((num, 1))  # 当前性能相对于当前最好性能的改进

        cur_index_performance = self.default_index_performance.copy()

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0]  # 只需要召回率和查询执行的平均距离计算次数
        cur_state_index_performance[:, 1] = cur_index_performance[:, 1]

        # 状态向量：共计13个特征）
        init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.default_paras), cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1)

        init_state[:, 5] = np.log10(init_state[:, 5])
        init_state[:, 11] = np.log10(init_state[:, 11])
        init_state[:, 13] = np.log10(init_state[:, 13])

        init_state_ = self.state_scaler.transform(init_state)  # 状态向量要归一化再存储

        return init_state_

    def _get_next_state(self, cur_index_performance, best_index_performance, cur_paras, best_paras, target_rec, num):
        best_rec = best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = (cur_rec - target_rec).reshape((num, 1))
        deltat20 = (best_rec - target_rec).reshape((num, 1))
        deltat = (cur_rec - best_rec).reshape((num, 1))

        target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
        target_dec = target_dec.reshape((num, 1))

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0]  # 只需要召回率和查询执行的平均距离计算次数
        cur_state_index_performance[:, 1] = cur_index_performance[:, 1]

        next_state = np.concatenate((cur_paras, best_paras, cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1)

        return next_state

    def _step(self, actions, data_feature, performance_filename,
              paras_filename):  # paras是推荐的参数（没有经过归一化处理）， data_feature是数据集输入特征（也没有经过归一化）
        self.steps += 1  # 执行一次就加一次

        num = self.target_rec.shape[0]
        target_rec = self.target_rec.reshape(-1)

        cur_paras = self._get_action(actions)

        feature_input = np.concatenate((np.copy(cur_paras), np.copy(data_feature)), axis=1)
        feature_input[:, 5] = np.log10(feature_input[:, 5])

        cur_index_performance = self._get_index_performance(feature_input)

        reward, average_reward = self._get_reward3(cur_index_performance)

        # global方式
        _ = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        best_now_performance = self._get_best_now(performance_filename)
        best_now_paras = self._get_best_paras_now(paras_filename)

        self.best_index_performance = best_now_performance.copy()  # 更新最优性能

        next_state = self._get_next_state(cur_index_performance, best_now_performance, cur_paras, best_now_paras, target_rec, num)

        next_state[:, 5] = np.log10(next_state[:, 5])
        next_state[:, 11] = np.log10(next_state[:, 11])
        next_state[:, 13] = np.log10(next_state[:, 13])

        next_state_ = self.state_scaler.transform(next_state)  # 状态向量要归一化再存储

        terminate = np.zeros((num, 1), dtype=bool)

        return reward, next_state_, terminate, self.score, average_reward, cur_index_performance, cur_paras

    @staticmethod
    def _calculate_reward(delta):
        reward_positive = (1 + delta) ** 2 - 1  # 后面再测试一下reward_positive = (1 + delta) ** 2 - 1 ，奖励不同确实影响很大。
        reward_negative = -(1 - delta) ** 2 + 1

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta >= 0, reward_positive, reward_negative)
        # _reward = np.where(delta >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec(delta0, deltat):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        reward_positive = ((1 + delta0) ** 2 - 1) * np.abs(1 + deltat)
        reward_negative = -((1 - delta0) ** 2 - 1) * np.abs(1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta0 >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec2(delta0, deltat, deltat20):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        _reward = np.zeros_like(delta0)

        cond1 = (deltat20 < 0) & (delta0 < 0)
        cond2 = (deltat20 < 0) & (delta0 >= 0)
        cond3 = (deltat20 > 0) & (delta0 < 0)

        reward_cond1 = -(1 - delta0) ** 2 + 1
        reward_cond2 = ((1 + delta0) ** 2) * (1 + deltat)
        reward_cond3 = (-(1 - delta0) ** 2) * (1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward[cond1] = reward_cond1[cond1]
        _reward[cond2] = reward_cond2[cond2]
        _reward[cond3] = reward_cond3[cond3]

        return _reward

    def _get_reward3(self, cur_index_performance):  # 优先使用reward1，reward1是分别计算ct和st的奖励，再加权求和
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = cur_rec - target_rec
        deltat = cur_rec - best_rec
        deltat20 = best_rec - target_rec

        reward = self._calculate_reward_rec2(delta0, deltat, deltat20)

        st_counts_dec = self._get_performance_improvement(self.best_index_performance, cur_index_performance)

        target_dec = st_counts_dec

        # 综合计算奖励
        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        average_reward = np.mean(reward)
        self.score += average_reward

        reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward)  # 不注释是global，注释是general

        # reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward


class IndexEnv_new_state_onlys_eval_nsg(object):  # 见2024.09.02的日志记录
    def __init__(self, num_dataset, default_performance, target_rec_lis, args_r, args_p, predict_model_save_path, standard_path, dv):
        self.device = dv

        # print('-------------加载索引性能预测模型-------------')
        self.args_p = args_p
        self.predict_model = Direct_Predict_MLP_nsg(eval(args_p.dipredict_layer_sizes_nsg)).to(self.device)
        self._get_predict_model(predict_model_save_path)  # 加载索引性能预测模型

        self.feature_scaler = Scaler_minmax_new_gpu_nsg(9, dv)
        self.performance_scaler = Scaler_minmax_new_gpu_nsg(0, dv)
        self.para_scaler = Scaler_para_nsg()  # 这个操作的对象是numpy
        self.state_scaler = Scaler_state_new_onlys_nsg(18)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.args_r = args_r

        self.target_rec = np.tile(np.array(target_rec_lis).reshape(-1, 1), (num_dataset, 1))  # target_rec_lis=[0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
        print(self.target_rec)

        self.score = 0.0
        # self.score_array = np.tile(np.array([0]), self.target_rec.shape[0])
        self.steps = 0
        self.max_steps = args_r.max_steps
        self.nochange_steps = 0
        self.nochange_episodes = 0

        # print(self.target_rec)
        self.default_paras = np.tile(np.array([[100, 100, 150, 5, 300, 10]]), (self.target_rec.shape[0], 1))
        self.default_index_performance = default_performance.copy()
        self.best_index_performance = default_performance.copy()

    def _get_predict_model(self, model_save_path):
        optimizer = optim.Adam(self.predict_model.parameters(), lr=self.args_p.dml_lr,
                               weight_decay=self.args_p.weight_decay)
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path):
        self.feature_scaler.load_parameters(None, standard_path, self.device)
        # self.state_scaler.load_parameters(standard_path)

    def _get_action(self, actions):  # 输入的actions是一个N*3的数组，在传入到这里之前actions已经是一个在cpu上的数组
        paras = self.para_scaler.inverse_transform(actions)  # N*3的数组
        paras[:, 5] = np.power(10, paras[:, 5])
        paras = np.floor(paras + 0.5)  # 要4舍5入转成整数

        paras[:, 1] = np.where(paras[:, 1] < paras[:, 0], paras[:, 0], paras[:, 1])

        return paras

    def _get_index_performance(self, feature_input):
        # 预测索引性能，输入feature_input是一个N*14的张量，没有经过归一化
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        # index_performance = torch.cat(index_performance_lis, dim=0)
        # index_performance = index_performance.cpu().numpy()  # 要用逆归一化后的数据

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1] = torch.pow(10, real_index_performance[:, 1])

        real_index_performance = real_index_performance.cpu().numpy()

        return real_index_performance  # 同时返回归一化后的性能和真实性能，一个用于组成状态向量，一个用于更新最优性能和计算奖励

    @staticmethod
    def _get_performance_improvement(last_index_performance, current_index_performance):  # 这里其实要考虑的是用真实性能计算还是归一化后的性能计算
        st_counts_dec = (last_index_performance[:, 1] - current_index_performance[:, 1]) / last_index_performance[:, 1]

        return st_counts_dec

    def _get_best_now(self, filename):
        with open(filename, 'rb') as f:
            best_index_performance = pickle.load(f)
        return best_index_performance

    def _get_best_paras_now(self, filename):
        with open(filename, 'rb') as f:
            best_paras = pickle.load(f)
        return best_paras

    def _record_best2(self, cur_index_performance, cur_paras, performance_filename,
                      paras_filename):  # 计算目标函数有两种方式，第一种是ct_counts和st_counts都用平均值；第二种是这两个不用平均值，但是计算目标时用归一化后的性能计算
        if os.path.exists(performance_filename):  # 我觉得应该用平均值，然后也用归一化后的性能计算目标，具体要看一下数据情况
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            best_rec = best_index_performance[:, 0]
            cur_rec = cur_index_performance[:, 0]

            target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)

            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)

            # 更新条件1: 最佳召回率未达到目标，当前召回率达到或超过目标
            cond_a = (cond1 & cond4)
            # 更新条件2: 最佳召回率未达到目标，当前召回率也未达到目标，但当前召回率更高
            cond_b = (cond1 & cond3 & cond5)
            # 更新条件3: 最佳召回率已达到目标，当前召回率也达到或超过目标，当前目标值更低
            cond_c = (cond2 & cond4 & cond6)

            if cond_a.any() or cond_b.any() or cond_c.any():
                self.nochange_steps = 0

                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)

            else:
                self.nochange_steps += 1

            return cond_c

        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

            return np.zeros(cur_index_performance.shape[0], dtype=bool)

    def _initialize(self):
        self.steps = 0
        self.score = 0.0
        self.nochange_steps = 0

        num = self.default_index_performance.shape[0]

        self.best_index_performance = self.default_index_performance.copy()  # self.best_index_performance 存储当前最优性能

        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]

        deltat20 = (best_rec - target_rec).reshape((num, 1))
        delta0 = deltat20.copy()
        deltat = np.zeros((num, 1))

        target_dec = np.zeros((num, 1))  # 当前性能相对于当前最好性能的改进

        cur_index_performance = self.default_index_performance.copy()

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0]  # 只需要召回率和查询执行的平均距离计算次数
        cur_state_index_performance[:, 1] = cur_index_performance[:, 1]

        # 状态向量：共计13个特征）
        init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.default_paras),
                                     cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1)

        init_state[:, 5] = np.log10(init_state[:, 5])
        init_state[:, 11] = np.log10(init_state[:, 11])
        init_state[:, 13] = np.log10(init_state[:, 13])

        init_state_ = self.state_scaler.transform(init_state)  # 状态向量要归一化再存储

        return init_state_

    def _get_next_state(self, cur_index_performance, best_index_performance, cur_paras, best_paras, target_rec, num):
        best_rec = best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = (cur_rec - target_rec).reshape((num, 1))
        deltat20 = (best_rec - target_rec).reshape((num, 1))
        deltat = (cur_rec - best_rec).reshape((num, 1))

        target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
        target_dec = target_dec.reshape((num, 1))

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0]  # 只需要召回率和查询执行的平均距离计算次数
        cur_state_index_performance[:, 1] = cur_index_performance[:, 1]

        next_state = np.concatenate(
            (cur_paras, best_paras, cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1)
        return next_state

    def _step(self, actions, data_feature, performance_filename,
              paras_filename):  # paras是推荐的参数（没有经过归一化处理）， data_feature是数据集输入特征（也没有经过归一化）
        self.steps += 1  # 执行一次就加一次

        num = self.target_rec.shape[0]
        target_rec = self.target_rec.reshape(-1)

        cur_paras = self._get_action(actions)

        feature_input = np.concatenate((np.copy(cur_paras), np.copy(data_feature)), axis=1)
        feature_input[:, 5] = np.log10(feature_input[:, 5])

        cur_index_performance = self._get_index_performance(feature_input)

        reward, average_reward = self._get_reward3(cur_index_performance)

        # global方式
        _ = self._record_best2(cur_index_performance, cur_paras, performance_filename, paras_filename)

        best_now_performance = self._get_best_now(performance_filename)
        best_now_paras = self._get_best_paras_now(paras_filename)

        self.best_index_performance = best_now_performance.copy()  # 更新最优性能

        next_state = self._get_next_state(cur_index_performance, best_now_performance, cur_paras, best_now_paras, target_rec, num)

        next_state[:, 5] = np.log10(next_state[:, 5])
        next_state[:, 11] = np.log10(next_state[:, 11])
        next_state[:, 13] = np.log10(next_state[:, 13])

        next_state_ = self.state_scaler.transform(next_state)  # 状态向量要归一化再存储

        terminate = np.zeros((num, 1), dtype=bool)

        return reward, next_state_, terminate, self.score, average_reward, cur_index_performance, cur_paras

    @staticmethod
    def _calculate_reward(delta):
        reward_positive = (1 + delta) ** 2 - 1  # 后面再测试一下reward_positive = (1 + delta) ** 2 - 1 ，奖励不同确实影响很大。
        reward_negative = -(1 - delta) ** 2 + 1

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta >= 0, reward_positive, reward_negative)
        # _reward = np.where(delta >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec(delta0, deltat):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        reward_positive = ((1 + delta0) ** 2 - 1) * np.abs(1 + deltat)
        reward_negative = -((1 - delta0) ** 2 - 1) * np.abs(1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward = np.where(delta0 >= 0, reward_positive, reward_negative)

        return _reward

    @staticmethod
    def _calculate_reward_rec2(delta0, deltat, deltat20):  # delta0是当前召回率相对于目标召回率的差值， deltat是当前召回率相对于当前最高 召回率的差值
        _reward = np.zeros_like(delta0)

        cond1 = (deltat20 < 0) & (delta0 < 0)
        cond2 = (deltat20 < 0) & (delta0 >= 0)
        cond3 = (deltat20 > 0) & (delta0 < 0)

        reward_cond1 = -(1 - delta0) ** 2 + 1
        reward_cond2 = ((1 + delta0) ** 2) * (1 + deltat)
        reward_cond3 = (-(1 - delta0) ** 2) * (1 - deltat)

        # 根据 delta0 的值选择正负奖励
        _reward[cond1] = reward_cond1[cond1]
        _reward[cond2] = reward_cond2[cond2]
        _reward[cond3] = reward_cond3[cond3]

        return _reward

    def _get_reward3(self, cur_index_performance):  # 优先使用reward1，reward1是分别计算ct和st的奖励，再加权求和
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = cur_rec - target_rec
        deltat = cur_rec - best_rec
        deltat20 = best_rec - target_rec

        reward = self._calculate_reward_rec2(delta0, deltat, deltat20)

        st_counts_dec = self._get_performance_improvement(self.best_index_performance, cur_index_performance)

        target_dec = st_counts_dec
        # min_dec = np.min(target_dec)

        # 分开计算奖励再综合
        # ct_counts_reward = self._calculate_reward(ct_counts_dec)
        # st_counts_reward = self._calculate_reward(st_counts_dec)
        # counts_reward = self.args_r.lamb * ct_counts_reward + st_counts_reward

        # 综合计算奖励
        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        average_reward = np.mean(reward)
        self.score += average_reward

        # reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward) #不注释是global，注释是general

        reward = reward.reshape((reward.shape[0], 1))

        return reward, average_reward