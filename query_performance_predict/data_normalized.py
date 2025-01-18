#在开始训练之前先运行这个文件对数据进行归一化，获取相关参数
import os
import numpy as np
import pandas as pd
import torch

from utils import read_data, df2np, np2ts, Scaler_raw, Scaler_standard, Scaler_minmax, Scaler_minmax_new_gpu

def get_input_feature(df_data_feature, df_config_unit):
    efSs = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 300, 340, 380, 420, 460, 500, 540, 580, 620, 660, 700, 760, 820, 880, 940, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000])

    df_config_whole = df_config_unit.loc[df_config_unit.index.repeat(len(efSs))].reset_index(drop=True)
    df_config_whole['efSearch'] = np.tile(efSs, len(df_config_unit))

    df_feature = pd.merge(df_data_feature, df_config_whole, on='FileName', how='right')

    return df_feature

use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

current_directory = os.getcwd()  # 返回当前工作目录的路径
parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

feature_standard_path = os.path.join(current_directory, 'scaler_paras/feature_standard.npz')
performance_standard_path = os.path.join(current_directory, 'scaler_paras/performance_standard.npz')

train_data_path = "../Data/train_data.csv"
config_unit_data_path = "../Data/config_unit_data.csv"
data_fetaure_path = os.path.join(parent_directory, 'Data/whole_data_feature.csv')

# df_data_feature = pd.read_csv(data_fetaure_path, sep=',', header = 0)
#
# df_combined = []
#
# filename_dic = {'glove_tr': 'glove_1_1.0_100_1', 'paper_tr':'paper_1_2.0_200_1', 'crawl_tr':'crawl_1_1.9_300_1', 'msong_tr':'msong_0_9.0_420_1', 'glove': 'glove_1_1.183514_100', 'paper':'paper_1_2.029997_200', 'crawl':'crawl_1_1.989995_300', 'msong':'msong_0_9.92272_420', 'gist':'gist_1_1.0_960', 'deep':'deep_2_1_96', 'sift1':'sift_1_1_128_1', 'sift50':'sift_2_5_128_1'}
# dataset_name_lis = ['glove_tr', 'paper_tr', 'crawl_tr', 'msong_tr', 'sift1']
#
# for dataset_name in dataset_name_lis:
#     filename = filename_dic[dataset_name]
#
#     df_data_feature_labeled = df_data_feature[df_data_feature['FileName'] == filename]
#
#     df_config_unit = pd.read_csv(config_unit_data_path, sep=',', header=0)
#     df_config_unit['FileName'] = filename
#
#     df_feature_labeled = get_input_feature(df_data_feature_labeled, df_config_unit)
#
#     df_combined.append(df_feature_labeled)
#
# labeled_df = pd.concat(df_combined, ignore_index=True)
#
# df_f = labeled_df[
#     ['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist',
#      'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]  # 初始选择未标注数据时就可以用到

df_train = pd.read_csv(train_data_path, sep=',', header=0)
df_f = df_train[['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist',
                 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]

feature = df2np(df_f)
feature[:, 0] = np.log10(feature[:, 0])
feature[:, 2:5] = np.log10(feature[:, 2:5])

feature = np2ts(feature).to(device)

#数据归一化
print('-------------------数据归一化-------------------')
feature_scaler = Scaler_minmax_new_gpu(6, device)
if os.path.exists(feature_standard_path):
    feature_scaler.load_parameters(None, feature_standard_path, device)
    print(feature_scaler.mean)
    print(feature_scaler.std)
    print('特征数据已经进行过归一化')
else:
    feature_scaler.fit(feature)
    feature_scaler.save_parameters(None, feature_standard_path)
    print(feature_scaler.mean)
    print(feature_scaler.std)
    print('特征数据完成归一化')

# performance_scaler = Scaler_minmax(2)
# if os.path.exists(performance_standard_path):
#     print('性能数据已经进行过归一化')
# else:
#     performance_scaler.fit(performance)
#     performance_scaler.save_parameters(None, performance_standard_path)
#     print('性能数据完成归一化')
