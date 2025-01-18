#在开始训练之前先运行这个文件对数据进行归一化，获取相关参数
import os
import numpy as np
import pandas as pd
import torch

from utils import df2np, np2ts, Scaler_minmax_new_gpu_nsg

use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

current_directory = os.getcwd()  # 返回当前工作目录的路径
parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

feature_standard_path = os.path.join(current_directory, 'scaler_paras/NSG_KNNG/feature_standard.npz')
performance_standard_path = os.path.join(current_directory, 'scaler_paras/NSG_KNNG/performance_standard.npz')

train_data_path = "../NSG_KNNG/Data/train_data.csv"
config_unit_data_path = "../NSG_KNNG/Data/config_unit_data.csv"
data_fetaure_path = os.path.join(parent_directory, 'NSG_KNNG/Data/whole_data_feature.csv')


df_train = pd.read_csv(train_data_path, sep=',', header=0)
df_f = df_train[['K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist',
                 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']]

feature = df2np(df_f)
feature[:, 5:8] = np.log10(feature[:, 5:8])

feature = np2ts(feature).to(device)

#数据归一化
print('-------------------数据归一化-------------------')
feature_scaler = Scaler_minmax_new_gpu_nsg(9, device)
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
