import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import random

from models import Direct_Predict_MLP_nsg
from utils import read_data_new_nsg, get_dataset, df2np, np2ts, CustomDataset, Scaler_minmax_new_gpu_nsg, calculate_errors, load_model
from trainer import dipredict_train
from Args import args

torch.autograd.set_detect_anomaly(True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.enabled = False
cudnn.benchmark = False
cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

current_directory = os.getcwd()  # 返回当前工作目录的路径
parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/train_data.csv')  # 表示是1w-100w的数据，不包含右边界

# feature_minmax_path = os.path.join(current_directory, 'scaler_paras/feature_minmax.npz')
# performance_minmax_path = os.path.join(current_directory, 'scaler_paras/performance_minmax.npz')
feature_standard_path = os.path.join(current_directory, 'scaler_paras/NSG_KNNG/feature_standard.npz')
performance_standard_path = os.path.join(current_directory, 'scaler_paras/NSG_KNNG/performance_standard.npz')

# 将numpy数组换成torch张量
print('-------------------加载数据-------------------')
df_train, df_valid, df_test = get_dataset(data_path)

df_train_f, df_train_p = read_data_new_nsg(df_train)
df_valid_f, df_valid_p = read_data_new_nsg(df_valid)
df_test_f, df_test_p = read_data_new_nsg(df_test)

df_train_f = df_train_f.drop(['FileName'], axis=1)
df_valid_f = df_valid_f.drop(['FileName'], axis=1)
df_test_f = df_test_f.drop(['FileName'], axis=1)

print(df_train_f.columns)

feature_train_raw = df2np(df_train_f)
performance_train_raw = df2np(df_train_p)

feature_valid_raw = df2np(df_valid_f)
performance_valid_raw = df2np(df_valid_p)
# has_nan_1 = np.any(np.isnan(feature_valid_raw))
# print(has_nan_1)
feature_test_raw = df2np(df_test_f)
performance_test_raw = df2np(df_test_p)

# performance_train_raw[:, 1] = performance_train_raw[:, 1] * 100
# performance_valid_raw[:, 1] = performance_valid_raw[:, 1] * 100
# performance_test_raw[:, 1] = performance_test_raw[:, 1] * 100
feature_train_raw[:, 5:8] = np.log10(feature_train_raw[:, 5:8])
feature_valid_raw[:, 5:8] = np.log10(feature_valid_raw[:, 5:8])
feature_test_raw[:, 5:8] = np.log10(feature_test_raw[:, 5:8])

performance_train_raw[:, 1] = np.log10(performance_train_raw[:, 1])
performance_valid_raw[:, 1] = np.log10(performance_valid_raw[:, 1])

feature_train_raw = np2ts(feature_train_raw).to(device)
feature_valid_raw = np2ts(feature_valid_raw).to(device)
feature_test_raw = np2ts(feature_test_raw).to(device)
performance_train_raw = np2ts(performance_train_raw).to(device)
performance_valid_raw = np2ts(performance_valid_raw).to(device)
performance_test_raw = np2ts(performance_test_raw).to(device)

# 数据归一化
print('-------------------数据归一化-------------------')
feature_scaler = Scaler_minmax_new_gpu_nsg(9, device)
if os.path.exists(feature_standard_path):
    feature_scaler.load_parameters(None, feature_standard_path, device)
else:
    feature_raw = torch.cat((feature_train_raw, feature_valid_raw, feature_test_raw), dim=0)

    feature_scaler.fit(feature_raw)
    feature_scaler.save_parameters(None, feature_standard_path)
    print('特征数据完成归一化')

feature_train = feature_scaler.transform(feature_train_raw)
feature_valid = feature_scaler.transform(feature_valid_raw)
feature_test = feature_scaler.transform(feature_test_raw)
# has_nan_1 = torch.any(torch.isnan(feature_train))
# print(has_nan_1)
performance_scaler = Scaler_minmax_new_gpu_nsg(0, device)
# performance_scaler.load_parameters(None, performance_standard_path)
performance_train = performance_scaler.transform(performance_train_raw)
performance_valid = performance_scaler.transform(performance_valid_raw)

performance_valid_raw[:, 1] = torch.pow(10, performance_valid_raw[:, 1])
# performance_valid_raw = torch.pow(10, performance_valid_raw)

if args.model == 'dim':
    dipredict_model_save_path = os.path.join(current_directory, 'model_checkpoints/DiPredict/NSG_KNNG/{}_{}_{}_{}_checkpoint.pth'.format(
                                                 args.dipredict_layer_sizes_nsg, args.dipredict_n_epochs, args.dipredict_batch_size, args.dipredict_lr))

    # 创建Dataset和DataLoader
    print('-------------------创建Dataset和DataLoader-------------------')
    dataset = CustomDataset(feature_train, performance_train)
    dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

    # 初始化模型和优化器
    print('-------------------初始化模型和优化器-------------------')
    layer_sizes = eval(args.dipredict_layer_sizes_nsg)
    inner_dim = args.inner_dim

    model = Direct_Predict_MLP_nsg(layer_sizes)
    #model = Direct_Predict_MLP_with_uncertainty(layer_sizes, inner_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=1e-2, threshold_mode='rel')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)  # step_size设为400， gamma设为0.1

    # 开始训练
    print('-------------------开始训练-------------------')
    dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                    performance_scaler, args, dipredict_model_save_path, None, device)

    print('-------------------训练结束, 开始测试-------------------')
    model, _, _ = load_model(model, optimizer, dipredict_model_save_path)
    model.eval()

    predicted_performances = model(feature_test)
    #predicted_performances, predicted_variance = model(feature_test)

    predicted_performances = performance_scaler.inverse_transform(predicted_performances)
    predicted_performances[:, 1] = torch.pow(10, predicted_performances[:, 1])

    print('------计算误差------')
    mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_test_raw, predicted_performances)  # 这里误差是一个3维张量，是所有验证样本的平均

    print('预测误差为：')
    print(f'mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}, mean_qerror:{mean_qerrors}')
