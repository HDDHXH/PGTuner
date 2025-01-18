import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import random

from models import Direct_Predict_MLP, Direct_Predict_MLP_Bayesian, Direct_Predict_MLP_with_uncertainty, Mt_Direct_Predict_MLP, It_Direct_Predict_MLP, Direct_Predict_ATT, Direct_Predict_Conv2d, Direct_Predict_Conv1d, Direct_Predict_MOE_MLP
from utils import read_data_new, read_data_new_ct, get_dataset, df2np, np2ts, CustomDataset, Scaler_raw, Scaler_standard, Scaler_minmax_new_gpu, Scaler_minmax_new_ct_gpu, calculate_errors
from trainer import dipredict_train, dipredict_train_with_uncertainty, dipredict_train_Bayesian
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

data_path = os.path.join(parent_directory, 'Data/train_data.csv')  # 表示是1w-100w的数据，不包含右边界

# feature_minmax_path = os.path.join(current_directory, 'scaler_paras/feature_minmax.npz')
# performance_minmax_path = os.path.join(current_directory, 'scaler_paras/performance_minmax.npz')
feature_standard_path = os.path.join(current_directory, 'scaler_paras/feature_standard.npz')
performance_standard_path = os.path.join(current_directory, 'scaler_paras/performance_standard.npz')

# 将numpy数组换成torch张量
print('-------------------加载数据-------------------')
df_train, df_valid, df_test = get_dataset(data_path)

df_train_f, df_train_p = read_data_new(df_train)
df_valid_f, df_valid_p = read_data_new(df_valid)
df_test_f, df_test_p = read_data_new(df_test)

df_train_f = df_train_f.drop(['FileName'], axis=1)
df_valid_f = df_valid_f.drop(['FileName'], axis=1)
df_test_f = df_test_f.drop(['FileName'], axis=1)

print(df_train_f.columns)

feature_train_raw = df2np(df_train_f)
performance_train_raw = df2np(df_train_p)

feature_valid_raw = df2np(df_valid_f)
performance_valid_raw = df2np(df_valid_p)

feature_test_raw = df2np(df_test_f)
performance_test_raw = df2np(df_test_p)

# performance_train_raw[:, 1] = performance_train_raw[:, 1] * 100
# performance_valid_raw[:, 1] = performance_valid_raw[:, 1] * 100
# performance_test_raw[:, 1] = performance_test_raw[:, 1] * 100
feature_train_raw[:, 0] = np.log10(feature_train_raw[:, 0])
feature_train_raw[:, 2:5] = np.log10(feature_train_raw[:, 2:5])

feature_valid_raw[:, 0] = np.log10(feature_valid_raw[:, 0])
feature_valid_raw[:, 2:5] = np.log10(feature_valid_raw[:, 2:5])

feature_test_raw[:, 0] = np.log10(feature_test_raw[:, 0])
feature_test_raw[:, 2:5] = np.log10(feature_test_raw[:, 2:5])

performance_train_raw[:, 1:] = np.log10(performance_train_raw[:, 1:])
performance_valid_raw[:, 1:] = np.log10(performance_valid_raw[:, 1:])

# feature_train_raw[:, 0] = np.log10(feature_train_raw[:, 0])
# feature_train_raw[:, 2] = np.log10(feature_train_raw[:, 2])
#
# feature_valid_raw[:, 0] = np.log10(feature_valid_raw[:, 0])
# feature_valid_raw[:, 2] = np.log10(feature_valid_raw[:, 2])
#
# feature_test_raw[:, 0] = np.log10(feature_test_raw[:, 0])
# feature_test_raw[:, 2] = np.log10(feature_test_raw[:, 2])
#
# performance_train_raw = np.log10(performance_train_raw)
# performance_valid_raw = np.log10(performance_valid_raw)

feature_train_raw = np2ts(feature_train_raw).to(device)
feature_valid_raw = np2ts(feature_valid_raw).to(device)
feature_test_raw = np2ts(feature_test_raw).to(device)
performance_train_raw = np2ts(performance_train_raw).to(device)
performance_valid_raw = np2ts(performance_valid_raw).to(device)
performance_test_raw = np2ts(performance_test_raw).to(device)

# feature_raw = np.vstack((feature_train_raw, feature_valid_raw, feature_test_raw))
# performance_raw = np.vstack((performance_train_raw, performance_valid_raw, performance_test_raw))
#
# min = np.min(performance_raw, axis=0)
# mean = np.mean(performance_raw, axis=0)
# max = np.max(performance_raw, axis=0)
# std = np.std(performance_raw, axis=0)
# print(min, mean, max, std)

# 数据归一化
print('-------------------数据归一化-------------------')
feature_scaler = Scaler_minmax_new_gpu(6, device)
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

performance_scaler = Scaler_minmax_new_gpu(0, device)
# performance_scaler.load_parameters(None, performance_standard_path)
performance_train = performance_scaler.transform(performance_train_raw)
performance_valid = performance_scaler.transform(performance_valid_raw)

performance_valid_raw[:, 1:] = torch.pow(10, performance_valid_raw[:, 1:])
# performance_valid_raw = torch.pow(10, performance_valid_raw)

if args.model == 'dim':
    dipredict_model_save_path = os.path.join(current_directory,
                                             'model_checkpoints/DiPredict/{}_{}_{}_{}_checkpoint.pth'.format(
                                                 args.dipredict_layer_sizes, args.dipredict_n_epochs,
                                                 args.dipredict_batch_size, args.dipredict_lr))

    # dipredict_loss_result_path = os.path.join(current_directory, 'results/DiPredict/{}_{}_{}'.format(args.dipredict_layer_sizes,
    #                                                                               args.dipredict_n_epochs,
    #                                                                               args.dipredict_lr))

    # writer = SummaryWriter(dipredict_loss_result_path)

    # 创建Dataset和DataLoader
    print('-------------------创建Dataset和DataLoader-------------------')
    dataset = CustomDataset(feature_train, performance_train)
    dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

    # 初始化模型和优化器
    print('-------------------初始化模型和优化器-------------------')
    layer_sizes = eval(args.dipredict_layer_sizes)
    inner_dim = args.inner_dim

    model = Direct_Predict_MLP(layer_sizes)
    #model = Direct_Predict_MLP_with_uncertainty(layer_sizes, inner_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=1e-2, threshold_mode='rel')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)  # step_size设为400， gamma设为0.1

    # 开始训练
    print('-------------------开始训练-------------------')
    dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                    performance_scaler, args, dipredict_model_save_path, None, device)
    # dipredict_train_with_uncertainty(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
    #                 performance_scaler, args, dipredict_model_save_path, None, device)

    # writer.close()
    print('-------------------训练结束, 开始测试-------------------')
    predicted_performances = model(feature_test)
    #predicted_performances, predicted_variance = model(feature_test)

    predicted_performances = performance_scaler.inverse_transform(predicted_performances)
    predicted_performances[:, 1:] = torch.pow(10, predicted_performances[:, 1:])

    print('------计算误差------')
    mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_test_raw, predicted_performances)  # 这里误差是一个3维张量，是所有验证样本的平均

    print('预测误差为：')
    print(f'mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}, mean_qerror:{mean_qerrors}')

    # print('预测方差为：')
    # print(f'min_variance:{torch.min(predicted_variance, dim=0).values}, mean_variance:{torch.mean(predicted_variance, dim=0)}, max_variance:{torch.max(predicted_variance, dim=0).values}')

elif args.model == 'dim_att':
    dipredict_att_model_save_path = os.path.join(current_directory,
                                             'model_checkpoints/DiPredict_ATT/{}_{}_{}_{}_checkpoint.pth'.format(
                                                 args.dipredict_layer_sizes, args.dipredict_n_epochs,
                                                 args.dipredict_batch_size, args.dipredict_lr))

    dipredict_att_loss_result_path = os.path.join(current_directory,
                                              'results/DiPredict_ATT/{}_{}_{}'.format(args.dipredict_layer_sizes,
                                                                                  args.dipredict_n_epochs,
                                                                                  args.dipredict_lr))

    writer = SummaryWriter(dipredict_att_loss_result_path)

    # 创建Dataset和DataLoader
    print('-------------------创建Dataset和DataLoader-------------------')
    dataset = CustomDataset(feature_train, performance_train)
    dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

    # 初始化模型和优化器
    print('-------------------初始化模型和优化器-------------------')
    layer_sizes = eval(args.dipredict_layer_sizes)

    model = Direct_Predict_ATT(layer_sizes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=1e-2, threshold_mode='rel')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)  # step_size设为400， gamma设为0.1

    # 开始训练
    print('-------------------开始训练-------------------')
    dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                    performance_scaler, args, dipredict_att_model_save_path, writer, device)

    writer.close()
    print('-------------------训练结束-------------------')

elif args.model == 'dim_conv2d':
    dipredict_att_model_save_path = os.path.join(current_directory,
                                             'model_checkpoints/DiPredict_Conv2d/{}_{}_{}_{}_checkpoint.pth'.format(
                                                 args.dipredict_conv_layer_sizes, args.dipredict_n_epochs,
                                                 args.dipredict_batch_size, args.dipredict_lr))

    dipredict_att_loss_result_path = os.path.join(current_directory,
                                              'results/DiPredict_Conv2d/{}_{}_{}'.format(args.dipredict_conv_layer_sizes,
                                                                                  args.dipredict_n_epochs,
                                                                                  args.dipredict_lr))

    writer = SummaryWriter(dipredict_att_loss_result_path)

    # 创建Dataset和DataLoader
    print('-------------------创建Dataset和DataLoader-------------------')
    dataset = CustomDataset(feature_train, performance_train)
    dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

    # 初始化模型和优化器
    print('-------------------初始化模型和优化器-------------------')
    layer_sizes = eval(args.dipredict_conv_layer_sizes)

    model = Direct_Predict_Conv2d(layer_sizes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=1e-2, threshold_mode='rel')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)  # step_size设为400， gamma设为0.1

    # 开始训练
    print('-------------------开始训练-------------------')
    dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                    performance_scaler, args, dipredict_att_model_save_path, writer, device)

    writer.close()
    print('-------------------训练结束-------------------')

elif args.model == 'dim_conv1d':
    dipredict_att_model_save_path = os.path.join(current_directory,
                                             'model_checkpoints/DiPredict_Conv1d/{}_{}_{}_{}_checkpoint.pth'.format(
                                                 args.dipredict_conv_layer_sizes, args.dipredict_n_epochs,
                                                 args.dipredict_batch_size, args.dipredict_lr))

    dipredict_att_loss_result_path = os.path.join(current_directory,
                                              'results/DiPredict_Conv1d/{}_{}_{}'.format(args.dipredict_conv_layer_sizes,
                                                                                  args.dipredict_n_epochs,
                                                                                  args.dipredict_lr))

    writer = SummaryWriter(dipredict_att_loss_result_path)

    # 创建Dataset和DataLoader
    print('-------------------创建Dataset和DataLoader-------------------')
    dataset = CustomDataset(feature_train, performance_train)
    dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

    # 初始化模型和优化器
    print('-------------------初始化模型和优化器-------------------')
    layer_sizes = eval(args.dipredict_conv_layer_sizes)

    model = Direct_Predict_Conv1d(layer_sizes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=1e-2, threshold_mode='rel')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)  # step_size设为400， gamma设为0.1

    # 开始训练
    print('-------------------开始训练-------------------')
    dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                    performance_scaler, args, dipredict_att_model_save_path, writer, device)

    writer.close()
    print('-------------------训练结束-------------------')

elif args.model == 'mdim':
    mt_dipredict_model_save_path = os.path.join(current_directory,
                                                'model_checkpoints/MtDiPredict/{}_{}_{}_{}_{}_checkpoint.pth'.format(
                                                    args.shared_layer_sizes, args.private_layer_sizes,
                                                    args.dipredict_n_epochs, args.dipredict_batch_size,
                                                    args.dipredict_lr))

    mt_dipredict_loss_result_path = os.path.join(current_directory,
                                                 'results/MtDiPredict/{}_{}_{}_{}'.format(args.shared_layer_sizes,
                                                                                          args.private_layer_sizes,
                                                                                          args.dipredict_n_epochs,
                                                                                          args.dipredict_lr))

    writer = SummaryWriter(mt_dipredict_loss_result_path)

    # 创建Dataset和DataLoader
    print('-------------------创建Dataset和DataLoader-------------------')
    dataset = CustomDataset(feature_train, performance_train)
    dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

    # 初始化模型和优化器
    print('-------------------初始化模型和优化器-------------------')
    shared_layer_sizes = eval(args.shared_layer_sizes)
    private_layer_sizes = eval(args.private_layer_sizes)

    model = Mt_Direct_Predict_MLP(shared_layer_sizes, private_layer_sizes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=1e-2, threshold_mode='rel')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)  # step_size设为400， gamma设为0.1

    # 开始训练
    print('-------------------开始训练-------------------')
    mt_dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                       performance_scaler, args, mt_dipredict_model_save_path, writer, device)

    writer.close()
    print('-------------------训练结束-------------------')

else:
    it_dipredict_model_save_path_R = os.path.join(current_directory,
                                                  'model_checkpoints/ItDiPredict_R/{}_{}_{}_{}_checkpoint.pth'.format(
                                                      args.individual_layer_sizes, args.dipredict_n_epochs,
                                                      args.dipredict_batch_size, args.dipredict_lr))

    it_dipredict_model_save_path_C = os.path.join(current_directory,
                                                  'model_checkpoints/ItDiPredict_C/{}_{}_{}_{}_checkpoint.pth'.format(
                                                      args.individual_layer_sizes, args.dipredict_n_epochs,
                                                      args.dipredict_batch_size, args.dipredict_lr))

    it_dipredict_model_save_path_Q = os.path.join(current_directory,
                                                  'model_checkpoints/ItDiPredict_Q/{}_{}_{}_{}_checkpoint.pth'.format(
                                                      args.individual_layer_sizes, args.dipredict_n_epochs,
                                                      args.dipredict_batch_size, args.dipredict_lr))

    if not os.path.exists(it_dipredict_model_save_path_R):
        it_dipredict_loss_result_path_R = os.path.join(current_directory, 'results/ItDiPredict_R/{}_{}_{}'.format(
            args.individual_layer_sizes, args.dipredict_n_epochs, args.dipredict_lr))

        writer_R = SummaryWriter(it_dipredict_loss_result_path_R)

        # 创建Dataset和DataLoader
        print('-------------------创建Dataset和DataLoader-------------------')
        dataset = CustomDataset(feature_train, performance_train[:, 0])
        dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

        # 初始化模型和优化器
        print('-------------------初始化召回率预测模型和优化器-------------------')
        individual_layer_sizes = eval(args.individual_layer_sizes)

        model_R = It_Direct_Predict_MLP(individual_layer_sizes)
        model_R.to(device)
        optimizer_R = optim.Adam(model_R.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
        scheduler_R = optim.lr_scheduler.StepLR(optimizer_R, step_size=400, gamma=0.1)  # step_size设为400， gamma设为0.1

        # 开始训练
        print('-------------------开始训练-------------------')
        it_dipredict_train(dataloader, model_R, optimizer_R, scheduler_R, feature_valid, performance_valid[:, 0],
                           performance_valid_raw[:, 0], args, it_dipredict_model_save_path_R, writer_R, device, 'r')

        writer_R.close()
        print('-------------------训练结束-------------------')

    if not os.path.exists(it_dipredict_model_save_path_C):
        it_dipredict_loss_result_path_C = os.path.join(current_directory, 'results/ItDiPredict_C/{}_{}_{}'.format(
            args.individual_layer_sizes, args.dipredict_n_epochs, args.dipredict_lr))

        writer_C = SummaryWriter(it_dipredict_loss_result_path_C)

        # 创建Dataset和DataLoader
        print('-------------------创建Dataset和DataLoader-------------------')
        dataset = CustomDataset(feature_train, performance_train[:, 1])
        dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

        # 初始化模型和优化器
        print('-------------------初始化索引构建时间预测模型和优化器-------------------')
        individual_layer_sizes = eval(args.individual_layer_sizes)

        model_C = It_Direct_Predict_MLP(individual_layer_sizes)
        model_C.to(device)
        optimizer_C = optim.Adam(model_C.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
        scheduler_C = optim.lr_scheduler.StepLR(optimizer_C, step_size=400, gamma=0.1)  # step_size设为400， gamma设为0.1

        # 开始训练
        print('-------------------开始训练-------------------')
        it_dipredict_train(dataloader, model_C, optimizer_C, scheduler_C, feature_valid, performance_valid[:, 1],
                           performance_valid_raw[:, 1], args, it_dipredict_model_save_path_C, writer_C, device, 'c')

        writer_C.close()
        print('-------------------训练结束-------------------')

    if not os.path.exists(it_dipredict_model_save_path_Q):
        it_dipredict_loss_result_path_Q = os.path.join(current_directory, 'results/ItDiPredict_Q/{}_{}_{}'.format(
            args.individual_layer_sizes, args.dipredict_n_epochs, args.dipredict_lr))

        writer_Q = SummaryWriter(it_dipredict_loss_result_path_Q)

        # 创建Dataset和DataLoader
        print('-------------------创建Dataset和DataLoader-------------------')
        dataset = CustomDataset(feature_train, performance_train[:, 2])
        dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

        # 初始化模型和优化器
        print('-------------------初始化qps预测模型和优化器-------------------')
        individual_layer_sizes = eval(args.individual_layer_sizes)

        model_Q = It_Direct_Predict_MLP(individual_layer_sizes)
        model_Q.to(device)
        optimizer_Q = optim.Adam(model_Q.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
        scheduler_Q = optim.lr_scheduler.StepLR(optimizer_Q, step_size=600, gamma=0.1)  # step_size设为400， gamma设为0.1

        # 开始训练
        print('-------------------开始训练-------------------')
        it_dipredict_train(dataloader, model_Q, optimizer_Q, scheduler_Q, feature_valid, performance_valid[:, 2],
                           performance_valid_raw[:, 2], args, it_dipredict_model_save_path_Q, writer_Q, device, 'q')

        writer_Q.close()
        print('-------------------训练结束-------------------')




