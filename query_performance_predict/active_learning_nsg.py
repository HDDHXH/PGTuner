import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
import random
from tqdm import tqdm
# import cupy as cp
# from cuml.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
from tqdm import tqdm

from models import Direct_Predict_MLP_nsg
from trainer import dipredict_train
from utils import read_data_new_nsg, split_data,  df2np, np2ts, CustomDataset, load_model, Scaler_minmax_new_gpu_nsg, calculate_errors
from Args import args

'''
注意，对于NSG,同样是考虑选择5%的构建参数配置，最大选择轮数的计算公式为： int(N * 0.05 / 2) + 1
不过对于NSG,为了防止选择轮数太大，直接选择所有的KNN构建参数配置，即16个；然后只在NSG构建参数配置(220个)中做主动学习，计算得到的最大选择轮数为6,然后总的最大选择的构建参数配置数就是6*2*16=192
对于PGTuner采用上述方式，对于GMM，则从整个候选构建配置中选择192个；对于RandomSearch，则从整个构建参数配置空间中随机选择192个。
'''
def check_is_normal_distribution(data):
    stat, p = stats.shapiro(data.flatten())
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('数据可能服从正态分布')
    else:
        print('数据不服从正态分布')

    mean = np.mean(data)
    std = np.std(data)
    stat, p = stats.kstest(data.flatten(), 'norm', args=(mean, std))
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('数据可能服从正态分布')
    else:
        print('数据不服从正态分布')

    result = stats.anderson(data.flatten(), dist='norm')
    print('Statistic: %.3f' % result.statistic)
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            print(f'在显著性水平 {sl} 下，数据可能服从正态分布')
        else:
            print(f'在显著性水平 {sl} 下，数据不服从正态分布')

    skewness = stats.skew(data.flatten())
    kurtosis = stats.kurtosis(data.flatten()) + 3  # scipy.kurtosis默认是excess kurtosis
    print(f"偏度: {skewness}")
    print(f"峰度: {kurtosis}")

    save_path = 'result.png'
    plt.hist(data, bins=100, color='blue', alpha=0.7, edgecolor='black')  # bins定义了直方图的条形数

    # 设置图表标题和坐标轴标签
    plt.title('Histogram of Data')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    plt.savefig(save_path)
    plt.close()


def get_feature_vectors(model, input_feature):
    model.eval()

    with torch.no_grad():
        input_feature_vectors = model.get_feature_vectors(input_feature, 3)
        input_feature_vectors_l2 = F.normalize(input_feature_vectors, p=2, dim=1)

    return input_feature_vectors_l2 

def get_nn_dist(labeled_feature_vectors, query_feature_vectors, k): #这里输入都是L2规范化后的特征向量，如果查询向量就是标注特征向量，那么k取2，；如果是未标注向量，则k取1

    labeled_feature_vectors_np = labeled_feature_vectors.cpu().numpy()
    query_feature_vectors_np = query_feature_vectors.cpu().numpy()

    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    nn.fit(labeled_feature_vectors_np)

    # 获取距离
    distances, _= nn.kneighbors(query_feature_vectors_np)  
    
    min_distances = distances[:, -1]
    min_distances_np = min_distances

    del distances
    del nn

    return min_distances_np

def get_input_feature(df_data_feature, df_config_unit):
    df_KNN_config = pd.read_csv('../NSG_KNNG/Data/KNN_config_unit_data.csv', sep=',', header=0)

    L_nsg_Ss = np.array( [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
            430, 460, 490, 520, 550, 580, 610, 640, 670, 700, 740, 780, 820, 860, 900, 960, 1020, 1080, 1140, 1200, 1300, 1400, 1500])

    df_config_temp = pd.merge(df_KNN_config, df_config_unit, how='cross')

    df_config_whole = df_config_temp.loc[df_config_temp.index.repeat(len(L_nsg_Ss))].reset_index(drop=True)
    df_config_whole['L_nsg_S'] = np.tile(L_nsg_Ss, len(df_config_temp))

    df_feature = pd.merge(df_data_feature, df_config_whole, on='FileName', how='right')

    return df_feature

def get_distance_statistics(distance):
    # 计算统计信息
    mean_value = np.mean(min_distance)
    std_value = np.std(min_distance)
    min_value = np.min(min_distance)
    max_value = np.max(min_distance)
    quartiles_value = np.percentile(min_distance, [25, 50, 75, 90, 95, 96, 98, 99])

    # 打印统计信息
    print(f"均值: {mean_value}")
    print(f"标准差: {std_value}")
    print(f"最小值: {min_value}")
    print(f"最大值: {max_value}")
    print(f"四分位数: {quartiles_value}")

    return mean_value, std_value, min_value, max_value, quartiles_value

def feature_df2feature_tensor(df_feature, device, feature_scaler):
    feature_raw = df2np(df_feature)
    feature_raw[:, 5:8] = np.log10(feature_raw[:, 5:8])

    feature_raw = np2ts(feature_raw).to(device)

    feature_tensor = feature_scaler.transform(feature_raw)

    del feature_raw

    return feature_tensor

def performance_df2performance_tensor(df_performance, device, performance_scaler):
    performance_raw = df2np(df_performance)

    performance_raw[:, 1] = np.log10(performance_raw[:, 1])

    performance_raw = np2ts(performance_raw).to(device)

    performance_tensor = performance_scaler.transform(performance_raw)

    del performance_raw

    return performance_tensor

def CoreSetSelecting2(labeled_feature_vectors, query_feature_vectors, selected_num):
    min_distances = get_nn_dist(labeled_feature_vectors, query_feature_vectors, 1)
    min_distances = min_distances.reshape((-1, 832))
    min_distances = np.mean(min_distances, axis=1)  #将每个efC和M下所有efS的距离的平均值作为该efC和M组合的距离

    min_index = np.argmin(min_distances)
    max_index = np.argmax(min_distances)

    mean_value = np.mean(min_distances )
    mean_index = np.argmin(np.abs(min_distances  - mean_value))

    if selected_num ==  1:
        selected_indices = [max_index]
    elif selected_num ==  2:
        selected_indices = [mean_index, max_index]
    else:
        selected_indices = [min_index, mean_index, max_index]


    return selected_indices


if __name__ == '__main__':
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

    L_nsg_Ss = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
            430, 460, 490, 520, 550, 580, 610, 640, 670, 700, 740, 780, 820, 860, 900, 960, 1020, 1080, 1140, 1200, 1300, 1400, 1500])

    init_model_save_path = os.path.join(current_directory, 'model_checkpoints/DiPredict/NSG_KNNG/{}_{}_{}_{}_checkpoint.pth'.format(
                                                        args.dipredict_layer_sizes_nsg, args.dipredict_n_epochs,
                                                        args.dipredict_batch_size, args.dipredict_lr))

    init_feature_standard_path = os.path.join(current_directory, 'scaler_paras/NSG_KNNG/feature_standard.npz')

    data_fetaure_path = os.path.join(parent_directory, 'NSG_KNNG/Data/whole_data_feature.csv') #注意，现在的whole_data_feature只包含输入特征，不包含其它特征

    #注意：这里两个文件都是只包含输入特征和输出特征的数据，不包含其他的信息，是彻底的训练和测试数据，总的文件在前面加上whole
    init_train_data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/train_data.csv') #用于加载训练数据（初始标注数据）
    init_test_data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/test_data_main.csv') #用于加载选择的未标注参数组合的标注数据

    NSG_config_unit_data_path = "../NSG_KNNG/Data/NSG_config_unit_data.csv"
    #selected_config_base_path = "../Data/selected_config_base.csv"

    filename_dic = {'deep1': 'deep_0_1_96_1', 'paper': 'paper_0_2_200_1', 'gist': 'gist_0_1_960_1'}

    #selected_num = 2
    selected_group_num = 12  #研究selected_num的作用时，需要固定最大的选择的配置组合的数量，然后来比较不同selected_group_num的影响。sift50用18,另外3个用24；这个同时也能体现出在固定selected_num的情况下，selected_rounds的作用
    #selected_rounds = selected_group_num // selected_num  #研究selected_rounds的作用时，就将selected_num固定为2, selected_rounds的最大值

    #'gist_25', 'gist_50', 'gist_75', 'gist_100', 'deep2_25', 'deep2_50', 'deep2_75', 'deep2_100'
    #'deep2', 'deep3', 'deep4', 'deep5', 'sift2', 'sift3', 'sift4', 'sift5'
    for selected_num in [2]:
        for dataset_name in ['gist']:
            print(dataset_name)
            selected_rounds = selected_group_num // selected_num
            #print('训练数据')
            filename = filename_dic[dataset_name]
            print(filename)

            NSG_config_unit_data_path = "../NSG_KNNG/Data/NSG_config_unit_data.csv"

            new_train_data_path = "../Data/active_learning_data/NSG_KNNG/{}_train_data_{}_{}.csv".format(dataset_name, selected_num, selected_rounds)
            selected_config_path = "../Data/active_learning_data/NSG_KNNG/{}_selected_config_{}_{}.csv".format(dataset_name, selected_num, selected_rounds)
            fig_save_path = "../Data/active_learning_data/NSG_KNNG/{}_distance_{}_{}.png".format(dataset_name, selected_num, selected_rounds)

            new_feature_standard_path = os.path.join(current_directory, 'scaler_paras/NSG_KNNG/{}_feature_standard_{}_{}.npz'.format(dataset_name, selected_num, selected_rounds))
            new_model_save_path = os.path.join(current_directory, 'model_checkpoints/DiPredict/NSG_KNNG/{}_{}_{}_{}_{}_{}_{}_checkpoint.pth'.format(dataset_name,
                                                            args.dipredict_layer_sizes_nsg, args.dipredict_n_epochs,
                                                            args.dipredict_batch_size, args.dipredict_lr, selected_num, selected_rounds))

            test_error_dic = {'rec_MAE': [], 'NSG_st_dc_MAE': [], 'rec_MAPE': [], 'NSG_st_dc_MAPE': [], 'distance_threshold95': [], 'distance_threshold96': [],
                              'distance_threshold98': [], 'distance_threshold99': [], 'mean_distance': [], 'duration_time': [], 'each_detect_time': []}
            test_error_path = "../Data/active_learning_data/NSG_KNNG/{}_test_error_{}_{}.csv".format(dataset_name, selected_num, selected_rounds)

            print('-------------------加载归一化器-------------------')
            feature_scaler = Scaler_minmax_new_gpu_nsg(9, device)
            feature_scaler.load_parameters(None, init_feature_standard_path, device)
            performance_scaler = Scaler_minmax_new_gpu_nsg(0, device)

            print('-------------------加载模型-------------------')
            dipredict_layer_sizes_nsg = eval(args.dipredict_layer_sizes_nsg)

            model = Direct_Predict_MLP_nsg(dipredict_layer_sizes_nsg)
            model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)

            model, _, _ = load_model(model, optimizer, init_model_save_path)

            print('-------------------加载初始标注数据并生成特征向量-------------------')
            df_data_feature = pd.read_csv(data_fetaure_path, sep=',', header=0)

            current_train_df = pd.read_csv(init_train_data_path, sep=',', header=0)  #先用于获取参与训练数据的高维向量数据集的名称，后面还会参与主动学习

            # active_learning的区别是基输入特征向量直接使用训练数据的输入特征向量，而不是组合
            selected_df_feature_labeled = current_train_df[ ['K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist',
                 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]  # 直接从训练数据中提取的输入特征向量是用于配置选择

            selected_labeled_feature = feature_df2feature_tensor(selected_df_feature_labeled, device, feature_scaler)
            selected_labeled_feature_vectors = get_feature_vectors(model, selected_labeled_feature)

            detected_df_feature_labeled = current_train_df[
                ['K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist',
                 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]

            detected_labeled_feature = feature_df2feature_tensor(detected_df_feature_labeled, device, feature_scaler)
            detected_labeled_feature_vectors = get_feature_vectors(model, detected_labeled_feature)

            print('-------------------计算标注数据特征向量与其最近邻的距离-------------------')
            min_distance = get_nn_dist(detected_labeled_feature_vectors,detected_labeled_feature_vectors, 2)

            #_, _, _, _,  _ = get_distance_statistics(min_distance)
            #check_is_normal_distribution(min_distance)
            distance_threshold = np.percentile(min_distance, 95)  # 更新距离阈值
            # distance_threshold2 = np.percentile(min_distance, 96)
            # distance_threshold3 = np.percentile(min_distance, 98)
            # distance_threshold4 = np.percentile(min_distance, 99)
            print(distance_threshold)
            # print(distance_threshold2)
            # print(distance_threshold3)
            # print(distance_threshold4)

            print('-------------------加载未标注数据并生成特征向量-------------------')
            df_data_feature_test = df_data_feature[df_data_feature['FileName'] == filename]

            df_config_unit = pd.read_csv(NSG_config_unit_data_path, sep=',', header=0)
            df_config_unit['FileName'] = filename

            df_feature_unlabeled = get_input_feature(df_data_feature_test, df_config_unit)

            df_feature_query = df_feature_unlabeled[['K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']]

            query_feature = feature_df2feature_tensor(df_feature_query, device, feature_scaler)
            query_feature_vectors = get_feature_vectors(model, query_feature)

            print('-------------------计算未标注数据特征向量与其在标注数据特征向量中最近邻的距离-------------------')
            min_distance = get_nn_dist(detected_labeled_feature_vectors, query_feature_vectors, 1)
            #config_min_distance = min_distance.reshape((-1, 63))
            #_, _, _, _, _ = get_distance_statistics(min_distance)
            mean_distance = np.mean(min_distance)
            print(mean_distance)
            dist_flag = (mean_distance <= distance_threshold)

            if dist_flag:
                #当前未标注数据与已标注数据相似，直接使用模型
                print('当前未标注数据与已标注数据相似，可直接使用模型')
            #'''
            else:
                #当前未标注数据与已标注数据不相似，开始主动学习
                real_data_test_df = pd.read_csv(init_test_data_path, sep=',', header=0)
                real_data_test_df = real_data_test_df[real_data_test_df['FileName'] == filename]
            
                current_detected_df_feature_labeled = detected_df_feature_labeled.copy() #用于更新数据相似检测的所有标注组合数据
                current_selected_df_feature_labeled = selected_df_feature_labeled.copy() #用于更新配置选择的所有训练输入特征数据
                current_df_config_unlabeled = df_config_unit.copy()  ##用于更新剩余未标注的efC和M组合数据

                #每次选择完未标注数据进行标注，然后重新训练更新模型后，都要做一次数据相似检测，如果flag未True则可以停止主动学习了；或者选择轮数达到最大轮数，同样终止主动学习
                ts = time.time()
                for round_num in tqdm(range(selected_rounds), total = selected_rounds):
                    print('-------------------开始选择未标注数据------------------')
                    #selected_indices = CoreSetSelecting(labeled_feature_vectors, query_feature_vectors, selected_num)  #每轮选2个efC和M组合，最多选12轮，即24个（10%）。注意，这里的labeled_feature_vectors和query_feature_vectors均是当前最新的
                    selected_indices = CoreSetSelecting2(selected_labeled_feature_vectors, query_feature_vectors, selected_num)
                    df_selected_config = current_df_config_unlabeled.iloc[selected_indices]
                    # print(df_selected_config)

                    if not os.path.exists(selected_config_path):
                        # 如果文件不存在，写入文件（包括列名）
                        df_selected_config.to_csv(selected_config_path, index=False, mode='w', header=True)
                    else:
                        # 如果文件已存在，附加到文件末尾（不包括列名）
                        df_selected_config.to_csv(selected_config_path, index=False, mode='a', header=False)

                    #更新未标注的efC和M的组合
                    current_df_config_unlabeled = current_df_config_unlabeled.drop(selected_indices)
                    current_df_config_unlabeled = current_df_config_unlabeled.reset_index(drop=True)
 
                    #获取选择的config的训练数据，与已有的训练数据混合重新训练模型
                    selected_data_df = pd.merge(real_data_test_df, df_selected_config, on=['FileName', 'L_nsg_C', 'R_nsg', 'C'], how='right')
                    current_train_df = pd.concat([current_train_df, selected_data_df], axis=0)

                    current_selected_df_feature_labeled = current_train_df[['K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist',
                         'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]
                         
                    current_detected_df_feature_labeled = current_train_df[['K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist',
                                                                'Sum_K_MaxDist',  'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]
 
                    print('-------------------获取新的训练数据，进行数据处理------------------')
                    df_train, df_test, df_valid = split_data(current_train_df)
                    df_train = pd.concat([df_train, df_test], axis=0)

                    df_train_f, df_train_p = read_data_new_nsg(df_train)
                    df_valid_f, df_valid_p = read_data_new_nsg(df_valid)

                    df_train_f = df_train_f.drop(['FileName'], axis=1)
                    df_valid_f = df_valid_f.drop(['FileName'], axis=1)

                    whole_df_f = pd.concat([df_train_f, df_valid_f], axis=0) #还是要用全部数据更新标准化参数
                    whole_feature = df2np(whole_df_f)
                    whole_feature[:, 5:8] = np.log10(whole_feature[:, 5:8])

                    whole_feature = np2ts(whole_feature).to(device)
                    feature_scaler.fit(whole_feature)
                    feature_scaler.save_parameters(None, new_feature_standard_path)

                    feature_train =feature_df2feature_tensor(df_train_f, device, feature_scaler)
                    feature_valid =feature_df2feature_tensor(df_valid_f, device, feature_scaler)

                    performance_train = performance_df2performance_tensor(df_train_p, device, performance_scaler)
                    performance_valid = performance_df2performance_tensor(df_valid_p, device, performance_scaler)

                    performance_valid_raw = performance_scaler.inverse_transform(performance_valid)
                    performance_valid_raw[:, 1] = torch.pow(10, performance_valid_raw[:, 1])

                    dataset = CustomDataset(feature_train, performance_train)
                    dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

                    print('开始训练模型')
                    model = Direct_Predict_MLP_nsg(dipredict_layer_sizes_nsg)
                    model.to(device)

                    optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

                    dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                                    performance_scaler, args, new_model_save_path, None, device)

                    #测试新模型的预测性能
                    df_test_f, df_test_p = read_data_new_nsg(real_data_test_df)
                    df_test_f = df_test_f.drop(['FileName'], axis=1)

                    feature_test = feature_df2feature_tensor(df_test_f, device, feature_scaler) #所以这里labeled_feature并不是前面的feature_train
                    performance_test_raw = df2np(df_test_p)
                    performance_test_raw = np2ts(performance_test_raw).to(device)

                    model.eval()
                    with torch.no_grad():
                        predicted_performances = model(feature_test)
                        predicted_performances = performance_scaler.inverse_transform(predicted_performances)
                        predicted_performances[:, 1] = torch.pow(10, predicted_performances[:, 1])

                        mean_errors, mean_errors_percent, _ = calculate_errors(performance_test_raw, predicted_performances)  # 这里误差是一个3维张量，是所有验证样本的平均
                        print('预测误差为：')
                        print(f'mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}')

                        test_error_dic['rec_MAE'].append(mean_errors[0].item())
                        test_error_dic['NSG_st_dc_MAE'].append(mean_errors[1].item())
                        test_error_dic['rec_MAPE'].append(mean_errors_percent[0].item())
                        test_error_dic['NSG_st_dc_MAPE'].append(mean_errors_percent[1].item())

                    print('模型训练完毕，重新检测')
                    #切记，数据相似检测是用的是所有标注组合的特征数据，而不是用标注的训练数据
                    t1 = time.time()
                    detected_labeled_feature = feature_df2feature_tensor(current_detected_df_feature_labeled, device, feature_scaler) #所以这里labeled_feature并不是前面的feature_train
                    detected_labeled_feature_vectors = get_feature_vectors(model, detected_labeled_feature)

                    selected_labeled_feature = feature_df2feature_tensor(current_selected_df_feature_labeled, device, feature_scaler)
                    selected_labeled_feature_vectors = get_feature_vectors(model, selected_labeled_feature)

                    min_distance = get_nn_dist(detected_labeled_feature_vectors, detected_labeled_feature_vectors, 2)
                    distance_threshold1 = np.percentile(min_distance, 95)  # 更新距离阈值
                    distance_threshold2 = np.percentile(min_distance, 96)
                    distance_threshold3 = np.percentile(min_distance, 98)
                    distance_threshold4 = np.percentile(min_distance, 99)

                    df_feature_unlabeled = get_input_feature(df_data_feature_test, current_df_config_unlabeled)

                    df_feature_query = df_feature_unlabeled[['K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist',
                                                             'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']]

                    query_feature = feature_df2feature_tensor(df_feature_query, device, feature_scaler)
                    query_feature_vectors = get_feature_vectors(model, query_feature)

                    min_distance = get_nn_dist(detected_labeled_feature_vectors, query_feature_vectors, 1)
                    mean_distance = np.mean(min_distance)
                    dist_flag = (mean_distance <= distance_threshold1)

                    test_error_dic['distance_threshold95'].append(distance_threshold1)
                    test_error_dic['distance_threshold96'].append(distance_threshold2)
                    test_error_dic['distance_threshold98'].append(distance_threshold3)
                    test_error_dic['distance_threshold99'].append(distance_threshold4)
                    test_error_dic['mean_distance'].append(mean_distance)
                    # print(f'distance_threshold:{distance_threshold}')
                    # print(f'mean_distance:{mean_distance}')

                    td = time.time()
                    duration_time = td - ts
                    detect_time = td - t1

                    test_error_dic['duration_time'].append(duration_time)
                    test_error_dic['each_detect_time'].append(detect_time)
                    
                    test_error_df = pd.DataFrame(test_error_dic)
                    test_error_df['FileName'] = filename
                    test_error_df.to_csv(test_error_path, index=False, mode='w', header=True)

                    if dist_flag:
                        break

            current_train_df.to_csv(new_train_data_path, mode='w', index=False)
            #'''






                












        





    
    
    







    
    



