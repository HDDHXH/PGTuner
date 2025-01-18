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

from models import Direct_Predict_MLP
from trainer import dipredict_train
from utils import read_data, read_data_new, read_unlabeld_data_new, split_data, df2np, np2ts, CustomDataset, \
    Scaler_minmax, load_model, save_model, Scaler_minmax_new_gpu, calculate_errors
from Args import args

'''
用于检测基输入特征向量是用的组合特征，而用于选择的基输入特征向量是用的训练数据。这种方式字啊检测和放那花效果上是最好的，也是最符合直觉的，所以主动学习就用这个
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


def get_nn_dist(labeled_feature_vectors, query_feature_vectors,
                k):  # 这里输入都是L2规范化后的特征向量，如果查询向量就是标注特征向量，那么k取2，；如果是未标注向量，则k取1
    # labeled_feature_vectors_cp = cp.asarray(labeled_feature_vectors)  #cuml暂时下载不了
    # query_feature_vectors_cp = cp.asarray(query_feature_vectors)

    # nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    # nn.fit(labeled_feature_vectors_cp)

    # # 获取距离
    # distances, _= nn.kneighbors(query_feature_vectors_cp)

    # if k == 2: #第一列的距离全为0，每个特征向量与其最近邻的距离其实是第2列
    #     min_distances = distances[:, 1]  #min_distances表示的是每个查询特征向量与所有标注的特征向量的距离的最小值，这不就是查询特征向量与其在标注特征向量中的最近邻的距离
    # else:
    #     min_distances = distances

    # min_distances_np = cp.asnumpy(min_distances)
    # **************************************************************************************************************************
    labeled_feature_vectors_np = labeled_feature_vectors.cpu().numpy()
    query_feature_vectors_np = query_feature_vectors.cpu().numpy()

    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    nn.fit(labeled_feature_vectors_np)

    # 获取距离
    distances, _ = nn.kneighbors(query_feature_vectors_np)

    min_distances = distances[:, -1]
    min_distances_np = min_distances

    del distances
    del nn

    return min_distances_np


def get_input_feature(df_data_feature, df_config_unit):
    efSs = np.array(
        [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320,
         340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640,
         660, 680, 700, 730, 760, 790, 820, 850, 880, 910, 940, 970, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
         1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100,
         3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
         5000])

    df_config_whole = df_config_unit.loc[df_config_unit.index.repeat(len(efSs))].reset_index(drop=True)
    df_config_whole['efSearch'] = np.tile(efSs, len(df_config_unit))

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

    feature_raw[:, 0] = np.log10(feature_raw[:, 0])
    feature_raw[:, 2:5] = np.log10(feature_raw[:, 2:5])

    feature_raw = np2ts(feature_raw).to(device)

    feature_tensor = feature_scaler.transform(feature_raw)

    del feature_raw

    return feature_tensor


def performance_df2performance_tensor(df_performance, device, performance_scaler):
    performance_raw = df2np(df_performance)

    performance_raw[:, 1:] = np.log10(performance_raw[:, 1:])

    performance_raw = np2ts(performance_raw).to(device)

    performance_tensor = performance_scaler.transform(performance_raw)

    del performance_raw

    return performance_tensor


def CoreSetSelecting(labeled_feature_vectors, query_feature_vectors, amount):
    selected_indices = []

    min_distances = get_nn_dist(labeled_feature_vectors, query_feature_vectors, 1)
    min_distances = min_distances.reshape((-1, 94))
    min_distances = np.mean(min_distances, axis=1)  # 将每个efC和M下所有efS的距离的平均值作为该efC和M组合的距离

    farthest = np.argmax(min_distances)
    selected_indices.append(farthest)

    for i in range(amount - 1):
        new_selected_indice = selected_indices[-1]
        new_selected_feature_vectors = query_feature_vectors[new_selected_indice * 63: (new_selected_indice + 1) * 63,
                                       :]

        new_min_distances = get_nn_dist(new_selected_feature_vectors, query_feature_vectors, 1)
        new_min_distances = new_min_distances.reshape((-1, 94))
        new_min_distances = np.mean(new_min_distances, axis=1)

        min_distances = np.minimum(min_distances, new_min_distances)

        farthest = np.argmax(min_distances)
        selected_indices.append(farthest)

    return selected_indices


def CoreSetSelecting2(labeled_feature_vectors, query_feature_vectors, selected_num):
    min_distances = get_nn_dist(labeled_feature_vectors, query_feature_vectors, 1)
    min_distances = min_distances.reshape((-1, 94))
    min_distances = np.mean(min_distances, axis=1)  # 将每个efC和M下所有efS的距离的平均值作为该efC和M组合的距离

    min_index = np.argmin(min_distances)
    max_index = np.argmax(min_distances)

    mean_value = np.mean(min_distances)
    mean_index = np.argmin(np.abs(min_distances - mean_value))

    if selected_num == 1:
        selected_indices = [max_index]
    elif selected_num == 2:
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

    efSs = np.array(
        [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320,
         340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640,
         660, 680, 700, 730, 760, 790, 820, 850, 880, 910, 940, 970, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
         1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100,
         3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
         5000])
    # print(len(efSs)) 94

    init_model_save_path = os.path.join(current_directory,
                                        'model_checkpoints/DiPredict/{}_{}_{}_{}_checkpoint.pth'.format(
                                            args.dipredict_layer_sizes, args.dipredict_n_epochs,
                                            args.dipredict_batch_size, args.dipredict_lr))

    init_feature_standard_path = os.path.join(current_directory, 'scaler_paras/feature_standard.npz')

    data_fetaure_path = os.path.join(parent_directory,  'Data/whole_data_feature.csv')  # 注意，现在的whole_data_feature只包含输入特征，不包含其它特征

    # 注意：这里两个文件都是只包含输入特征和输出特征的数据，不包含其他的信息，是彻底的训练和测试数据，总的文件在前面加上whole
    init_train_data_path = os.path.join(parent_directory, 'Data/train_data.csv')  # 用于加载训练数据（初始标注数据）
    #init_test_data_path = os.path.join(parent_directory, 'Data/test_data_ds_change.csv')
    init_test_data_path = os.path.join(parent_directory, 'Data/test_data_qw_change.csv')  # 用于加载选择的未标注参数组合的标注数据
    #init_test_data_path = os.path.join(parent_directory, 'Data/test_data_main.csv')

    config_unit_data_path = "../Data/config_unit_data.csv"
    # selected_config_base_path = "../Data/selected_config_base.csv"

    filename_dic = {'deep1': 'deep_1_1_96_1', 'sift1': 'sift_1_1_128_1', 'glove': 'glove_1_1.183514_100',
                    'paper': 'paper_1_2.029997_200', 'crawl': 'crawl_1_1.989995_300', 'msong': 'msong_0_9.92272_420',
                    'gist': 'gist_1_1.0_960', 'deep10': 'deep_2_1_96', 'sift50': 'sift_2_5_128_1', 'deep2': 'deep_1_2_96_1', 'deep3': 'deep_1_3_96_1', 'deep4': 'deep_1_4_96_1',
                    'deep5': 'deep_1_5_96_1', 'sift2': 'sift_1_2_128_1', 'sift3': 'sift_1_3_128_1', 'sift4': 'sift_1_4_128_1',
                    'sift5': 'sift_1_5_128_1', 'gist_25': 'gist_1_1.0_960_25', 'gist_50': 'gist_1_1.0_960_50',
                    'gist_75': 'gist_1_1.0_960_75', 'gist_100': 'gist_1_1.0_960_100', 'deep2_25': 'deep_1_2_96_1_25',
                    'deep2_50': 'deep_1_2_96_1_50', 'deep2_75': 'deep_1_2_96_1_75', 'deep2_100': 'deep_1_2_96_1_100'}

    # selected_num = 1
    selected_group_num = 14  # 研究selected_num的作用时，需要固定最大的选择的配置组合的数量，然后来比较不同selected_group_num的影响。sift50用18,另外3个用24；这个同时也能体现出在固定selected_num的情况下，selected_rounds的作用
    # selected_rounds = selected_group_num // selected_num  #研究selected_rounds的作用时，就将selected_num固定为2, selected_rounds的最大值
    selected_num = 2

    mdoe = 'qw_change'
    #mdoe = 'dast_change'

    # 'gist_25', 'gist_50', 'gist_75', 'gist_100', 'deep2_25', 'deep2_50', 'deep2_75', 'deep2_100'
    #dataset_name_list = ['deep2', 'deep3', 'deep4', 'deep5']
    #dataset_name_list = ['sift2', 'sift3', 'sift4', 'sift5']
    #dataset_name_list = ['deep2', 'deep2_25', 'deep2_50', 'deep2_75', 'deep2_100']
    dataset_name_list = ['gist', 'gist_25', 'gist_50', 'gist_75', 'gist_100']

    #dataset_name_list = ['deep2', 'deep10']
    #dataset_name_list = ['sift2', 'sift50']
    #dataset_name_list = ['gist', 'gist_100']

    #dataset_name_list = ['glove', 'sift50', 'deep10', 'gist']

    for dataset_name in ['gist_100']:
        print(dataset_name)
        i = dataset_name_list.index(dataset_name)

        selected_rounds = selected_group_num // selected_num
        # print('训练数据')
        filename = filename_dic[dataset_name]
        print(filename)

        if dataset_name == 'sift50':
            config_unit_data_path = "../Data/config_unit_data_sift.csv"

        else:
            config_unit_data_path = "../Data/config_unit_data.csv"

        if i >= 1:  # 从第二次数据更新开始，每一次用的模型、归一化数据和训练数据都是上一次数据更新后产生的
            last_dataset_name = dataset_name_list[i - 1]
            init_train_data_path = "../Data/active_learning_data/{}/{}_train_data_{}_{}.csv".format(mdoe, last_dataset_name, selected_num, selected_rounds)
            init_feature_standard_path = os.path.join(current_directory,
                                                      'scaler_paras/{}/{}_feature_standard_{}_{}.npz'.format(mdoe, last_dataset_name, selected_num,  selected_rounds))
            init_model_save_path = os.path.join(current_directory, 'model_checkpoints/DiPredict/{}/{}_{}_{}_{}_{}_{}_{}_checkpoint.pth'.format(
                                                    mdoe, last_dataset_name, args.dipredict_layer_sizes, args.dipredict_n_epochs,
                                                    args.dipredict_batch_size, args.dipredict_lr, selected_num,  selected_rounds))

        new_train_data_path = "../Data/active_learning_data/{}/{}_train_data_{}_{}.csv".format(mdoe, dataset_name, selected_num, selected_rounds)
        selected_config_path = "../Data/active_learning_data/{}/{}_selected_config_{}_{}.csv".format(mdoe, dataset_name, selected_num, selected_rounds)
        fig_save_path = "../Data/active_learning_data/{}/{}_distance_{}_{}.png".format(mdoe, dataset_name, selected_num, selected_rounds)

        new_feature_standard_path = os.path.join(current_directory, 'scaler_paras/{}/{}_feature_standard_{}_{}.npz'.format(mdoe, dataset_name, selected_num, selected_rounds))
        new_model_save_path = os.path.join(current_directory,
                                           'model_checkpoints/DiPredict/{}/{}_{}_{}_{}_{}_{}_{}_checkpoint.pth'.format(mdoe, dataset_name, args.dipredict_layer_sizes, args.dipredict_n_epochs,
                                               args.dipredict_batch_size, args.dipredict_lr, selected_num, selected_rounds))

        test_error_dic = {'rec_MAE': [], 'ct_dc_MAE': [], 'st_dc_MAE': [], 'rec_MAPE': [], 'ct_dc_MAPE': [], 'st_dc_MAPE': [], 'distance_threshold95': [], 'distance_threshold96': [],
                          'distance_threshold98': [], 'distance_threshold99': [], 'mean_distance': [], 'duration_time': [], 'each_detect_time': []}
        test_error_path = "../Data/active_learning_data/{}/{}_test_error_{}_{}.csv".format(mdoe, dataset_name,  selected_num, selected_rounds)

        print('-------------------加载归一化器-------------------')
        feature_scaler = Scaler_minmax_new_gpu(6, device)
        feature_scaler.load_parameters(None, init_feature_standard_path, device)
        performance_scaler = Scaler_minmax_new_gpu(0, device)

        print('-------------------加载模型-------------------')
        dipredict_layer_sizes = eval(args.dipredict_layer_sizes)

        model = Direct_Predict_MLP(dipredict_layer_sizes)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)

        model, _, _ = load_model(model, optimizer, init_model_save_path)

        print('-------------------加载初始标注数据并生成特征向量-------------------')
        df_data_feature = pd.read_csv(data_fetaure_path, sep=',', header=0)

        current_train_df = pd.read_csv(init_train_data_path, sep=',', header=0)  # 先用于获取参与训练数据的高维向量数据集的名称，后面还会参与主动学习

        # active_learning的区别是基输入特征向量直接使用训练数据的输入特征向量，而不是组合
        selected_df_feature_labeled = current_train_df[
            ['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist',
             'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio',
             'q_K_StdRatio']]  # 直接从训练数据中提取的输入特征向量是用于配置选择

        selected_labeled_feature = feature_df2feature_tensor(selected_df_feature_labeled, device, feature_scaler)
        selected_labeled_feature_vectors = get_feature_vectors(model, selected_labeled_feature)

        lanbeled_filename_lis = current_train_df['FileName'].unique().tolist()

        # df_combined = []
        #
        # for labeled_filename in lanbeled_filename_lis:
        #     df_data_feature_labeled = df_data_feature[df_data_feature['FileName'] == labeled_filename]
        #
        #     df_config_unit = pd.read_csv(config_unit_data_path, sep=',', header=0)
        #     df_config_unit['FileName'] = labeled_filename
        #
        #     df_feature_labeled = get_input_feature(df_data_feature_labeled, df_config_unit)
        #
        #     df_combined.append(df_feature_labeled)
        #
        # detected_labeled_df = pd.concat(df_combined, ignore_index=True)
        # # 组合得到的输入特征向量用于检测
        # detected_df_feature_labeled = detected_labeled_df[
        #     ['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist',
        #      'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]  # 初始选择未标注数据时就可以用到

        detected_df_feature_labeled = current_train_df[['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist',
                                                        'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]

        detected_labeled_feature = feature_df2feature_tensor(detected_df_feature_labeled, device, feature_scaler)
        detected_labeled_feature_vectors = get_feature_vectors(model, detected_labeled_feature)

        print('-------------------计算标注数据特征向量与其最近邻的距离-------------------')
        min_distance = get_nn_dist(detected_labeled_feature_vectors, detected_labeled_feature_vectors, 2)

        # _, _, _, _,  _ = get_distance_statistics(min_distance)
        # check_is_normal_distribution(min_distance)
        distance_threshold = np.percentile(min_distance, 95)  # 更新距离阈值
        # distance_threshold2 = np.percentile(min_distance, 96)
        # distance_threshold3 = np.percentile(min_distance, 98)
        # distance_threshold4 = np.percentile(min_distance, 99)
        #print(distance_threshold)
        # print(distance_threshold2)
        # print(distance_threshold3)
        # print(distance_threshold4)

        print('-------------------加载未标注数据并生成特征向量-------------------')
        df_data_feature_test = df_data_feature[df_data_feature['FileName'] == filename]

        df_config_unit = pd.read_csv(config_unit_data_path, sep=',', header=0)
        df_config_unit['FileName'] = filename

        df_feature_unlabeled = get_input_feature(df_data_feature_test, df_config_unit)

        df_feature_query = df_feature_unlabeled[
            ['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist',
             'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]

        query_feature = feature_df2feature_tensor(df_feature_query, device, feature_scaler)
        query_feature_vectors = get_feature_vectors(model, query_feature)

        print('-------------------计算未标注数据特征向量与其在标注数据特征向量中最近邻的距离-------------------')
        min_distance = get_nn_dist(detected_labeled_feature_vectors, query_feature_vectors, 1)
        # config_min_distance = min_distance.reshape((-1, 63))
        # _, _, _, _, _ = get_distance_statistics(min_distance)
        mean_distance = np.mean(min_distance)
        print(mean_distance)
        dist_flag = (mean_distance <= distance_threshold)

        # config_mean_distance = np.mean(config_min_distance, axis=1)
        # plt.figure(figsize=(28, 12))
        # plt.plot(config_mean_distance, marker='o', linestyle='-', color='red')
        # plt.title('Distance per Config')
        # plt.xlabel('Config')
        # plt.ylabel('Distance')
        # plt.grid(True)
        # plt.savefig(fig_save_path)
        # plt.close()

        if dist_flag:
            # 当前未标注数据与已标注数据相似，直接使用模型
            print('当前未标注数据与已标注数据相似，可直接使用模型')
            feature_scaler.save_parameters(None, new_feature_standard_path)
            save_model(model, optimizer, args.dipredict_n_epochs, new_model_save_path)
            current_train_df.to_csv(new_train_data_path, mode='w', index=False)

        # '''
        else:
            # 当前未标注数据与已标注数据不相似，开始主动学习
            real_data_test_df = pd.read_csv(init_test_data_path, sep=',', header=0)
            real_data_test_df = real_data_test_df[real_data_test_df['FileName'] == filename]

            current_detected_df_feature_labeled = detected_df_feature_labeled.copy()  # 用于更新数据相似检测的所有标注组合数据
            current_selected_df_feature_labeled = selected_df_feature_labeled.copy()  # 用于更新配置选择的所有训练输入特征数据
            current_df_config_unlabeled = df_config_unit.copy()  ##用于更新剩余未标注的efC和M组合数据

            # 每次选择完未标注数据进行标注，然后重新训练更新模型后，都要做一次数据相似检测，如果flag未True则可以停止主动学习了；或者选择轮数达到最大轮数，同样终止主动学习
            ts = time.time()
            for round_num in tqdm(range(selected_rounds), total=selected_rounds):
                print('-------------------开始选择未标注数据------------------')
                # selected_indices = CoreSetSelecting(labeled_feature_vectors, query_feature_vectors, selected_num)  #每轮选2个efC和M组合，最多选12轮，即24个（10%）。注意，这里的labeled_feature_vectors和query_feature_vectors均是当前最新的
                selected_indices = CoreSetSelecting2(selected_labeled_feature_vectors, query_feature_vectors, selected_num)
                df_selected_config = current_df_config_unlabeled.iloc[selected_indices]
                # print(df_selected_config)

                if not os.path.exists(selected_config_path):
                    # 如果文件不存在，写入文件（包括列名）
                    df_selected_config.to_csv(selected_config_path, index=False, mode='w', header=True)
                else:
                    # 如果文件已存在，附加到文件末尾（不包括列名）
                    df_selected_config.to_csv(selected_config_path, index=False, mode='a', header=False)

                # 更新未标注的efC和M的组合
                current_df_config_unlabeled = current_df_config_unlabeled.drop(selected_indices)
                current_df_config_unlabeled = current_df_config_unlabeled.reset_index(drop=True)

                # 更新全部的标注组合数据
                # labeled_df_selected = get_input_feature(df_data_feature_test, df_selected_config)
                # df_feature_labeled_selected = labeled_df_selected[
                #     ['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist',
                #      'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio',
                #      'q_K_StdRatio']]
                #
                # current_detected_df_feature_labeled = pd.concat(
                #     [current_detected_df_feature_labeled, df_feature_labeled_selected], axis=0)

                # 获取选择的config的训练数据，与已有的训练数据混合重新训练模型
                selected_data_df = pd.merge(real_data_test_df, df_selected_config,
                                            on=['FileName', 'efConstruction', 'M'], how='right')
                current_train_df = pd.concat([current_train_df, selected_data_df], axis=0)

                current_selected_df_feature_labeled = current_train_df[
                    ['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist',
                     'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio',
                     'q_K_StdRatio']]

                current_detected_df_feature_labeled = current_train_df[['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist',
                                                                'Sum_K_MaxDist',  'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]

                print('-------------------获取新的训练数据，进行数据处理------------------')
                # if round_num == 0:
                #     current_feature = df2np(current_df_feature_labeled)
                #     current_feature[:, 0] = np.log10(current_feature[:, 0])
                #     current_feature[:, 2:5] = np.log10(current_feature[:, 2:5])
                #
                #     current_feature = np2ts(current_feature).to(device)
                #     feature_scaler.fit(current_feature)
                #     feature_scaler.save_parameters(None, new_feature_standard_path)

                df_train, df_test, df_valid = split_data(current_train_df)
                df_train = pd.concat([df_train, df_test], axis=0)

                df_train_f, df_train_p = read_data_new(df_train)
                df_valid_f, df_valid_p = read_data_new(df_valid)

                df_train_f = df_train_f.drop(['FileName'], axis=1)
                df_valid_f = df_valid_f.drop(['FileName'], axis=1)

                whole_df_f = pd.concat([df_train_f, df_valid_f], axis=0)  # 还是要用全部数据更新标准化参数
                whole_feature = df2np(whole_df_f)
                whole_feature[:, 0] = np.log10(whole_feature[:, 0])
                whole_feature[:, 2:5] = np.log10(whole_feature[:, 2:5])

                whole_feature = np2ts(whole_feature).to(device)
                feature_scaler.fit(whole_feature)
                feature_scaler.save_parameters(None, new_feature_standard_path)

                feature_train = feature_df2feature_tensor(df_train_f, device, feature_scaler)
                feature_valid = feature_df2feature_tensor(df_valid_f, device, feature_scaler)

                performance_train = performance_df2performance_tensor(df_train_p, device, performance_scaler)
                performance_valid = performance_df2performance_tensor(df_valid_p, device, performance_scaler)

                performance_valid_raw = performance_scaler.inverse_transform(performance_valid)
                performance_valid_raw[:, 1:] = torch.pow(10, performance_valid_raw[:, 1:])

                dataset = CustomDataset(feature_train, performance_train)
                dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

                print('开始训练模型')
                model = Direct_Predict_MLP(dipredict_layer_sizes)
                model.to(device)

                optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

                dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid,
                                performance_valid_raw,
                                performance_scaler, args, new_model_save_path, None, device)

                # 测试新模型的预测性能
                df_test_f, df_test_p = read_data_new(real_data_test_df)
                df_test_f = df_test_f.drop(['FileName'], axis=1)

                feature_test = feature_df2feature_tensor(df_test_f, device,
                                                         feature_scaler)  # 所以这里labeled_feature并不是前面的feature_train
                performance_test_raw = df2np(df_test_p)
                performance_test_raw = np2ts(performance_test_raw).to(device)

                model.eval()
                with torch.no_grad():
                    predicted_performances = model(feature_test)
                    predicted_performances = performance_scaler.inverse_transform(predicted_performances)
                    predicted_performances[:, 1:] = torch.pow(10, predicted_performances[:, 1:])

                    mean_errors, mean_errors_percent, _ = calculate_errors(performance_test_raw,
                                                                           predicted_performances)  # 这里误差是一个3维张量，是所有验证样本的平均
                    print('预测误差为：')
                    print(f'mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}')

                    test_error_dic['rec_MAE'].append(mean_errors[0].item())
                    test_error_dic['ct_dc_MAE'].append(mean_errors[1].item())
                    test_error_dic['st_dc_MAE'].append(mean_errors[2].item())
                    test_error_dic['rec_MAPE'].append(mean_errors_percent[0].item())
                    test_error_dic['ct_dc_MAPE'].append(mean_errors_percent[1].item())
                    test_error_dic['st_dc_MAPE'].append(mean_errors_percent[2].item())

                print('模型训练完毕，重新检测')
                # 切记，数据相似检测是用的是所有标注组合的特征数据，而不是用标注的训练数据
                t1 = time.time()
                detected_labeled_feature = feature_df2feature_tensor(current_detected_df_feature_labeled, device,
                                                                     feature_scaler)  # 所以这里labeled_feature并不是前面的feature_train
                detected_labeled_feature_vectors = get_feature_vectors(model, detected_labeled_feature)

                selected_labeled_feature = feature_df2feature_tensor(current_selected_df_feature_labeled, device,
                                                                     feature_scaler)
                selected_labeled_feature_vectors = get_feature_vectors(model, selected_labeled_feature)

                min_distance = get_nn_dist(detected_labeled_feature_vectors, detected_labeled_feature_vectors, 2)
                distance_threshold1 = np.percentile(min_distance, 95)  # 更新距离阈值
                distance_threshold2 = np.percentile(min_distance, 96)
                distance_threshold3 = np.percentile(min_distance, 98)
                distance_threshold4 = np.percentile(min_distance, 99)

                df_feature_unlabeled = get_input_feature(df_data_feature_test, current_df_config_unlabeled)

                df_feature_query = df_feature_unlabeled[
                    ['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist',
                     'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio',
                     'q_K_StdRatio']]

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
        # '''








































