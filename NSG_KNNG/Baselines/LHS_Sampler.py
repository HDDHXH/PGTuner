import os
from pathlib import Path
import subprocess
from scipy.stats import qmc
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import time
import struct

'''
实现3种LSH：
LSH1：以efC和M为基本单位采样，在网格搜索的配置中采样10%（即24个组合），采样的是网格搜索配置中的索引
LSH2：以efC、M和efS为基本单位采样，在网格搜索的配置中采样10%（即1530个组合），采样的是网格搜索配置中的索引
LSH3：以efC、M和efS为基本单位，在整个配置空间中采样1530个组合，采样的是具体的数值
'''

def LHS_sample1(dimension, num_points, seed, bounds):
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    samples = sampler.random(n=num_points)

    # 手动调整每个维度的样本到指定范围
    lower, upper = bounds

    scaled_samples = np.zeros_like(samples)
    
    scaled_samples = lower + (upper - lower) * samples
    scaled_samples = np.floor(scaled_samples + 0.5)  # 要4舍5入转成整数

    scaled_samples = scaled_samples.reshape((-1)).tolist()

    return scaled_samples

def LHS_sample2(dimension, num_points, seed, bounds):
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    samples = sampler.random(n=num_points)

    # 手动调整每个维度的样本到指定范围
    scaled_samples = np.zeros_like(samples)
    for i, (lower, upper) in enumerate(bounds):
        scaled_samples[:, i] = lower + (upper - lower) * samples[:, i]

    scaled_samples = np.floor(scaled_samples + 0.5)  # 要4舍5入转成整数

    scaled_samples[:, 1] = np.where(scaled_samples[:, 1] < scaled_samples[:, 0], scaled_samples[:, 0], scaled_samples[:, 1])

    return scaled_samples

def LHS_sample3(dimension, num_points, seed, bounds):
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    samples = sampler.random(n=num_points)

    # 手动调整每个维度的样本到指定范围
    scaled_samples = np.zeros_like(samples)
    for i, (lower, upper) in enumerate(bounds):
        scaled_samples[:, i] = lower + (upper - lower) * samples[:, i]

    scaled_samples = np.floor(scaled_samples + 0.5)  # 要4舍5入转成整数

    scaled_samples[:, 0] = np.where(scaled_samples[:, 0] < scaled_samples[:, 1], scaled_samples[:, 1], scaled_samples[:, 0])

    return scaled_samples

def read_ivecs(file_path):
    indices = []
    with open(file_path, 'rb') as f:
        while True:
            k_bytes = f.read(4)
            if not k_bytes:
                break
            k, = struct.unpack('I', k_bytes)
            vector_bytes = f.read(k * 4)  # For ivecs, each dimension is an int (4 bytes)
            indice = np.frombuffer(vector_bytes, dtype=np.int32)
            indices.append(indice)
    return np.array(indices)


def calculate_recall_rate(gt, qs):
    recall_rates = []

    K = qs.shape[1]
    gt = gt[:, :K]

    for row1, row2 in zip(gt, qs):
        recall_count = sum(elem in row1 for elem in row2)
        # 计算召回率：在A中找到的元素数 / B行中的元素总数
        recall_rate = recall_count / K
        recall_rates.append(recall_rate)

    # 计算所有行的平均召回率
    average_recall = np.mean(recall_rates)
    return average_recall

if __name__ == '__main__':
    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录
    three_level_directory = Path.cwd().parents[1]


    selected_config_path_GMM = "../Data/LHS_configs/selected_config_GMM.csv"
    selected_config_path_random1 = "../Data/LHS_configs/selected_config_random1.csv"
    selected_config_path_LSH3 = "../Data/LHS_configs/selected_config_LSH3.csv"

    # KNN_config_unit_data_path = "../Data/KNN_config_unit_data.csv"
    # NSG_config_unit_data_path = "../Data/NSG_config_unit_data.csv"
    #
    # df_KNN_config = pd.read_csv(KNN_config_unit_data_path, sep=',', header=0)
    # df_config_unit = pd.read_csv(NSG_config_unit_data_path, sep=',', header=0)
    #
    # df_config_whole = pd.merge(df_KNN_config, df_config_unit, how='cross')

    seed = 42

    # dimension1 = 1
    # num_points1 = 192
    # bounds1 = (0, 3519)

    dimension2 = 5
    num_points2 = 192
    bounds2 = [(100, 400), (100, 400), (150, 350), (5, 90), (300, 600)]

    # dimension3 = 3
    # num_points3 = 50
    # bounds3 = [(20, 800), (4, 100), (10, 5000)]
    #
    # selected_samples1 = LHS_sample1(dimension1, num_points1, seed, bounds1)
    # df_selected_config_LSH1 = df_config_whole.iloc[selected_samples1]
    # df_selected_config_LSH1.to_csv(selected_config_path_GMM, index=False, mode='w', header=True)
    #
    #
    # selected_samples2 = LHS_sample2(dimension2, num_points2, seed, bounds2)
    # df_selected_config_LSH2 = pd.DataFrame(selected_samples2, columns=[ 'K', 'L', 'L_nsg_C', 'R_nsg', 'C'])
    # df_selected_config_LSH2.to_csv(selected_config_path_random1, index=False, mode='w', header=True)

    
    selected_samples_list = []

    target_rec_lis = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]

    filename_dic = {'deep1': '0_1_96_1', 'paper': '0_2_200_1', 'gist': '0_1_960_1'}

    ite = 12
    S = 15
    R = 100

    # NSG搜索参数
    L_nsg_Ss = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
                430, 460, 490, 520, 550, 580, 610, 640, 670, 700, 740, 780, 820, 860, 900, 960, 1020, 1080, 1140, 1200, 1300, 1400, 1500]

    base_dir = os.path.join(three_level_directory, 'Data/Base')
    query_dir =  os.path.join(three_level_directory, 'Data/Query')
    groundtruth_dir = os.path.join(three_level_directory, 'Data/GroundTruth')
    KNN_graph_dir = "../KNN_graph"
    NSG_graph_dir = "../NSG_graph"

    dataset_name = 'gist'

    subdir = re.match(r'\D+', dataset_name).group()
    filename = filename_dic[dataset_name]

    whole_name = subdir + '_' + filename

    subdir_path = os.path.join(base_dir, subdir)

    filename_list = filename.split('_')
    level = int(filename_list[0])
    num = float(filename_list[1])
    dim = int(filename_list[2])

    size = int(pow(10, level) * 100000 * num)

    base_path = os.path.join(subdir_path, filename+'.fvecs')
    query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, dim))
    gt_indice_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename))
#'''
    for method in ['random1']:
        index_performance_csv1 = "../Data/index_performance_KNNG_{}.csv".format(method)  # 记录每个数据集在不同参数下构建KNNG的时间、距离计算次数、占用内存等信息
        index_performance_csv2 = "../Data//index_performance_NSG_{}.csv".format(method)  # 记录每个数据集在固定KNNG和在不同参数下构建NSG的时间、距离计算次数、占用内存等信息
        index_performance_csv3 = "../Data/index_performance_Search_{}.csv".format(method)  # 记录每个数据集在固定NSG和不同参数下搜索的时间、距离计算次数、占用内存等信息
        index_performance_csv4 = "../Data/index_performance_Search_Recall_{}.csv".format(method)
        whole_index_performance_csv = '../Data/index_performance_{}.csv'.format(method)

        # index_csv =  os.path.join(parent_directory, 'Data/index_performance_{}.csv'.format(method))

        # exist_index_performance_csv = os.path.join(parent_directory, 'Data/index_performance_{}.csv'.format(method))
        #
        # exist_df = pd.read_csv(exist_index_performance_csv)
        # exist_df = exist_df[exist_df['FileName'] == filename]

        if method == 'random1':
            # exist_paras = exist_df[['efConstruction', 'M']].to_numpy().tolist()
            #'''
            selected_samples = LHS_sample2(dimension2, num_points2, seed, bounds2)
            # print(selected_samples)
            selected_samples = selected_samples.tolist()

            for para in tqdm(selected_samples, total = len(selected_samples)):
                K, L, L_nsg_C, R_nsg, C = para
                K = int(K)
                L = int(L)
                L_nsg_C = int(L_nsg_C)
                R_nsg = int(R_nsg)
                C = int(C)

                KNN_graph_path = os.path.join(KNN_graph_dir, '{}/{}_{}_{}.graph'.format(subdir, filename, K, L))
                NSG_graph_path = os.path.join(NSG_graph_dir,  '{}/{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C))

                print(f'-------------------开始构建KNNG: {K}_{L}，只需构建一次-------------------')
                if not os.path.exists(KNN_graph_path):
                    t1 = time.time()
                    run_command1 = ['../FFANNA_KNNG/build/tests/test_nndescent', base_path, KNN_graph_path, str(K), str(L), str(ite), str(S), str(R), whole_name, index_performance_csv1]
                    result1 = subprocess.run(run_command1, check=True, text=True, capture_output=True)
                    t2 = time.time()

                    KNN_graph_time = t2 - t1

                print(f'-------------------开始构建NSG: {L_nsg_C}_{R_nsg}_{C}-------------------')
                t3 = time.time()
                run_command2 = ['../NSG/build/tests/test_nsg_index', base_path, KNN_graph_path, str(L_nsg_C), str(R_nsg), str(C), NSG_graph_path, str(K), str(L), whole_name, index_performance_csv2]
                result2 = subprocess.run(run_command2, check=True, text=True, capture_output=True)
                t4 = time.time()

                NSG_graph_time = t4 - t3

                print('-------------------开始搜索-------------------')
                recs = []
                for L_nsg_S in L_nsg_Ss:
                    qs_indice_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C, L_nsg_S))

                    run_command3 = ['../NSG/build/tests/test_nsg_optimized_search', base_path, query_path, NSG_graph_path, str(L_nsg_S), str(10), qs_indice_path, str(K),
                                    str(L), str(L_nsg_C), str(R_nsg), str(C), whole_name, index_performance_csv3]
                    result3 = subprocess.run(run_command3, check=True, text=True, capture_output=True)

                    gt = read_ivecs(gt_indice_path)
                    qs = read_ivecs(qs_indice_path)

                    rec = calculate_recall_rate(gt, qs)
                    recs.append((whole_name, K, L, L_nsg_C, R_nsg, C, L_nsg_S, rec))

                    os.remove(qs_indice_path) #用完了就删除

                    if L_nsg_S >= 300 and rec >= 0.995:  # L_nsg_S >= 500 这个得测试一下
                        break

                rec_df = pd.DataFrame(recs, columns=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'recall'])
                rec_df.to_csv(index_performance_csv4, mode='a', header=False, index=False)

                if NSG_graph_time < 1800:
                    os.remove(NSG_graph_path)  # NSG构建时间较短就删除
            #'''
        df_KNNG = pd.read_csv(index_performance_csv1, sep=',', header=0)
        df_NSG = pd.read_csv(index_performance_csv2, sep=',', header=0)
        df_Search = pd.read_csv(index_performance_csv3, sep=',', header=0)
        df_Search_Recall = pd.read_csv(index_performance_csv4, sep=',', header=0)

        df_merged = pd.merge(df_KNNG, df_NSG, on=['FileName', 'K', 'L'], how='left')

        # 合并上一步的结果与 C
        df_merged = pd.merge(df_merged, df_Search, on=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C'], how='left')

        # 最后将得到的结果与 D 合并
        df_merged = pd.merge(df_merged, df_Search_Recall, on=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S'], how='left')

        df_merged.to_csv(whole_index_performance_csv, mode='w', header=True, index=False)
#'''









