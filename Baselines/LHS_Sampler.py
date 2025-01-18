import os
import subprocess
from scipy.stats import qmc
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

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

    scaled_samples[:, 0] = np.where(scaled_samples[:, 0] < scaled_samples[:, 1], scaled_samples[:, 1], scaled_samples[:, 0])

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

def get_whole_config(df_config_unit):
    efSs = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 300, 340, 380, 420, 460, 500, 540, 580, 620, 660, 700, 760, 820, 880, 940, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000])

    df_config_whole = df_config_unit.loc[df_config_unit.index.repeat(len(efSs))].reset_index(drop=True)
    df_config_whole['efSearch'] = np.tile(efSs, len(df_config_unit))

    return df_config_whole

def get_performance_data_random1(exist_paras, exist_df, efC, m, subdir, filename, base_path, query_path, indice_path, index_csv, size, dim):
    whole_name = subdir + '_' + filename

    if [efC, m] in exist_paras:  #如果这个参数的性能数据已经有了，就直接从文件中获取
        config_df = pd.DataFrame([(efC, m)], columns=['efConstruction', 'M'])

        config_data_df = pd.merge(exist_df, config_df, on=['efConstruction', 'M'], how='right')
        config_data_df.to_csv(index_csv, mode='a', index=False, header = False)


    else: #没有的话再构建索引跑数据
        index_path = os.path.join('../Index', '{}/{}_{}.bin'.format(subdir, filename, efC, m))

        # 构建运行命令
        run_command = ['../index_construct_test_new', whole_name, base_path, query_path, indice_path, index_path, index_csv, subdir, str(size), str(dim), str(efC), str(m)]
        #print(" ".join(run_command))
        print(f'{whole_name}_{efC}_{m}')
        print('-------------------开始构建索引并执行测试-------------------')
        result = subprocess.run(run_command, check=True, text=True, capture_output=True)
        print('-------------------索引构建与测试结束-------------------')

def get_performance_data_random2(exist_paras, exist_df, efC, m, efS, target_recall, subdir, filename, base_path, query_path, indice_path, index_csv, size, dim):
    whole_name = subdir + '_' + filename

    if [efC, m, efS] in exist_paras:  #如果这个参数的性能数据已经有了，就直接从文件中获取
        config_df = pd.DataFrame([(efC, m, efS)], columns=['efConstruction', 'M', 'efSearch'])

        config_data_df = pd.merge(exist_df, config_df, on=['efConstruction', 'M', 'efSearch'], how='right')

        config_data_df['target_recall'] = target_recall
        config_data_df.to_csv(index_csv, mode='a', index=False, header = False)


    else: #没有的话再构建索引跑数据
        index_path = os.path.join('../Index', '{}/{}_{}_{}.bin'.format(subdir, filename, efC, m, efS))

        # 构建运行命令
        run_command = ['../index_construct_VDTuner', whole_name, base_path, query_path, indice_path, index_path, index_csv,
                    subdir, str(size), str(dim), str(efC), str(m), str(efS), str(target_recall)]
        #print(" ".join(run_command))
        print(f'{whole_name}_{target_recall}_{efC}_{m}_{efS}')
        print('-------------------开始构建索引并执行测试-------------------')
        result = subprocess.run(run_command, check=True, text=True, capture_output=True)
        print('-------------------索引构建与测试结束-------------------')


if __name__ == '__main__':
    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

    selected_config_path_LSH1 = "../Data/LSH_configs/selected_config_LSH1.csv"
    selected_config_sift_path_LSH1 = "../Data/LSH_configs/selected_config_sift_LSH1.csv"
    selected_config_path_LSH2 = "../Data/LSH_configs/selected_config_LSH2.csv"
    selected_config_path_LSH3 = "../Data/LSH_configs/selected_config_LSH3.csv"

    # df_config_unit = pd.read_csv(config_unit_data_path, sep=',', header=0)
    # df_config_unit_sift = pd.read_csv(config_unit_data_sift_path, sep=',', header=0)

    # df_config_whole = get_whole_config(df_config_unit)

    seed = 42

    # dimension1 = 1
    # num_points1 = 14
    # bounds1 = (0, 242)
    # bounds_sift1 = (0, 165)

    dimension2 = 2
    num_points2 = 14
    bounds2 = [(20, 800), (4, 100)]

    dimension3 = 3
    num_points3 = 50
    bounds3 = [(20, 800), (4, 100), (10, 5000)]

    # selected_samples1 = LHS_sample1(dimension1, num_points1, seed, bounds1)
    # selected_samples_sift1 = LHS_sample1(dimension1, num_points1, seed, bounds_sift1)
    
    # df_selected_config_LSH1 = df_config_unit.iloc[selected_samples1] 
    # df_selected_config_sift_LSH1 = df_config_unit_sift.iloc[selected_samples_sift1] 

    # df_selected_config_LSH1.to_csv(selected_config_path_LSH1, index=False, mode='w', header=True)
    # df_selected_config_sift_LSH1.to_csv(selected_config_sift_path_LSH1, index=False, mode='w', header=True)

    # selected_samples2 = LHS_sample2(dimension2, num_points2, seed, bounds2)
    # df_selected_config_LSH2 = pd.DataFrame(selected_samples2, columns=['efConstruction', 'M'])
    # df_selected_config_LSH2.to_csv(selected_config_path_LSH2, index=False, mode='w', header=True)

    
    selected_samples_list = []
    # for seed in [21, 42, 88, 121, 360, 520, 666, 888, 999]:
    #     selected_samples = LHS_sample3(dimension3, num_points3, seed, bounds3)
    #     selected_samples_list.append(selected_samples)
    
    # selected_samples3 = np.vstack(selected_samples_list)
    
    # df_selected_config_LSH3 = pd.DataFrame(selected_samples3, columns=['efConstruction', 'M', 'efSearch'])
    # df_selected_config_LSH3.to_csv(selected_config_path_LSH3, index=False, mode='w', header=True)



     # 编译cpp文件
    # compile_command = ['g++', '-Ofast', '-lrt', '-std=c++11', '-DHAVE_CXX0X', '-march=native', '-fpic', '-w',
    #                 '-fopenmp', '-ftree-vectorize', '-ftree-vectorizer-verbose=0', '../index_construct_VDTuner.cpp', '-o',
    #                 '../index_construct_VDTuner'
    #                 ]
    # subprocess.run(compile_command, check=True)
    # print('编译完成')

    target_rec_lis = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]

    filename_dic = {'deep1': '1_1_96_1', 'sift1': '1_1_128_1', 'glove': '1_1.183514_100', 'paper': '1_2.029997_200',
                    'crawl': '1_1.989995_300', 'msong': '0_9.92272_420',
                    'gist': '1_1.0_960', 'deep10': '2_1_96', 'sift50': '2_5_128_1', 'deep2': '1_2_96_1',
                    'deep3': '1_3_96_1', 'deep4': '1_4_96_1', 'deep5': '1_5_96_1',
                    'sift2': '1_2_128_1', 'sift3': '1_3_128_1', 'sift4': '1_4_128_1', 'sift5': '1_5_128_1',
                    'gist_25': '1_1.0_960_25', 'gist_50': '1_1.0_960_50',
                    'gist_75': '1_1.0_960_75', 'gist_100': '1_1.0_960_100', 'deep2_25': '1_2_96_1_25',
                    'deep2_50': '1_2_96_1_50', 'deep2_75': '1_2_96_1_75', 'deep2_100': '1_2_96_1_100'}

    base_dir = os.path.join(parent_directory, 'Data/Base')
    query_dir =  os.path.join(parent_directory, 'Data/Query')
    groundtruth_dir = os.path.join(parent_directory, 'Data/GroundTruth')
    index_dir = os.path.join(parent_directory, 'Index')

    # dataset_name = 'sift2'
    for dataset_name in ['sift3', 'sift4', 'sift5']:
        subdir = re.match(r'\D+', dataset_name).group()
        filename = filename_dic[dataset_name]

        subdir_path = os.path.join(base_dir, subdir)

        filename_list = filename.split('_')
        level = int(filename_list[0])
        num = float(filename_list[1])
        dim = int(filename_list[2])

        size = int(pow(10, level) * 100000 * num)

        if subdir in ['sift']:
            base_path = os.path.join(subdir_path, filename+'.bvecs')
            query_path = os.path.join(query_dir, '{}/{}.bvecs'.format(subdir, dim))
        else:
            base_path = os.path.join(subdir_path, filename+'.fvecs')
            query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, dim))


        indice_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename))

        for method in ['random1']:
            index_csv =  os.path.join(parent_directory, 'Data/index_performance_{}.csv'.format(method))

            exist_index_performance_csv = os.path.join(parent_directory, 'Data/index_performance_{}.csv'.format(method))

            exist_df = pd.read_csv(exist_index_performance_csv)
            exist_df = exist_df[exist_df['FileName'] == filename]

            if method == 'random1':
                exist_paras = exist_df[['efConstruction', 'M']].to_numpy().tolist()

                selected_samples = LHS_sample2(dimension2, num_points2, seed, bounds2)
                selected_samples = selected_samples.tolist()
                print(selected_samples)

                # for para in tqdm(selected_samples, total = len(selected_samples)):
                #     efC, M = para
                #     efC = int(efC)
                #     M = int(M)
                #
                #     get_performance_data_random1(exist_paras, exist_df, efC, M, subdir, filename, base_path, query_path, indice_path, index_csv, size, dim)

            else:
                exist_paras = exist_df[['efConstruction', 'M', 'efSearch']].to_numpy().tolist()

                seed_list = [21, 42, 88, 121, 360, 520, 666, 888, 999]

                for i in tqdm(range(len(target_rec_lis)), total = len(target_rec_lis)):
                    target_recall = target_rec_lis[i]
                    temp_seed = seed_list[i]

                    selected_samples = LHS_sample3(dimension3, num_points3, temp_seed, bounds3)
                    selected_samples = selected_samples.tolist()

                    for para in tqdm(selected_samples, total = len(selected_samples)):
                        efC, M, efS = para
                        efC = int(efC)
                        M = int(M)
                        efS = int(efS)

                        get_performance_data_random2(exist_paras, exist_df, efC, M, efS, target_recall, subdir, filename, base_path, query_path, indice_path, index_csv, size, dim)










