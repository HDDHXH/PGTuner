import os
import subprocess
import pandas as pd
import struct
import numpy as np
from tqdm import tqdm
import time
import random
import re

#这个代码用于接收drl推荐的参数，然后构建索引并执行查询，在推荐爱你的efS的基础上搜索能达到指定召回率的最小efS
s = 20
np.random.seed(s)
random.seed(s)
# 编译cpp文件
compile_command = ['g++', '-Ofast', '-lrt', '-std=c++11', '-DHAVE_CXX0X', '-march=native', '-fpic', '-w',
                   '-fopenmp', '-ftree-vectorize', '-ftree-vectorizer-verbose=0', 'index_construct_verify.cpp', '-o',
                   'index_construct_verify'
                   ]
subprocess.run(compile_command, check=True)
print('编译完成')

filename_dic = {'deep1':'1_1_96_1', 'sift1':'1_1_128_1', 'glove': '1_1.183514_100', 'paper':'1_2.029997_200', 'crawl':'1_1.989995_300', 'msong':'-1_1_420_1',
                    'gist':'1_1.0_960', 'deep10':'2_1_96', 'sift50':'2_5_128_1', 'deep2':'1_2_96_1', 'deep3':'1_3_96_1', 'deep4':'1_4_96_1', 'deep5':'1_5_96_1',
                    'sift2':'1_2_128_1', 'sift3':'1_3_128_1', 'sift4':'1_4_128_1', 'sift5':'1_5_128_1', 'gist_25':'1_1.0_960_25', 'gist_50':'1_1.0_960_50',
                    'gist_75':'1_1.0_960_75', 'gist_100':'1_1.0_960_100', 'deep2_25':'1_2_96_1_25', 'deep2_50':'1_2_96_1_50', 'deep2_75':'1_2_96_1_75', 'deep2_100':'1_2_96_1_100'}

para_dic_test_main = {'deep10': [[0.9, 260, 40, 30], [0.92, 420, 16, 70], [0.95, 420, 24, 70], [0.98, 620, 32, 100], [0.99, 800, 48, 120]], 'gist': [[0.85, 500, 48, 45], [0.88, 800, 20, 100], [0.9, 800, 24, 100], [0.92, 500, 48, 80], [0.94, 620, 56, 90], [0.95, 800, 40, 120], [0.96, 680, 40, 140], [0.98, 800, 32, 240], [0.99, 800, 56, 260]]}

para_dic_test_main_ep8000_mt5000_128_onlys = {'deep10': [[0.9, 800.0, 14.0, 49.0], [0.92, 800.0, 15.0, 54.0], [0.95, 799.0, 14.0, 91.0], [0.98, 800.0, 25.0, 128.0], [0.99, 640.0, 31.0, 150.0]]}

para_dic_test_main_ep8000_mt5000_128_onlys_new = {'deep10': [[0.9, 800.0, 14.0, 51.0], [0.92, 795.0, 17.0, 50.0], [0.95, 773.0, 20.0, 68.0], [0.98, 800.0, 23.0, 140.0], [0.99, 742.0, 36.0, 139.0]], 'gist': [[0.85, 800.0, 65.0, 38.0], [0.88, 791.0, 75.0, 49.0], [0.9, 800.0, 78.0, 59.0], [0.92, 800.0, 79.0, 73.0], [0.94, 800.0, 58.0, 102.0], [0.95, 800.0, 40.0, 151.0], [0.96, 800.0, 38.0, 174.0], [0.98, 800.0, 37.0, 273.0], [0.99, 631.0, 45.0, 309.0]], 'glove': [[0.85, 800.0, 57.0, 78.0], [0.88, 800.0, 59.0, 101.0], [0.9, 795.0, 62.0, 120.0], [0.92, 800.0, 60.0, 147.0], [0.94, 800.0, 55.0, 210.0], [0.95, 800.0, 54.0, 255.0], [0.96, 800.0, 59.0, 301.0], [0.98, 800.0, 63.0, 458.0], [0.99, 800.0, 65.0, 578.0]]}

para_dic_test_main_ep4800_mt5000_128_onlys_pec_reward10 = {'deep10': [[0.9, 800.0, 14.0, 49.0], [0.92, 800.0, 14.0, 58.0], [0.95, 800.0, 18.0, 70.0], [0.98, 457.0, 28.0, 119.0], [0.99, 391.0, 33.0, 144.0]], 'gist': [[0.85, 762.0, 38.0, 59.0], [0.88, 239.0, 25.0, 117.0], [0.9, 164.0, 33.0, 123.0], [0.92, 236.0, 31.0, 141.0], [0.94, 248.0, 30.0, 169.0], [0.95, 244.0, 32.0, 180.0], [0.96, 246.0, 34.0, 191.0], [0.98, 231.0, 42.0, 249.0], [0.99, 412.0, 46.0, 270.0]], 'glove': [[0.85, 800.0, 47.0, 85.0], [0.88, 800.0, 51.0, 105.0], [0.9, 794.0, 54.0, 121.0], [0.92, 800.0, 71.0, 143.0], [0.94, 800.0, 61.0, 206.0], [0.95, 800.0, 64.0, 246.0], [0.96, 800.0, 65.0, 297.0], [0.98, 800.0, 76.0, 449.0], [0.99, 800.0, 79.0, 561.0]]}

para_dic_test_main_ep4800_mt5000_128_onlys_pec_reward10_new = {'deep10': [[0.95, 783.0, 17.0, 72.0], [0.98, 425.0, 23.0, 121.0], [0.99, 418.0, 29.0, 151.0]], 'gist':[[0.85, 793.0, 34.0, 63.0], [0.88, 794.0, 38.0, 78.0], [0.9, 775.0, 41.0, 91.0], [0.92, 485.0, 26.0, 146.0], [0.94, 440.0, 28.0, 165.0], [0.95, 431.0, 30.0, 175.0], [0.96, 395.0, 33.0, 186.0], [0.98, 349.0, 40.0, 242.0], [0.99, 407.0, 49.0, 261.0]], 'glove': [[0.85, 739.0, 58.0, 81.0], [0.88, 604.0, 44.0, 127.0], [0.9, 705.0, 71.0, 125.0], [0.92, 800.0, 52.0, 156.0], [0.94, 800.0, 99.0, 187.0], [0.95, 800.0, 99.0, 224.0], [0.96, 800.0, 100.0, 274.0], [0.98, 800.0, 99.0, 430.0], [0.99, 799.0, 98.0, 544.0]]}

para_dic_test_main_ep8000_mt5000_128_onlys_alone = {'deep10': [[0.9, 800.0, 14.0, 64.0], [0.92, 792.0, 14.0, 74.0], [0.95, 794.0, 19.0, 76.0], [0.98, 785.0, 29.0, 106.0], [0.99, 793.0, 38.0, 146.0]], 'gist': [[0.85, 800.0, 25.0, 77.0], [0.88, 798.0, 32.0, 87.0], [0.9, 794.0, 36.0, 99.0], [0.92, 792.0, 39.0, 115.0], [0.94, 797.0, 37.0, 141.0], [0.95, 787.0, 37.0, 158.0], [0.96, 800.0, 35.0, 182.0], [0.98, 800.0, 39.0, 265.0], [0.99, 626.0, 45.0, 312.0]], 'glove': [[0.85, 799.0, 60.0, 78.0], [0.88, 792.0, 69.0, 100.0], [0.9, 795.0, 71.0, 119.0], [0.92, 800.0, 68.0, 144.0], [0.94, 797.0, 63.0, 206.0], [0.95, 800.0, 67.0, 244.0], [0.96, 800.0, 67.0, 295.0], [0.98, 800.0, 63.0, 462.0], [0.99, 800.0, 60.0, 587.0]]}

para_dic_test_main_ep4800_mt5000_128_onlys_pec_reward10_new_alone = {'gist': [[0.85, 772.0, 76.0, 36.0], [0.88, 774.0, 68.0, 52.0], [0.9, 794.0, 46.0, 82.0], [0.92, 787.0, 48.0, 97.0], [0.94, 445.0, 45.0, 132.0], [0.95, 622.0, 44.0, 143.0], [0.96, 545.0, 47.0, 154.0], [0.98, 448.0, 56.0, 208.0], [0.99, 265.0, 73.0, 366.0]]}

para_dic_test_main_ep8000_mt5000_128_onlys_alone_sift = {'sift50': [[0.9, 800.0, 25.0, 43.0], [0.95, 800.0, 26.0, 68.0], [0.99, 750.0, 38.0, 171.0]]}

para_dic_test_ds_change_ep8000_mt5000_128_onlys_alone = {'sift3': [[0.95, 800.0, 19.0, 52.0]], 'sift4': [[0.95, 800.0, 19.0, 51.0]], 'sift5': [[0.95, 784.0, 20.0, 53.0]]}

para_dic_test_dast_change_order1 = {'glove': [[0.9, 800.0, 85.0, 104.0], [0.95, 800.0, 100.0, 180.0], [0.99, 800.0, 100.0, 459.0]],
                                    'gist': [[0.9, 800.0, 21.0, 113.0], [0.95, 788.0, 40.0, 152.0], [0.99, 795.0, 71.0, 328.0]],
                                    'deep10': [[0.9, 785.0, 15.0, 54.0], [0.95, 790.0, 21.0, 66.0], [0.99, 800.0, 33.0, 134.0]]}

para_dic_test_dast_change_order2 = {'glove': [[0.9, 800.0, 90.0, 126.0], [0.95, 798.0, 100.0, 299.0], [0.99, 800.0, 100.0, 796.0]],
                                    'gist': [[0.9, 800.0, 51.0, 60.0], [0.95, 800.0, 40.0, 135.0], [0.99, 800.0, 73.0, 335.0]],
                                    'sift50': [[0.9, 800.0, 26.0, 49.0], [0.95, 796.0, 24.0, 96.0], [0.99, 721.0, 36.0, 109.0]]}

para_dic_test_dast_change_order3 = {'glove': [[0.9, 800.0, 62.0, 122.0], [0.95, 797.0, 74.0, 315.0], [0.99, 800.0, 92.0, 841.0]],
                                    'deep10': [[0.9, 771.0, 19.0, 49.0], [0.95, 800.0, 17.0, 97.0], [0.99, 800.0, 31.0, 118.0]],
                                    'sift50': [[0.9, 800.0, 17.0, 71.0], [0.95, 784.0, 23.0, 79.0], [0.99, 794.0, 38.0, 151.0]]}

para_dic_test_dast_change_order4 = {'gist': [[0.9, 800.0, 21.0, 119.0], [0.95, 795.0, 39.0, 166.0], [0.99, 797.0, 78.0, 380.0]],
                                    'deep10': [[0.9, 800.0, 17.0, 57.0], [0.95, 795.0, 21.0, 86.0], [0.99, 800.0, 32.0, 118.0]],
                                    'sift50': [[0.9, 800.0, 19.0, 59.0], [0.95, 800.0, 22.0, 96.0], [0.99, 680.0, 29.0, 174.0]]}

base_dir = "./Data/Base"
query_dir = "./Data/Query"
groundtruth_dir = "./Data/GroundTruth"
index_dir = "./Index"
index_csv = "./Data/experiments_results/test_ds_change/index_performance_verify_test_ds_change_compare.csv"  # 每个数据集在每个参数配置下构建的索引的相关数据，索引构建时间、内存占用、查询召回率、查询时间等数据

all_index_performance_csv = "./Data/experiments_results/test_ds_change/index_performance_verify_test_ds_change_compare.csv"

df = pd.read_csv(all_index_performance_csv, sep=',', header=0)
#exist_para = df[['FileName', 'target_recall', 'efConstruction', 'M', 'pr_efSearch']].to_numpy().tolist()
exist_para = []

tasks = []

for para_dic in [para_dic_test_ds_change_ep8000_mt5000_128_onlys_alone]:
    for dataset_name in para_dic.keys():
        para_list = para_dic[dataset_name]

        filename = filename_dic[dataset_name]
        subdir = re.match(r'\D+', dataset_name).group()

        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            if subdir == 'sift':
                file_text = filename + '.bvecs'
            else:
                file_text = filename + '.fvecs'

            whole_name = subdir + '_' + filename

            filename_list = filename.split('_')
            if len(filename_list) == 2:
                print(os.path.join(subdir_path, file_text))
            else:
                level = int(filename_list[0])
                num = float(filename_list[1])
                dim = int(filename_list[2])

                size = int(pow(10, level) * 100000 * num)

                # if  (5e6 <= size < 1e7) and dim in [100] and i == 1:
                if dim in [96, 100, 128, 200, 300, 420, 960]:
                    base_path = os.path.join(subdir_path, file_text)
                    indice_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename))

                    if subdir in ['deep', 'glove', 'sift', 'paper', 'uqv', 'crawl', 'msong', 'gist']:
                        if dim == 128:
                            query_path = os.path.join(query_dir, '{}/{}.bvecs'.format(subdir, dim))
                        else:
                            query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, dim))

                    else:
                        query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, filename))

                    for para in para_list:
                        target_recall, efC, m, efS = para
                        efC = int(efC)
                        m = int(m)
                        efS = int(efS)

                        if [whole_name, target_recall, efC, m, efS] not in exist_para:
                            tasks.append((whole_name, base_path, query_path, indice_path, filename, index_csv, subdir, size, dim, efC, m, efS, target_recall))

print(len(tasks))
for task in tqdm(tasks, total=len(tasks)):
    whole_name, base_path, query_path, indice_path, filename, index_csv, subdir, size, dim, efC, m, efS, target_recall = task
    # print(target_recall)
    # print(str(target_recall))
    index_path = os.path.join('./Index', '{}/{}_{}_{}.bin'.format(subdir, filename, efC, m))

    # 构建运行命令
    run_command = ['./index_construct_verify', whole_name, base_path, query_path, indice_path, index_path, index_csv,
                   subdir, str(size), str(dim), str(efC), str(m), str(efS), str(target_recall)]
    #print(" ".join(run_command))
    print(f'{whole_name}_{target_recall}_{efC}_{m}_{efS}')
    print('-------------------开始构建索引并执行测试-------------------')
    result = subprocess.run(run_command, check=True, text=True, capture_output=True)
    print('-------------------索引构建与测试结束-------------------')













