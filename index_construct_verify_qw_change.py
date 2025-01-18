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
                   '-fopenmp', '-ftree-vectorize', '-ftree-vectorizer-verbose=0', 'index_construct_verify_qw_change.cpp', '-o',
                   'index_construct_verify_qw_change'
                   ]
subprocess.run(compile_command, check=True)
print('编译完成')

filename_dic = {'deep1':'1_1_96_1', 'sift1':'1_1_128_1', 'glove': '1_1.183514_100', 'paper':'1_2.029997_200', 'crawl':'1_1.989995_300', 'msong':'0_9.92272_420',
                    'gist':'1_1.0_960', 'deep10':'2_1_96', 'sift50':'2_5_128_1', 'deep2':'1_2_96_1', 'deep3':'1_3_96_1', 'deep4':'1_4_96_1', 'deep5':'1_5_96_1',
                    'sift2':'1_2_128_1', 'sift3':'1_3_128_1', 'sift4':'1_4_128_1', 'sift5':'1_5_128_1', 'gist_25':'1_1.0_960_25', 'gist_50':'1_1.0_960_50',
                    'gist_75':'1_1.0_960_75', 'gist_100':'1_1.0_960_100', 'deep2_25':'1_2_96_1_25', 'deep2_50':'1_2_96_1_50', 'deep2_75':'1_2_96_1_75', 'deep2_100':'1_2_96_1_100'}

para_dic_test_qw_change_ep8000_mt5000_128_onlys_alone = {'gist': [[0.95, 791.0, 78.0, 217.0], [0.95, 797.0, 100.0, 169.0], [0.95, 800.0, 60.0, 322.0], [0.95, 796.0, 54.0, 296.0]]}

base_dir = "./Data/Base"
query_dir = "./Data/Query"
groundtruth_dir = "./Data/GroundTruth"
index_dir = "./Index"
index_csv = "./Data/experiments_results/test_qw_change/index_performance_verify_test_qw_change_compare.csv"  # 每个数据集在每个参数配置下构建的索引的相关数据，索引构建时间、内存占用、查询召回率、查询时间等数据

all_index_performance_csv = "./Data/experiments_results/test_qw_change/index_performance_verify_test_qw_change_compare.csv"

df = pd.read_csv(all_index_performance_csv, sep=',', header=0)
#exist_para = df[['target_recall', 'efConstruction', 'M', 'pr_efSearch']].to_numpy().tolist()
exist_para = []

tasks = []

for para_dic in [para_dic_test_qw_change_ep8000_mt5000_128_onlys_alone]:
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

                    filename1 = filename + '_25'
                    filename2 = filename + '_50'
                    filename3 = filename + '_75'
                    filename4 = filename + '_100'

                    indice_path1 = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename1))
                    indice_path2 = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename2))
                    indice_path3 = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename3))
                    indice_path4 = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename4))

                    if subdir in ['deep', 'glove', 'sift', 'paper', 'uqv', 'crawl', 'msong', 'gist']:
                        if dim == 128:
                            query_path1 = os.path.join(query_dir, '{}/{}_25.bvecs'.format(subdir, dim))
                            query_path2 = os.path.join(query_dir, '{}/{}_50.bvecs'.format(subdir, dim))
                            query_path3 = os.path.join(query_dir, '{}/{}_75.bvecs'.format(subdir, dim))
                            query_path4 = os.path.join(query_dir, '{}/{}_100.bvecs'.format(subdir, dim))
                        else:
                            query_path1 = os.path.join(query_dir, '{}/{}_25.fvecs'.format(subdir, dim))
                            query_path2 = os.path.join(query_dir, '{}/{}_50.fvecs'.format(subdir, dim))
                            query_path3 = os.path.join(query_dir, '{}/{}_75.fvecs'.format(subdir, dim))
                            query_path4 = os.path.join(query_dir, '{}/{}_100.fvecs'.format(subdir, dim))

                    for para in para_list:
                        target_recall, efC, m, efS = para
                        efC = int(efC)
                        m = int(m)
                        efS = int(efS)

                        if [target_recall, efC, m, efS] not in exist_para:
                            tasks.append((whole_name, base_path, filename, query_path1, query_path2, query_path3, query_path4, indice_path1, indice_path2, indice_path3, indice_path4,
                                          index_csv, subdir, size, dim, efC, m, efS, target_recall))

print(len(tasks))
for task in tqdm(tasks, total=len(tasks)):
    whole_name, base_path, filename, query_path1, query_path2, query_path3, query_path4, indice_path1, indice_path2, indice_path3, indice_path4, index_csv, subdir, size, dim, efC, m, efS, target_recall = task
    # print(target_recall)
    # print(str(target_recall))
    index_path = os.path.join('./Index', '{}/{}_{}_{}.bin'.format(subdir, filename, efC, m))

    # 构建运行命令
    run_command = ['./index_construct_verify_qw_change', whole_name, base_path, query_path1, query_path2, query_path3, query_path4, indice_path1, indice_path2, indice_path3, indice_path4,
                   index_path, index_csv, subdir, str(size), str(dim), str(efC), str(m), str(efS), str(target_recall)]
    #print(" ".join(run_command))
    print(f'{whole_name}_{target_recall}_{efC}_{m}_{efS}')
    print('-------------------开始构建索引并执行测试-------------------')
    result = subprocess.run(run_command, check=True, text=True, capture_output=True)
    print('-------------------索引构建与测试结束-------------------')













