import os
import subprocess
import pandas as pd
import struct
import numpy as np
from tqdm import tqdm
import time
import random

s = 20
np.random.seed(s)
random.seed(s)
# 编译cpp文件
compile_command = ['g++', '-Ofast', '-lrt', '-std=c++11', '-DHAVE_CXX0X', '-march=native', '-fpic', '-w',
                   '-fopenmp', '-ftree-vectorize', '-ftree-vectorizer-verbose=0', 'index_construct_test_new.cpp', '-o',
                   'index_construct_test_new'
                   ]
subprocess.run(compile_command, check=True)
print('编译完成')

# Ks = [10]

# ef = 50

# efCs = [20, 40, 60, 80, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 560, 620, 680, 740, 800]
# ms = [4, 8, 16, 24, 32, 48, 64, 80, 100]
efCs = [20, 40, 60, 80, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 560, 620, 680, 740, 800]
ms = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 100]
# ms = [4, 8, 16, 24, 32, 48, 64, 80]
# efCs = [740, 800]
# ms = [32, 48]

# para_list = []
# for efC in efCs:
#     for m in ms:
#         if m <= efC:
#             para_list.append((efC, m))
#         else:
#             break
# print(len(para_list))
# para_list = []
tasks = []

base_dir = "./Data/Base"
query_dir = "./Data/Query"
groundtruth_dir = "./Data/GroundTruth"
index_dir = "./Index"
index_csv = "./Data/index_performance_ds_change_sift2.csv"  # 每个数据集在每个参数配置下构建的索引的相关数据，索引构建时间、内存占用、查询召回率、查询时间等数据

all_index_performance_csv = './Data/index_performance_ds_change_sift2.csv'

df = pd.read_csv(all_index_performance_csv, sep=',', header=0)
exist_para = df[['FileName', 'efConstruction', 'M']].to_numpy().tolist()

# construct_paras = []

# ['glove', 'sift', 'paper', 'crawl', 'msong', 'gist']
for subdir in ['sift']:
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path):
        if os.listdir(subdir_path):  # 检查子文件夹是否为空
            # 遍历子文件夹中的文件
            for file_text in os.listdir(subdir_path):
                filename = os.path.splitext(file_text)[0]  # 获取文件名

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
                    if 2e6 <= size < 3e6 and dim in [96, 100, 128, 200, 300, 420, 960] and len(filename_list) == 4:
                        # if  (1e6 <= size < 2e6 or 4e6 <= size < 5e6) and dim in [100, 200, 300] and i in [2, 3]:
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
                            efC, m = para

                            # construct_paras.append([whole_name, efC, m])
                            # tasks.append((whole_name, base_path, query_path, indice_path, filename, index_csv, subdir,
                            #               size, dim, efC, m))
                            if [whole_name, efC, m] not in exist_para:
                                tasks.append((whole_name, base_path, query_path, indice_path, filename, index_csv,
                                              subdir, size, dim, efC, m))

# print(len(tasks))
# names = sorted(list(set([task[0] for task in tasks])))
# for n in names:
#     print(n)
# print(len(construct_paras))
# index_list = list(range(1, len(construct_paras)+1))
#
# sample_num = int(len(tasks) * 0.1)
# sample_index = np.random.choice(index_list, size=sample_num, replace=False)
# sample_tasks = []
#
# for idx in sample_index:
#     para = construct_paras[idx]
#     # print(para)
#     if para not in exist_para:
#         if '960' in para[0]:
#             sample_tasks.append(tasks[idx])
# print(len(sample_tasks))

# for task in tqdm(sample_tasks, total=len(sample_tasks)):
for task in tqdm(tasks, total=len(tasks)):
    whole_name, base_path, query_path, indice_path, filename, index_csv, subdir, size, dim, efC, m = task

    index_path = os.path.join('./Index', '{}/{}_{}_{}.bin'.format(subdir, filename, efC, m))

    # 构建运行命令
    run_command = ['./index_construct_test_new', whole_name, base_path, query_path, indice_path, index_path, index_csv,
                   subdir, str(size), str(dim), str(efC), str(m)]
    # print(" ".join(run_command))
    print(f'{whole_name}_{efC}_{m}')
    print('-------------------开始构建索引并执行测试-------------------')
    result = subprocess.run(run_command, check=True, text=True, capture_output=True)
    print('-------------------索引构建与测试结束-------------------')













