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
                   '-fopenmp', '-ftree-vectorize', '-ftree-vectorizer-verbose=0', 'index_construct_test_qw_change.cpp', '-o',
                   'index_construct_test_qw_change'
                   ]
subprocess.run(compile_command, check=True)
print('编译完成')

# efCs = [20, 40, 60, 80, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 560, 620, 680, 740, 800]
# ms = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 100]

efCs = [20, 40, 60, 80, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 560, 620, 680, 740, 800]
ms = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56]

# efCs = [100, 500]
# ms = [100]

# para_list = []
# for efC in efCs:
#     for m in ms:
#         if m <= efC:
#             para_list.append((efC, m))
#         else:
#             break
# print(len(para_list))
#print(para_list)
para_list = [[311, 28], [752, 95], [572, 32], [479, 5], [403, 83], [445, 87], [96, 19], [720, 16], [602, 58], [252, 41], [201, 70], [634, 60], [144, 51], [79, 79]]
tasks = []

base_dir = "./Data/Base"
query_dir = "./Data/Query"
groundtruth_dir = "./Data/GroundTruth"
index_dir = "./Index"
index_csv = "./Data/index_performance_qw_change_random1.csv"  # 每个数据集在每个参数配置下构建的索引的相关数据，索引构建时间、内存占用、查询召回率、查询时间等数据

all_index_performance_csv = './Data/index_performance_qw_change_random1.csv'

df = pd.read_csv(all_index_performance_csv, sep=',', header=0)
#exist_para = df[['efConstruction', 'M']].to_numpy().tolist()
exist_para = []

for subdir in ['gist']:
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

                    if 1e6 <= size < 2e6 and dim in [96, 100, 128, 200, 300, 420, 960] and len(filename_list) == 3:
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
                            efC, m = para

                            if [efC, m] not in exist_para:
                                tasks.append((whole_name, base_path, filename, query_path1, query_path2, query_path3, query_path4, indice_path1, indice_path2, indice_path3, indice_path4,
                                              index_csv, subdir, size, dim, efC, m))

for task in tqdm(tasks, total=len(tasks)):
    whole_name, base_path, filename, query_path1, query_path2, query_path3, query_path4, indice_path1, indice_path2, indice_path3, indice_path4, index_csv, subdir, size, dim, efC, m = task

    index_path = os.path.join('./Index', '{}/{}_{}_{}.bin'.format(subdir, filename, efC, m))

    # 构建运行命令
    run_command = ['./index_construct_test_qw_change', whole_name, base_path, query_path1, query_path2, query_path3, query_path4, indice_path1, indice_path2, indice_path3, indice_path4, index_path, index_csv, subdir, str(size), str(dim), str(efC), str(m)]
    #print(" ".join(run_command))
    print(f'{whole_name}_{efC}_{m}')
    print('-------------------开始构建索引并执行测试-------------------')
    result = subprocess.run(run_command, check=True, text=True, capture_output=True)
    print('-------------------索引构建与测试结束-------------------')













