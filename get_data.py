import os
import pandas as pd
import struct
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
import random
from multiprocessing import Pool

from utils.data_generate import data_sample_random, data_sample_random_float, data_sample_sequential, data_sample_sequential_float, generate_data_by_normal_float, generate_multicluster_data_float

from utils.tools import save_bvecs, save_fvecs, save_ivecs

#多进程版本
def process_sample_task(args):  #处理抽样
    b_path, dim, max_num, size, dtname, level, num, i, current_directory, seed = args

    np.random.seed(seed)
    random.seed(seed)

    start_id = np.random.randint(0, max_num)

    if dim == 128:
        if size >= 1 * 1e6:
            vectors = data_sample_sequential(b_path, dim, max_num, size, start_id)
            save_path = os.path.join(current_directory, 'Data/Base/{}/{}_{}_{}_{}.bvecs'.format(dtname, level, num, dim, i))
            save_bvecs(vectors, save_path)
        else:
            vectors = data_sample_random(b_path, dim, max_num, size, flag=0)
            save_path = os.path.join(current_directory, 'Data/Base/{}/{}_{}_{}_{}.bvecs'.format(dtname, level, num, dim, i))
            save_bvecs(vectors, save_path)
    else:
        if size >= 1 * 1e6:
            vectors = data_sample_sequential_float(b_path, dim, max_num, size, start_id)
            save_path = os.path.join(current_directory, 'Data/Base/{}/{}_{}_{}_{}.fvecs'.format(dtname, level, num, dim, i))
            save_fvecs(vectors, save_path)
        else:
            vectors = data_sample_random_float(b_path, dim, max_num, size, flag=0)
            save_path = os.path.join(current_directory, 'Data/Base/{}/{}_{}_{}_{}.fvecs'.format(dtname, level, num, dim, i))
            save_fvecs(vectors, save_path)
    print(vectors.shape)
    print('数据存储完毕')


def sample_main(current_directory, dim2dtname, dim_list, sample_seed_list):
    # exist_group = [(1e5, 100), (1e6, 100), (1e6, 200)]
    exist_group = []
    # numbers = [x / 2 for x in range(2, 20)]
    numbers = [1]

    tasks = []
    for base_num in [1e4, 1e5]:
        level = {1e4: -1, 1e5: 0, 1e6: 1, 1e7: 2}.get(base_num, 3)

        for num in numbers:
            size = int(num * base_num)
            for dim in dim_list:
                dim = int(dim)

                if dim in dim2dtname.keys():
                    if (size, dim) not in exist_group:
                        dtname, max_num = dim2dtname[dim]
                        max_num = int(max_num)

                        if size < max_num:
                            if dim == 128:
                                b_path = os.path.join(current_directory, 'Data/{}-{}/{}_base.bvecs'.format(dtname, dim, dtname))
                            else:
                                b_path = os.path.join(current_directory, 'Data/{}-{}/{}_base.fvecs'.format(dtname, dim, dtname))

                            for i, seed in enumerate(sample_seed_list):
                                if i in [1]:
                                    tasks.append((b_path, dim, max_num, size, dtname, level, num, i, current_directory, seed))

    print(len(tasks))

    # 使用 Pool 来处理任务
    # with Pool(processes=3) as pool:
    #     for _ in tqdm(pool.imap_unordered(process_sample_task, tasks), total=len(tasks)):
    #         pass
    for task  in tqdm(tasks, total=len(tasks)):
        process_sample_task(task)


def process_generate_task(args):  # 处理抽样
    dim, size, i, level, num, current_directory, seed = args

    np.random.seed(seed)
    random.seed(seed)

    # mean = np.random.uniform(-500, 500)
    # std = np.random.uniform(1, 50)
    n_cluster = np.random.randint(5, 100)

    # if i < 1:
    #     vectors1 = generate_data_by_normal_float(size, dim, mean, std)

    #     save_path1 = os.path.join(current_directory, 'Data/Base/normal/{}_{}_{}_{}.fvecs'.format(level, num, dim, i))
    #     save_fvecs(vectors1, save_path1)


    vectors2 = generate_multicluster_data_float(size+10000, dim, n_cluster)

    indices = np.random.choice(size+10000, 10000, replace=False)
    all_indices = list(range(size+10000))
    other_indices = list(set(all_indices) - set(indices))

    b_vectors2 = vectors2[other_indices]
    q_vectors2 = vectors2[indices]

    b_save_path2 = os.path.join(current_directory, 'Data/Base/multicluster/{}_{}_{}_{}.fvecs'.format(level, num, dim, i))
    q_save_path2 = os.path.join(current_directory, 'Data/Query/multicluster/{}_{}_{}_{}.fvecs'.format(level, num, dim, i))
    save_fvecs(b_vectors2, b_save_path2)
    save_fvecs(q_vectors2, q_save_path2)
    print('数据存储完毕')


def generate_main(current_directory, dim2dtname, dim_list, generate_seed_list, sub_list):
    # exist_group = [(1e5, 100), (1e6, 100), (1e6, 200)]
    numbers = [x / 1 for x in range(1, 3)]
    exist_group = []

    tasks = []

    for base_num in [1e6]:
        level = {1e4: -1, 1e5: 0, 1e6: 1, 1e7: 2}.get(base_num, 3)

        for num in numbers:
            size = int(num * base_num)
            for dim in dim_list:
                dim = int(dim)

                if (size, dim) not in exist_group:
                    for i, seed in enumerate(generate_seed_list):
                        fn = '{}_{}_{}_{}'.format(level, num, dim, i)
                        if fn not in sub_list and i in [1]:
                            tasks.append((dim, size, i, level, num, current_directory, seed))
                    # i = random.choice([1, 2, 3])
                    # seed = generate_seed_list[i]
                    # fn = '{}_{}_{}_{}'.format(level, num, dim, i)
                    # if fn not in sub_list and i in [1, 2, 3]:
                    #     tasks.append((dim, size, i, level, num, current_directory, seed))
    print(len(tasks))

    # with Pool(processes=4) as pool:
    #     for _ in tqdm(pool.imap_unordered(process_generate_task, tasks), total=len(tasks)):
    #         pass
    for task  in tqdm(tasks, total=len(tasks)):
        process_generate_task(task)


if __name__ == '__main__':
    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

    # sample_seed_list = [21, 42, 88, 100, 520]
    sample_seed_list = [21, 42]
    generate_seed_list = [21, 42, 88, 100, 520, 666]

    dim2dtname = {96:['deep', 1000000], 100:['glove', 1183514], 128:['sift', 5e7], 200:['paper', 2029997],
                  256:['uqv', 1000000], 300:['crawl', 1989995], 420:['msong', 992272], 960:['gist', 1000000]}

    dim_list = [420]
    # dim_list = [100, 128, 200, 300, 420, 512, 640, 768, 896, 960]
    '''
    获取10w-100维、100w-100维、100w-200维的向量数据集，均有真实数据可以抽取
    '''
    root_dir = "./Data/Base"
    sub_list = []

    for subdir in tqdm(os.listdir(root_dir), total = len(os.listdir(root_dir))):
        if subdir in ['multicluster']:
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path) and os.listdir(subdir_path):
                for file_text in os.listdir(subdir_path):
                    filename = os.path.splitext(file_text)[0]
                    filename_list = filename.split('_')
                    if len(filename_list) == 2:
                        pass
                    else:
                        sub_list.append(filename)


    sub_list =sorted(sub_list)

#多进程版本
    sample_main(current_directory, dim2dtname, dim_list, sample_seed_list)
    #generate_main(current_directory, dim2dtname, dim_list, generate_seed_list, sub_list)

'''
#单进程版本
for base_num in [1e5, 1e6]:
    for num in numbers:
        size = int(num * base_num)
        for dim in dim_list:
            dim = int(dim)

            if dim in dim2dtname.keys():
                if (size, dim) not in exist_group:
# for size_dim in tqdm([], total = 3):
#     size = int(size_dim[0])
#     dim = int(size_dim[1])

                    dtname, max_num = dim2dtname[dim]
                    max_num = int(max_num)

                    # if 100000 <= size < 1000000:
                    #     level = 0
                    # elif 1000000 <= size < 10000000:
                    #     level = 1
                    # elif 10000000 <= size < 100000000:
                    #     level = 2
                    # else:
                    #     level = 3

                    # num = size / (np.power(10, level) * 100000)
                    if base_num == 1e5:
                        level = 0
                    elif base_num == 1e6:
                        level = 1
                    elif base_num == 1e7:
                        level = 2
                    else:
                        level = 3

                    if dim == 128:
                        b_path = os.path.join(current_directory, '{}-{}/{}_base.bvecs'.format(dtname, dim, dtname))
                    else:
                        b_path = os.path.join(current_directory, '{}-{}/{}_base.fvecs'.format(dtname, dim, dtname))

                    for i in tqdm(range(len(sample_seed_list)), total=len(sample_seed_list)):
                        seed = sample_seed_list[i]
                        np.random.seed(seed)
                        random.seed(seed)

                        if dim == 128:
                            vectors = data_sample_random(b_path, dim, max_num, size, flag=0)

                            save_path = os.path.join(current_directory, 'Data/Base/{}/{}_{}_{}_{}.bvecs'.format(dtname, level, num, dim, i))
                            save_bvecs(vectors, save_path)
                        else:
                            vectors = data_sample_random_float(b_path, dim, max_num, size, flag=0)
                            # print(vectors.shape)

                            save_path = os.path.join(current_directory, 'Data/Base/{}/{}_{}_{}_{}.fvecs'.format(dtname, level, num, dim, i))
                            save_fvecs(vectors, save_path)


# 没有真实数据集，全部生成合成数据，然后需要单独生成查询向量集
# for size_dim in tqdm([(1e5, 100), (1e6, 100), (1e6, 200)], total=3):
#     size = int(size_dim[0])
#
#     dim = int(size_dim[1])
#
#     if 100000 <= size < 1000000:
#         level = 0
#     elif 1000000 <= size < 10000000:
#         level = 1
#     elif 10000000 <= size < 100000000:
#         level = 2
#     else:
#         level = 3
#
#     num = size / (np.power(10, level) * 100000)

for base_num in [1e5, 1e6]:
    for num in numbers:
        size = int(num * base_num)
        for dim in dim_list:
            dim = int(dim)

            if (size, dim) not in exist_group:
                dtname, max_num = dim2dtname[dim]
                max_num = int(max_num)

                if base_num == 1e5:
                    level = 0
                elif base_num == 1e6:
                    level = 1
                elif base_num == 1e7:
                    level = 2
                else:
                    level = 3

                if dim in dim2dtname.keys():  #如果这个维度有真实数据集，那么就只需要生成15个合成数据集，5个normal，10个multicluster
                    seed_list = part_generate_seed_list
                else:  #否则需要生成20个合成数据集，5个normal，15个multicluster
                    seed_list = generate_seed_list

                # for i in tqdm(range(len(part_generate_seed_list)), total=len(part_generate_seed_list)):
                for i in tqdm(range(len(seed_list)), total=len(seed_list)):
                    # seed = part_generate_seed_list[i]
                    seed = seed_list[i]
                    np.random.seed(seed)
                    random.seed(seed)

                    mean = np.random.uniform(-500, 500)
                    std = np.random.uniform(1, 50)
                    n_cluster = np.random.randint(3, 500)

                    if 0 <= i < 5:
                        vectors1 = generate_data_by_normal_float(size, dim, mean, std)
                        # print(vectors1.shape)

                        save_path1 = os.path.join(current_directory, 'Data/Base/normal/{}_{}_{}_{}.fvecs'.format(level, num, dim, i))
                        save_fvecs(vectors1, save_path1)

                    vectors2 = generate_multicluster_data_float(size, dim, n_cluster)
                    # print(vectors2.shape)

                    save_path2 = os.path.join(current_directory, 'Data/Base/multicluster/{}_{}_{}_{}.fvecs'.format(level, num, dim, i))
                    save_fvecs(vectors2, save_path2)
'''

# 生成查询向量集，每个维度每个随机种子生成一个就行
# for dim in tqdm(dim_list, total=len(dim_list)):
#     dim = int(dim)
#     size = 10000

#     for i in tqdm(range(len(generate_seed_list)), total=len(generate_seed_list)):
#         seed = generate_seed_list[i]
#         np.random.seed(seed)
#         random.seed(seed)

#         mean = np.random.uniform(-500, 500)
#         std = np.random.uniform(1, 50)
#         n_cluster = np.random.randint(3, 500)

#         vectors1 = generate_data_by_normal_float(size, dim, mean, std)

#         save_path1 = os.path.join(current_directory, 'Data/Query/normal/{}_{}.fvecs'.format(dim, i))
#         save_fvecs(vectors1, save_path1)

#         vectors2 = generate_multicluster_data_float(size, dim, n_cluster)
#         save_path2 = os.path.join(current_directory, 'Data/Query/multicluster/{}_{}.fvecs'.format(dim, i))
#         save_fvecs(vectors2, save_path2)
