import os
import pandas as pd
import struct
import numpy as np
import cupy as cp
from cuml.neighbors import NearestNeighbors
from tqdm import tqdm
import time
import random
from sklearn.neighbors import NearestNeighbors as NearestNeighbors_CPU

from utils.tools import read_bvecs, read_fvecs, save_ivecs

def process_search(args):
    subdir, file_text, root_dir, query_dir, groundtruth_dir  = args

    subdir_path = os.path.join(root_dir, subdir)
    filename = os.path.splitext(file_text)[0]
    whole_name = subdir + '_' + filename

    print(whole_name)

    filename_list = filename.split('_')
    dim = int(filename_list[2])

    file_path = os.path.join(subdir_path, file_text)

    # if subdir in ['glove', 'sift', 'paper', 'crawl', 'msong, 'gist'', 'gist']:
    if subdir in ['deep', 'glove', 'sift', 'paper', 'crawl', 'msong', 'gist']:
        if dim == 128:
            query_path = os.path.join(query_dir, '{}/{}.bvecs'.format(subdir, dim))

            base_vectors = read_bvecs(file_path, None)
            query_vectors = read_bvecs(query_path, None)

        else:
            query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, dim))

            base_vectors = read_fvecs(file_path, None)
            query_vectors = read_fvecs(query_path, None)

    else:
        query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, filename))

        base_vectors = read_fvecs(file_path, None)
        query_vectors = read_fvecs(query_path, None)
    print('开始搜索真实最近邻')

    nn = NearestNeighbors_CPU(n_neighbors=100, algorithm='brute', metric='euclidean')
    nn.fit(base_vectors)

    t1 = time.time()
    _, indices = nn.kneighbors(query_vectors)
    t2 = time.time()

    search_time = t2 - t1
    print(search_time)

    del base_vectors
    del query_vectors
    del nn

    # try:
    #     print(base_vectors.shape)
    #     print(query_vectors.shape)
    #     base_vectors = cp.asarray(base_vectors)
    #     query_vectors = cp.asarray(query_vectors)
    #
    #     nn = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='euclidean')
    #     nn.fit(base_vectors)
    #
    #     t1 = time.time()
    #     _, indices = nn.kneighbors(query_vectors)
    #     indices = np.asarray(indices.get())
    #     t2 = time.time()
    #
    #     search_time = t2 - t1
    #     print(search_time)
    #
    #     del base_vectors
    #     del query_vectors
    #     del nn
    #
    # except: #不行就放到cpu上跑
    #     nn = NearestNeighbors_CPU(n_neighbors=100, algorithm='brute', metric='euclidean')
    #     nn.fit(base_vectors)
    #
    #     t1 = time.time()
    #     _, indices = nn.kneighbors(query_vectors)
    #     t2 = time.time()
    #
    #     search_time = t2 - t1
    #     print(search_time)
    #
    #     del base_vectors
    #     del query_vectors
    #     del nn

    save_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename))
    save_ivecs(indices, save_path)
    print('真实最近邻搜索结束')

    time_data = {"FileName": whole_name, "SearchTime": search_time}

    return time_data

if __name__=='__main__':
    root_dir = "./Data/Base"
    query_dir = "./Data/Query"
    groundtruth_dir = "./Data/GroundTruth"
    bruteforce_search_time_csv = "./Data/bruteforce_search_time2.csv"

    all_bruteforce_search_time_csv = './Data/bruteforce_search_time2.csv'
    # if os.path.exists(all_bruteforce_search_time_csv):
    #     df = pd.read_csv(all_bruteforce_search_time_csv, sep=',', header=0)
    #     exist_FileName = df['FileName'].tolist()
    # else:
    #     exist_FileName = []

    exist_FileName = []

    file_tasks = []
    for subdir in tqdm(os.listdir(root_dir), total = len(os.listdir(root_dir))):
        if subdir in ['msong']:
        # if subdir in ['glove', 'paper', 'crawl', 'msong', 'gist', 'sift']:
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path) and os.listdir(subdir_path):
                for file_text in os.listdir(subdir_path):
                    filename = os.path.splitext(file_text)[0]  # 获取文件名
                    whole_name = subdir + '_' + filename

                    filename_list = filename.split('_')
                    if len(filename_list) == 2:
                        # print(os.path.join(subdir_path, file))
                        pass

                    else:
                        level = int(filename_list[0])
                        num = float(filename_list[1])
                        dim = int(filename_list[2])
                        size = int(pow(10, level) * 100000 * num)

                        if  1e4 <= size < 5e5 and whole_name not in exist_FileName:
                            args = (subdir, file_text, root_dir, query_dir, groundtruth_dir)
                            file_tasks.append(args)


    print(len(file_tasks))
    for task in tqdm(file_tasks, total = len(file_tasks)):
        result = process_search(task)

        if result:
            write_header = not os.path.exists(bruteforce_search_time_csv)
            df = pd.DataFrame(result, index=[0])
            df.to_csv(bruteforce_search_time_csv, mode='a', header=write_header, index=False)


 





