import os
import multiprocessing as mp
from multiprocessing import Pool
import pandas as pd
import struct
import numpy as np
from tqdm import tqdm
import time
import random
import cupy as cp
from cuml.neighbors import NearestNeighbors
from utils.tools import read_bvecs, read_fvecs
from utils.data_generate import data_sample_sequential, data_sample_sequential_float
import math

def uniform_sample(max_num, sample_size):  #max_num是原始数据集数据量
    indices = np.random.choice(max_num, sample_size, replace=False)
    return indices

def get_q_k_neighbor_dists(b_vectors, q_vectors):
    print(q_vectors.shape)
    b_vectors = cp.array(b_vectors)
    q_vectors = cp.array(q_vectors)

    nn = NearestNeighbors(n_neighbors=1000, algorithm='brute', metric='euclidean')
    nn.fit(b_vectors)

    distances, _ = nn.kneighbors(q_vectors)

    top_10_distances = distances[:, :10]
    other_distances = distances[:, 10:]

    mean_top_10_distances = top_10_distances.mean(axis=1)
    mean_other_distances = other_distances.mean(axis=1)

    zero_rows = cp.isclose(mean_other_distances, 0)
    indices_to_remove = cp.where(zero_rows)[0]

    mean_other_distances_non_zero = mean_other_distances[~zero_rows]
    mean_top_10_distances_non_zero = mean_top_10_distances[~zero_rows]

    ratios = mean_top_10_distances_non_zero / mean_other_distances_non_zero

    min_ratio = ratios.min()
    max_ratio = ratios.max()
    mean_ratio = ratios.mean()
    std_ratio = ratios.std()
    median_ratio = cp.percentile(ratios, 50)

    sum_top_10_distances = top_10_distances.sum(axis=1)
    sum_other_distances = other_distances.sum(axis=1)

    zero_rows = cp.isclose(sum_other_distances, 0)
    indices_to_remove = cp.where(zero_rows)[0]

    sum_other_distances_non_zero = sum_other_distances[~zero_rows]
    sum_top_10_distances_non_zero = sum_top_10_distances[~zero_rows]

    sum_ratios = sum_top_10_distances_non_zero / sum_other_distances_non_zero

    min_sum_ratio = sum_ratios.min()
    max_sum_ratio = sum_ratios.max()
    mean_sum_ratio = sum_ratios.mean()
    std_sum_ratio = sum_ratios.std()
    median_sum_ratio = cp.percentile(sum_ratios, 50)

    return min_ratio, median_ratio, max_ratio, mean_ratio,  std_ratio, min_sum_ratio, median_sum_ratio, max_sum_ratio, mean_sum_ratio,  std_sum_ratio

        
    # whole_mean_dist = distances.mean()
    # distances = distances / whole_mean_dist

    # k_neighbor_dists = distances[:, -1]
    # sum_k_neighbor_dists = distances.sum(axis=1)

    # min_dist = k_neighbor_dists.min()
    # max_dist = k_neighbor_dists.max()
    # mean_dist = k_neighbor_dists.mean()
    # std_dist = k_neighbor_dists.std()
    # median_dist = cp.percentile(k_neighbor_dists, 50)

    # sum_min_dist = sum_k_neighbor_dists.min()
    # sum_max_dist = sum_k_neighbor_dists.max()
    # sum_mean_dist = sum_k_neighbor_dists.mean()
    # sum_std_dist = sum_k_neighbor_dists.std()
    # sum_median_dist = cp.percentile(sum_k_neighbor_dists, 50)

    # return min_dist, median_dist, max_dist, mean_dist,  std_dist, sum_min_dist, sum_median_dist, sum_max_dist, sum_mean_dist,  sum_std_dist


def process_file(args):  #经过简单的测试，采样多次后得到的平均距离与直接在整个数据上暴力搜索得到的距离相差很小，后续在真是数据及上再验证一下，现在先完成想嘎u你的代码
    subdir, file_text, b_root_dir,  q_root_dir = args

    b_subdir_path = os.path.join(b_root_dir, subdir)
    q_subdir_path = os.path.join(q_root_dir, subdir)

    filename = os.path.splitext(file_text)[0]
    whole_name = subdir + '_' + filename

    filename_list = filename.split('_')
    level = int(filename_list[0])
    num = float(filename_list[1])
    dim = int(filename_list[2])
    size = int(pow(10, level) * 100000 * num)

    b_file_path = os.path.join(b_subdir_path, file_text)
    print(whole_name)

    if dim == 128 and subdir == 'sift':
        q_file_path = os.path.join(q_subdir_path, filename_list[2]+'.bvecs')

        b_vectors = read_bvecs(b_file_path, None)
        q_vectors = read_bvecs(q_file_path, None)
    else:
        q_file_path = os.path.join(q_subdir_path, filename_list[2]+'.fvecs')

        b_vectors = read_fvecs(b_file_path, None)
        q_vectors = read_fvecs(q_file_path, None)

    if b_vectors.dtype != np.float32:
        # 如果数据不是 float32 类型，则转换为 float32
        b_vectors = b_vectors.astype(np.float32)
        q_vectors = q_vectors.astype(np.float32)

    q_size = int(q_vectors.shape[0])

    t1 = time.time()
    min_ratio, median_ratio, max_ratio, mean_ratio,  std_ratio, min_sum_ratio, median_sum_ratio, max_sum_ratio, mean_sum_ratio,  std_sum_ratio = get_q_k_neighbor_dists(b_vectors, q_vectors)
    # min_dist, median_dist, max_dist, mean_dist, std_dist, sum_min_dist, sum_median_dist, sum_max_dist, sum_mean_dist,  sum_std_dist = get_q_k_neighbor_dists(b_vectors, q_vectors)
    t2 = time.time()
    search_time = t2 - t1
    print(f'暴力搜索10近邻的时间为: {search_time}')

    # feature_data = {
    #     "FileName": whole_name, "q_K_MinRatio": min_ratio, "q_K_MedianRatio": median_ratio, "q_MaxRatio": max_ratio, "q_K_MeanRatio": mean_ratio, "q_K_StdRatio": std_ratio,
    #     "q_sum_K_MinRatio": min_sum_ratio, "q_sum_K_MedianRatio": median_sum_ratio, "q_sum_MaxRatio": max_sum_ratio, "q_sum_K_MeanRatio": mean_sum_ratio, "q_sum_K_StdRatio": std_sum_ratio, "q_SearchTime": search_time}
    feature_data = {
        "FileName": whole_name, "q_SIZE": q_size, "q_K_MinRatio": min_ratio, "q_K_MaxRatio": max_ratio, "q_K_MeanRatio": mean_ratio, "q_K_StdRatio": std_ratio, "q_SearchTime": search_time}

    return feature_data



if __name__ == '__main__':
    b_root_dir = "./Data/Base"
    q_root_dir = "./Data/Query"
    query_K_neighbor_dist_feature_csv = "./Data/query_K_neighbor_dist_ratio_feature.csv"

    exist_name = []
    if os.path.exists(query_K_neighbor_dist_feature_csv):
        df = pd.read_csv(query_K_neighbor_dist_feature_csv, sep=',', header=0)
        exist_name = list(df['FileName'])

    file_tasks = []

    for subdir in os.listdir(b_root_dir):
        # if subdir not in ['glove', 'sift', 'paper', 'crawl', 'msong', 'gist', 'multicluster']:
        if subdir in ['deep', 'sift']:
            subdir_path = os.path.join(b_root_dir, subdir)
            if os.path.isdir(subdir_path) and os.listdir(subdir_path):
                for file_text in os.listdir(subdir_path):
                    filename = os.path.splitext(file_text)[0]
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

                        if 2e6 <= size <6e6  and dim in [96, 100, 128, 200, 300, 420, 960] and whole_name not in exist_name:
                            # if whole_name in exist_whole_name:
                            args = (subdir, file_text, b_root_dir, q_root_dir)
                            file_tasks.append(args)

    # print(len(file_tasks))

    for task in tqdm(file_tasks, total=len(file_tasks)):
        # process_file_test(task)
        result = process_file(task)
        # results.append(result)

        if result:
            write_header = not os.path.exists(query_K_neighbor_dist_feature_csv)
            df = pd.DataFrame(result, index=[0])
            df.to_csv(query_K_neighbor_dist_feature_csv, mode='a', header=write_header, index=False)


