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

def get_k_neighbor_dists(vectors):

    # def compute_incremental_cov_matrix(data, batch_size):
    #     n_samples, n_features = data.shape
    #     mean = cp.zeros(n_features)
    #     cov_matrix = cp.zeros((n_features, n_features))

    #     for start in range(0, n_samples, batch_size):
    #         end = min(start + batch_size, n_samples)
    #         batch = data[start:end]

    #         batch_mean = cp.mean(batch, axis=0)
    #         mean += batch_mean * (end - start)

    #         centered_batch = batch - batch_mean
    #         cov_matrix += cp.dot(centered_batch.T, centered_batch)

    #     mean /= n_samples
    #     cov_matrix /= (n_samples - 1)
    #     return mean, cov_matrix

    # batch_size = 10000  # Adjust batch size based on memory constraints
    # mean, cov_matrix = compute_incremental_cov_matrix(vectors, batch_size)
    # inv_cov_matrix = cp.linalg.inv

    # def mahalanobis_dist(u, v, inv_cov_matrix):
    #     delta = u - v
    #     dist = cp.sqrt(cp.dot(cp.dot(delta, inv_cov_matrix), delta.T))
    #     return dist
    
    vectors = np.unique(vectors, axis=0)
    vectors = cp.array(vectors)

    nn = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='euclidean')
    nn.fit(vectors)

    distances, _ = nn.kneighbors(vectors)

    # mahalanobis_distances = cp.zeros_like(distances)
    # for i in range(vectors.shape[0]):
    #     for j in range(1, distances.shape[1]):  # start from 1 to skip self-neighbor
    #         u = vectors[i]
    #         v = vectors[indices[i, j]]
    #         mahalanobis_distances[i, j] = mahalanobis_dist(u, v, inv_cov_matrix)
        
    whole_mean_dist = distances[:, 1:-1].mean()
    distances = distances / whole_mean_dist

    k_neighbor_dists = distances[:, -1]
    sum_k_neighbor_dists = distances[:, 1:-1].sum(axis=1)

    min_dist = k_neighbor_dists.min()
    max_dist = k_neighbor_dists.max()
    mean_dist = k_neighbor_dists.mean()
    std_dist = k_neighbor_dists.std()
    median_dist = cp.percentile(k_neighbor_dists, 50)

    sum_min_dist = sum_k_neighbor_dists.min()
    sum_max_dist = sum_k_neighbor_dists.max()
    sum_mean_dist = sum_k_neighbor_dists.mean()
    sum_std_dist = sum_k_neighbor_dists.std()
    sum_median_dist = cp.percentile(sum_k_neighbor_dists, 50)

    return min_dist, median_dist, max_dist, mean_dist,  std_dist, sum_min_dist, sum_median_dist, sum_max_dist, sum_mean_dist,  sum_std_dist
def process_file_test(args):  #经过简单的测试，采样多次后得到的平均距离与直接在整个数据上暴力搜索得到的距离相差很小，后续在真是数据及上再验证一下，现在先完成想嘎u你的代码
    subdir, file_text, root_dir = args

    subdir_path = os.path.join(root_dir, subdir)
    filename = os.path.splitext(file_text)[0]
    whole_name = subdir + '_' + filename

    filename_list = filename.split('_')
    level = int(filename_list[0])
    num = float(filename_list[1])
    dim = int(filename_list[2])
    size = int(pow(10, level) * 100000 * num)

    file_path = os.path.join(subdir_path, file_text)
    print(whole_name)

    if dim == 128 and subdir == 'sift':
        vectors = read_bvecs(file_path, None)
    else:
        vectors = read_fvecs(file_path, None)

    if vectors.dtype != np.float32:
        # 如果数据不是 float32 类型，则转换为 float32
        vectors = vectors.astype(np.float32)

    t1 = time.time()
    min_dist, median_dist, max_dist, mean_dist, std_dist, sum_min_dist, sum_median_dist, sum_max_dist, sum_mean_dist,  sum_std_dist = get_k_neighbor_dists(vectors)
    t2 = time.time()
    print(f'暴力搜索10近邻的时间为: {t2-t1}')
    #
    # print(min_dist, median_dist, max_dist, mean_dist,  std_dist)

    raw_dist_data = [min_dist, median_dist, max_dist, mean_dist,  std_dist]
    print(f'原始数据暴力搜索的距离数据为： {raw_dist_data}')

    # sampled_min_dist_data = []
    # sampled_median_dist_data = []
    # sampled_max_dist_data = []
    # sampled_mean_dist_data = []
    # sampled_std_dist_data = []
    #
    # sample_num = int(size / 1e5)
    # sample_ids = uniform_sample(size, sample_num)
    # t3 = time.time()
    # for start_id in tqdm(sample_ids, total = len(sample_ids)):
    #     sample_vectors = data_sample_sequential_float(file_path, dim, size, int(1e5), start_id)
    #     sample_min_dist, sample_median_dist, sample_max_dist,sample_mean_dist, sample_std_dist = get_k_neighbor_dists(sample_vectors)
    #
    #     sampled_min_dist_data.append(sample_min_dist)
    #     sampled_median_dist_data.append(sample_median_dist)
    #     sampled_max_dist_data.append(sample_max_dist)
    #     sampled_mean_dist_data.append(sample_mean_dist)
    #     sampled_std_dist_data.append(sample_std_dist)
    # t4 = time.time()
    # print(f'采样暴力搜索10近邻的时间为: {t4 - t3}')
    #
    # sampled_dist_data = [sum(sampled_min_dist_data) / sample_num, sum(sampled_median_dist_data) / sample_num, sum(sampled_max_dist_data) / sample_num,
    #                      sum(sampled_mean_dist_data) / sample_num, sum(sampled_std_dist_data) / sample_num]
    # print(f'采样数据暴力搜索的距离数据为： {sampled_dist_data}')

    # t1 = time.time()
    # lid = intrinsic_dim(vectors, 'MLE_NN')
    # t2 = time.time()
    # print(f'暴力搜索计算lid的时间: {t2-t1}, lid为: {lid}')


    # feature_data = {
    #     "FileName": whole_name, "SIZE": size, "DIM": dim, "LID": lid, "ClustersNum": cluster_num,
    #     "MeanDist": mean_distance, "StdDist": std_distance, "LIDTime": lid_time, "ReductionDimTime": dr_time,
    #     "ClusterTime": cluster_time
    # }
    #
    # return feature_data


def process_file(args):  #经过简单的测试，采样多次后得到的平均距离与直接在整个数据上暴力搜索得到的距离相差很小，后续在真是数据及上再验证一下，现在先完成想嘎u你的代码
    subdir, file_text, root_dir = args

    subdir_path = os.path.join(root_dir, subdir)
    filename = os.path.splitext(file_text)[0]
    whole_name = subdir + '_' + filename

    filename_list = filename.split('_')
    level = int(filename_list[0])
    num = float(filename_list[1])
    dim = int(filename_list[2])
    size = int(pow(10, level) * 100000 * num)

    file_path = os.path.join(subdir_path, file_text)
    print(whole_name)

    if dim == 128 and subdir == 'sift':
        vectors = read_bvecs(file_path, None)
    else:
        vectors = read_fvecs(file_path, None)

    if vectors.dtype != np.float32:
        # 如果数据不是 float32 类型，则转换为 float32
        vectors = vectors.astype(np.float32)

    if size <= 1e6:
        t1 = time.time()
        min_dist, median_dist, max_dist, mean_dist, std_dist, sum_min_dist, sum_median_dist, sum_max_dist, sum_mean_dist,  sum_std_dist = get_k_neighbor_dists(vectors)
        t2 = time.time()
        search_time = t2 - t1
        print(f'暴力搜索10近邻的时间为: {search_time}')


    else:
        sampled_min_dist_data = []
        sampled_median_dist_data = []
        sampled_max_dist_data = []
        sampled_mean_dist_data = []
        sampled_std_dist_data = []

        sampled_sum_min_dist_data = []
        sampled_sum_median_dist_data = []
        sampled_sum_max_dist_data = []
        sampled_sum_mean_dist_data = []
        sampled_sum_std_dist_data = []

        if size < 1e7:
            sample_num = int(size / 1e6 + 0.5)
        else:
            sample_num = int(size / 5e6 + 0.5)
        sample_ids = uniform_sample(size, sample_num)
        t3 = time.time()
        for start_id in tqdm(sample_ids, total = len(sample_ids)):
            if size < 1e7:
                if subdir == 'sift':
                    sample_vectors = data_sample_sequential(file_path, dim, size, int(1e6), start_id)
                else:
                    sample_vectors = data_sample_sequential_float(file_path, dim, size, int(1e6), start_id)
            else:
                if subdir == 'sift':
                    sample_vectors = data_sample_sequential(file_path, dim, size, int(5e6), start_id)
                else:
                    sample_vectors = data_sample_sequential_float(file_path, dim, size, int(5e6), start_id)
            sample_min_dist, sample_median_dist, sample_max_dist,sample_mean_dist, sample_std_dist, sample_sum_min_dist, sample_sum_median_dist, sample_sum_max_dist, sample_sum_mean_dist,  sample_sum_std_dist = get_k_neighbor_dists(sample_vectors)

            sampled_min_dist_data.append(sample_min_dist)
            sampled_median_dist_data.append(sample_median_dist)
            sampled_max_dist_data.append(sample_max_dist)
            sampled_mean_dist_data.append(sample_mean_dist)
            sampled_std_dist_data.append(sample_std_dist)

            sampled_sum_min_dist_data.append(sample_sum_min_dist)
            sampled_sum_median_dist_data.append(sample_sum_median_dist)
            sampled_sum_max_dist_data.append(sample_sum_max_dist)
            sampled_sum_mean_dist_data.append(sample_sum_mean_dist)
            sampled_sum_std_dist_data.append(sample_sum_std_dist)

        min_dist = sum(sampled_min_dist_data) / sample_num
        median_dist = sum(sampled_median_dist_data) / sample_num
        max_dist = sum(sampled_max_dist_data) / sample_num
        mean_dist = sum(sampled_mean_dist_data) / sample_num
        std_dist = sum(sampled_std_dist_data) / sample_num

        sum_min_dist = sum(sampled_sum_min_dist_data) / sample_num
        sum_median_dist = sum(sampled_sum_median_dist_data) / sample_num
        sum_max_dist = sum(sampled_sum_max_dist_data) / sample_num
        sum_mean_dist = sum(sampled_sum_mean_dist_data) / sample_num
        sum_std_dist = sum(sampled_sum_std_dist_data) / sample_num
        print(sum_min_dist, sum_max_dist, sum_std_dist)

        t4 = time.time()
        search_time = t4 - t3
        print(f'采样暴力搜索10近邻的时间为: {search_time}')

    # feature_data = {
    #     "FileName": whole_name, "K_MinDist": min_dist, "K_MedianDist": median_dist, "K_MaxDist": max_dist,
    #     "K_MeanDist": mean_dist, "K_StdDist": std_dist, "Sum_K_MinDist": sum_min_dist, "Sum_K_MedianDist": sum_median_dist, "Sum_K_MaxDist": sum_max_dist,
    #     "Sum_K_MeanDist": sum_mean_dist, "Sum_K_StdDist": sum_std_dist, "SearchTime": search_time}
    feature_data = {
        "FileName": whole_name, "Sum_K_MinDist": sum_min_dist, "Sum_K_MaxDist": sum_max_dist, "Sum_K_StdDist": sum_std_dist, "SearchTime": search_time}

    return feature_data



if __name__ == '__main__':
    root_dir = "./Data/Base"
    K_neighbor_dist_feature_csv = "./Data/K_neighbor_dist_feature.csv"

    exist_name = []
    if os.path.exists(K_neighbor_dist_feature_csv):
        df = pd.read_csv(K_neighbor_dist_feature_csv, sep=',', header=0)
        exist_name = list(df['FileName'])

    file_tasks = []

    for subdir in os.listdir(root_dir):
        # if subdir not in ['glove', 'sift', 'paper', 'crawl', 'msong', 'gist', 'multicluster']:
        if subdir in ['deep', 'glove', 'sift', 'paper', 'crawl', 'msong', 'gist']:
            subdir_path = os.path.join(root_dir, subdir)
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

                        if 2e7 <= size < 6e7  and dim in [96, 100, 128, 200, 300, 420, 960] and whole_name not in exist_name:
                            # if whole_name in exist_whole_name:
                            args = (subdir, file_text, root_dir)
                            file_tasks.append(args)

    # print(len(file_tasks))

    for task in tqdm(file_tasks, total=len(file_tasks)):
        # process_file_test(task)
        result = process_file(task)
        # results.append(result)

        if result:
            write_header = not os.path.exists(K_neighbor_dist_feature_csv)
            df = pd.DataFrame(result, index=[0])
            df.to_csv(K_neighbor_dist_feature_csv, mode='a', header=write_header, index=False)


