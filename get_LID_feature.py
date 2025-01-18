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

from utils.LID_estimate import intrinsic_dim
from utils.tools import read_bvecs, read_fvecs
from utils.dim_reduction import pca_dr, incre_pca_dr, pca_dr_cpu
from utils.cluster import hdbscan_cluster_gpu_only


def process_file(args):
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
        vectors = read_bvecs(file_path,None)
    else:
        vectors = read_fvecs(file_path, None)

    if vectors.dtype != np.float32:
    # 如果数据不是 float32 类型，则转换为 float32
        vectors = vectors.astype(np.float32)

    if size <= 1e6:
        t1 = time.time()
        print('开始计算lid')
        lid = intrinsic_dim(vectors, 'MLE_NN')

        t2 = time.time()
        lid_time = t2 - t1

    else:
        t1 = time.time()
        print('开始计算lid')
        lids = []

        if size < 1e7:
            sample_size = int(1e6)
            sample_num = int(size / sample_size + 0.5)
        else:
            sample_size = int(5e6)
            sample_num = int(size / 5e6 + 0.5)

        for i in tqdm(range(sample_num), total = sample_num):
            sample_indexs = random.sample(range(0, size), int(sample_size))
            sample_vectors = vectors[sample_indexs]
            lid = intrinsic_dim(sample_vectors, 'MLE_NN')
            lids.append(lid)
            print(lid)

        lid = sum(lids) / sample_num

        t2 = time.time()
        lid_time = t2 - t1

    feature_data = {"FileName": whole_name, "SIZE": size, "LID": lid, "LIDTime": lid_time}

    return feature_data

    
if __name__=='__main__':
    root_dir = "./Data/Base"
    LID_feature_csv = "./Data/LID_data_feature.csv"

    exist_name = []
    if os.path.exists(LID_feature_csv):
        df = pd.read_csv(LID_feature_csv, sep=',', header=0)
        exist_name = list(df['FileName'])

    file_tasks = []
    for subdir in os.listdir(root_dir):
        # if subdir not in ['glove', 'sift', 'paper', 'crawl', 'msong', 'gist', 'multicluster]:
        if subdir in ['deep', 'glove', 'sift', 'paper', 'crawl', 'msong', 'gist']:
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path) and os.listdir(subdir_path):
                for file_text in os.listdir(subdir_path):
                    filename = os.path.splitext(file_text)[0]
                    whole_name = subdir + '_' + filename

                    filename_list = filename.split('_')

                    if len(filename_list) == 2:
                        pass

                    else:
                        level = int(filename_list[0])
                        num = float(filename_list[1])
                        dim = int(filename_list[2])

                        size = int(pow(10, level) * 100000 * num)

                        if 1e7 <= size < 2e7 and dim in [96, 100, 128, 200, 300, 420, 960] and whole_name not in exist_name:
                        # if whole_name in exist_whole_name:
                            args = (subdir, file_text, root_dir)
                            file_tasks.append(args)
                            
    print(len(file_tasks))

    for task in tqdm(file_tasks, total=len(file_tasks)):
        result = process_file(task)

        if result:
            write_header = not os.path.exists(LID_feature_csv)
            df = pd.DataFrame(result, index=[0])
            df.to_csv(LID_feature_csv, mode='a', header=write_header, index=False)


