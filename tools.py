import sys
sys.path.append('./utils')
import os
import pandas as pd
import struct
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
from tqdm import tqdm
# from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ProcessPoolExecutor
import time
# from LID_estimate import intrinsic_dim
from pathlib import Path
import random

'''
数据读取测试代码 2024.04.04
'''
#从bvecs文件中读取高维向量数据，针对SIFT和GIST
def read_bvecs_test(file_path, row_index, dim):
    with open(file_path, 'rb') as f:
        # 跳转到指定的行，每行数据的开头有4个字节表示向量维度
        f.seek(row_index * (4 + dim))  # 此处假设每个向量的维度固定为128
        dim_r = int.from_bytes(f.read(4), byteorder='little')  # 读取向量维度
        if dim_r == dim:
            vector_bytes = f.read(dim_r)  # For fvecs, each dimension is a float (4 bytes)
            vector = np.frombuffer(vector_bytes, dtype=np.int8)
        else:
            return 0
    return vector


#从fvecs文件中读取高维向量数据（float），或每个查询向量与其1000个最近邻的距离（按距离增加的顺序排序）
def read_fvecs_test(file_path, row_index, k):
    with open(file_path, 'rb') as f:
        f.seek(row_index * (4 + k * 4))  # Skip to the desired row
        k_r = int.from_bytes(f.read(4), byteorder='little')
        if k_r == k:
            vector_bytes = f.read(k_r * 4)  # For fvecs, each dimension is a float (4 bytes)
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
        else:
            return 0
    return vector

#从ivecs文件中读取每个查询向量的1000个最近邻的id（按距离增加的顺序排序）
def read_ivecs_test(file_path, row_index, k):
    with open(file_path, 'rb') as f:
        f.seek(row_index * (4 + k * 4))  # Skip to the desired row
        k_r = int.from_bytes(f.read(4), byteorder='little')
        if k_r == k:
            vector_bytes = f.read(k_r * 4)  # For fvecs, each dimension is a float (4 bytes)
            indices = np.frombuffer(vector_bytes, dtype=np.float32)
        else:
            return 0
    return indices

'''
--------------------------------------实际数据读取代码 2024.04.04--------------------------------------
'''
#从bvecs文件中读取高维向量数据  针对SIFT
def read_bvecs(file_path, num=None):
    vectors = []
    with open(file_path, 'rb') as f:
        count = 0
        while True:
            if num is not None and count >= num:
                break
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim, = struct.unpack('I', dim_bytes)
            vector_bytes = f.read(dim)
            vector = np.frombuffer(vector_bytes, dtype=np.uint8)
            vectors.append(vector)
            count += 1
    return np.array(vectors)

def read_bvecs_wo_dim(file_path, dim, num=None):
    vectors = []
    with open(file_path, 'rb') as f:
        count = 0

        while True:
            if num is not None and count >= num:
                break
            # 直接读取整个向量的数据，跳过了读取维度的步骤
            vector_bytes = f.read(dim)
            if not vector_bytes:
                break
            vector = np.frombuffer(vector_bytes, dtype=np.uint8)
            vectors.append(vector)
            count += 1
    return np.array(vectors)

#从fvecs文件中读取高维向量数据（float），或每个查询向量与其1000个最近邻的距离（按距离增加的顺序排序）
def read_fvecs(file_path, num=None):
    vectors = []
    count = 0
    with open(file_path, 'rb') as f:
        while True:
            if num is not None and count >= num:
                break
            k_bytes = f.read(4)
            if not k_bytes:
                break
            k, = struct.unpack('I', k_bytes)
            vector_bytes = f.read(k * 4)  # For fvecs, each dimension is a float (4 bytes)
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            vectors.append(vector)
            count += 1
    return np.array(vectors)

#从ivecs文件中读取每个查询向量的100个最近邻的id（按距离增加的顺序排序）
'''
sift 100最近邻
gist 100最近邻
glove 100最近邻
msong 100最近邻
uqv 100最近邻
paper 100最近邻
crawl 100最近邻
'''
def read_ivecs(file_path):
    indices = []
    with open(file_path, 'rb') as f:
        while True:
            k_bytes = f.read(4)
            if not k_bytes:
                break
            k, = struct.unpack('I', k_bytes)
            vector_bytes = f.read(k * 4)  # For ivecs, each dimension is an int (4 bytes)
            indice = np.frombuffer(vector_bytes, dtype=np.int32)
            indices.append(indice)
    return np.array(indices)


'''
--------------------------------------数据存储代码 2024.04.04--------------------------------------
'''
#向量数据存储 bvecs 针对SIFT数据集
def save_bvecs(vectors, file_path):
    with open(file_path, 'wb') as f:
        dim = vectors.shape[1]
        for vector in vectors:
            f.write(struct.pack('I', dim))
            f.write(vector.tobytes())

def save_bvecs_wo_dim(vectors, file_path):
    with open(file_path, 'wb') as f:
        for vector in vectors:
            f.write(vector.tobytes())


#将存储向量或将每个查询向量与其100个最近邻的距离（按距离增加的顺序排序）写入fvecs文件
def save_fvecs(vectors, file_path):
    dim = vectors.shape[1]
    with open(file_path, 'wb') as f:
        for vector in vectors:
            # 获取向量的维度
            # 写入维度信息
            f.write(struct.pack('I', dim))
            # 写入向量数据
            f.write(vector.astype(np.float32).tobytes())

#将每个查询向量的100个最近邻的id（按距离增加的顺序排序）写入ivecs文件def save_ivecs(indices, file_path):
def save_ivecs(indices, file_path):
    dim = indices.shape[1]
    with open(file_path, 'wb') as f:
        for indice in indices:
            # 获取向量的维度
            # 写入维度信息
            f.write(struct.pack('I', dim))
            # 写入向量数据
            f.write(indice.astype(np.int32).tobytes())

if __name__ == '__main__':
    s = 20
    np.random.seed(s)
    random.seed(s)

    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

    # b_path = os.path.join(parent_directory, 'hnswlib/bigann/bigann_base.bvecs')
    # q_path = os.path.join(parent_directory, 'hnswlib/bigann/bigann_query.bvecs')
    # d_path = os.path.join(parent_directory, 'hnswlib/bigann/gnd/dis_1M.fvecs')
    # i_path = os.path.join(parent_directory, 'hnswlib/bigann/gnd/idx_1M.ivecs')

    dtname2dim = {'deep':96, 'glove':100, 'sift':128, 'gist':960}
    dtname2size = {'deep': 10000, 'glove': 10000, 'sift': 10000, 'gist': 1000}

    dataset_name = 'deep'
    dim = dtname2dim[dataset_name]
    max_size = int(dtname2size[dataset_name])

    if not dataset_name == 'sift':
        init_q_path = os.path.join(parent_directory, 'Data/Query/{}/{}.fvecs'.format(dataset_name, dim))

        q_path1 = os.path.join(parent_directory, 'Data/Query/{}/{}_25.fvecs'.format(dataset_name, dim))
        q_path2 = os.path.join(parent_directory, 'Data/Query/{}/{}_50.fvecs'.format(dataset_name, dim))
        q_path3 = os.path.join(parent_directory, 'Data/Query/{}/{}_75.fvecs'.format(dataset_name, dim))
        q_path4 = os.path.join(parent_directory, 'Data/Query/{}/{}_100.fvecs'.format(dataset_name, dim))

        init_q_vectors = read_fvecs(init_q_path, num=None)
    else:
        init_q_path = os.path.join(parent_directory, 'Data/Query/{}/{}.bvecs'.format(dataset_name, dim))

        q_path1 = os.path.join(parent_directory, 'Data/Query/{}/{}_25.bvecs'.format(dataset_name, dim))
        q_path2 = os.path.join(parent_directory, 'Data/Query/{}/{}_50.bvecs'.format(dataset_name, dim))
        q_path3 = os.path.join(parent_directory, 'Data/Query/{}/{}_75.bvecs'.format(dataset_name, dim))
        q_path4 = os.path.join(parent_directory, 'Data/Query/{}/{}_100.bvecs'.format(dataset_name, dim))

        init_q_vectors = read_bvecs(init_q_path, num=None)

    print(init_q_vectors)
    print(init_q_vectors.min())
    print(init_q_vectors.max())
    temp_index = init_q_vectors.shape[0]
    half_dim = int(dim / 2)

    index1 = int(temp_index*0.25)
    index2 = int(temp_index*0.5)
    index3 = int(temp_index*0.75)

    indices1 = np.random.choice(max_size, index1, replace=False)
    indices2 = np.random.choice(max_size, index2, replace=False)
    indices3 = np.random.choice(max_size, index3, replace=False)

    q_vectors1 = init_q_vectors.copy()
    q_vectors2 = init_q_vectors.copy()
    q_vectors3 = init_q_vectors.copy()

    #gist
    if dataset_name == 'gist':
        mean=0.1
        std = 0.1  #glove:1, gist:0.2 or glove:0.5, gist:0.1
    elif dataset_name == 'deep':
        mean = 0
        std = 0.15
    elif dataset_name == 'sift':
        mean = 127.5
        std = 25

    print(np.random.normal(mean, std, size=(index1, dim)))
    if not dataset_name == 'sift':
        q_vectors1[indices1] = init_q_vectors[indices1] + np.clip(np.random.normal(mean, std, size=(index1, dim)), 0, 10)
        q_vectors2[indices2] = init_q_vectors[indices2] + np.clip(np.random.normal(mean, std, size=(index2, dim)), 0, 10)
        q_vectors3[indices3] = init_q_vectors[indices3] + np.clip(np.random.normal(mean, std, size=(index3, dim)), 0, 10)
        q_vectors4 = init_q_vectors + np.clip(np.random.normal(mean, std, size=(temp_index, dim)), 0, 10)

        save_fvecs(q_vectors1, q_path1)
        save_fvecs(q_vectors2, q_path2)
        save_fvecs(q_vectors3, q_path3)
        save_fvecs(q_vectors4, q_path4)
    else:
        q_vectors1[indices1] = init_q_vectors[indices1] + np.clip(np.random.normal(mean, std, size=(index1, dim)), 0, 255).astype(int)
        q_vectors2[indices2] = init_q_vectors[indices2] + np.clip(np.random.normal(mean, std, size=(index2, dim)), 0, 255).astype(int)
        q_vectors3[indices3] = init_q_vectors[indices3] + np.clip(np.random.normal(mean, std, size=(index3, dim)), 0, 255).astype(int)
        q_vectors4 = init_q_vectors + np.clip(np.random.normal(mean, std, size=(temp_index, dim)), 0, 255).astype(int)

        save_bvecs(q_vectors1, q_path1)
        save_bvecs(q_vectors2, q_path2)
        save_bvecs(q_vectors3, q_path3)
        save_bvecs(q_vectors4, q_path4)
















