import sys
sys.path.append('./utils')
import os
import numpy as np
import struct
from sklearn.datasets import make_blobs, make_s_curve, make_swiss_roll
import time
from tqdm import tqdm
from tools import read_bvecs, read_fvecs
# from LID_estimate import intrinsic_dim

# np.random.seed(42)

'''
--------------------------------------数据抽样函数--------------------------------------
'''
#按均匀分布生成索引
def uniform_sample(max_num, sample_size):  #max_num是原始数据集数据量
    indices = np.random.choice(max_num, sample_size, replace=False)
    return indices

#按正态分布生成索引
def normal_sample(max_num, sample_size, mean=0.5, std=0.1):
    indices = (np.random.normal(mean, std, sample_size) * max_num).astype(int)
    indices = np.clip(indices, 0, max_num-1)  # 限制索引在合法范围内
    return indices

#从原数据集中随机抽样 针对SIFT数据集
def data_sample_random(file_path, dim, max_num, sample_size, flag=0):  #先按某种分布随机生成给定抽样数据量的索引，然后从文件中读取每个索引对应的数据
    if flag == 0:
        indices = uniform_sample(max_num, sample_size)
        # print(indices)
    else:
        indices = normal_sample(max_num, sample_size, mean=0.5, std=0.1)

    byte_per_vector = 4 + dim  # 每个向量的字节数，包括4字节的维度信息和dim个字节的向量数据
    sampled_vectors = []  # 存储抽样结果的列表

    with open(file_path, 'rb') as f:
        # for id in tqdm(indices, total = len(indices)):
        for id in indices:
            f.seek(id * byte_per_vector)
            dim_bytes = f.read(4)
            dim, = struct.unpack('I', dim_bytes)
            vector_bytes = f.read(dim)
            vector = np.frombuffer(vector_bytes, dtype=np.uint8)
            sampled_vectors.append(vector)
 
    # vectors = read_bvecs(file_path)
    # sampled_vectors = vectors[indices]

    return np.array(sampled_vectors)
    # return sampled_vectors

#从原数据集中随机抽样 针对向量元素为浮点数的数据集
def data_sample_random_float(file_path, dim, max_num, sample_size, flag=0):  #先按某种分布随机生成给定抽样数据量的索引，然后从文件中读取每个索引对应的数据
    if flag == 0:
        indices = uniform_sample(max_num, sample_size)
    else:
        indices = normal_sample(max_num, sample_size, mean=0.5, std=0.1)

    byte_per_vector = 4 + 4 * dim  # 每个向量的字节数，包括4字节的维度信息和dim个字节的向量数据
    sampled_vectors = []  # 存储抽样结果的列表

    # with open(file_path, 'rb') as f:
    #     # for id in tqdm(indices, total = len(indices)):
    #     for id in indices:
    #         f.seek(id * byte_per_vector)
    #         dim_bytes = f.read(4)
            #   dim, = struct.unpack('I', dim_bytes)
    #         vector_bytes = f.read(4 * dim)
    #         vector = np.frombuffer(vector_bytes, dtype=np.float32)
    #         sampled_vectors.append(vector)

    vectors = read_fvecs(file_path)
    sampled_vectors = vectors[indices]

    # return np.array(sampled_vectors)
    return sampled_vectors

#从原数据集中顺序抽样 针对SIFT数据集
def data_sample_sequential(file_path, dim, max_num, sample_size, start_id):  #先随机生成一个初始索引，然后从这个索引开始顺序读取给定数据量的数据
    byte_per_vector = 4 + dim  # 每个向量的字节数，包括4字节的维度信息和dim个字节的向量数据
    vectors = []  # 存储抽样结果的列表

    # # 随机生成起始索引id
    # start_id = np.random.randint(0, max_num)
    end_id = start_id + sample_size

    with open(file_path, 'rb') as f:
        if end_id <= max_num:
            # 如果end_id没有超过max_num，直接从start_id开始读取sample_size个向量
            f.seek(start_id * byte_per_vector)
            for _ in range(sample_size):
                dim_bytes = f.read(4)
                dim_, = struct.unpack('I', dim_bytes)
                vector_bytes = f.read(dim_)
                vector = np.frombuffer(vector_bytes, dtype=np.uint8)
                vectors.append(vector)
        else:
            # 如果end_id超过了max_num，先从start_id读取到max_num-1，然后从头开始读取剩下的数据
            f.seek(start_id * byte_per_vector)
            for _ in range(max_num - start_id):
                dim_bytes = f.read(4)
                dim, = struct.unpack('I', dim_bytes)
                vector_bytes = f.read(dim)
                vector = np.frombuffer(vector_bytes, dtype=np.uint8)
                vectors.append(vector)
            f.seek(0)
            for _ in range(end_id - max_num):
                dim_bytes = f.read(4)
                dim, = struct.unpack('I', dim_bytes)
                vector_bytes = f.read(dim)
                vector = np.frombuffer(vector_bytes, dtype=np.uint8)
                vectors.append(vector)

    return np.array(vectors)

def data_sample_sequential_float(file_path, dim, max_num, sample_size, start_id):  #先随机生成一个初始索引，然后从这个索引开始顺序读取给定数据量的数据
    byte_per_vector = 4 + 4 * dim  # 每个向量的字节数，包括4字节的维度信息和dim个字节的向量数据
    vectors = []  # 存储抽样结果的列表

    # # 随机生成起始索引id
    # start_id = np.random.randint(0, max_num)
    end_id = start_id + sample_size

    with open(file_path, 'rb') as f:
        if end_id <= max_num:
            # 如果end_id没有超过max_num，直接从start_id开始读取sample_size个向量
            f.seek(start_id * byte_per_vector)
            for _ in range(sample_size):
                k_bytes = f.read(4)
                if not k_bytes:
                    break
                k, = struct.unpack('I', k_bytes)
                vector_bytes = f.read(k * 4)  # For fvecs, each dimension is a float (4 bytes)
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                vectors.append(vector)
        else:
            # 如果end_id超过了max_num，先从start_id读取到max_num-1，然后从头开始读取剩下的数据
            f.seek(start_id * byte_per_vector)
            for _ in range(max_num - start_id):
                k_bytes = f.read(4)
                if not k_bytes:
                    break
                k, = struct.unpack('I', k_bytes)
                vector_bytes = f.read(k * 4)  # For fvecs, each dimension is a float (4 bytes)
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                vectors.append(vector)
            f.seek(0)
            for _ in range(end_id - max_num):
                k_bytes = f.read(4)
                if not k_bytes:
                    break
                k, = struct.unpack('I', k_bytes)
                vector_bytes = f.read(k * 4)  # For fvecs, each dimension is a float (4 bytes)
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                vectors.append(vector)

    return np.array(vectors)

'''
--------------------------------------数据生成函数 for SIFT  先考虑以下4个函数--------------------------------------
'''
#正态分布
def generate_data_by_normal(n_samples, n_features, value_range):
    X = np.random.normal(loc=127.5 , scale=1000, size=(n_samples, n_features))
    X = np.clip(X, value_range[0], value_range[1]).astype(np.uint8)
    return X

#贝塔分布
def generate_data_by_beta(n_samples, n_features, value_range):  #参数a、b需要根据数据集进行调整，先不考虑贝塔分布
    X = np.random.beta(500,1, size=(n_samples, n_features)) * 255
    X = np.clip(X, value_range[0], value_range[1]).astype(np.uint8)
    return X

#球体
def generate_sphere_data(n_samples, n_features, value_range): #考虑乘法乘法加法
    vec = np.random.randn(n_features, n_samples)
    vec /= np.linalg.norm(vec, axis=0)
    X =vec.T
    # X = X * 255
    X = X + np.random.randint(low=value_range[0], high=value_range[1] + 1, size=(n_samples, n_features))
    X = np.clip(X, value_range[0], value_range[1]).astype(np.uint8)
    # print(X)
    return X

def generate_multicluster_data(n_samples, n_features, value_range, num_clusters): #考虑乘法加法 参数num_clusters需要根据数据集进行调整，对SIFT数据集是5
    """
    生成具有多个聚类的数据集。

    Parameters:
    - n_samples: int, 样本数量。
    - n_features: int, 特征（维度）数量。
    - n_clusters: int, 聚类中心的数量。

    Returns:
    - X: np.array, 形状为 (n_samples, n_features) 的数据集。
    """
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=num_clusters)
    #测试centers[1,5,10,20,30,40,50,100,500,1000, 200]
    #对应的LID[15, 12, 11, 11, 11, 11, 11, 11, 8, 4, 10]
    #可以测试[5, 10]
    # print(X)
    # X = X * 255
    X = X + np.random.randint(low=value_range[0], high=value_range[1] + 1, size=(n_samples, n_features))
    X = np.clip(X, value_range[0], value_range[1]).astype(np.uint8)
    # print(X)
    return X

'''
def generate_swiss_roll_data(n_samples, n_features, noise, value_range):
    """
    生成“Swiss roll”形状的数据集，并在高维空间中添加随机噪声。

    Parameters:
    - n_samples: int, 样本数量。
    - n_features: int, 特征（维度）数量，至少为3。
    - noise: float, 噪声标准差。

    Returns:
    - X: np.array, 形状为 (n_samples, n_features) 的数据集。
    """
    X, _ = make_swiss_roll(n_samples, noise=noise)
    # print(X)
    extra_features = noise * np.random.randn(n_samples, n_features-3)
    X = np.hstack((X, extra_features))
    # X = X * 255
    X = X + np.random.randint(low=value_range[0], high=value_range[1] + 1, size=(n_samples, n_features))
    X = np.clip(X, value_range[0], value_range[1]).astype(np.uint8)
    print(X)
    return X

def generate_ring_data(n_samples, n_features, noise, value_range):
    """
    生成具有环形结构的数据集，并添加噪声。

    Parameters:
    - n_samples: int, 样本数量。
    - n_features: int, 特征（维度）数量，至少为2。
    - noise: float, 噪声标准差。

    Returns:
    - X: np.array, 形状为 (n_samples, n_features) 的数据集。
    """
    theta = np.linspace(0, 2 * np.pi, n_samples)
    # x = np.cos(theta) + noise * np.random.randn(n_samples)
    # y = np.sin(theta) + noise * np.random.randn(n_samples)

    x = np.cos(theta) + noise * np.random.randn(n_samples)
    y = np.sin(theta) + noise * np.random.randn(n_samples)

    extra_features = noise * np.random.randn(n_samples, n_features - 2)
    X = np.column_stack((x, y, extra_features))
    # X = X * 255
    X = X + np.random.randint(low=value_range[0], high=value_range[1] + 1, size=(n_samples, n_features))
    X = np.clip(X, value_range[0], value_range[1]).astype(np.uint8)
    print(X)
    return X


def generate_spiral_data(n_samples, n_features, noise, value_range):
    """
    生成高维螺旋数据集。

    Parameters:
    - n_samples: int, 样本数量。
    - n_features: int, 特征（维度）数量，至少为2。
    - noise: float, 噪声标准差。

    Returns:
    - X: np.array, 形状为 (n_samples, n_features) 的数据集。
    """
    t = np.linspace(0, 4 * np.pi, n_samples)
    x = t * np.cos(t) + noise * np.random.randn(n_samples)
    y = t * np.sin(t) + noise * np.random.randn(n_samples)

    extra_features = noise * np.random.randn(n_samples, n_features - 2)
    X = np.column_stack((x, y, extra_features))
    # X = X * 255
    X = X + np.random.randint(low=value_range[0], high=value_range[1] + 1, size=(n_samples, n_features))
    X = np.clip(X, value_range[0], value_range[1]).astype(np.uint8)
    print(X)
    return X


def generate_s_shape_data(n_samples, n_features, noise, value_range):
    """
    生成具有"S"形状的数据集，并在高维空间中添加随机噪声。

    Parameters:
    - n_samples: int, 样本数量。
    - n_features: int, 特征（维度）数量，至少为3。
    - noise: float, 噪声标准差。

    Returns:
    - X: np.array, 形状为 (n_samples, n_features) 的数据集。
    """
    X, _ = make_s_curve(n_samples, noise=noise)
    # print(X)
    extra_features = noise * np.random.randn(n_samples, n_features - 3)
    X = np.hstack((X, extra_features))
    # X = X * 255
    X = X + np.random.randint(low=value_range[0], high=value_range[1] + 1, size=(n_samples, n_features))
    X = np.clip(X, value_range[0], value_range[1]).astype(np.uint8)
    print(X)
    return X


def generate_moebius_data(n_samples, n_features, noise, value_range):
    """
    生成高维“Moebius带”数据集。

    Parameters:
    - n_samples: int, 样本数量。
    - n_features: int, 特征（维度）数量，至少为3。
    - noise: float, 噪声标准差。

    Returns:
    - X: np.array, 形状为 (n_samples, n_features) 的数据集。
    """
    theta = np.linspace(0, 2 * np.pi, n_samples)
    w = noise * np.random.rand(n_samples) - 0.5
    x = np.cos(theta) + w * np.cos(2 * theta)
    y = np.sin(theta) + w * np.sin(2 * theta)
    z = w

    extra_features = noise * np.random.randn(n_samples, n_features - 3)
    X = np.column_stack((x, y, z, extra_features))
    # X = X * 255
    X = X + np.random.randint(low=value_range[0], high=value_range[1] + 1, size=(n_samples, n_features))
    X = np.clip(X, value_range[0], value_range[1]).astype(np.uint8)
    print(X)
    return X
'''

'''
--------------------------------------数据生成函数 for 浮点数向量（归一化）  先考虑以下4个函数--------------------------------------
'''
# 正态分布
def generate_data_by_normal_float(n_samples, n_features, mean, std):
    vec = np.random.normal(mean, std, size=(n_features, n_samples))

    # vec = np.random.randn(n_features, n_samples)
    vec /= np.linalg.norm(vec, axis=0)
    X = vec.T
    # X = np.clip(X, value_range[0], value_range[1])  # 确保元素在-1到1之间
    return X

# 贝塔分布
def generate_data_by_beta_float(n_samples, n_features, a, b):
    vec= np.random.beta(a, b, size=(n_features, n_samples))
    vec = 2 * (vec - 0.5)
    vec /= np.linalg.norm(vec, axis=0)
    X = vec.T
    # X = np.clip(X, value_range[0], value_range[1])  # 确保元素在-1到1之间
    return X


def generate_multicluster_data_float(n_samples, n_features, num_clusters):
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=num_clusters)
    vec = X.T
    vec /= np.linalg.norm(vec, axis=0)
    X = vec.T
    # X = np.clip(X, value_range[0], value_range[1])  # 确保元素在-1到1之间
    return X


if __name__ == '__main__':
    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

    n_sample = 10000
    dim = 128
    value_range = (-1, 1)
    noise = 0.1
    n_clusters = 5
    sample_sizes = [10000]

    # vector_data = generate_sphere_data(n_sample, dim, value_range)
    # save_path1 = os.path.join(current_directory, 'BaseData/0.01M/SIFT_sphere_0.01_128.bvecs')  # 三个{}分别表示数据量级别、数据量，以及抽样次数
    # save_bvecs(vector_data, save_path1)
    # print(0)
    # vector_data = generate_ring_data(n_sample, dim, noise, value_range)
    # save_path2 = os.path.join(current_directory, 'BaseData/0.01M/SIFT_ring_0.01_128.bvecs')  # 三个{}分别表示数据量级别、数据量，以及抽样次数
    # save_bvecs(vector_data, save_path2)
    # print(0)
    # vector_data = generate_multicluster_data(n_sample, dim, n_clusters, value_range)
    # save_path3 = os.path.join(current_directory, 'BaseData/0.01M/SIFT_multicluster_0.01_128.bvecs')  # 三个{}分别表示数据量级别、数据量，以及抽样次数
    # save_bvecs(vector_data, save_path3)
    # print(0)
    # vector_data = generate_spiral_data(n_sample, dim, noise, value_range)
    # save_path4 = os.path.join(current_directory, 'BaseData/0.01M/SIFT_spiral_0.01_128.bvecs')  # 三个{}分别表示数据量级别、数据量，以及抽样次数
    # save_bvecs(vector_data, save_path4)
    # print(0)
    # vector_data = generate_s_shape_data(n_sample, dim, noise, value_range)
    # save_path5 = os.path.join(current_directory, 'BaseData/0.01M/SIFT_s_0.01_128.bvecs')  # 三个{}分别表示数据量级别、数据量，以及抽样次数
    # save_bvecs(vector_data, save_path5)
    # print(0)
    # vector_data = generate_moebius_data(n_sample, dim, noise, value_range)
    # save_path6 = os.path.join(current_directory, 'BaseData/0.01M/SIFT_moebius_0.01_128.bvecs')  # 三个{}分别表示数据量级别、数据量，以及抽样次数
    # save_bvecs(vector_data, save_path6)
    # print(0)
    # vector_data = generate_swiss_roll_data(n_sample, dim, noise, value_range)
    # save_path7 = os.path.join(current_directory, 'BaseData/0.01M/SIFT_swiss_0.01_128.bvecs')  # 三个{}分别表示数据量级别、数据量，以及抽样次数
    # save_bvecs(vector_data, save_path7)
    # print(0)



    # for data_path in [save_path3]:#[save_path1, save_path2, save_path3, save_path4, save_path5, save_path6, save_path7]:
    #     t1 = time.time()
    #     data = read_bvecs(data_path, k=None)
    #     t2 = time.time()
    #     print(f'数据读取时间：{t2 - t1}')
    #     # print('开始估计LID')
    #     t3 = time.time()
    #     lid = intrinsic_dim(data, 'MLE_NN')
    #     t4 = time.time()
    #     print(f'LID计算时间：{t4 - t3}')
    #     print(f'LID：{lid}')

    # for sample_size in tqdm(sample_sizes, total=len(sample_sizes)):
    #     if 100000 <= sample_size < 1000000:
    #         level = 0.1
    #     elif 1000000 <= sample_size < 10000000:
    #         level = 1
    #     elif 10000000 <= sample_size < 100000000:
    #         level = 10
    #     else:
    #         level = 100
    #
    #     num = sample_size / 1000000
    #
    #     # id_set = set()
    #     for i in range(10):
    #         data =np.random.poisson(25.5, size=(sample_size, dim))
    #         t3 = time.time()
    #         lid = intrinsic_dim(data, 'MLE_NN')
    #         t4 = time.time()
    #         print(f'LID计算时间：{t4 - t3}')
    #         print(f'LID：{lid}')

    t1 = time.time()
    # data =  generate_data_by_beta_float(n_sample, dim)
    # data = generate_sphere_data_float(n_sample, dim)
    data = generate_multicluster_data_float(n_sample, dim, n_clusters)
    print(data[0])
    t2 = time.time()
    print(f'数据生成时间：{t2 - t1}')
    # print('开始估计LID')
    t3 = time.time()
    lid = intrinsic_dim(data, 'MLE_NN')
    t4 = time.time()
    print(f'LID计算时间：{t4 - t3}')
    print(f'LID：{lid}')




