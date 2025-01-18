import sys
sys.path.append('./utils')
import os
# from sklearn.cluster import Birch, DBSCAN
# from hdbscan import HDBSCAN
from scipy.spatial.distance import pdist, mahalanobis
from scipy.linalg import pinv
import numpy as np
import cupy as cp
from cuml.cluster.hdbscan import HDBSCAN
import random
import time
from tqdm import tqdm
# from data_generate import data_sample_random, data_sample_random_float, generate_data_by_normal_float, generate_multicluster_data_float
# from LID_estimate import intrinsic_dim

# np.random.seed(42)


# 使用 HDBSCAN 进行聚类  对于使用generate_multicluster_data函数生成的数据，使用HDBSCAN计算得到的聚类数与generate_multicluster_data中指定的聚类数一致（或者相差1），我觉得可以先考虑用这个。
def hdbscan_cluster(data):
    def mahalanobis_distances(data):   #使用马氏距离，可以消除数值量纲的影响
        cov_matrix = np.cov(data, rowvar=False)
        inv_cov_matrix = pinv(cov_matrix)

        mdist = pdist(data, lambda u, v: mahalanobis(u, v, inv_cov_matrix))
        return mdist

    # t3 = time.time()
    cluster_ = HDBSCAN(min_cluster_size=5).fit(data)
    # t4 = time.time()
    # print(f'聚类时间：{t4-t3}')
    labels = cluster_.labels_

    labels = cp.asnumpy(labels)
    data = cp.asnumpy(data)

    # 计算 HDBSCAN 聚类的聚类中心和距离的平均值与方差
    unique_labels = np.unique(labels[labels >= 0])
    # print(unique_labels)
    cluster_num = len(unique_labels)
    # print(cluster_num)

    print('开始计算距离')
    t3 = time.time()
    if cluster_num == 0:  #就把每个点看成一个聚类
        cluster_num = data.shape[0]

        # distances = pdist(data, 'euclidean')
        distances = mahalanobis_distances(data)

        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        cv = 0

        # print("平均距离:", mean_distance)
        # print("距离标准差:", std_distance)
        # print("变异系数:", cv)

    elif cluster_num == 1:
        mean_distance = 0
        std_distance = 0
        cv = 0
        # print("平均距离:", mean_distance)
        # print("距离标准差:", std_distance)
        # print("变异系数:", cv)

    else:
        centers = np.array([data[labels == label].mean(axis=0) for label in unique_labels])

        # distances = pdist(centers, 'euclidean')
        distances = mahalanobis_distances(centers)

        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        cv = 0

        # print("平均距离:", mean_distance)
        # print("距离标准差:", std_distance)
        # print("变异系数:", cv)
    t4 = time.time()
    print(f'距离计算时间：{t4-t3}')


    return cluster_num, mean_distance, std_distance, cv

def hdbscan_cluster_gpu_only(data):
    def mahalanobis_distances(data):
        # 计算协方差矩阵
        cov_matrix = cp.cov(data, rowvar=False)
        # 计算协方差矩阵的逆
        inv_cov_matrix = cp.linalg.pinv(cov_matrix + 1e-6 * cp.eye(cov_matrix.shape[0]))

        # 扩展数据形状以利用广播
        data_expanded = cp.expand_dims(data, axis=1)
        diff = data_expanded - data_expanded.transpose((1, 0, 2))

        # 使用 einsum 计算马氏距离中的矩阵乘法部分
        temp = cp.einsum('ijk,kl->ijl', diff, inv_cov_matrix)
        # 计算差异向量与临时结果的点积，求和并开根号得到马氏距离
        distances = cp.sqrt(cp.einsum('ijk,ijk->ij', temp, diff))

        # 取矩阵的上三角部分（不包括对角线）
        # upper_triangle_indices = cp.triu_indices_from(distances, k=1)
        # return distances[upper_triangle_indices]
        rows, cols = cp.where(cp.triu(cp.ones_like(distances), k=1) == 1)
        return distances[rows, cols]

    def core_cluster(data, cluster_size, samples):
        hdb =  HDBSCAN(min_cluster_size=cluster_size, min_samples=samples)
        cluster_ = hdb.fit(data)
     
        labels = cluster_.labels_
        unique_labels = cp.unique(labels[labels >= 0])
        cluster_num = len(unique_labels)

        del hdb

        del cluster_

        return labels, unique_labels, cluster_num

    # t3 = time.time()
    # print(isinstance(data, cp.ndarray))
                
    cluster_size = 100
    samples = 10

    labels, unique_labels, cluster_num = core_cluster(data, cluster_size, samples)
    
    if cluster_num < 1:
        cluster_size = 50
        samples = 10

        labels, unique_labels, cluster_num = core_cluster(data, cluster_size, samples)
        
        if cluster_num < 1:
            cluster_size = 10
            samples = 5

            labels, unique_labels, cluster_num = core_cluster(data, cluster_size, samples)

            if cluster_num < 1:
                cluster_size = 5
                samples = 2

                labels, unique_labels, cluster_num = core_cluster(data, cluster_size, samples)


    # print(cluster_num)
    print('聚类完成，开始计算距离')
    # print(isinstance(unique_labels, cp.ndarray))

    # print(f'cluster_num: {cluster_num}')
    # print('开始计算距离')
    t5 = time.time()
    if cluster_num == 1: 
        mean_distance = 0
        std_distance = 0

        # print("平均距离:", mean_distance)
        # print("距离标准差:", std_distance)
        # print("变异系数:", cv)

    else:
        centers = cp.array([data[labels == label].mean(axis=0) for label in unique_labels])

        # distances = pdist(centers, 'euclidean')
        distances = mahalanobis_distances(centers)

        mean_distance = cp.mean(distances)
        std_distance = cp.std(distances)

        del distances


        # print("平均距离:", mean_distance)
        # print("距离标准差:", std_distance)
        # print("变异系数:", cv)
    t6 = time.time()
    print(f'距离计算完成，计算时间：{t6-t5}')
    

    return cluster_num, mean_distance, std_distance
    # return cluster_num,0,0



# 使用 BIRCH 进行聚类
def birch_cluster(data):
    t3 = time.time()
    cluster_ = Birch(n_clusters=None).fit(data)
    t4 = time.time()
    print(f'聚类时间：{t4 - t3}')
    labels = cluster_.labels_

    # 计算 BIRCH 聚类的聚类中心和距离的平均值与方差
    unique_labels = np.unique(labels[labels >= 0])
    # print(unique_labels)
    cluster_num = len(unique_labels)
    print(cluster_num)
    centers = np.array([data[labels == label].mean(axis=0) for label in unique_labels])

    # 计算聚类中心之间的距离矩阵
    distance_matrix = euclidean_distances(centers)
    upper_triangular_indices = np.triu_indices_from(distance_matrix, k=1)
    distances = distance_matrix[upper_triangular_indices]

    # 计算距离的平均值
    mean_distance = np.mean(distances)
    variance_distance = np.std(distances)

    print("平均距离:", mean_distance)
    print("距离方差:", variance_distance)

# 使用 DBSCAN 进行聚类
def dbscan_cluster(data):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    t1 = time.time()
    labels = dbscan.fit_predict(data)
    print(labels)
    t2 = time.time()
    print(f'聚类耗费时间{t2 - t1}s')

    # 聚类个数（排除噪声）
    if -1 in labels:
        unique_labels = set(labels) - {-1}
    else:
        unique_labels = set(labels)

    cluster_num = len(unique_labels)
    print("聚类数:",cluster_num)
    t3 = time.time()
    # 计算每个聚类的聚类中心
    cluster_centers = []
    for k in unique_labels:
        class_member_mask = (labels == k)
        # print(class_member_mask)
        cluster_center = data[class_member_mask].mean(axis=0)
        cluster_centers.append(cluster_center)

    # 将聚类中心转换为NumPy数组以便计算
    cluster_centers = np.array(cluster_centers)
    # print(cluster_centers)

    # 计算聚类中心之间的距离矩阵
    distance_matrix = euclidean_distances(cluster_centers)
    upper_triangular_indices = np.triu_indices_from(distance_matrix, k=1)
    distances = distance_matrix[upper_triangular_indices]

    mean_distance = np.mean(distances)
    variance_distance = np.std(distances)
    t4 = time.time()
    print(f'聚类耗费时间{t4 - t3}s')

    print("平均距离:", mean_distance)
    print("距离方差:", variance_distance)

if __name__ == '__main__':
    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

    # file_path = os.path.join(current_directory, 'msong-420/msong_base.fvecs')
    file_path = os.path.join(parent_directory, 'hnswlib/bigann/bigann_base.bvecs')

    # 示例数据
    max_num = 992272
    dim = 100
    # value_range = (-1, 1)
    n_sample = 10000
    n_clusters = [5]

    seed_list = [21, 42, 72, 88, 100, 103, 107, 122, 131, 150, 152, 189, 192, 215, 258, 271, 294, 309, 331, 344, 349, 360, 373, 386, 414, 436, 444, 459, 467, 492]
    for i in range(1):
        # seed = seed_list[i]
        # np.random.seed(seed)
        # random.seed(seed)
        # mean = np.random.uniform(-500, 500)
        # std = np.random.uniform(1, 50)
        # a = np.random.randint(1, 100)
        # b = np.random.randint(1, 100)
        # n_cluster = np.random.randint(2, 500)
        print(f'第{i+1}次')
        # data1 = generate_data_by_normal_float(n_sample, dim, mean, std)
        # print(data1[0])
        # data2 = generate_data_by_beta_float(n_sample, dim, a, b)
        # print(data2[0])
        # data3= generate_sphere_data_float(n_sample, dim)
        # data4 = generate_multicluster_data_float(n_sample, dim, n_cluster)
        data5 = data_sample_random(file_path, dim, max_num, n_sample, flag=1)
        print(data5[0])

    #     # print(data)
    #     # n_clusters, mean_distance, variance_distance = data_cluster(data, eps=100000, min_samples=10)
    #     hdbscan_cluster(data1)
    #     lid1 = intrinsic_dim(data1, 'MLE_NN')
    #     print(f'{lid1}\n')
    #     hdbscan_cluster(data2)
    #     lid2 = intrinsic_dim(data2, 'MLE_NN')
    #     print(f'{lid2}\n')
    #     # hdbscan_cluster(data3)
    #     # lid3 = intrinsic_dim(data3, 'MLE_NN')
    #     # print(f'{lid3}\n')
    #     hdbscan_cluster(data4)
    #     lid4 = intrinsic_dim(data4, 'MLE_NN')
    #     print(f'{lid4}\n')
        hdbscan_cluster(data5)
        lid5 = intrinsic_dim(data5, 'MLE_NN')
        print(f'{lid5}\n')
    #     # birch_cluster(data)
    #     # dbscan_cluster(data)




