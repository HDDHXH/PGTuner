import os
import subprocess
import pandas as pd
import struct
import numpy as np
from tqdm import tqdm
import time
import random

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

def calculate_recall_rate(gt, qs):
    recall_rates = []

    K = qs.shape[1]
    gt = gt[:, :K]

    for row1, row2 in zip(gt, qs):
        recall_count = sum(elem in row1 for elem in row2)
        # 计算召回率：在A中找到的元素数 / B行中的元素总数
        recall_rate = recall_count / K
        recall_rates.append(recall_rate)
    
    # 计算所有行的平均召回率
    average_recall = np.mean(recall_rates)
    return average_recall


def calculate_recall_rate_given_dataset_name(dataset_name, filename):
    groundtruth_dir = "../Data/GroundTruth"
    gt_indice_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(dataset_name, filename))
    qs_indice_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(dataset_name, filename+'_test'))

    gt = read_ivecs(gt_indice_path)
    qs = read_ivecs(qs_indice_path)

    rec = calculate_recall_rate(gt, qs)
    print(rec)


# dataset_name = 'paper'
# filename = '0_2_200_1'

# calculate_recall_rate_given_dataset_name(dataset_name, filename)


s = 20
np.random.seed(s)
random.seed(s)

#KNNG参数
Ks = [100, 200, 300, 400]
Ls = [100, 150, 200, 250, 300, 350, 400] # K <= L
# Ks = [100]
# Ls = [100] # K <= L
ite = 12
S = 15
R = 100

#NSG构建参数
L_nsg_Cs = [150, 200, 250, 300, 350]
R_nsgs = [5, 10, 15, 20, 25, 30, 35, 40, 50, 70, 90] # K >= R_nsg
Cs = [300, 400, 500, 600] #C >= R_nsg
# L_nsg_Cs = [150]
# R_nsgs = [20] # K >= R_nsg
# Cs = [400]

#NSG搜索参数
L_nsg_Ss = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
            430, 460, 490, 520, 550, 580, 610, 640, 670, 700, 740, 780, 820, 860, 900, 960, 1020, 1080, 1140, 1200, 1300, 1400, 1500]
# L_nsg_Ss = [500, 1000, 1500]

KNN_para_list = []
NSG_para_list = []
for L in Ls:
    for K in Ks:
        if K <= L:
            KNN_para_list.append((K, L))
        else:
            break

for L_nsg_C in L_nsg_Cs:
    for R_nsg in R_nsgs:
        for C in Cs:
            if R_nsg <= C:
                NSG_para_list.append((L_nsg_C, R_nsg, C))
            else:
                break

print(len(KNN_para_list)) #16
print(len(NSG_para_list)) #220
print(len(L_nsg_Ss)) #52  total:183040


base_dir = "../Data/Base"
query_dir = "../Data/Query"
groundtruth_dir = "../Data/GroundTruth"
KNN_graph_dir = "./KNN_graph"
NSG_graph_dir = "./NSG_graph"
 
index_performance_csv1 = "./Data/index_performance_KNNG.csv"               #记录每个数据集在不同参数下构建KNNG的时间、距离计算次数、占用内存等信息
index_performance_csv2 = "./Data/index_performance_NSG.csv"                #记录每个数据集在固定KNNG和在不同参数下构建NSG的时间、距离计算次数、占用内存等信息
index_performance_csv3= "./Data/index_performance_Search.csv"              #记录每个数据集在固定NSG和不同参数下搜索的时间、距离计算次数、占用内存等信息
index_performance_csv4= "./Data/index_performance_Search_Recall.csv"       #记录每个数据集在固定NSG和不同参数下搜索的召回率信息

whole_index_performance_csv = './Data/index_performance_main.csv'                 #将上述4个文件合并后的总文件
    
df = pd.read_csv(index_performance_csv2, sep=',', header=0)
exist_para = df[['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C']].to_numpy().tolist()
#exist_para = []
# print(exist_para)
'''
for subdir in ['gist']:
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path):
        if os.listdir(subdir_path): # 检查子文件夹是否为空
            # 遍历子文件夹中的文件
            for file_text in os.listdir(subdir_path):
                filename = os.path.splitext(file_text)[0] #获取文件名
                
                whole_name = subdir + '_' + filename

                filename_list = filename.split('_')
                if len(filename_list) == 2:
                    print(os.path.join(subdir_path, file_text))
                else:
                    level = int(filename_list[0])
                    num = float(filename_list[1])
                    dim = int(filename_list[2])

                    size = int(pow(10, level) * 100000 * num)

                    if 1e5 <= size < 3e5:
                        base_path = os.path.join(subdir_path, file_text)
                        query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, dim))
                        gt_indice_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename))

                        for KNN_para in tqdm(KNN_para_list, total = len(KNN_para_list)):
                            K, L = KNN_para
                            KNN_graph_path = os.path.join(KNN_graph_dir, '{}/{}_{}_{}.graph'.format(subdir, filename, K, L))
                            KNN_graph_time = 0

                            for NSG_para in tqdm(NSG_para_list, total = len(NSG_para_list)):
                                L_nsg_C, R_nsg, C = NSG_para

                                if [whole_name, K, L, L_nsg_C, R_nsg, C] not in exist_para:
                                    NSG_graph_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C))

                                    print(f'{whole_name}_{K}_{L}_{L_nsg_C}_{R_nsg}_{C}')

                                    if not os.path.exists(KNN_graph_path): #同一K, L下KNN_graph_path只用构建一次
                                        print(f'-------------------开始构建KNNG: {K}_{L}，只需构建一次-------------------')
                                        t1 = time.time()
                                        run_command1 = ['./FFANNA_KNNG/build/tests/test_nndescent', base_path, KNN_graph_path, str(K), str(L), str(ite), str(S), str(R), whole_name, index_performance_csv1]  
                                        result1 = subprocess.run(run_command1, check=True, text=True, capture_output=True)
                                        t2 = time.time()

                                        KNN_graph_time = t2 - t1
                                        # print(" ".join(run_command))
                                    
                                    print(f'-------------------开始构建NSG: {L_nsg_C}_{R_nsg}_{C}-------------------')
                                    NSG_graph_time = 0
                                    if not os.path.exists(NSG_graph_path):
                                        t3 = time.time()
                                        run_command2 = ['./NSG/build/tests/test_nsg_index', base_path, KNN_graph_path, str(L_nsg_C), str(R_nsg), str(C), NSG_graph_path, str(K), str(L), whole_name, index_performance_csv2]
                                        result2 = subprocess.run(run_command2, check=True, text=True, capture_output=True)
                                        t4 = time.time()

                                        NSG_graph_time = t4 - t3

                                    print('-------------------开始搜索-------------------')
                                    recs = []
                                    for L_nsg_S in L_nsg_Ss:
                                        qs_indice_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C, L_nsg_S))

                                        run_command3 = ['./NSG/build/tests/test_nsg_optimized_search', base_path, query_path, NSG_graph_path, str(L_nsg_S), str(10), qs_indice_path, str(K), str(L), str(L_nsg_C), str(R_nsg), str(C), whole_name, index_performance_csv3]
                                        result3 = subprocess.run(run_command3, check=True, text=True, capture_output=True)

                                        gt = read_ivecs(gt_indice_path)
                                        qs = read_ivecs(qs_indice_path)

                                        rec = calculate_recall_rate(gt, qs)
                                        recs.append((whole_name, K, L, L_nsg_C, R_nsg, C, L_nsg_S, rec))

                                        #os.remove(qs_indice_path) #用完了就删除

                                        if L_nsg_S >= 300 and rec >= 0.995:  #L_nsg_S >= 500 这个得测试一下
                                            break

                                    rec_df = pd.DataFrame(recs, columns=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'recall'])
                                    rec_df.to_csv(index_performance_csv4, mode='a', header=False, index=False)

                                    if NSG_graph_time < 1800:
                                        os.remove(NSG_graph_path) #NSG构建时间较短就删除

                            # if KNN_graph_time < 900:
                            #     os.remove(KNN_graph_path) #KNNG构建时间较短就删除
'''
    
df_KNNG = pd.read_csv(index_performance_csv1, sep=',', header=0)   
df_NSG = pd.read_csv(index_performance_csv2, sep=',', header=0)  
df_Search = pd.read_csv(index_performance_csv3, sep=',', header=0)  
df_Search_Recall = pd.read_csv(index_performance_csv4, sep=',', header=0)  

df_merged = pd.merge(df_KNNG, df_NSG, on=['FileName', 'K', 'L'], how='left')

# 合并上一步的结果与 C
df_merged = pd.merge(df_merged, df_Search, on=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C'], how='left')

# 最后将得到的结果与 D 合并
df_merged = pd.merge(df_merged, df_Search_Recall, on=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S'], how='left')

df_merged.to_csv(whole_index_performance_csv, mode='w', header=True, index=False)
#'''











