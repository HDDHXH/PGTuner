import os
import subprocess
import pandas as pd
import struct
import numpy as np
from tqdm import tqdm
import time
import random
import re

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

filename_dic = {'deep1':'1_1_96_1', 'sift1':'1_1_128_1', 'glove': '1_1.183514_100', 'paper':'1_2.029997_200', 'crawl':'1_1.989995_300', 'msong':'-1_1_420_1',
                    'gist':'1_1.0_960', 'deep10':'2_1_96', 'sift50':'2_5_128_1', 'deep2':'1_2_96_1', 'deep3':'1_3_96_1', 'deep4':'1_4_96_1', 'deep5':'1_5_96_1',
                    'sift2':'1_2_128_1', 'sift3':'1_3_128_1', 'sift4':'1_4_128_1', 'sift5':'1_5_128_1', 'gist_25':'1_1.0_960_25', 'gist_50':'1_1.0_960_50',
                    'gist_75':'1_1.0_960_75', 'gist_100':'1_1.0_960_100', 'deep2_25':'1_2_96_1_25', 'deep2_50':'1_2_96_1_50', 'deep2_75':'1_2_96_1_75', 'deep2_100':'1_2_96_1_100'}

para_dic_test_main = {'deep10': [[0.9, 200, 200, 200, 40, 500, 100]]}

ite = 12
S = 15
R = 100

base_dir = "../Data/Base"
query_dir = "../Data/Query"
groundtruth_dir = "../Data/GroundTruth"
KNN_graph_dir = "./KNN_graph"
NSG_graph_dir = "./NSG_graph"
 
index_performance_csv1 = "./Data/experiments_results/index_performance_KNNG_compare.csv"               #记录每个数据集在不同参数下构建KNNG的时间、距离计算次数、占用内存等信息
index_performance_csv2 = "./Data/experiments_results/index_performance_NSG_compare.csv"                #记录每个数据集在固定KNNG和在不同参数下构建NSG的时间、距离计算次数、占用内存等信息
index_performance_csv3= "./Data/experiments_results/index_performance_Search_compare.csv"              #记录每个数据集在固定NSG和不同参数下搜索的时间、距离计算次数、占用内存等信息
index_performance_csv4= "./Data/experiments_results/index_performance_Search_Recall_compare.csv"       #记录每个数据集在固定NSG和不同参数下搜索的召回率信息

whole_index_performance_csv = './Data/experiments_results/index_performance_compare.csv'                 #将上述4个文件合并后的总文件
    
# df = pd.read_csv(index_performance_csv2, sep=',', header=0)
# exist_para = df[['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C']].to_numpy().tolist()
exist_para = []
#'''
for para_dic in tqdm([para_dic_test_main], total = len([para_dic_test_main])):
    for dataset_name in tqdm(para_dic.keys(), total = len(para_dic.keys().tolist())):
        para_list = para_dic[dataset_name]

        filename = filename_dic[dataset_name]
        subdir = re.match(r'\D+', dataset_name).group()

        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            file_text = filename + '.fvecs'

            whole_name = subdir + '_' + filename

            filename_list = filename.split('_')
            if len(filename_list) == 2:
                print(os.path.join(subdir_path, file_text))
            else:
                level = int(filename_list[0])
                num = float(filename_list[1])
                dim = int(filename_list[2])

                size = int(pow(10, level) * 100000 * num)

                # if  (5e6 <= size < 1e7) and dim in [100] and i == 1:
                if dim in [96, 100, 128, 200, 300, 420, 960]:
                    base_path = os.path.join(subdir_path, file_text)
                    gt_indice_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename))

                    query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, dim))

                    for para in tqdm(para_list, total = len(para_list)):
                        target_recall, K, L, L_nsg_C, R_nsg, C, pr_L_nsg_S = para
                        K = int(K)
                        L = int(L)
                        L_nsg_C = int(L_nsg_C)
                        R_nsg = int(R_nsg)
                        C = int(C)
                        pr_L_nsg_S = int(pr_L_nsg_S)

                        if [whole_name, target_recall, K, L, L_nsg_C, R_nsg, C, pr_L_nsg_S] not in exist_para:
                            KNN_graph_path = os.path.join(KNN_graph_dir, '{}/{}_{}_{}.graph'.format(subdir, filename, K, L))
                            NSG_graph_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C))

                            print(f'-------------------开始构建KNNG: {K}_{L}，只需构建一次-------------------')
                            t1 = time.time()
                            run_command1 = ['./FFANNA_KNNG/build/tests/test_nndescent', base_path, KNN_graph_path, str(K), str(L), str(ite), str(S), str(R), whole_name, index_performance_csv1]
                            result1 = subprocess.run(run_command1, check=True, text=True, capture_output=True)
                            t2 = time.time()

                            KNN_graph_time = t2 - t1
                                    
                            print(f'-------------------开始构建NSG: {L_nsg_C}_{R_nsg}_{C}-------------------')
                            t3 = time.time()
                            run_command2 = ['./NSG/build/tests/test_nsg_index', base_path, KNN_graph_path, str(L_nsg_C), str(R_nsg), str(C), NSG_graph_path, str(K), str(L), whole_name, index_performance_csv2]
                            result2 = subprocess.run(run_command2, check=True, text=True, capture_output=True)
                            t4 = time.time()

                            NSG_graph_time = t4 - t3

                            print('-------------------开始搜索-------------------')
                            dealt = 0
                            if (pr_L_nsg_S < 100):
                                dealt = 1
                            elif (100 <= pr_L_nsg_S < 200):
                                dealt = 5
                            elif (200 <= pr_L_nsg_S < 400):
                                dealt = 10
                            elif (400 <= pr_L_nsg_S < 700):
                                dealt = 15
                            elif (700 <= pr_L_nsg_S < 900):
                                dealt = 20
                            elif (900 <= pr_L_nsg_S < 1200):
                                dealt = 30
                            elif (1200 <= pr_L_nsg_S < 1500):
                                dealt = 50
                            else:
                                dealt = 100


                            qs_indice_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C, pr_L_nsg_S))

                            run_command3 = ['./NSG/build/tests/test_nsg_optimized_search', base_path, query_path, NSG_graph_path, str(pr_L_nsg_S), str(10), qs_indice_path, str(K), str(L), str(L_nsg_C), str(R_nsg), str(C), whole_name, index_performance_csv3]
                            result3 = subprocess.run(run_command3, check=True, text=True, capture_output=True)

                            gt = read_ivecs(gt_indice_path)
                            qs = read_ivecs(qs_indice_path)

                            rec = calculate_recall_rate(gt, qs)

                            os.remove(qs_indice_path) #用完了就删除
                            ##'FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'pr_L_nsg_S', 'real_L_nsg_S', 'target_recall', 'real_recall', 'paras_search_time'
                            t5 = time.time()
                            init_paras_search_time = t5 - t4

                            recs = [(whole_name, K, L, L_nsg_C, R_nsg, C, pr_L_nsg_S, pr_L_nsg_S, target_recall, rec, init_paras_search_time)]

                            L_nsg_S = pr_L_nsg_S
                            if rec > target_recall:
                                while(True):
                                    if (L_nsg_S < 100):
                                        dealt = 1
                                    elif (100 <= L_nsg_S < 200):
                                        dealt = 5
                                    elif (200 <= L_nsg_S < 400):
                                        dealt = 10
                                    elif (400 <= L_nsg_S < 700):
                                        dealt = 15
                                    elif (700 <= L_nsg_S < 900):
                                        dealt = 20
                                    elif (900 <= L_nsg_S < 1200):
                                        dealt = 30
                                    elif (1200 <= L_nsg_S < 1500):
                                        dealt = 50
                                    else:
                                        dealt = 100

                                    L_nsg_S = L_nsg_S - dealt

                                    qs_indice_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C, pr_L_nsg_S))

                                    run_command3 = ['./NSG/build/tests/test_nsg_optimized_search', base_path,  query_path, NSG_graph_path, str(pr_L_nsg_S), str(10),
                                                    qs_indice_path, str(K), str(L), str(L_nsg_C), str(R_nsg), str(C), whole_name, index_performance_csv3]
                                    result3 = subprocess.run(run_command3, check=True, text=True, capture_output=True)

                                    gt = read_ivecs(gt_indice_path)
                                    qs = read_ivecs(qs_indice_path)

                                    rec = calculate_recall_rate(gt, qs)

                                    os.remove(qs_indice_path)  # 用完了就删除

                                    if rec >= target_recall:
                                        t6 = time.time()
                                        paras_search_time = t6 - t4

                                        #保证索引搜索时的数据是最新的
                                        df =  pd.read_csv(index_performance_csv3, sep=',', header=0)
                                        df = df.drop(df.index[-2])
                                        df.to_csv(index_performance_csv3, mode='w', header=False, index=False)

                                        recs = [(whole_name, K, L, L_nsg_C, R_nsg, C, pr_L_nsg_S, L_nsg_S, target_recall, rec, paras_search_time)]
                                    else:
                                        df = pd.read_csv(index_performance_csv3, sep=',', header=0)
                                        df = df.drop(df.index[-1])
                                        df.to_csv(index_performance_csv3, mode='w', header=False, index=False)

                                        break
                            else:
                                while (True):
                                    if (L_nsg_S < 100):
                                        dealt = 1
                                    elif (100 <= L_nsg_S < 200):
                                        dealt = 5
                                    elif (200 <= L_nsg_S < 400):
                                        dealt = 10
                                    elif (400 <= L_nsg_S < 700):
                                        dealt = 15
                                    elif (700 <= L_nsg_S < 900):
                                        dealt = 20
                                    elif (900 <= L_nsg_S < 1200):
                                        dealt = 30
                                    elif (1200 <= L_nsg_S < 1500):
                                        dealt = 50
                                    else:
                                        dealt = 100

                                    L_nsg_S = L_nsg_S + dealt

                                    qs_indice_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C, pr_L_nsg_S))

                                    run_command3 = ['./NSG/build/tests/test_nsg_optimized_search', base_path, query_path, NSG_graph_path, str(pr_L_nsg_S), str(10),
                                                    qs_indice_path, str(K), str(L), str(L_nsg_C), str(R_nsg), str(C), whole_name, index_performance_csv3]
                                    result3 = subprocess.run(run_command3, check=True, text=True, capture_output=True)

                                    gt = read_ivecs(gt_indice_path)
                                    qs = read_ivecs(qs_indice_path)

                                    rec = calculate_recall_rate(gt, qs)

                                    os.remove(qs_indice_path)  # 用完了就删除

                                    if rec >= target_recall:
                                        t6 = time.time()
                                        paras_search_time = t6 - t4

                                        # 保证索引搜索时的数据是最新的
                                        df = pd.read_csv(index_performance_csv3, sep=',', header=0)
                                        df = df.drop(df.index[-2])
                                        df.to_csv(index_performance_csv3, mode='w', header=False, index=False)

                                        recs = [(whole_name, K, L, L_nsg_C, R_nsg, C, pr_L_nsg_S, L_nsg_S, target_recall, rec, paras_search_time)]

                                        break
                                    else:
                                        df = pd.read_csv(index_performance_csv3, sep=',', header=0)
                                        df = df.drop(df.index[-1])
                                        df.to_csv(index_performance_csv3, mode='w', header=False, index=False)

                            rec_df = pd.DataFrame(recs, columns=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'pr_L_nsg_S', 'real_L_nsg_S', 'target_recall', 'real_recall', 'paras_search_time'])
                            rec_df.to_csv(index_performance_csv4, mode='a', header=False, index=False)

                            if NSG_graph_time < 1200:
                                os.remove(NSG_graph_path) #NSG构建时间较短就删除

                            if KNN_graph_time < 1200:
                                os.remove(KNN_graph_path) #KNNG构建时间较短就删除

    
# df_KNNG = pd.read_csv(index_performance_csv1, sep=',', header=0)
# df_NSG = pd.read_csv(index_performance_csv2, sep=',', header=0)
# df_Search = pd.read_csv(index_performance_csv3, sep=',', header=0)
# df_Search_Recall = pd.read_csv(index_performance_csv4, sep=',', header=0)
#
# df_merged = pd.merge(df_KNNG, df_NSG, on=['FileName', 'K', 'L'], how='left')
#
# # 合并上一步的结果与 C
# df_merged = pd.merge(df_merged, df_Search, on=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C'], how='left')
#
# # 最后将得到的结果与 D 合并
# df_merged = pd.merge(df_merged, df_Search_Recall, on=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S'], how='left')
#
# df_merged.to_csv(whole_index_performance_csv, mode='w', header=True, index=False)
#'''











