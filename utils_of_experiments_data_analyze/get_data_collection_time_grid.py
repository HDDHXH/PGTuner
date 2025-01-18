import os
import sys
sys.path.append('../')
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_whole(row):
    data_size_dic = {'glove_1_1.0_100': [1e6, 1e4], 'glove_1_1.183514_100': [1183514, 1e4], 'paper_1_2.0_200': [2e6, 1e4], 'paper_1_2.029997_200': [2029997, 1e4], 'sift_1_1_128': [1e6, 1e4], 
                    'sift_2_5_128': [5e7, 1e4], 'crawl_1_1.9_300': [1900000, 1e4], 'crawl_1_1.989995_300': [1989995, 1e4], 'msong_0_9.0_420': [900000, 200], 'msong_0_9.92272_420': [992272, 200],
                    'gist_0_9.0_960': [900000, 1e3], 'gist_1_1.0_960': [1e6, 1e3], 'deep_2_1_96': [1e7, 1e4]}

    for key, value in data_size_dic.items():
        if key in row['FileName']:
            row['whole_search_time'] =row['search_time'] * value[1]

            break
    return row


def get_train_data_collection_time(df):
    train_time = {}

    groups = df.groupby('FileName')

    for filename, group in groups:
        train_time[filename] = []

        new_group = group.drop_duplicates(subset=['efConstruction', 'M'], keep='first')

        all_ct = sum(new_group['construction_time'].unique().tolist())
        all_st = group['whole_search_time'].sum()
        all_t = all_ct + all_st

        train_time[filename] = [all_ct, all_st, all_t]

    print(train_time)

    time_lis = list(train_time.values())

    all_ct = 0
    all_st = 0

    for tup in time_lis:
        ct, st, _ = tup
        all_ct += ct
        all_st += st

    all_t = all_ct + all_st
    print(f'总构建时间：{all_ct}')
    print(f'总搜索时间：{all_st}')
    print(f'总时间：{all_t}')


def get_grid_data_collection_time(df):  # 获取每个目标召回率下网格搜索花费的时间
    # recall_thresholds = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    recall_thresholds = [0.95, 0.99]

    real_test_time = {}

    groups = df.groupby('FileName')
    for filename, group in groups:
        real_test_time[filename] = {}

        new_group = group.drop_duplicates(subset=['efConstruction', 'M'], keep='first')

        all_ct = sum(new_group['construction_time'].unique().tolist())
        all_st = {}
        for tr in recall_thresholds:
            all_st[tr] = 0

        sub_groups = group.groupby(['efConstruction', 'M'])

        # 遍历每个子组
        for (efConstruction, M), sub_group in sub_groups:
            # 按 efSearch 从小到大排序
            sub_group = sub_group.sort_values(by='efSearch')

            # 找到 recall 大于等于目标阈值的第一行
            for tr in recall_thresholds:
                target_rows = sub_group[sub_group['recall'] >= tr]

                if not target_rows.empty:
                    # 如果找到符合条件的行
                    target_row = target_rows.iloc[0]
                    # 计算从第一行到目标行的 whole_search_time 总和
                    total_whole_search_time = sub_group.loc[:target_row.name, 'whole_search_time'].sum()
                    all_st[tr] += total_whole_search_time
                else:
                    # 如果没有找到符合条件的行，计算整个子组的 whole_search_time 总和
                    total_whole_search_time = sub_group['whole_search_time'].sum()
                    all_st[tr] += total_whole_search_time

        for tr in recall_thresholds:
            all_t = all_ct + all_st[tr]
            real_test_time[filename][tr] = [all_ct, all_st[tr], all_t]

    print(real_test_time)

current_directory = os.getcwd()  # 返回当前工作目录的路径
parent_directory = os.path.dirname(current_directory)

metrics = ['ct_dec', 'st_dec', 'cdc_dec', 'sdc_dec', 'ot_dec', 'odc_dec']
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown']


filename_dic = {'deep1':'deep_1_1_96_1', 'sift1':'sift_1_1_128_1', 'glove': 'glove_1_1.183514_100', 'paper':'paper_1_2.029997_200', 'crawl':'crawl_1_1.989995_300', 'msong':'msong_0_9.92272_420',
                    'gist':'gist_1_1.0_960', 'deep10':'deep_2_1_96', 'sift50':'sift_2_5_128_1', 'deep2':'deep_1_2_96_1', 'deep3':'deep_1_3_96_1', 'deep4':'deep_1_4_96_1', 'deep5':'deep_1_5_96_1',
                    'sift2':'sift_1_2_128_1', 'sift3':'sift_1_3_128_1', 'sift4':'sift_1_4_128_1', 'sift5':'sift_1_5_128_1', 'gist_25':'gist_1_1.0_960_25', 'gist_50':'gist_1_1.0_960_50',
                    'gist_75':'gist_1_1.0_960_75', 'gist_100':'gist_1_1.0_960_100', 'deep2_25':'deep_1_2_96_1_25', 'deep2_50':'deep_1_2_96_1_50', 'deep2_75':'deep_1_2_96_1_75', 'deep2_100':'deep_1_2_96_1_100'}



root_path = os.path.join(parent_directory, 'Data')

train_data_path = os.path.join(parent_directory, 'Data/train_data.csv')
test_data_main_path = os.path.join(parent_directory, 'Data/test_data_main.csv')
test_data_ds_change_path = os.path.join(parent_directory, 'Data/test_data_ds_change.csv')
test_data_qw_change_path = os.path.join(parent_directory, 'Data/test_data_qw_change.csv')

# test_data_qw_change_test_path  = os.path.join(parent_directory, 'Data/index_performance_qw_change_test.csv')
test_data_ds_change_sift2_path = os.path.join(parent_directory, 'Data/index_performance_ds_change_sift2.csv')

for dataset_name in ['sift3', 'sift4', 'sift5']: #'sift50', 'deep10', 'glove', 'gist'
    filename = filename_dic[dataset_name]

    #selected_config_path = "../Data/active_learning_data/{}_selected_config_{}_{}.csv".format(dataset_name, 2, 7)
    selected_config_path = "../Data/active_learning_data/ds_change/{}_selected_config_{}_{}.csv".format(dataset_name, 2, 7)
    #selected_config_path = "../Data/active_learning_data/qw_change/{}_selected_config_{}_{}.csv".format(dataset_name, 2, 7)
    #selected_config_path = "../Data/active_learning_data/dast_change/order4/{}_selected_config_{}_{}.csv".format(dataset_name, 2, 7)
    #selected_config_path = "../Data/selected_config_GMM.csv"

    # train_df = pd.read_csv(train_data_path, sep=',', header=0)
    #real_test_df = pd.read_csv(test_data_main_path, sep=',', header=0)
    real_test_df = pd.read_csv(test_data_ds_change_path, sep=',', header=0)
    #real_test_df = pd.read_csv(test_data_qw_change_path, sep=',', header=0)
    real_test_df = real_test_df[real_test_df['FileName']==filename]

    # partial_config =  [(20, 4), (20, 8), (20, 12), (20, 16), (20, 20), (40, 4), (40, 8), (40, 12), (40, 16), (40, 20), (40, 24), (40, 32), (40, 40), (60, 4), (60, 8), (60, 12),
    #                    (60, 16), (60, 20), (60, 24), (80, 4), (80, 8), (80, 12), (80, 16), (100, 4), (100, 8), (100, 12), (100, 16), (140, 4), (140, 8), (140, 12), (180, 4), (180, 8),
    #                    (180, 12), (220, 4), (220, 8), (220, 12), (260, 4), (260, 8), (260, 12), (300, 4), (300, 8), (300, 12), (340, 4), (340, 8), (340, 12), (380, 4), (380, 8), (380, 12),
    #                    (420, 4), (420, 8), (420, 12), (460, 4), (460, 8), (460, 12), (500, 4), (500, 8), (500, 12), (560, 4), (560, 8), (560, 12), (620, 4), (620, 8), (620, 12), (680, 4),
    #                    (680, 8), (680, 12), (740, 4), (740, 8), (740, 12), (800, 4), (800, 8), (800, 12)]
    # partial_config_df = pd.DataFrame(partial_config, columns=['efConstruction', 'M'])
    # partial_config_df['FileName'] = filename
    # partial_real_test_df = real_test_df.merge(partial_config_df, on=['FileName', 'efConstruction', 'M'], how='inner')
    # print(len(partial_real_test_df))

    partial_config_df = pd.read_csv(selected_config_path, sep=',', header=0)
    partial_config_df['FileName'] = filename
    partial_real_test_df = real_test_df.merge(partial_config_df, on=['FileName', 'efConstruction', 'M'], how='inner')

    #get_grid_data_collection_time(real_test_df)
    get_train_data_collection_time(partial_real_test_df)

    '''
    # para_list = []
    # groups = real_test_df.groupby(['efConstruction', 'M'])
    # for i, group in groups:
    #     if group['efSearch'].max() >= 300:
    #         para_list.append(i)
    # print(len(para_list))
    # print(para_list)

    # real_test_df_sift2 = pd.read_csv(test_data_ds_change_sift2_path, sep=',', header=0)
    # print(len(real_test_df_sift2))
    # all_df = pd.concat([partial_real_test_df, real_test_df_sift2], axis=0)
    # print(len(all_df))
    # all_df.to_csv(test_data_ds_change_sift2_path, mode='w', index=False)
    '''




#get_train_data_collection_time(train_df)
#get_grid_data_collection_time(real_test_df)
#get_train_data_collection_time(partial_real_test_df)



