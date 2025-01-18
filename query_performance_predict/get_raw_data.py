import os
import pandas as pd

def get_GT_data(data_path, train_data_path, GT_data_path):
    df = pd.read_csv(data_path, sep=',', header=0)

    df['FileName'] = df['FileName'].astype(str)
    df['NumElements'] = df['FileName'].apply(lambda x: len(x.split('_')))
    print(df['NumElements'].unique())

    # 分别提取元素数量为3和4的数据
    df_4_elements = df[df['NumElements'] == 4]
    df_5_elements = df[df['NumElements'] == 5]

    # 删除 NumElements 列
    df_5_elements = df_5_elements.drop(columns=['NumElements'])
    df_4_elements = df_4_elements.drop(columns=['NumElements'])

    # 将数据保存到两个不同的 CSV 文件中
    df_5_elements.to_csv(train_data_path, index=False)
    df_4_elements.to_csv(GT_data_path, index=False)

    print(len(df_5_elements))
    print(len(df_4_elements))

def calculate_average(row):
    data_size_dic = {'glove_1_1.0_100': [1e6, 1e4], 'glove_1_1.183514_100': [1183514, 1e4], 'paper_1_2.0_200': [2e6, 1e4], 'paper_1_2.029997_200': [2029997, 1e4],
                    'sift_2_5_128_1': [5e7, 1e4], 'crawl_1_1.9_300': [1900000, 1e4], 'crawl_1_1.989995_300': [1989995, 1e4], 'msong_0_9.0_420': [900000, 200], 'msong_0_9.92272_420': [992272, 200],
                    'gist_0_9.0_960': [900000, 1e3], 'gist_1_1.0_960': [1e6, 1e3], 'deep_2_1_96': [1e7, 1e4], 'deep_1_1_96_1': [1e6, 1e4], 'deep_1_2_96_1': [2e6, 1e4], 'deep_1_3_96_1': [3e6, 1e4],
                     'deep_1_4_96_1': [4e6, 1e4], 'deep_1_5_96_1': [5e6, 1e4], 'sift_1_1_128_1': [1e6, 1e4], 'sift_1_2_128_1': [2e6, 1e4], 'sift_1_3_128_1': [3e6, 1e4], 'sift_1_4_128_1': [1e6, 4e4], 'sift_1_5_128_1': [5e6, 1e4],
                     'gist_1_1.0_960_25': [1e6, 1e3], 'gist_1_1.0_960_50': [1e6, 1e3], 'gist_1_1.0_960_75': [1e6, 1e3], 'gist_1_1.0_960_100': [1e6, 1e3], 'deep_1_2_96_1_25': [2e6, 1e4], 'deep_1_2_96_1_50': [2e6, 1e4],
                     'deep_1_2_96_1_75': [2e6, 1e4], 'deep_1_2_96_1_100': [2e6, 1e4]}

    for key, value in data_size_dic.items():
        if key in row['FileName']:
            row['average_construct_dc_counts'] = int(row['construct_dc_counts'] / value[0] + 1)
            row['average_search_dc_counts'] = int(row['search_dc_counts'] / value[1] + 1)
            row['whole_search_time'] = row['search_time'] * value[1]
            break
    return row

def get_data_feature(LID_feature_path, K_neighbor_feature_path, query_K_neighbor_feature_path, data_feature_path):
    df1 = pd.read_csv(LID_feature_path, sep=',', header=0)
    df2 = pd.read_csv(K_neighbor_feature_path, sep=',', header=0)
    df3 = pd.read_csv(query_K_neighbor_feature_path, sep=',', header=0)

    df1.drop(columns=['LIDTime'], inplace = True)
    df2.drop(columns=['SearchTime'], inplace=True)
    df3.drop(columns=['q_SearchTime'], inplace=True)
    # 使用 merge 函数进行合并
    result_df = pd.merge(df1, df2, on='FileName', how='left')
    result_df = pd.merge(result_df, df3, on='FileName', how='left')
    result_df.to_csv(data_feature_path, mode='w', index=False)

def process_raw_data(data_path):
    df = pd.read_csv(data_path, sep=',', header=0)
    df = df.apply(calculate_average, axis=1)

    # 使用 merge 函数进行合并
    df.to_csv(data_path, mode='w', index=False)

def split_train_test_main_raw_data(data_path, train_data_path, test_data_path):
    df = pd.read_csv(data_path, sep=',', header=0)

    df_train = df[df['FileName'].isin(['deep_1_1_96_1', 'sift_1_1_128_1', 'paper_1_2.029997_200', 'crawl_1_1.989995_300', 'msong_0_9.92272_420']) ]
    df_test = df[df['FileName'].isin(['glove_1_1.183514_100', 'gist_1_1.0_960', 'deep_2_1_96', 'sift_2_5_128_1'])]

    df_train.to_csv(train_data_path, mode='w', index=False)
    df_test.to_csv(test_data_path, mode='w', index=False)

def get_train_test_data(data_feature_path, raw_data_path, data_path):
    df1 = pd.read_csv(data_feature_path, sep=',', header=0)
    df2 = pd.read_csv(raw_data_path, sep=',', header=0)

    df2 = df2.apply(calculate_average, axis=1)

    # 使用 merge 函数进行合并
    result_df = pd.merge(df1, df2, on='FileName', how='right')
    result_df.to_csv(data_path, mode='w', index=False)

def get_train_raw_data_new(data_path1, data_path2, train_raw_data_path):  #分别是final_data_feature_train和all_index_performance_new_count
    df1 = pd.read_csv(data_path1, sep=',', header=0)
    df2 = pd.read_csv(data_path2, sep=',', header=0)

    # 使用 merge 函数进行合并
    result_df = pd.merge(df1, df2, on='FileName', how='right')
    print(result_df.describe())

    result_df.to_csv(train_raw_data_path, mode='w', index=False)

def get_final_data_feature(data_feature_path1, data_feature_path2, data_feature_path3, final_data_feature_path): #分别是all_data_feature、k_neighbor_dist_feature、query_k_neighbor_dist_feature
    df1 = pd.read_csv(data_feature_path1, sep=',', header=0)
    df2 = pd.read_csv(data_feature_path2, sep=',', header=0)
    df3 = pd.read_csv(data_feature_path3, sep=',', header=0)
    # print(df2['construction_time'].describe())

    # 使用 merge 函数进行合并
    temp_result_df = pd.merge(df1, df2, on='FileName', how='right')
    result_df = pd.merge(temp_result_df, df3, on='FileName', how='right')

    result_df.to_csv(final_data_feature_path, mode='w', index=False)

def get_default_raw_data(raw_data_path, def_raw_data_path):
    df = pd.read_csv(raw_data_path, sep=',', header=0)

    df = df[(df['efConstruction'] == 20) & (df['M'] == 4)]
    df.to_csv(def_raw_data_path, mode='w', index=False)
    print(len(df))

def get_unlabeled_data(unlabeled_data_path):
    efCs = [20, 40, 60, 80, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 560, 620, 680, 740, 800]
    # ms = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 100]
    ms = [4, 8, 16, 24, 32, 48, 64, 80, 100]
    efSs = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 300, 340, 380, 420, 460, 500, 540, 580, 620, 660, 700, 760, 820, 880, 940, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800, 5000]
    print(len(efSs))

    para_list = []

    for efC in efCs:
        for m in ms:
            if m <= efC:
                para_list.append([efC, m])
            else:
                break

    df = pd.DataFrame(para_list, columns=['efConstruction', 'M'])
    df.to_csv(unlabeled_data_path, mode='w', index=False)


if __name__ == '__main__':
    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

    LID_feature_path = os.path.join(parent_directory, 'Data/LID_data_feature.csv')  # 表示是1w-100w的数据，不包含右边界
    K_neighbor_feature_path = os.path.join(parent_directory, 'Data/K_neighbor_dist_feature.csv')  # 表示是1w-100w的数据，不包含右边界
    query_K_neighbor_feature_path = os.path.join(parent_directory, 'Data/query_K_neighbor_dist_ratio_feature.csv')  # 表示是1w-100w的数据，不包含右边界
    data_feature_path = os.path.join(parent_directory, 'Data/whole_data_feature.csv')  # 表示是1w-100w的数据，不包含右边界

    data_performance_path = os.path.join(parent_directory, 'Data/index_performance_main.csv')
    train_data_performance_path = os.path.join(parent_directory, 'Data/index_performance_main_train.csv')
    test_data_performance_path = os.path.join(parent_directory, 'Data/index_performance_main_test.csv')
    ds_data_performance_path = os.path.join(parent_directory, 'Data/index_performance_ds_change.csv')
    ds_data_performance_sift2_path = os.path.join(parent_directory, 'Data/index_performance_ds_change_sift2.csv')
    qw_data_performance_path = os.path.join(parent_directory, 'Data/index_performance_qw_change.csv')

    train_data_path = os.path.join(parent_directory, 'Data/train_data.csv') 
    test_data_main_path = os.path.join(parent_directory, 'Data/test_data_main.csv')
    test_data_ds_change_path = os.path.join(parent_directory, 'Data/test_data_ds_change.csv')
    test_data_ds_change_sift2_path = os.path.join(parent_directory, 'Data/test_data_ds_change_sift2.csv')
    test_data_qw_change_path = os.path.join(parent_directory, 'Data/test_data_qw_change.csv')

    config_unit_data_path = os.path.join(parent_directory, 'Data/config_unit_data.csv')

    #get_data_feature(LID_feature_path, K_neighbor_feature_path, query_K_neighbor_feature_path, data_feature_path)
    #print(0)
    # split_train_test_main_raw_data(data_performance_path, train_data_performance_path, test_data_performance_path)
    # print(1)
    # get_train_test_data(data_feature_path, train_data_performance_path,train_data_path)
    # print(2)
    # get_train_test_data(data_feature_path, test_data_performance_path, test_data_main_path)
    #get_train_test_data(data_feature_path, qw_data_performance_path, test_data_qw_change_path)
    get_train_test_data(data_feature_path, ds_data_performance_sift2_path, test_data_ds_change_sift2_path)



    # df1 = pd.read_csv(train_data_path, sep=',', header=0)
    # df2 = pd.read_csv(test_data_main_path, sep=',', header=0)
    #
    # df = pd.concat([df1, df2], axis=0)
    # print(df.describe())
    #
    # groups = df.groupby('FileName')
    # for filename, group in groups:
    #     print(filename)
    #     print(group.describe())




    