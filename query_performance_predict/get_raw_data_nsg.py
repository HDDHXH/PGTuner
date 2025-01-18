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
    data_size_dic = {'paper_0_2_200_1': [2e5, 1e4], 'deep_0_1_96_1': [1e5, 1e4], 'gist_0_1_960_1': [1e5, 1e3],}

    for key, value in data_size_dic.items():
        if key in row['FileName']:
            row['average_NSG_s_dc_counts'] = int(row['NSG_s_dc_counts'] / value[1] + 1)
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

    df_train = df[df['FileName'].isin(['deep_0_1_96_1', 'paper_0_2_200_1']) ]
    df_test = df[df['FileName'].isin(['gist_0_1_960_1'])]

    df_train.to_csv(train_data_path, mode='w', index=False)
    df_test.to_csv(test_data_path, mode='w', index=False)

def get_train_test_data(data_feature_path, raw_data_path, data_path):
    df1 = pd.read_csv(data_feature_path, sep=',', header=0)
    df2 = pd.read_csv(raw_data_path, sep=',', header=0)
    print(df2.columns)
    df2 = df2.apply(calculate_average, axis=1)

    # 使用 merge 函数进行合并
    result_df = pd.merge(df1, df2, on='FileName', how='right')
    result_df.to_csv(data_path, mode='w', index=False)

def get_final_data_feature(data_feature_path1, data_feature_path2, data_feature_path3, final_data_feature_path): #分别是all_data_feature、k_neighbor_dist_feature、query_k_neighbor_dist_feature
    df1 = pd.read_csv(data_feature_path1, sep=',', header=0)
    df2 = pd.read_csv(data_feature_path2, sep=',', header=0)
    df3 = pd.read_csv(data_feature_path3, sep=',', header=0)
    # print(df2['construction_time'].describe())

    # 使用 merge 函数进行合并
    temp_result_df = pd.merge(df1, df2, on='FileName', how='right')
    result_df = pd.merge(temp_result_df, df3, on='FileName', how='right')

    result_df.to_csv(final_data_feature_path, mode='w', index=False)

def get_unlabeled_data_NSG(unlabeled_data_NSG_path):
    L_nsg_Cs = [150, 200, 250, 300, 350]
    R_nsgs = [5, 10, 15, 20, 25, 30, 35, 40, 50, 70, 90]  # K >= R_nsg
    Cs = [300, 400, 500, 600]  # C >= R_nsg

    NSG_para_list = []

    for L_nsg_C in L_nsg_Cs:
        for R_nsg in R_nsgs:
            for C in Cs:
                if R_nsg <= C:
                    NSG_para_list.append((L_nsg_C, R_nsg, C))
                else:
                    break

    df = pd.DataFrame(NSG_para_list, columns=['L_nsg_C', 'R_nsg', 'C'])
    df.to_csv(unlabeled_data_NSG_path, mode='w', index=False)

def get_unlabeled_data_KNN(unlabeled_data_KNN_path):
    Ks = [100, 200, 300, 400]
    Ls = [100, 150, 200, 250, 300, 350, 400]  # K <= L

    KNN_para_list = []
    for L in Ls:
        for K in Ks:
            if K <= L:
                KNN_para_list.append((K, L))
            else:
                break

    df = pd.DataFrame(KNN_para_list, columns=['K', 'L'])
    df.to_csv(unlabeled_data_KNN_path, mode='w', index=False)

if __name__ == '__main__':
    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

    LID_feature_path = os.path.join(parent_directory, 'Data/LID_data_feature.csv')  # 表示是1w-100w的数据，不包含右边界
    K_neighbor_feature_path = os.path.join(parent_directory, 'Data/K_neighbor_dist_feature.csv')  # 表示是1w-100w的数据，不包含右边界
    query_K_neighbor_feature_path = os.path.join(parent_directory, 'Data/query_K_neighbor_dist_ratio_feature.csv')  # 表示是1w-100w的数据，不包含右边界
    data_feature_path = os.path.join(parent_directory, 'NSG_KNNG/Data/whole_data_feature.csv')  # 表示是1w-100w的数据，不包含右边界

    data_performance_path = os.path.join(parent_directory, 'NSG_KNNG/Data/index_performance_main.csv')
    train_data_performance_path = os.path.join(parent_directory, 'NSG_KNNG/Data/index_performance_main_train.csv')
    test_data_performance_path = os.path.join(parent_directory, 'NSG_KNNG/Data/index_performance_main_test.csv')

    train_data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/train_data.csv')
    test_data_main_path = os.path.join(parent_directory, 'NSG_KNNG/Data/test_data_main.csv')


    NSG_config_unit_data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/NSGconfig_unit_data.csv')
    KNN_config_unit_data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/KNN_config_unit_data.csv')

    #get_data_feature(LID_feature_path, K_neighbor_feature_path, query_K_neighbor_feature_path, data_feature_path)
    #print(0)
    # split_train_test_main_raw_data(data_performance_path, train_data_performance_path, test_data_performance_path)
    # print(1)
    # get_train_test_data(data_feature_path, train_data_performance_path,train_data_path)
    # print(2)
    # get_train_test_data(data_feature_path, test_data_performance_path, test_data_main_path)

    get_unlabeled_data_KNN(KNN_config_unit_data_path)
    # get_unlabeled_data_NSG(NSG_config_unit_data_path)


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




    