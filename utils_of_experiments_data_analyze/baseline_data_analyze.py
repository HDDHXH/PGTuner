import os
import sys
sys.path.append('../')
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_average(row):
    data_size_dic = {'glove_1_1.0_100': [1e6, 1e4], 'glove_1_1.183514_100': [1183514, 1e4], 'paper_1_2.0_200': [2e6, 1e4], 'paper_1_2.029997_200': [2029997, 1e4],
                    'sift_2_5_128_1': [5e7, 1e4], 'sift_2_5_128': [5e7, 1e4], 'crawl_1_1.9_300': [1900000, 1e4], 'crawl_1_1.989995_300': [1989995, 1e4], 'msong_0_9.0_420': [900000, 200], 'msong_0_9.92272_420': [992272, 200],
                    'gist_0_9.0_960': [900000, 1e3], 'gist_1_1.0_960': [1e6, 1e3], 'deep_2_1_96': [1e7, 1e4], 'deep_1_1_96_1': [1e6, 1e4], 'deep_1_2_96_1': [2e6, 1e4], 'deep_1_3_96_1': [3e6, 1e4],
                     'deep_1_4_96_1': [4e6, 1e4], 'deep_1_5_96_1': [5e6, 1e4], 'sift_1_1_128_1': [1e6, 1e4], 'sift_1_2_128_1': [2e6, 1e4], 'sift_1_3_128_1': [3e6, 1e4], 'sift_1_4_128_1': [1e6, 4e4], 'sift_1_5_128_1': [5e6, 1e4],
                     'gist_1_1.0_960_25': [1e6, 1e3], 'gist_1_1.0_960_50': [1e6, 1e3], 'gist_1_1.0_960_75': [1e6, 1e3], 'gist_1_1.0_960_100': [1e6, 1e3], 'deep_1_2_96_1_25': [2e6, 1e4], 'deep_1_2_96_1_50': [2e6, 1e4],
                     'deep_1_2_96_1_75': [2e6, 1e4], 'deep_1_2_96_1_100': [2e6, 1e4]}

    for key, value in data_size_dic.items():
        if key in row['FileName']:
            # row['average_construct_dc_counts'] = int(row['construct_dc_counts'] / value[0] + 1)
            # row['average_search_dc_counts'] = int(row['search_dc_counts'] / value[1] + 1)
            row['whole_search_time'] = row['search_time'] * value[1]
            break
    return row

def process_raw_data(data_path):
    df = pd.read_csv(data_path, sep=',', header=0)
    df = df.apply(calculate_average, axis=1)

    # 使用 merge 函数进行合并
    df.to_csv(data_path, mode='w', index=False)

def get_best_para_by_tr_VDTuner(df): #获取VDTuner, random2, Ottertune等baseline的每个目标召回率的最佳参数
    # 创建一个空的DataFrame来存储结果
    result = pd.DataFrame()
    para_dic = {}

    # 根据FileName和target_recall进行分组
    for (filename, target_recall), group in df.groupby(['FileName', 'target_recall']):
        # 筛选出recall >= target_recall的行
        filtered = group[group['recall'] >= target_recall]
        # 如果存在符合条件的行，找到search_time最小的那一行并添加到结果DataFrame中
        if not filtered.empty:
            best_row = filtered.loc[filtered['search_time'].idxmin()]
            best_df = pd.DataFrame(best_row).transpose()
            result = pd.concat([result, best_df], ignore_index=True)

    # 重置索引
    result = result.reset_index(drop=True)

    para_df = result[['FileName', 'target_recall', 'efConstruction', 'M', 'efSearch']]
    groups = para_df.groupby('FileName')
    for filename, group in groups:
        para_dic[filename] = group[['target_recall', 'efConstruction', 'M', 'efSearch']].values.tolist()

    print(para_dic)

def get_best_para_by_tr_grid(df): #获取网格搜索, random1等baseline的每个目标召回率的最佳参数
    # 创建一个空的DataFrame来存储结果
    #recall_thresholds = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    recall_thresholds = [0.95]
    result = pd.DataFrame()
    para_dic = {}

    grouped = df.groupby('FileName')

    # 遍历每个分组
    for filename, group in grouped:
        # 遍历每个 recall 阈值
        for recall_threshold in recall_thresholds:
            # 根据 recall 阈值筛选数据
            filtered_data = group[group['recall'] >= recall_threshold]

            # 如果筛选后的数据非空，获取 search_time 最小的行
            if not filtered_data.empty:
                min_search_time_row = filtered_data.loc[filtered_data['search_time'].idxmin()]
                min_search_time_df = pd.DataFrame(min_search_time_row).transpose()
                min_search_time_df['target_recall'] = recall_threshold

                result = pd.concat([result, min_search_time_df], ignore_index=True)

    result = result.reset_index(drop=True)

    para_df = result[['FileName', 'target_recall', 'efConstruction', 'M', 'efSearch']]
    groups = para_df.groupby('FileName')
    for filename, group in groups:
        para_dic[filename] = group[['target_recall', 'efConstruction', 'M', 'efSearch']].values.tolist()

    print(para_dic)


def get_data_collection_time_VDTuner(df): #获取VDTuner, random2, Ottertune等baseline的每个目标召回率下网格搜索花费的时间
    #df['whole_search_time'] = df['search_time'] * 10000

    real_test_time = {}

    groups = df.groupby('FileName')
    for filename, group in groups:
        real_test_time[filename] = {}

        sub_groups = group.groupby('target_recall')

        for target_recall, new_group in sub_groups:
            all_t = sum(new_group['construction_time'].tolist()) + sum(new_group['whole_search_time'].tolist())
            real_test_time[filename][target_recall] = all_t

            
    print( real_test_time)

def get_data_collection_time_grid(df): #获取网格搜索, random1等baseline的每个目标召回率下网格搜索花费的时间
    #recall_thresholds = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    recall_thresholds = [0.95]

    real_test_time = {}

    groups = df.groupby('FileName')
    for filename, group in groups:
        real_test_time [filename] = {}

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
            
    print( real_test_time)

def draw_main_expeirments(df):
    grouped = df.groupby('FileName')

    # 遍历每个分组
    for filename, group in grouped:
        # 创建一个新的画布，包含两个子图
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # 绘制 qps 的条形图
        sns.barplot(ax=axes[0], x='target_recall', y='qps', hue='data_source', data=group)
        axes[0].set_title(f'{name} - QPS vs Target Recall')
        axes[0].set_xlabel('Target Recall')
        axes[0].set_ylabel('QPS')

        # 绘制 tune_time 的条形图
        sns.barplot(ax=axes[1], x='target_recall', y='tune_time', hue='data_source', data=group)
        axes[1].set_title(f'{name} - Tune Time vs Target Recall')
        axes[1].set_xlabel('Target Recall')
        axes[1].set_ylabel('Tune Time')

        # 调整图例显示在图像外部
        axes[0].legend(loc='upper right')
        axes[1].legend(loc='upper right')

        # 调整布局，确保子图不会重叠
        plt.tight_layout()

        # 保存图片，以FileName命名文件
        plt.savefig(f'{name}_qps_tune_time.png')

        # 关闭当前画布，防止下次循环时覆盖
        plt.close()

def calculate_improvement(df):
    df['qps'] = (1 / df['search_time'] + 0.5).astype(int)
    df['tune_time'] = df['tune_time'] / 3600

    grouped = df.groupby('FileName')

    for filename, group in grouped:
        print(filename)

        group_grid = group[group['data_source'] == 'grid' ]
        group_HDVITune = group[group['data_source'] == 'PGTuner']
        group_VDTuner = group[group['data_source'] == 'VDTuner']
        group_random1 = group[group['data_source'] == 'random1']
        group_random2 = group[group['data_source'] == 'random2']

        group_PGTuner.columns = ['FileName','construction_time_P','memory_P','target_recall', 'real_recall_P',
                                    'search_time_P', 'tune_time_P','data_source_P', 'qps_P']

        group_VDTuner.columns = ['FileName','construction_time_V','memory_V','target_recall', 'real_recall_V',
                                    'search_time_V', 'tune_time_V','data_source_V', 'qps_V']

        group_random1.columns = ['FileName','construction_time_R1','memory_R1','target_recall', 'real_recall_R1',
                                    'search_time_R1', 'tune_time_R1','data_source_R1', 'qps_R1']

        group_random2.columns = ['FileName','construction_time_R2','memory_R2','target_recall', 'real_recall_R2',
                                    'search_time_R2', 'tune_time_R2','data_source_R2', 'qps_R2']

        all_group_P2G = pd.merge(group_grid, group_PGTuner, on=['FileName', 'target_recall'], how='right')
        all_group_P2V = pd.merge(group_PGTuner, group_VDTuner, on=['FileName', 'target_recall'], how='right')
        all_group_P2R1 = pd.merge(group_PGTuner, group_random1, on=['FileName', 'target_recall'], how='right')
        all_group_P2R2 = pd.merge(group_PGTuner, group_random2, on=['FileName', 'target_recall'], how='right')

        all_group_P2G['qps_inc'] = (all_group_P2G['qps_P'] - all_group_P2G['qps']) / all_group_P2G['qps'] * 100
        all_group_P2G['tune_time_speedup'] = all_group_P2G['tune_time'] / all_group_P2G['tune_time_P']

        all_group_P2V['qps_inc'] = (all_group_P2V['qps_P'] - all_group_P2V['qps_V']) / all_group_P2V['qps_V'] * 100
        all_group_P2V['tune_time_speedup'] = all_group_P2V['tune_time_V'] / all_group_P2V['tune_time_P']

        all_group_P2R1['qps_inc'] = (all_group_P2R1['qps_P'] - all_group_P2R1['qps_R1']) / all_group_P2R1['qps_R1'] * 100
        all_group_P2R1['tune_time_speedup'] = all_group_P2R1['tune_time_R1'] / all_group_P2R1['tune_time_P']

        all_group_P2R2['qps_inc'] = (all_group_P2R2['qps_P'] - all_group_P2R2['qps_R2']) / all_group_P2R2['qps_R2'] * 100
        all_group_P2R2['tune_time_speedup'] = all_group_P2R2['tune_time_R2'] / all_group_P2R2['tune_time_P']

        print(all_group_P2G[['target_recall', 'qps_inc', 'tune_time_speedup']])
        print(all_group_P2V[['target_recall', 'qps_inc', 'tune_time_speedup']])
        print(all_group_P2R1[['target_recall', 'qps_inc', 'tune_time_speedup']])
        print(all_group_P2R2[['target_recall', 'qps_inc', 'tune_time_speedup']])
    
if __name__ == '__main__':
    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)

    #获取不同baseline对应的最佳参数及其性能数据和调优时间
    baseline_data_path_grid = os.path.join(parent_directory, 'Data/index_performance_main_test.csv')
    baseline_data_path_VDTuner = os.path.join(parent_directory, 'Data/experiments_results/test_main/index_performance_VDTuner.csv')
    baseline_data_qw_change_path_VDTuner = os.path.join(parent_directory, 'Data/experiments_results/test_qw_change/index_performance_VDTuner.csv')
    baseline_data_path_random1 = os.path.join(parent_directory, 'Data/experiments_results/test_main/index_performance_random1.csv')
    baseline_data_qw_change_path_random1 = os.path.join(parent_directory, 'Data/experiments_results/test_qw_change/index_performance_qw_change_random1.csv')
    baseline_data_qw_change_path_grid = os.path.join(parent_directory, 'Data/experiments_results/test_qw_change/index_performance_qw_change.csv')


    baseline_ds_change_data_path_grid = os.path.join(parent_directory, 'Data/index_performance_ds_change.csv')
    baseline_ds_change_data_sift2_path_grid = os.path.join(parent_directory, 'Data/index_performance_ds_change_sift2.csv')

    test_data_qw_change_test_path = os.path.join(parent_directory, 'Data/experiments_results/test_qw_change/index_performance_qw_change_test.csv')

    #process_raw_data(baseline_data_path_VDTuner)
    #process_raw_data(baseline_data_qw_change_path_grid)
    #process_raw_data(baseline_ds_change_data_sift2_path_grid)


    # Grid
    # print('Grid:')
    # baseline_grid_df = pd.read_csv(baseline_data_qw_change_path_grid, sep=',', header=0)
    # _ = get_best_para_by_tr_grid(baseline_grid_df.copy())
    # get_data_collection_time_grid(baseline_grid_df)

    #VDTuner
    # print('VDTuner:')
    baseline_VDTuner_df = pd.read_csv(baseline_data_path_VDTuner, sep=',', header=0)
    get_best_para_by_tr_VDTuner(baseline_VDTuner_df.copy())
    get_data_collection_time_VDTuner(baseline_VDTuner_df)

    #random2
    # print('random2:')
    # baseline_random2_df = pd.read_csv(baseline_data_path_random2, sep=',', header=0)
    # _ = get_best_para_by_tr_VDTuner(baseline_random2_df.copy())
    # get_data_collection_time_VDTuner(baseline_random2_df)

    #random1
    #print('random1:')
    # baseline_random1_df = pd.read_csv(baseline_data_qw_change_path_random1, sep=',', header=0)
    # get_best_para_by_tr_grid(baseline_random1_df.copy())
    # get_data_collection_time_grid(baseline_random1_df)

    #获取主实验的实验结果
    # main_experiment_data_path = os.path.join(parent_directory, 'experiments_results/main experiment/verify_real_test_compare_test.csv')
    # main_experiment_df = pd.read_csv(main_experiment_data_path, sep=',', header=0)
    #
    # draw_main_expeirments(main_experiment_df.copy())
    # calculate_improvement(main_experiment_df)
