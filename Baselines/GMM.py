# 导入必要的库
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib  # 用于保存和加载模型
import time
import random
# 假设你的DataFrame是df，目标变量是'target_column'，输入特征是其他列
def train_random_forest(df, target_column, feature_columns, model_save_path):
    # 提取输入和输出
    X = df[feature_columns]  # 输入特征
    y = df[target_column]    # 输出目标
    
    # 创建随机森林模型，树的深度为None，使用MSE作为指标
    model = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=42)
    
    # 使用5折交叉验证评估模型的表现
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mean_cv_score = cv_scores.mean()  # 获取交叉验证的平均得分
    
    # 输出交叉验证的平均得分
    print(f"Cross-Validation Mean Squared Error (negative): {mean_cv_score}")
    
    # 在整个训练集上训练模型
    model.fit(X, y)
    
    # 保存训练好的模型
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

def calculate_error(df, target_column, feature_columns, model):
    # 提取输入和输出
    X = df[feature_columns]  # 输入特征
    y = df[target_column]  # 输出目标

    # 创建随机森林模型，树的深度为None，使用MSE作为指标
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100

    print(f"mae:{mae}, mape:{mape}")
    
def predict_with_models(df, model_rr, model_adcn):
    # 预测召回率和搜索时间
    predicted_rr = model_rr.predict(df)
    predicted_adcn = model_adcn.predict(df)

    # 将预测结果添加到原始数据中
    df['predicted_recall_rate'] = predicted_rr
    df['predicted_average_search_dc_counts'] = predicted_adcn

    return df

# 从预测结果中筛选出符合条件的输入数据（召回率超过目标值且搜索时间最短）
def filter_best_results(df, target_recall):
    # 筛选出召回率超过目标值的行
    filtered_df = df[df['predicted_recall_rate'] >= target_recall]

    if len(filtered_df) > 0:
        # 如果有多个符合条件的行，选取搜索时间最短的行
        best_row = filtered_df.loc[filtered_df['predicted_average_search_dc_counts'].idxmin()]

        best_efC = best_row['efConstruction']
        best_m = best_row['M']
        best_efS = best_row['efSearch']
    else:
        best_row = df.loc[df['predicted_recall_rate'].idxmax()]

        best_efC = best_row['efConstruction']
        best_m = best_row['M']
        best_efS = best_row['efSearch']

    return best_efC, best_m, best_efS

def get_input_feature(df_data_feature, df_config_unit):
    efSs = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640,
                     660, 680, 700, 730, 760, 790, 820, 850, 880, 910, 940, 970, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 
                     3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500,4600, 4700, 4800, 4900, 5000])

    df_config_whole = df_config_unit.loc[df_config_unit.index.repeat(len(efSs))].reset_index(drop=True)
    df_config_whole['efSearch'] = np.tile(efSs, len(df_config_unit))

    df_feature = pd.merge(df_data_feature, df_config_whole, on='FileName', how='right')

    return df_feature

# 使用示例
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

    filename_dic = {'deep1': 'deep_1_1_96_1', 'sift1': 'sift_1_1_128_1', 'glove': 'glove_1_1.183514_100',
                    'paper': 'paper_1_2.029997_200', 'crawl': 'crawl_1_1.989995_300', 'msong': 'msong_0_9.92272_420',
                    'gist': 'gist_1_1.0_960', 'deep10': 'deep_2_1_96', 'sift50': 'sift_2_5_128_1',
                    'deep2': 'deep_1_2_96_1', 'deep3': 'deep_1_3_96_1', 'deep4': 'deep_1_4_96_1',
                    'deep5': 'deep_1_5_96_1', 'sift2': 'sift_1_2_128_1', 'sift3': 'sift_1_3_128_1',
                    'sift4': 'sift_1_4_128_1',
                    'sift5': 'sift_1_5_128_1', 'gist_25': 'gist_1_1.0_960_25', 'gist_50': 'gist_1_1.0_960_50',
                    'gist_75': 'gist_1_1.0_960_75', 'gist_100': 'gist_1_1.0_960_100', 'deep2_25': 'deep_1_2_96_1_25',
                    'deep2_50': 'deep_1_2_96_1_50', 'deep2_75': 'deep_1_2_96_1_75', 'deep2_100': 'deep_1_2_96_1_100'}

    data_fetaure_path = os.path.join(parent_directory, 'Data/whole_data_feature.csv') #注意，现在的whole_data_feature只包含输入特征，不包含其它特征

    train_data_path = os.path.join(parent_directory, 'Data/train_data.csv')  # 表示是1w-100w的数据，不包含右边界
    test_data_main_path = os.path.join(parent_directory, 'Data/test_data_main.csv')
    test_data_ds_change_path = os.path.join(parent_directory, 'Data/test_data_ds_change.csv')
    test_data_qw_change_path = os.path.join(parent_directory, 'Data/test_data_qw_change.csv')

    selected_config_path = os.path.join(parent_directory, 'Data/selected_config_GMM.csv')
    config_unit_data_path = "../Data/config_unit_data.csv"

    # 设置输入特征和目标列
    input_feature_columns = ['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']  # 替换成你的特征列
    target_column_rr = 'recall'  # 替换成你的目标列
    target_column_adcn = 'average_search_dc_counts'
    
    # 加载模型并在新数据上预测
    # target_rec_lis = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    target_rec_lis = [0.95]

    df_data_feature = pd.read_csv(data_fetaure_path, sep=',', header=0)

    df_train = pd.read_csv(train_data_path)
    #df_test = pd.read_csv(test_data_main_path)  # 替换成你的数据路径
    #df_test = pd.read_csv(test_data_ds_change_path)
    df_test = pd.read_csv(test_data_qw_change_path)
    df_selected_config = pd.read_csv(selected_config_path)

    result_para_dic = {}
    all_time_dic = {}

    for dataset_name in ['gist_25', 'gist_50', 'gist_75']: #'sift50', 'deep10', 'gist', 'glove'
        # 模型保存路径
        model_rr_save_path = os.path.join(current_directory, 'GMM_models/{}_model_rr_500.pkl'.format(dataset_name))
        model_adcn_save_path = os.path.join(current_directory, 'GMM_models/{}_model_adcn_500.pkl'.format(dataset_name))

        if dataset_name == 'sift50':
            config_unit_data_path = "../Data/config_unit_data_sift.csv"
        else:
            config_unit_data_path = "../Data/config_unit_data.csv"

        print(dataset_name)
        result_para_dic[dataset_name] = []

        filename = filename_dic[dataset_name]

        #获取更新模型的数据
        df_test_temp = df_test[df_test['FileName'] == filename]
        df_test_selected = df_test_temp.merge(df_selected_config, on=['efConstruction', 'M'], how='inner')
        temp_train_df = pd.concat([df_train, df_test_selected], axis=0)

        #'''
        # 训练模型并保存
        print('开始训练模型')
        t1 = time.time()
        train_random_forest(temp_train_df, target_column_rr, input_feature_columns, model_rr_save_path)
        print('rr预测模型训练完毕')
        t2 = time.time()
        train_random_forest(temp_train_df, target_column_adcn, input_feature_columns, model_adcn_save_path)
        print('adcn预测模型训练完毕')
        t3 = time.time()
        
        # 加载模型，生成待预测的所有候选参数配置
        model_rr = joblib.load(model_rr_save_path)
        model_adcn = joblib.load(model_adcn_save_path)

        print('rr预测误差:')
        calculate_error(df_test_temp, target_column_rr, input_feature_columns, model_rr)
        print('adcn预测误差:')
        calculate_error(df_test_temp, target_column_adcn, input_feature_columns, model_adcn)

        df_data_feature_test = df_data_feature[df_data_feature['FileName'] == filename]

        df_config_unit = pd.read_csv(config_unit_data_path, sep=',', header=0)
        df_config_unit['FileName'] = filename

        df_feature_unlabeled = get_input_feature(df_data_feature_test, df_config_unit)
        df_feature_predicted = df_feature_unlabeled[['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']]

        df_predictions = predict_with_models(df_feature_predicted, model_rr, model_adcn)
        print('测试数据预测完毕')
        t4 = time.time()

        for target_rec in target_rec_lis:
            best_efC, best_m, best_efS = filter_best_results(df_predictions, target_rec)
            result_para_dic[dataset_name].append([target_rec, best_efC, best_m, best_efS])
        t5 = time.time()
        print('参数配置推荐完毕')

        all_time_dic[dataset_name] = [t2-t1, t3-t2, t4-t3, t5-t1]  #时间从左到右分别为每个数据集的两个模型的训练时间、预测时间和参数配置推荐时间

    print(all_time_dic)
    print(result_para_dic)
    #'''


