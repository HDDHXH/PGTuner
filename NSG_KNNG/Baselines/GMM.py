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
    model = RandomForestRegressor(n_estimators=1000, max_depth=None, random_state=42)
    
    # 使用5折交叉验证评估模型的表现
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
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
    df['predicted_average_NSG_search_dc_counts'] = predicted_adcn

    return df

# 从预测结果中筛选出符合条件的输入数据（召回率超过目标值且搜索时间最短）
def filter_best_results(df, target_recall):
    # 筛选出召回率超过目标值的行
    filtered_df = df[df['predicted_recall_rate'] >= target_recall]

    if len(filtered_df) > 0:
        # 如果有多个符合条件的行，选取搜索时间最短的行
        best_row = filtered_df.loc[filtered_df['predicted_average_NSG_search_dc_counts'].idxmin()]

        best_K = best_row['K']
        best_L = best_row['L']
        best_L_nsg_C = best_row['L_nsg_C']
        best_R_nsg = best_row['R_nsg']
        best_C = best_row['C']
        best_L_nsg_S = best_row['L_nsg_S']
    else:
        best_row = df.loc[df['predicted_recall_rate'].idxmax()]

        best_K = best_row['K']
        best_L = best_row['L']
        best_L_nsg_C = best_row['L_nsg_C']
        best_R_nsg = best_row['R_nsg']
        best_C = best_row['C']
        best_L_nsg_S = best_row['L_nsg_S']

    return best_K , best_L, best_L_nsg_C, best_R_nsg, best_C, best_L_nsg_S

def get_input_feature(df_data_feature, df_config_unit):
    df_KNN_config = pd.read_csv('../Data/KNN_config_unit_data.csv', sep=',', header=0)

    L_nsg_Ss = np.array( [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
            430, 460, 490, 520, 550, 580, 610, 640, 670, 700, 740, 780, 820, 860, 900, 960, 1020, 1080, 1140, 1200, 1300, 1400, 1500])

    df_config_temp = pd.merge(df_KNN_config, df_config_unit, how='cross')

    df_config_whole = df_config_temp.loc[df_config_temp.index.repeat(len(L_nsg_Ss))].reset_index(drop=True)
    df_config_whole['L_nsg_S'] = np.tile(L_nsg_Ss, len(df_config_temp))

    df_feature = pd.merge(df_data_feature, df_config_whole, on='FileName', how='right')

    return df_feature

# 使用示例
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录

    filename_dic = {'deep1': 'deep_0_1_96_1', 'paper': 'paper_0_2_200_1', 'gist': 'gist_0_1_960_1'}

    data_fetaure_path = os.path.join(parent_directory, 'Data/whole_data_feature.csv') #注意，现在的whole_data_feature只包含输入特征，不包含其它特征

    train_data_path = os.path.join(parent_directory, 'Data/train_data.csv')  # 表示是1w-100w的数据，不包含右边界
    test_data_main_path = os.path.join(parent_directory, 'Data/test_data_main.csv')


    selected_config_path = os.path.join(parent_directory, 'Data/LHS_configs/selected_config_GMM.csv')
    config_unit_data_path = "../Data/NSG_config_unit_data.csv"
    #best_K , best_L, best_L_nsg_C, best_R_nsg, best_C, best_L_nsg_S
    # 设置输入特征和目标列
    input_feature_columns = ['K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']  # 替换成你的特征列
    target_column_rr = 'recall'  # 替换成你的目标列
    target_column_adcn = 'average_NSG_s_dc_counts'
    
    # 加载模型并在新数据上预测
    #target_rec_lis = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    target_rec_lis = [0.9, 0.95, 0.99]

    df_data_feature = pd.read_csv(data_fetaure_path, sep=',', header=0)

    df_train = pd.read_csv(train_data_path)
    df_test = pd.read_csv(test_data_main_path)  # 替换成你的数据路径
    #df_test = pd.read_csv(test_data_ds_change_path)
    #df_test = pd.read_csv(test_data_qw_change_path)
    df_selected_config = pd.read_csv(selected_config_path)

    result_para_dic = {}
    all_time_dic = {}

    for dataset_name in ['gist']: #'sift50', 'deep10', 'gist', 'glove'
        # 模型保存路径
        model_rr_save_path = os.path.join(current_directory, 'GMM_models/{}_model_rr_1000.pkl'.format(dataset_name))
        model_adcn_save_path = os.path.join(current_directory, 'GMM_models/{}_model_adcn_1000.pkl'.format(dataset_name))

        config_unit_data_path = "../Data/NSG_config_unit_data.csv"

        print(dataset_name)
        result_para_dic[dataset_name] = []

        filename = filename_dic[dataset_name]

        #获取更新模型的数据
        df_test_temp = df_test[df_test['FileName'] == filename]
        df_test_selected = df_test_temp.merge(df_selected_config, on=['K', 'L', 'L_nsg_C', 'R_nsg', 'C'], how='inner')
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
        df_feature_predicted = df_feature_unlabeled[['K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']]

        df_predictions = predict_with_models(df_feature_predicted, model_rr, model_adcn)
        print('测试数据预测完毕')
        t4 = time.time()

        for target_rec in target_rec_lis:
            best_K , best_L, best_L_nsg_C, best_R_nsg, best_C, best_L_nsg_S = filter_best_results(df_predictions, target_rec)
            result_para_dic[dataset_name].append([target_rec, best_K , best_L, best_L_nsg_C, best_R_nsg, best_C, best_L_nsg_S])
        t5 = time.time()
        print('参数配置推荐完毕')

        all_time_dic[dataset_name] = [t2-t1, t3-t2, t4-t3, t5-t1]  #时间从左到右分别为每个数据集的两个模型的训练时间、预测时间和参数配置推荐时间

    print(all_time_dic)
    print(result_para_dic)
    #'''


