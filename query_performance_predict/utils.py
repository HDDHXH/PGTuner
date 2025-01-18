import numpy as np
import torch
import pandas as pd
# import cupy as cp
# from cuml.neighbors import NearestNeighbors
import random
from torch.utils.data import Dataset
from Args import args

random.seed(args.seed)
np.random.seed(args.seed)

'''
----------------------读取datafram数据---------------------------------
'''
def read_data(df):
    df['Level'] = np.log10(df['SIZE'] / 1e4).astype(int)
    df['Num'] = df['SIZE'] / (1e4 * 10 ** df['Level'])
    df_f = df[['FileName', 'efConstruction', 'M', 'Level', 'Num', 'DIM', 'LID', 'ClustersNum', 'MeanDist', 'StdDist']]
    # df_f = df[['FileName', 'efConstruction', 'M', 'Level', 'Num', 'DIM', 'LID']]
    df_p = df[['recall', 'construction_time', 'qps']]
    return df_f, df_p

def read_data2(df):
    df['Num'] = np.log10(df['SIZE'] / 1e4)
    df_f = df[['FileName', 'efConstruction', 'M', 'Num', 'DIM', 'LID', 'ClustersNum', 'MeanDist', 'StdDist']]
    # df_f = df[['FileName', 'efConstruction', 'M', 'Level', 'Num', 'DIM', 'LID']]
    df_p = df[['recall', 'construction_time', 'qps']]
    return df_f, df_p

def read_data_new(df):
    # df_f = df[['FileName', 'efConstruction', 'M', 'efS', 'Num', 'q_Num', 'DIM', 'LID', 'K_MinDist', 'K_MaxDist', 'K_StdDist', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist',
    #            'q_K_MinDist', 'q_K_MaxDist', 'q_K_StdDist', 'q_Sum_K_MinDist', 'q_Sum_K_MaxDist', 'q_Sum_K_StdDist']]
    df_f = df[['FileName', 'efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']]
    df_p = df[['recall', 'average_construct_dc_counts', 'average_search_dc_counts']]
    return df_f, df_p

def read_data_new_nsg(df):
    # df_f = df[['FileName', 'efConstruction', 'M', 'efS', 'Num', 'q_Num', 'DIM', 'LID', 'K_MinDist', 'K_MaxDist', 'K_StdDist', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist',
    #            'q_K_MinDist', 'q_K_MaxDist', 'q_K_StdDist', 'q_Sum_K_MinDist', 'q_Sum_K_MaxDist', 'q_Sum_K_StdDist']]
    df_f = df[['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']]
    df_p = df[['recall', 'average_NSG_s_dc_counts']]
    return df_f, df_p

def read_data_new_ct(df):
    # df_f = df[['FileName', 'efConstruction', 'M', 'efS', 'Num', 'q_Num', 'DIM', 'LID', 'K_MinDist', 'K_MaxDist', 'K_StdDist', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist',
    #            'q_K_MinDist', 'q_K_MaxDist', 'q_K_StdDist', 'q_Sum_K_MinDist', 'q_Sum_K_MaxDist', 'q_Sum_K_StdDist']]
    df_f = df[['FileName', 'efConstruction', 'M', 'SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']]
    df_p = df[['average_construct_dc_counts']]
    return df_f, df_p

def read_unlabeld_data(df):
    df['Num'] = np.log10(df['SIZE'] / 1e4)
    df_f = df[['FileName', 'efConstruction', 'M', 'Num', 'DIM', 'LID', 'ClustersNum', 'MeanDist', 'StdDist']]

    return df_f

def read_unlabeld_data_new(df):
    # df_f = df[['FileName', 'efConstruction', 'M', 'efS', 'Num', 'q_Num', 'DIM', 'LID', 'K_MinDist', 'K_MaxDist', 'K_StdDist', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist',
    #            'q_K_MinDist', 'q_K_MaxDist', 'q_K_StdDist', 'q_Sum_K_MinDist', 'q_Sum_K_MaxDist', 'q_Sum_K_StdDist']]
    df_f = df[['FileName', 'efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist',
               'Sum_K_StdDist', 'q_Sum_K_MinDist', 'q_Sum_K_MaxDist', 'q_Sum_K_StdDist']]

    return df_f

'''
----------------------划分数据，构建训练、验证和测试集---------------------------------
'''
def get_dataset(file_path):
    def split_df(group, train_frac=0.8, val_frac=0.1):
        shuffled_indices = np.random.permutation(len(group))
        train_end = int(len(group) * train_frac)
        val_end = int(len(group) * (train_frac + val_frac))

        if train_end % 2 == 0:
            train_indices = shuffled_indices[:train_end]
            val_indices = shuffled_indices[train_end:val_end]
        else:
            train_indices = shuffled_indices[:train_end+1]
            val_indices = shuffled_indices[train_end+1:val_end]
        test_indices = shuffled_indices[val_end:]

        return {
            'train': group.iloc[train_indices],
            'valid': group.iloc[val_indices],
            'test': group.iloc[test_indices]
        }

    df = pd.read_csv(file_path, sep=',', header=0)
    print(f'数据集大小：{len(df)}')
    # df = df.head(100)

    # grouped = df.groupby(['DIM', 'LID', 'ClustersNum', 'MeanDist', 'StdDist'])
    # result = grouped.apply(split_df)
    #
    # df_train = pd.concat([group['train'] for group in result])
    # df_valid = pd.concat([group['valid'] for group in result])
    # df_test = pd.concat([group['test'] for group in result])

    result = split_df(df)

    df_train = result['train']
    df_valid = result['valid']
    df_test = result['test']

    return df_train, df_valid, df_test

def split_data(df):
    def split_df(group, train_frac=0.8, val_frac=0.1):
        shuffled_indices = np.random.permutation(len(group))
        train_end = int(len(group) * train_frac)
        val_end = int(len(group) * (train_frac + val_frac))

        if train_end % 2 == 0:
            train_indices = shuffled_indices[:train_end]
            val_indices = shuffled_indices[train_end:val_end]
        else:
            train_indices = shuffled_indices[:train_end+1]
            val_indices = shuffled_indices[train_end+1:val_end]
        test_indices = shuffled_indices[val_end:]

        return {
            'train': group.iloc[train_indices],
            'valid': group.iloc[val_indices],
            'test': group.iloc[test_indices]
        }

    result = split_df(df)

    df_train = result['train']
    df_valid = result['valid']
    df_test = result['test']

    return df_train, df_valid, df_test

'''
----------------------datafram转numpy，numpy转tensor---------------------------------
'''
def df2np(df):
    np_data = df.to_numpy()
    return np_data

def np2ts(np_data):
    ts_data = torch.tensor(np_data).to(torch.float32)
    return ts_data

'''
----------------------创建dataset类---------------------------------
'''
class CustomDataset(Dataset):
    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def __len__(self):
        return len(self.tensor1)  # 假设张量1和张量2的长度相同

    def __getitem__(self, index):
        return self.tensor1[index], self.tensor2[index]

# dataset = CustomDataset(ts_data1, ts_data2)
# dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

class CustomDataset2(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)  # 假设张量1和张量2的长度相同

    def __getitem__(self, index):
        return self.tensor[index]



'''
----------------------创建归一化类---------------------------------
'''
#计划对输入向量前efC、M、Size、Dim这两个维度采用MinMax归一化，其它维度采用Standard归一化；对性能向量QPS采用Standard归一化， 构建时间采用MinMax归一化。
class Scaler_raw:
    def __init__(self, num):  #num为5或2
        self.num = num

        self.min = None
        self.max = None
        self.mean = None
        self.std = None

        if self.num == 5:
            self.min = np.array([20, 6, 0, 1, 100])
            self.max = np.array([300, 100, 4, 10, 1000])
        else:
            self.min = np.array([0])
            self.max = np.array([7200])

    def fit(self, data):
        self.mean = np.mean(data[:, self.num:], axis=0)
        self.std = np.std(data[:, self.num:], axis=0)

    def transform(self, data):
        if self.num == 5:
            head_data = (data[:, :self.num] - self.min) / (self.max - self.min)
            tail_data = (data[:, self.num:] - self.mean) / self.std

            normalized_data = np.hstack((head_data , tail_data))
        else:
            head_data = data[:, :self.num-1]
            mid_data = (data[:, self.num-1:self.num] - self.min) / (self.max - self.min)
            tail_data = (data[:, self.num:] - self.mean) / self.std

            normalized_data = np.hstack((head_data, mid_data, tail_data))
        return normalized_data

    def inverse_transform(self, data):
        if self.num == 5:
            head_data = data[:, :self.num] * (self.max - self.min) + self.min
            tail_data = data[:, self.num:] * self.std + self.mean

            raw_data = np.hstack((head_data, tail_data))
        else:
            head_data = data[:, :self.num-1]
            mid_data = data[:, self.num-1:self.num] * (self.max - self.min) + self.min
            tail_data = data[:, self.num:] * self.std + self.mean

            raw_data = np.hstack((head_data, mid_data, tail_data))
        return raw_data

    def save_parameters(self, minmax_path, standard_path):
        # np.savez(minmax_path, min=self.min, max=self.max)
        np.savez(standard_path, mean=self.mean, std=self.std)

    def load_parameters(self, minmax_path, standard_path):
        # minmax_params = np.load(minmax_path)
        # self.min = minmax_params['min']
        # self.max = minmax_params['max']

        standard_params = np.load(standard_path)
        self.mean = standard_params['mean']
        self.std = standard_params['std']

class Scaler_standard:
    def __init__(self, num):  #num为5或2
        self.num = num

        self.min = None
        self.max = None
        self.mean = None
        self.std = None

        if self.num == 5:
            self.min = np.array([20, 6, 0, 1, 100])
            self.max = np.array([300, 100, 4, 10, 1000])
        else:
            self.min = np.array([0])
            self.max = np.array([7200])

    def fit(self, data):
        if self.num == 5:
            self.mean = np.mean(data[:, self.num:], axis=0)
            self.std = np.std(data[:, self.num:], axis=0)
        else:
            self.mean = np.mean(data[:, self.num-1:], axis=0)
            self.std = np.std(data[:, self.num-1:], axis=0)

    def transform(self, data):
        if self.num == 5:
            head_data = (data[:, :self.num] - self.min) / (self.max - self.min)
            tail_data = (data[:, self.num:] - self.mean) / self.std

            normalized_data = np.hstack((head_data , tail_data))
        else:
            head_data = data[:, :self.num-1]
            tail_data = (data[:, self.num-1:] - self.mean) / self.std

            normalized_data = np.hstack((head_data, tail_data))
        return normalized_data

    def inverse_transform(self, data):
        if self.num == 5:
            head_data = data[:, :self.num] * (self.max - self.min) + self.min
            tail_data = data[:, self.num:] * self.std + self.mean

            raw_data = np.hstack((head_data, tail_data))
        else:
            head_data = data[:, :self.num-1]
            tail_data = data[:, self.num-1:] * self.std + self.mean

            raw_data = np.hstack((head_data, tail_data))
        return raw_data

    def save_parameters(self, minmax_path, standard_path):
        # np.savez(minmax_path, min=self.min, max=self.max)
        np.savez(standard_path, mean=self.mean, std=self.std)

    def load_parameters(self, minmax_path, standard_path):
        # minmax_params = np.load(minmax_path)
        # self.min = minmax_params['min']
        # self.max = minmax_params['max']
        if self.num == 5:
            standard_params = np.load(standard_path)
            self.mean = standard_params['mean']
            self.std = standard_params['std']
        else:
            pass

class Scaler_minmax:
    def __init__(self, num):  #num为5或0
        self.num = num

        self.min = None
        self.max = None
        self.mean = None
        self.std = None

        if self.num == 5:
            self.min = np.array([20, 6, 0, 1, 100])
            self.max = np.array([300, 100, 4, 10, 1000])
        else:
            self.min = np.array([0, np.log10(1), np.log10(10)]) #这是目前效果最好的配置
            self.max = np.array([1, np.log10(10000000), np.log10(100000)])
            # self.min = np.array([0.046538, 346]) #这是目前效果最好的配置
            # self.max = np.array([7200, 59546])

    def fit(self, data):
        if self.num == 5:
            self.mean = np.mean(data[:, self.num:], axis=0)
            self.std = np.std(data[:, self.num:], axis=0)

    def transform(self, data):
        if self.num == 5:
            head_data = (data[:, :self.num] - self.min) / (self.max - self.min)
            tail_data = (data[:, self.num:] - self.mean) / self.std

            normalized_data = np.hstack((head_data , tail_data))
        else:
            normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        if self.num == 5:
            head_data = data[:, :self.num] * (self.max - self.min) + self.min
            tail_data = data[:, self.num:] * self.std + self.mean

            raw_data = np.hstack((head_data, tail_data))
        else:
            raw_data= data * (self.max - self.min) + self.min

        return raw_data

    def save_parameters(self, minmax_path, standard_path):
        # np.savez(minmax_path, min=self.min, max=self.max)
        np.savez(standard_path, mean=self.mean, std=self.std)

    def load_parameters(self, minmax_path, standard_path):
        # minmax_params = np.load(minmax_path)
        # self.min = minmax_params['min']
        # self.max = minmax_params['max']

        standard_params = np.load(standard_path)
        self.mean = standard_params['mean']
        self.std = standard_params['std']

class Scaler_minmax2:
    def __init__(self, num):  #num为5或0
        self.num = num

        self.min = None
        self.max = None
        self.mean = None
        self.std = None

        if self.num == 4:
            self.min = np.array([20, 6, 0, 100])
            self.max = np.array([300, 100, 4, 1000])
        else:
            self.min = np.array([0, np.log10(1), np.log10(10)])  # 这是目前效果最好的配置
            self.max = np.array([1, np.log10(10000000), np.log10(100000)])

    def fit(self, data):
        if self.num == 4:
            self.mean = np.mean(data[:, self.num:], axis=0)
            self.std = np.std(data[:, self.num:], axis=0)

    def transform(self, data):
        if self.num == 4:
            head_data = (data[:, :self.num] - self.min) / (self.max - self.min)
            tail_data = (data[:, self.num:] - self.mean) / self.std

            normalized_data = np.hstack((head_data , tail_data))
        else:
            normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        if self.num == 4:
            head_data = data[:, :self.num] * (self.max - self.min) + self.min
            tail_data = data[:, self.num:] * self.std + self.mean

            raw_data = np.hstack((head_data, tail_data))
        else:
            raw_data= data * (self.max - self.min) + self.min

        return raw_data

    def save_parameters(self, minmax_path, standard_path):
        # np.savez(minmax_path, min=self.min, max=self.max)
        np.savez(standard_path, mean=self.mean, std=self.std)

    def load_parameters(self, minmax_path, standard_path):
        # minmax_params = np.load(minmax_path)
        # self.min = minmax_params['min']
        # self.max = minmax_params['max']

        standard_params = np.load(standard_path)
        self.mean = standard_params['mean']
        self.std = standard_params['std']

class Scaler_minmax_new:
    def __init__(self, num):  #num为6或0
        self.num = num

        self.min = None
        self.max = None
        self.mean = None
        self.std = None

        if self.num == 5:
            self.min = np.array([20, 4, 10, 0, 100])
            self.max = np.array([800, 100, 1000, 3, 1000])
        else:
            self.min = np.array([0, np.log10(1), np.log10(1)])  # 这个等数据出来后再根据统计信息修改
            self.max = np.array([1, np.log10(100000), np.log10(100000)])

    def fit(self, data):
        if self.num == 5:
            self.mean = np.mean(data[:, self.num:], axis=0)
            self.std = np.std(data[:, self.num:], axis=0)

    def transform(self, data):
        if self.num == 5:
            head_data = (data[:, :self.num] - self.min) / (self.max - self.min)
            tail_data = (data[:, self.num:] - self.mean) / self.std

            normalized_data = np.hstack((head_data , tail_data))
        else:
            normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        if self.num == 5:
            head_data = data[:, :self.num] * (self.max - self.min) + self.min
            tail_data = data[:, self.num:] * self.std + self.mean

            raw_data = np.hstack((head_data, tail_data))
        else:
            raw_data= data * (self.max - self.min) + self.min

        return raw_data

    def save_parameters(self, minmax_path, standard_path):
        # np.savez(minmax_path, min=self.min, max=self.max)
        np.savez(standard_path, mean=self.mean, std=self.std)

    def load_parameters(self, minmax_path, standard_path):
        # minmax_params = np.load(minmax_path)
        # self.min = minmax_params['min']
        # self.max = minmax_params['max']

        standard_params = np.load(standard_path)
        self.mean = standard_params['mean']
        self.std = standard_params['std']

class Scaler_minmax_new_gpu:
    def __init__(self, num, device):  # num为6或3或0
        self.num = num

        self.min = None
        self.max = None
        self.mean = None
        self.std = None

        if self.num == 6:
            self.min = torch.tensor([np.log10(20), 4, 1, 5, 2, 100], dtype=torch.float32).to(device)
            self.max = torch.tensor([np.log10(800), 100, torch.log10(torch.tensor(5000)), 8, 4, 1000], dtype=torch.float32).to(device)
        elif self.num == 3:
            self.min = torch.tensor([5, 2, 100], dtype=torch.float32).to(device)
            self.max = torch.tensor([8, 4, 1000], dtype=torch.float32).to(device)
        else:
            self.min = torch.tensor([0, 2, 2], dtype=torch.float32).to(device)
            self.max = torch.tensor([1, 5, torch.log10(torch.tensor(500000))], dtype=torch.float32).to(device)

    def fit(self, data):
        if self.num != 0:
            self.mean = torch.mean(data[:, self.num:], dim=0)
            self.std = torch.std(data[:, self.num:], dim=0)

    def transform(self, data):
        if self.num != 0:
            head_data = (data[:, :self.num] - self.min) / (self.max - self.min)
            tail_data = (data[:, self.num:] - self.mean) / self.std

            normalized_data = torch.cat((head_data, tail_data), dim=1)
        else:
            normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        if self.num != 0:
            head_data = data[:, :self.num] * (self.max - self.min) + self.min
            tail_data = data[:, self.num:] * self.std + self.mean

            raw_data = torch.cat((head_data, tail_data), dim=1)
        else:
            raw_data = data * (self.max - self.min) + self.min

        return raw_data

    def save_parameters(self, minmax_path, standard_path):
        # torch.save({'min': self.min, 'max': self.max}, minmax_path)
        torch.save({'mean': self.mean, 'std': self.std}, standard_path)

    def load_parameters(self, minmax_path, standard_path, device):
        # minmax_params = torch.load(minmax_path)
        # self.min = minmax_params['min']
        # self.max = minmax_params['max']

        standard_params = torch.load(standard_path)
        self.mean = standard_params['mean'].to(device)
        self.std = standard_params['std'].to(device)

class Scaler_minmax_new_gpu_nsg:
    def __init__(self, num, device):  # num为6或3或0
        self.num = num

        self.min = None
        self.max = None
        self.mean = None
        self.std = None

        if self.num == 9:
            self.min = torch.tensor([100, 100, 150, 5, 300, 1, 5, 2, 100], dtype=torch.float32).to(device)
            self.max = torch.tensor([400, 400, 350, 90, 600, torch.log10(torch.tensor(1500)), 6, 4, 1000],
                                    dtype=torch.float32).to(device)
        elif self.num == 3:
            self.min = torch.tensor([5, 2, 100], dtype=torch.float32).to(device)
            self.max = torch.tensor([6, 4, 1000], dtype=torch.float32).to(device)
        else:
            self.min = torch.tensor([0, 1], dtype=torch.float32).to(device)
            self.max = torch.tensor([1, torch.log10(torch.tensor(50000))], dtype=torch.float32).to(device)

    def fit(self, data):
        if self.num != 0:
            self.mean = torch.mean(data[:, self.num:], dim=0)
            self.std = torch.std(data[:, self.num:], dim=0) + 1e-8

    def transform(self, data):
        if self.num != 0:
            head_data = (data[:, :self.num] - self.min) / (self.max - self.min)
            tail_data = (data[:, self.num:] - self.mean) / self.std

            normalized_data = torch.cat((head_data, tail_data), dim=1)
        else:
            normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        if self.num != 0:
            head_data = data[:, :self.num] * (self.max - self.min) + self.min
            tail_data = data[:, self.num:] * self.std + self.mean

            raw_data = torch.cat((head_data, tail_data), dim=1)
        else:
            raw_data = data * (self.max - self.min) + self.min

        return raw_data

    def save_parameters(self, minmax_path, standard_path):
        # torch.save({'min': self.min, 'max': self.max}, minmax_path)
        torch.save({'mean': self.mean, 'std': self.std}, standard_path)

    def load_parameters(self, minmax_path, standard_path, device):
        # minmax_params = torch.load(minmax_path)
        # self.min = minmax_params['min']
        # self.max = minmax_params['max']

        standard_params = torch.load(standard_path)
        self.mean = standard_params['mean'].to(device)
        self.std = standard_params['std'].to(device)

class Scaler_minmax_new_partial_gpu:  #这个不包含efC、M和efS
    def __init__(self, num, device):  # num为3
        self.num = num

        self.min = None
        self.max = None
        self.mean = None
        self.std = None

        self.min = torch.tensor([5, 2, 100], dtype=torch.float32).to(device)
        self.max = torch.tensor([8, 4, 1000], dtype=torch.float32).to(device)
        
    def fit(self, data):
        self.mean = torch.mean(data[:, self.num:], dim=0)
        self.std = torch.std(data[:, self.num:], dim=0)

    def transform(self, data):
        head_data = (data[:, :self.num] - self.min) / (self.max - self.min)
        tail_data = (data[:, self.num:] - self.mean) / self.std

        normalized_data = torch.cat((head_data, tail_data), dim=1)
        
        return normalized_data

    def inverse_transform(self, data):
        head_data = data[:, :self.num] * (self.max - self.min) + self.min
        tail_data = data[:, self.num:] * self.std + self.mean

        raw_data = torch.cat((head_data, tail_data), dim=1)
      

        return raw_data

    def save_parameters(self, minmax_path, standard_path):
        # torch.save({'min': self.min, 'max': self.max}, minmax_path)
        torch.save({'mean': self.mean, 'std': self.std}, standard_path)

    def load_parameters(self, minmax_path, standard_path, device):
        # minmax_params = torch.load(minmax_path)
        # self.min = minmax_params['min']
        # self.max = minmax_params['max']

        standard_params = torch.load(standard_path)
        self.mean = standard_params['mean'].to(device)
        self.std = standard_params['std'].to(device)

class Scaler_minmax_new_ct_gpu:
    def __init__(self, num, device):  # num为6或0
        self.num = num

        self.min = None
        self.max = None
        self.mean = None
        self.std = None

        if self.num == 4:
            self.min = torch.tensor([1, 4, 5, 100], dtype=torch.float32).to(device)
            self.max = torch.tensor([3, 100, 8, 1000], dtype=torch.float32).to(device)
        else:
            self.min = torch.tensor([2], dtype=torch.float32).to(device)
            self.max = torch.tensor([torch.log10(torch.tensor(5000000))], dtype=torch.float32).to(device)

    def fit(self, data):
        if self.num == 4:
            self.mean = torch.mean(data[:, self.num:], dim=0)
            self.std = torch.std(data[:, self.num:], dim=0)

    def transform(self, data):
        if self.num == 4:
            head_data = (data[:, :self.num] - self.min) / (self.max - self.min)
            tail_data = (data[:, self.num:] - self.mean) / self.std

            normalized_data = torch.cat((head_data, tail_data), dim=1)
        else:
            normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        if self.num == 4:
            head_data = data[:, :self.num] * (self.max - self.min) + self.min
            tail_data = data[:, self.num:] * self.std + self.mean

            raw_data = torch.cat((head_data, tail_data), dim=1)
        else:
            raw_data = data * (self.max - self.min) + self.min

        return raw_data

    def save_parameters(self, minmax_path, standard_path):
        # torch.save({'min': self.min, 'max': self.max}, minmax_path)
        torch.save({'mean': self.mean, 'std': self.std}, standard_path)

    def load_parameters(self, minmax_path, standard_path, device):
        # minmax_params = torch.load(minmax_path)
        # self.min = minmax_params['min']
        # self.max = minmax_params['max']

        standard_params = torch.load(standard_path)
        self.mean = standard_params['mean'].to(device)
        self.std = standard_params['std'].to(device)

'''
----------------------创建正负样本对、三元组，以及对应的Miner策略---------------------------------
Miner是在训练过程中，动态地在每一批输入向量中进一步挖掘出队训练更有效的样本对或三元组
'''
def create_pairs(batch_performance_norm, performance_norm, threshold, batch_start_index):
    # 计算该批次性能向量与全体性能向量之间的余弦相似度
    sim_matrix = torch.mm(batch_performance_norm, performance_norm.t())

    # 正样本对的提取
    positive_mask = (sim_matrix > threshold)
    positive_indices = positive_mask.nonzero()

    # 负样本对的提取
    negative_mask = (sim_matrix < threshold)
    negative_indices = negative_mask.nonzero()

    # 调整第一维度的索引 第一维度的索引是每一批数据中的索引，需要调整成在原始数据中的索引，而第二维度的索引本身就是原始数据中的索引
    positive_indices[:, 0] = positive_indices[:, 0] + batch_start_index
    negative_indices[:, 0] = negative_indices[:, 0] + batch_start_index

    return positive_indices, negative_indices

def create_pairs_with_SemiHardMiner(batch_feature, feature, batch_performance_norm, performance_norm, threshold, margin, batch_start_index):  #挖掘候选样本对中满足距离条件的样本对，即正样本对距离小于margin，负样本对距离大于margin
    # 计算该批次性能向量与全体性能向量之间的余弦相似度
    sim_matrix = torch.mm(batch_performance_norm, performance_norm.t())

    # 样本对的提取
    positive_mask = (sim_matrix >= threshold)
    negative_mask = (sim_matrix < threshold)

    # 计算batch_feature和feature之间的欧氏距离
    diff = batch_feature.unsqueeze(1) - feature.unsqueeze(0)
    dist_matrix = torch.sqrt(torch.sum(diff * diff, dim=2))

    # 从先前提取的正样本对和负样本对中选择难样本对
    hard_positive_mask = positive_mask & (dist_matrix >= 0.5 * margin)
    hard_positive_indices = hard_positive_mask.nonzero()

    hard_negative_mask = negative_mask & (dist_matrix <= margin)
    hard_negative_indices = hard_negative_mask.nonzero()

    # 调整第一维度的索引
    hard_positive_indices[:, 0] += batch_start_index
    hard_negative_indices[:, 0] += batch_start_index

    return hard_positive_indices, hard_negative_indices

def create_triplets_with_SemiHardMiner(batch_feature, feature, batch_performance_norm, performance_norm, threshold, margin, batch_start_index):
    sim_matrix = torch.mm(batch_performance_norm, performance_norm.t())

    positive_mask = sim_matrix >= threshold
    negative_mask = sim_matrix < threshold
    # print(sim_matrix)

    diff = batch_feature.unsqueeze(1) - feature.unsqueeze(0)
    dist_matrix = torch.sqrt(torch.sum(diff * diff, dim=2))
    # print(dist_matrix)
    # print(positive_mask)
    # print(negative_mask)

    # 一次性计算所有行的负样本距离，避免在循环中重复计算
    neg_value = torch.tensor(float('-1'), device='cuda:0')
    modified_dist_matrix = dist_matrix.clone()
    modified_dist_matrix[~positive_mask] = neg_value
    # print(modified_dist_matrix)

    max_pos_dist, max_pos_indices = torch.max(modified_dist_matrix, dim=1)
    # print(max_pos_dist)
    # print(max_pos_indices)

    # 创建阈值比较矩阵
    margin_matrix = max_pos_dist.unsqueeze(1) + margin
    # print(margin_matrix)

    # 找到所有满足条件的负样本
    valid_negatives_mask = (dist_matrix <= margin_matrix) & negative_mask
    valid_negatives_indices = valid_negatives_mask.nonzero()

    # 提取anchor, positive, negative索引
    anchors = valid_negatives_indices[:, 0] + batch_start_index
    positives = max_pos_indices[valid_negatives_indices[:, 0]]
    negatives = valid_negatives_indices[:, 1]
    # print(anchors)
    # print(positives)
    # print(negatives)

    return anchors, positives, negatives # 返回三元组索引列表

'''
----------------------保存和加载模型---------------------------------
'''
# 保存模型参数
def save_model(model, optimizer, epoch, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, file_path)

# 加载模型参数
def load_model(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

'''
----------------------利用cuml NearestNeighbors找k近邻---------------------------------
'''
#找出每个测试特征向量在特征向量池中的k近邻的索引
def find_k_neighbors(features_pool, features_test, k, threshold):
    # features_pool_cp = cp.asarray(features_pool.cpu().numpy())
    # features_test_cp = cp.asarray(features_test.cpu().numpy())
    features_pool_cp = cp.asarray(features_pool)
    features_test_cp = cp.asarray(features_test)

    nn = NearestNeighbors(n_neighbors=k, algorithm='brute')
    nn.fit(features_pool_cp)

    # 获取距离和索引
    distances, indices = nn.kneighbors(features_test_cp)

    # 将CuPy数组转换回PyTorch张量
    distances = torch.tensor(distances.get()).cuda()
    indices = torch.tensor(indices.get()).cuda()

    # 创建一个距离阈值内的有效掩码
    valid_mask = distances < threshold
    filtered_indices = torch.where(valid_mask, indices, torch.tensor(-1).cuda())  # 无效索引设置为-1

    valid_indices = [idx[idx != -1].tolist() for idx in filtered_indices]

    return valid_indices

'''
----------------------基于找出的k近邻预测每个测试特征向量的性能---------------------------------
'''
def predict_performance(performances_pool, valid_indices):
    # 初始化一个列表来存储预测结果
    predictions = []

    # 处理每组有效的索引
    for indices in valid_indices:
        if len(indices) > 0:
            indices_tensor = torch.tensor(indices)

            selected_performances = performances_pool[indices_tensor]
            mean_performance = torch.mean(selected_performances, dim=0)
        else:
            # 如果没有有效近邻，使用NaN填充
            mean_performance = torch.tensor([float('nan'), float('nan'), float('nan')], dtype=torch.float32)

        # 添加到结果列表
        predictions.append(mean_performance)

    predicted_performances = torch.stack(predictions)
    return predicted_performances

'''
----------------------计算预测误差：mean error 和 mean qerror---------------------------------
'''
def calculate_errors(performances_test, predicted_performances):
    # 计算mean error
    errors = torch.abs(performances_test - predicted_performances)
    mean_errors = torch.mean(errors, dim=0)

    errors_percent = errors / ((performances_test + predicted_performances) / 2)
    mean_errors_percent = torch.mean(errors_percent, dim=0)

    # 计算q error
    qerrors = torch.max(performances_test / predicted_performances, predicted_performances / performances_test)
    mean_qerrors = torch.mean(qerrors, dim=0)

    return mean_errors, mean_errors_percent, mean_qerrors


'''
----------------------计算特征向量差异，更新特征字典---------------------------------
'''
def get_init_fetaure_dic(data_fetaure_path, feature_standard_path, fetaure_dic_path):  #获取初始的特征向量矩阵及其对应的特征差异阈值矩阵，用字典存储
    df = pd.read_csv(data_fetaure_path, sep=',', header=0)
    feature_df = df[['SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']]

    feature_pool =  feature_df.to_numpy()

    standard_params = torch.load(feature_standard_path)
    std = standard_params['std'].cpu().numpy()
    std = std.reshape((-1, len(std)))

    std_mat = np.tile(std, (feature_pool.shape[0], 1))
    print(std_mat.shape)

    feature_difference_threshold_mat = np.zeros_like(feature_pool)

    feature_difference_threshold_mat[:, 3:] = std_mat * 0.5

    feature_pool1 = feature_pool[:, 0]
    feature_pool2 = feature_pool[:, 1]

    feature_difference_threshold1 = feature_difference_threshold_mat[:, 0]
    feature_difference_threshold2 = feature_difference_threshold_mat[:, 1]

    cond11 = (1e5 <= feature_pool1) & (feature_pool1 < 1e6)
    cond12 = (1e6 <= feature_pool1) & (feature_pool1 < 1e7)
    cond13 = (1e7 <= feature_pool1) & (feature_pool1 < 1e8)

    cond21 = (1e2 <= feature_pool2) & (feature_pool2 < 1e3)
    cond22 = (1e3 <= feature_pool2) & (feature_pool2 < 1e4)
    cond23 = (1e4 <= feature_pool2) & (feature_pool2 < 1e5)


    feature_difference_threshold1[cond11] = 1e5
    feature_difference_threshold1[cond12] = 5e5
    feature_difference_threshold1[cond13] = 25 * 1e5

    feature_difference_threshold2[cond21] = 1e2
    feature_difference_threshold2[cond22] = 5e2
    feature_difference_threshold2[cond23] = 25 * 1e2


    feature_difference_threshold_mat[:, 0] = feature_difference_threshold1
    feature_difference_threshold_mat[:, 1] = feature_difference_threshold2
    feature_difference_threshold_mat[:, 2] = np.tile(np.array([50]), feature_pool.shape[0])
    
    feature_dic = {}
    feature_dic['fp'] = feature_pool
    feature_dic['fdt'] = feature_difference_threshold_mat

    # print(feature_pool)
    # print(feature_difference_threshold_mat)
    
    with open(fetaure_dic_path, 'wb') as f:
        pickle.dump(feature_dic, f)


def compare_fetaure(new_feature_vector, feature_pool, feature_difference_threshold_mat):
    differences = np.abs(feature_pool - new_feature_vector)
    
    # Compare differences with threshold_vector
    comparison = np.all(differences <= feature_difference_threshold_mat, axis=1)
    
    # Check if any row in comparison is True
    if np.any(comparison):
        return True
    else:
        return False


def update_fetaure_dic(new_feature_vector, feature_standard_path, fetaure_dic_path):  #讲新的特征向量与已有的特征向量池中的特征向量进行比较，并对fetaure_dic进行更新。
    def get_top3_threshold(vec):
        d_vec = np.zeros_like(vec)

        size = vec[0]
        q_size = vec[1]

        if 1e5 <= size < 1e6:
            d_vec[0] = 1e5
        elif 1e6 <= size < 1e7:
            d_vec[0] = 5e5
        elif 1e7 <= size < 1e8:
            d_vec[0] = 25 * 1e5

        if 1e2 <= q_size < 1e3:
            d_vec[1] = 1e2
        elif 1e3 <= q_size < 1e4:
            d_vec[1] = 5e2
        elif 1e4 <= q_size < 1e5:
            d_vec[1] = 25 * 1e2

        d_vec[2] = 50

        return d_vec

    with open(feature_dic_path, 'rb') as f:
        feature_dic = pickle.load(f)
    
    feature_pool = feature_dic['fp']
    feature_difference_threshold_mat = feature_dic['fdt']

    standard_params = torch.load(feature_standard_path)   #特征数据的std可能发生了改变，所以每次更新是都要现重新对特征差异阈值矩阵进行更新
    std = standard_params['std'].cpu().numpy()
    std = std.reshape((-1, len(std)))

    std_mat = np.tile(std, (feature_pool.shape[0], 1))  
    feature_difference_threshold_mat[:, 3:] = std_mat * 0.5

    flag = compare_fetaure(new_feature_vector, feature_pool, feature_difference_threshold_mat)

    cur_num = feature_pool.shape[0]
    dim = feature_pool.shape[1]

    new_feature_pool = np.zeros((cur_num + 1, dim))
    new_feature_difference_threshold_mat = np.zeros((cur_num + 1, dim))

    new_feature_pool[:-1, :] = feature_pool
    new_feature_pool[-1, :] = new_feature_vector  #注意，传入的new_feature_vector是一行dim列的二维数组

    new_feature_difference_threshold_mat[:-1, :] = feature_difference_threshold_mat
    new_feature_difference_threshold_mat[-1, 3:] = std * 0.5

    vec = new_feature_vector[0, 0:3]
    new_feature_difference_threshold_mat[-1, 0:3] = get_top3_threshold(vec)

    feature_dic['fp'] = new_feature_pool
    feature_dic['fdt'] = new_feature_difference_threshold_mat
    
    with open(feature_dic_path, 'wb') as f:
        pickle.dump(feature_dic, f)



