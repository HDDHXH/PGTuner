import os
import subprocess
import pandas as pd
import numpy as np

import torch
from torch.backends import cudnn

from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition import ConstrainedExpectedImprovement
from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels import ProductKernel
from gpytorch.priors.torch_priors import GammaPrior

from tqdm import tqdm
import time
import random
import pickle
import re

class CEIBO:
    def __init__(self, target_rec, seed):
        self.seed = seed

        self.target_rec = target_rec 
        self.bounds = torch.tensor([[0.0] * 3, [1.0]* 3])
        
        self.X_init = None
        self.Y_init = None

        self.kernel_init()
    
    def kernel_init(self):
        covar_module = MaternKernel(
                nu=2.5,
                # ard_num_dims=1,
                active_dims=(0, 1, 2),
                # batch_shape=torch.Size([]),
                lengthscale_prior=GammaPrior(3.0, 6.0),
            )

        self.covar_module = ScaleKernel(
            covar_module,
            # batch_shape=self._aug_batch_shape,
            outputscale_prior=GammaPrior(2.0, 0.15),
            )
    
    def recommend(self, q):
        # assume 2-dim output: [recall, qps]
        #qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        
        acq_func = LogConstrainedExpectedImprovement(
            model=self.model,
            best_f = torch.tensor([1]),
            constraints={0: [self.target_rec, 1]},  #形式是字典，键是要约束的目标的索引，键值是该约束目标的值的范围
            objective_index=1  #这个就是要优化的目标的索引
        )

        candidate, acq_value = optimize_acqf(acq_func, bounds=self.bounds, q=q, num_restarts=10, raw_samples=100, options={'seed':self.seed})

        new_x = candidate.detach()

        return new_x.numpy()
    
    def update_samples(self, norm_X, norm_Y,):
        self.X_init = torch.tensor(norm_X,dtype=torch.float64)
        self.Y_init = torch.tensor(norm_Y,dtype=torch.float64)

        models = []

        for i in range(self.Y_init.shape[-1]):
            train_y = self.Y_init[..., i : i + 1]

            models.append(SingleTaskGP(self.X_init, train_y, covar_module=self.covar_module, outcome_transform = Standardize(m=1)))
            #models.append(SingleTaskGP(self.X_init, train_y, outcome_transform=Standardize(m=1)))

            
        self.model = ModelListGP(*models)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(self.mll)

class BayesianOptimization:
    def __init__(self, target_rec=0.85, seed=1206):
        self.default_para = np.array([[256, 32, 500]])
        self.min_para = np.array([[20, 4, 10]])
        self.max_para =  np.array([[800, 100, 5000]])

        self.X = []  #二维嵌套列表
        self.Y = []  #二维嵌套列表

        self.vbo = CEIBO(target_rec, seed=seed)

    def transform(self, X):
        norm_X = (X - self.min_para) / (self.max_para - self.min_para)

        return norm_X

    def inverse_transform(self, norm_X):
        X = self.min_para +  (self.max_para - self.min_para) * norm_X

        return X

    def init_sample(self, x_init, y_init):
        self.X.append(x_init)
        self.Y.append(y_init)

        self.update_model()

    def step(self, q=1): 
        norm_paras = self.vbo.recommend(q)

        paras = self.inverse_transform(norm_paras)  # N*3的数组
        paras = np.floor(paras + 0.5)  # 要4舍5入转成整数

        paras[:, 0] = np.where(paras[:, 0] < paras[:, 1], paras[:, 1], paras[:, 0])

        return paras.tolist()

    def reward_transform(self):
        Y = []

        Y_arr = np.array(self.Y)
      
        Y_arr[:,0] = (Y_arr[:,0] + 1e-6)/ (np.max(Y_arr[:,0]) + 1e-6)
        Y_arr[:,1] = (Y_arr[:,1] + 1e-6) / (np.max(Y_arr[:,1]) + 1e-6)

        Y += Y_arr.tolist()

        X = np.array(self.X)
        self.norm_X = self.transform(X) #二维嵌套列表

        self.norm_Y = Y #二维嵌套列表

    def update_model(self): #x是新的参数列表[efC, M, efS]， y是新的性能数据列表[rec, qps]
        self.reward_transform()
        self.vbo.update_samples(self.norm_X, self.norm_Y)

    def save_model(self, save_path):
        # 保存模型参数和其他需要的信息
        torch.save({
            'X': torch.tensor(self.X),
            'Y': torch.tensor(self.Y),
        }, save_path)

    def load_model(self, save_path):
        checkpoint = torch.load(save_path)
        self.X = checkpoint['X'].tolist()
        self.Y = checkpoint['Y'].tolist()

def get_performance_data(exist_paras, exist_df, efC, m, efS, target_recall, subdir, filename, base_path, query_path, indice_path, index_csv, size, dim):
    whole_name = subdir + '_' + filename

    if [efC, m, efS] in exist_paras:  #如果这个参数的性能数据已经有了，就直接从文件中获取
        config_df = pd.DataFrame([(efC, m, efS)], columns=['efConstruction', 'M', 'efSearch'])

        config_data_df = pd.merge(exist_df, config_df, on=['efConstruction', 'M', 'efSearch'], how='right')
        config_data_df = config_data_df.iloc[[0]]

        rec = config_data_df['recall'].iloc[0]
        st = config_data_df['search_time'].iloc[0]

        config_data_df['target_recall'] = target_recall
        config_data_df.to_csv(index_csv, mode='a', index=False, header = False)


    else: #没有的话再构建索引跑数据
        index_path = os.path.join('../Index', '{}/{}_{}_{}.bin'.format(subdir, filename, efC, m))

        # 构建运行命令
        run_command = ['../index_construct_VDTuner', whole_name, base_path, query_path, indice_path, index_path, index_csv,
                    subdir, str(size), str(dim), str(efC), str(m), str(efS), str(target_recall)]
        #print(" ".join(run_command))
        print(f'{whole_name}_{target_recall}_{efC}_{m}_{efS}')
        print('-------------------开始构建索引并执行测试-------------------')
        result = subprocess.run(run_command, check=True, text=True, capture_output=True)
        print('-------------------索引构建与测试结束-------------------')

        data_df = pd.read_csv(index_csv)
        rec = data_df['recall'].iloc[-1]
        st = data_df['search_time'].iloc[-1]
    
    qps = 1 / st

    return rec, qps


if __name__ == '__main__':
    seed = 1

    torch.autograd.set_detect_anomaly(True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = False
    cudnn.benchmark = False
    cudnn.deterministic = True

    current_directory = os.getcwd()  # 返回当前工作目录的路径
    parent_directory = os.path.dirname(current_directory)

    # 编译cpp文件
    compile_command = ['g++', '-Ofast', '-lrt', '-std=c++11', '-DHAVE_CXX0X', '-march=native', '-fpic', '-w',
                    '-fopenmp', '-ftree-vectorize', '-ftree-vectorizer-verbose=0', '../index_construct_VDTuner.cpp', '-o',
                    '../index_construct_VDTuner'
                    ]
    subprocess.run(compile_command, check=True)
    print('编译完成')

    target_rec_lis = [0.95]

    filename_dic = {'deep1': '1_1_96_1', 'sift1': '1_1_128_1', 'glove': '1_1.183514_100', 'paper': '1_2.029997_200',
                    'crawl': '1_1.989995_300', 'msong': '0_9.92272_420',
                    'gist': '1_1.0_960', 'deep10': '2_1_96', 'sift50': '2_5_128_1', 'deep2': '1_2_96_1',
                    'deep3': '1_3_96_1', 'deep4': '1_4_96_1', 'deep5': '1_5_96_1',
                    'sift2': '1_2_128_1', 'sift3': '1_3_128_1', 'sift4': '1_4_128_1', 'sift5': '1_5_128_1',
                    'gist_25': '1_1.0_960_25', 'gist_50': '1_1.0_960_50',
                    'gist_75': '1_1.0_960_75', 'gist_100': '1_1.0_960_100', 'deep2_25': '1_2_96_1_25',
                    'deep2_50': '1_2_96_1_50', 'deep2_75': '1_2_96_1_75', 'deep2_100': '1_2_96_1_100'}

    # dim_dic = {'glove': 100, 'paper':200, 'crawl':300, 'msong': 420, 'gist': 960, 'deep': 96, 'sift': 128}

    base_dir = os.path.join(parent_directory, 'Data/Base')
    query_dir =  os.path.join(parent_directory, 'Data/Query')
    groundtruth_dir = os.path.join(parent_directory, 'Data/GroundTruth')
    index_dir = os.path.join(parent_directory, 'Index')
    index_csv =  os.path.join(parent_directory, 'Data/index_performance_VDTuner.csv')

    exist_index_performance_csv = os.path.join(parent_directory, 'Data/index_performance_VDTuner.csv')

    # dataset_name = 'gist_100'
    for dataset_name in ['gist_25', 'gist_50', 'gist_75']:
        dataset_name_list = dataset_name.split('_')

        model_save_path = os.path.join(current_directory, '{}_VDTuner_model.pth'.format(dataset_name))

        subdir = re.match(r'\D+', dataset_name_list[0]).group()
        filename = filename_dic[dataset_name]

        subdir_path = os.path.join(base_dir, subdir)

        filename_list = filename.split('_')
        level = int(filename_list[0])
        num = float(filename_list[1])
        dim = int(filename_list[2])

        size = int(pow(10, level) * 100000 * num)

        if subdir in ['sift']:
            if len(dataset_name_list) == 2: #表明是查询负载变化
                last_index = filename.rfind('_')

                temp_filaname = filename[: last_index]
                change_ratio = filename[last_index+1: ]

                base_path = os.path.join(subdir_path, temp_filaname + '.bvecs')
                query_path = os.path.join(query_dir, '{}/{}_{}.bvecs'.format(subdir, dim, change_ratio))
            else:
                base_path = os.path.join(subdir_path, filename+'.bvecs')
                query_path = os.path.join(query_dir, '{}/{}.bvecs'.format(subdir, dim))

        else:
            if len(dataset_name_list) == 2:  # 表明是查询伏在变化
                last_index = filename.rfind('_')

                temp_filaname = filename[: last_index]
                change_ratio = filename[last_index + 1:]

                base_path = os.path.join(subdir_path, temp_filaname + '.fvecs')
                query_path = os.path.join(query_dir, '{}/{}_{}.fvecs'.format(subdir, dim, change_ratio))
            else:
                base_path = os.path.join(subdir_path, filename + '.fvecs')
                query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, dim))


        indice_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename))

        whole_name = subdir + '_' + filename
        exist_df = pd.read_csv(exist_index_performance_csv)
        exist_df = exist_df[exist_df['FileName'] == whole_name]

        exist_paras = exist_df[['efConstruction', 'M', 'efSearch']].to_numpy().tolist()

        time_dic = {}
        time_dic_path = os.path.join(current_directory, '{}_VDTuner_time_dic.pkl'.format(dataset_name))

        for target_rec in tqdm(target_rec_lis, total = len(target_rec_lis)):
            print(f'---------------{target_rec}---------------')
            #初始化模型
            t1 = time.time()
            model = BayesianOptimization(target_rec=target_rec, seed=1)

            if os.path.exists(model_save_path): #存在的话，说明已经有相关的数据，直接加载这些数据训练初始的模型
                model.load_model(model_save_path)
                model.update_model()
            else:
                #第一次要用默认参数初始化
                init_para = [256, 32, 500]
                init_efC, init_m, init_efS = init_para

                init_rec, init_qps = get_performance_data(exist_paras, exist_df, init_efC, init_m, init_efS, target_rec, subdir, filename, base_path, query_path, indice_path, index_csv, size, dim)

                init_performance = [init_rec, init_qps]

                model.init_sample(init_para, init_performance)

            #接下来是基于高斯回归和约束期望改进的贝叶斯优化的参数推荐，每次推荐一组(efConstrucion, M, efSearch)参数；每个目标召回率迭代100次？
            for i in tqdm(range(50), total = 50):
                exist_df = pd.read_csv(exist_index_performance_csv)
                exist_df = exist_df[exist_df['FileName'].str.contains(dataset_name)]

                exist_paras = exist_df[['efConstruction', 'M', 'efSearch']].to_numpy().tolist()

                recommend_paras = model.step(q=1)

                new_efC, new_m, new_efS = recommend_paras[0]
                new_efC = int(new_efC)
                new_m = int(new_m)
                new_efS = int(new_efS)

                new_rec, new_qps = get_performance_data(exist_paras, exist_df, new_efC, new_m, new_efS, target_rec, subdir, filename, base_path, query_path, indice_path, index_csv, size, dim)

                #最后利用新的efC, m, efS和对应的rec, qps更新模型
                new_para = [new_efC, new_m, new_efS]
                new_performance = [new_rec, new_qps]

                model.X.append(new_para)
                model.Y.append(new_performance)
                model.update_model()

            model.save_model(model_save_path)

            t2 = time.time()

            time_dic[target_rec] = t2-t1
            with open(time_dic_path, 'wb') as f:
                pickle.dump(time_dic, f)

        print(time_dic)
        print(sum(list(time_dic.values())))







    

