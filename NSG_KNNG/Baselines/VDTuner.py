import os
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import struct
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

class CEIBO:
    def __init__(self, target_rec, seed):
        self.seed = seed

        self.target_rec = target_rec 
        self.bounds = torch.tensor([[0.0] * 6, [1.0]* 6])
        
        self.X_init = None
        self.Y_init = None

        self.kernel_init()
    
    def kernel_init(self):
        covar_module = MaternKernel(
                nu=2.5,
                # ard_num_dims=1,
                active_dims=(0, 1, 2, 3, 4, 5),
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
        self.default_para = np.array([[200, 250, 250, 30, 400, 500]])
        self.min_para = np.array([[100, 100, 150, 5, 300, 10]])
        self.max_para =  np.array([[400, 400, 350, 90, 600, 1500]])

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

        paras[:, 1] = np.where(paras[:, 1] < paras[:, 0], paras[:, 0], paras[:, 1])

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

def get_performance_data(K, L, ite, S, R, L_nsg_C, R_nsg, C, L_nsg_S, target_recall, subdir, filename, base_path, query_path, gt_indice_path,
                         index_performance_csv1, index_performance_csv2, index_performance_csv3, index_performance_csv4):
    whole_name = subdir + '_' + filename

    KNN_graph_path = os.path.join(KNN_graph_dir, '{}/{}_{}_{}.graph'.format(subdir, filename, K, L))
    NSG_graph_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C))

    print(f'-------------------开始构建KNNG: {K}_{L}，只需构建一次-------------------')
    if not os.path.exists(KNN_graph_path):
        t1 = time.time()
        run_command1 = ['../FFANNA_KNNG/build/tests/test_nndescent', base_path, KNN_graph_path, str(K), str(L), str(ite),
                        str(S), str(R), whole_name, index_performance_csv1]
        result1 = subprocess.run(run_command1, check=True, text=True, capture_output=True)
        t2 = time.time()

        KNN_graph_time = t2 - t1

    print(f'-------------------开始构建NSG: {L_nsg_C}_{R_nsg}_{C}-------------------')
    t3 = time.time()
    run_command2 = ['../NSG/build/tests/test_nsg_index', base_path, KNN_graph_path, str(L_nsg_C), str(R_nsg), str(C),
                    NSG_graph_path, str(K), str(L), whole_name, index_performance_csv2]
    result2 = subprocess.run(run_command2, check=True, text=True, capture_output=True)
    t4 = time.time()

    NSG_graph_time = t4 - t3

    qs_indice_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C, L_nsg_S))

    run_command3 = ['../NSG/build/tests/test_nsg_optimized_search', base_path, query_path, NSG_graph_path,  str(L_nsg_S), str(10), qs_indice_path, str(K), str(L), str(L_nsg_C), str(R_nsg), str(C),
                    whole_name, index_performance_csv3]
    result3 = subprocess.run(run_command3, check=True, text=True, capture_output=True)

    gt = read_ivecs(gt_indice_path)
    qs = read_ivecs(qs_indice_path)

    rec = calculate_recall_rate(gt, qs)

    os.remove(qs_indice_path)  # 用完了就删除

    recs = [(whole_name, K, L, L_nsg_C, R_nsg, C, L_nsg_S, target_recall, rec)]
    rec_df = pd.DataFrame(recs, columns=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'target_recall', 'recall'])
    rec_df.to_csv(index_performance_csv4, mode='a', header=False, index=False)

    if NSG_graph_time < 1800:
        os.remove(NSG_graph_path)  # NSG构建时间较短就删除

    search_df = pd.read_csv(index_performance_csv3, sep=',', header=0)
    st = search_df['search_time'].iloc[-1]

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
    parent_directory = os.path.dirname(current_directory)  # 返回当前工作目录所指路径的父目录
    three_level_directory = Path.cwd().parents[1]

    target_rec_lis = [0.9]

    filename_dic = {'deep1': '0_1_96_1', 'paper': '0_2_200_1', 'gist': '0_1_960_1'}

    ite = 12
    S = 15
    R = 100

    # NSG搜索参数
    L_nsg_Ss = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 430,
                460, 490, 520, 550, 580, 610, 640, 670, 700, 740, 780, 820, 860, 900, 960, 1020, 1080, 1140, 1200, 1300, 1400, 1500]

    base_dir = os.path.join(three_level_directory, 'Data/Base')
    query_dir = os.path.join(three_level_directory, 'Data/Query')
    groundtruth_dir = os.path.join(three_level_directory, 'Data/GroundTruth')
    KNN_graph_dir = "../KNN_graph/VDTuner"
    NSG_graph_dir = "../NSG_graph/VDTuner"

    exist_index_performance_csv = os.path.join(parent_directory, 'Data/index_performance_VDTuner.csv')

    dataset_name = 'gist'
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

    base_path = os.path.join(subdir_path, filename + '.fvecs')
    query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, dim))
    gt_indice_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename))

    index_performance_csv1 = "./Data/experiments_results/index_performance_KNNG_VDTuner.csv"  # 记录每个数据集在不同参数下构建KNNG的时间、距离计算次数、占用内存等信息
    index_performance_csv2 = "./Data/experiments_results/index_performance_NSG_VDTuner.csv"  # 记录每个数据集在固定KNNG和在不同参数下构建NSG的时间、距离计算次数、占用内存等信息
    index_performance_csv3 = "./Data/experiments_results/index_performance_Search_VDTuner.csv"  # 记录每个数据集在固定NSG和不同参数下搜索的时间、距离计算次数、占用内存等信息
    index_performance_csv4 = "./Data/experiments_results/index_performance_Search_Recall_VDTuner.csv"
    whole_index_performance_csv = './Data/index_performance_VDTuner.csv'


    whole_name = subdir + '_' + filename

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
            init_para = [200, 250, 250, 30, 400, 500]
            init_K, init_L, init_L_nsg_C, init_R_nsg, init_C, init_L_nsg_S = init_para

            init_rec, init_qps = get_performance_data( init_K, init_L, ite, S, R, init_L_nsg_C, init_R_nsg, init_C, init_L_nsg_S, target_rec, subdir, filename, base_path, query_path, gt_indice_path,
                                                        index_performance_csv1, index_performance_csv2, index_performance_csv3, index_performance_csv4)
            init_performance = [init_rec, init_qps]

            model.init_sample(init_para, init_performance)

        #接下来是基于高斯回归和约束期望改进的贝叶斯优化的参数推荐，每次推荐一组(efConstrucion, M, efSearch)参数；每个目标召回率迭代100次？
        for i in tqdm(range(500), total = 500):
            recommend_paras = model.step(q=1)

            new_K, new_L, new_L_nsg_C, new_R_nsg, new_C, new_L_nsg_S = recommend_paras[0]
            new_K = int(new_K)
            new_L = int(new_L)
            new_L_nsg_C = int(new_L_nsg_C)
            new_R_nsg = int(new_R_nsg)
            new_C = int(new_C)
            new_L_nsg_S = int(new_L_nsg_S)

            new_rec, new_qps = get_performance_data(new_K, new_L, ite, S, R, new_L_nsg_C, new_R_nsg, new_C, new_L_nsg_S, target_rec, subdir, filename, base_path, query_path,
                                                      gt_indice_path, index_performance_csv1, index_performance_csv2, index_performance_csv3, index_performance_csv4)
            #最后利用新的efC, m, efS和对应的rec, qps更新模型
            new_para = [new_K, new_L, new_L_nsg_C, new_R_nsg, new_C, new_L_nsg_S]
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







    

