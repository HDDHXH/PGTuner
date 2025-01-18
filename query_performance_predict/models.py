import torch
import torch.nn as nn
import torch.nn.functional as F
from blitz.modules import BayesianLinear
from blitz.losses import kl_divergence_from_nn
from blitz.utils import variational_estimator


class DML_MLP(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]  #[8, 128, 128, 32]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            if i < final_relu_layer:
                layer_list.append(nn.Dropout(0.3, inplace=False))
        self.net = nn.Sequential(*layer_list)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                # nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        o = self.net(x)
        return o

    def contrastive_loss(self, e1, e2, label, margin):
        euclidean_distance = nn.functional.pairwise_distance(e1, e2)
        loss = torch.mean(label * torch.pow(euclidean_distance, 2) + (1 - label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
        return loss

    def triplet_loss(self, a, p, n, margin):
        distance_positive = nn.functional.pairwise_distance(a, p).pow(2) # L2距离平方
        distance_negative = nn.functional.pairwise_distance(a, n).pow(2)  # L2距离平方
        loss = torch.mean(torch.relu(distance_positive + margin - distance_negative))
        return loss

class Predict_MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes] #[32, 128, 128, 64, 3]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            if i < final_relu_layer:
                layer_list.append(nn.Dropout(0.3, inplace=False))
        self.net = nn.Sequential(*layer_list)
        self._init_weights()

        self.weights = torch.tensor([[1, 0.5, 1]]).cuda()
        self.loss = nn.MSELoss(reduction='none')


    def _init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                # nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        o = self.net(x)
        return o

    def calculate_loss(self, inp, label):
        weighted_loss = self.loss(inp, label)* self.weights
        loss = torch.mean(weighted_loss)

        return loss

class Direct_Predict_MLP(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]  # [14, 128, 256, 64, 3]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))

            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
                layer_list.append(nn.ReLU(inplace=False))
            # if i < final_relu_layer:
            #     layer_list.append(nn.Dropout(0, inplace=False))

        layer_list.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layer_list)

        self._init_weights()

        self.weights = torch.tensor([[1, 5, 20]]).cuda()

        self.loss = nn.MSELoss(reduction='none')

    def _init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        o = self.net(x)  #最后一层使用relu比用sigmoid好，relu在作为后一层或者放在forward函数中都可以
        # o = F.relu(self.net(x))
        return o

    def get_feature_vectors(self, x, feature_layer=3):
        for i, layer in enumerate(self.net):
            x = layer(x)
            # We multiply by 3 because each 'block' includes three layers
            if i == feature_layer * 3 - 1:  
                break

        return x

    def calculate_loss(self, inp, label):
        weighted_loss = self.loss(inp, label) * self.weights
        loss = torch.mean(weighted_loss)

        return loss

class Direct_Predict_MLP_Bayesian(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]  # [14, 128, 256, 64, 3]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(BayesianLinear(input_size, curr_size))
            # if i < final_relu_layer:
            #     layer_list.append(nn.BatchNorm1d(curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
               
        layer_list.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layer_list)
        # self.sigmoid = nn.Sigmoid()
        self._init_weights()

        # self.sigmoid = nn.Sigmoid()
        self.weights = torch.tensor([[1, 5, 5]]).cuda()
        self.loss = nn.MSELoss(reduction='none')

    def _init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        o = self.net(x)  #最后一层使用relu比用sigmoid好，relu在作为后一层或者放在forward函数中都可以
        # o = F.relu(self.net(x))
        return o

    def nn_kl_divergence(self):
        kl_divergence = kl_divergence_from_nn(self)
        return kl_divergence

    def calculate_loss(self, inp, label, sample_nbr=3, complexity_cost_weight=1/50000):
        loss = 0
        for _ in range(sample_nbr):
            oup = self.net(inp)
            loss += torch.mean(self.loss(oup, label) * self.weights)
            loss += self.nn_kl_divergence() * complexity_cost_weight

        final_loss = loss / sample_nbr
            
        return final_loss

class Direct_Predict_MLP_with_uncertainty(nn.Module):
    def __init__(self, layer_sizes, inner_dim):
        super().__init__()
        
        self.fc_mu1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.bn_mu1= nn.BatchNorm1d(layer_sizes[1])

        self.fc_mu2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.bn_mu2 = nn.BatchNorm1d(layer_sizes[2])

        self.fc_mu3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.bn_mu3 = nn.BatchNorm1d(layer_sizes[3])

        self.fc_mu4 = nn.Linear(layer_sizes[3], layer_sizes[4])

        self.fc_sigma1 = nn.Linear(layer_sizes[1], inner_dim)
        self.bn_sigma1= nn.BatchNorm1d(inner_dim)

        self.fc_sigma2 = nn.Linear(layer_sizes[2], inner_dim)
        self.bn_sigma2 = nn.BatchNorm1d(inner_dim)

        self.fc_sigma3 = nn.Linear(layer_sizes[3], inner_dim)
        self.bn_sigma3 = nn.BatchNorm1d(inner_dim)

        self.fc_sigma4 = nn.Linear(3*inner_dim, layer_sizes[4])

        self.relu = nn.ReLU(inplace=False)

        self._init_weights()

        self.weights = torch.tensor([[1, 5, 5]]).cuda()

        self.loss = nn.GaussianNLLLoss(reduction='none')

    def _init_weights(self):
        for m in [self.fc_mu1, self.fc_mu2, self.fc_mu3, self.fc_mu4, self.fc_sigma1, self.fc_sigma2, self.fc_sigma3, self.fc_sigma4]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-2)
            nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        o1 = self.relu(self.bn_mu1(self.fc_mu1(x)))
        o2 = self.relu(self.bn_mu2(self.fc_mu2(o1)))
        o3 = self.relu(self.bn_mu3(self.fc_mu3(o2)))
        mu = self.relu(self.fc_mu4(o3))

        uo1 = self.relu(self.bn_sigma1(self.fc_sigma1(o1)))
        uo2 = self.relu(self.bn_sigma2(self.fc_sigma2(o2)))
        uo3 = self.relu(self.bn_sigma3(self.fc_sigma3(o3)))

        uo = torch.cat((uo1, uo2, uo3), dim=1)

        log_sigma = self.fc_sigma4(uo)
        sigma = torch.exp(log_sigma)

        return mu, sigma

    def calculate_loss(self, inp, target, sigma):
        losses = self.loss(inp, target, sigma)  #loss(input, target, var)，所以inp是mu，target是标签，var是sigma
        loss = torch.mean(losses * self.weights)

        return loss

class Mt_Direct_Predict_MLP(nn.Module):  #多任务框架
    def __init__(self, shared_layer_sizes, private_layer_sizes, final_relu=False):
        super().__init__()
        final_relu=True
        s_layer_list = []
        s_layer_sizes = [int(x) for x in shared_layer_sizes]  # [9, 128, 256, 256] # [256, 64, 1]
        s_num_layers = len(s_layer_sizes) - 1
        s_final_relu_layer = s_num_layers if final_relu else s_num_layers - 1
        for i in range(len(s_layer_sizes) - 1):
            input_size = s_layer_sizes[i]
            curr_size = s_layer_sizes[i + 1]
            s_layer_list.append(nn.Linear(input_size, curr_size))
            if i < s_final_relu_layer:
                s_layer_list.append(nn.BatchNorm1d(curr_size))
            if i < s_final_relu_layer:
                s_layer_list.append(nn.ReLU(inplace=False))
            if i < s_final_relu_layer:
                s_layer_list.append(nn.Dropout(0.3, inplace=False))
        self.s_net = nn.Sequential(*s_layer_list)

        final_relu=False
        p_layer_list = []
        p_layer_sizes = [int(x) for x in private_layer_sizes]  # [9, 128, 256, 256] # [256, 64, 1]
        p_num_layers = len(p_layer_sizes) - 1
        p_final_relu_layer = p_num_layers if final_relu else p_num_layers - 1
        for i in range(len(p_layer_sizes) - 1):
            input_size = p_layer_sizes[i]
            curr_size = p_layer_sizes[i + 1]
            p_layer_list.append(nn.Linear(input_size, curr_size))
            if i < p_final_relu_layer:
                p_layer_list.append(nn.BatchNorm1d(curr_size))
            if i < p_final_relu_layer:
                p_layer_list.append(nn.ReLU(inplace=False))
            if i < p_final_relu_layer:
                p_layer_list.append(nn.Dropout(0.3, inplace=False))
        self.p_net1 = nn.Sequential(*p_layer_list)
        self.p_net2 = nn.Sequential(*p_layer_list)
        self.p_net3 = nn.Sequential(*p_layer_list)
        
        self._init_weights()

        self.weights = torch.tensor([[1, 3, 1]]).cuda()
        self.loss = nn.MSELoss(reduction='none')

    def _init_weights(self):
        for net in [self.s_net, self.p_net1, self.p_net2, self.p_net3]:
            for m in net:
                if type(m) == nn.Linear:
                    # nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        s = self.s_net(x)
        o1 = self.p_net1(s)
        o2 = self.p_net2(s)
        o3 = self.p_net3(s)
        return o1, o2, o3

    def calculate_loss(self, inp1, inp2, inp3, label):
        inp = torch.cat((inp1, inp2, inp3), dim = 1)
        weighted_loss = self.loss(inp, label) * self.weights
        loss = torch.mean(weighted_loss)

        return loss

class It_Direct_Predict_MLP(nn.Module):
    def __init__(self, individual_layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in individual_layer_sizes]  # [9, 128, 256, 256, 64, 1]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            if i < final_relu_layer:
                layer_list.append(nn.Dropout(0.3, inplace=False))
        layer_list.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layer_list)
        # self.sigmoid = nn.Sigmoid()
        self._init_weights()

        self.loss = nn.MSELoss()

    def _init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        o = self.net(x).squeeze(1)
        return o

    def calculate_loss(self, inp, label):
        # label = label.unsqueeze(1)
        loss = self.loss(inp, label)

        return loss

class Direct_Predict_ATT(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]  # [128, 128, 64, 3]
        num_layers = len(layer_sizes) - 2
        final_relu_layer = num_layers if final_relu else num_layers - 1

        self.attention = nn.MultiheadAttention(embed_dim=layer_sizes[1], num_heads=4, dropout=0)

        self.input_q = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.input_k = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.input_v = nn.Linear(layer_sizes[0], layer_sizes[1])

        for i in range(len(layer_sizes) - 2):
            input_size = layer_sizes[i+1]
            curr_size = layer_sizes[i + 2]
            layer_list.append(nn.Linear(input_size, curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
                # layer_list.append(nn.PReLU())
                # layer_list.append(nn.ELU(inplace=False))
            # if i < final_relu_layer:
            #     layer_list.append(nn.Dropout(0.01, inplace=False))
        # layer_list.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layer_list)
        # self.sigmoid = nn.Sigmoid()
        self._init_weights()

        # self.sigmoid = nn.Sigmoid()
        self.weights = torch.tensor([[5, 5, 10]]).cuda()
        self.loss = nn.MSELoss(reduction='none')

    def _init_weights(self):
        for m in [self.input_q, self.input_k, self.input_v]:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.uniform_(m.bias, -0.1, 0.1)

        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.uniform_(m.bias, -0.1, 0.1)

        for m in self.net:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        q = self.input_q(x).unsqueeze(0)
        k = self.input_k(x).unsqueeze(0)
        v = self.input_v(x).unsqueeze(0)

        attn_output, _ = self.attention(q, k, v)
        attn_output = attn_output.squeeze()  #

        o = self.net(attn_output)  #最后一层使用relu比用sigmoid好，relu在作为后一层或者放在forward函数中都可以
        # o = F.relu((self.net(x))
        return o

    def calculate_loss(self, inp, label):
        weighted_loss = self.loss(inp, label) * self.weights
        loss = torch.mean(weighted_loss)

        return loss

class Direct_Predict_Conv2d(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]  # [128, 128, 64, 3]
        num_layers = len(layer_sizes) - 2
        final_relu_layer = num_layers if final_relu else num_layers - 1

        # 二维卷积层
        self.num_filters = int(layer_sizes[1] / layer_sizes[0])
        self.conv = nn.Conv2d(1, self.num_filters, kernel_size=3, padding=1)

        self.bn_conv = nn.BatchNorm2d(self.num_filters)
        self.relu = nn.ReLU(inplace=False)

        for i in range(len(layer_sizes) - 2):
            input_size = layer_sizes[i+1]
            curr_size = layer_sizes[i + 2]
            layer_list.append(nn.Linear(input_size, curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
                # layer_list.append(nn.PReLU())
                # layer_list.append(nn.ELU(inplace=False))
            # if i < final_relu_layer:
            #     layer_list.append(nn.Dropout(0.01, inplace=False))
        # layer_list.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layer_list)
        # self.sigmoid = nn.Sigmoid()
        self._init_weights()

        # self.sigmoid = nn.Sigmoid()
        self.weights = torch.tensor([[5, 5, 10]]).cuda()
        self.loss = nn.MSELoss(reduction='none')

    def _init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        x = x.view(-1, 2, 4) # 二维重塑
        x = x.unsqueeze(1) # 添加一个通道维度，使其成为 [N, 1, 2, 4]

        feature_matrix = self.relu(self.bn_conv(self.conv(x)))
        feature_vector = feature_matrix.view(feature_matrix.size(0), -1)

        o = self.net(feature_vector)  #最后一层使用relu比用sigmoid好，relu在作为后一层或者放在forward函数中都可以
        # o = F.relu((self.net(x))
        return o

    def calculate_loss(self, inp, label):
        weighted_loss = self.loss(inp, label) * self.weights
        loss = torch.mean(weighted_loss)

        return loss


class Direct_Predict_Conv1d(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]  # [128, 128, 64, 3]
        num_layers = len(layer_sizes) - 2
        final_relu_layer = num_layers if final_relu else num_layers - 1

        # 一维卷积层
        self.num_filters = int(layer_sizes[1] / layer_sizes[0])
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.num_filters, kernel_size=3, padding=1)

        self.bn_conv = nn.BatchNorm1d(self.num_filters)
        self.relu = nn.ReLU(inplace=False)

        for i in range(len(layer_sizes) - 2):
            input_size = layer_sizes[i + 1]
            curr_size = layer_sizes[i + 2]
            layer_list.append(nn.Linear(input_size, curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
                # layer_list.append(nn.PReLU())
                # layer_list.append(nn.ELU(inplace=False))
            # if i < final_relu_layer:
            #     layer_list.append(nn.Dropout(0.01, inplace=False))
        # layer_list.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layer_list)
        # self.sigmoid = nn.Sigmoid()
        self._init_weights()

        # self.sigmoid = nn.Sigmoid()
        self.weights = torch.tensor([[5, 5, 10]]).cuda()
        self.loss = nn.MSELoss(reduction='none')

    def _init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加一个通道维度，使其成为 [N, 1, 8]

        feature_matrix = self.relu(self.bn_conv(self.conv(x)))
        feature_vector = feature_matrix.view(feature_matrix.size(0), -1)

        o = self.net(feature_vector)  # 最后一层使用relu比用sigmoid好，relu在作为后一层或者放在forward函数中都可以
        # o = F.relu((self.net(x))
        return o

    def calculate_loss(self, inp, label):
        weighted_loss = self.loss(inp, label) * self.weights
        loss = torch.mean(weighted_loss)

        return loss

class Direct_Predict_MOE_MLP(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()

        layer_sizes = [int(x) for x in layer_sizes]  # [9, 128, 256, 256, 64, 3]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1

        # Create separate layer lists for each expert
        self.net1 = self._create_network(layer_sizes, final_relu_layer)
        self.net2 = self._create_network(layer_sizes, final_relu_layer)
        self.net3 = self._create_network(layer_sizes, final_relu_layer)
        self.net4 = self._create_network(layer_sizes, final_relu_layer)

        self.att_wgt = nn.Linear(layer_sizes[0], 4)

        self._init_weights()

        # self.sigmoid = nn.Sigmoid()
        self.weights = torch.tensor([[1, 5, 5]]).cuda()
        self.loss = nn.MSELoss(reduction='none')

    def _create_network(self, layer_sizes, final_relu_layer):
        layer_list = []

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
            if i < final_relu_layer:
                # layer_list.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
                layer_list.append(nn.ReLU(inplace=False))
                # layer_list.append(nn.PReLU())
                # layer_list.append(nn.ELU(inplace=False))
            # if i < final_relu_layer:
            #     layer_list.append(nn.Dropout(0.01, inplace=False))
        layer_list.append(nn.ReLU(inplace=False))

        return nn.Sequential(*layer_list)

    def _init_weights(self):
        for net in [self.net1, self.net2, self.net3, self.net4]:
            for m in net:
                if type(m) == nn.Linear:
                    nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                    # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.uniform_(m.bias, -0.1, 0.1)

        nn.init.normal_(self.att_wgt.weight, mean=0.0, std=1e-2)
        # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.uniform_(self.att_wgt.bias, -0.1, 0.1)

    def forward(self, x):
        att_wgt = torch.softmax(self.att_wgt(x), dim=1)
        # print(att_wgt[1, :])

        o1 = self.net1(x)  #最后一层使用relu比用sigmoid好，relu在作为后一层或者放在forward函数中都可以
        o2 = self.net2(x)
        o3 = self.net3(x)
        o4 = self.net4(x)
        # print(o1[1, :])
        # print(o2[1, :])
        # print(o3[1, :])
        # print(o4[1, :])

        weighted_o = att_wgt[:, 0].unsqueeze(1) * o1 + att_wgt[:, 1].unsqueeze(1) * o2 + att_wgt[:, 2].unsqueeze(1) * o3 + att_wgt[:, 3].unsqueeze(1) * o4
        # print(weighted_o[1, :])
        return weighted_o

    def calculate_loss(self, inp, label):
        weighted_loss = self.loss(inp, label) * self.weights
        loss = torch.mean(weighted_loss)

        return loss


class Encoder(nn.Module):
    def __init__(self, layer_sizes):
        super(Encoder, self).__init__()
        layer_list = []

        layer_sizes = [int(x) for x in layer_sizes]  # [14, 128, 256, 64]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers 

        for i in range(len(layer_sizes) - 2):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))

            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))

        self.shared_net = nn.Sequential(*layer_list)
                
        self.fc_mu = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.fc_logvar = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        self._init_weights()

    def _init_weights(self):
        for m in self.shared_net:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                nn.init.uniform_(m.bias, -0.1, 0.1)

        for m in [self.fc_mu, self.fc_logvar]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-2)
            nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        x = self.shared_net(x)
       
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, layer_sizes):
        super(Decoder, self).__init__()
        layer_list = []

        layer_sizes = [int(x) for x in layer_sizes]  # [14, 128, 256, 64]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers 

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[-(i+1)]
            curr_size = layer_sizes[-(i+2)]
            layer_list.append(nn.Linear(input_size, curr_size))

            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))

        self.net = nn.Sequential(*layer_list)

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, z):
        output = self.net(z)

        return output  

class VAE(nn.Module):
    def __init__(self, layer_size, belta = 1):
        super(VAE, self).__init__()
        self.encoder = Encoder(layer_size)
        self.decoder = Decoder(layer_size)

        self.belta = belta

        self.loss = nn.MSELoss(reduction='none')

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        z = self.decoder(z)
        return z, mu, logvar

    def KL_div(self, mu, logvar, reduction='avg'):
        mu = mu.view(mu.size(0), mu.size(1))
        logvar = logvar.view(logvar.size(0), logvar.size(1))
        if reduction == 'sum':
            KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        
        return KL

    def calculate_loss(self, x, z, mu, logvar):
        recon_loss = torch.mean(torch.sum(self.loss(x, z), 1))
        kl_loss = torch.mean(self.KL_div(mu, logvar))

        loss = recon_loss + self.belta * kl_loss

        return loss

class Performance_Predict(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        
        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.bn1= nn.BatchNorm1d(layer_sizes[1])

        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.bn2 = nn.BatchNorm1d(layer_sizes[2])

        self.fc3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.bn3 = nn.BatchNorm1d(layer_sizes[3])

        self.fc4 = nn.Linear(layer_sizes[3], layer_sizes[4])

        self.relu = nn.ReLU(inplace=False)

        self._init_weights()

        self.weights = torch.tensor([[1, 5, 5]]).cuda()
        self.loss = nn.MSELoss(reduction='none')

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-2)
            nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        o1 = self.relu(self.bn1(self.fc1(x)))
        o2 = self.relu(self.bn2(self.fc2(o1)))
        o3 = self.relu(self.bn3(self.fc3(o2)))
        o = self.relu(self.fc4(o3))

        return o1, o2, o3, o

    def calculate_loss(self, inp, label):
        losses = self.loss(inp, label)
        loss = torch.mean(losses * self.weights)

        return loss

class Loss_Predict(nn.Module):
    def __init__(self, layer_sizes, inner_dim):
        super().__init__()
        
        self.fc1 = nn.Linear(layer_sizes[1], inner_dim)
        self.bn1= nn.BatchNorm1d(inner_dim)

        self.fc2 = nn.Linear(layer_sizes[2], inner_dim)
        self.bn2 = nn.BatchNorm1d(inner_dim)

        self.fc3 = nn.Linear(layer_sizes[3], inner_dim)
        self.bn3 = nn.BatchNorm1d(inner_dim)

        self.fc4 = nn.Linear(3*inner_dim, layer_sizes[4])

        self.relu = nn.ReLU(inplace=False)

        self.weights = torch.tensor([[1, 1, 1]]).cuda()

        self.loss = nn.MSELoss(reduction='none')

        self._init_weights()


    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-2)
            nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, o1, o2, o3):
        lo1 = self.relu(self.bn1(self.fc1(o1)))
        lo2 = self.relu(self.bn2(self.fc2(o2)))
        lo3 = self.relu(self.bn3(self.fc3(o3)))

        lo = torch.cat((lo1, lo2, lo3), dim=1)
        lo = self.relu(self.fc4(lo))

        return lo 

    def calculate_loss(self, inp, label):
        weighted_loss = self.loss(inp, label) * self.weights
        loss = torch.mean(weighted_loss)

        return loss

class Direct_Predict_MLP_nsg(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]  # [17, 128, 256, 64, 2]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))

            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
                layer_list.append(nn.ReLU(inplace=False))
            # if i < final_relu_layer:
            #     layer_list.append(nn.Dropout(0, inplace=False))

        layer_list.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layer_list)

        self._init_weights()

        self.weights = torch.tensor([[1, 20]]).cuda()

        self.loss = nn.MSELoss(reduction='none')

    def _init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        o = self.net(x)  # 最后一层使用relu比用sigmoid好，relu在作为后一层或者放在forward函数中都可以
        # o = F.relu(self.net(x))
        return o

    def get_feature_vectors(self, x, feature_layer=3):
        for i, layer in enumerate(self.net):
            x = layer(x)
            # We multiply by 3 because each 'block' includes three layers
            if i == feature_layer * 3 - 1:
                break

        return x

    def calculate_loss(self, inp, label):
        weighted_loss = self.loss(inp, label) * self.weights
        loss = torch.mean(weighted_loss)

        return loss

