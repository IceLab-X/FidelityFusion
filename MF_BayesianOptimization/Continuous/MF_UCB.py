# -*- coding = utf-8 -*-
# @Time : 25/9/23 14:31
# @Author : Alison_W
# @File : MF_UCB_optimise.py
# @Software : PyCharm
import torch
import math
import numpy as np
import torch.nn as nn

class upper_confidence_bound_continuous(nn.Module):
    def __init__(self, x_dimension, search_range, posterior_function, model_cost, seed):
        super(upper_confidence_bound_continuous, self).__init__()
        self.N = 300
        self.x_dimension = x_dimension
        self.search_range = search_range
        tem = []
        for i in range(self.x_dimension):
            np.random.seed(seed[0] + 117 + i)
            tt = np.random.rand(self.N, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            tem.append(tt)
        tt = np.concatenate(tem, axis=1)
        self.x_range = tt
        np.random.seed(seed[0] + 127)
        self.z_range = np.array(sorted(np.random.rand(self.N)*(search_range[-1][1]-search_range[-1][0])+search_range[-1][0])).reshape(-1, 1)

        # 传入的用于计算的函数/参数
        self.pre_func = posterior_function
        self.model_cost = model_cost
        self.log_length_scale = nn.Parameter(torch.zeros(x_dimension))    # ARD length scale
        self.log_scale = nn.Parameter(torch.zeros(1))   # kernel scale

        # select criteria
        self.seed = seed[0]
        self.beta = 0.5 * int(self.x_dimension) * math.log(2 * int(seed[1]) + 1)
        self.d = x_dimension
        self.k_0 = 1
        self.p = 1

    def kernel(self, X1, X2):
    # the common RBF kernel
        X1 = X1 / self.log_length_scale.exp()
        X2 = X2 / self.log_length_scale.exp()
        X1_norm2 = torch.sum(X1 * X1, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(0))  #this is the effective Euclidean distance matrix between X1 and X2.
        K = self.log_scale.exp() * torch.exp(-0.5 * K)
        return K

    def information_gap(self, input):
        if input == None:
            input = self.z_range
        else:
            input = np.ones(1).reshape(-1, 1)*input

        phi = self.kernel(torch.from_numpy(input), torch.ones(1).reshape(-1, 1).double())
        phi = phi.detach().numpy()
        ksin = np.sqrt(1-np.power(phi, 2))
        return ksin

    def gamma_z(self, ksin_z):
        q = 1 / (self.p + self.d + 2)
        lambda_balance = np.power(self.model_cost.compute_cost(self.z_range)/self.model_cost.compute_cost(1), q)
        gamma_z = np.sqrt(self.k_0) * ksin_z * lambda_balance
        return gamma_z

    # def negative_ucb(self):
    #     mean, var = self.pre_func(self.x, np.ones(1).reshape(-1, 1)*self.search_range[-1][-1])
    #     # mean, var = self.pre_func(self.x)
    #     ucb = mean + self.beta * var
    #     return -ucb

    # def optimise_adam(self, niteration, lr):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    #     # optimizer.zero_grad()
    #     for i in range(niteration):
    #         optimizer.zero_grad()
    #         loss = self.negative_ucb()
    #         loss.backward()
    #         optimizer.step()
    #         print('iter'+str(i)+'/'+str(niteration), 'loss_negative_ucb:', loss.item(), end='\r')

    def compute_next(self):

        # optimize x
        np.random.seed(self.seed+10086)
        # self.x = nn.Parameter(torch.from_numpy(tt.reshape(1, self.x_dimension)).double(),  requires_grad=True)
        # self.optimise_adam(niteration=20, lr=0.01)

        mean, var = self.pre_func(self.x_range, self.z_range)
        ucb = mean + self.beta * var
        # new_x = s
        idx = np.argmax(ucb.detach().numpy())
        new_x = self.x_range[idx]
        new_x = new_x.reshape(1, self.x_dimension)
        new_s = self.z_range[idx].reshape(1, 1)

        # tau_z_mean = []
        # tau_z_std = []

        # np.ones(1).reshape(1, 1) * self.search_range[-1][-1]
        # tau_z_mean = []
        # tau_z_std = []
        # for z in self.z_range:
        #     z = z.reshape(-1, 1)
        #     m, v = self.pre_func(new_x, z)
        #     tau_z_mean.append(m.detach().numpy())
        #     tau_z_std.append(np.sqrt(v.detach().numpy()))
        #
        # ksin_z = self.information_gap(None)
        # gamma_z = self.gamma_z(ksin_z)
        #
        # possible_z = []
        # for i in range(self.z_range.shape[0]):
        #     condition_1 = tau_z_std[i][0][0] > gamma_z[i]
        #     condition_2 = ksin_z[i] > self.information_gap(np.sqrt(self.p)) / np.sqrt(self.beta)
        #     if condition_1 and condition_2:
        #         possible_z.append(self.z_range[i])
        #
        # if len(possible_z) == 0:
        #     new_s = 0.1
        # else:
        #     new_s = min(possible_z)

        if isinstance(new_x, torch.Tensor):
            new_x = new_x.detach().numpy()

        return new_x, new_s
