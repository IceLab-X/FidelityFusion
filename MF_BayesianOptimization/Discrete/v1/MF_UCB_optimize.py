# -*- coding = utf-8 -*-
# @Time : 25/9/23 14:31
# @Author : Alison_W
# @File : MF_UCB_optimise.py
# @Software : PyCharm
import torch
import math
import torch.nn as nn
import numpy as np

class upper_confidence_bound(nn.Module):
    def __init__(self, x_dimension, fidelity_num, posterior_function, model_cost, seed):
        super(upper_confidence_bound, self).__init__()

        # 传入的用于计算的函数/参数
        self.fidelity_num = fidelity_num
        self.x_dimension = x_dimension
        self.pre_func = posterior_function

        # select criteria
        self.beta = 0.2 * int(self.x_dimension) * math.log(int(seed[1]+1))
        self.gamma = 1.0
        self.seed = seed[0]

        # Optimizer para/target
        # self.x = nn.Parameter(torch.ones(x_dimension))

    def optimise_adam(self, fidelity_indicator, niteration=100, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_ucb(fidelity_indicator)
            loss.backward()
            optimizer.step()
            self.x.data.clamp_(0.0, 1.0)
            print('iter', i, 'x:', self.x, 'loss_negative_ucb:',loss.item(), end='\n')

    def negative_ucb(self, fidelity_indicator):
        mean, var = self.pre_func(self.x, fidelity_indicator)
        ucb = mean + self.beta * var
        return -ucb

    def compute_next(self):
        N_UCB = []
        UCB_x = []
        for i in range(self.fidelity_num):
            np.random.seed(self.seed+i+10086)
            tt = np.random.rand(self.x_dimension)
            print(tt)
            self.x = nn.Parameter(torch.from_numpy(tt.reshape(1, self.x_dimension)).double())
            self.optimise_adam(fidelity_indicator=i+1, niteration=15, lr=0.01)
            UCB_x.append(self.x.detach())
            N_UCB.append(self.negative_ucb(fidelity_indicator=i+1))

        new_x = UCB_x[N_UCB.index(min(N_UCB))]


        m, v = self.pre_func(new_x, 1)

        if self.beta * v > self.gamma:
            new_s = 1
        else:
            new_s = 2
        return new_x, new_s


