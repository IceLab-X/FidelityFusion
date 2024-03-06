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
    def __init__(self, x_dimension, fidelity_num, data_manager, posterior_function, model_cost, seed):
        super(upper_confidence_bound, self).__init__()

        # 传入的用于计算的函数/参数
        self.fidelity_num = fidelity_num
        self.x_dimension = x_dimension
        self.data_manager = data_manager
        self.pre_func = posterior_function

        # select criteria
        self.beta = 0.2 * int(self.x_dimension) * math.log(int(seed[1]+1.1))
        self.gamma = 0.01
        self.seed = seed[0]

    def compute_next(self):
        # np.random.seed(self.seed+1007)
        torch.manual_seed(self.seed+1007)
        N = 100
        self.x_range =torch.rand(N, 1).double()
        mean_low, var_low = self.pre_func(self.data_manager, self.x_range, 1)
        mean_high, var_high = self.pre_func(self.data_manager, self.x_range, 2)
        ucb_low = mean_low + self.beta * torch.diag(var_low)
        ucb_high = mean_high + self.beta * torch.diag(var_high)
        # new_x = s
        idx = torch.argmax(torch.cat((ucb_low, ucb_high), 0))
        if idx >= N:
            idx = idx - N
        new_x = self.x_range[idx]
        new_x = new_x.reshape(1, self.x_dimension)

        m, v = self.pre_func(self.data_manager, new_x, 1)
        print(self.beta * v)
        if self.beta * v > self.gamma:
            new_s = 1
        else:
            new_s = 2
        return new_x, new_s

    def get_value(self, x):

        mean_low, var_low = self.pre_func(x, 1)
        mean_high, var_high = self.pre_func(x, 2)
        ucb_low = mean_low + self.beta * var_low
        ucb_high = mean_high + self.beta * var_high

        return ucb_low, ucb_high

