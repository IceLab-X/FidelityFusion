# -*- coding = utf-8 -*-
# @Time : 25/9/23 14:31
# @Author : Alison_W
# @File : MF_UCB_optimise.py
# @Software : PyCharm
import torch
import numpy as np
import torch.nn as nn

# class MF_acquisition_function(nn.Module):

def MF_acq_optimise(self, niteration, lr):
    
def MF_acq_next():
    
def MF_acq_compute_next(mf_acq_func, search_range, model_cost, seed):
    
    # optimize x
    np.random.seed(self.seed+10086)
    tem = []
    for i in range(self.x_dimension):
        tt = np.random.rand(1, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
        tem.append(tt)
    tt = np.concatenate(tem, axis=1)
    print(tt)
    self.x = nn.Parameter(torch.from_numpy(tt.reshape(1, self.x_dimension)).double(),  requires_grad=True)
    self.optimise_adam(niteration=20, lr=0.01)

    new_x = self.x.detach()

    tau_z_mean = []
    tau_z_std = []

    # np.ones(1).reshape(1, 1) * self.search_range[-1][-1]
    tau_z_mean = []
    tau_z_std = []
    for z in self.z_range:
        z = z.reshape(-1, 1)
        m, v = self.pre_func(new_x, z)
        tau_z_mean.append(m.detach().numpy())
        tau_z_std.append(np.sqrt(v.detach().numpy()))

    ksin_z = self.information_gap(None)
    gamma_z = self.gamma_z(ksin_z)

    possible_z = []
    for i in range(self.z_range.shape[0]):
        condition_1 = tau_z_std[i][0][0] > gamma_z[i]
        condition_2 = ksin_z[i] > self.information_gap(np.sqrt(self.p)) / np.sqrt(self.beta)
        if condition_1 and condition_2:
            possible_z.append(self.z_range[i])

    if len(possible_z) == 0:
        new_s = 0.1
    else:
        new_s = min(possible_z)

    if isinstance(new_x, torch.Tensor):
        new_x = new_x.detach().numpy()

    return new_x, new_s

    
class MF_UCB(nn.Module):
       # mean_func is also an nn.Module taking in input x, fidelity indicator t, and returning the acquisition function value
       # the input to MF_UCB is also x, t
       # they should be considered as two concatenating layers of a neural network
    def __init__(self, mean_func, var_func, cost_func):
        super(upper_confidence_bound_continuous, self).__init__()
        self.mean_func = mean_func
        self.var_func = var_func
        self.cost_func = cost_func
        
        # select criteria
        self.seed = seed
        self.beta = 1.0
        self.d = x_dimension
        self.k_0 = 1
        self.p = 1

    def information_gap(self, input):
        if input == None:
            input = self.z_range
        else:
            input = np.ones(1).reshape(-1, 1)*input

        phi = self.kernel(torch.from_numpy(input), torch.ones(1).reshape(-1, 1).double())
        phi = phi.detach().numpy()
        ksin = np.sqrt(1-np.power(phi, 2))
        return ksin

    # gamma_z is ？？？
    def gamma_z(self, ksin_z):
        q = 1 / (self.p + self.d + 2)
        lambda_balance = np.power(self.model_cost.compute_cost(self.z_range)/self.model_cost.compute_cost(1), q)
        gamma_z = np.sqrt(self.k_0) * ksin_z * lambda_balance
        return gamma_z

    def negative_ucb(self):
        mean, var = self.pre_func(self.x, np.ones(1).reshape(-1, 1)*self.search_range[-1][-1])
        # mean, var = self.pre_func(self.x)
        ucb = mean + self.beta * var
        return -ucb
    
  
  class MF_ES():
    def __init__(self, mean_func, var_func, cost_func):
        super(upper_confidence_bound_continuous, self).__init__()
        self.mean_func = mean_func
        self.var_func = var_func
        self.cost_func = cost_func
        
        # select criteria
        self.seed = seed
        self.beta = 1.0
        self.d = x_dimension
        self.k_0 = 1
        self.p = 1
