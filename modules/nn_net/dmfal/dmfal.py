from copy import deepcopy
# import gpytorch
import math
import torch
import numpy as np
import os
import sys
import random

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)


from modules.nn_net.dmfal.BaseNet import AdaptiveBaseNet
from utils import *


# optimize for main_controller

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415


default_module_config = {
    # according to original inplement
    # h_w, h_d determine laten dim
    # net_param
    'nn_param': {
        'hlayers_w': [40, 40],
        'hlayers_d': [2, 2],
        'base_dim': [32, 32], # ?
        'activation': 'relu', # ['tanh','relu','sigmoid']
        'out_shape': [(100,1000), (100, 2000)],
        'in_shape': [(100, 5)],
    },
}


class DeepMFnet:
    # def __init__(self, grid_params_list, kernel_list, target_list, normalize=True, restrict_method= 'exp') -> None:
    def __init__(self, module_config) -> None:
        default_module_config.update(module_config)
        self.module_config = deepcopy(default_module_config)
        module_config = deepcopy(default_module_config)

        self.M = len(module_config['nn_param']['out_shape'])
        self.init_model_params()

    def get_train_params(self):
        opt_params = []
        for i in range(self.M):
            for nn_param_name, nn_param in self.nn_list[i].parameters().items():
                opt_params.append(nn_param)
            opt_params.append(self.log_tau_list[i])
        return {"params": opt_params}


    def init_model_params(self):
        nn_param = self.module_config['nn_param']

        self.nn_list = []
        self.log_tau_list = []

        in_shape = nn_param['in_shape']
        out_shape = nn_param['out_shape']

        for i in range(self.M):
            _layers = []
            if i == 0:
                _layers.append(in_shape[0][1]) #input dim
            else:
                _layers.append(in_shape[0][1] + nn_param['base_dim'][i-1])
            _layers.extend([nn_param['hlayers_w'][i]]* nn_param['hlayers_d'][i] ) #hidden dim
            _layers.append(nn_param['base_dim'][i]) #transition dim
            _layers.append(out_shape[i][1])
            self.nn_list.append(AdaptiveBaseNet(_layers, nn_param['activation'], 'cpu', torch.float))
            self.log_tau_list.append(torch.tensor(0.0, device='cpu', requires_grad=True, dtype=torch.float))


    def forward(self, x, m, sample=False):
        Y_m, base_m = self.nn_list[0].forward(x, sample)
        # propagate to the other fidelity levels
        for i in range(1,m+1):
            X_concat = torch.cat((base_m, x), dim=1)
            # print(X_concat.shape)
            Y_m, base_m = self.nn_list[i].forward(X_concat, sample)
        return Y_m, base_m


    def eval_llh(self, x, y, m):
        llh_samples_list = []
        y_shape = y.shape[0]
        x = x[:y_shape]
        pred_sample, _ = self.forward(x, m, sample=True)
        log_prob_sample = torch.sum(-0.5*torch.square(torch.exp(self.log_tau_list[m]))*torch.square(pred_sample-y) +\
                                self.log_tau_list[m] - 0.5*np.log(2*np.pi))
        llh_samples_list.append(log_prob_sample)
        return sum(llh_samples_list)

    # def batch_eval_kld(self):
    #     kld_list = []
    #     for m in range(self.M):
    #         kld_list.append(self.nn_list[m]._eval_kld())
    #     return sum(kld_list)

    # def batch_eval_reg(self):
    #     reg_list = []
    #     for m in range(self.M):
    #         reg_list.append(self.nn_list[m]._eval_reg())
    #     return sum(reg_list)

    def predict(self, x_list, sample=False):
        y_predict, _ = self.forward(x_list[0], self.M - 1, sample)
        return y_predict


    def compute_loss(self, x_list, y_list):
        # inputs should as [x, y_0, y_1, ..., y_n]
        llh_list = []
        for m in range(self.M):
            # llh_m = self.eval_llh(x_list[m], y_list[m], m)
            llh_m = self.eval_llh(x_list[0], y_list[m], m)
            llh_list.append(llh_m)
        loss = - sum(llh_list)
        return loss

