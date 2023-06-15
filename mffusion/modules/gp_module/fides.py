# import gpytorch
import torch
import numpy as np
import os
import sys

from copy import deepcopy

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils import *
from modules.kernel import kernel_utils
from modules.gp_module.basic_gp_model import BASE_GP_MODEL

default_config = {
    'noise': 1.,
    'noise_exp_format': True,
    'kernel': {
                'K1': {'kernel_res': {'noise_exp_format':True, 'length_scale':1., 'scale': 1., 
                                      'length_scale_z':1.}},
              },
}

class FIDES_MODULE(BASE_GP_MODEL):
    def __init__(self, gp_model_config) -> None:
        super().__init__(gp_model_config)
        _final_config = smart_update(default_config, gp_model_config)
        self.gp_model_config = _final_config

        if self.gp_model_config['noise_exp_format'] is True:
            self.noise = torch.nn.Parameter(torch.log(torch.tensor(self.gp_model_config['noise'], dtype=torch.float32)))
        else:
            self.noise = torch.nn.Parameter(torch.tensor(self.gp_model_config['noise'], dtype=torch.float32))

        self.inputs_tr = None
        self.outputs_tr = None
        self.already_set_train_data = False
        self.kernel_list = None
        kernel_utils.register_kernel(self, self.gp_model_config['kernel'])

        self.fi_define=False

    def set_fidelity(self, l1, h1, l2, h2):
        self.l1 = l1
        self.h1 = h1
        self.l2 = l2
        self.h2 = h2
        self.fi_define=True


    def _get_noise_according_exp_format(self):
        if self.gp_model_config['noise_exp_format'] is True:
            return torch.exp(self.noise)
        else:
            return self.noise
        

    def predict_with_var(self, inputs, vars=None):
        x_test = inputs[0]
        l1, h1, l2, h2 = self.l1, self.h1, self.l2, self.h2
        _noise = self._get_noise_according_exp_format()

        sigma = self.kernel_list[0](self.inputs_tr[0], self.inputs_tr[0], l1, h1, l2, h2) + _noise.pow(-1) * torch.eye(self.inputs_tr[0].size(0), device=list(self.parameters())[0].device)
        sigma = sigma + JITTER * torch.eye(self.inputs_tr[0].size(0), device=list(self.parameters())[0].device)

        kx = self.kernel_list[0](self.inputs_tr[0], x_test, l1, h1, l2, h2)
        L = torch.cholesky(sigma)
        LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

        mean = kx.t() @ torch.cholesky_solve(self.outputs_tr[0], L)  # torch.linalg.cholesky()
        
        var_diag = self.kernel_list[0](x_test, x_test, l1, h1, l2, h2).diag().view(-1, 1) \
            - (LinvKx**2).sum(dim = 0).view(-1, 1)
        # add the noise uncertainty
        var_diag = var_diag + _noise.pow(-1)

        return mean, var_diag


    def compute_loss(self, inputs, outputs):
        self.inputs_tr = inputs
        self.outputs_tr = outputs
        self.already_set_train_data = True

        _noise = self._get_noise_according_exp_format()
        x = inputs[0]
        y = outputs[0]
        l1, h1, l2, h2 = self.l1, self.h1, self.l2, self.h2

        sigma = self.kernel_list[0](x, x, l1, h1, l2, h2) + _noise.pow(-1) * torch.eye(x.size(0), device=list(self.parameters())[0].device)
        sigma = sigma + JITTER * torch.eye(x.size(0), device=list(self.parameters())[0].device)

        L = torch.linalg.cholesky(sigma)
        y_num, y_dimension = y.shape
        Gamma,_ = torch.triangular_solve(y, L, upper = False)
        nll =  0.5 * (Gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension

        return nll
    

    def get_train_params(self):
        train_params_list = {}
        train_params_list['noise'] = self.noise

        train_params_list['kernel'] = []
        for _k in self.kernel_list:
            _k.get_param(train_params_list['kernel'])
        return train_params_list