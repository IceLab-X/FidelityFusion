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
from utils.data_utils import data_register
from utils.mlgp_hook import set_function_as_module_to_catch_error
from modules.gp_module.basic_gp_model import BASE_GP_MODEL


default_config = {
    'noise': 1.,
    'noise_exp_format': True,
    'kernel': {
                'K1': {'SE': {'noise_exp_format':True, 'length_scale':1., 'scale': 1.}},
              },
}

class CIGP_MODULE(BASE_GP_MODEL):
    def __init__(self, gp_model_config=None) -> None:
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

    def _get_noise_according_exp_format(self):
        if self.gp_model_config['noise_exp_format'] is True:
            return torch.exp(self.noise)
        else:
            return self.noise

    def predict_with_var(self, inputs, vars=None):
        if self.already_set_train_data is False:
            assert False, "gp model model hasn't been trained. predict failed"
    
        with torch.no_grad():
            Sigma = self.kernel_list[0](self.inputs_tr[0], self.inputs_tr[0]) + JITTER * torch.eye(self.inputs_tr[0].size(0), device=list(self.parameters())[0].device)
            _noise = self._get_noise_according_exp_format()
            Sigma = Sigma + _noise.pow(-1) * torch.eye(self.inputs_tr[0].size(0), device=list(self.parameters())[0].device)
            self.k_l_inv = Sigma

            kx = self.kernel_list[0](self.inputs_tr[0], inputs[0])
            self.kstar = kx
            L = torch.linalg.cholesky(Sigma)
            LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

            u = kx.t() @ torch.cholesky_solve(self.outputs_tr[0], L)

            var_diag = self.kernel_list[0](inputs[0], inputs[0]).diag().view(-1, 1) - (LinvKx**2).sum(dim = 0).view(-1, 1)
            _noise = self._get_noise_according_exp_format()
            var_diag = var_diag + _noise.pow(-1)
            var_diag = var_diag.expand_as(u)
        return u, var_diag

    def compute_loss(self, inputs, outputs, inputs_var=None, outputs_var=None):
        # TODO checking if inputs/outputs was changed
        self.inputs_tr = inputs
        self.outputs_tr = outputs
        self.already_set_train_data = True

        Sigma = self.kernel_list[0](inputs[0], inputs[0]) + JITTER * torch.eye(inputs[0].size(0), device=list(self.parameters())[0].device)
        _noise = self._get_noise_according_exp_format()

        Sigma = Sigma + _noise.pow(-1) * torch.eye(inputs[0].size(0), device=list(self.parameters())[0].device)

        L = torch.linalg.cholesky(Sigma)
        #option 1 (use this if torch supports)
        # Gamma,_ = torch.triangular_solve(self.Y, L, upper = False)
        #option 2

        gamma = L.inverse() @ outputs[0]       # we can use this as an alternative because L is a lower triangular matrix.

        y_num, y_dimension = outputs[0].shape
        nll =  0.5 * (gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI, device=list(self.parameters())[0].device)) * y_dimension
        return nll

    def get_train_params(self):
        train_params_list = {}
        train_params_list['noise'] = self.noise

        train_params_list['kernel'] = []
        for _k in self.kernel_list:
            _k.get_param(train_params_list['kernel'])
        return train_params_list


def basic_test():
    # prepare data
    x = np.load('./data/sample/input.npy')
    y = np.load('./data/sample/output_fidelity_2.npy')
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()

    # normalizer now is outsider of the model
    from utils.normalizer import Dateset_normalize_manager
    dnm = Dateset_normalize_manager([x], [y])
    x = dnm.normalize_inputs([x])[0]        # reset x = norm_x
    y = dnm.normalize_outputs([y])[0]       # resrt y = norm_y

    train_x = [x[:128,:]]
    train_y = [y[:128,...].reshape(128, -1)]
    eval_x = [x[128:,:]]
    eval_y = [y[128:,...].reshape(128, -1)]
    source_shape = y[128:,...].shape

    # init model
    cigp = CIGP_MODULE(default_config)

    # init optimizer, optimizer is also outsider of the model
    lr_dict = {'kernel': 0.01, 'noise': 0.01}
    params_dict = cigp.get_train_params()
    optimizer_dict = [{'params': params_dict[_key], 'lr': lr_dict[_key]} for _key in params_dict.keys()]
    optimizer = torch.optim.Adam(optimizer_dict)
    
    for epoch in range(300):
        print('epoch {}/{}'.format(epoch+1, 300), end='\r')
        optimizer.zero_grad()
        loss = cigp.compute_loss(train_x, train_y)
        print('loss_nll:', loss.item())
        loss.backward()
        optimizer.step()
    print('\n')
    cigp.eval()
    predict_y = cigp.predict(eval_x)

    # plot result
    from visualize_tools.plot_field import plot_container
    data_list = [eval_y[0], predict_y[0], (eval_y[0] - predict_y[0]).abs()]
    data_list = [_d.reshape(source_shape) for _d in data_list]
    label_list = ['groundtruth', 'predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()


def gp_model_block_test():
    # prepare data
    x = np.load('./data/sample/input.npy')
    y = np.load('./data/sample/output_fidelity_2.npy')
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    data_len = x.shape[0]
    source_shape = [-1, *y.shape[1:]]

    x = x.reshape(data_len, -1)
    y = y.reshape(data_len, -1)
    train_x = [x[:128,:]]
    train_y = [y[:128,:]]
    eval_x = [x[128:,:]]
    eval_y = [y[128:,:]]

    # normalizer now is outsider of the model
    from utils.normalizer import Dateset_normalize_manager
    dnm = Dateset_normalize_manager([train_x[0]], [train_y[0]])

    # init model
    cigp = CIGP_MODULE(default_config)

    from gp_model_block import GP_model_block
    gp_model_block = GP_model_block()
    gp_model_block.dnm = dnm
    gp_model_block.gp_model = cigp

    # init optimizer, optimizer is also outsider of the model
    lr_dict = {'kernel': 0.01, 'noise': 0.01}
    params_dict = gp_model_block.get_train_params()
    optimizer_dict = [{'params': params_dict[_key], 'lr': lr_dict[_key]} for _key in params_dict.keys()]
    optimizer = torch.optim.Adam(optimizer_dict)
    
    for epoch in range(300):
        print('epoch {}/{}'.format(epoch+1, 300), end='\r')
        optimizer.zero_grad()
        loss = gp_model_block.compute_loss(train_x, train_y)
        print('loss_nll:', loss.item())
        loss.backward()
        optimizer.step()

    print('\n')
    gp_model_block.eval()
    predict_y = gp_model_block.predict(eval_x)

    # plot result
    from visualize_tools.plot_field import plot_container
    data_list = [eval_y[0], predict_y[0], (eval_y[0] - predict_y[0]).abs()]
    data_list = [_d.reshape(source_shape) for _d in data_list]
    label_list = ['groundtruth', 'predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()


if __name__ == '__main__':
    # basic_test()
    gp_model_block_test()

