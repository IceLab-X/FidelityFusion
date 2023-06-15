# import gpytorch
import torch
import numpy as np
import os
import sys
import math
from copy import deepcopy

import tensorly
from tensorly import tucker_to_tensor
tensorly.set_backend('pytorch')


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
    'noise_exp_format': False,
    'kernel': {
                'K1': {'SE': {'noise_exp_format':True, 'length_scale':1., 'scale': 1.}},
              },
    'output_shape': None,

    'learnable_grid': False,
    'learnable_mapping': False,
}

class HOGP_MODULE(BASE_GP_MODEL):
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

        self.K = []
        self.K_eigen = []

        # broadcast kernel
        if self.gp_model_config['output_shape'] is None:
            raise ValueError('output_shape must be set')
        for _k in range(len(self.gp_model_config['output_shape'])):
            self.gp_model_config['kernel']['K{}'.format(_k+2)] = deepcopy(self.gp_model_config['kernel']['K1'])
        kernel_utils.register_kernel(self, self.gp_model_config['kernel'])

        # set grid
        self.grid = []
        for _value in self.gp_model_config['output_shape']:
            self.grid.append(torch.nn.Parameter(torch.tensor(range(_value)).reshape(-1,1).float()))
        if self.gp_model_config['learnable_grid'] is False:
            for i in range(len(self.grid)):
                self.grid[i].requires_grad = False

        # set mapping
        self.mapping_vector = []
        for _value in self.gp_model_config['output_shape']:
            self.mapping_vector.append(torch.nn.Parameter(torch.eye(_value)))
        if self.gp_model_config['learnable_mapping'] is False:
            for i in range(len(self.mapping_vector)):
                self.mapping_vector[i].requires_grad = False

    def _get_noise_according_exp_format(self):
        if self.gp_model_config['noise_exp_format'] is True:
            return torch.exp(self.noise)
        else:
            return self.noise

    def predict_with_var(self, inputs, inputs_vars=None):
        if self.already_set_train_data is False:
            assert False, "gp model model hasn't been trained. predict failed"

        with torch.no_grad():
            # Get predict mean
            K_star = self.kernel_list[0](inputs[0], self.inputs_tr[0])
            K_predict = [K_star] + self.K[1:]

            '''
            predict_u = tensorly.tenalg.kronecker(K_predict)@self.g_vec #-> memory error
            so we use tensor.tenalg.multi_mode_dot instead
            '''
            predict_u = tensorly.tenalg.multi_mode_dot(self.g, K_predict)

            # /*** Get predict var***/
            # NOTE: now only work for the normal predict
            n_dim = len(self.K_eigen) - 1
            _init_value = torch.tensor([1.0]).reshape(*[1 for i in range(n_dim)])
            diag_K_dims = tucker_to_tensor(( _init_value, [K.diag().reshape(-1,1) for K in self.K[1:]]))
            diag_K_dims = diag_K_dims.unsqueeze(0)
            diag_K_x = self.kernel_list[0](inputs[0], inputs[0]).diag()
            for i in range(n_dim):
                diag_K_x = diag_K_x.unsqueeze(-1)
            diag_K = diag_K_x*diag_K_dims

            S = self.A * self.A.pow(-1/2)
            S_2 = S.pow(2)
            # S_product = tensorly.tenalg.multi_mode_dot(S_2, [(K_star@K_p.inverse()@eigen_vector_p).pow(2), eigen_vector_d1.pow(2), eigen_vector_d2.pow(2)])
            eigen_vectors_x = K_star@self.K[0] + JITTER*torch.eye(K_star.shape[0], self.K[0].shape[0]).pow(2)
            eigen_vectors_dims = [self.K_eigen[i+1].vector.pow(2) for i in range(n_dim)]
            
            eigen_vectors = [eigen_vectors_x] + eigen_vectors_dims
            S_product = tensorly.tenalg.multi_mode_dot(S_2, eigen_vectors)
            var_diag = diag_K + S_product

            # /*** Get predict var***/
            # if inputs_vars is not None:
            #     diag_K_vars = self.kernel_list[0](inputs_vars[0], inputs_vars[0]).diag()* diag_K_dims
            #     var_diag = var_diag + diag_K_vars
            # if getattr(self, 'outputs_tr_var', None) is not None:
            #     _noise = self._get_noise_according_exp_format()
            #     k_r_inv = torch.inverse(self.K[0]) + _noise.pow(-1)* tensorly.ones(self.K[0].size(0)) + JITTER
            #     # gamma = K_star.T@k_r_inv - self.k_star_l.T@self.k_l_inv
            #     gamma = K_star.T@k_r_inv
            #     pred_uncertainty = gamma @ self.outputs_tr_var[0] @ gamma.T
            #     var_diag += pred_uncertainty.diag().reshape(-1, 1)

        return predict_u, var_diag

    def compute_loss_with_var(self, inputs, outputs, inputs_var=None, outputs_var=None):
        # TODO checking if inputs/outputs was changed
        self.inputs_tr = inputs
        self.outputs_tr = outputs
        self.inputs_tr_var = inputs_var
        self.outputs_tr_var = outputs_var
        self.already_set_train_data = True

        # compute kernel
        self.K.clear()
        self.K_eigen.clear()

        self.K.append(self.kernel_list[0](inputs[0], inputs[0]))
        self.K_eigen.append(eigen_pairs(self.K[-1]))

        for i in range(0, len(self.kernel_list)-1):
            _in = tensorly.tenalg.mode_dot(self.grid[i], self.mapping_vector[i], 0)
            self.K.append(self.kernel_list[i+1](_in, _in))
            self.K_eigen.append(eigen_pairs(self.K[-1]))

        # Kruskal operator
        # compute log(|S|) = sum over the logarithm of all the elements in A. O(nd) complexity.
        _init_value = torch.tensor([1.0],  device=list(self.parameters())[0].device).reshape(*[1 for i in self.K])
        lambda_list = [eigen.value.reshape(-1, 1) for eigen in self.K_eigen]
        A = tucker_to_tensor((_init_value, lambda_list))

        _noise = self._get_noise_according_exp_format()

        A = A + _noise.pow(-1)* tensorly.ones(A.shape,  device=list(self.parameters())[0].device)
        if outputs_var is not None:
            A += outputs_var[0]
        
        # TODO: add jitter limite here?
        # vec(z).T@ S.inverse @ vec(z) = b.T @ b,  b = S.pow(-1/2) @ vec(z)
        T_1 = tensorly.tenalg.multi_mode_dot(self.outputs_tr[0], [eigen.vector.T for eigen in self.K_eigen])
        T_2 = T_1 * A.pow(-1/2)
        T_3 = tensorly.tenalg.multi_mode_dot(T_2, [eigen.vector for eigen in self.K_eigen])
        b = tensorly.tensor_to_vec(T_3)

        # g = S.inverse@vec(z)
        g = tensorly.tenalg.multi_mode_dot(T_1 * A.pow(-1), [eigen.vector for eigen in self.K_eigen])
        # g_vec = tensorly.tensor_to_vec(g)

        self.b = b
        self.A = A
        self.g = g
        # self.g_vec = g_vec

        nd = torch.prod(torch.tensor([value for value in self.A.shape]))
        loss = -1/2* nd * torch.log(torch.tensor(2 * math.pi, device=list(self.parameters())[0].device))
        loss += -1/2* torch.log(self.A).sum()
        loss += -1/2* self.b.t() @ self.b

        loss = -loss/nd
        return loss

    def get_train_params(self):
        train_params_list = {}
        train_params_list['noise'] = self.noise

        train_params_list['kernel'] = []
        for _k in self.kernel_list:
            _k.get_param(train_params_list['kernel'])

        train_params_list['others'] = []
        train_params_list['others'].extend(self.mapping_vector)
        train_params_list['others'].extend(self.grid)

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
    train_y = [y[:128,...]]
    eval_x = [x[128:,:]]
    eval_y = [y[128:,...]]
    source_shape = y[128:,...].shape

    # init model
    test_config = deepcopy(default_config)
    test_config['output_shape'] = train_y[0][0,...].shape
    hogp = HOGP_MODULE(test_config)

    # init optimizer, optimizer is also outsider of the model
    lr_dict = {'kernel': 0.01, 'noise': 0.01, 'others': 0.01}
    params_dict = hogp.get_train_params()
    optimizer_dict = [{'params': params_dict[_key], 'lr': lr_dict[_key]} for _key in params_dict.keys()]
    optimizer = torch.optim.Adam(optimizer_dict)
    
    max_epoch = 100
    for epoch in range(max_epoch):
        print('epoch {}/{}'.format(epoch+1, max_epoch), end='\r')
        optimizer.zero_grad()
        loss = hogp.compute_loss(train_x, train_y)
        print('loss_nll:', loss.item())
        loss.backward()
        optimizer.step()
    print('\n')
    hogp.eval()
    predict_y = hogp.predict(eval_x)

    # plot result
    from visualize_tools.plot_field import plot_container
    data_list = [eval_y[0], predict_y[0].get_mean(), (eval_y[0] - predict_y[0].get_mean()).abs()]
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

    train_x = [x[:128,:]]
    train_y = [y[:128,:]]
    eval_x = [x[128:,:]]
    eval_y = [y[128:,:]]
    source_shape = y[128:,...].shape

    # normalizer now is outsider of the model
    from utils.normalizer import Dateset_normalize_manager
    dnm = Dateset_normalize_manager([train_x[0]], [train_y[0]])

    # init model
    test_config = deepcopy(default_config)
    test_config['output_shape'] = train_y[0][0,...].shape
    hogp = HOGP_MODULE(test_config)

    from gp_model_block import GP_model_block
    gp_model_block = GP_model_block()
    gp_model_block.dnm = dnm
    gp_model_block.gp_model = hogp

    # init optimizer, optimizer is also outsider of the model
    lr_dict = {'kernel': 0.01, 'noise': 0.01, 'others': 0.01}
    params_dict = gp_model_block.get_train_params()
    optimizer_dict = [{'params': params_dict[_key], 'lr': lr_dict[_key]} for _key in params_dict.keys()]
    optimizer = torch.optim.Adam(optimizer_dict)
    
    max_epoch = 100
    for epoch in range(max_epoch):
        print('epoch {}/{}'.format(epoch+1, max_epoch), end='\r')
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
    data_list = [eval_y[0], predict_y[0].get_mean(), (eval_y[0] - predict_y[0].get_mean()).abs()]
    data_list = [_d.reshape(source_shape) for _d in data_list]
    label_list = ['groundtruth', 'predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()


if __name__ == '__main__':
    # basic_test()
    gp_model_block_test()

