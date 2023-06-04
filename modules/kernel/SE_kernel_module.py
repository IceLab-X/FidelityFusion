import torch
import os
import sys

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.mlgp_decorator import *

@class_init_param_check
class SE_kernel(torch.nn.Module):
    # Squared Exponential Kernel
    def __init__(self, noise_exp_format, length_scale=1., scale=1.) -> None:
        super().__init__()
        self.noise_exp_format = noise_exp_format

        length_scale = torch.tensor(length_scale)
        scale = torch.tensor(scale)

        if noise_exp_format is True:
            self.length_scale = torch.nn.Parameter(torch.log(length_scale))
            self.scale = torch.nn.Parameter(torch.log(scale))
        else:
            self.length_scale = torch.nn.Parameter(length_scale)
            self.scale = torch.nn.Parameter(scale)

        # TODO should we add noise here?

    def forward(self, X, X2):
        if self.noise_exp_format is True:
            length_scale = torch.exp(self.length_scale).view(1, -1)
            scale = torch.exp(self.scale).view(1, -1)
        else:
            length_scale = self.length_scale.view(1, -1)
            scale = self.scale.view(1, -1)

        # optimize for multi dim.
        if len(X.shape)>2 :
            assert len(X2.shape)>2, "X and X2 should be same dim"
            X = X.reshape(X.size(0), -1)
            X2 = X2.reshape(X2.size(0), -1)

        X = X / length_scale.expand(X.size(0), length_scale.size(1))
        X2 = X2 / length_scale.expand(X2.size(0), length_scale.size(1))

        X_norm2 = torch.sum(X * X, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        # compute effective distance
        K = -2.0 * X @ X2.t() + X_norm2.expand(X.size(0), X2.size(0)) + X2_norm2.t().expand(X.size(0), X2.size(0))
        K = scale * torch.exp(-0.5 * K)

        return K

    # def get_param(self):
    #     return self.length_scale, self.log_scale

    def get_param(self, optimize_param_list):
        optimize_param_list.append(self.length_scale)
        optimize_param_list.append(self.scale)
        return optimize_param_list

    def set_param(self, param_list):
        with torch.no_grad():
            self.length_scale.copy_(param_list[0])
            self.scale.copy_(param_list[1])

    def get_params_need_check(self):
        _temp_list = []
        self.get_param(_temp_list)
        return _temp_list

    def clamp_to_positive(self):
        if 'GP_DEBUG' in os.environ and os.environ['GP_DEBUG'] == 'True':
            # if self.length_scale < 0:
                print('DEBUG WARNING length_scale:{} clamp to 0'.format(self.length_scale.data))
                print('DEBUG WARNING scale:{} clamp to 0'.format(self.scale.data))
                print('\n')

        with torch.no_grad():
            self.length_scale.copy_(self.length_scale.clamp_(min=0))
            self.scale.copy_(self.scale.clamp_(min=0))


if __name__ == '__main__':
    print('test1')
    ke = SE_kernel(True)

    print('test2')
    ke = SE_kernel(False, 2.)

    print('test3')
    ke = SE_kernel(False, 2., 3.)