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
class Matern_kernel(torch.nn.Module):
    # Squared Exponential Kernel
    def __init__(self, noise_exp_format, length_scale=1., scale=1., const_item=torch.tensor(3.).sqrt()) -> None:
        super().__init__()
        self.noise_exp_format = noise_exp_format

        length_scale = torch.tensor(length_scale)
        scale = torch.tensor(scale)
        self.const_item = const_item

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

        X = X / length_scale.expand(X.size(0), length_scale.size(1))
        X2 = X2 / length_scale.expand(X2.size(0), length_scale.size(1))

        #! error on torch.dist, shape not match
        # distance = self.const_item* torch.dist(X, X2, p=2)
        distance = self.const_item*(X @ X2.t())

        k_matern3 = scale * (1. + distance) * torch.exp(-distance)
        return k_matern3

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
    ke = Matern_kernel(True)

    print('test2')
    ke = Matern_kernel(False, 2.)

    print('test3')
    ke = Matern_kernel(False, 2., 3.)