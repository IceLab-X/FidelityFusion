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
class Kernel_res(torch.nn.Module):
    # Squared Exponential Kernel
    def __init__(self, noise_exp_format, length_scale=1., scale=1., length_scale_z=1., const_item=torch.tensor(3.).sqrt()) -> None:
        super().__init__()
        self.noise_exp_format = noise_exp_format

        length_scale = torch.tensor(length_scale)
        scale = torch.tensor(scale)
        length_scale_z = torch.tensor(length_scale_z)
        self.const_item = const_item

        if noise_exp_format is True:
            self.length_scale = torch.nn.Parameter(torch.log(length_scale))
            self.scale = torch.nn.Parameter(torch.log(scale))
            self.length_scale_z = torch.nn.Parameter(torch.log(length_scale_z))
        else:
            self.length_scale = torch.nn.Parameter(length_scale)
            self.scale = torch.nn.Parameter(scale)
            self.length_scale_z = torch.nn.Parameter(length_scale_z)

        # add exp format?
        self.b = torch.nn.Parameter(torch.tensor(1.))

        # TODO should we add noise here?
        self.seed = 1024

    def warp(self, l1, h1, l2, h2):
        lf1, hf1, lf2, hf2 = l1, h1, l2, h2
        return lf1, hf1, lf2, hf2

    def forward(self, X1, X2, l1, h1, l2, h2):
        if self.noise_exp_format is True:
            length_scale = torch.exp(self.length_scale).view(1, -1)
            scale = torch.exp(self.scale).view(1, -1)
            length_scale_z = torch.exp(self.length_scale_z).view(1, -1)
        else:
            length_scale = self.length_scale.view(1, -1)
            scale = self.scale.view(1, -1)
            length_scale_z = self.length_scale_z.view(1, -1)

        lf1, hf1, lf2, hf2 = self.warp(l1, h1, l2, h2)

        N = 100
        torch.manual_seed(self.seed)
        z1 = torch.rand(N) * (hf1 - lf1) + lf1 # 这块需要用来调整z选点的范围
        z2 = torch.rand(N) * (hf2 - lf2) + lf2

        X1 = X1 / length_scale
        X2 = X2 / length_scale
        X1_norm2 = torch.sum(X1 * X1, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(0))  
        #this is the effective Euclidean distance matrix between X1 and X2.
        K = scale * torch.exp(-0.5 * K)
        
        # z part use MCMC to calculate the integral
        dist_z = (z1 / length_scale_z - z2 / length_scale_z) ** 2
        z_part1 = -self.b * (z1 - hf1)
        z_part2 = -self.b * (z2 - hf2)
        z_part  = (z_part1 + z_part2 - 0.5 * dist_z).exp()
        z_part_mc = z_part.mean() * (hf1 - lf1) * (hf2 - lf2)
        # z_part_mc = z_part.mean()
        
        K_ard = z_part_mc * K
        return K_ard

    def get_param(self, optimize_param_list):
        optimize_param_list.append(self.length_scale)
        optimize_param_list.append(self.scale)
        optimize_param_list.append(self.length_scale_z)
        optimize_param_list.append(self.b)
        return optimize_param_list

    def set_param(self, param_list):
        with torch.no_grad():
            self.length_scale.copy_(param_list[0])
            self.scale.copy_(param_list[1])
            self.length_scale_z.copy_(param_list[2])
            self.b.copy_(param_list[3])

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
    ke = Kernel_res(True)

    print('test2')
    ke = Kernel_res(False, 2.)

    print('test3')
    ke = Kernel_res(False, 2., 3.)