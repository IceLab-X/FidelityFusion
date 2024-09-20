import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import torch
import numpy as np

from Data_simulation.Cost_Function.cost_pow_10 import cost_discrete as cost_pow_10
from Data_simulation.Cost_Function.cost_linear import cost_discrete as cost_linear
from Data_simulation.Cost_Function.cost_log import cost_discrete as cost_log
cost_list = {'pow_10': cost_pow_10,'linear': cost_linear, 'log': cost_log}

_alpha6_low = torch.tensor([0.5, 0.5, 2.0, 4.0]).view(-1, 1)
_alpha6_high = torch.tensor([1.0, 1.2, 3.0, 3.2]).view(-1, 1)
_A6 = torch.tensor([
    [10.00,  3.0, 17.00,  3.5,  1.7,  8],
    [ 0.05, 10.0, 17.00,  0.1,  8.0, 14],
    [ 3.00,  3.5,  1.70, 10.0, 17.0,  8],
    [17.00,  8.0,  0.05, 10.0,  0.1, 14],
]).t().unsqueeze(0)

_P6 = torch.tensor([
    [.1312, .1696, .5569, .0124, .8283, .5886],
    [.2329, .4135, .8307, .3736, .1004, .9991],
    [.2348, .1451, .3522, .2883, .3047, .6650],
    [.4047, .8828, .8732, .5743, .1091, .0381],
]).t().unsqueeze(0)

_four_nine_exp = torch.exp(torch.tensor(-4 / 9))

class hartmann():
    def __init__(self, cost_type, total_fidelity_num):
        self.total_fidelity_num = total_fidelity_num
        self.x_dim = 6
        self.search_range = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
        self.cost = cost_list[cost_type](self.search_range[-1])
    
    def w_h(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        w = torch.log10(9 * t + 1)
        return w
    
    def w_l(self, t):
        w = 1 - self.w_h(t)
        return w
    
    def hartmann6_hf(self, xx):
        xx = xx.unsqueeze(-1)

        tmp1 = (xx - _P6) ** 2 * _A6
        tmp2 = torch.exp(-torch.sum(tmp1, dim=1)).double()
        tmp3 = tmp2.matmul(_alpha6_high.double()) + 2.58

        return -(1/1.94) * tmp3.view(-1)

    def hartmann6_lf(self, xx):
        xx = xx.unsqueeze(-1)

        tmp1 = (xx - _P6) ** 2 * _A6
        tmp2 = self._f_exp(-torch.sum(tmp1, dim=1)).double()
        tmp3 = tmp2.matmul(_alpha6_low.double()) + 2.58

        return -(1/1.94) * tmp3.view(-1)
    
    def _f_exp(self, xx):
        return (_four_nine_exp + (_four_nine_exp * (xx + 4) / 9)) ** 9
    
    def get_cmf_data(self, input_x, input_s):
        
        xtr = input_x
        y_l = self.hartmann6_lf(xtr).reshape(-1,1)
        y_h = self.hartmann6_hf(xtr).reshape(-1,1)
        ytr_fid = self.w_l(input_s) * y_l + self.w_h(input_s) * y_h
        ytr_fid = ytr_fid + 0.001 * torch.randn_like(ytr_fid) ## add noise 正则化
        
        return ytr_fid
    
    def get_data(self, input_x, input_s):
        
        xtr = input_x
        y_l = self.hartmann6_lf(xtr).reshape(-1,1)
        y_h = self.hartmann6_hf(xtr).reshape(-1,1)
        ytr_fid = self.w_l(input_s) * y_l + self.w_h(input_s) * y_h
        ytr_fid = ytr_fid + 0.001 * torch.randn_like(ytr_fid) ## add noise 正则化
        
        return ytr_fid
    
    def Initiate_data(self, index, seed):
        torch.manual_seed(seed)
        
        low = []
        for i in range(self.x_dim):
            tt = torch.rand(index[0], 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            low.append(tt)
        xtr_low = torch.cat(low, dim=1)
        
        high = []
        for i in range(self.x_dim):
            tt = torch.rand(index[1], 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            high.append(tt)
        xtr_high = torch.cat(high, dim=1)
        
        xtr = [xtr_low, xtr_high]

        ytr_low = self.get_cmf_data(xtr_low, 0)
        ytr_high = self.get_cmf_data(xtr_high, 1)
        ytr = [ytr_low, ytr_high]

        return xtr, ytr
    
    def find_max_value_in_range(self):
        
        torch.manual_seed(1)
        
        # Generate random points within the search range
        num_points = 1000
        
        tem = []
        
        for i in range(self.x_dim):
            torch.manual_seed(i +55)
            tt = torch.rand(num_points, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            tem.append(tt)

        x = torch.cat(tem, dim = 1)
        fidelity_indicator = torch.ones(num_points, 1)
        
        y = self.get_cmf_data(x, fidelity_indicator)
        
        # Find the maximum value and its index
        max_value, max_index = torch.max(y, dim=0)

        return max_value.item(), x.reshape(-1, self.x_dim)
    
if __name__ == "__main__":
    data = hartmann('pow_10', 2)
    max,_ = data.find_max_value_in_range()
    ## max = -1.33
    xtr, ytr = data.Initiate_data({0: 10, 1: 4}, 1)
    pass
    
    