import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import torch
import math

from Data_simulation.Cost_Function.cost_pow_10 import cost_discrete as cost_pow_10
from Data_simulation.Cost_Function.cost_linear import cost_discrete as cost_linear
from Data_simulation.Cost_Function.cost_log import cost_discrete as cost_log
cost_list = {'pow_10': cost_pow_10,'linear': cost_linear, 'log': cost_log}
class non_linear_sin():
    def __init__(self, cost_type, total_fidelity_num):
        self.total_fidelity_num = total_fidelity_num
        self.x_dim = 1
        self.search_range = [[0, 1.5], [0, 1.5], [0, 1]]
        self.cost = cost_list[cost_type](self.search_range[-1])

    def w_h(self, t):
        # w = t ** 2 + 0.1 * torch.sin(10 * torch.pi * t)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float64)
        w = torch.log10(9 * t + 1)
        return w

    def w_l(self, t):
        w = 1 - self.w_h(t)
        return w

    def get_data(self, input_x, input_s):

        # fcn, space = multi_fidelity_non_linear_sin()

        xtr = input_x
        Ytr_l = torch.sin(xtr * 8 * math.pi)
        Ytr_h = (xtr - torch.sqrt(torch.tensor(2.0))) * torch.pow(Ytr_l, 2)

        Ytr = [Ytr_l]
        fidelity_list = torch.linspace(0, 1, self.total_fidelity_num).view(-1, 1)
        fidelity_list = fidelity_list[1:-1]
        if len(fidelity_list) != 0:
            for s in fidelity_list:
                ytr_fid = self.w_l(s) * Ytr_l + self.w_h(s) * Ytr_h
                Ytr.append(ytr_fid)
            Ytr.append(Ytr_h)
        else:
            Ytr.append(Ytr_h)

        new_y = Ytr[input_s]

        if len(new_y.shape) == 1:
            d = new_y.shape[0]
            new_y = new_y.reshape(1, d)

        return new_y
    
    def get_cmf_data(self, input_x, input_s):
        
        xtr = input_x
        Ytr_l = torch.sin(xtr * 8 * math.pi)
        Ytr_h = (xtr - torch.sqrt(torch.tensor(2.0))) * torch.pow(Ytr_l, 2)
        ytr_fid = self.w_l(input_s) * Ytr_l + self.w_h(input_s) * Ytr_h
        
        return ytr_fid
    
    def get_discrete_data(self, index, seed):
        
        torch.manual_seed(seed)
        xtr = []
        ytr = []

        for i in range(self.total_fidelity_num):
            xtr_fi = torch.rand(index[i], 1).double() * (self.search_range[0][1] - self.search_range[0][0]) + self.search_range[0][0]
            fidelity_indicator = torch.ones(index[i], 1) * i / self.total_fidelity_num
            ytr_fi = self.get_cmf_data(xtr_fi, fidelity_indicator).reshape(index[i], 1)
            xtr.append(xtr_fi)
            ytr.append(ytr_fi)

        return xtr, ytr


    def Initiate_data(self, index, seed):
        torch.manual_seed(seed)
        xtr_low = self.search_range[0][1]*torch.rand(index[0], 1).double()  # x is from 0 to 1.5
        xtr_high = torch.cat((xtr_low[:int(index[1] - index[1]/2),:], torch.rand(int(index[1]/2), 1)*self.search_range[0][1])).double()
        xtr = [xtr_low, xtr_high]

        ytr_low = self.get_data(xtr_low, 0)
        ytr_high = self.get_data(xtr_high, 1)
        ytr = [ytr_low, ytr_high]

        return xtr, ytr
    
    def find_max_value_in_range(self):
        
        # Generate random points within the search range
        torch.manual_seed(1)
        num_points = 1000
        
        x_samples = torch.rand(num_points, 1) * (self.search_range[0][1] - self.search_range[0][0]) + self.search_range[0][0]
        
        y_samples = self.get_data(x_samples, 1)
        
        # Find the maximum value and its index
        max_value, max_index = torch.max(y_samples[:, 0], dim=0)

        return max_value.item(),x_samples.reshape(-1,1)


if __name__ == "__main__":
    data = non_linear_sin('pow_10', 2)
    max,_ = data.find_max_value_in_range()
    pass
