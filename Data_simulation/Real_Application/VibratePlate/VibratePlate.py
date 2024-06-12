import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import torch
import numpy as np

from Data_simulation.Cost_Function.cost_pow_10 import cost_discrete as cost_pow_10
from Data_simulation.Cost_Function.cost_linear import cost_discrete as cost_linear
from Data_simulation.Cost_Function.cost_log import cost_discrete as cost_log

import matlab
import matlab.engine
cost_list = {'pow_10': cost_pow_10,'linear': cost_linear, 'log': cost_log}

class VibPlate:
    """ Airfoil """
    def __init__(self, cost_type, total_fidelity_num = None):
        
        self.x_dim = 3
        self.flevels = 2
        self.maximum = 250
        self.search_range = [[100e9, 500e9], [0.2, 0.6], [6000, 10000],[0, 1]]
        self.cost = cost_list[cost_type](self.search_range[-1])
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(r'H:\eda\nips2024\mfbo_v2\Data_simulation\Real_Application', nargout=0) # Add the path of the matlab code
    
    def get_data(self, input_x, input_s):
        if isinstance(input_x, torch.Tensor):
            input_x = input_x.numpy()
        if isinstance(input_s, torch.Tensor):
            input_s = input_s.numpy()
        
        y = self.eng.VibratePlateQuerySilent(matlab.double(input_x.tolist()), matlab.double(input_s.tolist()))
        y_np = np.array(y)
        Y = torch.tensor(y_np)

        return Y

    def get_cmf_data(self, input_x, input_s):
        if isinstance(input_x, torch.Tensor):
            input_x = input_x.numpy()
        if isinstance(input_s, torch.Tensor):
            input_s = input_s.numpy()
        y = self.eng.VibratePlateQuerySilent(matlab.double(input_x.tolist()), matlab.double(input_s.tolist()))
        y_np = np.array(y)
        Y = torch.tensor(y_np)

        return Y
    
    def Initiate_data(self, index, seed):
        
        torch.manual_seed(seed)
        low = []
        for i in range(self.x_dim):
            tt = torch.rand(index[0], 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            low.append(tt)
        xtr_low = torch.cat(low, dim=1)

        ytr_low = self.get_data(xtr_low, torch.tensor(0.)).reshape(index[0], 1)
        
        torch.manual_seed(seed+3)
        high = []
        for i in range(self.x_dim):
            tt = torch.rand(index[1], 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            high.append(tt)
        xtr_high = torch.cat(high, dim=1)

        ytr_high = self.get_data(xtr_high, torch.tensor(1.)).reshape(index[1], 1)
        
        xtr = [xtr_low, xtr_high]
        ytr = [ytr_low, ytr_high]

        return xtr, ytr
    
    def find_max_value_in_range(self):
        
        # Generate random points within the search range
        num_points = 100
        
        tem = []
        
        for i in range(self.x_dim):
            torch.manual_seed(i +55)
            tt = torch.rand(num_points, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            tem.append(tt)

        x = torch.cat(tem, dim = 1)
        # fidelity_indicator = torch.ones(num_points, 1)
        
        # y = self.get_data(x, 1.)
        
        # Find the maximum value and its index
        # max_value, max_index = torch.max(y, dim=0)

        max_value = torch.tensor(self.maximum)

        return max_value.item(), x.reshape(-1, self.x_dim)
        