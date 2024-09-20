import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import torch
import math

from Data_simulation.Cost_Function.cost_pow_10 import cost_discrete as cost_pow_10
from Data_simulation.Cost_Function.cost_linear import cost_discrete as cost_linear
from Data_simulation.Cost_Function.cost_log import cost_discrete as cost_log
cost_list = {'pow_10': cost_pow_10,'linear': cost_linear, 'log': cost_log}

class Branin():
    def __init__(self,cost_type, total_fidelity_num = None):
        self.x_dim = 2
        self.search_range = [[0, 1.5], [0, 1.5], [0, 1]]
        self.cost = cost_list[cost_type](self.search_range[-1])
        self.b = 5.1 / (4 * math.pow(math.pi, 2))
        self.c = 5 / math.pi
        self.r = 6
        self.t = 1 / (8 * math.pi)
        self.fi_num = total_fidelity_num

    def get_data(self, input_x, input_s):
        if input_x.shape[0] == 1:
            # 单独生成new_x, new_s
            x = torch.cat((input_x.reshape(1, 2), input_s * torch.ones(1)[:, None]), dim=1)
            Y = (torch.pow(x[:, 1] - (self.b - 0.1 * (1 - x[:, 2])) * torch.pow(x[:, 0], 2) + self.c * x[:, 0] - self.r, 2)
                  + 10 * (1 - self.t) * torch.cos(x[:, 0]) + 10)

        else:
            x = torch.cat((input_x, input_s.reshape(input_x.shape[0], 1)), dim=1)
            Y = (torch.pow(x[:, 1] - (self.b - 0.1 * (1 - x[:, 2])) * torch.pow(x[:, 0], 2) + self.c * x[:, 0] - self.r, 2)
                  + 10 * (1 - self.t) * torch.cos(x[:, 0]) + 10)

        return Y.reshape(-1, 1)
    
    def get_cmf_data(self, input_x, input_s):
        if input_x.shape[0] == 1:
            # 单独生成new_x, new_s
            x = torch.cat((input_x.reshape(1, 2), input_s * torch.ones(1)[:, None]), dim=1)
            Y = (torch.pow(x[:, 1] - (self.b - 0.1 * (1 - x[:, 2])) * torch.pow(x[:, 0], 2) + self.c * x[:, 0] - self.r, 2)
                  + 10 * (1 - self.t) * torch.cos(x[:, 0]) + 10)

        else:
            x = torch.cat((input_x, input_s.reshape(input_x.shape[0], 1)), dim=1)
            Y = (torch.pow(x[:, 1] - (self.b - 0.1 * (1 - x[:, 2])) * torch.pow(x[:, 0], 2) + self.c * x[:, 0] - self.r, 2)
                  + 10 * (1 - self.t) * torch.cos(x[:, 0]) + 10)

        return Y.reshape(-1, 1)
    

    def Initiate_data(self, index, seed):
        
        torch.manual_seed(seed)
        xtr_low = torch.rand(index[0], 2).double() * (self.search_range[0][1] - self.search_range[0][0]) + self.search_range[0][0]
        fidelity_indicator1 = torch.ones(index[0], 1) * 0
        ytr_low = self.get_data(xtr_low, fidelity_indicator1).reshape(index[0], 1)
        
        # xtr_mid = torch.rand(index[1], 2).double() * (self.search_range[1][1] - self.search_range[1][0]) + self.search_range[1][0]
        # fidelity_indicator2 = torch.ones(index[1], 1) * 0.5
        # ytr_middle = self.get_data(xtr_mid, fidelity_indicator2).reshape(index[1], 1)
        
        xtr_high = torch.rand(index[1], 2).double() * (self.search_range[1][1] - self.search_range[1][0]) + self.search_range[1][0]
        fidelity_indicator3 = torch.ones(index[1], 1) * 1
        ytr_high = self.get_data(xtr_high, fidelity_indicator3).reshape(index[1], 1)
        
        # xtr = torch.cat((xtr_low, xtr_mid, xtr_high), dim=0)
        xtr = [xtr_low, xtr_high]
        
        # fidelity_indicator = torch.cat((fidelity_indicator1, fidelity_indicator2, fidelity_indicator3), dim=0)
        # total_num = index[0] + index[1] + index[2]
        # ytr = self.get_data(xtr, fidelity_indicator).reshape(total_num, 1)
        ytr = [ytr_low, ytr_high]

        return xtr, ytr
    
    def get_discrete_data(self, index, seed):
            
        torch.manual_seed(seed)
        xtr = []
        ytr = []

        for i in range(self.fi_num):
            xtr_fi = torch.rand(index[i], 2).double() * (self.search_range[0][1] - self.search_range[0][0]) + self.search_range[0][0]
            fidelity_indicator = torch.ones(index[i], 1) * i / self.fi_num
            ytr_fi = self.get_data(xtr_fi, fidelity_indicator).reshape(index[i], 1)
            xtr.append(xtr_fi)
            ytr.append(ytr_fi)

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
        
        y = self.get_data(x, fidelity_indicator)
        
        # Find the maximum value and its index
        max_value, max_index = torch.max(y, dim=0)

        return max_value.item(), x.reshape(-1, self.x_dim)


if __name__ == "__main__":
    data = Branin('pow_10')
    max_value, _ = data.find_max_value_in_range()
    xtr, ytr, fidelity_indicator = data.Initiate_data(8, 1)
    #cost is a tensor
    Cost = data.cost.compute_model_cost(dataset = ytr, s_index = fidelity_indicator)
