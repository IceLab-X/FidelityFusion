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

class HeatedBlock:
    """ HeatedBlock """
    def __init__(self, cost_type, total_fidelity_num = None):
        
        self.x_dim = 3
        self.flevels = 2
        self.maximum = 2.0
        # self.bounds = ((0.1,0.4), (0.1,0.4),(0,2*np.pi))
        self.search_range = [[0.1, 0.4], [0.1, 0.4], [0, 2*np.pi],[0, 1]]
        self.cost = cost_list[cost_type](self.search_range[-1])
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(r'H:\eda\nips2024\mfbo_v2\Data_simulation\Real_Application', nargout=0) # Add the path of the matlab code
        # self.lb = torch.tensor([bound[0] for bound in self.bounds])
        # self.ub = torch.tensor([bound[1] for bound in self.bounds])
    
    def get_data(self, input_x, input_s):
        if isinstance(input_x, torch.Tensor):
            input_x = input_x.numpy()
        if isinstance(input_s, torch.Tensor):
            input_s = input_s.numpy()
        
        y = self.eng.HeatedBlockQuerySilent(matlab.double(input_x.tolist()), matlab.double(input_s.tolist()))
        y_np = np.array(y)
        Y = torch.tensor(y_np)

        return Y

    def get_cmf_data(self, input_x, input_s):
        if isinstance(input_x, torch.Tensor):
            input_x = input_x.numpy()
        if isinstance(input_s, torch.Tensor):
            input_s = input_s.numpy()
        y = self.eng.HeatedBlockQuerySilent(matlab.double(input_x.tolist()), matlab.double(input_s.tolist()))
        y_np = np.array(y)
        Y = torch.tensor(y_np)

        return Y
        
#         X = X.numpy()
        
#         if X.ndim == 1:
#             # X = torch.unsqueeze(X, 0)
#             X = np.expand_dims(X, 0)
            
#         matlab_input = '['
#         for i in range(X.shape[0]):

#             s = np.array2string(X[i,:], separator=',',floatmode='maxprec',max_line_width=10000000)
#             # s = ', '.join(map(str, X[i,:].tolist()))
#             matlab_input += s
#             if i < X.shape[0] - 1:
#                 matlab_input += ';'

#         matlab_input += ']'
        
#         matlab_cmd = 'addpath(genpath(\'HeatedBlock\'));'
#         matlab_cmd += 'HeatedBlockQuery(' + matlab_input  + ',' + str(m) + ');'

#         # matlab_cmd += 'quit force'
        
#         process = subprocess.Popen(["matlab", "-nodesktop", "-r", matlab_cmd],
#                              stdout=subprocess.PIPE, 
#                              stderr=subprocess.PIPE)
#         stdout, stderr = process.communicate()
        
# #         print(stdout)
# #         print(stderr)
        
#         message_out = stdout.decode("utf-8")
#         message_err = stderr.decode("utf-8")
        
#         print(message_out)
#         print(message_err)
        
#         [start, end] = [i for i,ltr in enumerate(message_out) if ltr=='$']
        
#         matlab_output = message_out[start+1:end]
        
#         ym = torch.tensor([float(yi) for yi in matlab_output.split(',')])

        # return ym
    
    def Initiate_data(self, index, seed):
        
        torch.manual_seed(seed)
        low = []
        for i in range(self.x_dim):
            tt = torch.rand(index[0], 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            low.append(tt)
        xtr_low = torch.cat(low, dim=1)

        # fidelity_indicator1 = torch.ones(index[0], 1) * 0
        # ytr_low = self.get_data(xtr_low, fidelity_indicator1).reshape(index[0], 1)
        ytr_low = self.get_data(xtr_low, torch.tensor(0.)).reshape(index[0], 1)
        
        torch.manual_seed(seed+3)
        high = []
        for i in range(self.x_dim):
            tt = torch.rand(index[1], 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            high.append(tt)
        xtr_high = torch.cat(high, dim=1)

        # fidelity_indicator3 = torch.ones(index[1], 1) * 1

        # ytr_high = self.get_data(xtr_high, fidelity_indicator3).reshape(index[1], 1)
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
        
        y = self.get_data(x, torch.tensor(1.))
        
        # Find the maximum value and its index
        max_value, max_index = torch.max(y, dim=0)

        # max_value = torch.tensor(1.60)

        return max_value.item(), x.reshape(-1, self.x_dim)
    

if __name__ == '__main__':
        
    instance = HeatedBlock(cost_type='pow_10')
    # xtr, ytr = instance.Initiate_data(index=[10, 4], seed=0)
    max_value, _ = instance.find_max_value_in_range()
    # print(xtr)
        
