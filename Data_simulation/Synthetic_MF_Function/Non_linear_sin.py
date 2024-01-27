import torch
import math
import numpy as np

from Data_simulation.Cost_Function.cost_pow_10 import cost_discrete as cost

class non_linear_sin():
    def __init__(self, total_fidelity_num):
        self.total_fidelity_num = total_fidelity_num
        self.x_dim = 1
        self.search_range = [[0, 1], [0, 1]]
        self.cost = cost(self.search_range[-1])

    def w_h(self, t):
        w = t ** 2 + 0.1 * torch.sin(10 * torch.pi * t)
        return w

    def w_l(self, t):
        w = 1 - self.w_h(t)
        return w

    def get_data(self, input_x, input_s):

        # fcn, space = multi_fidelity_non_linear_sin()

        xtr = input_x
        Ytr_l = np.sin(xtr * 8 * math.pi)
        Ytr_h = (xtr - np.sqrt(2)) * np.power(Ytr_l, 2)

        Ytr = [torch.from_numpy(Ytr_l)]
        fidelity_list = torch.linspace(0, 1, self.total_fidelity_num).view(-1, 1)
        fidelity_list = fidelity_list[1:-1]
        if len(fidelity_list) != 0:
            for s in fidelity_list:
                ytr_fid = self.w_l(s).detach().numpy() * Ytr_l + self.w_h(s).detach().numpy() * Ytr_h
                Ytr.append(ytr_fid)
            Ytr.append(torch.from_numpy(Ytr_h))
        else:
            Ytr.append(torch.from_numpy(Ytr_h))

        new_y = Ytr[input_s - 1]

        if len(new_y.shape) == 1:
            d = new_y.shape[0]
            new_y = new_y.reshape(1, d)

        return new_y


    def Initiate_data(self, index, seed):
        np.random.seed(seed)
        xtr_low = np.random.rand(index[1])[:, None]
        xtr_high = np.concatenate((xtr_low[:(index[2] - 2), :], np.random.rand(2)[:, None]), axis=0)
        xtr = [torch.from_numpy(xtr_low), torch.from_numpy(xtr_high)]

        ytr_low = self.get_data(xtr_low, 1)
        ytr_high = self.get_data(xtr_high, 2)
        ytr = [ytr_low, ytr_high]

        return xtr, ytr


if __name__ == "__main__":
    data = non_linear_sin(2)
    xtr, ytr = data.Initiate_data({1: 10, 2: 4}, 1)
    data.cost.compute_model_cost(ytr)
    print(ytr)
