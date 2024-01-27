import torch
import math
import numpy as np

from Simulation.Cost_Function.cost_pow_10 import cost


class Branin():
    def __init__(self):
        self.x_dim = 2
        self.search_range = [[-5, 10], [0, 15], [0, 1]]
        self.cost = cost(self.search_range[-1])
        self.b = 5.1 / (4 * pow(math.pi, 2))
        self.c = 5 / math.pi
        self.r = 6
        self.t = 1 / (8 * math.pi)

    def get_data(self, input_x, input_s):
        if input_x.shape[0] == 1:
            # 单独生成new_x, new_s
            x = np.concatenate((input_x.reshape(1, 2), input_s * np.ones(1)[:, None]), axis=1)
            Y = -(pow(x[:, 1] - (self.b - 0.1 * (1 - x[:, 2])) * pow(x[:, 0], 2) + self.c * x[:, 0] - self.r, 2)
                  + 10 * (1 - self.t) * np.cos(x[:, 0]) + 10)
            Y = Y

        else:
            x = np.concatenate((input_x, input_s.reshape(input_x.shape[0], 1)), axis=1)
            Y = -(pow(x[:, 1] - (self.b - 0.1 * (1 - x[:, 2])) * pow(x[:, 0], 2) + self.c * x[:, 0] - self.r, 2)
                  + 10 * (1 - self.t) * np.cos(x[:, 0]) + 10)

        return Y.reshape(-1, 1)

    def Initiate_data(self, num, seed):
        tem = []
        for i in range(self.x_dim):
            np.random.seed(seed + 217 + i)
            tt = np.random.rand(num, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            tem.append(tt)

        xtr = np.concatenate(tem, axis=1)
        fidelity_indicator = np.random.rand(num, 1) * (self.search_range[-1][1] - self.search_range[-1][0]) + \
                             self.search_range[-1][0]

        ytr = self.get_data(xtr, fidelity_indicator).reshape(num, 1)

        return xtr, ytr, fidelity_indicator


if __name__ == "__main__":
    data = Branin()
    data.Initiate_data(8, 1)
    data.cost.compute_model_cost()
