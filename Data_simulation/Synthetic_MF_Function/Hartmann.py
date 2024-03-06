import torch
import numpy as np

from botorch.test_functions.multi_fidelity import AugmentedHartmann
from Simulation.Cost_Function.cost_pow_10 import cost

class Hartmann():
    def __init__(self):
        self.x_dim = 6
        self.search_range = [[0, 1] for i in range(7)]

        tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
        self.problem = AugmentedHartmann(negate=True).to(**tkwargs)
        self.cost = cost(self.search_range[-1])

    def get_data(self, input_x, input_s):
        if input_x.shape[0] == 1:
            # 单独生成new_x, new_s
            x = torch.from_numpy(np.concatenate((input_x.reshape(1, self.x_dim), input_s * np.ones(1)[:, None]), axis=1))
            Y = self.problem(x)
            if isinstance(Y, torch.Tensor):
                Y = Y.numpy()
            if len(Y.shape) == 1:
                d = Y.shape[0]
                Y = Y.reshape(1, d)
        else:
            if isinstance(input_x, np.ndarray):
                # 初始化数据
                input_x = torch.from_numpy(np.concatenate((input_x, input_s.reshape(input_x.shape[0], 1)), axis=1))
            else:
                input_x = torch.cat([input_x, input_s * torch.ones(1).reshape(1, 1)], dim=1)

            x = input_x
            Y = self.problem(x)

        return Y

    def Initiate_data(self, num, seed):
        tem = []
        for i in range(self.x_dim):
            np.random.seed(seed + 217 + i)
            tt = np.random.rand(num, 1) * (self.search_range[i][1] - self.search_range[i][0]) + self.search_range[i][0]
            tem.append(tt)

        xtr = np.concatenate(tem, axis=1)
        fidelity_indicator = np.random.rand(num, 1) * (self.search_range[-1][1] - self.search_range[-1][0]) + self.search_range[-1][0]

        ytr = self.get_data(xtr, fidelity_indicator).reshape(num, 1)

        return xtr, ytr, fidelity_indicator

        return y

if __name__ == "__main__":
    data = Hartmann()
    x, y, f = data.Initiate_data(8, 1)
    print(y)