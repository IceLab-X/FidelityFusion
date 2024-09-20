import numpy as np
class cost_discrete():
    def __init__(self, fidelity_range):
        self.s_min = fidelity_range[0]
        self.s_max = fidelity_range[-1]

    def compute_cost(self, z):
        if isinstance(z, np.ndarray):
            c = np.power(z,2) + 0.1
        else:
            c = pow(z,2) + 0.1
        return c

    def compute_model_cost(self, dataset):
        C = 0
        for i in range(len(dataset)):
            C += self.compute_cost(i) * dataset[i].shape[0]
        return C
    
    def compute_gp_cost(self, dataset):
        C = 0
        for i in range(len(dataset[0])):
            C += self.compute_cost(dataset[0][i][-1]-1)
        return C

    def compute_index(self, index):
        C = 0
        for key in index.keys():
            C += self.compute_cost(int(key)) * int(index[key])
        return C

    def compute_model_cost_fabolas(self, X, y):
        C= 0
        for i in X:
            C += self.compute_cost(i[-1] + 1)
        return C