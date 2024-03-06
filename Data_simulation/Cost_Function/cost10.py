import numpy as np

class cost():
    def __init__(self, fidelity_range):
        self.s_min = fidelity_range[0]
        self.s_max = fidelity_range[-1]

    def compute_cost(self, z):
        if isinstance(z, np.ndarray):
            c = np.floor((z+1) * 5)
        else:
            c = int(z * 5)
        return c

    def compute_cost(self, z):
        if isinstance(z, np.ndarray):
            c = np.floor((z+1) * 10)
        else:
            c = int(z * 10)
        return c

    def compute_model_cost(self, dataset):
        C = 0
        for i in range(len(dataset)):
            C += self.compute_cost(i+1) * dataset[i].shape[0]
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


class cost_continuous():
    def __init__(self, fidelity_range):
        self.s_min = fidelity_range[0]
        self.s_max = fidelity_range[-1]

    def transform(self, s):
        s_cost = (np.log2(s) - np.log2(self.s_min)) / (np.log2(self.s_max) - np.log2(self.s_min))
        return s_cost

    def return_fidelity(self):
        pass
        return 0

    def compute_cost(self, z):
        if isinstance(z, np.ndarray):
            c = (z+1) * 2
        else:
            c = z * 2
        return c

    def compute_model_cost(self, dataset, s_index):
        C = 0

        for j in range(len(dataset)):
            C += self.compute_cost(s_index[j])
        return C[0]


    def compute_model_cost_fabolas(self, X, y):
        C= 0
        for i in X:
            C += self.compute_cost(i[-1] + 1)
        return C


if __name__ == "__main__":
    cost = cost([0,1])
    print(cost.compute_index({1: 32, 2: 8}))