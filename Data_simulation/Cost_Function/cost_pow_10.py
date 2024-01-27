import numpy as np

class cost():
    def __init__(self, fidelity_range):
        self.s_min = fidelity_range[0]
        self.s_max = fidelity_range[-1]

    def compute_cost(self, z):
        if isinstance(z, np.ndarray):
            c = np.power(10, z + 1)
        else:
            c = pow(10, z)
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

    def compute_model_cost_smac(self, dataset):
        C = 0
        for i in range(len(dataset)):
            C += self.compute_cost(i+1) * dataset[i].shape[0]
        return C


class cost_discrete():
    def __init__(self, fidelity_range):
        self.s_min = fidelity_range[0]
        self.s_max = fidelity_range[-1]

    def compute_cost(self, z):
        if isinstance(z, np.ndarray):
            c = np.floor(np.power(10, z))
        else:
            c = int(pow(10, z))
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

if __name__ == "__main__":
    cost = cost([0,1])
    print(cost.compute_cost(np.array([1,2])))