import numpy as np
import torch
class cost():
    def __init__(self, fidelity_range):
        self.s_min = fidelity_range[0]
        self.s_max = fidelity_range[-1]

    def compute_cost(self, z):
        if isinstance(z, np.ndarray):
            c = np.power(10, z)
        else:
            c = pow(10, z)
        return c


    def compute_model_cost(self, dataset, s_index):
        C = 0
        for j in range(len(dataset)):
            C += self.compute_cost(s_index[j])

        return int(C)
    
    def compute_gp_cost(self, dataset):
        C = 0
        for i in range(len(dataset[0])):
            C += self.compute_cost(dataset[0][i][1])
        return C


    def compute_model_cost_fabolas(self, X, y):
        C= 0
        for i in X:
            C += self.compute_cost(i[-1])
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
            c = torch.floor(torch.pow(10, torch.tensor(z)))
            # c = int(pow(10, z))
        return c
    
    def compute_gp_cost(self, dataset):
        C = 0
        for i in range(len(dataset[0])):
            C += self.compute_cost(dataset[0][i][-1]-1)
        return C

    def compute_model_cost(self, dataset):
        C = 0
        for i in range(len(dataset)):
            C += self.compute_cost(i) * dataset[i].shape[0]
        return C
    
    def compute_model_ConDis_cost(self, dataset, fi):
        C = 0
        for i in range(len(dataset)):
            C += self.compute_cost(i/fi) * dataset[i].shape[0]
        return C
    
    def compute_model_ConDisGP_cost(self, dataset, fi):
        C = 0
        for i in range(len(dataset[0])):
            C += self.compute_cost((dataset[0][i][-1]-1)/fi)
        return C

    def compute_index(self, index):
        C = 0
        for key in index.keys():
            C += self.compute_cost(int(key)) * int(index[key])
        return C

    def compute_model_cost_fabolas(self, X, s):
        C= 0
        for i in range(X.shape[0]):
            # C += self.compute_cost(s[i]-1)
            C += torch.pow(10, torch.tensor(s[i]-1))
        return C
    
    def compute_model_cost_smac(self, ytr):
        C = 0
        for i in range(len(ytr)):
            C += self.compute_cost(i) * ytr[i].shape[0]
        return C

if __name__ == "__main__":
    cost = cost([0,1])
    print(cost.compute_cost(np.array([1,2])))