import numpy as np
import torch
import torch.nn as nn
import math

from Acquisition_Function.Discrete.ES import EntropySearch

class entropy_search():
    def __init__(self, fidelity_num, x_dimension, model_objective, model_cost, seed):
        np.random.seed(seed[0])
        self.x_dimension = x_dimension
        self.x_range = np.random.rand(100, x_dimension)
        np.random.seed(seed[0] * 2)
        self.z_range = np.array(sorted(np.random.rand(10)))
        self.fidelity_num = fidelity_num
        self.model_objective = model_objective
        self.model_cost = model_cost
        self.log_length_scale = nn.Parameter(torch.zeros(self.x_range.shape[1]))  # ARD length scale
        self.log_scale = nn.Parameter(torch.zeros(1))  # kernel scale
        self.beta = 0.2 * int(self.x_dimension) * math.log(int(seed[1]+1))
        self.d = self.x_range.shape[-1]
        self.p = 1
        self.k_0 = 1
        self.es = EntropySearch(model_objective, self.x_range)

    def kernel(self, X1, X2):
    # the common RBF kernel
        X1 = X1 / self.log_length_scale.exp()
        X2 = X2 / self.log_length_scale.exp()
        X1_norm2 = torch.sum(X1 * X1, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(0))  #this is the effective Euclidean distance matrix between X1 and X2.
        K = self.log_scale.exp() * torch.exp(-0.5 * K)
        return K

    def information_gap(self, input):
        if input == None:
            input = self.z_range
        else:
            input = np.ones(1)*input

        phi = self.kernel(torch.from_numpy(input[:, None]), torch.ones(1)[:, None].double())
        phi = phi.detach().numpy()
        ksin = np.sqrt(1-np.power(phi, 2))
        return ksin

    def gamma_z(self, ksin_z):
        q = 1 / (self.p + self.d + 2)
        gamma_z = np.sqrt(self.k_0) * ksin_z.flatten() * np.power(self.model_cost.compute_cost(self.z_range)/self.model_cost.compute_cost(1), q)
        return gamma_z

    def compute_next(self):

        es_range = self.es.evaluate(self.x_range)
        idx = np.argmax(es_range)
        new_x = self.x_range[idx]

        tau_z_mean = []
        tau_z_std = []
        for z in self.z_range:
            if z <= 0.5:
                re_z = 1
            else:
                re_z = 2
            m, v = self.model_objective.predict(new_x.reshape(1, self.x_dimension), re_z)
            tau_z_mean.append(m.detach().numpy())
            tau_z_std.append(np.sqrt(v.detach().numpy()))
        ksin_z = self.information_gap(None)
        gamma_z = self.gamma_z(ksin_z)

        possible_z = []
        for i in range(self.z_range.shape[0]):

            if tau_z_std[i] > gamma_z[i] and ksin_z[i] > np.sqrt(self.beta) * self.information_gap(np.sqrt(self.p)):
                possible_z.append(self.z_range[i])

        if len(possible_z) == 0:
            new_s = 0.1
        else:
            new_s = min(possible_z)

        if new_s >= 0.02:
            new_s = 2
        else:
            new_s = 1

        if len(new_x.shape) == 1:
            d = new_x.shape[0]
            new_x = new_x.reshape(1, d)

        return new_x, new_s

    def get_value(self):
        es_range = self.es.evaluate(self.x_range)

        return es_range, self.x_range
