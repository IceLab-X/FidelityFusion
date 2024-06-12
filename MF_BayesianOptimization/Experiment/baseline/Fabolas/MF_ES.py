import numpy as np
import torch
import torch.nn as nn
import sys

from model_MFBO.Acq_function_continuous.ES import EntropySearch


class entropy_search_continuous():
    def __init__(self, x_dimension, search_range, model_objective, model_cost, seed):
        np.random.seed(seed)
        self.seed = seed
        self.x_dimension = x_dimension
        tem = []
        for i in range(x_dimension):
            tt = np.random.rand(100, 1)*(search_range[i][1] - search_range[i][0])+search_range[i][0]
            tem.append(tt)
        self.x_range = np.concatenate(tem, axis = 1)
        np.random.seed(seed * 2)
        self.search_range = search_range
        self.z_range = np.array(sorted(np.random.rand(100) * (search_range[-1][1] - search_range[-1][0]))) + search_range[-1][0]
        self.model_objective = model_objective
        self.model_cost = model_cost

        self.log_length_scale = nn.Parameter(torch.zeros(self.x_range.shape[1]))  # ARD length scale
        self.log_scale = nn.Parameter(torch.zeros(1))  # kernel scale
        self.beta = 2.0
        self.d = self.x_range.shape[-1]
        self.p = 1
        self.k_0 = 1
        self.es = EntropySearch(model_objective, self.x_range, self.search_range, self.seed)

    def kernel(self, X1, X2):
        # the common RBF kernel
        X1 = X1 / self.log_length_scale.exp()
        X2 = X2 / self.log_length_scale.exp()
        X1_norm2 = torch.sum(X1 * X1, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(
            0))  # this is the effective Euclidean distance matrix between X1 and X2.
        K = self.log_scale.exp() * torch.exp(-0.5 * K)
        return K

    def information_gap(self, input):
        if input == None:
            input = self.z_range
        else:
            input = np.ones(1) * input

        phi = self.kernel(torch.from_numpy(input[:, None]), torch.ones(1)[:, None].double())
        phi = phi.detach().numpy()
        ksin = np.sqrt(1 - np.power(phi, 2))
        return ksin

    def gamma_z(self, ksin_z):
        q = 1 / (self.p + self.d + 2)
        gamma_z = np.sqrt(self.k_0) * ksin_z.flatten() * np.power(
            self.model_cost.compute_cost(self.z_range) / self.model_cost.compute_cost(1), q)
        return gamma_z

    def compute_next(self):

        es_range = self.es.evaluate(self.x_range, self.z_range)
        idx = np.argmax(es_range)
        new_x = self.x_range[idx]
        new_x = new_x.reshape(1, self.x_dimension)
        new_s = self.z_range[idx].reshape(-1, 1)
        # m, v = self.model_objective.predict(new_x, np.ones(1).reshape(1, 1)*self.search_range[-1][-1])
        # # tau_z_mean = m.detach().numpy()
        # tau_z_std = np.sqrt(v.detach().numpy())
        # ksin_z = self.information_gap(None)
        # gamma_z = self.gamma_z(ksin_z)
        #
        # possible_z = []
        # for i in range(self.z_range.shape[0]):
        #
        #     condition_1 = tau_z_std > gamma_z[i]
        #     condition_2 = ksin_z[i] > self.information_gap(np.sqrt(self.p)) / np.sqrt(self.beta)
        #
        #     if condition_1 and condition_2:
        #         possible_z.append(self.z_range[i])
        #
        # if len(possible_z) == 0:
        #     new_s = 0.1
        # else:
        #     new_s = min(possible_z)

        return new_x, new_s
