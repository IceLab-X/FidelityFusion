#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import torch.nn as nn
import torch
import sys
import copy
# from Utils.gen_data_continuous import gen_data as gen_data
# from Simulation.Synthetic_MF_Function.gen_data_discrete import gen_data as gen_data


class discrete_fidelity_knowledgement_gradient(nn.Module):
    def __init__(self,  posterior_function, model_objective_new, train_function_new, data_model, data_manager, model_cost, seed, total_fidelity_num):
        super(discrete_fidelity_knowledgement_gradient, self).__init__()

        self.pre_func = posterior_function
        self.data_model = data_model
        self.model_objective_new = model_objective_new
        self.train_function_new = train_function_new
        self.data_manager = data_manager
        self.model_cost = model_cost
        self.seed = seed
        self.search_range = [[0, 1], [0, 1]]
        self.total_fid_num = total_fidelity_num

    def optimise_adam(self, xtr, ytr, niteration=100, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_cfkg(xtr, ytr)
            loss.backward()
            optimizer.step()
            print('iter', i, 'x:', self.x, 's:', self.s, 'loss_negative_cfkg:',loss.item(), end='\n')


    def negative_cfkg(self, xtr, ytr, x, s):
        xtr_new = copy.deepcopy(xtr)
        ytr_new = copy.deepcopy(ytr)
        # s_index_new = copy.deepcopy(s_index)
        # prediction of xall in the highest fidelity
        # xall = initiate_data_discrete(self.data_name, index, self.total_fid_num, self.seed+117)
        # xall, yall, s_indexall, search_rangeall = initiate_data(self.data_name, 100, 117)

        # xall = torch.from_numpy(np.random.rand(100, 1))
        xall = torch.rand(100, 1).double()
        # s_indexall = np.floor(2 * np.random.rand(100, 1)) + 1
        # yall = gen_data(self.seed+117, self.data_name, xall, 2, 2)

        mean_y, sigma_y = self.pre_func(self.data_manager, xall, self.total_fid_num)# 预测最高精度
        max_mean_y = torch.max(mean_y)
        y = torch.tensor(self.data_model.get_data(x, int(s[0][0])).reshape(-1, 1))
        # y = torch.tensor(gen_data(self.seed, self.data_name, x, s).reshape(-1, 1))
        if isinstance(ytr_new, np.ndarray):
            ytr_new = torch.from_numpy(ytr_new)
        if isinstance(ytr, np.ndarray):
            ytr = torch.from_numpy(ytr)
        xtr_new[int(s[0][0])-1] = np.concatenate((xtr_new[int(s[0][0])-1], x), axis=0)
        # ytr_new = np.concatenate((ytr_new, y), axis=0)
        ytr_new[int(s[0][0])-1] = np.concatenate((ytr_new[int(s[0][0])-1], y.numpy()), axis=0)
        # s_index_new = np.concatenate((s_index_new, s), axis=0)
        # x_total = np.concatenate((xtr, self.x), axis=0)
        # y_total = np.concatenate((ytr, y), axis=0)
        # ytr = torch.cat((ytr, y), axis=0)
        # s_index = np.concatenate((s_index, self.s), axis=0)

        self.model_objective_new.train(xtr_new, ytr_new)
        
        self.train_function_new(self.model_objective_new, self.data_manager, max_iter=10, lr_init=0.01),

        mu, v = self.model_objective_new.predict(self.data_manager, xall, self.total_fid_num)  # tensor
        max_mu = torch.max(mu)
        c = self.model_cost.compute_cost(s)
        cfkg = (max_mu.detach().numpy() - max_mean_y.detach().numpy())/c
        # ytr_size = mean_y.size(0)
        # cfkg = (mu[ytr_size:] - torch.ones(c.shape[0], 1)*max_mean_y) / torch.from_numpy(c)

        return cfkg

    def compute_next(self, xtr, ytr):
        N = 20
        tt = []
        for i in range(len(self.search_range)-1):
            np.random.seed(self.seed + 86 + i)
            tt.append(np.random.uniform(self.search_range[i][0], self.search_range[i][-1], size=(N, 1)))
        tt = np.concatenate(tt, axis=1)

        np.random.seed(self.seed + 86 + 37)
        ts = np.random.uniform(self.search_range[-1][0], self.search_range[-1][-1], size=(N, 1))

        # if self.data_name == "Hartmann":
        #     np.random.seed(self.seed + 86 + 1)
        #     tt = np.random.uniform(self.search_range[0][0], self.search_range[0][-1], size=(N, 6))
        #     np.random.seed(self.seed + 86 + 2)
        #     ts = np.random.uniform(self.search_range[-1][0], self.search_range[-1][-1], size=(N, 1))
        # elif self.data_name == "Branin":
        #     np.random.seed(self.seed + 86 + 1)
        #     tt = np.random.uniform(self.search_range[0][0], self.search_range[0][-1], size=(N, 1))
        #     np.random.seed(self.seed + 86 + 3)
        #     tt2 = np.random.uniform(self.search_range[1][0], self.search_range[1][-1], size=(N, 1))
        #     tt = np.concatenate((tt, tt2), axis=1)
        #     np.random.seed(self.seed + 86 + 2)
        #     ts = np.random.uniform(self.search_range[-1][0], self.search_range[-1][-1], size=(N, 1))
        # elif self.data_name == "mln_mnist":
        #     np.random.seed(self.seed + 86 + 1)
        #     tt = np.random.uniform(self.search_range[0][0], self.search_range[0][-1], size=(N, 1))
        #     np.random.seed(self.seed + 86 + 3)
        #     tt2 = np.random.uniform(self.search_range[1][0], self.search_range[1][-1], size=(N, 1))
        #     tt = np.concatenate((tt, tt2), axis=1)
        #     np.random.seed(self.seed + 86 + 2)
        #     ts = np.random.uniform(self.search_range[-1][0], self.search_range[-1][-1], size=(N, 1))
        # print(tt)

        # self.x = nn.Parameter(torch.from_numpy(tt).double())
        # self.s = nn.Parameter(torch.from_numpy(ts).double())
        # self.optimise_adam(xtr=xtr, ytr=ytr, niteration=100, lr=0.1)
        # self.x = torch.from_numpy(tt).double()
        # self.s = torch.from_numpy(ts).double()
        s = np.ones(N) + 1
        max_cfkg = sys.float_info.min
        new_x = 0.5 * (self.search_range[0][-1] + self.search_range[0][0]) * np.ones(1).reshape(-1, 1)
        new_s = 2 * np.ones(1).reshape(-1, 1)
        tem = torch.from_numpy(tt[i].reshape(1, tt.shape[1]))
        for i in range(N):
            cfkg = self.negative_cfkg(xtr, ytr, tem, s[i].reshape(1, 1))
            if cfkg > max_cfkg:
                max_cfkg = cfkg
                new_x = tt[i].reshape(1, tt.shape[1])
                new_s = s[i].reshape(1, 1)
        # print(max_cfkg)
        # self.optimise_adam(xtr=xtr, ytr=ytr, niteration=100, lr=0.1)
        # idx = np.argmax(cfkg.detach().numpy())
        # new_x = self.x[idx]
        # new_x = new_x.reshape(1, self.x.shape[1])
        # new_s = self.s[idx].reshape(1, 1)

        # new_x = self.x.detach()
        # new_s = self.s.detach()

        return new_x, new_s

    def get_value(self, xtr, ytr, s_index, x):
        N = x.shape[0]
        s = np.ones(N)
        kg_val = []
        for i in range(N):
            cfkg = self.negative_cfkg(xtr, ytr, s_index, x[i].reshape(1, x.shape[1]), s[i].reshape(1, 1))
            kg_val.append(cfkg)

        return kg_val