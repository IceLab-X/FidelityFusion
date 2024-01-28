import numpy as np
import torch.nn as nn
import torch
from scipy.stats import norm
import sys
import copy


class expected_improvement(nn.Module):
    def __init__(self, x_dimension, fidelity_num, posterior_function, model_objective_new, data_name, target_func, cost_model, seed):
        super(expected_improvement, self).__init__()

        self.x_dimension = x_dimension
        self.fidelity_num = fidelity_num
        self.pre_func = posterior_function
        self.model_objective_new = model_objective_new
        self.data_name = data_name
        self.target_func = target_func
        self.cost_model = cost_model
        self.seed = seed

    def optimise_adam(self, xall, niteration=100, lr=0.1):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            loss = self.negative_ei(xall)
            loss.backward()
            optimizer.step()
            self.x.data.clamp_(0.0, 1.0)
            print('iter', i, 'x:', self.x, 'loss_negative_ei:',loss.item(), end='\n')

    def negative_ei(self, xall):
        # prediction of xall in the highest fidelity
        mean_y, var_y = self.pre_func(xall, self.fidelity_num)
        sigma_y = torch.sqrt(var_y)
        # prediction of new x in the highest fidelity
        mean_y_new, var_y_new = self.pre_func(self.x, self.fidelity_num)
        sigma_y_new = torch.sqrt(var_y_new)
        if sigma_y_new == 0.0:
            ei = torch.tensor(np.array([[1e-5]]), requires_grad=True)
        else:
            max_mean_y = torch.max(mean_y)
            z = (mean_y_new - max_mean_y) / sigma_y_new
            cdf_value = norm.cdf(z.detach().numpy())
            pdf_value = norm.pdf(z.detach().numpy())
            if cdf_value == 0:
                cdf_value = np.array([[1e-5]])
            if pdf_value == 0:
                pdf_value = np.array([[1e-5]])
            ei = (mean_y_new - max_mean_y) * cdf_value[0][0] + sigma_y_new * pdf_value[0][0]
        return -ei


    def compute_next(self, xtr, ytr, xall):
        np.random.seed(self.seed + 10086)
        tt = np.random.rand(1, self.x_dimension)
        print(tt)
        self.x = nn.Parameter(torch.from_numpy(tt).double())
        self.optimise_adam(xall=xall, niteration=100, lr=0.1)


        # choose fidelity
        new_x = self.x.detach()
        mean_y_opt, var_y_opt = self.pre_func(new_x, self.fidelity_num)
        sigma_y_opt = torch.sqrt(var_y_opt)
        min_mark = float(sys.maxsize)
        for i in range(self.fidelity_num):
            xtr_new = copy.deepcopy(xtr)
            ytr_new = copy.deepcopy(ytr)
            xtr_new[i] = np.concatenate((xtr_new[i], new_x), axis=0)
            if new_x.shape[1] == 6:
                new_input = np.concatenate((new_x.numpy(), np.double(i).reshape(1, 1)), axis=1)
            else:
                new_input = new_x.numpy()
            new_y = self.target_func(new_input, i + 1)
            if len(new_y.shape) == 1:
                d = new_y.shape[0]
                new_y = new_y.reshape(1, d)
            ytr_new[i] = np.concatenate((ytr_new[i], new_y), axis=0)
            self.model_objective_new.train(xtr_new, ytr_new)
            mean_m_y_opt, var_m_y_opt = self.model_objective_new.predict(new_x,
                                                                           self.fidelity_num)
            sigma_m_y_opt = torch.sqrt(var_m_y_opt)
            mark = self.cost_model.compute_cost(i + 1) / (sigma_y_opt.detach().numpy() ** 2 - sigma_m_y_opt.detach().numpy() ** 2)
            if mark < min_mark:
                min_mark = mark
                new_s = i + 1


            # new_x = new_x.numpy()

            if len(new_x.shape) == 1:
                d = new_x.shape[0]
                new_x = new_x.reshape(1, d)

        return new_x, new_s

    def get_value(self, xall, x_range):
        ei_list = []
        for x in x_range:
            # prediction of xall in the highest fidelity
            mean_y, var_y = self.pre_func(xall, self.fidelity_num)
            sigma_y = torch.sqrt(var_y)
            # prediction of new x in the highest fidelity
            mean_y_new, var_y_new = self.pre_func(x, self.fidelity_num)
            sigma_y_new = torch.sqrt(var_y_new)
            if sigma_y_new == 0.0:
                ei = torch.tensor(np.array([[1e-5]]), requires_grad=True)
            else:
                max_mean_y = torch.max(mean_y)
                z = (mean_y_new - max_mean_y) / sigma_y_new
                cdf_value = norm.cdf(z.detach().numpy())
                pdf_value = norm.pdf(z.detach().numpy())
                if cdf_value == 0:
                    cdf_value = np.array([[1e-5]])
                if pdf_value == 0:
                    pdf_value = np.array([[1e-5]])
                ei = (mean_y_new - max_mean_y) * cdf_value[0][0] + sigma_y_new * pdf_value[0][0]
            ei_list.append(ei)
        return ei_list
