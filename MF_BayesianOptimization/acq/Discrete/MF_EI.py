import numpy as np
import torch.nn as nn
import torch
from scipy.stats import norm


class expected_improvement(nn.Module):
    def __init__(self, x_dimension, fidelity_num, GP_model, cost, data_manager,search_range,model_name,threshold = 1e-6, seed = 0):
        super(expected_improvement, self).__init__()

        self.x_dimension = x_dimension
        self.fidelity_num = fidelity_num
        self.GP_model = GP_model
        self.cost = cost
        self.data_manager = data_manager
        self.threshold = threshold
        self.search_range = search_range
        self.model_name = model_name
        self.seed = seed

    def calculate_muti_muvar(self, x):
        var_mifi_inverse = 0
        mu_var = 0
        for fi in range(self.fidelity_num):
            # x = self.data_manager.normalizelayer[fi].normalize_x(x)
            mu, var = self.GP_model(self.data_manager, x, fi, normal = False)
            # mu, var = self.data_manager.normalizelayer[fi].denormalize(mu, var)
            var_mifi_inverse += 1 / var
            mu_var += mu / var

        var_mifi = 1 / var_mifi_inverse
        mu_mifi = var_mifi * mu_var
        return mu_mifi, var_mifi

    def negative_ei(self, x):
        # prediction of xall in the highest fidelity

        mu_mifi, var_mifi  = self.calculate_muti_muvar(x)

        if self.model_name in ['CMF_CAR','GP','CMF_CAR_dkl']:
            y_min = min(self.data_manager.get_data(0, normal = False)[1])
        else:
            y_min = min(min(self.data_manager.get_data(fidelity_num, normal = False)[1]) for fidelity_num in range(self.fidelity_num))
        # test_x = self.data_manager.normalizelayer[self.GP_model.fidelity_num-1].normalize_x(x)

        std_var_mifi = torch.sqrt(var_mifi)
            
        z = (y_min - mu_mifi) / std_var_mifi
        cdf_value = torch.tensor(norm.cdf(z.detach().numpy()), dtype=torch.float32)
        pdf_value = torch.tensor(norm.pdf(z.detach().numpy()), dtype=torch.float32)
        ei = (y_min - mu_mifi) * cdf_value + std_var_mifi * pdf_value
        ## loss is to minimize, so there is a negative sign
        return -ei
    
    def select_minmu(self,x):
        optimizer = torch.optim.Adam([x], lr = 1e-3)
        for i in range(100):
            optimizer.zero_grad()
            mu, _ = self.calculate_muti_muvar(x)
            mu.backward()
            optimizer.step()
            print('iter', i, 'x:', x, 'mu_value:', mu.item(), end='\r')

    def select_fidelity(self, x_next):
        fi_var_list = []
        fi_var_without_currfi = []
        var_mifi_inverse = 0
        for fi in range(self.fidelity_num):
            _, var = self.GP_model(self.data_manager, x_next, fi, normal = False)
            fi_var_list.append(var)
            var_mifi_inverse += 1 / var

        var_mifi = 1 / var_mifi_inverse
    
        for fi in range(self.fidelity_num):
            fi_cost = self.cost.compute_cost(fi)
            fi_var_without_currfi.append(var_mifi_inverse - 1 / fi_var_list[fi])
            fi_search_cost = fi_cost / (var_mifi-fi_var_without_currfi[fi])  ## need abs()?
            if fi == 0:
                min_search_cost = fi_search_cost
                new_s = fi
            else:
                if fi_search_cost < min_search_cost:
                    min_search_cost = fi_search_cost
                    new_s = fi
        return new_s
               
    def compute_next(self):
        torch.manual_seed(self.seed)

        ## flag is use to choose whether to use x_init2
        flag = 0
        x_init = torch.rand(1, self.x_dimension, dtype=torch.float64)*(self.search_range[1] - self.search_range[0]) + self.search_range[0]
        x_init2 = x_init.clone()
        
        self.x_init = nn.Parameter(x_init)
        self.x_init2 = nn.Parameter(x_init2)

        optimizer = torch.optim.Adam([self.x_init], lr = 1e-3)
        for i in range(100):
            optimizer.zero_grad()
            loss = self.negative_ei(self.x_init)
            loss.backward()
            optimizer.step()
            # self.x_init.clamp_(0.0, 1.0)
            print('iter', i, 'x:', self.x_init, 'loss_negative_ei:',loss.item(), end='\r')

        ## search strategy2
        if -loss.item() < self.threshold:
            flag = 1
            print('The improvement is less than the threshold, choose the minimum mean value')
            self.select_minmu(self.x_init2)

        # choose fidelity
        if flag == 0:
            new_x = self.x_init.detach()
        else:
            new_x = x_init2.detach()
        
        new_s = self.select_fidelity(new_x)

        return new_x.reshape(1,-1), new_s
    

