import sys
sys.path.append(r'H:\\eda\\mybranch')
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from two_fidelity_models.base.gp_basic import GP_basic as CIGP
from MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt

class autoRegression(nn.Module):
    # initialize the model
    def __init__(self,fidelity_num,kernel,rho_init=1.0):
        super().__init__()
        self.fidelity_num = fidelity_num
        # create the model
        self.cigp_list=[]
        for _ in range(self.fidelity_num):
            self.cigp_list.append(CIGP(kernel=kernel, noise_variance=1.0))
        
        self.cigp_list = torch.nn.ModuleList(self.cigp_list)

        self.rho_list=[]
        for _ in range(self.fidelity_num-1):
            self.rho_list.append(torch.nn.Parameter(torch.tensor(rho_init)))

        self.rho_list = torch.nn.ParameterList(self.rho_list)


    def forward(self,x_test):
        # predict the model
        for f in range(self.fidelity_num):
            if f == 0:
                y_pred_low, cov_pred_low = self.cigp_list[f](x_test)
            else:
                y_pred_res, cov_pred_res= self.cigp_list[f](x_test)
                y_pred_high = y_pred_low + self.rho_list[f-1] * y_pred_res
                cov_pred_high = cov_pred_low + (self.rho_list[f-1] **2) * cov_pred_res

                ## for next fidelity
                y_pred_low = y_pred_high
                cov_pred_low = cov_pred_high

        # return the prediction
        return y_pred_high, cov_pred_high
    

def train_AR(ARmodel,data_manager,max_iter=1000,lr_init=1e-1):
    
    # train the low fidelity GP
    optimizer = torch.optim.Adam(ARmodel.parameters(), lr=lr_init)
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = 0.
        for f in range(ARmodel.fidelity_num):
            if f == 0:
                x_low,y_low = data_manager.get_data(str(f))
                loss += -ARmodel.cigp_list[f].log_likelihood(x_low, y_low)
            else:
                _, y_low, subset_x,y_high = data_manager.get_overlap_input_data(str(f-1),str(f))
                y_residuaal = y_high - ARmodel.rho_list[f-1] * y_low
                loss += -ARmodel.cigp_list[f].log_likelihood(subset_x, y_residuaal)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))
    
# demo 
if __name__ == "__main__":

    torch.manual_seed(1)

    # generate the data
    x_all = torch.rand(500, 1) * 20
    xlow_indices = torch.randperm(500)[:300]
    x_low = x_all[xlow_indices]
    xhigh1_indices = torch.randperm(500)[:300]
    x_high1 = x_all[xhigh1_indices]
    xhigh2_indices = torch.randperm(500)[:250]
    x_high2 = x_all[xhigh2_indices]
    x_test = torch.linspace(0, 20, 100).reshape(-1, 1)

    y_low = torch.sin(x_low) + torch.rand(300, 1) * 0.6 - 0.3
    y_high1 = torch.sin(x_high1) + torch.rand(300, 1) * 0.4 - 0.2
    y_high2 = torch.sin(x_high2) + torch.rand(250, 1) * 0.2 - 0.1
    y_test = torch.sin(x_test)

    initial_data = [
        {'fidelity_index': '0', 'X': x_low, 'Y': y_low},
        {'fidelity_index': '1', 'X': x_high1, 'Y': y_high1},
        {'fidelity_index': '2', 'X': x_high2, 'Y': y_high2},
        {'fidelity_index': '-1', 'X': x_test, 'Y': y_test}
    ]

    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
    AR = autoRegression(fidelity_num=3,kernel=kernel1,rho_init=1.0)

    train_AR(AR,fidelity_manager, max_iter=200, lr_init=1e-2)

    with torch.no_grad():
        ypred, ypred_var = AR(x_test)

    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha=0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    plt.show() 
