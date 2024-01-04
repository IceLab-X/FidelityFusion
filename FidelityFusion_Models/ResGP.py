import sys
sys.path.append(r'H:\\eda\\mybranch')
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from GaussianProcess.gp_basic import GP_basic as CIGP
from MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt

class ResGP(nn.Module):
    # initialize the model
    def __init__(self,fidelity_num,kernel):
        super().__init__()
        self.fidelity_num = fidelity_num
        # create the model
        self.cigp_list=[]
        for _ in range(self.fidelity_num):
            self.cigp_list.append(CIGP(kernel=kernel, noise_variance=1.0))
        
        self.cigp_list = torch.nn.ModuleList(self.cigp_list)

    def forward(self,data_manager,x_test):
        # predict the model
        for f in range(self.fidelity_num):
            if f == 0:
                x_train,y_train = data_manager.get_data(f)
                y_pred_low, cov_pred_low = self.cigp_list[f](x_train,y_train,x_test)
                if self.fidelity_num == 1:
                    y_pred_high = y_pred_low
                    cov_pred_high = cov_pred_low
            else:
                x_train,y_train = data_manager.get_data(-f)
                y_pred_res, cov_pred_res= self.cigp_list[f](x_train,y_train,x_test)
                y_pred_high = y_pred_low +  y_pred_res
                cov_pred_high = cov_pred_low + cov_pred_res

                ## for next fidelity
                y_pred_low = y_pred_high
                cov_pred_low = cov_pred_high

        # return the prediction
        return y_pred_high, cov_pred_high
    
def train_ResGP(ResGPmodel,data_manager,max_iter=1000,lr_init=1e-1):
    
    for f in range(ResGPmodel.fidelity_num):
        optimizer = torch.optim.Adam(ResGPmodel.parameters(), lr=lr_init)
        if f == 0:
            x_low,y_low = data_manager.get_data(f)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -ResGPmodel.cigp_list[f].log_likelihood(x_low, y_low)
                loss.backward()
                optimizer.step()
                print('fidelity:', f, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
        else:
            _, y_low, subset_x,y_high = data_manager.get_overlap_input_data(f-1,f)
            y_residual = y_high - y_low
            data_manager.add_data(fidelity_index=-f,raw_fidelity_name='res-{}'.format(f),x=subset_x,y=y_residual)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -ResGPmodel.cigp_list[f].log_likelihood(subset_x, y_residual)
                loss.backward()
                optimizer.step()
                print('fidelity:', f, 'iter', i,'nll:{:.5f}'.format(loss.item()))
    
# demo 
if __name__ == "__main__":

    torch.manual_seed(1)

    # generate the data
    x_all = torch.rand(500, 1) * 20
    xlow_indices = torch.randperm(500)[:300]
    xlow_indices = torch.sort(xlow_indices).values
    x_low = x_all[xlow_indices]
    xhigh1_indices = torch.randperm(500)[:300]
    xhigh1_indices = torch.sort(xhigh1_indices).values
    x_high1 = x_all[xhigh1_indices]
    xhigh2_indices = torch.randperm(500)[:250]
    xhigh2_indices = torch.sort(xhigh2_indices).values
    x_high2 = x_all[xhigh2_indices]
    x_test = torch.linspace(0, 20, 100).reshape(-1, 1)

    y_low = torch.sin(x_low) - torch.rand(300, 1) * 0.2
    y_high1 = torch.sin(x_high1) - torch.rand(300, 1) * 0.1
    y_high2 = torch.sin(x_high2) + torch.rand(250, 1) * 0.1 -0.05
    y_test = torch.sin(x_test)

    initial_data = [
        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_low, 'Y': y_low},
        {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': x_high1, 'Y': y_high1},
        {'fidelity_indicator': 2, 'raw_fidelity_name': '2','X': x_high2, 'Y': y_high2},
    ]

    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
    myResGP = ResGP(fidelity_num=3,kernel=kernel1)

    train_ResGP(myResGP,fidelity_manager, max_iter=100, lr_init=1e-2)

    with torch.no_grad():
        ypred, ypred_var = myResGP(fidelity_manager,x_test)

    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha=0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    plt.show() 
