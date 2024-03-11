import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from GaussianProcess.cigp_v10 import cigp as GPR
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from Experiments.log_debugger import log_debugger
import matplotlib.pyplot as plt

class ad_alpha(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
class AR_aa(nn.Module):

    def __init__(self, fidelity_num, kernel_list, input_size, if_nonsubset=False):
        
        super().__init__()
        self.fidelity_num = fidelity_num
        self.gpr_list = []
        for i in range(self.fidelity_num):
            self.gpr_list.append(GPR(kernel=kernel_list[i], log_beta=1.0))
        self.gpr_list = torch.nn.ModuleList(self.gpr_list)

        self.rho_networks = torch.nn.ModuleList([
            ad_alpha(input_size, hidden_size = 4, output_size = 1) for _ in range(self.fidelity_num - 1)
        ])
        self.if_nonsubset = if_nonsubset

    def forward(self, data_manager, x_test, to_fidelity=None):
        
        if to_fidelity is not None:
            fidelity_level = to_fidelity
        else:
            fidelity_level = self.fidelity_num - 1
        for i_fidelity in range(fidelity_level + 1):
            if i_fidelity == 0:
                x_train, y_train = data_manager.get_data(i_fidelity, normal=True)
                y_pred_low, cov_pred_low = self.gpr_list[i_fidelity](x_train, y_train, x_test)
                if fidelity_level == 0:
                    y_pred_high = y_pred_low
                    cov_pred_high = cov_pred_low
            else:
                x_train, y_train = data_manager.get_data_by_name('res-{}'.format(i_fidelity))
                y_pred_res, cov_pred_res = self.gpr_list[i_fidelity](x_train, y_train, x_test)
                y_pred_high = y_pred_low + self.rho_networks[i_fidelity - 1](x_test) * y_pred_res
                cov_pred_high = cov_pred_low + (self.rho_networks[i_fidelity - 1](x_test) ** 2) * cov_pred_res
                y_pred_low = y_pred_high
                cov_pred_low = cov_pred_high

        return y_pred_high, cov_pred_high
#train_gp
    
def train_ARaa(ARmodel, data_manager, max_iter=1000, lr_init=1e-1, debugger=None):
    
    for i_fidelity in range(ARmodel.fidelity_num):
        optimizer = torch.optim.Adam(ARmodel.parameters(), lr=lr_init)
        if i_fidelity == 0:
            x_low, y_low = data_manager.get_data(i_fidelity, normal=True)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -ARmodel.gpr_list[i_fidelity].negative_log_likelihood(x_low, y_low)
                if debugger is not None:
                    debugger.get_status(ARmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')
        else:
            if ARmodel.if_nonsubset:
                with torch.no_grad():
                    subset_x, y_low, y_high = data_manager.get_nonsubset_fill_data(ARmodel, i_fidelity - 1, i_fidelity)
            else:
                _, y_low, subset_x, y_high = data_manager.get_overlap_input_data(i_fidelity - 1, i_fidelity, normal=True)
            for i in range(max_iter):
                optimizer.zero_grad()
                if ARmodel.if_nonsubset:
                    y_residual_mean = y_high[0] - ARmodel.rho_networks[i_fidelity - 1](subset_x) * y_low[0]
                    y_residual_var = abs(y_high[1] - ARmodel.rho_networks[i_fidelity - 1](subset_x) * y_low[1])
                else:
                    y_residual_mean = y_high - ARmodel.rho_networks[i_fidelity - 1](subset_x) * y_low
                    y_residual_var = None
                if i == max_iter - 1:
                    if y_residual_var is not None:
                        y_residual_var = y_residual_var.detach()
                    data_manager.add_data(raw_fidelity_name='res-{}'.format(i_fidelity), fidelity_index=None, x=subset_x.detach(), y=[y_residual_mean.detach(), y_residual_var])
                loss = -ARmodel.gpr_list[i_fidelity].negative_log_likelihood(subset_x, [y_residual_mean, y_residual_var])
                if debugger is not None:
                    debugger.get_status(ARmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i,'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')
            
# demo 
if __name__ == "__main__":

    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debugger=log_debugger("AR_aa")

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

    y_low = torch.sin(x_low) - 0.5 * torch.sin(2 * x_low) + torch.rand(300, 1) * 0.1 - 0.05
    y_high1 = torch.sin(x_high1) - 0.3 * torch.sin(2 * x_high1) + torch.rand(300, 1) * 0.1 - 0.05
    y_high2 = torch.sin(x_high2) + torch.rand(250, 1) * 0.1 - 0.05
    y_test = torch.sin(x_test)

    initial_data = [
        {'raw_fidelity_name': '0','fidelity_indicator': 0, 'X': x_low.to(device), 'Y': y_low.to(device)},
        {'raw_fidelity_name': '1','fidelity_indicator': 1, 'X': x_high1.to(device), 'Y': y_high1.to(device)},
        {'raw_fidelity_name': '2','fidelity_indicator': 2, 'X': x_high2.to(device), 'Y': y_high2.to(device)},
    ]
    fidelity_num = len(initial_data)

    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel_list = [kernel.SquaredExponentialKernel() for _ in range(fidelity_num)]
    myAR = AR_aa(fidelity_num = fidelity_num, kernel_list = kernel_list, input_size = x_low.shape[1], if_nonsubset=False).to(device)

    ## if nonsubset is False, max_iter should be 100 ,lr can be 1e-2
    train_ARaa(myAR, fidelity_manager, max_iter=200, lr_init=1e-2, debugger = debugger)

    debugger.logger.info('training finished,start predicting')
    with torch.no_grad():
        x_test = fidelity_manager.normalizelayer[myAR.fidelity_num-1].normalize_x(x_test)
        ypred, ypred_var = myAR(fidelity_manager,x_test)
        ypred, ypred_var = fidelity_manager.normalizelayer[myAR.fidelity_num-1].denormalize(ypred, ypred_var)

    debugger.logger.info('prepare to plot')
    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha = 0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    plt.show() 
