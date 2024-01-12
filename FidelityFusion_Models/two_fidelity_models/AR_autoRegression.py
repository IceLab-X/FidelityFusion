# auto regression for multi-fidelity fusion.
# This function assumes that the high fidelity input is a superset of the low fidelity input.
# author: Wei Xing
# date: 2023-12-12
# version: 1.0
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from base.gp_basic import GP_basic as GPR
import matplotlib.pyplot as plt

def find_subsets_and_indexes(x_low, x_high):
    # find the overlap set
    flat_x_low = x_low.flatten()
    flat_x_high = x_high.flatten()
    subset_indexes_low = torch.nonzero(torch.isin(flat_x_low, flat_x_high), as_tuple=True)[0]
    subset_indexes_high = torch.nonzero(torch.isin(flat_x_high, flat_x_low), as_tuple=True)[0]
    subset_x = flat_x_low[subset_indexes_low].reshape(-1,1)
    return subset_x, subset_indexes_low, subset_indexes_high

    
class autoRegression_twofidelity(nn.Module):
    # initialize the model
    def __init__(self,rho_init=1.0):
        super().__init__()
        
        # create the model
        kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
        self.low_fidelity_GP = GPR(kernel = kernel1, noise_variance = 1.0)
        
        kernel2 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
        self.Residual_GP = GPR(kernel = kernel2, noise_variance=1.0)
        
        self.rho = nn.Parameter(torch.tensor(rho_init))

    def forward(self,x_test):
        # predict the model
        y_pred_low, cov_pred_low = self.low_fidelity_GP(x_test)
        y_pred_res, cov_pred_res= self.Residual_GP(x_test)
        
        y_pred_high = y_pred_low + self.rho * y_pred_res
        cov_pred_high = cov_pred_low + (self.rho ** 2) * cov_pred_res
        
        # return the prediction
        return y_pred_high, cov_pred_high
    

def train_AR_twofidelity(ARmodel, x_train, y_train,max_iter=1000,lr_init=1e-1):
    # get the data
    x_low = x_train[0]
    y_low = y_train[0]
    x_high = x_train[1]
    y_high = y_train[1]
    
    # train the low fidelity GP
    optimizer_low = torch.optim.Adam(ARmodel.low_fidelity_GP.parameters(), lr = lr_init)
    for i in range(max_iter):
        optimizer_low.zero_grad()
        loss = -ARmodel.low_fidelity_GP.log_likelihood(x_low, y_low)
        loss.backward()
        optimizer_low.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))
    
    # get the high fidelity part that is subset of the low fidelity part
    subset_x, subset_indexes_low, subset_indexes_high = find_subsets_and_indexes(x_low, x_high)
    
    # train the Residual_GP
    optimizer_res = torch.optim.Adam(ARmodel.parameters(), lr=lr_init)

    for i in range(max_iter):
        optimizer_res.zero_grad()
        y_residual = y_high[subset_indexes_high] - ARmodel.rho * y_low[subset_indexes_low]
        loss = -ARmodel.Residual_GP.log_likelihood(subset_x, y_residual)
        loss.backward()
        optimizer_res.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))

        
# demo 
if __name__ == "__main__":
    
    torch.manual_seed(1)
    # generate the data
    x_all = torch.rand(500, 1) * 20
    xlow_indices = torch.randperm(500)[:300]
    xlow_indices = torch.sort(xlow_indices).values
    x_low = x_all[xlow_indices]
    xhigh_indices = torch.randperm(500)[:300]
    xhigh_indices = torch.sort(xhigh_indices).values
    x_high = x_all[xhigh_indices]
    x_test = torch.linspace(0, 20, 100).reshape(-1, 1)

    y_low = torch.sin(x_low) - 0.3 * torch.sin(2 * x_low) + torch.rand(300, 1) * 0.1 - 0.05
    y_high = torch.sin(x_high) + torch.rand(300, 1) * 0.1 - 0.05
    y_test = torch.sin(x_test)

    x_train = [x_low, x_high]
    y_train = [y_low, y_high]

    AR = autoRegression_twofidelity(rho_init = 1.0)
    print("rho_init:", AR.rho.item())
    train_AR_twofidelity(AR, x_train, y_train, max_iter = 600, lr_init = 1e-3)

    with torch.no_grad():
        ypred, ypred_var = AR(x_test)
    print("rho:", AR.rho.item())

    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt = 'r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha = 0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    plt.show() 
