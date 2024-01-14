import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from GaussianProcess.gp_basic import GP_basic as GPR
from MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt

def warp_function(lf, hf):
    return lf, hf

class fidelity_kernel_MCMC(nn.Module):
    """
    fidelity kernel module base ARD and use MCMC to calculate the integral.

    Args:
        input_dim (int): The input dimension.
        initial_length_scale (float): The initial length scale value. Default is 1.0.
        initial_signal_variance (float): The initial signal variance value. Default is 1.0.
        eps (float): A small constant to prevent division by zero. Default is 1e-9.

    Attributes:
        length_scales (nn.Parameter): The length scales for each dimension.
        signal_variance (nn.Parameter): The signal variance.
        eps (float): A small constant to prevent division by zero.

    """

    def __init__(self, input_dim, kernel1, b, initial_length_scale=0.0, initial_signal_variance=1.0, eps=1e-3):
        super().__init__()
        self.kernel1 = kernel1
        self.b = b
        self.log_length_scales = nn.Parameter(torch.ones(input_dim) * initial_length_scale)
        self.signal_variance = nn.Parameter(torch.tensor([initial_signal_variance]))
        self.eps = eps
        self.seed = 105

    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the ARD kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        X1 = x1[:, 0].reshape(-1, 1)
        X2 = x2[:, 0].reshape(-1, 1)
        fidelity_indicator_1 = x1[:, 1].reshape(-1, 1)
        fidelity_indicator_2 = x2[:, 1].reshape(-1, 1)

        N = 100
        torch.manual_seed(self.seed)
        t1 = torch.rand(N).float().reshape(N, 1) # 这块需要用来调整z选点的范围
        t2 = torch.rand(N).float().reshape(N, 1)
        
        tem = [fidelity_indicator_1 for i in range(fidelity_indicator_2.size(0))]
        S1 = torch.cat(tem, dim=1)
        tem = [fidelity_indicator_2 for i in range(fidelity_indicator_1.size(0))]
        S2 = torch.cat(tem, dim=1)
        
        z_11 = torch.pow(S1, 2)
        z_22 = torch.pow(S2, 2)
        t_prod = torch.sum(t1*t2) * torch.ones(fidelity_indicator_1.size(0), fidelity_indicator_2.size(0))
        z_1 = fidelity_indicator_1 @ t1.t()
        z_2 = fidelity_indicator_2 @ t2.t()
        z_12 = z_1 @ z_2.t()
        dist_z = (z_11 + z_22.t())*t_prod - 2*z_12
        dist_z = dist_z / (2 * self.log_length_scales).exp()
        
        S = S1 + S2.t()
        t_part = 0.005 * torch.sum((3-t1) + (3-t2)) * torch.ones(fidelity_indicator_1.size(0), fidelity_indicator_2.size(0))
        z_part = (-self.b*(S - t_part) - 0.5 * dist_z).exp()/N
        
        z_part_mc = z_part * S1*S2.t()

        return self.signal_variance.abs() * z_part_mc * self.kernel1(X1, X2)

class ContinuousAutoRegression_large(nn.Module):
    # initialize the model
    def __init__(self, fidelity_num, kernel_x, b_init=1.0):
        super().__init__()
        self.fidelity_num = fidelity_num
        self.b = torch.nn.Parameter(torch.tensor(b_init))

        kernel_full = fidelity_kernel_MCMC(kernel_x.length_scales.shape[0], kernel_x, self.b)
        self.cigp = GPR(kernel=kernel_full, noise_variance=1.0)

    def forward(self, data_manager, x_test):
        # predict the model

        x_train = []
        y_train = []
        fidelity_indicator = []
        for i_fidelity in range(self.fidelity_num):
            x, y = data_manager.get_data(i_fidelity)
            x_train.append(x)
            y_train.append(y)
            fidelity_indicator.append(torch.ones(x.shape[0]) * (i_fidelity+1))
        
        x_train = torch.cat(x_train, 0)
        y_train = torch.cat(y_train, 0)
        fidelity_indicator = torch.cat(fidelity_indicator, 0)
        x_train = torch.cat((x_train, fidelity_indicator.reshape(-1,1)), 1)
        
        x_test =  torch.cat((x_test, (torch.ones(x_test.shape[0]) * self.fidelity_num).reshape(-1,1)), 1)

        y_pred, cov_pred = self.cigp(x_train,y_train,x_test)

        # return the prediction
        return y_pred, cov_pred
    
def train_CAR(CARmodel, data_manager,max_iter=1000,lr_init=1e-1):
    x_train = []
    y_train = []
    fidelity_indicator = []
    for i_fidelity in range(CARmodel.fidelity_num):
        x, y = data_manager.get_data(i_fidelity)
        x_train.append(x)
        y_train.append(y)
        fidelity_indicator.append(torch.ones(x.shape[0]) * (i_fidelity+1))
    
    x_train = torch.cat(x_train, 0)
    y_train = torch.cat(y_train, 0)
    fidelity_indicator = torch.cat(fidelity_indicator, 0)
    x_train = torch.cat((x_train, fidelity_indicator.reshape(-1,1)), 1)

    optimizer = torch.optim.Adam(CARmodel.parameters(), lr=lr_init)
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = -CARmodel.cigp.log_likelihood(x_train, y_train)
        loss.backward()
        optimizer.step()
        print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
    
    
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

    y_low = torch.sin(x_low) - 0.5 * torch.sin(2 * x_low) + torch.rand(300, 1) * 0.1 - 0.05
    y_high1 = torch.sin(x_high1) - 0.3 * torch.sin(2 * x_high1) + torch.rand(300, 1) * 0.1 - 0.05
    y_high2 = torch.sin(x_high2) + torch.rand(250, 1) * 0.1 - 0.05
    y_train = torch.cat((y_low, y_high1, y_high2), 0)
    y_test = torch.sin(x_test)


    initial_data = [
        {'raw_fidelity_name': '0','fidelity_indicator': 0, 'X': x_low, 'Y': y_low},
        {'raw_fidelity_name': '1','fidelity_indicator': 1, 'X': x_high1, 'Y': y_high1},
        {'raw_fidelity_name': '2','fidelity_indicator': 2, 'X': x_high2, 'Y': y_high2},
    ]

    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel_x = kernel.ARDKernel(x_low.shape[1])
    CAR = ContinuousAutoRegression_large(fidelity_num=2, kernel_x=kernel_x, b_init=1.0)

    train_CAR(CAR,fidelity_manager, max_iter=100, lr_init=1e-2)

    with torch.no_grad():
        ypred, ypred_var = CAR(fidelity_manager,x_test)
 
    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha=0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    plt.show() 