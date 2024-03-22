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

# Reserve part for future development
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

    def __init__(self, kernel1, lf, hf, b, initial_length_scale=1.0, initial_signal_variance=1.0, eps=1e-3):
        super().__init__()
        self.kernel1 = kernel1
        self.b = b
        self.lf = lf
        self.hf = hf
        self.length_scales = nn.Parameter(torch.tensor([initial_length_scale]))
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
        length_scales = torch.abs(self.length_scales) + self.eps
        N = 100
        torch.manual_seed(self.seed)
        # print(torch.rand(1))
        z1 = torch.rand(N) * (self.hf - self.lf) + self.lf # 这块需要用来调整z选点的范围
        z2 = torch.rand(N) * (self.hf - self.lf) + self.lf

        dist_z = (z1 / length_scales - z2 / length_scales) ** 2

       
        z_part1 = -self.b * (self.hf - z1)
        z_part2 = -self.b * (self.hf - z2)

        ##z_part 计算被积函数在采样点上的取值
        z_part  = (z_part1 + z_part2 - 0.5 * dist_z).exp()
        ## z_part_mc 计算MC积分的估计值
        z_part_mc = z_part.mean() * (self.hf - self.lf) * (self.hf - self.lf)

        return self.signal_variance.abs() * z_part_mc * self.kernel1(x1, x2)

class ContinuousAutoRegression(nn.Module):
    # initialize the model
    def __init__(self, fidelity_num, kernel_list, b_init=1.0, if_nonsubset=False):
        super().__init__()
        self.fidelity_num = fidelity_num
        self.b = torch.nn.Parameter(torch.tensor(b_init))

        # create the model
        self.cigp_list=[]
        self.cigp_list.append(GPR(kernel=kernel_list[0], log_beta=1.0))

        for fidelity_low in range(self.fidelity_num - 1):
            low_fidelity_indicator, high_fidelity_indicator = warp_function(fidelity_low, fidelity_low+1)
            # input_dim = kernel_list[0].length_scale.shape[0]
            kernel_residual = fidelity_kernel_MCMC(kernel_list[fidelity_low+1],
                                                   low_fidelity_indicator, high_fidelity_indicator, self.b)
            self.cigp_list.append(GPR(kernel=kernel_residual, log_beta=1.0))
        
        self.cigp_list = torch.nn.ModuleList(self.cigp_list)
        self.if_nonsubset = if_nonsubset


    def forward(self, data_manager, x_test, to_fidelity=None):

        if to_fidelity is not None:
            fidelity_level = to_fidelity
        else:
            fidelity_level = self.fidelity_num - 1
        # predict the model
        for i_fidelity in range(fidelity_level + 1):
            if i_fidelity == 0:
                x_train,y_train = data_manager.get_data(i_fidelity, normal = True)
                y_pred_low, cov_pred_low = self.cigp_list[i_fidelity](x_train,y_train,x_test)
                if fidelity_level == 0:
                    y_pred_high = y_pred_low
                    cov_pred_high = cov_pred_low
            else:
                x_train, y_train = data_manager.get_data_by_name('res-{}'.format(i_fidelity))
                y_pred_res, cov_pred_res= self.cigp_list[i_fidelity](x_train,y_train,x_test)
                y_pred_high = y_pred_low + self.b * y_pred_res ## ?
                cov_pred_high = cov_pred_low + (self.b **2) * cov_pred_res 

                ## for next fidelity
                y_pred_low = y_pred_high
                cov_pred_low = cov_pred_high

        # return the prediction
        return y_pred_high, cov_pred_high
    
def train_CAR(CARmodel, data_manager, max_iter=1000, lr_init=1e-1, debugger = None):
    
    for i_fidelity in range(CARmodel.fidelity_num):
        optimizer = torch.optim.Adam(CARmodel.parameters(), lr=lr_init)
        if i_fidelity == 0:
            x_low,y_low = data_manager.get_data(i_fidelity, normal = True)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -CARmodel.cigp_list[i_fidelity].negative_log_likelihood(x_low, y_low)
                if debugger is not None:
                    debugger.get_status(CARmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')
        else:
            if CARmodel.if_nonsubset:
                with torch.no_grad():
                    subset_x, y_low, y_high = data_manager.get_nonsubset_fill_data(CARmodel, i_fidelity-1, i_fidelity)
            else:
                _, y_low, subset_x,y_high = data_manager.get_overlap_input_data(i_fidelity-1, i_fidelity, normal = True)
            for i in range(max_iter):
                optimizer.zero_grad()
                if CARmodel.if_nonsubset:
                    y_residual = y_high[0] - CARmodel.b.exp() * y_low[0] # 修改
                    y_residual_var = abs(y_high[1] - (CARmodel.b.exp() ** 2) * y_low[1])
                else:
                    y_residual = y_high - CARmodel.b.exp() * y_low
                    y_residual_var = None
                if i == max_iter-1:
                    if y_residual_var is not None:
                        y_residual_var = y_residual_var.detach()
                    data_manager.add_data(fidelity_index=None,raw_fidelity_name='res-{}'.format(i_fidelity),x=subset_x.detach(),y=[y_residual.detach(),y_residual_var])
                loss = -CARmodel.cigp_list[i_fidelity].negative_log_likelihood(subset_x, y_residual)
                if debugger is not None:
                    debugger.get_status(CARmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i,'b:',CARmodel.b.item(), 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{},b {}, nll: {}'.format(i_fidelity, i+1, max_iter,CARmodel.b.item(), loss.item()), end='\r')
            print('')
            
# demo 
if __name__ == "__main__":

    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debugger=log_debugger("CAR")

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

    fidelity_manager = MultiFidelityDataManager(initial_data)
    fidelity_num = 3
    kernel_list = [kernel.SquaredExponentialKernel() for _ in range(fidelity_num)]
    # kernel_residual = fidelity_kernel_MCMC(x_low.shape[1], kernel.ARDKernel(x_low.shape[1]), 1, 2)
    CAR = ContinuousAutoRegression(fidelity_num=fidelity_num, kernel_list=kernel_list, b_init=1.0).to(device)

    debugger.logger.info('training finished,start predicting')
    train_CAR(CAR,fidelity_manager, max_iter=200, lr_init=1e-2, debugger = debugger)

    with torch.no_grad():
        x_test = fidelity_manager.normalizelayer[CAR.fidelity_num-1].normalize_x(x_test)
        ypred, ypred_var = CAR(fidelity_manager,x_test)
        ypred, ypred_var = fidelity_manager.normalizelayer[CAR.fidelity_num-1].denormalize(ypred, ypred_var)
    
    debugger.logger.info('prepare to plot')
    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha=0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    # plt.plot(x_high1.flatten(), y_high1.flatten(), 'b+')
    plt.show() 