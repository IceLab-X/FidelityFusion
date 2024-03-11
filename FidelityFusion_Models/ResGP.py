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

class ResGP(nn.Module):
    """
    Residual Gaussian Process model.

    Args:
        fidelity_num (int): Number of fidelity levels.
        kernel_list (list): List of kernels for each fidelity level.
        if_nonsubset (bool, optional): Flag indicating if non-subset data is used. Defaults to False.
    """

    def __init__(self, fidelity_num, kernel_list, if_nonsubset=False):
        super().__init__()
        self.fidelity_num = fidelity_num
        self.gpr_list = []
        for i in range(self.fidelity_num):
            self.gpr_list.append(GPR(kernel=kernel_list[i], log_beta=1.0))
        self.gpr_list = torch.nn.ModuleList(self.gpr_list)
        self.if_nonsubset = if_nonsubset

    def forward(self, data_manager, x_test, to_fidelity=None):
        """
        Forward pass of the ResGP model.

        Args:
            data_manager (DataManager): Data manager object.
            x_test (torch.Tensor): Test input data.
            to_fidelity (int, optional): Fidelity level to predict. Defaults to None.

        Returns:
            torch.Tensor: Predicted output.
            torch.Tensor: Covariance of the predictions.
        """
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
                y_pred_high = y_pred_low + y_pred_res
                cov_pred_high = cov_pred_low + cov_pred_res

                # for next fidelity
                y_pred_low = y_pred_high
                cov_pred_low = cov_pred_high

        return y_pred_high, cov_pred_high
    
def train_ResGP(ResGPmodel, data_manager, max_iter=1000, lr_init=1e-1, debugger=None):
    """
    Trains the Residual Gaussian Process (ResGP) model.

    Args:
        ResGPmodel (ResGPModel): The ResGP model to train.
        data_manager (DataManager): The data manager object.
        max_iter (int, optional): The maximum number of iterations for training. Defaults to 1000.
        lr_init (float, optional): The initial learning rate for the optimizer. Defaults to 1e-1.
        debugger (Debugger, optional): The debugger object for monitoring training progress. Defaults to None.
    """
    for i_fidelity in range(ResGPmodel.fidelity_num):
        optimizer = torch.optim.Adam(ResGPmodel.parameters(), lr=lr_init)
        if i_fidelity == 0:
            x_low, y_low = data_manager.get_data(i_fidelity, normal=True)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -ResGPmodel.gpr_list[i_fidelity].negative_log_likelihood(x_low, y_low)
                if debugger is not None:
                    debugger.get_status(ResGPmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')
        else:
            if ResGPmodel.if_nonsubset:
                with torch.no_grad():
                    subset_x, y_low, y_high = data_manager.get_nonsubset_fill_data(ResGPmodel, i_fidelity - 1, i_fidelity)
                y_residual_mean = y_high[0] - y_low[0]
                y_residual_var = abs(y_high[1] - y_low[1])
            else:
                _, y_low, subset_x, y_high = data_manager.get_overlap_input_data(i_fidelity - 1, i_fidelity, normal=True)
                y_residual_mean = y_high - y_low
                y_residual_var = None
            if y_residual_var is not None:
                y_residual_var = y_residual_var.detach()
            data_manager.add_data(raw_fidelity_name='res-{}'.format(i_fidelity), fidelity_index=None, x=subset_x.detach(), y=[y_residual_mean.detach(), y_residual_var])
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -ResGPmodel.gpr_list[i_fidelity].negative_log_likelihood(subset_x, [y_residual_mean, y_residual_var])
                if debugger is not None:
                    debugger.get_status(ResGPmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')
    
# demo 
if __name__ == "__main__":

    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debugger=log_debugger("ResGP")

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
    myResGP = ResGP(fidelity_num = 3,kernel_list=kernel_list, if_nonsubset = True).to(device)

    ## if nonsubset is False, max_iter should be 100 ,lr can be 1e-2
    train_ResGP(myResGP, fidelity_manager, max_iter=200, lr_init=1e-2, debugger = debugger)

    debugger.logger.info('training finished,start predicting')
    with torch.no_grad():
        x_test = fidelity_manager.normalizelayer[myResGP.fidelity_num-1].normalize_x(x_test)
        ypred, ypred_var = myResGP(fidelity_manager, x_test)
        ypred, ypred_var = fidelity_manager.normalizelayer[myResGP.fidelity_num-1].denormalize(ypred, ypred_var)
        
    debugger.logger.info('prepare to plot')
    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt = 'r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha = 0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    plt.show() 
