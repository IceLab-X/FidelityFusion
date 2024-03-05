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

class NAR(nn.Module):
    """
    Nonlinear Autoregressive (NAR) model for fidelity fusion.
    
    Args:
        fidelity_num (int): Number of fidelity levels.
        kernel_list (list): List of kernels for Gaussian Process Regression (GPR) at each fidelity level.
        if_nonsubset (bool, optional): Flag indicating if non-subset training is used. Defaults to False.
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
        Forward pass of the NAR model.
        
        Args:
            data_manager (DataManager): Data manager object for accessing training data.
            x_test (torch.Tensor): Test input data.
            to_fidelity (int, optional): Fidelity level to predict. Defaults to None, which predicts the highest fidelity.
        
        Returns:
            torch.Tensor: Predicted output at the specified fidelity level.
            torch.Tensor: Covariance of the predicted output.
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
                x_train, y_train = data_manager.get_data_by_name('concat-{}'.format(i_fidelity))
                concat_input = torch.cat([x_test, y_pred_low.reshape(-1, 1)], dim=-1)
                y_pred_high, cov_pred_high = self.gpr_list[i_fidelity](x_train, y_train, concat_input)
                y_pred_low = y_pred_high
                cov_pred_low = cov_pred_high

        return y_pred_high, cov_pred_high
    
def train_NAR(NARmodel, data_manager, max_iter=1000, lr_init=1e-1, debugger=None):
    """
    Trains the NAR model using the specified data manager.

    Args:
        NARmodel: The NAR model to train.
        data_manager: The data manager object that provides the training data.
        max_iter (optional): The maximum number of iterations for training. Defaults to 1000.
        lr_init (optional): The initial learning rate for the optimizer. Defaults to 0.1.
        debugger (optional): The debugger object for monitoring the training process. Defaults to None.
    """

    for i_fidelity in range(NARmodel.fidelity_num):
        optimizer = torch.optim.Adam(NARmodel.parameters(), lr=lr_init)
        if i_fidelity == 0:
            x_low, y_low = data_manager.get_data(i_fidelity, normal=True)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -NARmodel.gpr_list[i_fidelity].negative_log_likelihood(x_low, y_low)
                if debugger is not None:
                    debugger.get_status(NARmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')
        else:
            if NARmodel.if_nonsubset:
                with torch.no_grad():
                    subset_x, y_low, y_high = data_manager.get_nonsubset_fill_data(NARmodel, i_fidelity - 1, i_fidelity)
                y_low_mean = y_low[0]
                y_high_mean = y_high[0]
                y_high_var = y_high[1]
            else:
                _, y_low_mean, subset_x, y_high_mean = data_manager.get_overlap_input_data(i_fidelity - 1, i_fidelity, normal=True)
                y_high_var = None
            concat_input = torch.cat([subset_x, y_low_mean], dim=-1)
            data_manager.add_data(raw_fidelity_name='concat-{}'.format(i_fidelity), fidelity_index=None, x=concat_input.detach(), y=[y_high_mean.detach(), y_high_var.detach()])
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -NARmodel.gpr_list[i_fidelity].negative_log_likelihood(concat_input, [y_high_mean, y_high_var])
                if debugger is not None:
                    debugger.get_status(NARmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')
    
# demo 
if __name__ == "__main__":

    torch.manual_seed(1)
    debugger=log_debugger("NAR")

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
        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_low, 'Y': y_low},
        {'fidelity_indicator': 1, 'raw_fidelity_name': '1', 'X': x_high1, 'Y': y_high1},
        {'fidelity_indicator': 2, 'raw_fidelity_name': '2','X': x_high2, 'Y': y_high2},
    ]
    fidelity_num = len(initial_data)

    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel_list = [kernel.SquaredExponentialKernel() for _ in range(fidelity_num)]
    
    myNAR = NAR(fidelity_num = 3,kernel_list= kernel_list, if_nonsubset = False)

    ## if nonsubset is False, max_iter should be 200 ,lr can be 1e-2
    train_NAR(myNAR,fidelity_manager, max_iter = 200, lr_init = 1e-2, debugger = debugger)

    debugger.logger.info('training finished,start predicting')
    with torch.no_grad():
        x_test = fidelity_manager.normalizelayer[myNAR.fidelity_num-1].normalize_x(x_test)
        ypred, ypred_var = myNAR(fidelity_manager, x_test)
        ypred, ypred_var = fidelity_manager.normalizelayer[myNAR.fidelity_num-1].denormalize(ypred, ypred_var)

    debugger.logger.info('prepare to plot')
    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt = 'r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha = 0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    plt.show() 
