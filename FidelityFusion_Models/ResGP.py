import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from GaussianProcess.gp_basic import GP_basic as GPR
from GaussianProcess.gp_transform import Normalize0_layer
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt

class ResGP(nn.Module):
    # initialize the model
    def __init__(self, fidelity_num, kernel, if_nonsubset = False):
        """
        Initialize the ResGP class.

        Args:
            fidelity_num (int): The number of fidelity levels.
            kernel: The kernel function for Gaussian Process Regression.
            if_nonsubset (bool): Flag indicating whether to use non-subset data for training selection.

        """
        super().__init__()
        self.fidelity_num = fidelity_num
        # create the model
        self.gpr_list=[]
        for _ in range(self.fidelity_num):
            self.gpr_list.append(GPR(kernel = kernel, noise_variance = 1.0))
        
        self.gpr_list = torch.nn.ModuleList(self.gpr_list)
        self.if_nonsubset = if_nonsubset

    def forward(self, data_manager, x_test, to_fidelity=None):
        """
        Forward pass of the ResGP model.

        Args:
            data_manager: The data manager object.
            x_test: The input test data.
            to_fidelity: The fidelity level to use for prediction.
                         The lowest prediction fidelity is 0

        Returns:
            y_pred_high: The predicted output at the highest fidelity level.
            cov_pred_high: The covariance of the predicted output at the highest fidelity level.
        """
        # predict the model
        if to_fidelity is not None:
            fidelity_level = to_fidelity
        else:
            fidelity_level = self.fidelity_num - 1
        for i_fidelity in range(fidelity_level + 1):
            if i_fidelity == 0:
                x_train, y_train = data_manager.get_data(i_fidelity)
                y_pred_low, cov_pred_low = self.gpr_list[i_fidelity](x_train, y_train, x_test)
                if fidelity_level == 0:
                    y_pred_high = y_pred_low
                    cov_pred_high = cov_pred_low
            else:
                x_train, y_train = data_manager.get_data_by_name('res-{}'.format(i_fidelity))
                y_pred_res, cov_pred_res = self.gpr_list[i_fidelity](x_train, y_train, x_test)
                y_pred_high = y_pred_low + y_pred_res
                cov_pred_high = cov_pred_low + cov_pred_res

                ## for next fidelity
                y_pred_low = y_pred_high
                cov_pred_low = cov_pred_high

        # return the prediction
        return y_pred_high, cov_pred_high
    
def train_ResGP(ResGPmodel, data_manager, max_iter=1000, lr_init=1e-1):
    """
    Trains the ResGP model using the given data manager.

    Args:
        ResGPmodel: The ResGP model to train.
        data_manager: The data manager object that provides the training data.
        max_iter (optional): The maximum number of iterations for training. Default is 1000.
        lr_init (optional): The initial learning rate for the optimizer. Default is 0.1.

    Returns:
        None
    """
    for i_fidelity in range(ResGPmodel.fidelity_num):
        optimizer = torch.optim.Adam(ResGPmodel.parameters(), lr=lr_init)
        if i_fidelity == 0:
            x_low, y_low = data_manager.get_data(i_fidelity)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -ResGPmodel.gpr_list[i_fidelity].log_likelihood(x_low, y_low)
                loss.backward()
                optimizer.step()
                print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
        else:
            if ResGPmodel.if_nonsubset:
                with torch.no_grad():
                    subset_x, y_low, y_high = data_manager.get_nonsubset_fill_data(ResGPmodel, i_fidelity - 1, i_fidelity)
                y_residual_mean = y_high[0] - y_low[0]
                y_residual_var = y_high[1] - y_low[1]
            else:
                _, y_low, subset_x, y_high = data_manager.get_overlap_input_data(i_fidelity - 1, i_fidelity)
                y_residual_mean = y_high - y_low
                y_residual_var = None
            data_manager.add_data(raw_fidelity_name='res-{}'.format(i_fidelity), fidelity_index=None, x=subset_x, y=[y_residual_mean, y_residual_var])
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -ResGPmodel.gpr_list[i_fidelity].log_likelihood(subset_x, [y_residual_mean, y_residual_var])
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
    y_test = torch.sin(x_test)

    # dnm_xlow = Normalize0_layer(x_low)
    # dnm_ylow = Normalize0_layer(y_low)
    # dnm_xhigh1 = Normalize0_layer(x_high1)
    # dnm_yhigh1 = Normalize0_layer(y_high1)
    # dnm_xhigh2 = Normalize0_layer(x_high2)
    # dnm_yhigh2 = Normalize0_layer(y_high2)
    # dnm_xtest = Normalize0_layer(x_test)
    # dnm_ytest = Normalize0_layer(y_test)

    # x_low = dnm_xlow.forward(x_low)
    # y_low = dnm_ylow.forward(y_low)
    # x_high1 = dnm_xhigh1.forward(x_high1)
    # y_high1 = dnm_yhigh1.forward(y_high1)
    # x_high2 = dnm_xhigh2.forward(x_high2)
    # y_high2 = dnm_yhigh2.forward(y_high2)
    # x_test1 = dnm_xtest.forward(x_test)
    # y_test1 = dnm_ytest.forward(y_test)

    initial_data = [
        {'raw_fidelity_name': '0','fidelity_indicator': 0, 'X': x_low, 'Y': y_low},
        {'raw_fidelity_name': '1','fidelity_indicator': 1, 'X': x_high1, 'Y': y_high1},
        {'raw_fidelity_name': '2','fidelity_indicator': 2, 'X': x_high2, 'Y': y_high2},
    ]

    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
    myResGP = ResGP(fidelity_num = 3, kernel = kernel1, if_nonsubset = True)

    ## if nonsubset is False, max_iter should be 100 ,lr can be 1e-2
    train_ResGP(myResGP, fidelity_manager, max_iter=100, lr_init=1e-2)

    with torch.no_grad():
        ypred, ypred_var = myResGP(fidelity_manager, x_test)
        # ypred = dnm_ytest.inverse(ypred)
        # ypred_var = ypred_var * dnm_ytest.std**2

    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt = 'r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha = 0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    plt.show() 
