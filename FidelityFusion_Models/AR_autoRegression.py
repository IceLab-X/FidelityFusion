import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from GaussianProcess.gp_basic import GP_basic as GPR
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt

class AR(nn.Module):
    # initialize the model
    def __init__(self, fidelity_num, kernel, rho_init=1.0, if_nonsubset=False):
        """
        Initialize the AR_autoRegression class.

        Args:
            fidelity_num (int): The number of fidelity levels.
            kernel: The kernel function for Gaussian Process Regression.
            rho_init (float): The initial value for the rho parameter.
            if_nonsubset (bool): Flag indicating whether to use non-subset data for training selection.

        """
        super().__init__()
        self.fidelity_num = fidelity_num
        # create the model
        self.gpr_list = []
        for _ in range(self.fidelity_num):
            self.gpr_list.append(GPR(kernel=kernel, noise_variance=1.0))
        
        self.gpr_list = torch.nn.ModuleList(self.gpr_list)

        self.rho_list = []
        for _ in range(self.fidelity_num-1):
            self.rho_list.append(torch.nn.Parameter(torch.tensor(rho_init)))

        self.rho_list = torch.nn.ParameterList(self.rho_list)
        self.if_nonsubset = if_nonsubset

    def forward(self, data_manager, x_test, to_fidelity=None):
        """
        Predicts the posterior given a new input `x_test`.

        Args:
            data_manager: The data manager object.
            x_test: The input data for prediction.
            to_fidelity: The fidelity level to use for prediction. If None, the default fidelity level will be used.
                         The lowest prediction fidelity is 0

        Returns:
            Tuple: A tuple containing the predicted output `y_pred_high` and the covariance `cov_pred_high`.
        """
        # predict the posterior given a new input x_test
        # if to_fidelity is not None and to_fidelity >= 1:
        if to_fidelity is not None :
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
                # get the residual data from data_manager using key word 'res-{}', which is created during model training
                x_train, y_train = data_manager.get_data_by_name('res-{}'.format(i_fidelity))
                y_pred_res, cov_pred_res = self.gpr_list[i_fidelity](x_train, y_train, x_test)
                y_pred_high = y_pred_low + self.rho_list[i_fidelity - 1] * y_pred_res
                cov_pred_high = cov_pred_low + (self.rho_list[i_fidelity - 1] ** 2) * cov_pred_res

                ## update: high fidelity not become the low fidelity for the next iteration
                y_pred_low = y_pred_high
                cov_pred_low = cov_pred_high

        # return the prediction
        return y_pred_high, cov_pred_high
#train_gp
    
def train_AR(ARmodel, data_manager, max_iter = 1000, lr_init = 1e-1):
    """
    Trains an auto-regression model using the specified ARmodel and data_manager.

    Parameters:
    - ARmodel: The auto-regression model to be trained.
    - data_manager: The data manager object that provides the training data.
    - max_iter: The maximum number of iterations for training. Default is 1000.
    - lr_init: The initial learning rate for the optimizer. Default is 0.1.

    Returns:
    None
    """
    
    for i_fidelity in range(ARmodel.fidelity_num):
        optimizer = torch.optim.Adam(ARmodel.parameters(), lr = lr_init)
        if i_fidelity == 0:
            x_low,y_low = data_manager.get_data(i_fidelity)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -ARmodel.gpr_list[i_fidelity].log_likelihood(x_low, y_low)
                loss.backward()
                optimizer.step()
                print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
        else:
            if ARmodel.if_nonsubset:
                with torch.no_grad():
                    subset_x,y_low,y_high = data_manager.get_nonsubset_fill_data(ARmodel, i_fidelity - 1, i_fidelity)
            else:
                _, y_low, subset_x,y_high = data_manager.get_overlap_input_data(i_fidelity - 1, i_fidelity)
            for i in range(max_iter):
                optimizer.zero_grad()
                if ARmodel.if_nonsubset:
                    y_residual_mean = y_high[0] - ARmodel.rho_list[i_fidelity - 1] * y_low[0]
                    y_residual_var = y_high[1] - ARmodel.rho_list[i_fidelity - 1] * y_low[1]
                else:
                    y_residual_mean = y_high - ARmodel.rho_list[i_fidelity - 1] * y_low
                    y_residual_var = None
                if i == max_iter-1:
                    data_manager.add_data(raw_fidelity_name = 'res-{}'.format(i_fidelity), fidelity_index = None, x = subset_x, y = [y_residual_mean, y_residual_var])
                loss = -ARmodel.gpr_list[i_fidelity].log_likelihood(subset_x, [y_residual_mean, y_residual_var])
                loss.backward()
                optimizer.step()
                print('fidelity:', i_fidelity, 'iter', i,'rho',ARmodel.rho_list[i_fidelity - 1].item(), 'nll:{:.5f}'.format(loss.item()))
            
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

    initial_data = [
        {'raw_fidelity_name': '0','fidelity_indicator': 0, 'X': x_low, 'Y': y_low},
        {'raw_fidelity_name': '1','fidelity_indicator': 1, 'X': x_high1, 'Y': y_high1},
        {'raw_fidelity_name': '2','fidelity_indicator': 2, 'X': x_high2, 'Y': y_high2},
    ]

    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
    myAR = AR(fidelity_num=3,kernel=kernel1,rho_init=1.0,if_nonsubset=True)

    ## if nonsubset is False, max_iter should be 100 ,lr can be 1e-2
    train_AR(myAR, fidelity_manager, max_iter=500, lr_init=1e-3)

    with torch.no_grad():
        ypred, ypred_var = myAR(fidelity_manager,x_test)

    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha = 0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    plt.show() 
