import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from GaussianProcess.cigp_v10 import cigp as GPR
import GaussianProcess.kernel as kernel
from GaussianProcess.gp_computation_pack import Tensor_linear
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from Experiments.log_debugger import log_debugger
import matplotlib.pyplot as plt
        
        
class CIGAR(torch.nn.Module):
    """
    CIGAR (ConditionalIndependentGAR) module.

    Args:
        fidelity_num (int): Number of fidelity levels.
        kernel_list (list): List of kernels for each fidelity level.
        data_shape_list (list): List of data shapes for each fidelity level.
        if_nonsubset (bool, optional): Flag indicating if non-subset data is used. Defaults to False.
    """

    def __init__(self, fidelity_num, kernel_list, data_shape_list, if_nonsubset=False):
        super().__init__()
        self.fidelity_num = fidelity_num
        self.gpr_list = []
        for i in range(self.fidelity_num):
            self.gpr_list.append(GPR(kernel=kernel_list[i], log_beta=1.0))
        self.gpr_list = torch.nn.ModuleList(self.gpr_list)

        self.Tensor_linear_list = []
        for i in range(self.fidelity_num - 1):
            self.Tensor_linear_list.append(Tensor_linear(data_shape_list[i], data_shape_list[i + 1]))
        self.Tensor_linear_list = torch.nn.ModuleList(self.Tensor_linear_list)

        self.if_nonsubset = if_nonsubset

    def forward(self, data_manager, x_test, to_fidelity=None):
        """
        Forward pass of the CIGAR module.

        Args:
            data_manager: Data manager object.
            x_test: Test input data.
            to_fidelity (int, optional): Fidelity level to evaluate. Defaults to None.

        Returns:
            tuple: Tuple containing the mean and variance of the output.
        """
        if to_fidelity is not None:
            fidelity_level = to_fidelity
        else:
            fidelity_level = self.fidelity_num - 1

        for i_fidelity in range(fidelity_level + 1):
            if i_fidelity == 0:
                x_train, y_train = data_manager.get_data(i_fidelity, normal=True)
                mean_low, var_low = self.gpr_list[i_fidelity].forward(x_train, y_train, x_test)
                if len(mean_low.shape) == 0:
                    mean_low = mean_low.reshape(1).unsqueeze(dim=0)
                if len(mean_low.shape) == 1:
                    mean_low = mean_low.unsqueeze(dim=1)
                var_low = var_low.diag().unsqueeze(dim=1).expand_as(mean_low)
                if fidelity_level == 0:
                    mean_high = mean_low
                    var_high = var_low
            else:
                x_train, y_train = data_manager.get_data_by_name('res-{}'.format(i_fidelity))
                mean_res, var_res = self.gpr_list[i_fidelity].forward(x_train, y_train, x_test)
                if len(mean_res.shape) == 1:
                    mean_res = mean_res.unsqueeze(dim=1)
                var_res = var_low.diag().unsqueeze(dim=1).expand_as(mean_res)
                mean_high = self.Tensor_linear_list[i_fidelity - 1](mean_low) + mean_res
                var_high = self.Tensor_linear_list[i_fidelity - 1](var_low) + var_res

                ## for next fidelity
                mean_low = mean_high
                var_low = var_high

        return mean_high, var_high
        
def train_CIGAR(CIGARmodel, data_manager, max_iter=1000, lr_init=1e-1, debugger=None):
    """
    Trains the CIGAR model using the specified data manager.

    Args:
        CIGARmodel: The CIGAR model to train.
        data_manager: The data manager object that provides the training data.
        max_iter: The maximum number of iterations for training (default: 1000).
        lr_init: The initial learning rate for the optimizer (default: 0.1).
        debugger: Optional debugger object for monitoring the training process (default: None).
    """

    for i_fidelity in range(CIGARmodel.fidelity_num):
        optimizer = torch.optim.Adam(CIGARmodel.parameters(), lr=lr_init)
        if i_fidelity == 0:
            x_low, y_low = data_manager.get_data(i_fidelity, normal=True)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -CIGARmodel.gpr_list[i_fidelity].negative_log_likelihood(x_low, y_low)
                if debugger is not None:
                    debugger.get_status(CIGARmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')
        else:
            if CIGARmodel.if_nonsubset:
                with torch.no_grad():
                    subset_x, y_low, y_high = data_manager.get_nonsubset_fill_data(CIGARmodel, i_fidelity - 1, i_fidelity)
            else:
                _, y_low, subset_x, y_high = data_manager.get_overlap_input_data(i_fidelity - 1, i_fidelity, normal=True)
            for i in range(max_iter):
                optimizer.zero_grad()
                if CIGARmodel.if_nonsubset:
                    y_residual_mean = y_high[0] - CIGARmodel.Tensor_linear_list[i_fidelity - 1](y_low[0])  # tensor linear layer
                    y_residual_var = abs(y_high[1] - y_low[1])
                else:
                    y_residual_mean = y_high - CIGARmodel.Tensor_linear_list[i_fidelity - 1](y_low)
                    y_residual_var = None

                if i == max_iter - 1:
                    if y_residual_var is not None:
                        y_residual_var = y_residual_var.detach()
                    data_manager.add_data(raw_fidelity_name='res-{}'.format(i_fidelity), fidelity_index=None, x=subset_x.detach(), y=[y_residual_mean.detach(), y_residual_var])
                loss = -CIGARmodel.gpr_list[i_fidelity].negative_log_likelihood(subset_x, [y_residual_mean, y_residual_var])
                if debugger is not None:
                    debugger.get_status(CIGARmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')

if __name__ == "__main__":
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debugger=log_debugger("CIGAR")

    x = np.load('assets/MF_data/Poisson_data/input.npy')
    x = torch.tensor(x, dtype=torch.float32)
    yl=np.load('assets/MF_data/Poisson_data/output_fidelity_0.npy')
    yl = torch.tensor(yl, dtype=torch.float32)
    yh = np.load('assets/MF_data/Poisson_data/output_fidelity_1.npy')
    yh = torch.tensor(yh, dtype=torch.float32)
    yh2 = np.load('assets/MF_data/Poisson_data/output_fidelity_2.npy')
    yh2 = torch.tensor(yh2, dtype = torch.float32)

    x_train = x[:128, :].to(device)
    y_l = yl[:128, :].to(device)
    y_h = yh[:128, :].to(device)
    y_h2 = yh2[:128, :].to(device)
    src_y_shape = y_h2.shape[1:]

    x_test = x[128:, :].to(device)
    y_test = yh2[128:, :].to(device)

    x_train = x_train.reshape(x_train.shape[0],-1)
    y_l = y_l.reshape(y_l.shape[0],-1)
    y_h = y_h.reshape(y_h.shape[0],-1)
    y_h2 = y_h2.reshape(y_h2.shape[0],-1)
    

    data_shape = [y_l[0].shape, y_h[0].shape, y_h2[0].shape]

    initial_data = [
        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_train, 'Y': y_l},
        {'fidelity_indicator': 1,'raw_fidelity_name': '1', 'X': x_train, 'Y': y_h},
        {'fidelity_indicator': 2,'raw_fidelity_name': '2', 'X': x_train, 'Y': y_h2}
    ]
    fidelity_num = len(initial_data)

    fidelity_manager = MultiFidelityDataManager(initial_data)

    kernel_list = [kernel.SquaredExponentialKernel() for _ in range(fidelity_num)]
    myCIGAR = CIGAR(fidelity_num, kernel_list, data_shape, if_nonsubset = True).to(device)

    train_CIGAR(myCIGAR, fidelity_manager, max_iter = 100, lr_init = 1e-3, debugger = debugger)

    debugger.logger.info('training finished,start predicting')
    with torch.no_grad():
        x_test = fidelity_manager.normalizelayer[myCIGAR.fidelity_num-1].normalize_x(x_test)
        ypred, ypred_var = myCIGAR(fidelity_manager, x_test)
        ypred, ypred_var = fidelity_manager.normalizelayer[myCIGAR.fidelity_num-1].denormalize(ypred, ypred_var)
    
    ypred = ypred.reshape(-1, * src_y_shape)

    debugger.logger.info('prepare to plot')
    ##plot the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # yte = dnm_yh.inverse(y_test)
    yte = y_test
    vmin = torch.min(yte[1])
    vmax = torch.max(yte[1])

    im = axs[0].imshow(yte[1].cpu(), cmap='hot', interpolation='nearest', vmin = vmin, vmax = vmax)
    axs[0].set_title('Groundtruth')

    axs[1].imshow(ypred[1].cpu(), cmap='hot', interpolation ='nearest', vmin = vmin, vmax = vmax)
    axs[1].set_title('Predict')

    axs[2].imshow((yte[1].cpu()-ypred[1].cpu()).abs(), cmap = 'hot', interpolation='nearest', vmin = vmin, vmax = vmax)
    axs[2].set_title('Difference')

    cbar_ax = fig.add_axes([0.95, 0.2, 0.03, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    plt.show()