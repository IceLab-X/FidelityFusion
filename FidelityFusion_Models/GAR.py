import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from FidelityFusion_Models.two_fidelity_models.hogp_simple import HOGP_simple
import GaussianProcess.kernel as kernel
from GaussianProcess.gp_computation_pack import Tensor_linear
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from Experiments.log_debugger import log_debugger
import matplotlib.pyplot as plt
        
class GAR(torch.nn.Module):
    """
    GeneralizedAutoAR (GAR) model.

    Args:
        fidelity_num (int): The number of fidelity levels.
        kernel_list (list): List of kernel values for each fidelity level.
        data_shape_list (list): List of data shapes for each fidelity level.
        if_nonsubset (bool, optional): Flag indicating if non-subset data is used. Defaults to False.
    """

    def __init__(self, fidelity_num, kernel_list, data_shape_list, if_nonsubset=False):
        super().__init__()
        self.fidelity_num = fidelity_num
        self.hogp_list = []
        for i in range(self.fidelity_num):
            # k = i + 1 if i < len(data_shape_list) - 1 else len(data_shape_list) - 1
            k = i
            self.hogp_list.append(HOGP_simple(kernel=[kernel_list[i] for _ in range(len(data_shape_list[k])+1)], noise_variance=1.0, output_shape=data_shape_list[k], learnable_grid=False, learnable_map=False))
        self.hogp_list = torch.nn.ModuleList(self.hogp_list)

        self.Tensor_linear_list = []
        for i in range(self.fidelity_num - 1):
            self.Tensor_linear_list.append(Tensor_linear(data_shape_list[i], data_shape_list[i + 1]))
        self.Tensor_linear_list = torch.nn.ModuleList(self.Tensor_linear_list)

        self.if_nonsubset = if_nonsubset

    def forward(self, data_manager, x_test, to_fidelity=None):
        """
        Forward pass of the GAR model.

        Args:
            data_manager: The data manager object.
            x_test: The test input data.
            to_fidelity (int, optional): The fidelity level to compute. Defaults to None.

        Returns:
            tuple: A tuple containing the mean and variance of the output.
        """
        if to_fidelity is not None:
            fidelity_level = to_fidelity
        else:
            fidelity_level = self.fidelity_num - 1

        for i_fidelity in range(fidelity_level + 1):
            if i_fidelity == 0:
                x_train, _ = data_manager.get_data(i_fidelity, normal=True)
                mean_low, var_low = self.hogp_list[i_fidelity].forward(x_train, x_test)
                if fidelity_level == 0:
                    mean_high = mean_low
                    var_high = var_low
            else:
                x_train, _ = data_manager.get_data_by_name('res-{}'.format(i_fidelity))
                mean_res, var_res = self.hogp_list[i_fidelity].forward(x_train, x_test)

                mean_high = self.Tensor_linear_list[i_fidelity - 1](mean_low) + mean_res
                var_high = self.Tensor_linear_list[i_fidelity - 1](var_low) + var_res

                mean_low = mean_high
                var_low = var_high

        return mean_high, var_high
        
def train_GAR(GARmodel, data_manager, max_iter=1000, lr_init=1e-1, debugger=None):
    """
    Trains the GARmodel using the specified data_manager.

    Args:
        GARmodel: The GAR model to be trained.
        data_manager: The data manager object that provides the training data.
        max_iter (optional): The maximum number of iterations for training. Default is 1000.
        lr_init (optional): The initial learning rate for the optimizer. Default is 0.1.
        debugger (optional): The debugger object for debugging purposes. Default is None.
    """

    for i_fidelity in range(GARmodel.fidelity_num):
        optimizer = torch.optim.Adam(GARmodel.parameters(), lr=lr_init)
        if i_fidelity == 0:
            x_low, y_low = data_manager.get_data(i_fidelity, normal=True)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = GARmodel.hogp_list[i_fidelity].log_likelihood(x_low, y_low)
                if debugger is not None:
                    debugger.get_status(GARmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')
        else:
            if GARmodel.if_nonsubset:
                with torch.no_grad():
                    subset_x, y_low, y_high = data_manager.get_nonsubset_fill_data(GARmodel, i_fidelity - 1, i_fidelity)
            else:
                _, y_low, subset_x, y_high = data_manager.get_overlap_input_data(i_fidelity - 1, i_fidelity, normal=True)
            for i in range(max_iter):
                optimizer.zero_grad()
                if GARmodel.if_nonsubset:
                    y_residual_mean = y_high[0] - GARmodel.Tensor_linear_list[i_fidelity - 1](y_low[0])  # tensor linear layer
                    y_residual_var = abs(y_high[1] - y_low[1])
                else:
                    y_residual_mean = y_high - GARmodel.Tensor_linear_list[i_fidelity - 1](y_low)
                    y_residual_var = None

                if i == max_iter - 1:
                    data_manager.add_data(raw_fidelity_name='res-{}'.format(i_fidelity), fidelity_index=None, x=subset_x.detach(), y=[y_residual_mean.detach(), y_residual_var.detach()])
                loss = GARmodel.hogp_list[i_fidelity].log_likelihood(subset_x, [y_residual_mean, y_residual_var])
                if debugger is not None:
                    debugger.get_status(GARmodel, optimizer, i, loss)
                loss.backward()
                optimizer.step()
                # print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
                print('fidelity {}, epoch {}/{}, nll: {}'.format(i_fidelity, i+1, max_iter, loss.item()), end='\r')
            print('')


if __name__ == "__main__":
    torch.manual_seed(1)
    debugger=log_debugger("GAR")

    x = np.load('assets/MF_data/Poisson_data/input.npy')
    x = torch.tensor(x, dtype=torch.float32)
    yl=np.load('assets/MF_data/Poisson_data/output_fidelity_0.npy')
    yl = torch.tensor(yl, dtype=torch.float32)
    yh = np.load('assets/MF_data/Poisson_data/output_fidelity_1.npy')
    yh = torch.tensor(yh, dtype=torch.float32)
    yh2 = np.load('assets/MF_data/Poisson_data/output_fidelity_2.npy')
    yh2 = torch.tensor(yh2, dtype = torch.float32)


    x_train = x[:128, :]
    y_l = yl[:128, :]
    y_h = yh[:128, :]
    y_h2 = yh2[:128, :]

    x_test = x[128:, :]
    y_test = yh2[128:, :]

    data_shape = [y_l[0].shape, y_h[0].shape, y_h2[0].shape]

    initial_data = [
        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_train, 'Y': y_l},
        {'fidelity_indicator': 1,'raw_fidelity_name': '1', 'X': x_train, 'Y': y_h},
        {'fidelity_indicator': 2,'raw_fidelity_name': '2', 'X': x_train, 'Y': y_h2}
    ]
    fidelity_num = len(initial_data)
    fidelity_manager = MultiFidelityDataManager(initial_data)

    kernel_list = [kernel.SquaredExponentialKernel() for _ in range(fidelity_num)]
    myGAR = GAR(fidelity_num, kernel_list, data_shape, if_nonsubset = True)

    train_GAR(myGAR, fidelity_manager, max_iter = 100, lr_init = 1e-3, debugger = debugger)

    debugger.logger.info('training finished,start predicting')
    with torch.no_grad():
        x_test = fidelity_manager.normalizelayer[myGAR.fidelity_num-1].normalize_x(x_test)
        ypred, ypred_var = myGAR(fidelity_manager, x_test)
        ypred, ypred_var = fidelity_manager.normalizelayer[myGAR.fidelity_num-1].denormalize(ypred, ypred_var)

    debugger.logger.info('prepare to plot')
    ##plot the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
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

    