import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from GaussianProcess.gp_basic import GP_basic as GPR
import GaussianProcess.kernel as kernel
from GaussianProcess.gp_transform import Normalize0_layer
from GaussianProcess.gp_computation_pack import Tensor_linear
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt
        
        
class CIGAR(torch.nn.Module):
    def __init__(self, fidelity_num, kernel, data_shape_list, if_nonsubset=False):
        """
        Initialize the CIGAR class.

        Args:
            fidelity_num (int): The number of fidelity levels.
            kernel: The kernel function for Gaussian Process Regression.
            data_shape_list (list): A list of data shapes for each fidelity level.
            if_nonsubset (bool): Flag indicating whether to use non-subset data for training selection.
        
        """
        super().__init__()
        self.fidelity_num = fidelity_num
        self.gpr_list = []
        for i in range(self.fidelity_num):
            self.gpr_list.append(GPR(kernel=kernel, noise_variance=1.0))
        self.gpr_list = torch.nn.ModuleList(self.gpr_list)

        self.Tensor_linear_list = []
        for i in range(self.fidelity_num - 1):
            self.Tensor_linear_list.append(Tensor_linear(data_shape_list[i], data_shape_list[i + 1]))
        self.Tensor_linear_list = torch.nn.ModuleList(self.Tensor_linear_list)

        self.if_nonsubset = if_nonsubset

    def forward(self, data_manager, x_test, to_fidelity=None):
        """
        Forward pass of the CIGAR model.

        Args:
            data_manager: The data manager object.
            x_test: The input test data.
            to_fidelity: The fidelity level to evaluate. If None, the default fidelity level will be used.
                         The lowest prediction fidelity is 0
        Returns:
            mean_high: The mean output at the highest fidelity level.
            var_high: The variance output at the highest fidelity level.
        """
        if to_fidelity is not None:
            fidelity_level = to_fidelity
        else:
            fidelity_level = self.fidelity_num -1

        for i_fidelity in range(fidelity_level + 1):
            if i_fidelity == 0:
                x_train, y_train = data_manager.get_data(i_fidelity)
                mean_low, var_low = self.gpr_list[i_fidelity].forward(x_train, y_train, x_test)
                if len(mean_low.shape) == 0:
                    mean_low = mean_low.reshape(1).unsqueeze(dim = 0)
                if len(mean_low.shape) == 1:
                    mean_low = mean_low.unsqueeze(dim = 1)
                # if mean_low.shape == var_low.diag().shape:
                #     var_low = var_low.diag()
                # else:
                #     var_low = var_low.diag().unsqueeze(dim = 1).expand_as(mean_low)
                var_low = var_low.diag().unsqueeze(dim = 1).expand_as(mean_low)
                if fidelity_level == 0:
                    mean_high = mean_low
                    var_high = var_low
            else:
                x_train, y_train = data_manager.get_data_by_name('res-{}'.format(i_fidelity))
                mean_res, var_res = self.gpr_list[i_fidelity].forward(x_train, y_train, x_test)
                if len(mean_res.shape) == 1:
                    mean_res = mean_res.unsqueeze(dim = 1)
                # if mean_res.shape == var_res.diag().shape:
                #     var_res = var_low.diag()
                # else:
                #     var_res = var_low.diag().unsqueeze(dim = 1).expand_as(mean_res)
                var_res = var_low.diag().unsqueeze(dim = 1).expand_as(mean_res)
                mean_high = self.Tensor_linear_list[i_fidelity - 1](mean_low) + mean_res
                var_high = self.Tensor_linear_list[i_fidelity - 1](var_low) + var_res

                ## for next fidelity
                mean_low = mean_high
                var_low = var_high

        return mean_high, var_high
        
def train_CIGAR(CIGARmodel, data_manager, max_iter=1000, lr_init=1e-1):
    """
    Trains the CIGAR model using the specified data manager.

    Args:
        CIGARmodel: The CIGAR model to train.
        data_manager: The data manager object that provides the training data.
        max_iter: The maximum number of iterations for training (default: 1000).
        lr_init: The initial learning rate for the optimizer (default: 0.1).

    Returns:
        None
    """
    for i_fidelity in range(CIGARmodel.fidelity_num):
        optimizer = torch.optim.Adam(CIGARmodel.parameters(), lr=lr_init)
        if i_fidelity == 0:
            x_low, y_low = data_manager.get_data(i_fidelity)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = -CIGARmodel.gpr_list[i_fidelity].log_likelihood(x_low, y_low)
                loss.backward()
                optimizer.step()
                print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
        else:
            if CIGARmodel.if_nonsubset:
                with torch.no_grad():
                    subset_x, y_low, y_high = data_manager.get_nonsubset_fill_data(CIGARmodel, i_fidelity - 1, i_fidelity)
            else:
                _, y_low, subset_x, y_high = data_manager.get_overlap_input_data(i_fidelity - 1, i_fidelity)
            for i in range(max_iter):
                optimizer.zero_grad()
                if CIGARmodel.if_nonsubset:
                    y_residual_mean = y_high[0] - CIGARmodel.Tensor_linear_list[i_fidelity - 1](y_low[0])  # tensor linear layer
                    y_residual_var = y_high[1] - y_low[1]
                else:
                    y_residual_mean = y_high - CIGARmodel.Tensor_linear_list[i_fidelity - 1](y_low)
                    y_residual_var = None

                if i == max_iter - 1:
                    data_manager.add_data(raw_fidelity_name='res-{}'.format(i_fidelity), fidelity_index=None, x=subset_x, y=[y_residual_mean, y_residual_var])
                loss = -CIGARmodel.gpr_list[i_fidelity].log_likelihood(subset_x, [y_residual_mean, y_residual_var])
                loss.backward()
                optimizer.step()
                print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))

if __name__ == "__main__":
    torch.manual_seed(1)

    x = np.load('assets/MF_data/Poisson_data/input.npy')
    x = torch.tensor(x, dtype=torch.float32)
    yl=np.load('assets/MF_data/Poisson_data/output_fidelity_0.npy')
    yl = torch.tensor(yl, dtype=torch.float32)
    yh = np.load('assets/MF_data/Poisson_data/output_fidelity_1.npy')
    yh = torch.tensor(yh, dtype=torch.float32)
    yh2 = np.load('assets/MF_data/Poisson_data/output_fidelity_2.npy')
    yh2 = torch.tensor(yh2, dtype = torch.float32)

    ## Standardization layer, currently using full dimensional standardization
    dnm_x = Normalize0_layer(x)
    dnm_yl = Normalize0_layer(yl)
    dnm_yh = Normalize0_layer(yh)
    dnm_yh2 = Normalize0_layer(yh2)

    #normalize the data
    x=dnm_x.forward(x)
    y_l=dnm_yl.forward(yl)
    y_h=dnm_yh.forward(yh)
    y_h2=dnm_yh2.forward(yh2)

    x_train = x[:128, :]
    y_l = yl[:128, :]
    y_h = yh[:128, :]
    y_h2 = yh2[:128, :]
    src_y_shape = y_h2.shape[1:]

    x_test = x[128:, :]
    y_test = yh2[128:, :]

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

    fidelity_manager = MultiFidelityDataManager(initial_data)

    kernel1 = kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.)
    myCIGAR = CIGAR(3, kernel1, data_shape, if_nonsubset = True)

    train_CIGAR(myCIGAR, fidelity_manager, max_iter = 300, lr_init = 1e-3)

    with torch.no_grad():
        ypred, ypred_var = myCIGAR(fidelity_manager, x_test)
        ypred = dnm_yh2.inverse(ypred)
    
    ypred = ypred.reshape(-1, * src_y_shape)

    ##plot the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    yte = dnm_yh.inverse(y_test)
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