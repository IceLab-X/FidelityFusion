import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from FidelityFusion_Models.two_fidelity_models.hogp_simple import HOGP_simple
import GaussianProcess.kernel as kernel
from GaussianProcess.gp_transform import Normalize0_layer
from GaussianProcess.gp_computation_pack import Tensor_linear
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt
        
class GAR(torch.nn.Module):
    def __init__(self, fidelity_num, kernel, l_shape, h_shape, nonsubset = False):
        super().__init__()
        self.l_shape = l_shape
        self.h_shape = h_shape
        self.fidelity_num = fidelity_num
        self.hogp_list = []
        for i in range(self.fidelity_num):
            self.hogp_list.append(HOGP_simple(kernel = kernel, noise_variance = 1.0, output_shape = h_shape[i]))
        self.hogp_list=torch.nn.ModuleList(self.hogp_list)

        self.Tensor_linear_list = []
        for i in range(self.fidelity_num - 1):
            self.Tensor_linear_list.append(Tensor_linear(l_shape[i], h_shape[i]))
        self.Tensor_linear_list=torch.nn.ModuleList(self.Tensor_linear_list)

        self.nonsubset = nonsubset

    def forward(self, data_manager, x_test, to_fidelity = None):
        
        if to_fidelity is not None and to_fidelity >= 1:
            fidelity_num = to_fidelity
        else:
            fidelity_num = self.fidelity_num

        for i_fidelity in range(fidelity_num):
            if i_fidelity == 0:
                x_train, _ = data_manager.get_data(i_fidelity)
                mean_low, var_low = self.hogp_list[i_fidelity].forward(x_train, x_test)
                if fidelity_num == 1:
                    mean_high = mean_low
                    var_high = var_low
                    # var_high = torch.diag_embed(torch.flatten(var_high))
            else:
                x_train, _ = data_manager.get_data_by_name('res-{}'.format(i_fidelity))
                mean_res, var_res = self.hogp_list[i_fidelity].forward(x_train, x_test)

                mean_high = self.Tensor_linear_list[i_fidelity - 1](mean_low) + mean_res
                var_high = self.Tensor_linear_list[i_fidelity - 1](var_low) + var_res
                # var_high = torch.diag_embed(torch.flatten(var_high))

                ## for next fidelity
                mean_low = mean_high
                var_low = var_high

        return mean_high, var_high
        
def train_GAR(GARmodel, data_manager, max_iter = 1000, lr_init =  1e-1):
    
    for i_fidelity in range(GARmodel.fidelity_num):
        optimizer = torch.optim.Adam(GARmodel.parameters(), lr = lr_init)
        if i_fidelity == 0:
            x_low,y_low = data_manager.get_data(i_fidelity)
            for i in range(max_iter):
                optimizer.zero_grad()
                loss = GARmodel.hogp_list[i_fidelity].log_likelihood(x_low, y_low)
                loss.backward()
                optimizer.step()
                print('fidelity:', i_fidelity, 'iter', i, 'nll:{:.5f}'.format(loss.item()))
        else:
            if GARmodel.nonsubset:
                with torch.no_grad():
                    subset_x, y_low, y_high = data_manager.get_nonsubset_fill_data(GARmodel, i_fidelity - 1, i_fidelity)
            else:
                _, y_low, subset_x, y_high = data_manager.get_overlap_input_data(i_fidelity - 1, i_fidelity)
            for i in range(max_iter):
                optimizer.zero_grad()
                if GARmodel.nonsubset:
                    y_residual_mean = y_high[0] - GARmodel.Tensor_linear_list[i_fidelity - 1](y_low[0])  #tensor linear layer
                    y_residual_var = y_high[1] - y_low[1]
                else:
                    y_residual_mean = y_high - GARmodel.Tensor_linear_list[i_fidelity - 1](y_low)
                    y_residual_var = None

                if i == max_iter-1:
                    data_manager.add_data(raw_fidelity_name='res-{}'.format(i_fidelity), fidelity_index = None, x = subset_x, y = [y_residual_mean, y_residual_var])
                loss = GARmodel.hogp_list[i_fidelity].log_likelihood(subset_x, [y_residual_mean,y_residual_var])
                loss.backward()
                optimizer.step()
                print('fidelity:', i_fidelity, 'iter', i,'nll:{:.5f}'.format(loss.item()))

if __name__ == "__main__":
    torch.manual_seed(1)

    # x = np.load('assets\\MF_data\\Poisson_data\\input.npy')
    # x = torch.tensor(x, dtype=torch.float32)
    # yl=np.load('assets\\MF_data\\Poisson_data\\output_fidelity_0.npy')
    # yl = torch.tensor(yl, dtype=torch.float32)
    # yh = np.load('assets\\MF_data\\Poisson_data\\output_fidelity_1.npy')
    # yh = torch.tensor(yh, dtype=torch.float32)
    # yh2 = np.load('assets\\MF_data\\Poisson_data\\output_fidelity_2.npy')
    # yh2 = torch.tensor(yh2, dtype = torch.float32)

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
    y_h2=dnm_yh.forward(yh2)

    x_train = x[:128, :]
    y_l = yl[:128, :]
    y_h = yh[:128, :]
    y_h2 = yh2[:128, :]

    x_test = x[128:, :]
    y_test = yh2[128:, :]

    low_shape = [y_l[0].shape, y_h[0].shape]
    high_shape = [y_h[0].shape, y_h2[0].shape, y_h2[0].shape] ##contain output shape

    initial_data = [
        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_train, 'Y': y_l},
        {'fidelity_indicator': 1,'raw_fidelity_name': '1', 'X': x_train, 'Y': y_h},
        {'fidelity_indicator': 2,'raw_fidelity_name': '2', 'X': x_train, 'Y': y_h2}
    ]

    fidelity_manager = MultiFidelityDataManager(initial_data)

    kernel1 = kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.)
    myGAR = GAR(3 ,kernel1, low_shape, high_shape, nonsubset = True)

    train_GAR(myGAR, fidelity_manager, max_iter = 100, lr_init = 1e-3)

    with torch.no_grad():
        ypred, ypred_var = myGAR(fidelity_manager, x_test)
        ypred = dnm_yh2.inverse(ypred)

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

    