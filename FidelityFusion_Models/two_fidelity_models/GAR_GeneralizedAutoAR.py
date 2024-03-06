import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import torch
import numpy as np
from hogp_simple import HOGP_simple
import GaussianProcess.kernel as kernel
from GaussianProcess.gp_transform import Normalize0_layer
import matplotlib.pyplot as plt
import tensorly
tensorly.set_backend('pytorch')

def find_subsets_and_indexes(x_low, x_high):
    
    subset_indexes_low = []
    subset_indexes_high = []

    for i, row_x1 in enumerate(x_low):
        for j, row_x2 in enumerate(x_high):
            if torch.equal(row_x1, row_x2):
                subset_indexes_low.append(i)
                subset_indexes_high.append(j)
    subset_x = x_low[subset_indexes_low]
    return subset_x, subset_indexes_low, subset_indexes_high

class GAR_twofidelity(torch.nn.Module):
    # def __init__(self,l_shape,h_shape,rho=1.):
    def __init__(self, l_shape, h_shape):
        super().__init__()
        self.l_shape = l_shape
        self.h_shape = h_shape

        self.low_fidelity_HOGP = HOGP_simple(kernel = kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.),
                                            noise_variance = 1.0, output_shape = h_shape)
        self.high_fidelity_HOGP = HOGP_simple(kernel = kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.),
                                            noise_variance = 1.0, output_shape = h_shape)
        ## Matric_l2h
        self.vectors = []
        for i in range(len(l_shape)):
            if l_shape[i] < h_shape[i]:
                init_tensor = torch.eye(l_shape[i])
                init_tensor = torch.nn.functional.interpolate(init_tensor.reshape(1, 1, *init_tensor.shape), 
                                                              (l_shape[i],h_shape[i]), mode='bilinear')
                init_tensor = init_tensor.squeeze().T
            elif l_shape[i] == h_shape[i]:
                init_tensor = torch.eye(l_shape[i])
            self.vectors.append(torch.nn.Parameter(init_tensor))
        self.vectors = torch.nn.ParameterList(self.vectors)
        # self.rho = torch.nn.Parameter(torch.tensor(rho, dtype=torch.float32))
        # self.rho.requires_grad = False


    def forward(self,x_train,x_test):
        x_low = x_train[0]
        x_high = x_train[1]
        mean_low,var_low = self.low_fidelity_HOGP.forward(x_low, x_test)
        subset_x, _, _ = find_subsets_and_indexes(x_low, x_high)

        mean_res,var_res=self.high_fidelity_HOGP.forward(subset_x, x_test)

        for i in range(len(self.l_shape)):
            mean_low = tensorly.tenalg.mode_dot(mean_low, self.vectors[i], i+1)
        # mean_high = mean_low*self.rho + mean_res
        mean_high = mean_low + mean_res

        for i in range(len(self.l_shape)):
            var_low = tensorly.tenalg.mode_dot(var_low, self.vectors[i], i+1)
        # var_high = var_low*self.rho + var_res
        var_high = var_low + var_res

        return mean_high, var_high
        

def train_GAR_twofidelity(GARmodel, x_train, y_train, max_iter = 1000, lr_init = 1e-1):
    x_low = x_train[0]
    y_low = y_train[0]
    x_high = x_train[1]
    y_high = y_train[1]
    
    # train the low fidelity GP
    optimizer_low = torch.optim.Adam(GARmodel.low_fidelity_HOGP.parameters(), lr = lr_init)
    for i in range(max_iter):
        optimizer_low.zero_grad()
        loss = GARmodel.low_fidelity_HOGP.log_likelihood(x_low, y_low)
        loss.backward()
        optimizer_low.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))
    
    # get the high fidelity part that is subset of the low fidelity part
    subset_x, subset_indexes_low, subset_indexes_high = find_subsets_and_indexes(x_low, x_high)
    y_low = y_low[subset_indexes_low]
    y_high = y_high[subset_indexes_high]
    # train the high_fidelity_GP
    optimizer_high = torch.optim.Adam(GARmodel.parameters(), lr=lr_init)
    for i in range(max_iter):
        optimizer_high.zero_grad()

        with torch.no_grad():
            for j in range(len(GARmodel.l_shape)):
                y_low = tensorly.tenalg.mode_dot(y_low, GARmodel.vectors[j], j+1)
        # res = y_high - y_low * GARmodel.rho
        res = y_high - y_low

        loss = GARmodel.high_fidelity_HOGP.log_likelihood(subset_x, res)
        loss.backward()
        optimizer_high.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))


if __name__ == "__main__":
    torch.manual_seed(1)

    x = np.load('assets\\MF_data\\Poisson_data\\input.npy')
    x = torch.tensor(x, dtype=torch.float32)
    yl = np.load('assets\\MF_data\\Poisson_data\\output_fidelity_1.npy')
    yl = torch.tensor(yl, dtype=torch.float32)
    yh = np.load('assets\\MF_data\\Poisson_data\\output_fidelity_2.npy')
    yh = torch.tensor(yh, dtype=torch.float32)

    ## Standardization layer, currently using full dimensional standardization
    dnm_x = Normalize0_layer(x)
    dnm_yl = Normalize0_layer(yl)
    dnm_yh = Normalize0_layer(yh)

    #normalize the data
    x=dnm_x.forward(x)
    y_l=dnm_yl.forward(yl)
    y_h=dnm_yh.forward(yh)

    x_train = x[:128, :]
    y_l = yl[:128, :]
    y_h = yh[:128, :]

    x_train = [x_train, x_train]
    y_train = [y_l, y_h]

    x_test = x[128:, :]
    y_test = yh[128:, :]
    low_shape = y_l[0].shape
    high_shape = y_h[0].shape

    GAR = GAR_twofidelity(low_shape, high_shape)
    train_GAR_twofidelity(GAR, x_train, y_train, max_iter = 100, lr_init = 1e-3)

    with torch.no_grad():
        ypred, ypred_var = GAR(x_train, x_test)
        ypred=dnm_yh.inverse(ypred)

    ##plot the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    yte = dnm_yh.inverse(y_test)
    vmin = torch.min(yte[1])
    vmax = torch.max(yte[1])

    im = axs[0].imshow(yte[1].cpu(), cmap='hot', interpolation='nearest', vmin = vmin, vmax = vmax)
    axs[0].set_title('Groundtruth')

    axs[1].imshow(ypred[1].cpu(), cmap='hot', interpolation='nearest', vmin = vmin, vmax = vmax)
    axs[1].set_title('Predict')

    axs[2].imshow((yte[1].cpu()-ypred[1].cpu()).abs(), cmap='hot', interpolation='nearest',vmin = vmin, vmax = vmax)
    axs[2].set_title('Difference')

    cbar_ax = fig.add_axes([0.95, 0.2, 0.03, 0.6])
    cbar=fig.colorbar(im, cax = cbar_ax)
    plt.show()

    