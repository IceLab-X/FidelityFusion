import torch
import numpy as np
from hogp_simple import HOGP_simple
import GaussianProcess.kernel as kernel
from GaussianProcess.gp_transform import Normalize0_layer
import matplotlib.pyplot as plt
import tensorly
tensorly.set_backend('pytorch')

def find_subsets_and_indexes(x_low, x_high):
    # find the overlap set
    flat_x_low = x_low.flatten()
    flat_x_high = x_high.flatten()
    subset_indexes_low = torch.nonzero(torch.isin(flat_x_low, flat_x_high), as_tuple=True)[0]
    subset_indexes_high = torch.nonzero(torch.isin(flat_x_high, flat_x_low), as_tuple=True)[0]
    subset_x = flat_x_low[subset_indexes_low].reshape(-1,1)
    return subset_x, subset_indexes_low, subset_indexes_high

class Matric_l2h(torch.nn.Module):
    def __init__(self,l_shape,h_shape,rho=1.,trainable_rho=False):
        super().__init__()
        self.l_shape = l_shape
        self.vectors = []
        ##eye
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
        self.rho = torch.nn.Parameter(torch.tensor(rho, dtype=torch.float32))
        if not trainable_rho:
            self.rho.requires_grad = False
    
    def forward(self, low_fidelity, high_fidelity):
        for i in range(len(self.l_shape)):
            low_fidelity = tensorly.tenalg.mode_dot(low_fidelity, self.vectors[i], i+1)

        res = high_fidelity - low_fidelity*self.rho
        return res
    
    def backward(self, low_fidelity, res):
        for i in range(len(self.l_shape)):
            low_fidelity = tensorly.tenalg.mode_dot(low_fidelity, self.vectors[i], i+1)

        high_fidelity = low_fidelity*self.rho + res
        return high_fidelity
        

class GAR_twofidelity(torch.nn.Module):
    def __init__(self,l_shape,h_shape):
        super().__init__()
        self.low_fidelity_HOGP=HOGP_simple(kernel=kernel.SquaredExponentialKernel(length_scale=1.,signal_variance=1.),
                                            noise_variance=1.0,output_shape=h_shape)
        self.high_fidelity_HOGP=HOGP_simple(kernel=kernel.SquaredExponentialKernel(length_scale=1.,signal_variance=1.),
                                            noise_variance=1.0,output_shape=h_shape)
        self.Matric=Matric_l2h(l_shape,h_shape)

    def forward(self,x_test):
        mean_low,var_low=self.low_fidelity_HOGP.forward(x_test)

        mean_res,var_res=self.high_fidelity_HOGP.forward(x_test)
        mean_high=self.Matric.backward(mean_low,mean_res)
        var_high=self.Matric.backward(var_low,var_res)
        return mean_high,var_high
        

def train_GAR_twofidelity(GARmodel, x_train, y_train,max_iter=1000,lr_init=1e-1):
    x_low = x_train[0]
    y_low = y_train[0]
    x_high = x_train[1]
    y_high = y_train[1]
    
    # train the low fidelity GP
    optimizer_low = torch.optim.Adam(GARmodel.low_fidelity_HOGP.parameters(), lr=lr_init)
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
        res = GARmodel.Matric.forward(y_low,y_high)
        loss = GARmodel.high_fidelity_HOGP.log_likelihood(subset_x,res)
        loss.backward()
        optimizer_high.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))

if __name__ == "__main__":
    torch.manual_seed(1)
    # generate the data
    # x_all = torch.rand(500, 1) * 20

    # xlow_indices = torch.randperm(500)[:300]
    # x_low = x_all[xlow_indices]
    # xhigh_indices = torch.randperm(500)[:300]

    # # x_high = x_all[xhigh_indices]
    # ## full subset
    # x_high = x_all[xlow_indices]

    # x_test = torch.linspace(0, 20, 100).reshape(-1, 1)

    # y_low = torch.sin(x_low) + torch.rand(300, 1) * 0.6 - 0.3
    # y_high = torch.sin(x_high) + torch.rand(300, 1) * 0.2 - 0.1
    # y_test = torch.sin(x_test)

    # x_train = [x_low, x_high]
    # y_train = [y_low, y_high]

    # low_shape=y_low[0].shape
    # high_shape=y_high[0].shape

    x = np.load('assets\\MF_data\\Poisson_data\\input.npy')
    x = torch.tensor(x, dtype=torch.float32)
    yl=np.load('assets\\MF_data\\Poisson_data\\output_fidelity_1.npy')
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

    x_train = x[:128,:]
    y_l = yl[:128,:]
    y_h = yh[:128,:]
    x_train = [x_train,x_train]
    y_train = [y_l, y_h]
    x_test = x[128:,:]
    y_test = y_h[128:,:]
    low_shape=y_l[0].shape
    high_shape=y_h[0].shape


    GAR=GAR_twofidelity(low_shape,high_shape)
    train_GAR_twofidelity(GAR, x_train, y_train, max_iter=100, lr_init=1e-2)
    ypred, ypred_var = GAR(x_test)
    ypred=dnm_yh.inverse(ypred)

    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha=0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    plt.show()

    