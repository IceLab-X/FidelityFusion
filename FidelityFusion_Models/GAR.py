import torch
import numpy as np
from two_fidelity_models.hogp_simple import HOGP_simple
import GaussianProcess.kernel as kernel
from GaussianProcess.gp_transform import Normalize0_layer
from MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt
import tensorly
tensorly.set_backend('pytorch')

class Matric_l2h(torch.nn.Module):
    def __init__(self,l_shape,h_shape):
        super().__init__()
        self.l_shape=l_shape
        self.h_shape=h_shape
        self.vectors = []
        for i in range(len(self.l_shape)):
            if self.l_shape[i] < self.h_shape[i]:
                init_tensor = torch.eye(self.l_shape[i])
                init_tensor = torch.nn.functional.interpolate(init_tensor.reshape(1, 1, *init_tensor.shape), 
                                                              (self.l_shape[i],self.h_shape[i]), mode='bilinear')
                init_tensor = init_tensor.squeeze().T
            elif self.l_shape[i] == self.h_shape[i]:
                init_tensor = torch.eye(self.l_shape[i])
            self.vectors.append(torch.nn.Parameter(init_tensor))
        self.vectors = torch.nn.ParameterList(self.vectors)

    def forward(self,low_fidelity):
        for i in range(len(self.l_shape)):
            low_fidelity = tensorly.tenalg.mode_dot(low_fidelity, self.vectors[i], i+1)
        return low_fidelity
        
class GAR(torch.nn.Module):
    def __init__(self,kernel,l_shape,h_shape,fidelity,rho=1.):
        super().__init__()
        self.l_shape=l_shape
        self.h_shape=h_shape
        self.fidelity=fidelity
        self.hogp_list=[]
        for i in range(self.fidelity):
            self.hogp_list.append(HOGP_simple(kernel=kernel,noise_variance=1.0,output_shape=h_shape[i]))
        self.hogp_list=torch.nn.ModuleList(self.hogp_list)

        ## Matric_l2h
        self.matric_list=[]
        for i in range(self.fidelity-1):
            self.matric_list.append(Matric_l2h(l_shape[i],h_shape[i]))
        self.matric_list=torch.nn.ModuleList(self.matric_list)

        self.rho_list=[]
        for _ in range(self.fidelity-1):
            self.rho_list.append(torch.nn.Parameter(torch.tensor(rho,requires_grad=False)))
        self.rho_list = torch.nn.ParameterList(self.rho_list)

    def forward(self,x_test):

        for f in range(self.fidelity):
            if f == 0:
                mean_low,var_low=self.hogp_list[f].forward(x_test)
                if self.fidelity == 1:
                    mean_high = mean_low
                    var_high = var_low
            else:
                mean_res,var_res=self.hogp_list[f].forward(x_test)
                mean_high = self.matric_list[f-1](mean_low)*self.rho_list[f-1] + mean_res
                var_high = self.matric_list[f-1](var_low)*self.rho_list[f-1]**2 + var_res

                ## for next fidelity
                mean_low = mean_high
                var_low = var_high

        return mean_high,var_high
        
def train_GAR(GARmodel,data_manager,max_iter=1000,lr_init=1e-1):
    
    optimizer = torch.optim.Adam(GARmodel.parameters(), lr=lr_init)
    for i in range(max_iter):
        optimizer.zero_grad()
        loss=0.
        for f in range(GARmodel.fidelity):
            if f == 0:
                x_low,y_low = data_manager.get_data(str(f))
                loss += GARmodel.hogp_list[f].log_likelihood(x_low, y_low)
            else:
                _, y_low, subset_x,y_high = data_manager.get_overlap_input_data(str(f-1),str(f))
                res = y_high - GARmodel.matric_list[f-1](y_low)*GARmodel.rho_list[f-1]
                loss += GARmodel.hogp_list[f].log_likelihood(subset_x, res)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))

if __name__ == "__main__":
    torch.manual_seed(1)

    x = np.load('assets\\MF_data\\Poisson_data\\input.npy')
    x = torch.tensor(x, dtype=torch.float32)
    yl=np.load('assets\\MF_data\\Poisson_data\\output_fidelity_0.npy')
    yl = torch.tensor(yl, dtype=torch.float32)
    yh = np.load('assets\\MF_data\\Poisson_data\\output_fidelity_1.npy')
    yh = torch.tensor(yh, dtype=torch.float32)
    yh2 = np.load('assets\\MF_data\\Poisson_data\\output_fidelity_2.npy')
    yh2 = torch.tensor(yh2, dtype=torch.float32)

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

    x_train = x[:128,:]
    y_l = yl[:128,:]
    y_h = yh[:128,:]
    y_h2 = yh2[:128,:]

    x_test = x[128:,:]
    y_test = yh2[128:,:]

    low_shape = [y_l[0].shape, y_h[0].shape]
    high_shape = [y_h[0].shape, y_h2[0].shape,y_h2[0].shape] ##contain output shape

    initial_data = [
        {'fidelity_index': '0', 'X': x_train, 'Y': y_l},
        {'fidelity_index': '1', 'X': x_train, 'Y': y_h},
        {'fidelity_index': '2', 'X': x_train, 'Y': y_h2}
    ]

    fidelity_manager = MultiFidelityDataManager(initial_data)

    kernel1 = kernel.SquaredExponentialKernel(length_scale=1.,signal_variance=1.)
    myGAR = GAR(kernel1,low_shape,high_shape,fidelity=3)

    train_GAR(myGAR, fidelity_manager, max_iter=100, lr_init=1e-3)

    with torch.no_grad():
        ypred, ypred_var = myGAR(x_test)
        ypred=dnm_yh2.inverse(ypred)

    ##plot the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    yte = dnm_yh.inverse(y_test)
    vmin = torch.min(yte[1])
    vmax = torch.max(yte[1])

    im = axs[0].imshow(yte[1].cpu(), cmap='hot', interpolation='nearest',vmin=vmin,vmax=vmax)
    axs[0].set_title('Groundtruth')

    axs[1].imshow(ypred[1].cpu(), cmap='hot', interpolation='nearest',vmin=vmin,vmax=vmax)
    axs[1].set_title('Predict')

    axs[2].imshow((yte[1].cpu()-ypred[1].cpu()).abs(), cmap='hot', interpolation='nearest',vmin=vmin,vmax=vmax)
    axs[2].set_title('Difference')

    cbar_ax = fig.add_axes([0.95, 0.2, 0.03, 0.6])
    cbar=fig.colorbar(im, cax=cbar_ax)
    plt.show()

    