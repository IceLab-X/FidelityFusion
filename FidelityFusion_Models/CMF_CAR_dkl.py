import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from GaussianProcess.cigp_v10 import cigp as GPR
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
import matplotlib.pyplot as plt


class fidelity_kernel_MC(nn.Module):
    """
    fidelity kernel module base ARD and use monte carlo to calculate the integral.

    Args:
        input_dim (int): The input dimension.
        initial_length_scale (float): The initial length scale value. Default is 1.0.
        initial_signal_variance (float): The initial signal variance value. Default is 1.0.
        eps (float): A small constant to prevent division by zero. Default is 1e-9.

    Attributes:
        length_scales (nn.Parameter): The length scales for each dimension.
        signal_variance (nn.Parameter): The signal variance.
        eps (float): A small constant to prevent division by zero.

    """

    def __init__(self, kernel1, b, initial_length_scale=0.0, initial_signal_variance=1.0, eps=1e-3):
        super().__init__()
        self.kernel1 = kernel1
        self.b = b
        # self.log_length_scales = nn.Parameter(torch.ones(input_dim) * initial_length_scale)
        self.log_length_scales = nn.Parameter(torch.tensor([initial_length_scale]))
        self.signal_variance = nn.Parameter(torch.tensor([initial_signal_variance]))
        self.eps = eps
        self.seed = 105

        self.k = nn.Parameter(torch.tensor([1.0]))
        self.c = nn.Parameter(torch.tensor([0.0]))

    def warp_function(self, f_1, f_2):
        wf_1 = f_1 * self.k + self.c
        wf_2 = f_2 * self.k + self.c
        return wf_1, wf_2
    
    def h(self, t, t_1):

        self.v = self.log_length_scales.exp() * self.b * 0.5
        tem_1 = (self.v**2).exp() / (2*self.b)
        tem_2 = (-self.b * t).exp() 
        tem_3 = (self.b * t_1).exp() * (torch.erf((t-t_1)/self.log_length_scales.exp() - self.v) + torch.erf((t_1)/self.log_length_scales.exp() + self.v))
        tem_4 = (-self.b * t_1).exp() * (torch.erf((t/self.log_length_scales.exp()) - self.v) + torch.erf(self.v))

        return tem_1 * tem_2 * (tem_3 - tem_4)

    def forward_interagl(self, x1, x2):
        """
        Compute the covariance matrix using the ARD kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        X1 = x1[:, :-1].reshape(-1, x1.shape[1]-1)
        X2 = x2[:, :-1].reshape(-1, x2.shape[1]-1)
        # raw_fidelity_indicator_1 = x1[:, 1].reshape(-1, 1) # t'
        # raw_fidelity_indicator_2 = x2[:, 1].reshape(-1, 1) # t

        # fidelity_indicator_1, fidelity_indicator_2 = self.warp_function(raw_fidelity_indicator_1, raw_fidelity_indicator_2)

        fidelity_indicator_1 = x1[:, -1].reshape(-1, 1) # t'
        fidelity_indicator_2 = x2[:, -1].reshape(-1, 1) # t
        # fidelity_indicator_1, fidelity_indicator_2 = self.warp_function(fidelity_indicator_1, fidelity_indicator_2)

        # print("length:", self.log_length_scales)
        # print("b:", self.b)

        tem = [fidelity_indicator_1 for i in range(fidelity_indicator_2.size(0))]
        T1 = torch.cat(tem, dim=1)
        tem = [fidelity_indicator_2 for i in range(fidelity_indicator_1.size(0))]
        T2 = torch.cat(tem, dim=1).T

        h_part_1 = self.h(T1, T2)
        h_part_2 = self.h(T2, T1)

        final_part = 0.5 * torch.sqrt(torch.tensor(torch.pi)) * self.log_length_scales.exp() * (h_part_1 + h_part_2) + (-self.b * (T1+T2)).exp()

        return self.signal_variance.abs() * final_part * self.kernel1(X1, X2)

    def forward(self, x1, x2):
        """
        Compute the covariance matrix using the ARD kernel.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The covariance matrix.

        """
        X1 = x1[:, :-1].reshape(-1, x1.shape[1]-1)
        X2 = x2[:, :-1].reshape(-1, x2.shape[1]-1)
        # raw_fidelity_indicator_1 = x1[:, 1].reshape(-1, 1) # t'
        # raw_fidelity_indicator_2 = x2[:, 1].reshape(-1, 1) # t

        

        fidelity_indicator_1 = x1[:, -1].reshape(-1, 1) # t'
        fidelity_indicator_2 = x2[:, -1].reshape(-1, 1) # t
        # fidelity_indicator_1, fidelity_indicator_2 = self.warp_function(fidelity_indicator_1, fidelity_indicator_2)
        length_scales = torch.abs(self.log_length_scales.exp()) + self.eps

        scaled_f1 = fidelity_indicator_1  / length_scales
        scaled_f2 = fidelity_indicator_2  / length_scales
        sqdist = torch.cdist(scaled_f1, scaled_f2, p=2)**2

        return self.signal_variance.abs() * torch.exp(-0.5 * sqdist) * self.kernel1(X1, X2)

class CMF_CAR_dkl(nn.Module):
    # initialize the model
    def __init__(self, input_dim, kernel_x, b_init=1.0):
        super().__init__()
        # self.fidelity_num = fidelity_num
        self.b = torch.nn.Parameter(torch.tensor(b_init))
        input_dim = input_dim +1
        self.FeatureExtractor = torch.nn.Sequential(nn.Linear(input_dim, input_dim *4),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(input_dim *4, input_dim * 4),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(input_dim * 4, input_dim * 4),
                                                    nn.LeakyReLU(),
                                                    nn.Linear(input_dim * 4, input_dim))

        kernel_full = fidelity_kernel_MC(kernel_x, self.b)
        self.cigp = GPR(kernel=kernel_full, log_beta=1.0)

    def forward(self, data_manager, x_test,fidelity_indicator = None, normal = False):
        '''
        # x_train = []
        # y_train = []
        # fidelity_indicator = []
        # for i_fidelity in range(self.fidelity_num):
        #     x, y = data_manager.get_data(i_fidelity)
        #     x_train.append(x)
        #     y_train.append(y)
        #     fidelity_indicator.append(torch.ones(x.shape[0]) * (i_fidelity+1))
        
        # x_train = torch.cat(x_train, 0)
        # y_train = torch.cat(y_train, 0)
        # fidelity_indicator = torch.cat(fidelity_indicator, 0)
        # x_train = torch.cat((x_train, fidelity_indicator.reshape(-1,1)), 1)
        
        # x_test =  torch.cat((x_test, (torch.ones(x_test.shape[0]) * self.fidelity_num).reshape(-1,1)), 1)
        '''
        
        if fidelity_indicator is not None:
            x_test = torch.cat([x_test.reshape(-1,x_test.shape[1]),(torch.tensor(fidelity_indicator)+1).reshape(-1,1)], dim = 1)
        x_train, y_train = data_manager.get_data(0, normal = normal)
        x_test = self.FeatureExtractor(x_test)
        x_train = self.FeatureExtractor(x_train.double())
        y_pred, cov_pred = self.cigp(x_train,y_train.double(),x_test)

        # return the prediction
        return y_pred, cov_pred
    
def train_CMFCAR_dkl(CARmodel, data_manager,max_iter=1000,lr_init=1e-1, normal = False):
    '''
    # x_train = []
    # y_train = []
    # fidelity_indicator = []
    # for i_fidelity in range(CARmodel.fidelity_num):
    #     x, y = data_manager.get_data(i_fidelity)
    #     x_train.append(x)
    #     y_train.append(y)
    #     fidelity_indicator.append(torch.ones(x.shape[0]) * (i_fidelity+1))
    
    # x_train = torch.cat(x_train, 0)
    # y_train = torch.cat(y_train, 0)
    # fidelity_indicator = torch.cat(fidelity_indicator, 0)
    # x_train = torch.cat((x_train, fidelity_indicator.reshape(-1,1)), 1)
    '''
    CARmodel = CARmodel.double()
    x_train, y_train = data_manager.get_data(0, normal = normal)

    optimizer = torch.optim.Adam(CARmodel.parameters(), lr=lr_init)

    for i in range(max_iter):
        optimizer.zero_grad()
        x_train_1 = CARmodel.FeatureExtractor(x_train.double())
        loss = CARmodel.cigp.negative_log_likelihood(x_train_1, y_train.double())
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()), end='\r')
    print(' ')
    
    
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
    y_train = torch.cat((y_low, y_high1, y_high2), 0)
    y_test = torch.sin(x_test)

    # x_low = torch.cat((x_low, 0.33*torch.ones(x_low.shape[0]).reshape(-1,1)), 1)
    # x_high1 = torch.cat((x_high1, 0.67*torch.ones(x_high1.shape[0]).reshape(-1,1)), 1)
    # x_high2 = torch.cat((x_high2, torch.ones(x_high2.shape[0]).reshape(-1,1)), 1)
    # x_test = torch.cat((x_test, torch.ones(x_test.shape[0]).reshape(-1,1)), 1)

    x_low = torch.cat((x_low, torch.ones(x_low.shape[0]).reshape(-1,1)), 1)
    x_high1 = torch.cat((x_high1, 2*torch.ones(x_high1.shape[0]).reshape(-1,1)), 1)
    x_high2 = torch.cat((x_high2, 3*torch.ones(x_high2.shape[0]).reshape(-1,1)), 1)
    x_test = torch.cat((x_test, 3*torch.ones(x_test.shape[0]).reshape(-1,1)), 1)

    x = torch.cat((x_low, x_high1, x_high2), 0)
    y = torch.cat((y_low, y_high1, y_high2), 0)

    # initial_data = [
    #     {'raw_fidelity_name': '0','fidelity_indicator': 0, 'X': x_low, 'Y': y_low},
    #     {'raw_fidelity_name': '1','fidelity_indicator': 1, 'X': x_high1, 'Y': y_high1},
    #     {'raw_fidelity_name': '2','fidelity_indicator': 2, 'X': x_high2, 'Y': y_high2},
    # ]
    initial_data = [
        {'raw_fidelity_name': '0','fidelity_indicator': 0, 'X': x.double(), 'Y': y.double()},
    ]

    fidelity_manager = MultiFidelityDataManager(initial_data)
    kernel_x = kernel.ARDKernel(x_low.shape[1])
    CAR = CMF_CAR_dkl(input_dim=x.shape[1]-1, kernel_x=kernel_x, b_init=1.0)

    train_CMFCAR_dkl(CAR, fidelity_manager, max_iter=200, lr_init=1e-2)

    with torch.no_grad():
        ypred, ypred_var = CAR(fidelity_manager,x_test.double())
 
    plt.figure()
    plt.errorbar(x_test[:,0].flatten(), ypred[:,0].reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(x_test[:,0].flatten(), ypred[:,0].detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred[:,0].detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha=0.2)
    plt.plot(x_test[:,0].flatten(), y_test[:,0], 'k+')
    plt.show()
    # plt.savefig('CMF_CAR_dkl.png') 

    # plt.figure()
    # plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    # plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha=0.2)
    # plt.plot(x_test.flatten(), y_test, 'k+')
    # plt.show() 