# commonly used functions for GP computation
# Author: Wei Xing
# Date: 2023-12-11
# Version: 1.0
# History:
# 1.0    2023-12-11    Initial version

import torch
import torch.nn as nn
import numpy as np
import tensorly
tensorly.set_backend('pytorch')


EPS = 1e-9
JITTER = 1e-6
PI = 3.1415
# define a normalization module

    
# TODO: add a warpping layer. follow https://botorch.org/tutorials/bo_with_warped_gp 
# class warp_layer(nn.Module):
#     def __init__(self, warp_func, if_trainable =False):
#         super().__init__()
#         self.warp_func = warp_func
#         self.warp_func.requires_grad = if_trainable
#     def forward(self, x):
#         return self.warp_func(x)
#     def inverse(self, x):
#         return self.warp_func.inverse(x)


# compute the log likelihood of a normal distribution
def Gaussian_log_likelihood(y, cov, Kinv_method='cholesky3'):
    """
    Compute the log-likelihood of a Gaussian distribution.

    Args:
        y (torch.Tensor): The observed values.
        mean (torch.Tensor): The mean of the Gaussian distribution.
        cov (torch.Tensor): The covariance matrix of the Gaussian distribution.
        Kinv_method (str, optional): The method to compute the inverse of the covariance matrix.
            Defaults to 'cholesky3'.

    Returns:
        torch.Tensor: The log-likelihood of the Gaussian distribution.

    Raises:
        ValueError: If Kinv_method is not 'direct' or 'cholesky'.
    """
    
    # assert if the correct dimension
    assert len(y.shape) == 2 and len(cov.shape) == 2, "y, mean, cov should be 2D tensors"
    
    if Kinv_method == 'cholesky1':
        L = torch.linalg.cholesky(cov)
        L_inv = torch.inverse(L)
        K_inv = L_inv.T @ L_inv
        return -0.5 * (y.T @ K_inv @ y + 2 * torch.logdet(cov) + len(y) * np.log(2 * np.pi))
    elif Kinv_method == 'cholesky2':
        L = torch.linalg.cholesky(cov)
        gamma = torch.cholesky_solve(y, L)
        return -0.5 * (gamma.T @ gamma + 2 * torch.logdet(cov) + len(y) * np.log(2 * np.pi))
    
    elif Kinv_method == 'cholesky3':
        # fastest implementation so far
        L = torch.linalg.cholesky(cov)
        # return -0.5 * (y_use.T @ torch.cholesky_solve(y_use, L) + L.diag().log().sum() + len(x_train) * np.log(2 * np.pi))
        if y.shape[1] > 1:
            Warning('y_use.shape[1] > 1, will treat each column as a sample (for the joint normal distribution) and sum the log-likelihood')
            # 
            # (Alpha ** 2).sum() = (Alpha @ Alpha^T).diag().sum() = \sum_i (Alpha @ Alpha^T)_{ii}
            # 
            y_dim = y.shape[1]
            log_det_K = 2 * torch.sum(torch.log(torch.diag(L)))
            gamma = torch.cholesky_solve(y, L, upper = False)
            return - 0.5 * ( (gamma ** 2).sum() + log_det_K * y_dim + len(y) * y_dim * np.log(2 * np.pi) )
        else:
            gamma = torch.cholesky_solve(y, L, upper = False)
            return -0.5 * (gamma.T @ gamma + 2 * L.diag().log().sum() + len(y) * np.log(2 * np.pi))

    elif Kinv_method == 'direct':
        K_inv = torch.inverse(cov)
        return -0.5 * (y.T @ K_inv @ y + 2 * torch.logdet(cov) + len(y) * np.log(2 * np.pi))
    elif Kinv_method == 'torch_distribution_MN1':
        L = torch.linalg.cholesky(cov)
        return torch.distributions.MultivariateNormal(y, scale_tril=L).log_prob(y)
    elif Kinv_method == 'torch_distribution_MN2':
        return torch.distributions.MultivariateNormal(y, cov).log_prob(y)
    else:
        raise ValueError('Kinv_method should be either direct or cholesky')
    
def conditional_Gaussian(y, Sigma, K_s, K_ss, Kinv_method='cholesky3'):
    # Sigma = Sigma + torch.eye(len(Sigma)) * EPS
    if Kinv_method == 'cholesky1':   # kernel inverse is not stable, use cholesky decomposition instead
        L = torch.linalg.cholesky(Sigma)
        L_inv = torch.inverse(L)
        K_inv = L_inv.T @ L_inv
        alpha = K_inv @ y
        mu = K_s.T @ alpha
        v = L_inv @ K_s
        cov = K_ss - v.T @ v
    elif Kinv_method == 'cholesky3':
        # recommended implementation, fastest so far
        L = torch.linalg.cholesky(Sigma)
        alpha = torch.cholesky_solve(y, L)
        mu = K_s.T @ alpha
        # v = torch.cholesky_solve(K_s, L)    # wrong implementation
        v = L.inverse() @ K_s   # correct implementation
        cov = K_ss - v.T @ v
    elif Kinv_method == 'direct':
        K_inv = torch.inverse(Sigma)
        mu = K_s.T @ K_inv @ y
        cov = K_ss - K_s.T @ K_inv @ K_s
    else:
        raise ValueError('Kinv_method should be either direct or cholesky')
    
    return mu, cov

def negative_log_likelihood(kernel, log_beta, x_train, y_train):
        
    ## A scheme proposed by WJ to make GP more stable -2024/3/4

    y_num, y_dimension = y_train.shape
    Sigma = kernel(x_train, x_train) + log_beta.exp().pow(-1) * torch.eye(
        x_train.size(0)) + JITTER * kernel(x_train, x_train).mean() * torch.eye(x_train.size(0))
    
    L = torch.linalg.cholesky(Sigma)
    #option 1 (use this if torch supports)
    Gamma,_ = torch.triangular_solve(y_train, L, upper = False)
    #option 2
    # gamma = L.inverse() @ Y       # we can use this as an alternative because L is a lower triangular matrix.

    nll =  0.5 * (Gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
        + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
    return -nll

class Tensor_linear(torch.nn.Module):
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

    def forward(self,x):
        for i in range(len(self.l_shape)):
            x = tensorly.tenalg.mode_dot(x, self.vectors[i], i+1)
        return x