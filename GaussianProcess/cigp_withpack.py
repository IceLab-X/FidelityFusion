'''
CIGP use the gp_computation_pack 2024/3/11
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.nn as nn
import kernel as kernel
from matplotlib import pyplot as plt
import time as time

import gp_computation_pack as gp_pack
import gp_transform as gp_transform
    
def zeroMean(x):
    return torch.zeros(x.shape[0], 3)

class constMean(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(output_dim))
    def forward(self, x):
        return self.mean.expand(x.shape[0], -1)

class CIGP(nn.Module):
    def __init__(self, kernel, noise_variance):
        super().__init__()
        self.kernel = kernel
        self.noise_variance = nn.Parameter(torch.tensor([noise_variance]))

    def forward(self, x_train, y_train, x_test):
        # xNormalizer = gp_pack.Normalize_layer(x_train, dim=0, if_trainable =False)
        # yNormalizer = gp_pack.Normalize0_layer(y_train, if_trainable =False)
        # x_train = xNormalizer(x_train)
        # y_train = yNormalizer(y_train)
        # x_test = xNormalizer(x_test)
        
        if isinstance(y_train, list):
            y_train_var = y_train[1]
            y_train = y_train[0]
        else:
            y_train_var = None

        K = self.kernel(x_train, x_train) + self.noise_variance.pow(2) * torch.eye(len(x_train)).to(x_train.device)
        K_s = self.kernel(x_train, x_test)
        K_ss = self.kernel(x_test, x_test)
        
        mu, cov = gp_pack.conditional_Gaussian(y_train, K, K_s, K_ss)
        return mu, cov

    def log_likelihood(self, x_train, y_train):
        if isinstance(y_train, list):
            y_train_var = y_train[1]
            y_train = y_train[0]
        else:
            y_train_var = None
        K = self.kernel(x_train, x_train) + self.noise_variance.pow(2) * torch.eye(len(x_train))
        if y_train_var is not None:
            K = K + y_train_var.diag()* torch.eye(x_train.size(0)).to(x_train.device)
        return gp_pack.Gaussian_log_likelihood(y_train, K)
        
# downstate here how to use the GP model
if __name__ == '__main__':
    xte = torch.linspace(0, 6, 100).view(-1, 1)
    yte = torch.sin(xte) + 10

    xtr = torch.rand(16, 1) * 6
    ytr = torch.sin(xtr) + torch.randn(16, 1) * 0.5 + 10

    kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
    model = CIGP(kernel = kernel1, noise_variance = 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)
    
    for i in range(200):
        startTime = time.time()
        optimizer.zero_grad()
        loss = -model.log_likelihood(xtr, ytr)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        timeElapsed = time.time() - startTime
        print('time elapsed: {:.3f}s'.format(timeElapsed))
    with torch.no_grad():
        ypred, ypred_var = model.forward(xtr,ytr,xte)

    plt.figure()
    plt.errorbar(xte.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(xte.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha = 0.2)
    plt.plot(xte.flatten(), yte, 'k+')
    plt.show()