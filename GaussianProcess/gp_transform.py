# gp_transform.py
# transform block for GP models
# Compared to the standard transform in torch, this transform also takes care of the covariance matrix.
# 
# Author: Wei Xing
# Date: 2023-12-13
# Version: 1.0
# History:
# 1.0    2023-12-13    Initial version

import torch
import torch.nn as nn

class Normalize0_layer(nn.Module):
    # special normalization, i.e., all dimensions are normalized together. This works well for conditional independent GP (CIGP) for normalizing y.
    def __init__(self, X0, if_trainable =False):
        super().__init__()
        self.mean = nn.Parameter(X0.mean(), requires_grad=if_trainable)
        self.std = nn.Parameter(X0.std(), requires_grad=if_trainable)
    def forward(self, x): 
        return (x - self.mean) / self.std
    def inverse(self, x):
        return x * self.std + self.mean
    
class Normalize_layer(nn.Module):
    # normal normalization. It is basically the pytorch batch normalization, but the mean and std are not trainable. 
    # It work well for normalizing the input x.
    def __init__(self, X0, dim=0, if_trainable =False):
        super().__init__()
        self.mean = nn.Parameter(X0.mean(dim), requires_grad=if_trainable)
        self.std = nn.Parameter(X0.std(dim), requires_grad=if_trainable)
    def forward(self, x):
        return (x - self.mean) / self.std
    def inverse(self, x):
        return x * self.std + self.mean


class Normalize0_DistributionLayer(nn.Module):
    # special normalization, i.e., all dimensions are normalized together. This works well for conditional independent GP (CIGP) for normalizing y.
    def __init__(self, X0, if_trainable =False):
        super().__init__()
        self.mean = nn.Parameter(X0.mean(), requires_grad=if_trainable)
        self.std = nn.Parameter(X0.std(), requires_grad=if_trainable)
    def forward(self, x, Sigma=0): 
        return (x - self.mean) / self.std, Sigma / (self.std**2)
    def inverse(self, x, Sigma=0):
        return x * self.std + self.mean, Sigma * (self.std**2)
    
class Normalize_DistributionLayer(nn.Module):
#  x must be vector, Sigma must be a matrix indicating the covariance matrix of x.
    def __init__(self, X0, dim=0, if_trainable =False):
        super().__init__()
        self.mean = nn.Parameter(X0.mean(dim), requires_grad=if_trainable) # mean vector
        self.std = nn.Parameter(X0.std(dim), requires_grad=if_trainable)    # std vector
    def forward(self, x, Sigma=0):
        mean_result = (x - self.mean) / self.std
        Sigma_result = self.std.inverse().view(-1, 1) @ Sigma @ self.std.inverse().view(1, -1)
        return mean_result, Sigma_result 
    def inverse(self, x,  Sigma=0):
        mean_result = x * self.std + self.mean
        Sigma_result = self.std.view(-1, 1) @ Sigma @ self.std.view(1, -1)
        return mean_result, Sigma_result
