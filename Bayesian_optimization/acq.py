import inspect
import math
from numbers import Real

import numpy as np
import torch
from scipy.stats import norm


def find_next_batch(acq, bounds, batch_size=1, n_samples=1000, f_best=0):
    """
    Find the next batch of points to sample by selecting the ones with the highest UCB from a large set of random samples.

    Args:
        bounds (np.ndarray): The bounds for each dimension of the input space.
        batch_size (int): The number of points in the batch.
        n_samples (int): The number of random points to sample for finding the maximum UCB.

    Returns:
        torch.Tensor: The next batch of points to sample.
    """

    # 获取forward函数的参数信息
    signature = inspect.signature(acq.forward)
    parameters = signature.parameters

    # 打印参数数量
    num_params = len(parameters)
    print(f"Number of parameters in forward function: {num_params}")

    X_selected = []
    for _ in range(batch_size):
        # Generate a large number of random points
        X_random = torch.FloatTensor(n_samples, bounds.shape[0]).uniform_(bounds[0, 0], bounds[0, 1])

        if num_params == 1:
        # Compute UCB for all random points
            UCB_values = acq.forward(X_random)
        if num_params == 2:
            UCB_values = acq.forward(X_random, f_best)
        # Select the point with the highest UCB value
        idx_max = torch.argmax(UCB_values)
        X_selected.append(X_random[idx_max])
    return torch.stack(X_selected)


class UCB:
    def __init__(self, mean_func, variance_func, kappa=2.0):
        """
        UCB Formula:
        UCB(x) = mean(x) + kappa * sqrt(variance(x)), where
        mean(x) is the mean predicted by the GP at input x,
        variance(x) is the variance predicted by the GP at input x,and kappa is the exploration-exploitation trade-off parameter.

        Args:
            mean_func (callable): Function to compute the mean of the GP at given points (PyTorch tensor).
            variance_func (callable): Function to compute the variance of the GP at given points (PyTorch tensor).
            kappa (float): Controls the balance between exploration and exploitation.
        """
        self.mean_func = mean_func
        self.variance_func = variance_func
        self.kappa = kappa
        
    # forward
    def forward(self, X):
        """
        Compute the UCB values for the given inputs.

        Args:
            X (torch.Tensor): The input points where UCB is to be evaluated.

        Returns:
            torch.Tensor: The UCB values for the input points.
        """
        mean = self.mean_func(X)
        variance = self.variance_func(X)
        return mean + self.kappa * torch.sqrt(variance)


class EI:
    def __init__(self, mean_func, variance_func, xi=0.01):
        """
        EI formula:
            EI(x) = (mean(x) - f_best - xi) * Phi(Z) + std(x) * phi(Z)
            where Z = (mean(x) - f_best - xi) / std(x),
            Phi(Z) is the cumulative distribution function of the standard normal distribution,
            and phi(Z) is the probability density function of the standard normal distribution.
            
        Args:
            mean_func (callable): Function to compute the mean of the GP at given points.
            variance_func (callable): Function to compute the variance of the GP at given points.
            xi (float): Controls the balance between exploration and exploitation.
        """
        self.mean_func = mean_func
        self.variance_func = variance_func
        self.xi = xi

    def forward(self, X, f_best):
        """
        Compute the EI values for the given inputs.

        Args:
            X (torch.Tensor): The input points where EI is to be evaluated.
            f_best (float): The best observed objective function value to date.

        Returns:
            torch.Tensor: The EI values for the input points.
        """
        mean = self.mean_func(X)
        variance = self.variance_func(X)
        std = torch.sqrt(variance)

        # Preventing division by zero in standard deviation
        std = torch.clamp(std, min=1e-9)

        Z = (mean - f_best - self.xi) / std
        ei = (mean - f_best - self.xi) * torch.tensor(norm.cdf(Z.numpy()), dtype=torch.float32) + std * torch.tensor(norm.pdf(Z.numpy()), dtype=torch.float32)
        return ei


class PI:
    def __init__(self, mean_func, variance_func, sita=0.01):
        """
        PI formula:
            PI(x) = Phi((mean(x) - f_best - sita) / std(x)),
            where Phi(Z) is the cumulative distribution function of the standard normal distribution.
            
        Args:
            mean_func (callable): Function to compute the mean of the GP at given points.
            variance_func (callable): Function to compute the variance of the GP at given points.
            xi (float): Controls the balance between exploration and exploitation.
        """
        self.mean_func = mean_func
        self.variance_func = variance_func
        self.sita = sita

    def forward(self, X, f_best):
        """
        Compute the PI values for the given inputs.

        Args:
            X (torch.Tensor): The input points where PI is to be evaluated.
            f_best (float): The best observed objective function value to date.

        Returns:
            torch.Tensor: The PI values for the input points.
        """
        mean = self.mean_func(X)
        variance = self.variance_func(X)
        std = torch.sqrt(variance)

        # Preventing division by zero in standard deviation
        std = torch.clamp(std, min=1e-9)

        Z = (mean - f_best - self.sita) / std
        pi = torch.tensor(norm.cdf(Z.numpy()), dtype=torch.float32)
        return pi


class KG:
    def __init__(self,mean_func, variance_func, num_fantasies=10):
        """
        KG Formula:
        KG(x) = E[max(f(x, z) - f_best)], where E is the expectation over fantasy samples,
        f(x, z) is the objective function with input x and fantasy sample z.

        Args:
            mean_func (callable): Function to compute the mean of the GP at given points.
            variance_func (callable): Function to compute the variance of the GP at given points.
            num_fantasies (int): The number of fantasy samples to approximate the KG.
        """
        self.mean_func = mean_func
        self.variance_func = variance_func
        self.num_fantasies = num_fantasies

    def forward(self, X, f_best):
        # 使用模型的predict_mean和predict_var方法获取预测均值和方差
        mean = self.mean_func(X)
        variance = self.variance_func(X)
        std = torch.sqrt(variance)

        # 防止标准差为零
        std = torch.clamp(std, min=1e-6)
        std = torch.nan_to_num(std, nan=1e-6)

        # 生成幻想样本
        normal_dist = torch.distributions.Normal(mean, std)
        fantasies = normal_dist.rsample(sample_shape=torch.Size([self.num_fantasies]))

        # 计算每个幻想样本的预期改善
        best_fantasies, _ = fantasies.max(dim=0)
        expected_improvement = best_fantasies - f_best

        # 对所有幻想样本求平均，以估计KG
        kg = expected_improvement.mean(dim=0)

        return kg
