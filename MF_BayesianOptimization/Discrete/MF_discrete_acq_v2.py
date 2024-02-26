import torch
import torch.nn as nn
from scipy.stats import norm

from torch.distributions import  Normal

class DiscreteAcquisitionFunction(nn.Module):
    """
    Discrete Acquisition Base Function for UCB, ES, EI and KG

    Args:
        mean_function (function): The mean function for posterior distribution.
        variance_function (function): The variance function for posterior distribution.
        fidelity_num (int): Total fidelity number e.g. 2 or 5.
        f_best (tensor): The best observed objective function value to date. If the acq don't need to use it, can be set as None.

    Attributes:
        UCB_MF: Compute the score of upper confidence bound for input x and targeted fidelity s.
        # ES_MF: Compute the score of Entropy Search for input x and targeted fidelity s.
        EI_MF: Compute the score of Expectation Improvement for input x and targeted fidelity s.
        PI_MF: Compute the score of Probability Improvement for input x and targeted fidelity s.
        KG_MF: Compute the score of Knowledge Gradient for input x and targeted fidelity s.
        acq_selection_fidelity: According to MF_GP_UCB to select fidelity strategy.

    """
    def __init__(self, mean_function, variance_function, fidelity_num, x_dimension, f_best):
        super(DiscreteAcquisitionFunction, self).__init__()
        self.mean_function = mean_function
        self.variance_function = variance_function
        self.fidelity_num = fidelity_num
        self.x_dimension = x_dimension

        if f_best is not None:
            self.f_best = f_best
        else:
            self.f_best = None


    def UCB_MF(self, x, s):
        '''
        Compute the score of upper confidence bound for input x and targeted fidelity s.

        Args:
            x (torch.Tensor): Targeted input.
            s (int): Targeted fidelity s.

        Returns:
            torch.Tensor: The score of UCB
        '''
        self.beta = 0.2 * int(self.x_dimension)
        # mean = self.mean_function(x, s)
        ucb = self.mean_function(x, s) + self.beta * self.variance_function(x, s)
        return ucb
    
    # def ES_MF(self, x, s):
    #     '''
    #     Compute the score of Entropy Search for input x and targeted fidelity s.

    #     Args:
    #         x (torch.Tensor): Targeted input.
    #         s (int): Targeted fidelity s.

    #     Returns:
    #         torch.Tensor: The score of UCB
    #     '''
    #     mean = self.mean_function(x, s)
    #     var = self.variance_function(x, s)
    #     normal = torch.normal(mean, var)
    #     entropy = normal.entropy()

    #     return entropy
    
    def EI_MF(self, x, s):
        """
        Compute the EI values for the given inputs.

        Args:
            x (torch.Tensor): The input points where EI is to be evaluated.
            s (int): Targeted fidelity s.

        Returns:
            torch.Tensor: The EI values for the input points.
        """
        xi = 0.01
        mean = self.mean_function(x, s)
        variance = self.variance_function(x, s)
        std = torch.sqrt(variance)

        # Preventing division by zero in standard deviation
        std = torch.clamp(std, min=1e-9)

        Z = (mean - self.f_best - xi) / std
        ei = (mean - self.f_best - xi) * torch.tensor(norm.cdf(Z.detach().numpy()), dtype=torch.float32) + std * torch.tensor(norm.pdf(Z.detach().numpy()), dtype=torch.float32)
        return ei
    
    def PI_MF(self, x, s):
        """
        Compute the PI values for the given inputs.

        Args:
            x (torch.Tensor): The input points where PI is to be evaluated.
            s (int): Targeted fidelity s.

        Returns:
            torch.Tensor: The PI values for the input points.
        """
        theta = 0.01
        mean = self.mean_function(x, s)
        variance = self.variance_function(x, s)
        std = torch.sqrt(variance)

        # Preventing division by zero in standard deviation
        std = torch.clamp(std, min=1e-9)

        Z = (mean - self.f_best - theta) / std
        pi = torch.tensor(norm.cdf(Z.numpy()), dtype=torch.float32)
        return pi
    
    def KG_MF(self, x, s):
        """
        Compute the KG values for the given inputs.

        Args:
            x (torch.Tensor): The input points where KG is to be evaluated.
            s (int): Targeted fidelity s.

        Returns:
            torch.Tensor: The KG values for the input points.
        """
        
        # 使用模型的predict_mean和predict_var方法获取预测均值和方差
        mean = self.mean_function(x, s)
        variance = self.variance_function(x, s)
        std = torch.sqrt(variance)

        # 防止标准差为零
        std = torch.clamp(std, min=1e-6)
        std = torch.nan_to_num(std, nan=1e-6)

        # 生成幻想样本
        normal_dist = torch.distributions.Normal(mean, std)
        fantasies = normal_dist.rsample(sample_shape=torch.Size([self.num_fantasies]))

        # 计算每个幻想样本的预期改善
        best_fantasies, _ = fantasies.max(dim=0)
        expected_improvement = best_fantasies - self.f_best

        # 对所有幻想样本求平均，以估计KG
        kg = expected_improvement.mean(dim=0)

        return kg
    
    def acq_selection_fidelity(self, gamma, new_x):
        '''
        According to MF_GP_UCB to select fidelity.

        Args:
            gamma (list): The threshold for whether choose the higher fidelity
            x (torch.Tensor): Targeted input.

        Returns:
            int: The next candidate fidelity
        '''

        for i in range(self.fidelity_num):
            v = self.variance_function(new_x, i)

            if self.beta * v > gamma[i]:
                new_s = i + 1
            else:
                new_s = i

        return new_s


def optimize_acq_mf(fidelity_manager, acq_mf, n_iterations = 10, learning_rate = 0.001):
    '''
    Optimize the acquisition function to get the next candidate point for acq.

    Args:
        fidelity_manager (module):The data manager object.
        acq_mf (AcquisitionFunction): An instance of the AcquisitionFunction class.
        n_iterations (int): Iteration times for optimize x.
        learning_rate (float): learning rate for optimize x.

    Returns:
        torch.Tensor: The next candidate input without fidelity
    '''

    fidelity_num = int((len(fidelity_manager.data_dict) +1) / 2)
    x_dimension = fidelity_manager.data_dict['0']['X'].shape[1]

    acquisiton_score_by_fidelity = []
    acquisiton_x_by_fidelity = []
    for i in range(fidelity_num):
        X_initial = nn.Parameter(torch.rand(x_dimension).reshape(-1, 1), requires_grad = True)
        optimizer = torch.optim.Adam([X_initial], lr=learning_rate)
        # optimizer.zero_grad()
        for j in range(n_iterations):
            # optimizer.zero_grad()
            loss = -1 * acq_mf(X_initial, i)
            loss.backward()
            optimizer.step()
            print('iter', j, 'x:', X_initial, 'Negative Acquisition Function:', loss.item(), end='\n')

        acquisiton_x_by_fidelity.append(X_initial.detach())
        acquisiton_score_by_fidelity.append(loss.item())

    new_x = acquisiton_x_by_fidelity[acquisiton_score_by_fidelity.index(min(acquisiton_score_by_fidelity))]

    return new_x
