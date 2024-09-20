import torch
import torch.nn as nn
from scipy.stats import norm
from GaussianProcess.kernel import ARDKernel

PI = 3.1415926
EPS = 1e-9

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
    
    X_initial = nn.Parameter(torch.rand(x_dimension).reshape(-1, 1), requires_grad = True)
    optimizer = torch.optim.Adam([X_initial], lr=learning_rate)
    # optimizer.zero_grad()
    for j in range(n_iterations):
        # optimizer.zero_grad()
        loss = -1 * acq_mf.forward(X_initial, fidelity_num)
        loss.backward()
        optimizer.step()
        print('iter', j, 'x:', X_initial, 'Negative Acquisition Function:', loss.item(), end='\n')

    new_x = X_initial.detach()

    return new_x


class CMF_UCB:
    def __init__(self, mean_function, variance_function, fidelity_num, x_dimension):
        """
        Initialize the Discrete Multi-Fidelity Upper Confidence Bound (UCB) acquisition function.
        Args:
            mean_function (function): The mean function for posterior distribution.
            variance_function (function): The variance function for posterior distribution.
            fidelity_num (int): Total fidelity number e.g. 2 or 5.
            x_dimension (int): The dimension of input.
        """
        self.mean_function = mean_function
        self.variance_function = variance_function
        self.fidelity_num = fidelity_num
        self.x_dimension = x_dimension
        self.beta = 0.2 * int(self.x_dimension)
        self.p = 1
        self.k_0 = 1

    # forward
    def forward(self, x, s):
        '''
        Compute the score of upper confidence bound for input x and targeted fidelity s.
        Args:
            x (torch.Tensor): Targeted input.
            s (int): Targeted fidelity s.
        Returns:
            torch.Tensor: The score of UCB
        '''

        # mean = self.mean_function(x, s)
        ucb = self.mean_function(x, s) + self.beta * self.variance_function(x, s)

        return ucb
    
    def information_gap(self, input_fidelity):
        '''
        Information gap is a function to measures the gap in information between input fidelity and the highest fidelity.
        Args:
            input_fidelity (float): The fidelity input to measure the information gap.

        Returns:
            torch.tensor: Gap of two fidelities.
        '''

        fidelity_kernel = ARDKernel(input_dim = 1, initial_length_scale=1.0, initial_signal_variance=1.0, eps=EPS)

        if input == None:
            input_fidelity = torch.ones(1, 1) * self.fidelity_num
        else:
            input_fidelity = torch.ones(1, 1) * input_fidelity

        phi = fidelity_kernel.forward(input_fidelity, torch.ones(1, 1) * self.fidelity_num)
        ksin = torch.sqrt(1 - torch.power(phi, 2))

        return ksin
    
    def gamma_z(self, ksin_z):
        '''
        Use cost and information gap to balance each other, 
        so that the choice is not completely inclined to low-fidelity or high-fidelity.
        Args:
            ksin_z (float): The fidelity input to measure the information gap.

        Returns:
            torch.tensor: Gap of two fidelities.
        '''
        q = 1 / (self.p + self.x_dimension + 2)
        lambda_balance = torch.pow(self.model_cost.compute_cost(self.fidelity_num)/self.model_cost.compute_cost(1), q)
        gamma_z = torch.sqrt(self.k_0) * ksin_z * lambda_balance

        return gamma_z

    
    def acq_selection_fidelity(self, new_x):
        '''
        According to MF_GP_UCB to select fidelity.
        Args:
            gamma (list): The threshold for whether choose the higher fidelity
            x (torch.Tensor): Targeted input.
        Returns:
            int: The next candidate fidelity
        '''
        tau_s_mean = []
        tau_s_std = []
        for s in list(range(self.fidelity_num)):
            m = self.mean_function(new_x, s)
            v = self.variance_function(new_x, s)
            tau_s_mean.append(m.detach())
            tau_s_std.append(torch.sqrt(v.detach()))

        ksin_z = self.information_gap(None)
        gamma_z = self.gamma_z(ksin_z)

        possible_z = []
        for i in range(list(range(self.fidelity_num))):
            condition_1 = tau_s_std[i][0][0] > gamma_z[i]
            condition_2 = ksin_z[i] > self.information_gap(torch.sqrt(self.p)) / torch.sqrt(self.beta)
            if condition_1 and condition_2:
                possible_z.append(list(range(self.fidelity_num)))

        if len(possible_z) == 0:
            new_s = 0.1
        else:
            new_s = min(possible_z)

        return new_s

    

class CMF_EI:
    def __init__(self, mean_function, variance_function, fidelity_num, x_dimension, xi):
        """
        Initialize the Discrete Multi-Fidelity Entropy Information (EI) acquisition function.
        Args:
            mean_function (function): The mean function for posterior distribution.
            variance_function (function): The variance function for posterior distribution.
            fidelity_num (int): Total fidelity number e.g. 2 or 5.
            x_dimension (int): The dimension of input.
            xi (float): Controls the balance between exploration and exploitation.
        """
        self.mean_function = mean_function
        self.variance_function = variance_function
        self.fidelity_num = fidelity_num
        self.x_dimension = x_dimension
        self.xi = xi
        self.beta = 0.2 * int(self.x_dimension)

    def forward(self, x, s, f_best):
        """
        Compute the EI values for the given inputs.
        Args:
            x (torch.Tensor): The input points where EI is to be evaluated.
            s (int): Targeted fidelity s.
            f_best (torch.Tensor): The best observed objective function value to date.
            # s_best (int): The best observed value corresponding fidelity indicator.
        Returns:
            torch.Tensor: The EI values for the input points.
        """

        mean = self.mean_function(x, s)
        variance = self.variance_function(x, s)
        std = torch.sqrt(variance)

        # Preventing division by zero in standard deviation
        std = torch.clamp(std, min=1e-9)

        Z = (mean - f_best - self.xi) / std
        ei = (mean - f_best - self.xi) * torch.tensor(norm.cdf(Z.detach().numpy()), dtype=torch.float32) + std * torch.tensor(norm.pdf(Z.detach().numpy()), dtype=torch.float32)

        return ei

class CMF_PI:
    def __init__(self, mean_function, variance_function, fidelity_num, x_dimension, theta = 0.01):
        """
        Initialize the Discrete Multi-Fidelity Probability Improvement (PI) acquisition function.
        Args:
            mean_function (function): The mean function for posterior distribution.
            variance_function (function): The variance function for posterior distribution.
            fidelity_num (int): Total fidelity number e.g. 2 or 5.
            x_dimension (int): The dimension of input.
            theta (float): Controls the balance between exploration and exploitation.
        """
        self.mean_function = mean_function
        self.variance_function = variance_function
        self.fidelity_num = fidelity_num
        self.x_dimension = x_dimension
        self.theta = theta
        self.beta = 0.2 * int(self.x_dimension)

    def forward(self, x, s, f_best):
        """
        Compute the PI values for the given inputs.
        Args:
            x (torch.Tensor): The input points where PI is to be evaluated.
            s (int): Targeted fidelity s.
            f_best (torch.Tensor): The best observed objective function value to date.
        Returns:
            torch.Tensor: The PI values for the input points.
        """
        mean = self.mean_function(x, s)
        variance = self.variance_function(x, s)
        std = torch.sqrt(variance)

        # Preventing division by zero in standard deviation
        std = torch.clamp(std, min=1e-9)

        Z = (mean - f_best - self.theta) / std
        pi = - torch.pow(Z, 2) * 0.5 - torch.log(torch.ones(1, 1)) - torch.log(torch.sqrt(2 * PI * torch.ones(1, 1) ))

        return pi

class CMF_KG:
    def __init__(self, mean_function, variance_function, fidelity_num, x_dimension, theta = 0.01):
        """
        Initialize the Discrete Multi-Fidelity Probability Improvement (PI) acquisition function.
        Args:
            mean_function (function): The mean function for posterior distribution.
            variance_function (function): The variance function for posterior distribution.
            fidelity_num (int): Total fidelity number e.g. 2 or 5.
            x_dimension (int): The dimension of input.
            theta (float): Controls the balance between exploration and exploitation.
        """
        self.mean_function = mean_function
        self.variance_function = variance_function
        self.fidelity_num = fidelity_num
        self.x_dimension = x_dimension
        self.theta = theta
        self.beta = 0.2 * int(self.x_dimension)

    def forward(self, x, s, f_best):
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
        num_fantasies = 10
        normal_dist = torch.distributions.Normal(mean, std)
        fantasies = normal_dist.rsample(sample_shape=torch.Size([num_fantasies]))

        # 计算每个幻想样本的预期改善
        best_fantasies, _ = fantasies.max(dim=0)
        expected_improvement = best_fantasies - f_best

        # 对所有幻想样本求平均，以估计KG
        kg = expected_improvement.mean(dim=0)

        return kg