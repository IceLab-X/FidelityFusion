import torch
import torch.nn as nn

from torch.distributions import  Normal

class DiscreteAcquisitionFunction(nn.Module):
    """
    Discrete Acquisition Base Function for UCB, ES, EI and cfKG

    Args:
        mean_function (function): The mean function for posterior distribution.
        variance_function (function): The variance function for posterior distribution.
        fidelity_num (int): Total fidelity number e.g. 2 or 5.
    """
    def __init__(self, mean_function, variance_function, fidelity_num, x_dimension):
        super(DiscreteAcquisitionFunction, self).__init__()
        self.mean_function = mean_function
        self.variance_function = variance_function
        self.fidelity_num = fidelity_num
        self.x_dimension = x_dimension

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
        mean = self.mean_function(x, s)
        ucb = self.mean_function(x, s) + self.beta * self.variance_function(x, s)

        return ucb
    
    def ES_MF(self, x, s):
        '''
        Compute the score of Entropy Search for input x and targeted fidelity s.

        Args:
            x (torch.Tensor): Targeted input.
            s (int): Targeted fidelity s.

        Returns:
            torch.Tensor: The score of UCB
        '''
        mean = self.mean_function(x, s)
        var = self.variance_function(x, s)
        normal = torch.normal(mean, var)
        entropy = normal.entropy()

        return entropy


    def UCB_optimize(self):
        self.gamma = 0.1
        N_UCB = []
        UCB_x = []
        for i in range(self.fidelity_num):
            tt = torch.rand(self.x_dimension).reshape(-1, 1)
            self.x = nn.Parameter(tt)
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            optimizer.zero_grad()
            for i in range(15):
                # optimizer.zero_grad()
                loss = - self.UCB_MF(self.x, i)
                loss.backward(retain_graph = True)
                optimizer.step()
                # self.x.data.clamp_(0.0, 1.0)
                print('iter', i, 'x:', self.x, 'loss_negative_ucb:',loss.item(), end='\n')
            UCB_x.append(self.x.detach())
            N_UCB.append(self.UCB_MF(self.x, fidelity_indicator=i+1))

        new_x = UCB_x[N_UCB.index(min(N_UCB))]


        m = self.mean_function(new_x, 0)
        v = self.variance_func(new_x, 0)

        if self.beta * v > self.gamma:
            new_s = 0
        else:
            new_s = 1
        return new_x, new_s
    
    def ES_optimize(self):
        self.gamma = 0.1
        N_ES = []
        ES_x = []
        for i in range(self.fidelity_num):
            tt = torch.rand(self.x_dimension)
            self.x = nn.Parameter(torch.from_numpy(tt.reshape(1, self.x_dimension)).double())
            self.optimise_adam(fidelity_indicator=i+1, niteration=15, lr=0.01)
            ES_x.append(self.x.detach())
            N_ES.append(self.ES_MF(self.x, fidelity_indicator=i+1))

        new_x = ES_x[N_ES.index(min(N_ES))]

        m = self.mean_function(new_x, 0)
        v = self.variance_func(new_x, 0)

        if self.beta * v > self.gamma:
            new_s = 0
        else:
            new_s = 1

        return new_x, new_s
        
