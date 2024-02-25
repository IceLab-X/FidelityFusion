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

    Attributes:
        UCB_MF: Compute the score of upper confidence bound for input x and targeted fidelity s.
        ES_MF: Compute the score of Entropy Search for input x and targeted fidelity s.
        UCB_selection_fidelity: According to MF_GP_UCB to select fidelity.

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
    
    def UCB_selection_fidelity(self, gamma, new_x):
        '''
        According to MF_GP_UCB to select fidelity.

        Args:
            gamma (float): The threshold for whether choose the higher fidelity
            x (torch.Tensor): Targeted input.

        Returns:
            int: The next candidate fidelity
        '''

        m = self.mean_function(new_x, 0)
        v = self.variance_func(new_x, 0)

        if self.beta * v > gamma:
            new_s = 0
        else:
            new_s = 1

        return new_s



def optimize_acq_mf(fidelity_manager, acq_mf, n_iterations = 10, learning_rate = 0.001):
    '''
    Optimize the acquisition function to get the next candidate point for UCB.

    Args:
        fidelity_manager (module):The data manager object.
        acq_mf (AcquisitionFunction): An instance of the AcquisitionFunction class.
        n_iterations (int): Iteration times for optimize x.
        learning_rate (float): learning rate for optimize x.

    Returns:
        torch.Tensor: The next candidate input without fidelity
    '''

    fidelity_num = len(fidelity_manager)
    x_dimension = fidelity_manager['X'].shape[1]

    acquisiton_score_by_fidelity = []
    acquisiton_x_by_fidelity = []
    for i in range(fidelity_num):
        X_initial = torch.rand(x_dimension).reshape(-1, 1)
        optimizer = torch.optim.Adam([X_initial], lr=learning_rate)
        optimizer.zero_grad()
        for i in range(n_iterations):
            # optimizer.zero_grad()
            loss = - acq_mf(X_initial, i)
            loss.backward(retain_graph = True)
            optimizer.step()
            # self.x.data.clamp_(0.0, 1.0)
            print('iter', i, 'x:', X_initial, 'Negative Acquisition Function:',loss.item(), end='\n')
        acquisiton_x_by_fidelity.append(X_initial.detach())
        acquisiton_score_by_fidelity.append(acq_mf(X_initial, fidelity_indicator=i+1))

    new_x = acquisiton_x_by_fidelity[acquisiton_score_by_fidelity.index(min(acquisiton_score_by_fidelity))]

    return new_x
    
        
