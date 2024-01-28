import torch
import torch.nn as nn

def acq_optimize(self, niteration, lr):
    optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    # optimizer.zero_grad()
    for i in range(niteration):
        optimizer.zero_grad()
        loss = self.negative_acq()
        loss.backward()
        optimizer.step()
        print('iter'+str(i)+'/'+str(niteration), 'loss_negative_acq:', loss.item(), end='\r')

class UCB(nn.Module):
       # mean_func is also an nn.Module taking in input x, fidelity indicator t, and returning the acquisition function value
       # the input to MF_UCB is also x, t
       # they should be considered as two concatenating layers of a neural network
    def __init__(self, mean_func, var_func):
        super(UCB, self).__init__()
        self.mean_func = mean_func
        self.var_func = var_func
        
        
    