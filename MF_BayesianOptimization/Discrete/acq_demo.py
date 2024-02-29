import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import GaussianProcess.kernel as kernel
from FidelityFusion_Models import *
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from MF_discrete_acq_v2 import DiscreteAcquisitionFunction
from MF_discrete_acq_v2 import optimize_acq_mf

def objective_function(x, s):
    xtr = x
    if s == 0:
        Ytr = torch.sin(xtr * 8 * torch.pi)
    else:
        Ytr_l = torch.sin(xtr * 8 * torch.pi)
        Ytr = (xtr - torch.sqrt(torch.ones(xtr.shape[0])*2).reshape(-1, 1)) * torch.pow(Ytr_l, 2)
    
    return Ytr


train_xl = torch.rand(8, 1) * 10
train_xh = torch.rand(4, 1) * 10
train_yl = objective_function(train_xl, 0)
train_yh = objective_function(train_xh, 1)


data_shape = [train_yl[0].shape, train_yh[0].shape]

initial_data = [
                    {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': train_xl, 'Y': train_yl},
                    {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': train_xh, 'Y': train_yh},
                ]

fidelity_manager = MultiFidelityDataManager(initial_data)
kernel1 = kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.)
# model = AR(fidelity_num=2, kernel=kernel1, rho_init=1.0, if_nonsubset=True)
# train_AR(model, fidelity_manager, max_iter=100, lr_init=1e-3)
model = ResGP(fidelity_num=2, kernel=kernel1, if_nonsubset=True)
train_ResGP(model, fidelity_manager, max_iter=100, lr_init=1e-3)

def mean_function(x, s):
    mean, _ = model.forward(fidelity_manager, x, s)
    return mean.reshape(-1, 1)
    
def variance_function(x, s):
    _, variance = model.forward(fidelity_manager, x, s)
    return variance

# if want to use the KG/EI/PI acq, need specify the current best function (f_best) when initiate
# UCB do not need the f_best    
acq = DiscreteAcquisitionFunction(mean_function, variance_function, 2, train_xl.shape[1], torch.ones(1).reshape(-1, 1))

# Use Opitmizer to find the new_x
new_x = optimize_acq_mf(fidelity_manager, acq.PI_MF, 10, 0.01) 
# Use different selection strategy to select next_s
new_s = acq.acq_selection_fidelity(gamma=[0.1, 0.1], new_x=new_x)
print(new_x, new_s)