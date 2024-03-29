import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import GaussianProcess.kernel as kernel
from FidelityFusion_Models import *
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from MF_BayesianOptimization.Discrete.DMF_acq import DiscreteAcquisitionFunction, optimize_acq_mf
import matplotlib.pyplot as plt
torch.manual_seed(1)

# def objective_function(x, s):
#     xtr = x
#     if s == 0:
#         Ytr = torch.sin(xtr * 8 * torch.pi)
#     else:
#         Ytr_l = torch.sin(xtr * 8 * torch.pi)
#         Ytr = (xtr - torch.sqrt(torch.ones(xtr.shape[0])*2).reshape(-1, 1)) * torch.pow(Ytr_l, 2)
    
#     return Ytr
def objective_function(x, s):
    xtr = x
    if s == 0:
        # Ytr = torch.sin(xtr) - 0.5 * torch.sin(2 * xtr) + torch.rand(xtr.shape[0], 1) * 0.1 - 0.05
        Ytr = torch.sin(xtr) - 0.5 * torch.sin(2 * xtr)
    else:
        # Ytr = torch.sin(xtr) + torch.rand(xtr.shape[0], 1) * 0.1 - 0.05
        Ytr = torch.sin(xtr)
    return Ytr

def mean_function(x, s):
    mean, _ = model.forward(fidelity_manager, x, s)
    return mean.reshape(-1, 1)
    
def variance_function(x, s):
    _, variance = model.forward(fidelity_manager, x, s)
    return variance

if __name__ == '__main__':
    # train_xl = torch.rand(10, 1) * 10
    train_xh = torch.rand(5, 1) * 10
    train_yl = objective_function(train_xh, 0)
    train_yh = objective_function(train_xh, 1)

    data_shape = [train_yl[0].shape, train_yh[0].shape]
    bounds = torch.tensor([[0, 10]] * train_xh.shape[1])

    initial_data = [
                        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': train_xh, 'Y': train_yl},
                        {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': train_xh, 'Y': train_yh},
                    ]
    fidelity_manager = MultiFidelityDataManager(initial_data)
    fi_num = len(initial_data)
    kernel1 = [kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.) for _ in range(fi_num)]
    # model = AR(fidelity_num = fi_num, kernel_list = kernel1, rho_init=1.0, if_nonsubset=True)
    # train_AR(model, fidelity_manager, max_iter=100, lr_init=1e-3)

    model = ResGP(fidelity_num = fi_num, kernel_list = kernel1, if_nonsubset = True)
    acq = DiscreteAcquisitionFunction(mean_function, variance_function, 2, train_xh.shape[1], torch.ones(1).reshape(-1, 1))
    
    key_iterations = [2,4,5,6,8,10]
    predictions = []
    observed_l = []
    observed_h = []

    test_points = torch.linspace(0, 10, 100).reshape(-1, 1)
    lowf_y = objective_function(test_points, 0)
    true_y = objective_function(test_points, 1)

    for iteration in range(10):

        train_ResGP(model, fidelity_manager, max_iter=200, lr_init=1e-2)

        # Use Opitmizer to find the new_x
        new_x = optimize_acq_mf(fidelity_manager, acq, 'KG', bounds, 10, 0.01) 
        # Use different selection strategy to select next_s
        new_s = acq.acq_selection_fidelity(gamma=[0.2, 0.2], new_x=new_x)
        print(new_x, new_s)
        new_y = objective_function(new_x, new_s)
        fidelity_manager.add_data(raw_fidelity_name=str(new_s), fidelity_index=new_s, x=new_x, y=new_y)
        if (iteration + 1) in key_iterations:
            model.eval()
            with torch.no_grad():
                test_x = fidelity_manager.normalizelayer[model.fidelity_num-1].normalize_x(test_points)
                ypred, ypred_var = model(fidelity_manager, test_x)
                ypred, ypred_var = fidelity_manager.normalizelayer[model.fidelity_num-1].denormalize(ypred, ypred_var)
                predictions.append((ypred, ypred_var))
                observed_l.append((fidelity_manager.get_data(0)))
                observed_h.append((fidelity_manager.get_data(1)))

    plt.figure(figsize=(15, 12))
    for i, (ypred, ypred_var) in enumerate(predictions):
        plt.subplot(3, 2, i+1)
        plt.ylim(-2, 2)
        plt.plot(test_points, true_y, 'k-', label='True function')
        plt.plot(test_points, lowf_y, 'r-', label='Low fidelity function')
        plt.plot(test_points, ypred, 'b--', label='GP mean')
        
        plt.fill_between(test_points.reshape(-1),
                        ypred.reshape(-1) - ypred_var.diag().sqrt(),
                        ypred.reshape(-1) + ypred_var.diag().sqrt(),
                        color='blue', alpha=0.2, label='GP uncertainty')

        plt.scatter(observed_h[i][0], observed_h[i][1], c='g', zorder=2, label='Observed points (High)')
        plt.scatter(observed_l[i][0], observed_l[i][1], c='r', zorder=2, label='Observed points (Low)')
        plt.title(f'Samples: {key_iterations[i]}')
        # plt.legend()
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
