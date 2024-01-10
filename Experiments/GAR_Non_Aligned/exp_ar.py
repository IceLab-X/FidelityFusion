import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import GaussianProcess.kernel as kernel
from FidelityFusion_Models.AR_autoRegression import autoRegression
from FidelityFusion_Models.AR_autoRegression import train_AR
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from Experiments.calculate_metrix import calculate_metrix

import torch
import time
import torch.nn as nn
import pandas as pd


real_dataset = ['FlowMix3D_MF',
                'MolecularDynamic_MF', 
                'plasmonic2_MF', 
                'SOFC_MF',]

gen_dataset = ['poisson_v4_02',
               'burger_v4_02',
               'Burget_mfGent_v5',
               'Burget_mfGent_v5_02',
                # 'Heat_mfGent_v5',
                'Piosson_mfGent_v5',
                'Schroed2D_mfGent_v1',
                'TopOP_mfGent_v5',]

interp_data = False

if __name__ == '__main__':
        
        for _data_name in ['sample_data']:
            for _seed in [0,1]:
                recording = {'train_sample_num':[], 'rmse':[], 'nrmse':[], 'r2':[], 'nll':[], 'time':[]}
                for _high_fidelity_num in [4, 8, 16, 32]:
                    torch.manual_seed(_seed)
                    # generate the data
                    x_all = torch.rand(500, 1) * 20

                    xlow_indices = torch.randperm(500)[:300]
                    xlow_indices = torch.sort(xlow_indices).values
                    x_low = x_all[xlow_indices]

                    xhigh_indices = torch.randperm(500)[:_high_fidelity_num]
                    xhigh_indices = torch.sort(xhigh_indices).values
                    x_high1 = x_all[xhigh_indices]

                    y_low = torch.sin(x_low) - torch.rand(300, 1) * 0.2 
                    y_high1 = torch.sin(x_high1) - torch.rand(_high_fidelity_num, 1) * 0.1

                    x_test = torch.linspace(0, 20, 100).reshape(-1, 1)
                    y_test = torch.sin(x_test)
                
                    initial_data = [
                                        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_low, 'Y': y_low},
                                        {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': x_high1, 'Y': y_high1},
                                    ]

                    T1 = time.time()
                    fidelity_manager = MultiFidelityDataManager(initial_data)
                    kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
                    AR = autoRegression(fidelity_num=2,kernel=kernel1,rho_init=1.0)

                    train_AR(AR,fidelity_manager, max_iter=100, lr_init=1e-2)

                    with torch.no_grad():
                        ypred, ypred_var = AR(fidelity_manager,x_test)


                    metrics = calculate_metrix(y_test = y_test, y_mean_pre = ypred.reshape(-1, 1), y_var_pre = ypred_var)

                    T2 = time.time()
                    recording['train_sample_num'].append(_high_fidelity_num)
                    recording['rmse'].append(metrics['rmse'])
                    recording['nrmse'].append(metrics['nrmse'])
                    recording['r2'].append(metrics['r2'])
                    recording['nll'].append(metrics['nll'])
                    recording['time'].append(T2 - T1)

                path_csv = os.path.join('Experiments', 'GAR_Non_Aligned', 'exp_results', str(_data_name))
                if not os.path.exists(path_csv):
                        os.makedirs(path_csv)

                record = pd.DataFrame(recording)
                record.to_csv(path_csv + '/ar_seed_' + str(_seed) + '.csv', index = False) # 将数据写入

                    