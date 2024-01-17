import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import GaussianProcess.kernel as kernel
from FidelityFusion_Models.GAR import GAR
from FidelityFusion_Models.GAR import train_GAR
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from Experiments.calculate_metrix import calculate_metrix

import torch
import time
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
                
                    # x_train = [x_low, x_high1]
                    # y_train = [y_low, y_high1]

                    initial_data = [
                                        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_low, 'Y': y_low},
                                        {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': x_high1, 'Y': y_high1},
                                    ]
                    fidelity_manager = MultiFidelityDataManager(initial_data)

                    T1 = time.time()

                    low_shape=[y_low[0].shape]
                    high_shape=[y_high1[0].shape, y_high1[0].shape]

                    kernel1 = kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.)
                    myGAR = GAR(kernel1, low_shape, high_shape, fidelity = 2)
                    train_GAR(myGAR, fidelity_manager, max_iter=100, lr_init=1e-2)
                    with torch.no_grad():
                        ypred, ypred_var = myGAR(fidelity_manager, x_test)


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
                record.to_csv(path_csv + '/gar_seed_' + str(_seed) + '.csv', index = False) # 将数据写入

                    