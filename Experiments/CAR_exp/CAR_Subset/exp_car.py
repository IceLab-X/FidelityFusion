import pandas as pd
import random 
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import GaussianProcess.kernel as kernel
from FidelityFusion_Models.CAR_ContinuousAutoRegression import ContinuousAutoRegression
from FidelityFusion_Models.CAR_ContinuousAutoRegression import train_CAR
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from Experiments.calculate_metrix import calculate_metrix
import torch


data_name_list = ['sample_data'] # 实验用的数据

dec_rate = 0.75
fidelity_num = 5

for _data_name in data_name_list:
        for _seed in [0,1]:
            recording = {'train_sample_num':[], 'rmse':[], 'nrmse':[], 'r2':[], 'nll':[], 'time':[]}

            for train_sample_num in [32, 64, 96, 128]:
                random.seed(_seed)

                x_all = torch.rand(500, 1) * 20

                xlow_indices = torch.randperm(500)[:train_sample_num]
                xlow_indices = torch.sort(xlow_indices).values
                x_low = x_all[xlow_indices]

                xhigh_indices = torch.randperm(train_sample_num)[:int(dec_rate * train_sample_num)]
                xhigh_indices = torch.sort(xhigh_indices).values
                x_high1 = x_low[xhigh_indices]

                y_low = torch.sin(x_low) - torch.rand(train_sample_num, 1) * 0.2 
                y_high1 = torch.sin(x_high1) - torch.rand(int(dec_rate * train_sample_num), 1) * 0.1

                x_test = torch.linspace(0, 20, 100).reshape(-1, 1)
                y_test = torch.sin(x_test)
                T1 = time.time()

                initial_data = [
                                    {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_low, 'Y': y_low},
                                    {'fidelity_indicator': 1,'raw_fidelity_name': '1','X': x_high1, 'Y': y_high1},
                                ]

                fidelity_manager = MultiFidelityDataManager(initial_data)
                fidelity_num = 2
                kernel_list = [kernel.ARDKernel(x_low.shape[1]) for _ in range(fidelity_num)]
                # kernel_residual = fidelity_kernel_MCMC(x_low.shape[1], kernel.ARDKernel(x_low.shape[1]), 1, 2)
                CAR = ContinuousAutoRegression(fidelity_num=fidelity_num, kernel_list=kernel_list, b_init=1.0)

                train_CAR(CAR, fidelity_manager, max_iter=100, lr_init=1e-2)

                with torch.no_grad():
                    ypred, ypred_var = CAR(fidelity_manager,x_test)
                
                metrics = calculate_metrix(y_test = y_test, y_mean_pre = ypred.reshape(-1, 1), y_var_pre = ypred_var)

                T2 = time.time()
                recording['train_sample_num'].append(train_sample_num)
                recording['rmse'].append(metrics['rmse'])
                recording['nrmse'].append(metrics['nrmse'])
                recording['r2'].append(metrics['r2'])
                recording['nll'].append(metrics['nll'])
                recording['time'].append(T2 - T1)

            path_csv = os.path.join('Experiments', 'CAR_subset', 'exp_results', str(_data_name))
            if not os.path.exists(path_csv):
                    os.makedirs(path_csv)

            record = pd.DataFrame(recording)
            record.to_csv(path_csv + '/car_' + str(dec_rate) + '_seed_' + str(_seed) + '.csv', index = False) # 将数据写入

            