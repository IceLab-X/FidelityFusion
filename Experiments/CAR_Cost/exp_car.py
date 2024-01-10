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


 

seed = list(range(3))
fidelity_num = 2

def get_cost(ones_num_list):
        cost = 0
        for fid in range(len(ones_num_list)):
                cost += ones_num_list[fid] * pow(2, fid+1)

        return cost


for _data_name in ['sample_data']:
        train_sample_num = 64
        recording = {'cost':[], 'rmse':[], 'r2':[], 'nll':[], 'nrmse':[], 'time':[]}
        for _seed in [0,1]:
            random.seed(_seed)
            sample_num = torch.randint(0, 64)
            x_all = torch.rand(500, 1) * 20

            xlow_indices = torch.randperm(500)[:train_sample_num]
            xlow_indices = torch.sort(xlow_indices).values
            x_low = x_all[xlow_indices]

            xhigh_indices = torch.randperm(500)[:train_sample_num-sample_num]
            xhigh_indices = torch.sort(xhigh_indices).values
            x_high1 = x_all[xhigh_indices]

            y_low = torch.sin(x_low) - torch.rand(train_sample_num, 1) * 0.2 
            y_high1 = torch.sin(x_high1) - torch.rand(train_sample_num-sample_num, 1) * 0.1

            x_test = torch.linspace(0, 20, 100).reshape(-1, 1)
            y_test = torch.sin(x_test)

            T1 = time.time()

            initial_data = [
                                {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_low, 'Y': y_low},
                                {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': x_high1, 'Y': y_high1},
                            ]

            fidelity_manager = MultiFidelityDataManager(initial_data)
            fidelity_num = 2
            kernel_list = [kernel.ARDKernel(x_low.shape[1]) for _ in range(fidelity_num)]
            CAR = ContinuousAutoRegression(fidelity_num=fidelity_num, kernel_list=kernel_list, b_init=1.0)

            train_CAR(CAR,fidelity_manager, max_iter=100, lr_init=1e-2)

            with torch.no_grad():
                ypred, ypred_var = CAR(fidelity_manager,x_test)
                
            metrics = calculate_metrix(y_test = y_test, y_mean_pre = ypred.reshape(-1, 1), y_var_pre = ypred_var)

            T2 = time.time()
            recording['cost'].append(int(train_sample_num-sample_num))
            recording['rmse'].append(metrics['rmse'])
            recording['nrmse'].append(metrics['nrmse'])
            recording['r2'].append(metrics['r2'])
            recording['nll'].append(metrics['nll'])
            recording['time'].append(T2 - T1)

            path_csv = os.path.join('Experiments', 'CAR_subset', 'exp_results', str(_data_name))
            if not os.path.exists(path_csv):
                    os.makedirs(path_csv)

            record = pd.DataFrame(recording)
            record.to_csv(path_csv + '/car_seed_' + str(_seed) + '.csv', index = False) # 将数据写入

            
