import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import GaussianProcess.kernel as kernel
from FidelityFusion_Models import *
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from Experiments.calculate_metrix import calculate_metrix
from Experiments.Load_Mfdata import get_full_name_list_with_fidelity, load_data_certain_fi, generate_nonsubset_data
from FidelityFusion_Models.GAR_res_rho.GAR_res import GAR_res, train_GARres
from FidelityFusion_Models.GAR_res_rho.GAR_rho import GAR_rho, train_GARrho

import torch
import time
import pandas as pd
import numpy as np


test_data_list = ["posion"]

model_dic = {'AR': AR, 'ResGP': ResGP, 'NAR': NAR, 'CIGAR': CIGAR, 'GAR': GAR, 'CAR': ContinuousAutoRegression,'GAR_rho': GAR_rho,'GAR_res': GAR_res}
train_dic = {'AR': train_AR,'ResGP': train_ResGP, 'NAR': train_NAR,'CIGAR': train_CIGAR, 'GAR': train_GAR, 'CAR': train_CAR,'GAR_rho': train_GARrho,'GAR_res': train_GARres}

if __name__ == '__main__':
        
    method_list = ['GAR_res','GAR_rho']
    # method_list = ['GAR','CIGAR']
        # print(_data_name)
    for method in method_list:
        print(method)
        for _seed in [0]:
            print(_seed)
            recording = {'train_sample_num':[], 'rmse':[], 'nrmse':[], 'r2':[], 'time':[]}
            for _high_fidelity_num in [8,16,32,64]:
                torch.manual_seed(_seed)

                x = np.load('assets/MF_data/Poisson_data/input.npy')
                x = torch.tensor(x, dtype=torch.float32)
                yl=np.load('assets/MF_data/Poisson_data/output_fidelity_0.npy')
                yl = torch.tensor(yl, dtype=torch.float32)
                yh = np.load('assets/MF_data/Poisson_data/output_fidelity_1.npy')
                yh = torch.tensor(yh, dtype=torch.float32)
                # yh2 = np.load('assets/MF_data/Poisson_data/output_fidelity_2.npy')
                # yh2 = torch.tensor(yh2, dtype = torch.float32)
                x_low = x[:100]
                # y_low = yl[:100].reshape(100,-1)
                y_low = yl[:100]
                x_high1 = x[:_high_fidelity_num]
                # y_high1 = yh[:_high_fidelity_num].reshape(_high_fidelity_num,-1)
                y_high1 = yh[:_high_fidelity_num]
                x_test = x[-100:]
                # y_test = yh[-100:].reshape(100, -1)
                y_test = yh[-100:]

                data_shape = [y_low[0].shape, y_high1[0].shape]
            
                initial_data = [
                                    {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_low, 'Y': y_low},
                                    {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': x_high1, 'Y': y_high1},
                                ]

                T1 = time.time()
                fidelity_manager = MultiFidelityDataManager(initial_data)
                kernel_list = [kernel.SquaredExponentialKernel(), kernel.SquaredExponentialKernel()]
                if method == 'AR':
                    model = model_dic[method](fidelity_num=2,kernel_list = kernel_list, rho_init=1.0)
                elif method in ['CIGAR', 'GAR','GAR_res','GAR_rho']:
                    model = model_dic[method](fidelity_num=2,kernel_list = kernel_list, data_shape_list=data_shape)
                
                # elif method == 'CAR':
                #     model = model_dic[method](fidelity_num=2,kernel_list = kernel_list,input_dim = x_low.shape[1])
                else:
                    model = model_dic[method](fidelity_num=2,kernel_list = kernel_list)

                if method in ['GAR','CIGAR']:
                    max_iter = 100
                    lr = 1e-3
                else:
                    max_iter = 100
                    lr = 1e-3
                train_dic[method](model, fidelity_manager, max_iter = max_iter, lr_init = lr)

                T1 = time.time()
                with torch.no_grad():
                    x_test = fidelity_manager.normalizelayer[model.fidelity_num-1].normalize_x(x_test)
                    ypred, ypred_var = model(fidelity_manager,x_test)
                    ypred, ypred_var = fidelity_manager.normalizelayer[model.fidelity_num-1].denormalize(ypred, ypred_var)
                T2 = time.time()
                # if method in ['GAR','CIGAR']:
                #     ypred_var = torch.diag_embed(torch.flatten(ypred_var))
                metrics = calculate_metrix(y_test = y_test.reshape(-1, 1), y_mean_pre = ypred.reshape(-1, 1))

                # T2 = time.time()
                recording['train_sample_num'].append(_high_fidelity_num)
                recording['rmse'].append(metrics['rmse'])
                recording['nrmse'].append(metrics['nrmse'])
                recording['r2'].append(metrics['r2'])
                # recording['nll'].append(metrics['nll'])
                recording['time'].append(T2 - T1)

            path_csv = os.path.join('Experiments', 'GAR_Aligned', 'exp_results', str("posion"))
            if not os.path.exists(path_csv):
                    os.makedirs(path_csv)

            record = pd.DataFrame(recording)
            record.to_csv(path_csv + '/' + method + '_seed_' + str(_seed) + '.csv', index = False) # 将数据写入
