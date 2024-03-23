import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import GaussianProcess.kernel as kernel
from FidelityFusion_Models import *
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from Experiments.calculate_metrix import calculate_metrix
from Experiments.Load_Mfdata import get_full_name_list_with_fidelity, load_data_certain_fi, generate_nonsubset_data

import torch
import time
import pandas as pd


all_data_name_list = ["colville", "nonlinearsin", "toal", "forrester",
                          "tl1", "tl2", "tl3", "tl4", "tl5", "tl6", "tl7", "tl8", "tl9", "tl10",
                          "p1", "p2", "p3", "p4", "p5",
                          "maolin1", "maolin5", "maolin6", "maolin7", "maolin8", "maolin10", "maolin12", "maolin13",
                          "maolin15",
                          "maolin19", "maolin20",
                          "shuo6", "shuo11", "shuo15", "shuo16",
                          "test3", "test4", "test5", "test6", "test7"]
test_data_list = ["toal"]


model_dic = {'AR': AR,'AR_aa':AR_aa, 'ResGP': ResGP, 'NAR': NAR, 'CIGAR': CIGAR, 'GAR': GAR }
train_dic = {'AR': train_AR,'AR_aa':train_ARaa,'ResGP': train_ResGP, 'NAR': train_NAR,'CIGAR': train_CIGAR, 'GAR': train_GAR}

if __name__ == '__main__':
        
    # method_list = ['AR','AR_aa','ResGP','NAR','GAR','CIGAR']
    method_list = ["AR_aa"]
    all_data_name_with_fi_list = get_full_name_list_with_fidelity(data_name_list=test_data_list)   
    for _data_name in all_data_name_with_fi_list:
        print(_data_name)
        for method in method_list:
            print(method)
            for hidden in [2,4,8,16,32]:
                print("hidden_size:{}".format(hidden))
                for _seed in [0,1,2]:
                    print("seed:{}".format(_seed) )
                    recording = {'train_sample_num':[], 'rmse':[], 'nrmse':[], 'r2':[], 'nll':[], 'time':[]}
                    for _high_fidelity_num in [8, 16, 32, 64]:
                        torch.manual_seed(_seed)

                        # xtr, Ytr, xte, Yte = load_data_certain_fi(seed = 0, data_name_with_fi = _data_name, n_train = 100, n_test = 100, x_normal=True, y_normal=True)
                        xtr, Ytr, xte, Yte = generate_nonsubset_data(_data_name, x_dim = 10, min_value = -100, max_value = 100, num_points = 450, n_train = 300, n_test = 300,subset = True)
                        x_low = xtr[0]
                        y_low = Ytr[0]
                        x_high1 = xtr[1][:_high_fidelity_num]
                        y_high1 = Ytr[1][:_high_fidelity_num]
                        x_test = xte
                        y_test = Yte

                        data_shape = [y_low[0].shape, y_high1[0].shape]
                    
                        initial_data = [
                                            {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_low, 'Y': y_low},
                                            {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': x_high1, 'Y': y_high1},
                                        ]

                        T1 = time.time()
                        fidelity_manager = MultiFidelityDataManager(initial_data)
                        # kernel1 = kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.)
                        kernel_list = [kernel.SquaredExponentialKernel(), kernel.SquaredExponentialKernel()]
                        if method == 'AR':
                            model = model_dic[method](fidelity_num=2,kernel_list = kernel_list, rho_init=1.0)
                        elif method == 'AR_aa':
                            model = model_dic[method](fidelity_num=2,kernel_list = kernel_list, input_size = x_low.shape[1],hidden_size = hidden, if_nonsubset=False)
                        elif method in ['CIGAR', 'GAR']:
                            model = model_dic[method](fidelity_num=2,kernel_list = kernel_list, data_shape_list=data_shape)
                        else:
                            model = model_dic[method](fidelity_num=2,kernel_list = kernel_list)

                        if method in ['GAR','CIGAR']:
                            max_iter = 100
                            lr = 1e-3
                        else:
                            max_iter = 300
                            lr = 1e-2
                        train_dic[method](model, fidelity_manager, max_iter = max_iter, lr_init = lr)

                        with torch.no_grad():
                            x_test = fidelity_manager.normalizelayer[model.fidelity_num-1].normalize_x(x_test)
                            ypred, ypred_var = model(fidelity_manager,x_test)
                            ypred, ypred_var = fidelity_manager.normalizelayer[model.fidelity_num-1].denormalize(ypred, ypred_var)
                        
                        if method in ['GAR','CIGAR']:
                            ypred_var = torch.diag_embed(torch.flatten(ypred_var))
                        metrics = calculate_metrix(y_test = y_test, y_mean_pre = ypred.reshape(-1, 1), y_var_pre = ypred_var)

                        T2 = time.time()
                        recording['train_sample_num'].append(_high_fidelity_num)
                        recording['rmse'].append(metrics['rmse'])
                        recording['nrmse'].append(metrics['nrmse'])
                        recording['r2'].append(metrics['r2'])
                        recording['nll'].append(metrics['nll'])
                        recording['time'].append(T2 - T1)

                    path_csv = os.path.join('Experiments', 'AR_aa_try', 'hidden_size_res', _data_name)
                    if not os.path.exists(path_csv):
                            os.makedirs(path_csv)

                    record = pd.DataFrame(recording)
                    record.to_csv(path_csv + '/' +'hidden'+ str(hidden) + '_seed_' + str(_seed) + '.csv', index = False) # 将数据写入

                        