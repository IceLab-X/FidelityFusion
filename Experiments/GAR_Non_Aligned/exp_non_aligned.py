import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import GaussianProcess.kernel as kernel
from FidelityFusion_Models import *
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from Experiments.calculate_metrix import calculate_metrix
from FidelityFusion_Models.Load_Mfdata import get_full_name_list_with_fidelity, load_data_certain_fi

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
all_data_name_list = ["colville", "nonlinearsin", "toal", "forrester",
                          "tl1", "tl2", "tl3", "tl4", "tl5", "tl6", "tl7", "tl8", "tl9", "tl10",
                          "p1", "p2", "p3", "p4", "p5",
                          "maolin1", "maolin5", "maolin6", "maolin7", "maolin8", "maolin10", "maolin12", "maolin13",
                          "maolin15",
                          "maolin19", "maolin20",
                          "shuo6", "shuo11", "shuo15", "shuo16",
                          "test3", "test4", "test5", "test6", "test7", "test8", "test9"]

interp_data = False

model_dic = {'AR': AR, 'ResGP': ResGP, 'NAR': NAR, 'CIGAR': CIGAR, 'GAR': GAR}
train_dic = {'AR': train_AR,'ResGP': train_ResGP, 'NAR': train_NAR,'CIGAR': train_CIGAR, 'GAR': train_GAR}

if __name__ == '__main__':
        
    method_list = ['AR','ResGP','NAR','GAR','CIGAR']
    all_data_name_with_fi_list = get_full_name_list_with_fidelity(data_name_list=all_data_name_list)   
    for _data_name in all_data_name_with_fi_list:
        print(_data_name)
        for method in method_list:
            print(method)
            for _seed in [0,1]:
                print(_seed)
                recording = {'train_sample_num':[], 'rmse':[], 'nrmse':[], 'r2':[], 'nll':[], 'time':[]}
                for _high_fidelity_num in [4, 8, 16, 32]:
                    torch.manual_seed(_seed)
                    
                    xtr, Ytr, xte, Yte = load_data_certain_fi(seed = 0, data_name_with_fi = _data_name, n_train = 100, n_test = 100, x_normal=True, y_normal=True)
                    
                    x_low = xtr
                    y_low = Ytr[0]
                    x_high1 = x_low[:_high_fidelity_num]
                    y_high1 = Ytr[1][:_high_fidelity_num]
                    x_test = xte
                    y_test = Yte[1]
                    # generate the data
                    # x_all = torch.rand(500, 1) * 20

                    # xlow_indices = torch.randperm(500)[:300]
                    # xlow_indices = torch.sort(xlow_indices).values
                    # x_low = x_all[xlow_indices]

                    # xhigh_indices = torch.randperm(300)[:_high_fidelity_num]
                    # xhigh_indices = torch.sort(xhigh_indices).values
                    # x_high1 = x_low[xhigh_indices]

                    # y_low = torch.sin(x_low) - 0.5 * torch.sin(2 * x_low) + torch.rand(300, 1) * 0.1 - 0.05
                    # y_high1 = torch.sin(x_high1) - torch.rand(_high_fidelity_num, 1) * 0.1 - 0.05

                    # x_test = torch.linspace(0, 20, 100).reshape(-1, 1)
                    # y_test = torch.sin(x_test)

                    data_shape = [y_low[0].shape, y_high1[0].shape]
                
                    initial_data = [
                                        {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': x_low, 'Y': y_low},
                                        {'fidelity_indicator': 1, 'raw_fidelity_name': '1','X': x_high1, 'Y': y_high1},
                                    ]

                    T1 = time.time()
                    fidelity_manager = MultiFidelityDataManager(initial_data)
                    kernel1 = kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.)

                    if method == 'AR':
                        model = model_dic[method](fidelity_num=2, kernel=kernel1, rho_init=1.0)
                    elif method in ['CIGAR', 'GAR']:
                        model = model_dic[method](fidelity_num=2, kernel=kernel1, data_shape_list = data_shape)
                    else:
                        model = model_dic[method](fidelity_num=2, kernel=kernel1)

                    train_dic[method](model, fidelity_manager, max_iter=100, lr_init=1e-3)

                    with torch.no_grad():
                        ypred, ypred_var = model(fidelity_manager,x_test)
    
                    
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

                path_csv = os.path.join('Experiments', 'GAR_Non_Aligned', 'exp_results', str(_data_name))
                if not os.path.exists(path_csv):
                        os.makedirs(path_csv)

                record = pd.DataFrame(recording)
                record.to_csv(path_csv + '/' + method + '_seed_' + str(_seed) + '.csv', index = False) # 将数据写入

                        