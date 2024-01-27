import numpy as np
import pandas as pd
import sys
import os
import time
import torch

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# os.chdir(sys.path[-1])

realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('FidelityFusion') + 1])
sys.path.append(realpath)
os.chdir(sys.path[-1])

import GaussianProcess.kernel as kernel
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from Experiments.calculate_metrix import calculate_metrix
from Data_simulation.Synthetic_MF_Function import *
from FidelityFusion_Models import *
from Acquisition_Function.Discrete import *



model_dic = {'AR': AR, 'ResGP': ResGP, 'NAR': NAR, 'CIGAR': CIGAR, 'GAR': GAR, 'CAR': ContinuousAutoRegression}
train_dic = {'AR': train_AR,'ResGP': train_ResGP, 'NAR': train_NAR,'CIGAR': train_CIGAR, 'GAR': train_GAR, 'CAR':train_CAR}
Acq_list = {'UCB': upper_confidence_bound, 'ES': entropy_search, 'EI': expected_improvement, 'cfKG': discrete_fidelity_knowledgement_gradient}
Data_list = {'non_linear_sin': non_linear_sin, 'forrester': forrester}


def MF_BO_discrete(exp_config):
        seed = exp_config["seed"]

        '''Initiate Setting'''
        data_model = exp_config["data_model"]
        total_fidelity_num = exp_config['total_fidelity_num']
        initial_index = exp_config['initial_index']
        BO_iterations = exp_config['BO_iterations']
        MF_iterations = exp_config['MF_iterations']
        MF_learning_rate = exp_config['MF_learning_rate']

        '''prepare initial data'''
        data = data_model(total_fidelity_num)
        index = initial_index
        xtr, ytr = data.Initiate_data(index, seed)
        low_shape = ytr[0].shape[1]
        high_shape = ytr[1].shape[1]
        initial_data = [
                    {'fidelity_indicator': 0,'raw_fidelity_name': '0', 'X': xtr[0], 'Y': ytr[0]},
                    {'fidelity_indicator': 1,'raw_fidelity_name': '1', 'X': xtr[1], 'Y': ytr[1]},
                ]

        model_cost = data.cost
        recording = {"cost": [model_cost.compute_model_cost(ytr)],
                     "incumbents": [max(ytr[1].tolist()[0])],
                     "operation_time": [float(0)]}

        for i in range(BO_iterations):
                print('iteration:', i + 1)
                T1 = time.time()
                fidelity_manager = MultiFidelityDataManager(initial_data)
                kernel1 = kernel.SquaredExponentialKernel(length_scale = 1., signal_variance = 1.)

                # Fit the Gaussian process model to the sampled points
                method = exp_config["MF_model"]

                if method == 'AR':
                    model_objective = model_dic[method](fidelity_num=total_fidelity_num, kernel=kernel1, rho_init=1.0, nonsubset = True)
                elif method in ['CIGAR', 'GAR']:
                    model_objective = model_dic[method](fidelity_num=total_fidelity_num, kernel=kernel1, l_shape=low_shape, h_shape=high_shape, nonsubset = True)
                else:
                    model_objective = model_dic[method](fidelity_num=total_fidelity_num, kernel=kernel1, nonsubset = True)

                train_dic[method](model_objective, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate)

                # Determine the point with the highest observed function value
                best_idx_high = np.argmax(ytr[-1])
                best_y_high = ytr[-1][best_idx_high]

                if exp_config["Acq_function"] == "ES":
                        Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                            fidelity_num=total_fidelity_num,
                                                                            model_objective=model_objective,
                                                                            data_manager=fidelity_manager,
                                                                            model_cost=model_cost,
                                                                            seed=(seed + 1234 + i, i))
                        new_x, new_s = Acq_function.compute_next()
                elif exp_config["Acq_function"] == "UCB":
                        Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                            fidelity_num=total_fidelity_num,
                                                                            data_manager=fidelity_manager,
                                                                            posterior_function=model_objective.forward,
                                                                            model_cost=model_cost,
                                                                            seed=(seed + 1234 + i, i))
                        new_x, new_s = Acq_function.compute_next()
                elif exp_config["Acq_function"] == "EI":
                        np.random.seed(1028)
                        xall = np.random.rand(100)[:, None]

                        if method == 'AR':
                             model_objective_new = model_dic[method](fidelity_num=total_fidelity_num, kernel=kernel1, rho_init=1.0, nonsubset = True)
                        elif method in ['CIGAR', 'GAR']:
                            model_objective_new = model_dic[method](fidelity_num=total_fidelity_num, kernel=kernel1, l_shape=low_shape, h_shape=high_shape, nonsubset = True)
                        else:
                            model_objective_new = model_dic[method](fidelity_num=total_fidelity_num, kernel=kernel1, nonsubset = True)

                        Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                            fidelity_num=total_fidelity_num,
                                                                            posterior_function=model_objective.forward,
                                                                            model_objective_new=model_objective_new,
                                                                            data_name=data_name,
                                                                            target_func=data.get_data,
                                                                            cost_model=model_cost,
                                                                            seed= seed + i + 1234)

                        # new_x, new_s = Acq_function.compute_next(xtr, ytr)
                        new_x, new_s = Acq_function.compute_next(xtr, ytr, xall)
                elif exp_config["Acq_function"] == "cfKG":
                    if method == 'AR':
                        model_objective_new = model_dic[method](fidelity_num=total_fidelity_num, kernel=kernel1, rho_init=1.0, nonsubset = True)
                    elif method in ['CIGAR', 'GAR']:
                        model_objective_new = model_dic[method](fidelity_num=total_fidelity_num, kernel=kernel1, l_shape=low_shape, h_shape=high_shape, nonsubset = True)
                    else:
                        model_objective_new = model_dic[method](fidelity_num=total_fidelity_num, kernel=kernel1, nonsubset = True)
                    # train_dic[method](model_objective, fidelity_manager, max_iter=MF_iterations, lr_init=MF_learning_rate)
                    Acq_function = Acq_list[exp_config["Acq_function"]](posterior_function=model_objective.forward,
                                                                        model_objective_new=model_objective_new,
                                                                        train_function_new = train_dic[method],
                                                                        data_model=data,
                                                                        data_manager=fidelity_manager,
                                                                        model_cost=model_cost,
                                                                        total_fidelity_num=total_fidelity_num,
                                                                        seed=seed + i + 1234)
                    new_x, new_s = Acq_function.compute_next(xtr, ytr)
                    new_s = int(new_s[0][0])



                new_y = data.get_data(new_x, new_s)

                print("finish", i, "times optimization", new_x, new_s, new_y)
                xtr[new_s - 1] = torch.cat((xtr[new_s - 1], new_x), axis=0)
                ytr[new_s - 1] = torch.cat((ytr[new_s - 1], new_y), axis=0)
                T2 = time.time()

                recording["cost"].append(model_cost.compute_model_cost(ytr))
                recording["incumbents"].append(best_y_high.tolist()[0])
                recording["operation_time"].append(T2 - T1)

        return recording



if __name__ == '__main__':
    # realpath = os.path.abspath(__file__)
    # _sep = os.path.sep
    # realpath = realpath.split(_sep)
    # realpath = _sep.join(realpath[:realpath.index('MFBO') + 1])
    # sys.path.append(realpath)


    data_name = "forrester"
    for mf_model in ['AR']:
        for acq in ["ES"]:
            for seed in [0]:
                exp_config = {
                            'seed': seed,
                            'data_model': Data_list[data_name],
                            'MF_model': mf_model,
                            'Acq_function': acq,
                            'total_fidelity_num': 2,
                            'initial_index': {1: 10, 2: 4},
                            'BO_iterations': 10,
                            'MF_iterations': 20,
                            'MF_learning_rate': 0.01,
                    }

                record = MF_BO_discrete(exp_config)

                path_csv = os.path.join(sys.path[-1], 'Experiments', 'MFBO_discrete', 'exp_results', data_name)
                if not os.path.exists(path_csv):
                    os.makedirs(path_csv)

                df = pd.DataFrame(record)
                df.to_csv(path_csv + '/'+mf_model+'_' + exp_config['Acq_function'] + '_seed_' + str(seed) + '.csv',
                          index=False)