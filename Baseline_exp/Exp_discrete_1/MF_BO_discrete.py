import numpy as np
import pandas as pd
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))


# from MF_model.FiDEs_discrete import FIDES
from MF_model.ResGP import ResGP
from MF_model.AR import AR
from Acquisition_Function.Discrete.MF_EI import expected_improvement as EI
from Acquisition_Function.Discrete.MF_ES import entropy_search as ES
from Acquisition_Function.Discrete.MF_UCB_optimize import upper_confidence_bound as UCB
from Acquisition_Function.Discrete.CFKG import discrete_fidelity_knowledgement_gradient as CFKG

from Simulation.Synthetic_MF_Function.Non_linear_sin import non_linear_sin
from Simulation.Synthetic_MF_Function.Forrester import forrester

MF_model_list = {'FiDEs': FIDES, 'resgp': ResGP, 'ar': AR}
Acq_list = {'UCB': UCB, 'ES': ES, 'EI': EI, 'cfKG': CFKG}
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

        model_cost = data.cost
        recording = {"cost": [model_cost.compute_model_cost(ytr)],
                     "incumbents": [max(ytr[1].tolist()[0])],
                     "operation_time": [float(0)]}

        for i in range(BO_iterations):
                print('iteration:', i + 1)
                T1 = time.time()
                mf_exp_config = {
                    "fidelity_num": total_fidelity_num,
                    "seed": seed,
                    "n_iterations": MF_iterations,
                    "learning_rate": MF_learning_rate,
                    "normal_y_mode": 0,
                    "data_structure": xtr[0].shape[1]
                }

                # Fit the Gaussian process model to the sampled points

                model_objective = MF_model_list[exp_config["MF_model"]](mf_exp_config)
                model_objective.train(xtr, ytr)

                # Determine the point with the highest observed function value
                best_idx_high = np.argmax(ytr[-1])
                best_y_high = ytr[-1][best_idx_high]

                if exp_config["Acq_function"] == "ES":
                        Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                            fidelity_num=total_fidelity_num,
                                                                            model_objective=model_objective,
                                                                            model_cost=model_cost,
                                                                            seed=(seed + 1234 + i, i))
                        new_x, new_s = Acq_function.compute_next()
                elif exp_config["Acq_function"] == "UCB":
                        Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                            fidelity_num=total_fidelity_num,
                                                                            posterior_function=model_objective.predict,
                                                                            model_cost=model_cost,
                                                                            seed=(seed + 1234 + i, i))
                        new_x, new_s = Acq_function.compute_next()
                elif exp_config["Acq_function"] == "EI":
                        np.random.seed(1028)
                        xall = np.random.rand(100)[:, None]
                        # yall = gen_data(seed, data_name, xall, total_fidelity_num, total_fidelity_num)
                        model_objective_new = MF_model_list[exp_config["MF_model"]](mf_exp_config)
                        Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr[0].shape[1],
                                                                            fidelity_num=total_fidelity_num,
                                                                            posterior_function=model_objective.predict,
                                                                            model_objective_new=model_objective_new,
                                                                            data_name=data_name,
                                                                            target_func=data.get_data,
                                                                            cost_model=model_cost,
                                                                            seed= seed + i + 1234)

                        # new_x, new_s = Acq_function.compute_next(xtr, ytr)
                        new_x, new_s = Acq_function.compute_next(xtr, ytr, xall)
                elif exp_config["Acq_function"] == "cfKG":
                    mf_exp_config_new = {
                        "fidelity_num": total_fidelity_num,
                        "seed": seed,
                        "n_iterations": MF_iterations,
                        "learning_rate": MF_learning_rate,
                        "normal_y_mode": 0,
                        "data_structure": xtr[0].shape[1]
                    }
                    model_objective_new = MF_model_list[exp_config["MF_model"]](mf_exp_config_new)
                    Acq_function = Acq_list[exp_config["Acq_function"]](posterior_function=model_objective.predict,
                                                                        model_objective_new=model_objective_new,
                                                                        data_model=data,
                                                                        model_cost=model_cost,
                                                                        total_fidelity_num=total_fidelity_num,
                                                                        seed=seed + i + 1234)
                    new_x, new_s = Acq_function.compute_next(xtr, ytr)
                    new_s = int(new_s[0][0])



                new_y = data.get_data(new_x, new_s)

                print("finish", i, "times optimization", new_x, new_s, new_y)
                xtr[new_s - 1] = np.concatenate((xtr[new_s - 1], new_x), axis=0)
                ytr[new_s - 1] = np.concatenate((ytr[new_s - 1], new_y), axis=0)
                T2 = time.time()

                recording["cost"].append(model_cost.compute_model_cost(ytr))
                recording["incumbents"].append(best_y_high.tolist()[0])
                recording["operation_time"].append(T2 - T1)

        return recording



if __name__ == '__main__':
    realpath = os.path.abspath(__file__)
    _sep = os.path.sep
    realpath = realpath.split(_sep)
    realpath = _sep.join(realpath[:realpath.index('MFBO') + 1])
    sys.path.append(realpath)

    data_name = "forrester"
    for mf_model in ["FiDEs"]:
        for acq in ["cfKG", "UCB", "ES", "EI"]:
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

                path_csv = os.path.join(sys.path[-1], 'Experiment', 'Exp_discrete_1', 'Exp_results',
                                        data_name)
                if not os.path.exists(path_csv):
                    os.makedirs(path_csv)

                df = pd.DataFrame(record)
                df.to_csv(path_csv + '/'+mf_model+'_' + exp_config['Acq_function'] + '_seed_' + str(seed) + '.csv',
                          index=False)