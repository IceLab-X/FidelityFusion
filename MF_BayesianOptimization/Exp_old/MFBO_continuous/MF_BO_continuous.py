import numpy as np
import pandas as pd
import sys
import time
import os


from MF_model.FiDEs import FIDES as FIDES
from Acquisition_Function.Continuous.MF_ES import entropy_search_continuous as ES
from Acquisition_Function.Continuous.MF_UCB import upper_confidence_bound_continuous as UCB
from Acquisition_Function.Continuous.CFKG import continuous_fidelity_knowledgement_gradient as CFKG

from Simulation.Synthetic_MF_Function.Branin import Branin
from Simulation.Synthetic_MF_Function.Hartmann import Hartmann

MF_model_list = {'fides': FIDES}
Acq_list = {'UCB': UCB, 'cfKG': CFKG, 'ES': ES}
Data_list = {'Branin': Branin, 'Hartmann': Hartmann}

def MF_BO_continuous(exp_config):
    seed = exp_config["seed"]

    '''Initiate Setting'''
    data_model = exp_config["data_model"]
    initial_num = exp_config['initial_num']
    BO_iterations = exp_config['BO_iterations']
    MF_iterations = exp_config['MF_iterations']
    MF_learning_rate = exp_config['MF_learning_rate']

    '''prepare initial data'''
    data = data_model()
    search_range = data.search_range
    xtr, ytr, s_index = data.Initiate_data(initial_num, seed)
    model_cost = data.cost

    recording = {"cost": [],
                 "incumbents": [],
                 "operation_time": []}

    for i in range(BO_iterations):
        print('iteration:', i + 1)
        T1 = time.time()
        mf_exp_config = {
            "seed": seed,
            "n_iterations": MF_iterations,
            "learning_rate": MF_learning_rate,
            "normal_y_mode": 0,
            "data_structure": xtr.shape[1],
            "log_beta": 0.1,
        }

        # Fit the Gaussian process model to the sampled points

        model_objective = MF_model_list[exp_config["MF_model"]](mf_exp_config)
        model_objective.train(xtr, ytr, s_index)

        if exp_config["Acq_function"] == "ES":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr.shape[1],
                                                                search_range=search_range,
                                                                model_objective=model_objective,
                                                                model_cost=model_cost,
                                                                seed=seed + i + 1234)
            new_x, new_s = Acq_function.compute_next()
        elif exp_config["Acq_function"] == "UCB":
            Acq_function = Acq_list[exp_config["Acq_function"]](x_dimension=xtr.shape[1],
                                                                search_range=search_range,
                                                                posterior_function=model_objective.predict,
                                                                model_cost=model_cost,
                                                                seed=[seed + i + 1234, i])
            new_x, new_s = Acq_function.compute_next()
        elif exp_config["Acq_function"] == "cfKG":
            mf_exp_config_new = {
                "seed": seed,
                "n_iterations": MF_iterations,
                "learning_rate": MF_learning_rate,
                "normal_y_mode": 0,
                "data_structure": xtr.shape[1],
                "log_beta": 0.0001,
            }
            model_objective_new = MF_model_list[exp_config["MF_model"]](mf_exp_config_new)
            Acq_function = Acq_list[exp_config["Acq_function"]](posterior_function=model_objective.predict,
                                                                model_objective_new=model_objective_new,
                                                                data_model=data,
                                                                model_cost=model_cost,
                                                                search_range=search_range,
                                                                seed=seed + i + 1234)
            new_x, new_s = Acq_function.compute_next(xtr, ytr, s_index)

        new_y = data.get_data(new_x, new_s)

        print("finish", i, "times optimization", new_x, new_s, new_y)

        xtr = np.concatenate((xtr, new_x), axis=0)
        ytr = np.concatenate((ytr, new_y), axis=0)
        new_s = np.array(new_s).reshape(1, 1)
        s_index = np.concatenate((s_index, new_s), axis=0)
        T2 = time.time()

        recording["cost"].append(model_cost.compute_model_cost(ytr, s_index))
        recording["incumbents"].append(max(ytr).tolist()[0])
        recording["operation_time"].append(T2 - T1)

    return recording


if __name__ == '__main__':
    realpath = os.path.abspath(__file__)
    _sep = os.path.sep
    realpath = realpath.split(_sep)
    realpath = _sep.join(realpath[:realpath.index('MFBO') + 1])
    sys.path.append(realpath)

    # "Branin", "Hartmann", "mln_mnist", "cnn_cifar"
    data_name = 'Hartmann'
    initial_num = 16
    for seed in [0]:
        for acq in ["UCB"]:
            exp_config = {
                'seed': seed,
                'data_model': Data_list[data_name],
                'MF_model': "fides",
                'Acq_function': acq,
                'initial_num': initial_num,
                'BO_iterations': 10,
                'MF_iterations': 10,
                'MF_learning_rate': 0.0001,
            }
            record = MF_BO_continuous(exp_config)

            path_csv = os.path.join(sys.path[-1], 'Experiment', 'Exp_continuous_1', 'Exp_results',
                                    data_name)
            if not os.path.exists(path_csv):
                os.makedirs(path_csv)

            df = pd.DataFrame(record)
            df.to_csv(path_csv + '/FiDEs_' + exp_config['Acq_function'] + '_seed_' + str(seed) + '.csv', index=False)