import numpy as np
import torch
from ConfigSpace import Configuration, ConfigurationSpace, Float


from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
from smac import RunHistory, Scenario

import pandas as pd
import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from Data_simulation.Synthetic_MF_Function import *

# Cost_list = {'cost_10': cost_10, 'cost_pow_10': cost_pow_10}

'''single fidelity: fidelity = 2'''
class TargetFunction():
    def __init__(self, data_name):
        self.data_name = data_name
        if data_name == "Hartmann":
            self.x_range = [[0, 1] for i in range(6)]
            self.fidelity_range = (0, 1)
        elif data_name == "Branin":
            self.x_range = [[0, 1] for i in range(2)]
            self.fidelity_range = (0, 1)
        elif data_name == "Park":
            self.x_range = [[-1, 1] for i in range(2)]
            self.fidelity_range = (0, 1)
        elif data_name == "Currin":
            self.x_range = [[0, 1] for i in range(2)]
            self.fidelity_range = (0, 1)
        elif data_name == "non_linear_sin":
            self.x_range = [[0, 1.5]]
            self.fidelity_range = (0, 1)
        elif data_name == "forrester":
            self.x_range = [[0, 1.5]]
            self.fidelity_range = (0, 1)
        elif data_name == "mln_mnist":
            self.x_range = [[0, 1], [200, 250]]
            self.fidelity_range = [20, 30]
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        for i in range(len(self.x_range)):
            x = Float("x" + str(i + 1), self.x_range[i])
            cs.add_hyperparameters([x])

        s = Float("s", self.fidelity_range)
        cs.add_hyperparameters([s])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        x = [config["x" + str(i + 1)] for i in range(len(self.x_range))]
        s = config["s"]

        data_model = exp_config["data_model"]
        cost_type = exp_config['cost_type']
        data = data_model(cost_type, total_fidelity_num = 2)
        # return -gen_data(seed, self.data_name, np.array([x]), s)[0][0]
        # return data.get_cmf_data(torch.from_numpy(x)[0], torch.from_numpy(x)[1])
        return data.get_cmf_data(torch.tensor(x).reshape(1, -1), torch.tensor(s))


def plotandrecording(runhistory: RunHistory, incumbent: Configuration, model_cost, data_model):
    # plt.figure()

    # Plot ground truth
    # x = list(np.linspace(0, 1, 100))
    # for i in range(total_fidelity):
    #     y = [gen_data(0, 'non_linear_sin', np.array([[xi]]), i+1, total_fidelity)[0][0] for xi in x]
    #     plt.plot(x, y)

    # model_cost = Cost_list[exp_config['Cost_function']]([0, 1]) # Todo: model_cost 
    recording_dic = {"cost": [],
                     "incumbents": [],
                     "operation_time": []}

    # Plot all trials
    ytr = [np.empty(shape=[0, 1]), np.empty(shape=[0, 1])]
    for k, v in runhistory.items():
        config = runhistory.get_config(k.config_id)
        # x = config["x"]
        s = config["s"]
        x = [config["x" + str(i + 1)] for i in range(len(config._values)-1)]
        y = v.cost  # type: ignore
        # y = data_model.get_cmf_data(x, s)
        if s < float(config.config_space["s"].default_value):
            ss = 1
        else:
            ss = 2

        ytr[ss - 1] = np.concatenate((ytr[ss - 1], np.array([[-y]])), axis=0)
        time = v.time
        # plt.scatter(x, -y, c="blue", alpha=0.1, zorder=9999, marker="o")
        if len(ytr[-1]) == 0:
            continue
        else:
            m = max(ytr[-1])
            recording_dic["incumbents"].append(m[0])
            recording_dic["operation_time"].append(time)
            recording_dic["cost"].append(model_cost.compute_model_cost_smac(ytr).item())

    return recording_dic

    # Plot incumbent
    # plt.scatter(incumbent["x"], gen_data(0, 'non_linear_sin', np.array([[incumbent["x"]]]), incumbent["s"], total_fidelity)[0][0], c="red",
    #             zorder=10000, marker="x")

    # plt.show()


def SMAC_continuous(exp_config):
    seed = exp_config["seed"]

    '''Initiate Setting'''
    BO_iterations = exp_config['BO_iterations']

    data_name = exp_config["data_name"]
    model = TargetFunction(data_name)

    cost_type = exp_config['cost_type']
    data_model = exp_config["data_model"]
    data = data_model(cost_type, total_fidelity_num = 2)
    model_cost = data.cost

    # smac模型部分
    scenario = Scenario(model.configspace, deterministic=True, n_trials=BO_iterations)
    smac = HPOFacade(
        scenario,
        model.train,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
    )
    incumbent = smac.optimize()
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {-incumbent_cost}")
    recording_dic = plotandrecording(smac.runhistory, incumbent, model_cost, data_model)

    # 删掉产出文件
    path = sys.path[-1]
    shutil.rmtree(path + "/smac3_output")

    return recording_dic

if __name__ == "__main__":

    # data_name = 'Branin'
    Data_list = {'non_linear_sin': non_linear_sin, 'forrester': forrester, 'Park': Park, 'Branin': Branin, 'Currin': Currin}
    data_name = 'Currin'
    for seed in [0]:
        exp_config = {
            'seed': seed,
            'data_name': data_name,
            'data_model': Data_list[data_name],
            'cost_type': "pow_10",
            'initial_index': {1: 10, 2: 4},
            'BO_iterations': 100,
        }

        record = SMAC_continuous(exp_config)

        # path_csv = os.path.join(sys.path[-1], 'exp', 'continuous', data_name, 'smac')
        path_csv = sys.path[-1] + '/Experiment/CMF/Exp_results/' + data_name + '/' + exp_config['cost_type'] +'/'


        df = pd.DataFrame(record)
        df.to_csv(path_csv + '/smac_seed_' + str(seed) + '.csv',
                index=False)


