import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import RunHistory, Scenario

from Simulation.Synthetic_MF_Function.Non_linear_sin import non_linear_sin
from Simulation.Synthetic_MF_Function.Forrester import forrester

import pandas as pd
import sys
import os
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.chdir(sys.path[-1])

'''single fidelity: fidelity = 2'''
class NonlinearSinFunction():
    def __init__(self, total_fidelity):
        self.total_fidelity = total_fidelity
        self.data_model = non_linear_sin(total_fidelity)

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x = Float("x", (0, 1))
        s = Integer("s", (1, 2))
        cs.add_hyperparameters([x,s])
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        x = config["x"]
        s = config["s"]
        return -self.data_model.get_data(np.array([[x]]), s)[0][0]

class ForresterFunction():
    def __init__(self, total_fidelity):
        self.total_fidelity = total_fidelity
        self.data_model = forrester(total_fidelity)
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x = Float("x", (0, 1))
        s = Integer("s", (1, 2))
        cs.add_hyperparameters([x,s])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        x = config["x"]
        s = config["s"]
        return -self.data_model.get_data(np.array([[x]]), s)[0][0]


def plotandrecording(runhistory: RunHistory, incumbent: Configuration, total_fidelity, model_cost, data_name):
    # plt.figure()

    # Plot ground truth
    # x = list(np.linspace(0, 1, 100))
    # for i in range(total_fidelity):
    #     y = [gen_data(0, 'non_linear_sin', np.array([[xi]]), i+1, total_fidelity)[0][0] for xi in x]
    #     plt.plot(x, y)

    # model_cost =  # Todo: model_cost
    recording_dic = {"cost": [],
                     "incumbents": [],
                     "operation_time": []}

    # Plot all trials
    ytr = [np.empty(shape=[0, 1]), np.empty(shape=[0, 1])]
    known = model_cost.compute_index(exp_config["initial_index"])
    for k, v in runhistory.items():
        config = runhistory.get_config(k.config_id)
        x = config["x"]
        s = config["s"]
        y = v.cost  # type: ignore
        ytr[s - 1] = np.concatenate((ytr[s - 1], np.array([[-y]])), axis=0)
        time = v.time
        # plt.scatter(x, -y, c="blue", alpha=0.1, zorder=9999, marker="o")
        if len(ytr[-1]) == 0:
            continue
        else:
            m = max(ytr[-1])
            recording_dic["incumbents"].append(m[0])
            recording_dic["operation_time"].append(time)
            recording_dic["cost"].append(model_cost.compute_model_cost(ytr)+known)

    return recording_dic

    # Plot incumbent
    # plt.scatter(incumbent["x"], gen_data(0, 'non_linear_sin', np.array([[incumbent["x"]]]), incumbent["s"], total_fidelity)[0][0], c="red",
    #             zorder=10000, marker="x")

    # plt.show()


def SMAC_discrete(exp_config):
    seed = exp_config["seed"]

    '''Initiate Setting'''
    total_fidelity = exp_config['total_fidelity_num']
    BO_iterations = exp_config['BO_iterations']

    data_name = exp_config["data_name"]
    if data_name == "non_linear_sin":
        model = NonlinearSinFunction(total_fidelity)
    elif data_name == "forrester":
        model = ForresterFunction(total_fidelity)

    model_cost = model.data_model.cost

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
    recording_dic = plotandrecording(smac.runhistory, incumbent, total_fidelity, model_cost, data_name)

    # 删掉产出文件
    path = sys.path[-1]
    shutil.rmtree(path + "/smac3_output")

    return recording_dic

if __name__ == "__main__":

    data_name = 'non_linear_sin'
    # data_name = 'forrester'
    for seed in [0]:
        exp_config = {
            'seed': seed,
            'total_fidelity_num': 2,
            'data_name': data_name,
            'initial_index': {1: 10, 2: 4},
            'BO_iterations': 10,
        }

        record = SMAC_discrete(exp_config)

        path_csv = os.path.join(sys.path[-1], 'Experiment', 'Exp_discrete_1', 'Exp_results',
                                data_name)
        if not os.path.exists(path_csv):
            os.makedirs(path_csv)

        df = pd.DataFrame(record)
        df.to_csv(path_csv + '/smac_seed_' + str(seed) + '.csv',
                  index=False)


