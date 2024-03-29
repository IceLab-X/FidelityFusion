# -*- coding = utf-8 -*-
# @Time : 26/9/23 16:40
# @Author : Alison_W
# @File : fabolas_demo.py
# @Software : PyCharm
import json
import logging
import numpy as np
import pandas as pd
import time
import os
import george
import sys
from Simulation.Synthetic_MF_Function.gen_data_discrete import gen_data
from Simulation.Cost_Function.cost10 import cost
from robo.acquisition_functions.ei import EI
from robo.acquisition_functions.information_gain_per_unit_cost import InformationGainPerUnitCost
from robo.acquisition_functions.marginalization import MarginalizationGPMCMC
from robo.maximizers.random_sampling import RandomSampling
from robo.models.fabolas_gp import FabolasGPMCMC
from robo.priors.env_priors import EnvPrior
from robo.util.incumbent_estimation import projected_incumbent_estimation


def fabolas(data_name, N_iteration):
    path_csv = os.path.join('../exp', data_name, os.path.split(sys.argv[0])[-1][:-8])
    total_fidelity_num = 2
    seed = 0

    model_cost_exp = cost([0, 1])

    def transform(s, s_min, s_max):
        s_transform = (np.log2(s) - np.log2(s_min)) / (np.log2(s_max) - np.log2(s_min))
        return s_transform

    def retransform(s_transform, s_min, s_max):
        s = np.rint(2 ** (s_transform * (np.log2(s_max) - np.log2(s_min)) + np.log2(s_min)))
        return int(s)

    '''fabolas 有不同的数据组成结构'''

    x = np.random.rand(32)[:, None]
    s_min = 1
    s_max = 5
    s_low = transform(2, s_min, s_max)
    s_high = transform(4, s_min, s_max)

    ss_low = retransform(s_low, s_min, s_max)
    ss_high = retransform(s_high, s_min, s_max)

    xtr_low = np.concatenate((x, s_low * np.ones(32)[:, None]), axis=1)
    xtr_high = np.concatenate((x, s_high * np.ones(32)[:, None]), axis=1)

    ytr_low = gen_data(seed, data_name, x, 1, total_fidelity_num)
    ytr_high = gen_data(seed, data_name, x, 2, total_fidelity_num)


    '''进入fabolas的循环'''
    logger = logging.getLogger(__name__)
    lower = np.array([0])
    upper = np.array([1])
    n_dims = lower.shape[0]
    n_hypers = 1
    rng = None
    time_start = time.time()
    if rng is None:
        rng = np.random.RandomState(np.random.randint(0, 10000))

    # Bookkeeping
    time_func_eval = []
    time_overhead = []
    incumbents = []
    runtime = []

    X = np.concatenate((xtr_low, xtr_high), axis=0)
    y = np.concatenate((ytr_low, ytr_high), axis=0).flatten()
    c = np.concatenate((1 * np.ones(32)[:, None], 5 * np.ones(32)[:, None]), axis=0).flatten()
    num_iterations = X.shape[0] + N_iteration

    recording = {"cost": [model_cost_exp.compute_model_cost_fabolas(X, y)],
                 "incumbents": [max(y)],
                 "operation_time": [0]}

    # Define model for the objective function
    cov_amp = 1  # Covariance amplitude
    kernel = cov_amp

    # ARD Kernel for the configuration space
    for d in range(n_dims):
        kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                ndim=n_dims + 1, axes=d)

    env_kernel = george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                   ndim=n_dims + 1,
                                                   axes=n_dims)

    kernel *= env_kernel

    # Take 3 times more samples than we have hyperparameters
    if n_hypers < 2 * len(kernel):
        n_hypers = 3 * len(kernel)
        if n_hypers % 2 == 1:
            n_hypers += 1

    prior = EnvPrior(len(kernel) + 1,
                         n_ls=n_dims,
                         n_lr=2,
                         rng=rng)

    quadratic_bf = lambda x: (1 - x) ** 2
    linear_bf = lambda x: x

    model_objective = FabolasGPMCMC(kernel,
                                    prior=prior,
                                    burnin_steps=100,
                                    chain_length=100,
                                    n_hypers=n_hypers,
                                    normalize_output=False,
                                    basis_func=quadratic_bf,
                                    lower=lower,
                                    upper=upper,
                                    rng=rng)

    # Define model for the cost function
    cost_cov_amp = 1

    cost_kernel = cost_cov_amp

    # ARD Kernel for the configuration space
    for d in range(n_dims):
        cost_kernel *= george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                     ndim=n_dims + 1, axes=d)

    cost_env_kernel = george.kernels.Matern52Kernel(np.ones([1]) * 0.01,
                                                    ndim=n_dims + 1,
                                                    axes=n_dims)
    cost_kernel *= cost_env_kernel

    cost_prior = EnvPrior(len(cost_kernel) + 1,
                          n_ls=n_dims,
                          n_lr=2,
                          rng=rng)

    model_cost = FabolasGPMCMC(cost_kernel,
                               prior=cost_prior,
                               burnin_steps=100,
                               chain_length=100,
                               n_hypers=n_hypers,
                               basis_func=linear_bf,
                               normalize_output=False,
                               lower=lower,
                               upper=upper,
                               rng=rng)

    # Extend input space by task variable
    extend_lower = np.append(lower, 0)
    extend_upper = np.append(upper, 1)
    is_env = np.zeros(extend_lower.shape[0])
    is_env[-1] = 1

    # Define acquisition function and maximizer
    ig = InformationGainPerUnitCost(model_objective,
                                    model_cost,
                                    extend_lower,
                                    extend_upper,
                                    sampling_acquisition=EI,
                                    is_env_variable=is_env,
                                    n_representer=50)
    acquisition_func = MarginalizationGPMCMC(ig)
    maximizer = RandomSampling(acquisition_func, extend_lower, extend_upper)
    inc_estimation = "mean"

    output_path = path_csv
    count = 0
    for it in range(X.shape[0], num_iterations):
        logger.info("Start iteration %d ... ", it)

        start_time = time.time()

        # Train models
        model_objective.train(X, y, do_optimize=True)
        model_cost.train(X, c, do_optimize=True)

        if inc_estimation == "last_seen":
            # Estimate incumbent as the best observed value so far
            best_idx = np.argmin(y)
            incumbent = X[best_idx][:-1]
            incumbent = np.append(incumbent, 1)
            incumbent_value = y[best_idx]
        else:
            # Estimate incumbent by projecting all observed points to the task of interest and
            # pick the point with the lowest mean prediction
            incumbent, incumbent_value = projected_incumbent_estimation(model_objective, X[:, :-1],
                                                                        proj_value=1)
        incumbents.append(incumbent[:-1])
        logger.info("Current incumbent %s with estimated performance %f",
                    str(incumbent), np.exp(incumbent_value))

        # Maximize acquisition function
        acquisition_func.update(model_objective, model_cost)
        new_x = maximizer.maximize()

        s = retransform(new_x[-1], s_min, s_max)  # Map s from log space to original linear space

        time_overhead.append(time.time() - start_time)
        logger.info("Optimization overhead was %f seconds", time_overhead[-1])

        # Evaluate the chosen configuration
        logger.info("Evaluate candidate %s on subset size %f", str(new_x[:-1]), s)
        start_time = time.time()
        if s > 2.5:
            s = 2
            new_c = 5
        else:
            s = 1
            new_c = 1
        new_y = gen_data(seed, 'non_linear_sin', new_x[:-1], s, total_fidelity_num)
        time_func_eval.append(time.time() - start_time)

        logger.info("Configuration achieved a performance of %f with cost %f", new_y, new_c)
        logger.info("Evaluation of this configuration took %f seconds", time_func_eval[-1])

        # Add new observation to the data
        X = np.concatenate((X, new_x[None, :]), axis=0)
        y = np.concatenate((y, new_y.flatten()), axis=0)  # Model the target function on a logarithmic scale
        c = np.concatenate((c, np.array([new_c])), axis=0)  # Model the cost function on a logarithmic scale

        runtime.append(time.time() - time_start)
        recording["cost"].append(model_cost_exp.compute_model_cost_fabolas(X, y))
        recording["incumbents"].append(incumbents[count].tolist()[0])
        recording["operation_time"].append(time.time() - time_start)


        if output_path is not None:
            data = dict()
            data["optimization_overhead"] = time_overhead[count]
            data["runtime"] = runtime[count]
            data["incumbent"] = incumbents[count].tolist()
            data["time_func_eval"] = time_func_eval[count]
            data["iteration"] = count
            count += 1
            json.dump(data, open(os.path.join(output_path, "fabolas_iter_%d.json" % it), "w"))

    # Estimate the final incumbent
    model_objective.train(X, y, do_optimize=True)
    incumbent, incumbent_value = projected_incumbent_estimation(model_objective, X[:, :-1],
                                                                proj_value=1)
    logger.info("Final incumbent %s with estimated performance %f",
                str(incumbent), incumbent_value)

    results = dict()
    results["x_opt"] = incumbent[:-1].tolist()
    results["incumbents"] = [inc.tolist() for inc in incumbents]
    results["runtime"] = runtime
    results["overhead"] = time_overhead
    results["time_func_eval"] = time_func_eval
    results["X"] = [x.tolist() for x in X]
    results["y"] = [np.exp(yi).tolist() for yi in y]
    results["c"] = [ci.tolist() for ci in c]

    if not os.path.exists(path_csv):
        os.makedirs(path_csv)

    df = pd.DataFrame(recording)  # 数据初始化成为DataFrame对象
    df.to_csv(path_csv + '/demo.csv', index=False)  # 将数据写入

if __name__ == '__main__':
    data_name = 'non_linear_sin'
    fabolas(data_name, 20)