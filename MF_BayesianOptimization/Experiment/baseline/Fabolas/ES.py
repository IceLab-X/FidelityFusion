# -*- coding = utf-8 -*-
# @Time : 26/9/23 11:06
# @Author : Alison_W
# @File : ES.py
# @Software : PyCharm

from typing import Callable, Union
import numpy as np
import scipy
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from emukit.bayesian_optimization import epmgp
from emukit.core.acquisition import Acquisition


from emukit.core.interfaces import IModel
from emukit.core.parameter_space import ParameterSpace
from emukit.samplers import AffineInvariantEnsembleSampler, McmcSampler
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.interfaces import IEntropySearchModel

class EntropySearch(Acquisition):
        def __init__(self, model_objective, x_range, search_range, seed):
            self.model_objective = model_objective
            self.x_range = x_range
            self.search_range = search_range
            self.seed = seed

            num_samples = 50
            # This is used later to calculate derivative of the stochastic part for the loss function
            # Derived following Ito's Lemma, see for example https://en.wikipedia.org/wiki/It%C3%B4%27s_lemma
            self.W = scipy.stats.norm.ppf(
                np.linspace(1.0 / (num_samples + 1), 1 - 1.0 / (num_samples + 1), num_samples))[
                     np.newaxis, :
                     ]

            # Initialize parameters to lazily compute them once needed
            self.representer_points = None
            self.representer_points_log = None
            self.logP = None
            self.p_min_entropy = None


        def update_parameters(self) -> None:
            """
            Updates p_min parameter
            """
            self.update_pmin()

        def update_pmin(self) -> np.ndarray:
            """
            Approximates the distribution of the global optimum  p(x=x_star|D) by doing the following steps:
                - discretizing the input space by representer points sampled from a proposal measure (default EI)
                - predicting mean and the covariance matrix of these representer points
                - uses EPMGP algorithm to compute the probability of each representer point being the minimum
            """
            N = 50
            x_dim = self.x_range.shape[1]
            # tem = []
            # for i in range(x_dim):
            #     np.random.seed(self.seed + 317 + i)
            #     tt = np.random.rand(N, 1) * (self.search_range[i][1] - self.search_range[i][0]) + \
            #          self.search_range[i][0]
            #     tem.append(tt)
            # tt = np.concatenate(tem, axis=1)
            # self.representer_points = tt
            self.representer_points = np.linspace(1, 2, N)[:, None]
            self.representer_points_log = np.log(self.representer_points)
            self.zz = np.linspace(self.search_range[-1][0], self.search_range[-1][1], N)[:, None]
            mu, var = self.model_objective.predict(torch.from_numpy(self.representer_points),
                                                   self.zz)
            mu = np.ndarray.flatten(mu.detach().numpy())
            var = (torch.eye(N) * var).detach().numpy()

            self.logP, self.dlogPdMu, self.dlogPdSigma, self.dlogPdMudMu = epmgp.joint_min(mu, var,
                                                                                           with_derivatives=True)
            self.logP = self.logP[:, np.newaxis]

            # Calculate the entropy of the distribution over the minimum given the current model
            self.p_min_entropy = np.sum(
                np.multiply(np.exp(self.logP), np.add(self.logP, self.representer_points_log)), axis=0
            )

            return self.logP

        def _required_parameters_initialized(self):
            """
            Checks if all required parameters are initialized.
            """
            return not (self.representer_points is None or self.representer_points_log is None or self.logP is None)

        def evaluate(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
            """
            Computes the information gain, i.e the change in entropy of p_min if we would evaluate x.

            :param x: points where the acquisition is evaluated.
            """
            if not self._required_parameters_initialized():
                self.update_pmin()

            # Check if we want to compute the acquisition function for multiple inputs
            if x.shape[0] > 1:
                results = np.zeros([x.shape[0], 1])
                for j in range(x.shape[0]):
                    results[j] = self.evaluate(x[j, None, :], z[j].reshape(1, 1))
                return results

            # Number of representer points locations
            N = self.logP.size

            # Evaluate innovations, i.e how much does mean and variance at the
            # representer points change if we would evaluate x
            dMdx, dVdx = self._innovations(x, z)

            dVdx = dVdx[np.triu(np.ones((N, N))).T.astype(bool), np.newaxis]

            dMdx_squared = dMdx.dot(dMdx.T)
            trace_term = np.sum(
                np.sum(
                    np.multiply(
                        self.dlogPdMudMu, np.reshape(dMdx_squared, (1, dMdx_squared.shape[0], dMdx_squared.shape[1]))
                    ),
                    2,
                ),
                1,
            )[:, np.newaxis]

            # Deterministic part of change:
            deterministic_change = self.dlogPdSigma.dot(dVdx) + 0.5 * trace_term
            # Stochastic part of change:
            stochastic_change = (self.dlogPdMu.dot(dMdx)).dot(self.W)

            # Update our pmin distribution
            predicted_logP = np.add(self.logP + deterministic_change, stochastic_change)
            max_predicted_logP = np.amax(predicted_logP, axis=0)

            # normalize predictions
            max_diff = max_predicted_logP + np.log(np.sum(np.exp(predicted_logP - max_predicted_logP), axis=0))
            lselP = max_predicted_logP if np.any(np.isinf(max_diff)) else max_diff
            predicted_logP = np.subtract(predicted_logP, lselP)

            # We maximize the information gain
            H_p = np.sum(np.multiply(np.exp(predicted_logP), np.add(predicted_logP, self.representer_points_log)),
                         axis=0)
            # H_p = np.sum(np.multiply(np.exp(predicted_logP), predicted_logP),
            #              axis=0)

            new_entropy = np.mean(H_p)
            entropy_change = new_entropy - self.p_min_entropy
            return entropy_change.reshape(-1, 1)

        def _innovations(self, x: np.ndarray, z: np.ndarray) -> tuple:
            """
            Computes the expected change in mean and variance at the representer
            points (cf. Section 2.4 in the paper).

            :param x: candidate for which to compute the expected change in the GP
            """

            # Get the standard deviation at x without noise
            m, v = self.model_objective.predict(torch.from_numpy(x), z)
            stdev_x = np.sqrt(v.detach().numpy())

            # Compute the variance between the test point x and the representer points
            tem = self.model_objective.kernel(torch.from_numpy(self.representer_points), self.zz,
                                              torch.from_numpy(x), z)
            sigma_x_rep = tem.detach().numpy()
            dm_rep = sigma_x_rep / stdev_x

            # Compute the deterministic innovation for the variance
            dv_rep = -dm_rep.dot(dm_rep.T)
            return dm_rep, dv_rep

        @property
        def has_gradients(self) -> bool:
            """Returns that this acquisition has gradients"""
            return False