# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter
from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from typing import Tuple



PI = 3.14159


def multi_fidelity_forrester_my(std=0) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    space = ParameterSpace([ContinuousParameter('x', 0, 1), InformationSourceParameter(2)])

    x_dim = 1

    def forrester_1(x, sd=std):
        """
            .. math::
        f(x) = (6x - 2)^2 \sin(12x - 4)
        """

        x = x.reshape((len(x), 1))
        n = x.shape[0]
        fval = ((6 * x - 2) ** 2) * np.sin(12 * x - 4)
        if sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, sd, n).reshape(n, 1)
        return fval.reshape(n, 1) + noise


    def forrester_2(x, sd=std):
        x = x.reshape((len(x), 1))
        n = x.shape[0]
        fval = ((5.5 * x - 2.5) ** 2) * np.sin(12 * x - 4)
        if sd == 0:
            noise = np.zeros(n).reshape(n, 1)
        else:
            noise = np.random.normal(0, sd, n).reshape(n, 1)
        return fval.reshape(n, 1) + noise


    def forrester_3(x, sd=std):
        high_fidelity = forrester_1(x, 0)
        return 0.75 * high_fidelity + 5 * (x[:, [0]] - 0.5) - 2 + np.random.randn(x.shape[0], 1) * sd


    def forrester_4(x, sd=std):
        """
            .. math::
        f_{low}(x) = 0.5 f_{high}(x) + 10 (x - 0.5) - 5
        """
        high_fidelity = forrester_1(x, 0)
        return 0.5 * high_fidelity + 10 * (x[:, [0]] - 0.5) - 5 + np.random.randn(x.shape[0], 1) * sd

    return MultiSourceFunctionWrapper([forrester_4, forrester_3, forrester_2, forrester_1]), space



