from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

def multi_fidelity_test3_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Reference:

    R. Tuo, P. Z. Qian, and C. J. Wu, “Comment: A brownian motion model for stochastic simulation with tunable precision,” *Technometrics*, vol. 55, no. 1, pp. 29–31, 2013
    """

    def test_low(x):
        x1 = x[:, 0]

        return ( np.exp(1.4 * x1) * np.cos(3.5 * np.pi *x1) )[:, None]


    def test_high(x):
        x1 = x[:, 0]

        return ( np.exp(x1) * np.cos(x1) + 1 / (x1**2) )[:, None]


    space = ParameterSpace([ContinuousParameter('x1', 0., 1.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space
