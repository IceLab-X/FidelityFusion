from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace


def multi_fidelity_test5_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Reference:
    [34]X. Cai, H. Qiu, L. Gao, and X. Shao, “Metamodeling for high dimensional design problems by multi-fifidelity simulations,” *Structural and**Multidisciplinary Optimization*, vol. 56, no. 1, pp. 151–166, 2017.
    """

    def test_low(x):
        high = test_high(0.7 * x).flatten()
        x1 = x[:, 0]
        x2 = x[:, 1]
        return ( high + x1*x2 - 65 )[:, None]


    def test_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return ( 4*(x1**2) - 2.1*(x1**4) + (x1**6)/3 - 4*(x2**2) + 4*x2**4 + x1*x2 )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', -2., 2.), ContinuousParameter('x2', -2., 2.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space
