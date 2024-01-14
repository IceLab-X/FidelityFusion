from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace


def multi_fidelity_test7_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Reference:
    [36]J. An and A. Owen, “Quasi-regression,” *Journal of complexity*, vol. 17, no. 4, pp. 588–607, 2001.
    """

    def test_low(x):
        Xs = []
        for i in range(8):
            Xs.append(x[:,i])

        x4_sum = 0
        for i in range(4):
            x4_sum = x4_sum +Xs[i]

        res = 0
        for i in range(4,8):
            res = res + Xs[i]*np.cos(x4_sum) + Xs[i]*np.sin(x4_sum)

        return res[:, None]


    def test_high(x):
        Xs = []
        for i in range(8):
            Xs.append(x[:,i])

        x4_sum = 0
        for i in range(4):
            x4_sum = x4_sum +Xs[i]

        res_cos = 0
        for i in range(4,8):
            res_cos = res_cos + Xs[i]*np.cos(x4_sum)

        res_sin = 0
        for i in range(4,8):
            res_sin = res_sin + Xs[i]*np.sin(x4_sum)

        return ( (res_sin**2 + res_cos**2)**0.5 )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', 0., 2.*np.pi),ContinuousParameter('x2', 0., 2.*np.pi),ContinuousParameter('x3', 0., 2.*np.pi),ContinuousParameter('x4', 0., 2.*np.pi),
                            ContinuousParameter('x5', 0., 1.),ContinuousParameter('x6', 0., 1.),ContinuousParameter('x7', 0., 1.),ContinuousParameter('x8', 0., 1.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space
