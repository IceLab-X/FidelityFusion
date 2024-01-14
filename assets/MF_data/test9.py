from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace


def multi_fidelity_test9_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Reference:
    [37]G. H. Cheng, A. Younis, K. Haji Hajikolaei, and G. Gary Wang, “Trust region based mode pursuing sampling method for global optimization of high dimensional design problems,” *Journal of Mechanical Design*, vol. 137, no. 2, 2015.
    """

    def test_low(x):
        x_high = test_high(x).flatten()

        Xs = []
        for i in range(30):
            Xs.append(x[:, i])

        return ( x_high**3 + x_high**2 + x_high )[:, None]


    def test_high(x):
        Xs = []
        for i in range(30):
            Xs.append(x[:,i])

        res= 0
        for i in range(0,29):
            res = res + (30-(i+1)) * (Xs[i]**2 - Xs[i+1])**2

        return ( (Xs[0]-1)**2 + (Xs[29]-1)**2 + 30*res )[:, None]


    space_list = []
    for i in range(1,31):
        space_list.append(ContinuousParameter('x'+str(i), -3., 2.))
    space_list.append(InformationSourceParameter(2))

    space = ParameterSpace(space_list)
    # space = ParameterSpace([ContinuousParameter('x1', 0., 2.*np.pi),ContinuousParameter('x2', 0., 2.*np.pi),ContinuousParameter('x3', 0., 2.*np.pi),ContinuousParameter('x4', 0., 2.*np.pi),
    #                         ContinuousParameter('x5', 0., 1.),ContinuousParameter('x6', 0., 1.),ContinuousParameter('x7', 0., 1.),ContinuousParameter('x8', 0., 1.),
    #                         InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space
