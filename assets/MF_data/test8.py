from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace


def multi_fidelity_test8_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Reference:
    [20]X. Meng and G. E. Karniadakis, “A composite neural network that learns from multi-fifidelity data: Application to function approximationand inverse pde problems,” *Journal of Computational Physics*, vol. 401, p. 109020, 2020.

    """

    def test_low(x):
        x_high = test_high(x).flatten()

        Xs = []
        for i in range(20):
            Xs.append(x[:,i])

        res= 0
        for i in range(0,19):
            res = res + ( 0.4*Xs[i]*Xs[i+1] )

        return ( 0.8*x_high - res - 50 )[:, None]


    def test_high(x):
        Xs = []
        for i in range(20):
            Xs.append(x[:,i])

        res= 0
        for i in range(1,20):
            res = res + ( 2*Xs[i]**2 - Xs[i-1] )**2

        return ( res + Xs[0]**2 )[:, None]


    space_list = []
    for i in range(1,21):
        space_list.append(ContinuousParameter('x'+str(i), -3., 3.))
    space_list.append(InformationSourceParameter(2))

    space = ParameterSpace(space_list)
    # space = ParameterSpace([ContinuousParameter('x1', 0., 2.*np.pi),ContinuousParameter('x2', 0., 2.*np.pi),ContinuousParameter('x3', 0., 2.*np.pi),ContinuousParameter('x4', 0., 2.*np.pi),
    #                         ContinuousParameter('x5', 0., 1.),ContinuousParameter('x6', 0., 1.),ContinuousParameter('x7', 0., 1.),ContinuousParameter('x8', 0., 1.),
    #                         InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space
