from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace


def multi_fidelity_test6_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Reference:
    [35] R. B. Gramacy and H. K. Lee, “Adaptive design and analysis of supercomputer experiments,” *Technometrics*, vol. 51, no. 2, pp. 130–145, 2009.
    """

    def test_low(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]
        x6 = x[:, 5]
        return ( 100*np.exp(np.sin(x1)) + 5*x2*x3 + x4 + np.exp(x5*x6) )[:, None]


    def test_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]
        x6 = x[:, 5]
        return ( np.exp( np.sin( (0.9*x1+0.9*0.48)**10 ) ) + x2*x3 + x4 )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.),ContinuousParameter('x2', 0., 1.),ContinuousParameter('x3', 0., 1.),
                            ContinuousParameter('x4', 0., 1.),ContinuousParameter('x5', 0., 1.),ContinuousParameter('x6', 0., 1.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space
