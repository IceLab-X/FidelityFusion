from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace


def multi_fidelity_test4_function() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    r"""
    Reference:
    [33]D. Higdon, “Space and space-time modeling using process convolutions,” in *Quantitative methods for current environmental issues*.Springer, 2002, pp. 37–56.
    """

    def test_low(x):
        x1 = x[:, 0]
        return ( np.sin(2*np.pi*x1/10) + 0.2*np.sin(2*np.pi*x1/2.5) )[:, None]


    def test_high(x):
        x1 = x[:, 0]
        return ( np.sin(2*np.pi*x1/2.5) + np.cos(2*np.pi*x1/2.5) )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', 0., 10.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_high]), space
