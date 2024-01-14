from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

def multi_fidelity_p2_simp(A=0) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    def sigmoid2(x):
        return 1 / ( 1 + np.exp( -32*(x+0.5)) )

    def test_high(x):
        x1 = x[:, 0]
        sum = np.sin(30*((x1-0.9)**4)) * np.cos(2*(x1-0.9)) + (x1-0.9)/2

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid2(x1)

        return ( sum )[:, None]

    def test_mid(x):
        x1 = x[:, 0]
        high = test_high(x).flatten()
        sum = ( high - 1 + x1 ) / ( 1 + 0.25*x1 )

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid2(x1)

        return (sum)[:, None]

    def test_low(x):
        x1 = x[:, 0]
        sum = np.sin(20*((x1-0.87)**4)) * np.cos(2*(x1-0.87)) + (x1-0.87)/2 - (2.5-(0.7*x1-0.14)**2) + 2*x1

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid2(x1)

        return ( sum )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_mid, test_high]), space

