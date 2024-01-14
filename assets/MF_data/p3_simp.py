from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

def multi_fidelity_p3_simp(A=0) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    def sigmoid1(x):
        return 1 / ( 1 + np.exp( 32*(x+0.5)) )

    def test_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        sum = 100*((x2 - x1**2)**2) + (1-x1)**2
        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    def test_mid(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        sum = 50*((x2 - x1**2)**2) + (-2-x1)**2 - 0.5*(x1+x2)

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    def test_low(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        high = test_high(x).flatten()
        sum = (high-4-0.5*(x1+x2))/(10+0.25*(x1+x2))

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    space = ParameterSpace([ContinuousParameter('x1', -2., 2.), ContinuousParameter('x2', -2., 2.),
                            InformationSourceParameter(2)])

    return MultiSourceFunctionWrapper([test_low, test_mid, test_high]), space

