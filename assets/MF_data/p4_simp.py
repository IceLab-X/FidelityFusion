from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

def multi_fidelity_p4_simp(A=0) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -6., 5.), ContinuousParameter('x2', -6., 5.),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def sigmoid1(x):
        return 1 / ( 1 + np.exp( 32*(x+0.5)) )

    def test_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        sum = (x1**2+x2**2)/25 - np.cos(x1)*np.cos(x2/(2**0.5)) + 1
        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    def test_mid(x):

        x1 = x[:, 0]
        x2 = x[:, 1]
        sum = np.cos(x1)*np.cos(x2/(2**0.5)) + 1
        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    def test_low(x):

        x1 = x[:, 0]
        x2 = x[:, 1]

        sum = (x1**2+x2**2)/20 - np.cos(x1/(2**0.5)) * np.cos(x2/(3**0.5)) - 1
        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid1(x1)

        return ( sum )[:, None]

    return MultiSourceFunctionWrapper([test_low, test_mid, test_high]), space

if __name__ == "__main__":

    pass