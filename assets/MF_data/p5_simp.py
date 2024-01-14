from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

PI = 3.14159

def multi_fidelity_p5_simp(A=0) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -0.1, 0.2), ContinuousParameter('x2', -0.1, 0.2),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def sigmoid3(x):
        return 1 / ( 1 + np.exp( -128*(x-0.05)) )

    def theta(fai):
        return 1-0.0001*fai

    def error(x, fai):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x_list = [x1, x2]
        sum = 0
        thetares = theta(fai)
        for x_now in x_list:
            sum = sum + thetares * np.cos( 10*PI*thetares*x_now + 0.5*PI*thetares + PI )**2

        return sum

    def test_1(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x_list = [x1, x2]

        sum = 0
        for x_now in x_list:
            sum = sum + x_now**2 + 1 - np.cos(10*PI*x_now)

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid3(x1)

        return ( sum )[:, None]

    def test_2(x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        high = test_1(x).flatten()
        err = error(x, fai=5000)
        sum = high + err

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid3(x1)

        return (sum)[:, None]

    def test_3(x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        high = test_1(x).flatten()
        err = error(x, fai=2500)
        sum = high + err

        R = np.max(sum) - np.min(sum)
        noise = np.random.normal(loc=0, scale=A*R, size=sum.shape[0])
        sum = sum + noise * sigmoid3(x1)

        return (sum)[:, None]

    return MultiSourceFunctionWrapper([test_3, test_2, test_1]), space
