from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

PI = 3.14159


def multi_fidelity_Toal(A=0.5) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -100., 100.), ContinuousParameter('x2', -100., 100.),
                            ContinuousParameter('x3', -100., 100.), ContinuousParameter('x4', -100., 100.),
                            ContinuousParameter('x5', -100., 100.), ContinuousParameter('x6', -100., 100.),
                            ContinuousParameter('x7', -100., 100.), ContinuousParameter('x8', -100., 100.),
                            ContinuousParameter('x9', -100., 100.), ContinuousParameter('x10', -100., 100.),
                            InformationSourceParameter(2)])
    
    x_dim = 10

    def test_high(z):

        sum1 = 0

        for i in range(x_dim):
            x_now = z[:, i]
            sum1 = sum1 + (x_now-1)**2

        sum2 = 0

        for i in np.arange(1, x_dim):
            x_now = z[:, i]
            x_last = z[:, i-1]
            sum2 = sum2 + x_last * x_now

        return (sum1-sum2)[:, None]

    def test_low(z):

        sum1 = 0

        for i in range(x_dim):
            x_now = z[:, i]
            sum1 = sum1 + (x_now-A)**2

        sum2 = 0

        for i in np.arange(1, x_dim):
            x_now = z[:, i]
            x_last = z[:, i-1]
            sum2 = sum2 + x_last * x_now * i * (A-0.65)

        return (sum1-sum2)[:, None]

    return MultiSourceFunctionWrapper([test_low, test_high]), space

    

# if __name__ == "__main__":
#     fcn, new_space = multi_fidelity_p1_simp()
#     from Code.Pakage.emukit.core.initial_designs import LatinDesign
#     latin = LatinDesign(new_space)
#
#     xtr = latin.get_samples(point_count=100)
#     Ytr = []
#
#     fidelity = 3
#
#     for i in range(fidelity):
#         Ytr.append(fcn.f[i](xtr))
#         print(f"f{i+1}:{Ytr[i].shape}")
#
#     print(xtr.shape)
