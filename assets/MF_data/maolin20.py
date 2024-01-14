from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

PI = 3.14159


def multi_fidelity_maolin20() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.), ContinuousParameter('x2', 0., 1.),
                            ContinuousParameter('x3', 0., 1.), ContinuousParameter('x4', 0., 1.),
                            ContinuousParameter('x5', 0., 1.), ContinuousParameter('x6', 0., 1.),
                            ContinuousParameter('x7', 0., 1.), ContinuousParameter('x8', 0., 1.),
                            InformationSourceParameter(2)])
    
    x_dim = 8

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        sum = 0
        for i in [3,4,5,6,7]:
            sum_temp = 0

            for j in np.arange(2,i+1):
                x_j = z[:, j]
                sum_temp = sum_temp + x_j

            sum = sum + (i+1) * np.log(1+sum_temp)

        sum = sum * 16 * ((x3+1)**0.5) * (2*x3-1)**2

        return ( 4*(x1-2+8*x2+8*x2**2)**2 + (3-4*x2)**2 +sum )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        sum = 0
        for i in [3,4,5,6,7]:
            sum_temp = 0

            for j in np.arange(2,i+1):
                x_j = z[:, j]
                sum_temp = sum_temp + x_j

            sum = sum + np.log(1+sum_temp)

        sum = sum * 16 * ((x3+1)**0.5) * ((2*x3-1)**2)

        return ( 4*(x1-2+8*x2+8*x2**2)**2 + (3-4*x2)**2 + sum )[:, None]

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
