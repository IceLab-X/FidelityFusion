from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

PI = 3.14159



def multi_fidelity_shuo16() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -2., 3.), ContinuousParameter('x2', -2., 3.),
                            ContinuousParameter('x3', -2., 3.), ContinuousParameter('x4', -2., 3.),
                            ContinuousParameter('x5', -2., 3.), ContinuousParameter('x6', -2., 3.),
                            ContinuousParameter('x7', -2., 3.), ContinuousParameter('x8', -2., 3.),
                            ContinuousParameter('x9', -2., 3.), ContinuousParameter('x10', -2., 3.),
                            InformationSourceParameter(2)])
    
    x_dim = 10

    def test_high(z):
        A = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.662, -22.179]

        sum = 0
        for i in range(x_dim):

            sum_temp = 0
            for k in range(x_dim):
                sum_temp = sum_temp + np.exp(z[:, k])

            sum = sum + np.exp(z[:,i]) * ( A[i] + z[:,i] - np.log(sum_temp) )

        return ( sum )[:, None]

    def test_low(z):
        B = [-10, -10, -20, -10, -20, -20, -20, -10, -20, -20]
        sum = 0
        for i in range(x_dim):

            sum_temp = 0
            for k in range(x_dim):
                sum_temp = sum_temp + np.exp(z[:, k])

            sum = sum + np.exp(z[:,i]) * ( B[i] + z[:,i] - np.log(sum_temp) )

        return ( sum )[:, None]

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
