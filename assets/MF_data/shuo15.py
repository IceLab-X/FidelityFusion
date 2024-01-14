from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

PI = 3.14159



def multi_fidelity_shuo15() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', 0., 1.), ContinuousParameter('x2', 0., 1.),
                            ContinuousParameter('x3', 0., 1.), ContinuousParameter('x4', 0., 1.),
                            ContinuousParameter('x5', 0., 1.), ContinuousParameter('x6', 0., 1.),
                            ContinuousParameter('x7', 0., 1.), ContinuousParameter('x8', 0., 1.),
                            InformationSourceParameter(2)])
    
    x_dim = 8

    def test_high(z):

        sum = 0
        for i in [1, 2]:
            term1 = (z[:,(4*i-3)-1] + 10*z[:,(4*i-2)-1])**2
            term2 = 5*(z[:, (4*i-1)-1] - z[:, (4*i)-1])**2
            term3 = (z[:, (4*i-2)-1] - 2*z[:, (4*i-1)-1])**4
            term4 = 10*(z[:, (4*i-3)-1] - z[:, (4*i)-1])**4
            sum = sum + term1 + term2 + term3 + term4


        return ( sum )[:, None]

    def test_low(z):

        sum = 0
        for i in [1, 2]:
            term1 = (z[:,(4*i-3)-1] + 10*z[:,(4*i-2)-1])**2
            term2 = 125*(z[:, (4*i-1)-1] - z[:, (4*i)-1])**2
            term3 = (z[:, (4*i-2)-1] - 2*z[:, (4*i-1)-1])**4
            term4 = 10*(z[:, (4*i-3)-1] - z[:, (4*i)-1])**4
            sum = sum + term1 + term2 + term3 + term4

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
