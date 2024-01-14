from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

PI = 3.14159


def multi_fidelity_maolin19() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -5., 10.), ContinuousParameter('x2', -5., 10.),
                            ContinuousParameter('x3', -5., 10.), ContinuousParameter('x4', -5., 10.),
                            ContinuousParameter('x5', -5., 10.), ContinuousParameter('x6', -5., 10.),
                            InformationSourceParameter(2)])
    
    x_dim = 6

    def test_high(z):
        sum = 0
        for i in range(x_dim-1):
            x_now = z[:,i]
            x_next = z[:,i+1]
            sum = sum + 100*(x_next - x_now**2)**2 + (x_now-1)**2

        return ( sum )[:, None]

    def test_low(z):
        sum = 0
        for i in range(x_dim-1):
            x_now = z[:,i]
            x_next = z[:,i+1]
            sum = sum + 100*(x_next - x_now)**2 + 4*(x_now-1)**4

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
