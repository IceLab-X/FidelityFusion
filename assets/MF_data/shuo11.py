from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

PI = 3.14159



def multi_fidelity_shuo11() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -1., 1.), ContinuousParameter('x2', -1., 1.),
                            ContinuousParameter('x3', -1., 1.), InformationSourceParameter(2)])
    
    x_dim = 3

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        sum = 0
        for i in range(x_dim):
            xi = z[:, i]
            sum = sum + 0.3 * np.sin(16/15*xi-1) + ( np.sin(16/15*xi-1) )**2

        return ( sum )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]
        x3 = z[:, 2]

        sum = 0
        for i in range(x_dim):
            xi = z[:, i]
            sum = sum + 0.3 * np.sin(16/15*xi-1) + 0.2*(np.sin(16/15*xi-1))**2

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