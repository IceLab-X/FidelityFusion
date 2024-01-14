from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

PI = 3.14159



def multi_fidelity_maolin10() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', 0., 0.5), ContinuousParameter('x2', 0., 0.5),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        return ( (1-np.exp(-0.5/x2)) * (2300*x1**3 + 1900*x1**2 + 2092*x2 + 60)/(100*x1**3 + 500*x1**2 + 4*x2 + 20) )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        high1 = test_high(z+0.05).flatten()

        x1_new = (x1 + 0.05).reshape([-1, 1])
        x2_new = (x2 - 0.05).reshape([-1, 1])
        x2_new[x2_new<0] = 0
        z_new = np.hstack([x1_new, x2_new])

        high2 = test_high(z_new).flatten()

        x1_new = (x1 - 0.05).reshape([-1, 1])
        x2_new = (x2 + 0.05).reshape([-1, 1])
        z_new = np.hstack([x1_new, x2_new])

        high3 = test_high(z_new).flatten()

        x1_new = (x1 - 0.05).reshape([-1, 1])
        x2_new = (x2 - 0.05).reshape([-1, 1])
        x2_new[x2_new<0] = 0
        z_new = np.hstack([x1_new, x2_new])

        high4 = test_high(z_new).flatten()

        return ( -0.4 * high1 + (high2 + high3 + high4) / 4 )[:, None]

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
