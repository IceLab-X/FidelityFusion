from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

PI = 3.14159



def multi_fidelity_shuo6() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -5., 10.), ContinuousParameter('x2', 0., 15.),
                            InformationSourceParameter(2)])
    
    x_dim = 2

    def test_high(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        term1 = (x2 - 1.275 * (x1/PI)**2 + 5*x1/PI - 6)**2
        term2 = 10 * (1 - 1/8/PI) * np.cos(x1)

        return ( term1 + term2 )[:, None]

    def test_low(z):
        x1 = z[:, 0]
        x2 = z[:, 1]

        term1 = 0.5 * (x2 - 1.275 * (x1/PI)**2 + 5*x1/PI - 6)**2
        term2 = 10 * (1 - 1/8/PI) * np.cos(x1)

        return ( term1 + term2 )[:, None]

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
