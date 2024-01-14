from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ContinuousParameter, InformationSourceParameter, ParameterSpace

PI = 3.14159

def multi_fidelity_Colville(A=0.5) -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:

    space = ParameterSpace([ContinuousParameter('x1', -1., 1.), ContinuousParameter('x2', -1., 1.),
                            ContinuousParameter('x3', -1., 1.), ContinuousParameter('x4', -1., 1.),
                            InformationSourceParameter(2)])
    
    x_dim = 4

    def test_high(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        # x_list = [x1, x2]

        return ( 100*(x1**2-x2)**2 + (x1-1)**2 + (x3-1)**2 + 90*(x3**2-x4)
                 + 10.1*((x2-1)**2+(x4-1)**2) + 19.8*(x2-1)*(x4-1) )[:, None]

    def test_low(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        # x_list = [x1, x2]

        high = test_high(A * A * x).flatten()

        return ( high - (A+0.5) * (5*x1**2 + 4*x2**2 + 3*x3**2 + x4**2) )[:, None]

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
