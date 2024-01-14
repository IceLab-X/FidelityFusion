# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:33:52 2022

@author: Lenovo
"""

from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter


def test_function_d2() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        return np.sin(2*np.pi*(x-0.1))+x**2
    
    def low(x):
        return np.sin(2*np.pi*(x-0.1))
    space = ParameterSpace([ContinuousParameter('x1', 0, 1), InformationSourceParameter(1)])
    return MultiSourceFunctionWrapper([low, high]), space   