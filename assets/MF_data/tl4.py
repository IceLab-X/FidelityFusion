# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:42:36 2022

@author: Lenovo
"""

from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter

def test_function_d4() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        return np.cos(3.5*np.pi*x)*np.exp(-1.4*x)
    
    def low(x):
        return np.cos(3.5*np.pi*x)*np.exp(-1.4*x)+0.75*x**2
    
    space = ParameterSpace([ContinuousParameter('x1', .0, 1), InformationSourceParameter(1)])
    return MultiSourceFunctionWrapper([low, high]), space