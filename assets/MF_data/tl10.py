# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:00:06 2022

@author: Lenovo
"""

from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter

def test_function_d10() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        Sum=0
        for i in range(8):
            Sum+=x[:,i]**4-16*x[:,i]**2+5*x[:,i]
        return Sum[:,None]
    
    def low(x):
        Sum=0
        for i in range(8):
            Sum+=0.3*x[:,i]**4-16*x[:,i]**2+5*x[:,i]
        return Sum[:,None]
    
    space = ParameterSpace([ContinuousParameter('x1', -3, 3), 
                            ContinuousParameter('x2', -3, 3),
                            ContinuousParameter('x3', -3, 3),
                            ContinuousParameter('x4', -3, 3),
                            ContinuousParameter('x5', -3, 3),
                            ContinuousParameter('x6', -3, 3),
                            ContinuousParameter('x7', -3, 3),
                            ContinuousParameter('x8', -3, 3),
                            InformationSourceParameter(8)])
    return MultiSourceFunctionWrapper([low, high]), space