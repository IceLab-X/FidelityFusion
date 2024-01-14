# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:45:22 2022

@author: Lenovo
"""

from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter

def test_function_d5() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        x1=x[:,0]
        x2=x[:,1]
        s= 4*x1**2-2.1*x1**4+1/3*x1**6+x1*x2-4*x2**2+4*x2**4
        return s[:,None]
    
    def low(x):
        x1=x[:,0]
        x2=x[:,1]
        s= 2*x1**2-2.1*x1**4+1/3*x1**6+0.5*x1*x2-4*x2**2+2*x2**4
        return s[:,None]
    
    space = ParameterSpace([ContinuousParameter('x1', -2, 2), ContinuousParameter('x2', -2, 2),
                            InformationSourceParameter(2)])
    return MultiSourceFunctionWrapper([low, high]), space