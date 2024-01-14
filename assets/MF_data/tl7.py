# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:45:29 2022

@author: Lenovo
"""

from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter

def test_function_d7() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        x1=x[:,0]
        x2=x[:,1]
        s= x1**4+x2**4-16*x1**2-16*x2**2+5*x1+5*x2
        return s[:,None]
    
    def low(x):
        x1=x[:,0]
        x2=x[:,1]
        s= x1**4+x2**4-16*x1**2-16*x2**2
        return s[:,None]
    
    space = ParameterSpace([ContinuousParameter('x1', -3, 4), ContinuousParameter('x2', -3, 4),
                            InformationSourceParameter(2)])
    return MultiSourceFunctionWrapper([low, high]), space