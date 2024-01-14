# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:54:20 2022

@author: Lenovo
"""

from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter

def test_function_d9() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        x1=x[:,0]
        x2=x[:,1]
        x3=x[:,2]
        s= (x1-1)**2+(x1-x2)**2+x2*x3+0.5
        return s[:,None]
    
    def low(x):
        x1=x[:,0]
        x2=x[:,1]
        x3=x[:,2]
        s= 0.2*((x1-1)**2+(x1-x2)**2+x2*x3+0.5) -0.5*x1-0.2*x1*x2-0.1
        return s[:,None]
    
    space = ParameterSpace([ContinuousParameter('x1', 0, 1), ContinuousParameter('x2', 0, 1),
                            ContinuousParameter('x3', 0, 1),
                            InformationSourceParameter(3)])
    return MultiSourceFunctionWrapper([low, high]), space