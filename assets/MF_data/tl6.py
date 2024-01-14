# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:55:54 2022

@author: Lenovo
"""

from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter

def test_function_d6() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        x1=x[:,0]
        x2=x[:,1]
        s= 1/6*((30+5*x1*np.sin(5*x1))*(4+np.exp(-5*x2))-100)
        return s[:,None]
    
    def low(x):
        x1=x[:,0]
        x2=x[:,1]
        s= 1/6*((30+5*x1*np.sin(5*x1))*(4+2/5*np.exp(-5*x2))-100)
        return s[:,None]
    
    space = ParameterSpace([ContinuousParameter('x1', 0, 1), ContinuousParameter('x2', 0, 1),
                            InformationSourceParameter(2)])
    return MultiSourceFunctionWrapper([low, high]), space