# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:50:32 2022

@author: Lenovo
"""

from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter

def test_function_d8() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        x1=x[:,0]
        x2=x[:,1]
        s=(1-2*x1+0.05*np.sin(4*np.pi*x2-x1))**2+(x2-0.5*np.sin(2*np.pi*x1))**2
        return s[:,None]
    
    def low(x):
        x1=x[:,0]
        x2=x[:,1]
        s=(1-2*x1+0.05*np.sin(4*np.pi*x2-x1))**2+4*(x2-0.5*np.sin(2*np.pi*x1))**2
        return s[:,None]
    
    space = ParameterSpace([ContinuousParameter('x1', 0, 1), ContinuousParameter('x2', 0, 1),
                            InformationSourceParameter(2)])
    return MultiSourceFunctionWrapper([low, high]), space