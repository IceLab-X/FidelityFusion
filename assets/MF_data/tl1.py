# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 09:49:10 2022

@author: Lenovo
"""

from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter


def test_function_d1() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        return (6*x-2)**2*np.sin(12*x-4)
    
    def low(x):
        return 0.56*((6*x-2)**2*np.sin(12*x-4))+10*(x-0.5)-5
    
    
    space = ParameterSpace([ContinuousParameter('x1', .0, 1.0), InformationSourceParameter(1)])
    return MultiSourceFunctionWrapper([low, high]), space    
