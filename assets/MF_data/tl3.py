# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:38:59 2022

@author: Lenovo
"""

from typing import Tuple

import numpy as np

from emukit.core.loop.user_function import MultiSourceFunctionWrapper
from emukit.core import ParameterSpace, ContinuousParameter, InformationSourceParameter

def test_function_d3() -> Tuple[MultiSourceFunctionWrapper, ParameterSpace]:
    def high(x):
        return x*np.sin(x)/10
    
    def low(x):
        return x*np.sin(x)/10+x/10
    
    space = ParameterSpace([ContinuousParameter('x1', .0, 10), InformationSourceParameter(1)])
    return MultiSourceFunctionWrapper([low, high]), space