#   coding:utf-8
#   This file is part of MF_Fusion.

__author__ = 'Guanjie Wang'
__email__ = "gjwang.buaa@gmail.edu.cn"
__version__ = 1.0
__init_date__ = '2023/06/15 09:48:05'
__maintainer__ = 'Guanjie Wang'
__update_date__ = '2023/06/15 09:48:05'


import unittest
from copy import deepcopy

import torch

from ..subset_tools import shuffle, Subset_check


class CustomTestCase(unittest.TestCase):
    def assertConditonAnyTrue(self, condition):
        message = "%.any() != True" % str(condition)
        if not condition.any():
            self.fail(message)
    
    def assertConditonAllTrue(self, condition):
        message = "%s.all() != True" % str(condition)
        if not condition.all():
            self.fail(message)

    
class TestSubset(CustomTestCase):
    
    def setUp(self):
        torch.manual_seed(0)
        
        # # subset 不能处理重复项
        # _x = torch.randn(25, 50, 50)
        # self.x = torch.cat((_x, _x, _x, _x), dim=0)
        # print(self.x.shape)
        
        self.x = torch.randn(100, 50, 50)
        
    def run_data(self, _tt):
        if _tt == 'numpy':
            self.x = self.x.numpy()
        shuffle_set = list(range(self.x.shape[0]))
        shuffle(shuffle_set)
        set_0 = deepcopy(shuffle_set[0:20])
        set_1 = deepcopy(shuffle_set[10:40])
        shuffle(set_1)
    
        x_0 = self.x[set_0]
        x_1 = self.x[set_1]
    
        sc = Subset_check(x_0)
        x_0_repeat_index, x_1_repeat_index = sc.get_subset(x_1, subset_type='index')
        diff = (x_0[x_0_repeat_index] - x_1[x_1_repeat_index])
        return diff
    
    def test_torch(self):
        diff = self.run_data(_tt='torch')
        self.assertConditonAllTrue(diff == 0)
        
    def test_numpy(self):
        diff = self.run_data(_tt='numpy')
        self.assertConditonAllTrue(diff == 0)


if __name__ == '__main__':
    unittest.main()
