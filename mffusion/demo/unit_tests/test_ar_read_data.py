#   coding:utf-8
#   This file is part of MF_Fusion.

__author__ = 'Guanjie Wang'
__email__ = "gjwang.buaa@gmail.edu.cn"
__version__ = 1.0
__init_date__ = '2023/06/15 09:48:05'
__maintainer__ = 'Guanjie Wang'
__update_date__ = '2023/06/15 09:48:05'

import os
import unittest

import numpy as np
import torch


def judge_type(condition1):
    if isinstance(condition1, np.ndarray):
        return condition1.tolist()
    elif isinstance(condition1, torch.Tensor):
        return judge_type(condition1.numpy())
    elif isinstance(condition1, tuple):
        return list(condition1)
    elif isinstance(condition1, list):
        return condition1
    else:
        raise TypeError(''.join(["Type error: %s" % type(condition1),
                                 'Only support ndarray, tensor, list and tuple.']))


class CustomTestCase(unittest.TestCase):
    
    def assertTensorNumpyListTrue(self, condition1, condition2):
        condition1 = judge_type(condition1)
        condition2 = judge_type(condition2)
        message = "%s != %s" % (str(condition1), str(condition2))
        if condition1 != condition2:
            self.fail(message)


class TestReadData(CustomTestCase):
    
    def setUp(self):
        head_data_dir = lambda dfp: os.path.join('..', '..', 'data', 'sample', dfp)
        self.x = np.load(head_data_dir('input.npy'))
        self.y_low = np.load(head_data_dir('output_fidelity_1.npy'))
        self.y_high = np.load(head_data_dir('output_fidelity_2.npy'))
        
    def test_origin_data(self):
        self.assertTensorNumpyListTrue(self.x.shape, (256, 5))
        self.assertTensorNumpyListTrue(self.y_low.shape, (256, 100, 100))
        self.assertTensorNumpyListTrue(self.y_high.shape, (256, 100, 100))
    
    def test_train_data(self):
        x = torch.tensor(self.x).float()
        y_low = torch.tensor(self.y_low).float()
        y_high = torch.tensor(self.y_high).float()
    
        data_len = x.shape[0]
        self.assertTrue(data_len == 256)
        
        source_shape = [-1, *y_low.shape[1:]]
        self.assertTensorNumpyListTrue(source_shape, [-1, 100, 100])
        
        x = x.reshape(data_len, -1)
        self.assertTensorNumpyListTrue(x.shape, [256, 5])
        
        y_low = y_low.reshape(data_len, -1)
        self.assertTensorNumpyListTrue(y_low.shape, [256, 100*100])
        y_high = y_high.reshape(data_len, -1)
        self.assertTensorNumpyListTrue(y_high.shape, [256, 100*100])
        # train_inputs = [x[:128, :], y_low[:128, :]]
        # train_outputs = [y_high[:128, :]]
        # eval_inputs = [x[128:, :], y_low[128:, :]]
        # eval_outputs = [y_high[128:, :]]


if __name__ == '__main__':
    # testsuite = unittest.TestLoader().discover('tests')
    # unittest.TextTestRunner(verbosity=3).run(testsuite)
    unittest.main(verbosity=3)
