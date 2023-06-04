import os
import sys

import matplotlib.pyplot as plt
import numpy as np

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.mlgp_log import mlgp_log
from utils.mlgp_result_record import MLGP_record_parser


if __name__ == '__main__':
    test_file = './record_ref.txt'

    _parser = MLGP_record_parser(test_file)
    data = _parser.get_data()

    for i in range(len(data)):
        result = data[i]['@record_result@']
        
        epoch_index = result[0].index('epoch')
        epoch = [_l[epoch_index] for _l in result[1:]]

        rmse_index = result[0].index('mae')
        rmse = [_l[rmse_index] for _l in result[1:]]

        _module = data[i]['@append_info@']['module']
        _seed = data[i]['@append_info@']['module_config']['dataset']['seed']

        plt.plot(epoch, rmse)
        plt.xlabel('epoch')
        plt.ylabel('mae')
        plt.title(_module+'\n'+ 'seed:{}'.format(_seed))
        plt.show()


    