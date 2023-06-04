import os
import sys


realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils.mlgp_log import mlgp_log
from utils.data_utils.data_preprocess import Data_preprocess
from utils.data_utils.data_loader import SP_DataLoader, Standard_mat_DataLoader, Custom_mat_DataLoader


def data_regist(module_calsee, dataset_config, cuda=False):
    mlgp_log.i('dataset_config name:', dataset_config['name'])
    loaded = False
    for _loader in [SP_DataLoader, Standard_mat_DataLoader, Custom_mat_DataLoader]:
        if dataset_config['name'] in _loader.dataset_available:
            _data_loader = _loader(dataset_config['name'], dataset_config['interp_data'])
            _data = _data_loader.get_data()
            loaded = True
            break
    if loaded is False:
        mlgp_log.e('dataset {} not found in all loader'.format(dataset_config['name']))

    dp = Data_preprocess(dataset_config)
    module_calsee.inputs_tr, module_calsee.outputs_tr, module_calsee.inputs_eval, module_calsee.outputs_eval = dp.do_preprocess(_data, numpy_to_tensor=True)
    
    if cuda is True:
        for _params in [module_calsee.inputs_tr, module_calsee.outputs_tr, module_calsee.inputs_eval, module_calsee.outputs_eval]:
            for i, _p in enumerate(_params):
                _params[i] = _p.cuda()