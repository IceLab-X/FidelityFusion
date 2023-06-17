import torch
import time
import os
import shutil
from copy import deepcopy
import sys
import datetime


realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from utils import mlgp_result_record
from utils.dict_tools import smart_update
from utils.mlgp_hook import register_nan_hook
from utils.path_tools import get_available_name, deep_mkdir

default_controller_config = {
    'batch_size': 1, # not implement
    'check_point': [1, 10, 100, 300, 500, 1000, 1500,2500,3000,4000, 5000,6000,7000,8000,9000,10000],
    'eval_batch_size': 1, # not implement
    'record_step': 50,
    'max_epoch': 1000,
    'record_file_dir': './exp/'
}


class controller(object):
    def __init__(self, module, controller_config, module_config, demo_name=None) -> None:
        self.module_config = module_config
        self.controller_config = smart_update(default_controller_config, controller_config)
        exp_name = demo_name if demo_name is not None else module.__name__
        if self.controller_config['record_file_dir'] == default_controller_config['record_file_dir']:
            self.controller_config['record_file_dir'] = os.path.join(self.controller_config['record_file_dir'], exp_name, module_config['dataset']['name'], datetime.datetime.now().strftime('%Y-%m-%d'))
        deep_mkdir(self.controller_config['record_file_dir'])
        self.controller_config['record_file_path'] = get_available_name(self.controller_config['record_file_dir'], 'txt')

        self.module = module(module_config)
        if self.module.module_config.get('cuda',False):
            self.module.cuda()
            inputs_name = ['inputs_eval', 'inputs_tr', 'outputs_eval','outputs_tr']
            for _key in inputs_name:
                _p_list = getattr(self.module, _key)
                for i, _p in enumerate(_p_list):
                    _p_list[i] = _p.cuda()
        # register_nan_hook(self.module)

        self.rc_file = mlgp_result_record.MLGP_recorder(self.controller_config['record_file_path'],
                                                        overlap=True, 
                                                        append_info={
                                                            'module': str(module),
                                                            'controller_config': self.controller_config, 
                                                            'module_config':self.module.module_config
                                                            })
        if 'BayeSTA' in self.module.module_config and self.module.module_config['BayeSTA'] is True:
            BayeSTA_method = [_s + '_STA' for _s in self.module.module_config['evaluate_method']]
            self.rc_file.register(['epoch', *self.module.module_config['evaluate_method'], *BayeSTA_method,'time'])
        else:
            self.rc_file.register(['epoch', *self.module.module_config['evaluate_method'], 'time'])
        os.environ['mlgp_record_file'] = self.controller_config['record_file_path']

    def start_train(self):
        self.init_time = time.time()
        for i in range(self.controller_config['max_epoch']):
            os.environ['mlgp_epoch'] = str(i)
            self.module.train()

            print('train {}/{}'.format(i, self.controller_config['max_epoch']), end='\r')

            # when reach record step, store model
            # if i%self.controller_config['record_step'] == 0 and i!= 0:
                # self.record_state()
                # print('step: {} record state'.format(i))
                # _result = self.start_eval()

            # when reach check_point, evaluation
            if (i+1) in self.controller_config['check_point']:
                print("\nepoch: {}".format(i+1))
                _result = self.start_eval()
                _result['time'] = int(time.time() - self.init_time)
                _result['epoch'] = i+1
                self.rc_file.record(_result)


    def start_eval(self):
        result = self.module.eval()
        return result

    def record_state(self):
        pass

    def restore_state(self, index):
        pass

    def data_query(self):
        pass
        
    def clear_record(self):
        open('record.txt', 'w').close()