import random
import torch
import numpy as np
from copy import deepcopy

fidelity_map = {
    'low': 0,
    'medium': 1,
    'high': 2
}

# fidelity order should be low -> high
preprocess_default_config_dict = {
    'seed': None,

    # define sample select
    'train_start_index': 0, 
    'train_sample': 8, 
    'eval_start_index': 0, 
    'eval_sample':256,
    
    # define multi fidelity input/output format
    'inputs_format': ['x[0]'],
    'outputs_format': ['y[0]', 'y[2]'],

    # others
    'force_2d': False,
    'x_sample_to_last_dim': False,
    'y_sample_to_last_dim': False,

    # now only available for dataset, which not seperate train and test before
    'slice_param': [0.6, 0.4],
}

def _get_format_slice_data(length, slice):
    if isinstance(slice[0], int):
        return slice[0], slice[1]

    elif isinstance(slice[0], float):
        assert slice[0] + slice[1] == 1, 'slice sum should be 1'
        _tr = int(length * slice[0])
        _te = int(length * slice[1])
        while _tr + _te > length:
            _tr -= 1
            _te -= 1
        return _tr, _te

def _flatten_inputs(inputs):
    _len_list = [len(_l) for _l in inputs]
    _temp_array_list = []
    [_temp_array_list.extend(deepcopy(_l)) for _l in inputs]
    return _temp_array_list, _len_list

def _reformat_inputs(array_list, _len_list):
    _index = 0
    outputs = []
    for _len in _len_list:
        outputs.append(array_list[_index:_index+_len])
        _index += _len
    return outputs

def _last_dim_to_fist(_tensor):
    _dim = [i for i in range(_tensor.ndim)]
    _dim.insert(0, _dim.pop())
    if isinstance(_tensor, torch.Tensor):
        return _tensor.permute(*_dim)
    elif isinstance(_tensor, np.ndarray):
        return _tensor.transpose(*_dim)
    else:
        assert False, '_tensor should be torch.Tensor or np.ndarray'

def _first_dim_to_last(_tensor):
    _dim = [i+1 for i in range(_tensor.ndim-1)]
    _dim.append(0)
    if isinstance(_tensor, torch.Tensor):
        return _tensor.permute(*_dim)
    elif isinstance(_tensor, np.ndarray):
        return _tensor.transpose(*_dim)
    else:
        assert False, '_tensor should be torch.Tensor or np.ndarray'


class Data_preprocess(object):
    # --------------------------------------------------
    # input data format:
    #           [x_train, y_train, x_eval, y_eval]
    # output data format:
    #           [x_train, y_train, x_eval, y_eval]
    # --------------------------------------------------
    def __init__(self, config_dict):
        default_config = deepcopy(preprocess_default_config_dict)
        default_config.update(config_dict)
        self.config_dict = default_config

    def do_preprocess(self, inputs, numpy_to_tensor=False):
        out = inputs
        if inputs[2] is None and inputs[3] is None:
            out = self._seperate_to_gen_eval_data(out)

        if self.config_dict['seed'] is not None:
            out = self._random_shuffle(out)

        out = self._get_sample(out)
        if self.config_dict['force_2d'] is True:
            out = self._force_2d(out)

        if self.config_dict['x_sample_to_last_dim'] is True:
            out[0] = [_first_dim_to_last(_array) for _array in out[0]]
            out[2] = [_first_dim_to_last(_array) for _array in out[2]]
        if self.config_dict['y_sample_to_last_dim'] is True:
            out[1] = [_first_dim_to_last(_array) for _array in out[1]]
            out[3] = [_first_dim_to_last(_array) for _array in out[3]]

        out = self._get_want_format(out)
        
        if numpy_to_tensor is True:
            out = self._numpy_to_tensor(out)
        return out

    def _seperate_to_gen_eval_data(self, inputs):
        outputs = []
        total_sample = inputs[0][0].shape[0]
        sample_tr, sample_te = _get_format_slice_data(total_sample, self.config_dict['slice_param'])
        outputs.append([_array[:sample_tr, ...] for _array in inputs[0]])
        outputs.append([_array[:sample_tr, ...] for _array in inputs[1]])
        outputs.append([_array[sample_tr:sample_tr+sample_te, ...] for _array in inputs[0]])
        outputs.append([_array[sample_tr:sample_tr+sample_te, ...] for _array in inputs[1]])
        return outputs

    def _numpy_to_tensor(self, inputs):
        _temp_array_list,_len_list = _flatten_inputs(inputs)
        _temp_array_list = [torch.from_numpy(_array).float() for _array in _temp_array_list]
        outputs = _reformat_inputs(_temp_array_list, _len_list)
        return outputs

    def _force_2d(self, inputs):
        _temp_array_list,_len_list = _flatten_inputs(inputs)
        _temp_array_list = [_array.reshape(_array.shape[0], -1) for _array in _temp_array_list]
        outputs = _reformat_inputs(_temp_array_list, _len_list)
        return outputs

    def _get_want_format(self, inputs):
        outputs = []
        x = deepcopy(inputs[0])
        y = deepcopy(inputs[1])
        
        _temp_list = []
        for _cmd in self.config_dict['inputs_format']:
            _temp_list.append(eval(_cmd))
        outputs.append(_temp_list) # inputs_tr
        
        _temp_list = []
        for _cmd in self.config_dict['outputs_format']:
            _temp_list.append(eval(_cmd))
        outputs.append(_temp_list) # outputs_tr

        x = deepcopy(inputs[2])
        y = deepcopy(inputs[3])
        _temp_list = []
        for _cmd in self.config_dict['inputs_format']:
            _temp_list.append(eval(_cmd))
        outputs.append(_temp_list) # inputs_eval
        
        _temp_list = []
        for _cmd in self.config_dict['outputs_format']:
            _temp_list.append(eval(_cmd))
        outputs.append(_temp_list) # outputs_eval
        return outputs

    def _get_sample(self, inputs):
        outputs = []
        for i in range(2):
            _a = self.config_dict['train_start_index']
            _b = self.config_dict['train_sample']
            _temp_list = [_array[_a:_a+_b,...] for _array in inputs[i]]
            outputs.append(_temp_list)

        for i in range(2,4):
            _a = self.config_dict['eval_start_index']
            _b = self.config_dict['eval_sample']
            _temp_list = [_array[_a:_a+_b,...] for _array in inputs[i]]
            outputs.append(_temp_list)
        return outputs

    def _random_shuffle(self, inputs):
        _temp_array_list,_len_list = _flatten_inputs(inputs[0:2])
        _temp_array_list = self._random_shuffle_array_list(_temp_array_list)
        outputs = _reformat_inputs(_temp_array_list, _len_list)
        outputs.extend(inputs[2:4])
        return outputs

    def _random_shuffle_array_list(self, np_array_list):
        # sample should on first dim
        random.seed(self.config_dict['seed'])

        dim_lenth = []
        for _np_array in np_array_list:
            dim_lenth.append(_np_array.shape[0])
        assert len(set(dim_lenth)) == 1, "length of dim is not the same"

        shuffle_index = [i for i in range(dim_lenth[0])]
        random.shuffle(shuffle_index)

        output_array_list = []
        for _np_array in np_array_list:
            output_array_list.append(_np_array[shuffle_index])
        return output_array_list


if __name__ == '__main__':
    from data_loader import Standard_mat_DataLoader
    Stand_data = Standard_mat_DataLoader('poisson_v4_02')
    data = Stand_data.get_data()

    data_preprocess = Data_preprocess({})
    data = data_preprocess.do_preprocess(data)