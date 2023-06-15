import torch
from modules.l2h_module.base_l2h_module import Basic_l2h
from utils import smart_update
from utils.type_define import *

import tensorly
tensorly.set_backend('pytorch')

def _smooth_mapping_matrix(_shape):
    if _shape[0] < _shape[1]:
        # up_sample
        _init_tensor = torch.ones(_shape[0], _shape[1])
        up_rate = _shape[1]/_shape[0]
        for i in range(_shape[0]):
            for j in range(_shape[1]):
                _init_tensor[i,j] /= ((i*up_rate - j)**2 + 1)
        _init_tensor = _init_tensor/ _init_tensor.sum(0, keepdim=True)

        # for tensorly multi-dot
        _init_tensor = _init_tensor.transpose(1,0)
    elif _shape[0] == _shape[1]:
        _init_tensor = torch.eye(_shape[0])
    else:
        # down_sample
        assert False, NotImplemented

    return _init_tensor


def _eye_distribution(_shape):
    if _shape[0] < _shape[1]:
        init_tensor = torch.eye(_shape[0])
        init_tensor = torch.nn.functional.interpolate(init_tensor.reshape(1, 1, *init_tensor.shape), _shape, mode='bilinear')
        init_tensor = init_tensor.squeeze().T
    elif _shape[0] == _shape[1]:
        init_tensor = torch.eye(_shape[0])
    return init_tensor


default_config = {
    'l_shape': None,
    'h_shape': None,
    'matrix_init_method': "smooth",     # smooth, eye

    'rho_value_init': 1.,
    'trainable_rho': False,
}


class Matrix_l2h(Basic_l2h):
    def __init__(self, config=None) -> None:
        super().__init__()
        self.config = smart_update(default_config, config)

        self.l_shape = self.config['l_shape']
        self.h_shape = self.config['h_shape']
        assert self.l_shape is not None and self.h_shape is not None, "l_shape and h_shape should be set"

        self.vectors = []
        for i in range(len(self.l_shape)):
            if self.config['matrix_init_method'] == 'smooth':
                _init_tensor = _smooth_mapping_matrix((self.l_shape[i], self.h_shape[i]))
            elif self.config['matrix_init_method'] == 'eye':
                _init_tensor = _eye_distribution((self.l_shape[i], self.h_shape[i]))
            self.vectors.append(torch.nn.Parameter(_init_tensor))

        self.rho = torch.nn.Parameter(torch.tensor(self.config['rho_value_init'], dtype=torch.float32))
        if not self.config['trainable_rho']:
            self.rho.requires_grad = False

    # train
    # inputs = [x, y_low]       ->  [x]
    # outputs = [y_high]        ->  [y_res]
    def pre_process_at_train(self, inputs, outputs):
        if isinstance(inputs[1], GP_val_with_var):
            x = inputs[0]
            y_low_mean = inputs[1].get_mean()
            y_low_var = inputs[1].get_var()
            y_low_var = y_low_var.sqrt()
            y_high = outputs[0]

            for i in range(len(self.l_shape)):
                y_low_mean = tensorly.tenalg.mode_dot(y_low_mean, self.vectors[i], i+1)
                y_low_var = tensorly.tenalg.mode_dot(y_low_var, self.vectors[i], i+1)
            y_low_var = y_low_var**2

            re_present_inputs = [x]
            if isinstance(y_high, GP_val_with_var):
                re_present_outputs_mean = [y_high.get_mean() - y_low*self.rho]
                re_present_outputs_var = [y_high.get_var() + y_low_var*self.rho]
            else:
                re_present_outputs_mean = [y_high - y_low_mean*self.rho]
                re_present_outputs_var = [torch.zeros_like(y_high) + y_low_var*self.rho]
            re_present_outputs = [GP_val_with_var(re_present_outputs_mean[0], re_present_outputs_var[0])]
            
        else:
            x = inputs[0]
            y_low = inputs[1]

            y_high = outputs[0]

            for i in range(len(self.l_shape)):
                y_low = tensorly.tenalg.mode_dot(y_low, self.vectors[i], i+1)

            re_present_inputs = [x]
            re_present_outputs = [y_high - y_low*self.rho]
        return re_present_inputs, re_present_outputs

    def pre_process_at_predict(self, inputs, outputs):
        x = inputs[0]
        y_low = inputs[1]

        return [inputs[0]], outputs

    
    # predict
    # inputs = [x, y_low]       ->  [x, y_low]
    # outputs = [y_res]         ->  [y_high]
    def post_process_at_predict(self, inputs, outputs):
        x = inputs[0]
        y_low = inputs[1]

        for i in range(len(self.l_shape)):
            y_low = tensorly.tenalg.mode_dot(y_low, self.vectors[i], i+1)

        res = outputs[0]

        # TODO: support res with var
        if isinstance(res, GP_val_with_var):
            mean = res.get_mean()
            vars = res.get_var()
            y_high_predict = GP_val_with_var(y_low*self.rho + mean, vars)
        else:
            y_high_predict = y_low*self.rho + res

        re_present_outputs = [y_high_predict]
        return inputs, re_present_outputs
    
    def get_train_params(self):
        return {'matrix': self.vectors, 'rho': self.rho}