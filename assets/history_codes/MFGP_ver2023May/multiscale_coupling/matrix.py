import torch
from MFGP_ver2023May.utils.dict_tools import update_dict_with_default


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
    'low_fidelity_shape': None,
    'high_fidelity_shape': None,
    'matrix_init_method': "smooth",     # smooth, eye

    'rho_value_init': 1.,
    'trainable_rho': False,
}


class Matrix_Mapping(torch.nn.Module):
    def __init__(self, config=None) -> None:
        super().__init__()
        self.config = update_dict_with_default(default_config, config)

        self.l_shape = self.config['low_fidelity_shape']
        self.h_shape = self.config['high_fidelity_shape']
        assert self.l_shape is not None and self.h_shape is not None, "low_fidelity_shape and high_fidelity_shape should be set"

        self.vectors = []
        for i in range(len(self.l_shape)):
            if self.config['matrix_init_method'] == 'smooth':
                _init_tensor = _smooth_mapping_matrix((self.l_shape[i], self.h_shape[i]))
            elif self.config['matrix_init_method'] == 'eye':
                _init_tensor = _eye_distribution((self.l_shape[i], self.h_shape[i]))
            self.vectors.append(torch.nn.Parameter(_init_tensor))
        self.vectors = torch.nn.ParameterList(self.vectors)

        self.rho = torch.nn.Parameter(torch.tensor(self.config['rho_value_init'], dtype=torch.float32))
        if not self.config['trainable_rho']:
            self.rho.requires_grad = False

    def forward(self, low_fidelity, high_fidelity):
        for i in range(len(self.l_shape)):
            low_fidelity = tensorly.tenalg.mode_dot(low_fidelity, self.vectors[i], i+1)

        res = high_fidelity - low_fidelity*self.rho
        return res


    def backward(self, low_fidelity, res):
        for i in range(len(self.l_shape)):
            low_fidelity = tensorly.tenalg.mode_dot(low_fidelity, self.vectors[i], i+1)

        high_fidelity = low_fidelity*self.rho + res
        return high_fidelity
    
    def var_forward(self, low_fidelity_var, high_fidelity_var):
        return self.forward(low_fidelity_var, high_fidelity_var)
    
    def var_backward(self, low_fidelity_var, res_var):
        return self.backward(low_fidelity_var, res_var)

    
