from asyncio.streams import FlowControlMixin
import torch
import os
import tensorly
tensorly.backend.set_backend('pytorch')

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

class rho_connection(torch.nn.Module):
    # rho connection between yl/yh
    def __init__(self, rho=1., trainable=True) -> None:
        super().__init__()
        self.trainable = trainable
        if trainable is True:
            self.rho = torch.nn.Parameter(torch.tensor(rho))
        else:
            self.rho = torch.tensor(rho)

    def forward(self, yl, yh):
        res = yh - yl*self.rho
        return res

    def low_2_high(self, yl, res):
        yh = yl*self.rho + res
        return yh

    def low_2_high_double_mapping(self, yl_var, res):
        yh = yl_var*self.rho + res
        return yh


    def high_2_low(self, yh, res):
        yl = (yh - res)/self.rho
        return yl

    def get_param(self, optimize_param_list):
        if self.trainable is True:
            optimize_param_list.append(self.rho)
        return optimize_param_list

    def set_param(self, param_list):
        if self.trainable is True:
            with torch.no_grad():
                self.rho.copy_(param_list[0])

    def get_params_need_check(self):
        return None


class mapping_connection(torch.nn.Module):
    # mapping connection between yl/yh
    def __init__(self, yl_shape, yh_shape, distribution_name='smooth_mapping_matrix',  sample_last_dim=True):
        # assume 10,20 -> 35, 50
        super().__init__()
        assert len(yl_shape) == len(yh_shape), "yl and yh should have same dims, but got {} and {}".format(yl_shape, yh_shape)
        
        self.yl_shape = yl_shape
        self.yh_shape = yh_shape
        self.sample_last_dim = sample_last_dim

        self.mapping_list = []
        for i in range(len(yl_shape)):
            if distribution_name == 'smooth_mapping_matrix':
                _init_tensor = _smooth_mapping_matrix((yl_shape[i], yh_shape[i]))
            elif distribution_name == 'eye':
                _init_tensor = _eye_distribution((yl_shape[i], yh_shape[i]))
            self.mapping_list.append(torch.nn.Parameter(_init_tensor))
        self.mapping_list = torch.nn.ParameterList(self.mapping_list)

    def forward(self, yl, yh):
        if self.sample_last_dim:
            map_y = tensorly.tenalg.multi_mode_dot(yl, self.mapping_list)
        else:
            _perm = [i+1 for i in range(yl.dim()-1)] + [0]
            _back_perm = [yl.dim()-1] + [i for i in range(yl.dim()-1)]
            map_y = tensorly.tenalg.multi_mode_dot(yl.permute(_perm), self.mapping_list).permute(_back_perm)
        res = yh - map_y
        return res

    def low_2_high(self, yl, res):
        if self.sample_last_dim:
            map_y = tensorly.tenalg.multi_mode_dot(yl, self.mapping_list)
        else:
            _perm = [i+1 for i in range(yl.dim()-1)] + [0]
            _back_perm = [yl.dim()-1] + [i for i in range(yl.dim()-1)]
            map_y = tensorly.tenalg.multi_mode_dot(yl.permute(_perm), self.mapping_list).permute(_back_perm)
        yh = map_y + res.reshape_as(map_y)
        return yh

    def low_2_high_double_mapping(self, yl_var, res):
        yl_var = yl_var.sqrt()
        if self.sample_last_dim:
            map_y = tensorly.tenalg.multi_mode_dot(yl_var, self.mapping_list)
        else:
            _perm = [i+1 for i in range(yl_var.dim()-1)] + [0]
            _back_perm = [yl_var.dim()-1] + [i for i in range(yl_var.dim()-1)]
            map_y = tensorly.tenalg.multi_mode_dot(yl_var.permute(_perm), self.mapping_list).permute(_back_perm)
        map_y = torch.pow(map_y, 2)
        yh_var = map_y + res
        return yh_var

    def high_2_low(self, yh, res):
        raise NotImplemented

    def get_param(self, optimize_param_list):
        for _matrix in self.mapping_list:
            optimize_param_list.append(_matrix)
        return optimize_param_list

    def set_param(self, param_list):
        with torch.no_grad():
            for i in range(len(self.mapping_list)):
                self.mapping_list[i].copy_(param_list[i])

    def get_params_need_check(self):
        return self.mapping_list


if __name__ == '__main__':
    mc = mapping_connection((10,20), (35,50))
    a = torch.randn(10,20)
    b = torch.randn(35,50)
    res = mc(a, b)
    pass