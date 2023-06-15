import torch
import os

class Combine_kernel(torch.nn.Module):
    # Rational Quadratic Kernel
    def __init__(self, kernel_list) -> None:
        super().__init__()
        self.kernel_list = kernel_list

    def forward(self, X, X2):
        K = 0
        for i, _kernel in enumerate(self.kernel_list):
            K += _kernel(X, X2)
        return K

    # def get_param(self):
    #     return self.length_scale, self.log_scale

    def get_param(self, optimize_param_list):
        for i, _kernel in enumerate(self.kernel_list):
            _kernel.get_param(optimize_param_list)
        return optimize_param_list

    def set_param(self, param_list):
        index = 0
        for i, _kernel in enumerate(self.kernel_list):
            _temp_param_list = []
            _kernel.get_param(_temp_param_list)
            param_length = len(_temp_param_list)
            _kernel.set_param(param_list[index:index + param_length])
            index += param_length

    def get_params_need_check(self):
        _temp_list = []
        self.get_param(_temp_list)
        return _temp_list

    def clamp_to_positive(self):
        for i, _kernel in enumerate(self.kernel_list):
            _kernel.clamp_to_positive()
