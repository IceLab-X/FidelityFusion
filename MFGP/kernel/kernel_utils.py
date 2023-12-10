import torch

from MFGP.kernel.SE_kernel import SE_kernel

def create_kernels(kernel_configs):

    kernel_list = []

    for _k_config in kernel_configs:
        for kernel_name, kernel_config in _k_config.items():
            if kernel_name == 'SE':
                kernel_list.append(SE_kernel(kernel_config))
            else:
                raise NotImplementedError

    return torch.nn.ModuleList(kernel_list)

def create_kernel(kernel_config):
    if isinstance(kernel_config, list) and len(kernel_config) == 1:
        kernel_config = kernel_config[0]

    for kernel_name, kernel_config in kernel_config.items():
        if kernel_name == 'SE':
            return SE_kernel(kernel_config)
        elif kernel_name == 'kernel_res':
            from MFGP.kernel.MCMC_res_kernel import Kernel_res
            return Kernel_res(kernel_config)
        else:
            raise NotImplementedError