import torch

from modules.kernel.Linear_kernel import Linear_kernel
from modules.kernel.SE_kernel_module import SE_kernel
from modules.kernel.RQ_kernel import RQ_kernel
from modules.kernel.Combine_kernel import Combine_kernel
from modules.kernel.Periodic_kernel import Periodic_kernel
from modules.kernel.Matern_kernel import Matern_kernel
from modules.kernel.MCMC_res_kernel import Kernel_res

from utils import *


KERNEL_AVAILABLE = {
    'linear': Linear_kernel,
    'SE': SE_kernel,
    'RQ': RQ_kernel,
    'Periodic': Periodic_kernel,
    'Local_Periodic': Periodic_kernel,
    'Matern': Matern_kernel,
    'kernel_res': Kernel_res,
}


def kernel_generator(kernel_name, config_dict):    
    if kernel_name in KERNEL_AVAILABLE:
        return KERNEL_AVAILABLE[kernel_name](**config_dict)

    elif kernel_name == 'Combine_Linear_SE':
        cb_list = [Linear_kernel(**config_dict['Linear']), SE_kernel(**config_dict['SE'])]
        return Combine_kernel(cb_list)

    else:
        assert False, "kernel not found for {}".format(kernel_name)


def register_kernel(module_class, kernel_config):
    # NOTE: This function will add kernel to the module.
    # check param conflict
    if hasattr(module_class, 'kernel_list') and module_class.kernel_list is not None:
        mlgp_log.e("Regist kernel failed! module already got 'kernel_list' param. It's making conflict and not allowed")

    module_class.kernel_list = []
    for key, value in kernel_config.items():
        if len(value.items()) != 1:
            mlgp_log.e("Regist kernel failed! {} should contain more than one line".format(value))
        for _kernel_type, _kernel_params in value.items():
            module_class.kernel_list.append(kernel_generator(_kernel_type, _kernel_params))
    
    # set as module
    module_class.kernel_list = torch.nn.ModuleList(module_class.kernel_list)
    return