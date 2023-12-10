import torch
from MFGP.utils.dict_tools import update_dict_with_default

default_config = {
    'rho_value_init': 1.,
    'trainable': True,
}

class Residual(torch.nn.Module):
    def __init__(self, config=None) -> None:
        super().__init__()
        self.config = update_dict_with_default(default_config, config)

        self.rho = torch.nn.Parameter(torch.tensor(self.config['rho_value_init'], dtype=torch.float32))
        if self.config['trainable']:
            self.rho.requires_grad = True
        else:
            self.rho.requires_grad = False

    def forward(self, low_fidelity, high_fidelity):
        res = high_fidelity - low_fidelity* self.rho
        return res

    def backward(self, low_fidelity, res):
        high_fidelity = low_fidelity*self.rho + res
        return high_fidelity

    def var_forward(self, low_fidelity_var, high_fidelity_var):
        res = high_fidelity_var - low_fidelity_var* self.rho
        return res
    
    def var_backward(self, low_fidelity_var, res_var):
        high_fidelity_var = low_fidelity_var*self.rho + res_var
        return high_fidelity_var