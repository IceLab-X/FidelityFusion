import torch

from MFGP.utils.mfgp_log import MFGP_LOG
from MFGP.utils.dict_tools import update_dict_with_default
from MFGP.gp.base_gp.cigp import CIGP_MODULE
from MFGP.gp.multiscale_coupling.Residual import Residual


default_cigp_model_config = {
    'noise': {'init_value': 100., 'format': 'exp'},
    'kernel': {'SE': {'noise_exp_format':True, 'length_scale':1., 'scale': 1.}},
}

default_ar_config = {
    'cigp_model_config': default_cigp_model_config,
    'fidelity_shapes': [],
}

class NAR_MODULE(torch.nn.Module):
    def __init__(self, nar_config) -> None:
        super().__init__()
        self.config = update_dict_with_default(default_ar_config, nar_config)
        self.cigp_list = None
        self.fidelity_num = len(self.config['fidelity_shapes'])

        self.init_cigp_model()

    def init_cigp_model(self):
        if self.fidelity_num <= 1:
            MFGP_LOG.e("fidelity_num must be greater than 1, set fidelity_num in config first")

        # expand_config
        if isinstance(self.config['cigp_model_config'], dict) or \
            (isinstance(self.config['cigp_model_config'], list) and len(self.config['cigp_model_config']) == 1):
            cigp_config_list = [self.config['cigp_model_config']] * self.fidelity_num
        else:
            if len(self.config['cigp_model_config']) != self.fidelity_num:
                MFGP_LOG.e("Stack {} cigp model with different cigp_model_config, but the length of hogp_model_config is not equal to stack_num")
            cigp_config_list = self.config['hogp_model_config']

        self.cigp_list = []
        for i, _config in enumerate(cigp_config_list):
            self.cigp_list.append(CIGP_MODULE(_config))

        self.cigp_list = torch.nn.ModuleList(self.cigp_list)

    def check_fidelity_index(self, fidelity_index):
        if fidelity_index < 0 or fidelity_index >= self.fidelity_num:
            MFGP_LOG.e("fidelity_index must be bigger than {}, and smaller than fidelity_num[{}]".format(0, self.fidelity_num))


    def single_fidelity_forward(self, x, low_fidelity_y, x_var=0., low_fidelity_y_var=0., fidelity_index=0):
        if self.cigp_list is None:
            MFGP_LOG.e("please train first")
        self.check_fidelity_index(fidelity_index)

        if fidelity_index == 0:
            return self.cigp_list[0].forward(x, x_var)
        else:
            concat_input = torch.cat([x, low_fidelity_y], dim=-1)
            # concat_var = torch.cat([x_var, low_fidelity_y_var], dim=-1)
            concat_var = 0.
            high_fidelity_mean, high_fidelity_var = self.cigp_list[fidelity_index].forward(concat_input, concat_var)
            return high_fidelity_mean, high_fidelity_var


    def single_fidelity_compute_loss(self, x, low_fidelity, high_fidelity_y, x_var=0., low_fidelity_var=0., high_fidelity_y_var=0., fidelity_index=0):
        self.check_fidelity_index(fidelity_index)
        if fidelity_index == 0:
            return self.cigp_list[0].compute_loss(x, high_fidelity_y)
        else:
            concat_input = torch.cat([x, low_fidelity], dim=-1)
            return self.cigp_list[fidelity_index].compute_loss(concat_input, high_fidelity_y, update_data=False)


    def forward(self, x, x_var=0., to_fidelity_n=-1):
        if self.cigp_list is None:
            MFGP_LOG.e("please train first")
        if to_fidelity_n < 0:
            to_fidelity_n = self.fidelity_num + to_fidelity_n
        self.check_fidelity_index(to_fidelity_n)

        for _fn in range(to_fidelity_n+1):
            if _fn == 0:
                mean, var = self.cigp_list[0].forward(x, x_var)
            else:
                concat_input = torch.cat([x, mean], dim=-1)
                # mean, var = self.cigp_list[_fn].forward(concat_input, var)
                mean, var = self.cigp_list[_fn].forward(concat_input)
        return mean, var


    def compute_loss(self, x, y_list, to_fidelity_n=-1):
        if not isinstance(y_list, list) or len(y_list) != self.fidelity_num:
            MFGP_LOG.e("y_list must be a list of tensor with length {}".format(self.fidelity_num))

        base_shape = y_list[0].shape
        if len(base_shape) != 2:
            MFGP_LOG.e("y must be a list of tensor with shape [batch_size, value_dim]")
        for _t in y_list:
            if _t.shape[1:] != base_shape[1:]:
                MFGP_LOG.e("y must be a list of tensor with same shape. Got {}".format([_t.shape for _t in y_list]))

        if to_fidelity_n < 0:
            to_fidelity_n = self.fidelity_num + to_fidelity_n
        self.check_fidelity_index(to_fidelity_n)

        loss = 0.
        for _fn in range(to_fidelity_n+1):
            if _fn == 0:
                loss += self.cigp_list[0].compute_loss(x, y_list[0])
            else:
                concat_input = torch.cat([x, y_list[_fn-1]], dim=-1)
                loss += self.cigp_list[_fn].compute_loss(concat_input, y_list[_fn], update_data=False)
        return loss