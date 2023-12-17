import torch

from MFGP_ver2023May.utils.mfgp_log import MFGP_LOG
from MFGP_ver2023May.utils.dict_tools import update_dict_with_default
from MFGP_ver2023May.base_gp.cigp import CIGP
from MFGP_ver2023May.multiscale_coupling.Residual import Residual
from MFGP_ver2023May.utils.subset_tools import Subset_checker


default_cigp_model_config = {
    'noise': {'init_value': 100., 'format': 'exp'},
    'kernel': {'SE': {'noise_exp_format':True, 'length_scale':1., 'scale': 1.}},
}

default_ar_config = {
    'cigp_model_config': default_cigp_model_config,
    'fidelity_shapes': [],
}

class NAR(torch.nn.Module):
    def __init__(self, nar_config) -> None:
        """
        Initializes the NAR.

        Args:
            nar_config (dict): Configuration parameters for the NAR.
        """
        super().__init__()
        self.config = update_dict_with_default(default_ar_config, nar_config)
        self.cigp_list = None
        self.fidelity_num = len(self.config['fidelity_shapes'])

        self.init_cigp_model()
        self.nonsubset = True


    def init_cigp_model(self):
        """
        Initializes the CIGP models for each fidelity level.
        """
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
            self.cigp_list.append(CIGP(_config))

        self.cigp_list = torch.nn.ModuleList(self.cigp_list)

    def check_fidelity_index(self, fidelity_index):
        """
        Checks if the fidelity index is valid.

        Args:
            fidelity_index (int): The fidelity index to check.

        Raises:
            Exception: If the fidelity index is out of range.
        """
        if fidelity_index < 0 or fidelity_index >= self.fidelity_num:
            MFGP_LOG.e("fidelity_index must be bigger than {}, and smaller than fidelity_num[{}]".format(0, self.fidelity_num))

    def single_fidelity_forward(self, x, low_fidelity_y, x_var=0., low_fidelity_y_var=0., fidelity_index=0):
        """
        Performs forward pass for a single fidelity level.

        Args:
            x (torch.Tensor): The input tensor.
            low_fidelity_y (torch.Tensor): The low fidelity output tensor.
            x_var (float, optional): The input variance. Defaults to 0.
            low_fidelity_y_var (float, optional): The low fidelity output variance. Defaults to 0.
            fidelity_index (int, optional): The fidelity index. Defaults to 0.

        Returns:
            torch.Tensor: The high fidelity mean tensor.
            torch.Tensor: The high fidelity variance tensor.
        """
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
        """
        Computes the loss for a single fidelity level.

        Args:
            x (torch.Tensor): The input tensor.
            low_fidelity (torch.Tensor): The low fidelity tensor.
            high_fidelity_y (torch.Tensor): The high fidelity output tensor.
            x_var (float, optional): The input variance. Defaults to 0.
            low_fidelity_var (float, optional): The low fidelity variance. Defaults to 0.
            high_fidelity_y_var (float, optional): The high fidelity output variance. Defaults to 0.
            fidelity_index (int, optional): The fidelity index. Defaults to 0.

        Returns:
            torch.Tensor: The computed loss.
        """
        self.check_fidelity_index(fidelity_index)
        if fidelity_index == 0:
            return self.cigp_list[0].compute_loss(x, high_fidelity_y)
        else:
            concat_input = torch.cat([x, low_fidelity], dim=-1)
            return self.cigp_list[fidelity_index].compute_loss(concat_input, high_fidelity_y, update_data=False)

    def forward(self, x, x_var=0., to_fidelity_n=-1):
        """
        Performs forward pass for multiple fidelity levels.

        Args:
            x (torch.Tensor): The input tensor.
            x_var (float, optional): The input variance. Defaults to 0.
            to_fidelity_n (int, optional): The target fidelity level. Defaults to -1.

        Returns:
            torch.Tensor: The mean tensor.
            torch.Tensor: The variance tensor.
        """
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
    

    def _get_nonsubset_data(self, x_low, x_high, y_low, y_high, y_high_fidelity_index):
        x_low_subset_index, x_high_subset_index = Subset_checker.get_subset(x_low, x_high)
        x_low_nonsubset_index, x_high_nonsubset_index = Subset_checker.get_non_subset(x_low, x_high)

        y_low_subset = y_low[x_low_subset_index]
        x_high_nonsubset = x_high[x_high_nonsubset_index]

        if 0 not in [len(x_high_nonsubset_index), len(x_high_subset_index)]:
            y_low_nonsubset = self.forward(x_high_nonsubset, to_fidelity_n=y_high_fidelity_index-1)[0]
            y_low = torch.cat([y_low_subset, y_low_nonsubset], dim=0)
            y_high = torch.cat([y_high[x_high_subset_index], y_high[x_high_nonsubset_index]], dim=0)
            x = torch.cat([x_high[x_high_subset_index], x_high_nonsubset], dim=0)
        elif len(x_high_nonsubset_index) == 0:
            # full subset
            y_low = y_low_subset
            y_high = y_high[x_high_subset_index]
            x = x_high[x_high_subset_index]
        elif len(x_high_subset_index) == 0:
            # full nonsubset
            y_low_nonsubset = self.forward(x_high_nonsubset, to_fidelity_n=y_high_fidelity_index-1)[0]
            y_low = y_low_nonsubset
            y_high = y_high[x_high_nonsubset_index]
            x = x_high[x_high_nonsubset_index]
        return x, y_low, y_high


    def compute_loss(self, x_list, y_list, to_fidelity_n=-1):
        """
        Computes the loss for multiple fidelity levels.

        Args:
            x (torch.Tensor): The input tensor.
            y_list (list): The list of high fidelity output tensors.
            to_fidelity_n (int, optional): The target fidelity level. Defaults to -1.

        Returns:
            torch.Tensor: The computed loss.
        """
        if not isinstance(y_list, list) or len(y_list) != self.fidelity_num:
            MFGP_LOG.e("y_list must be a list of tensor with length {}".format(self.fidelity_num))

        if isinstance(x_list, torch.Tensor):
            x_list = [x_list]
            self.nonsubset = False
        elif isinstance(x_list, list) and len(x_list) == 1:
            self.nonsubset = False

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
                loss += self.cigp_list[0].compute_loss(x_list[0], y_list[0])
            else:
                if self.nonsubset:
                    x, y_low, y_high = self._get_nonsubset_data(x_list[_fn-1], x_list[_fn], y_list[_fn-1], y_list[_fn], _fn)
                else:
                    x = x_list[0]
                    y_low = y_list[_fn-1]
                    y_high = y_list[_fn]

                concat_input = torch.cat([x, y_low], dim=-1)
                loss += self.cigp_list[_fn].compute_loss(concat_input, y_high, update_data=False)
        return loss