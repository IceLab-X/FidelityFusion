import torch

from MFGP_ver2023May.utils.mfgp_log import MFGP_LOG
from MFGP_ver2023May.utils.dict_tools import update_dict_with_default
from MFGP_ver2023May.base_gp.cigp import CIGP
from MFGP_ver2023May.base_gp.fides import FIDES
from MFGP_ver2023May.multiscale_coupling.Residual import Residual
from MFGP_ver2023May.utils.subset_tools import Subset_checker


default_cigp_model_config = {
    'noise': {'init_value': 1., 'format': 'exp'},
    'kernel': {'SE': {'noise_exp_format':True, 'length_scale':1., 'scale': 1.}},
}

default_fides_model_config = {
    'noise': {'init_value': 1., 'format': 'exp'},
    'kernel': {
                'kernel_res': {'noise_exp_format':True, 'length_scale':1., 'scale': 1., 'length_scale_z':1.},
              }
}

default_ar_config = {
    'Residual': {'rho_value_init': 1., 'trainable': True},
    'cigp_model_config': default_cigp_model_config,
    'fides_model_config': default_fides_model_config,
    'fidelity_shapes': [],
}

class CAR(torch.nn.Module):
    def __init__(self, ar_config) -> None:
        """
        Initialize the CAR.

        Args:
            ar_config (dict): Configuration for the AR model.
        """
        super().__init__()
        self.config = update_dict_with_default(default_ar_config, ar_config)
        self.cigp = None
        self.fides = None
        assert self.config['Residual']['trainable'], "AR must have trainable residual. Call ResGP to build with untrainable residual blocks"
        self.fidelity_num = len(self.config['fidelity_shapes'])

        self.init_cigp_model()
        self.init_fides_model()
        self.init_residual()
        self.nonsubset = True


    def init_residual(self):
        """
        Initialize the residual blocks.
        """
        self.residual_list = [Residual(self.config['Residual']) for _ in range(self.fidelity_num-1)]
        self.residual_list = torch.nn.ModuleList(self.residual_list)

    def init_cigp_model(self):
        """
        Initialize the CIGP model.
        """
        self.cigp = CIGP(self.config['cigp_model_config'])

    def init_fides_model(self):
        """
        Initialize the FIDES model.
        """
        self.fides = FIDES(self.config['fides_model_config'])

    def check_fidelity_index(self, fidelity_index):
        """
        Check if the fidelity index is valid.

        Args:
            fidelity_index (int): The fidelity index to check.

        Raises:
            MFGP_LOG.e: If the fidelity index is out of range.
        """
        if fidelity_index < 0 or fidelity_index >= self.fidelity_num:
            MFGP_LOG.e("fidelity_index must be bigger than {}, and smaller than fidelity_num[{}]".format(0, self.fidelity_num))

    def single_fidelity_forward(self, x, low_fidelity_y, x_var=0., low_fidelity_y_var=0., fidelity_index=0):
        """
        Perform forward pass for a single fidelity level.

        Args:
            x (tensor): The input tensor.
            low_fidelity_y (tensor): The low fidelity output tensor.
            x_var (float, optional): Variance of the input tensor. Defaults to 0.
            low_fidelity_y_var (float, optional): Variance of the low fidelity output tensor. Defaults to 0.
            fidelity_index (int, optional): The fidelity index. Defaults to 0.

        Returns:
            tuple: A tuple containing the high fidelity mean and variance tensors.
        """
        if self.cigp is None:
            MFGP_LOG.e("please train first")
        self.check_fidelity_index(fidelity_index)

        if fidelity_index == 0:
            return self.cigp.forward(x, x_var)
        else:
            self.fides.set_fidelity(fidelity_index-1, fidelity_index, fidelity_index-1, fidelity_index)
            res_mean, res_var = self.fides.forward(x)
            high_fidelity_mean = self.residual_list[fidelity_index-1].forward(low_fidelity_y, res_mean)
            high_fidelity_var = self.residual_list[fidelity_index-1].var_forward(low_fidelity_y_var, res_var)
            return high_fidelity_mean, high_fidelity_var

    def single_fidelity_compute_loss(self, x, low_fidelity, high_fidelity_y, x_var=0., low_fidelity_var=0., high_fidelity_y_var=0., fidelity_index=0):
        """
        Compute the loss for a single fidelity level.

        Args:
            x (tensor): The input tensor.
            low_fidelity (tensor): The low fidelity tensor.
            high_fidelity_y (tensor): The high fidelity output tensor.
            x_var (float, optional): Variance of the input tensor. Defaults to 0.
            low_fidelity_var (float, optional): Variance of the low fidelity tensor. Defaults to 0.
            high_fidelity_y_var (float, optional): Variance of the high fidelity output tensor. Defaults to 0.
            fidelity_index (int, optional): The fidelity index. Defaults to 0.

        Returns:
            tensor: The computed loss.
        """
        self.check_fidelity_index(fidelity_index)
        if fidelity_index == 0:
            return self.cigp.compute_loss(x, high_fidelity_y)
        else:
            res = self.residual_list[fidelity_index-1].forward(low_fidelity, high_fidelity_y)
            self.fides.set_fidelity(fidelity_index-1, fidelity_index, fidelity_index-1, fidelity_index)
            return self.fides.compute_loss(x, res, update_data=True)

    def forward(self, x, x_var=0., to_fidelity_n=-1):
        """
        Perform forward pass up to a specified fidelity level.

        Args:
            x (tensor): The input tensor.
            x_var (float, optional): Variance of the input tensor. Defaults to 0.
            to_fidelity_n (int, optional): The fidelity level to compute up to. Defaults to -1.

        Returns:
            tuple: A tuple containing the mean and variance tensors.
        """
        if self.cigp is None:
            MFGP_LOG.e("please train first")
        if to_fidelity_n < 0:
            to_fidelity_n = self.fidelity_num + to_fidelity_n
        self.check_fidelity_index(to_fidelity_n)

        for _fn in range(to_fidelity_n+1):
            if _fn == 0:
                mean, var = self.cigp.forward(x, x_var)
            else:
                self.fides.set_fidelity(_fn-1, _fn, _fn-1, _fn)
                res_mean, res_var = self.fides.forward(x)
                mean = self.residual_list[_fn-1].backward(mean, res_mean)
                var = self.residual_list[_fn-1].var_backward(var, res_var)
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
        Compute the loss up to a specified fidelity level.

        Args:
            x (tensor): The input tensor.
            y_list (list): A list of tensors representing the output at each fidelity level.
            to_fidelity_n (int, optional): The fidelity level to compute up to. Defaults to -1.

        Returns:
            tensor: The computed loss.
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
                loss += self.cigp.compute_loss(x_list[0], y_list[0])
            else:
                if self.nonsubset:
                    x, y_low, y_high = self._get_nonsubset_data(x_list[_fn-1], x_list[_fn], y_list[_fn-1], y_list[_fn], _fn)
                else:
                    x = x_list[0]
                    y_low = y_list[_fn-1]
                    y_high = y_list[_fn]

                res = self.residual_list[_fn-1].forward(y_low, y_high)
                self.fides.set_fidelity(_fn-1, _fn, _fn-1, _fn)
                loss += self.fides.compute_loss(x, res, update_data=True)
        return loss