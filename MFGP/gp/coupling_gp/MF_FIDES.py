import torch

from MFGP.utils.mfgp_log import MFGP_LOG
from MFGP.utils.dict_tools import update_dict_with_default
from MFGP.gp.base_gp.cigp import CIGP_MODULE
from MFGP.gp.base_gp.fides import FIDES_MODULE
from MFGP.gp.multiscale_coupling.Residual import Residual


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

class MF_FIDES_MODULE(torch.nn.Module):
    def __init__(self, ar_config) -> None:
        """
        Initialize the MF_FIDES_MODULE.

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
        self.cigp = CIGP_MODULE(self.config['cigp_model_config'])

    def init_fides_model(self):
        """
        Initialize the FIDES model.
        """
        self.fides = FIDES_MODULE(self.config['fides_model_config'])

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

    def compute_loss(self, x, y_list, to_fidelity_n=-1):
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
                loss += self.cigp.compute_loss(x, y_list[0])
            else:
                res = self.residual_list[_fn-1].forward(y_list[_fn-1], y_list[_fn])
                self.fides.set_fidelity(_fn-1, _fn, _fn-1, _fn)
                loss += self.fides.compute_loss(x, res, update_data=True)
        return loss