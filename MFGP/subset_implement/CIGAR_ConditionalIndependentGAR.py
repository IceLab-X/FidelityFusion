import torch

from MFGP.utils.mfgp_log import MFGP_LOG
from MFGP.utils.dict_tools import update_dict_with_default
from MFGP.base_gp.cigp import CIGP
from MFGP.multiscale_coupling.matrix import Matrix_Mapping


default_cigp_model_config = {
    'noise': {'init_value': 1., 'format': 'exp'},
    'kernel': {'SE': {'noise_exp_format':True, 'length_scale':1., 'scale': 1.}},
}

default_matrix_mapping_config = {
    'low_fidelity_shape': None,
    'high_fidelity_shape': None,
    'matrix_init_method': "smooth",     # smooth, eye

    'rho_value_init': 1.,
    'trainable_rho': False,
}

default_cigar_config = {
    'cigp_model_config': default_cigp_model_config,
    'fidelity_shapes': [],
}


class CIGAR(torch.nn.Module):
    """
    CIGAR class represents a module for Coupled Input Gaussian Process (CIGP) with multiple fidelity levels.

    Args:
        cigar_config (dict): Configuration for the CIGAR module.

    Attributes:
        config (dict): Configuration for the CIGAR module.
        cigp_list (torch.nn.ModuleList): List of CIGP modules for each fidelity level.
        fidelity_num (int): Number of fidelity levels.
        matrix_list (torch.nn.ModuleList): List of Matrix_Mapping modules for each fidelity level.

    """

    def __init__(self, cigar_config) -> None:
        super().__init__()
        self.config = update_dict_with_default(default_cigar_config, cigar_config)
        self.cigp_list = None
        self.fidelity_num = len(self.config['fidelity_shapes'])

        self.init_cigp_model()
        self.init_matrix_mapping()

    def init_matrix_mapping(self):
        """
        Initializes the matrix mapping modules based on the fidelity shapes in the configuration.
        """
        matrix_config = [default_matrix_mapping_config]*(self.fidelity_num-1)
        for i in range(self.fidelity_num-1):
            matrix_config[i]['low_fidelity_shape'] = self.config['fidelity_shapes'][i]
            matrix_config[i]['high_fidelity_shape'] = self.config['fidelity_shapes'][i+1]
        self.matrix_list = [Matrix_Mapping(matrix_config[i]) for i in range(self.fidelity_num-1)]
        self.matrix_list = torch.nn.ModuleList(self.matrix_list)
        

    def init_cigp_model(self):
        """
        Initializes the CIGP modules based on the cigp_model_config in the configuration.
        """
        if self.fidelity_num <= 1:
            MFGP_LOG.e("fidelity_num must be greater than 1, set fidelity_num in config first")
        
        # expand_config
        if isinstance(self.config['cigp_model_config'], dict) or \
            (isinstance(self.config['cigp_model_config'], list) and len(self.config['cigp_model_config']) == 1):
            cigp_config_list = [self.config['cigp_model_config']] * self.fidelity_num
        else:
            if len(self.config['cigp_model_config']) != self.fidelity_num:
                MFGP_LOG.e("Stack {} cigp model with different cigp_model_config, but the length of cigp_model_config is not equal to stack_num")
            cigp_config_list = self.config['cigp_model_config']

        # create multi cigp model
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
            MFGP_LOG.e: If the fidelity index is out of range.
        """
        if fidelity_index < 0 or fidelity_index >= self.fidelity_num:
            MFGP_LOG.e("fidelity_index must be bigger than {}, and smaller than fidelity_num[{}]".format(0, self.fidelity_num))


    def single_fidelity_forward(self, x, low_fidelity_y, x_var=0., low_fidelity_y_var=0., fidelity_index=0):
        """
        Computes the forward pass for a single fidelity level.

        Args:
            x (torch.Tensor): The input tensor.
            low_fidelity_y (torch.Tensor): The low fidelity output tensor.
            x_var (float, optional): The variance of the input tensor. Defaults to 0.
            low_fidelity_y_var (float, optional): The variance of the low fidelity output tensor. Defaults to 0.
            fidelity_index (int, optional): The fidelity index. Defaults to 0.

        Returns:
            torch.Tensor: The output mean tensor.
            torch.Tensor: The output variance tensor.
        """
        if self.cigp_list is None:
            MFGP_LOG.e("please train first")
        self.check_fidelity_index(fidelity_index)

        if fidelity_index == 0:
            return self.cigp_list[0].forward(x, x_var)
        else:
            res_mean, res_var = self.cigp_list[fidelity_index].forward(x, x_var)
            high_fidelity_mean = self.matrix_list[fidelity_index-1].forward(low_fidelity_y, res_mean)
            high_fidelity_var = self.matrix_list[fidelity_index-1].var_forward(low_fidelity_y_var, res_var)
            return high_fidelity_mean, high_fidelity_var


    def single_fidelity_compute_loss(self, x, low_fidelity, high_fidelity_y, x_var=0., low_fidelity_var=0., high_fidelity_y_var=0., fidelity_index=0):
        """
        Computes the loss for a single fidelity level.

        Args:
            x (torch.Tensor): The input tensor.
            low_fidelity (torch.Tensor): The low fidelity tensor.
            high_fidelity_y (torch.Tensor): The high fidelity output tensor.
            x_var (float, optional): The variance of the input tensor. Defaults to 0.
            low_fidelity_var (float, optional): The variance of the low fidelity tensor. Defaults to 0.
            high_fidelity_y_var (float, optional): The variance of the high fidelity output tensor. Defaults to 0.
            fidelity_index (int, optional): The fidelity index. Defaults to 0.

        Returns:
            torch.Tensor: The computed loss.
        """
        self.check_fidelity_index(fidelity_index)
        if fidelity_index == 0:
            return self.cigp_list[0].compute_loss(x, high_fidelity_y)
        else:
            res = self.matrix_list[fidelity_index-1].forward(low_fidelity, high_fidelity_y)
            return self.cigp_list[fidelity_index].compute_loss(x, res, update_data=True)


    def forward(self, x, x_var=0., to_fidelity_n=-1):
        """
        Computes the forward pass for multiple fidelity levels.

        Args:
            x (torch.Tensor): The input tensor.
            x_var (float, optional): The variance of the input tensor. Defaults to 0.
            to_fidelity_n (int, optional): The target fidelity level. Defaults to -1.

        Returns:
            torch.Tensor: The output mean tensor.
            torch.Tensor: The output variance tensor.
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
                res_mean, res_var = self.cigp_list[_fn].forward(x, x_var)
                mean = self.matrix_list[_fn-1].backward(mean, res_mean)
                var = self.matrix_list[_fn-1].var_backward(var, res_var)
        return mean, var


    def compute_loss(self, x, y_list, to_fidelity_n=-1):
        """
        Computes the loss for multiple fidelity levels.

        Args:
            x (torch.Tensor): The input tensor.
            y_list (list): List of tensors representing the output at each fidelity level.
            to_fidelity_n (int, optional): The target fidelity level. Defaults to -1.

        Returns:
            torch.Tensor: The computed loss.
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
                loss += self.cigp_list[0].compute_loss(x, y_list[0])
            else:
                res = self.matrix_list[_fn-1].forward(y_list[_fn-1], y_list[_fn])
                loss += self.cigp_list[_fn].compute_loss(x, res, update_data=True)
        return loss