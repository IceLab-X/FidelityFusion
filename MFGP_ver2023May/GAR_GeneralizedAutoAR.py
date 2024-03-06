import sys
sys.path.append(r'H:\\eda\\mybranch')
import torch

from MFGP_ver2023May.utils.mfgp_log import MFGP_LOG
from MFGP_ver2023May.utils.dict_tools import update_dict_with_default
from MFGP_ver2023May.base_gp.hogp import HOGP
from MFGP_ver2023May.multiscale_coupling.matrix import Matrix_Mapping
from MFGP_ver2023May.utils.subset_tools import Subset_checker


default_hogp_model_config = {
    'noise': {'init_value': 1., 'format': 'linear'},
    'kernel': [{'SE': {'noise_exp_format':True, 'length_scale':1., 'scale': 1.}}],

    'learnable_grid': False,
    'learnable_mapping': False,
}

default_matrix_mapping_config = {
    'low_fidelity_shape': None,
    'high_fidelity_shape': None,
    'matrix_init_method': "smooth",     # smooth, eye

    'rho_value_init': 1.,
    'trainable_rho': False,
}

default_gar_config = {
    'hogp_model_config': default_hogp_model_config,
    'fidelity_shapes': [[1],[1]],
}


class GAR(torch.nn.Module):
    def __init__(self, gar_config) -> None:
        """
        Initializes the GAR.

        Args:
            gar_config (dict): Configuration parameters for the GAR.
        """
        super().__init__()
        self.config = update_dict_with_default(default_gar_config, gar_config)
        self.hogp_list = None
        self.fidelity_num = len(self.config['fidelity_shapes'])

        self.init_hogp_model()
        self.init_matrix_mapping()
        self.nonsubset = True


    def init_matrix_mapping(self):
        """
        Initializes the matrix mapping for each fidelity level.
        """
        matrix_config = [default_matrix_mapping_config]*(self.fidelity_num-1)
        for i in range(self.fidelity_num-1):
            matrix_config[i]['low_fidelity_shape'] = self.config['fidelity_shapes'][i]
            matrix_config[i]['high_fidelity_shape'] = self.config['fidelity_shapes'][i+1]
        self.matrix_list = [Matrix_Mapping(matrix_config[i]) for i in range(self.fidelity_num-1)]
        self.matrix_list = torch.nn.ModuleList(self.matrix_list)
        

    def init_hogp_model(self):
        """
        Initializes the HOGP models for each fidelity level.
        """
        if self.fidelity_num <= 1:
            MFGP_LOG.e("fidelity_num must be greater than 1, set fidelity_num in config first")
        
        # expand_config
        if isinstance(self.config['hogp_model_config'], dict) or \
            (isinstance(self.config['hogp_model_config'], list) and len(self.config['hogp_model_config']) == 1):
            hogp_config_list = [self.config['hogp_model_config']] * self.fidelity_num
        else:
            if len(self.config['hogp_model_config']) != self.fidelity_num:
                MFGP_LOG.e("Stack {} cigp model with different hogp_model_config, but the length of hogp_model_config is not equal to stack_num")
            hogp_config_list = self.config['hogp_model_config']

        self.hogp_list = []
        for i, _config in enumerate(hogp_config_list):
            _config['fidelity_shapes'] =  self.config['fidelity_shapes'][i]
            self.hogp_list.append(HOGP(_config))

        self.hogp_list = torch.nn.ModuleList(self.hogp_list)

    def check_fidelity_index(self, fidelity_index):
        """
        Checks if the given fidelity index is valid.

        Args:
            fidelity_index (int): The fidelity index to check.

        Raises:
            Exception: If the fidelity index is out of range.
        """
        if fidelity_index < 0 or fidelity_index >= self.fidelity_num:
            MFGP_LOG.e("fidelity_index must be bigger than {}, and smaller than fidelity_num[{}]".format(0, self.fidelity_num))


    def single_fidelity_forward(self, x, low_fidelity_y, x_var=0., low_fidelity_y_var=0., fidelity_index=0):
        """
        Performs a forward pass for a single fidelity level.

        Args:
            x (tensor): The input tensor.
            low_fidelity_y (tensor): The low fidelity output tensor.
            x_var (float, optional): The variance of the input tensor. Defaults to 0.
            low_fidelity_y_var (float, optional): The variance of the low fidelity output tensor. Defaults to 0.
            fidelity_index (int, optional): The fidelity index. Defaults to 0.

        Returns:
            tuple: A tuple containing the mean and variance of the high fidelity output tensor.
        """
        if self.hogp_list is None:
            MFGP_LOG.e("please train first")
        self.check_fidelity_index(fidelity_index)

        if fidelity_index == 0:
            return self.hogp_list[0].forward(x, x_var)
        else:
            res_mean, res_var = self.hogp_list[fidelity_index].forward(x, x_var)
            high_fidelity_mean = self.matrix_list[fidelity_index-1].forward(low_fidelity_y, res_mean)
            high_fidelity_var = self.matrix_list[fidelity_index-1].var_forward(low_fidelity_y_var, res_var)
            return high_fidelity_mean, high_fidelity_var


    def single_fidelity_compute_loss(self, x, low_fidelity, high_fidelity_y, x_var=0., low_fidelity_var=0., high_fidelity_y_var=0., fidelity_index=0):
        """
        Computes the loss for a single fidelity level.

        Args:
            x (tensor): The input tensor.
            low_fidelity (tensor): The low fidelity tensor.
            high_fidelity_y (tensor): The high fidelity output tensor.
            x_var (float, optional): The variance of the input tensor. Defaults to 0.
            low_fidelity_var (float, optional): The variance of the low fidelity tensor. Defaults to 0.
            high_fidelity_y_var (float, optional): The variance of the high fidelity output tensor. Defaults to 0.
            fidelity_index (int, optional): The fidelity index. Defaults to 0.

        Returns:
            tensor: The computed loss.
        """
        self.check_fidelity_index(fidelity_index)
        if fidelity_index == 0:
            return self.hogp_list[0].compute_loss(x, high_fidelity_y)
        else:
            res = self.matrix_list[fidelity_index-1].forward(low_fidelity, high_fidelity_y)
            return self.hogp_list[fidelity_index].compute_loss(x, res, update_data=True)


    def forward(self, x, x_var=0., to_fidelity_n=-1):
        """
        Performs a forward pass through the GAR.

        Args:
            x (tensor): The input tensor.
            x_var (float, optional): The variance of the input tensor. Defaults to 0.
            to_fidelity_n (int, optional): The fidelity level to propagate to. Defaults to -1.

        Returns:
            tuple: A tuple containing the mean and variance of the output tensor.
        """
        if self.hogp_list is None:
            MFGP_LOG.e("please train first")
        if to_fidelity_n < 0:
            to_fidelity_n = self.fidelity_num + to_fidelity_n
        self.check_fidelity_index(to_fidelity_n)

        for _fn in range(to_fidelity_n+1):
            if _fn == 0:
                mean, var = self.hogp_list[0].forward(x, x_var)
            else:
                res_mean, res_var = self.hogp_list[_fn].forward(x, x_var)
                mean = self.matrix_list[_fn-1].backward(mean, res_mean)
                var = self.matrix_list[_fn-1].var_backward(var, res_var)
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
        Computes the loss for the GAR.

        Args:
            x (tensor): The input tensor.
            y_list (list): A list of tensors representing the high fidelity outputs at each fidelity level.
            to_fidelity_n (int, optional): The fidelity level to propagate to. Defaults to -1.

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
        for _t in y_list:
            if _t.shape[1:] != base_shape[1:]:
                MFGP_LOG.e("y must be a list of tensor with same shape. Got {}".format([_t.shape for _t in y_list]))

        if to_fidelity_n < 0:
            to_fidelity_n = self.fidelity_num + to_fidelity_n
        self.check_fidelity_index(to_fidelity_n)

        loss = 0.
        for _fn in range(to_fidelity_n+1):
            if _fn == 0:
                loss += self.hogp_list[0].compute_loss(x_list[0], y_list[0])
            else:
                if self.nonsubset:
                    x, y_low, y_high = self._get_nonsubset_data(x_list[_fn-1], x_list[_fn], y_list[_fn-1], y_list[_fn], _fn)
                else:
                    x = x_list[0]
                    y_low = y_list[_fn-1]
                    y_high = y_list[_fn]
                res = self.matrix_list[_fn-1].forward(y_low, y_high)
                loss += self.hogp_list[_fn].compute_loss(x, res, update_data=True)
        return loss
    
if __name__ == "__main__":
    torch.manual_seed(1)
    # generate the data
    x_all = torch.rand(500, 1) * 20
    xlow_indices = torch.randperm(500)[:300]
    x_low = x_all[xlow_indices]
    xhigh_indices = torch.randperm(500)[:300]
    x_high = x_all[xhigh_indices]
    x_test = torch.linspace(0, 20, 100).reshape(-1, 1)

    y_low = torch.sin(x_low) + torch.rand(300, 1) * 0.6 - 0.3
    y_high = torch.sin(x_high) + torch.rand(300, 1) * 0.2 - 0.1
    y_test = torch.sin(x_test)

    x_train = [x_low, x_high]
    y_train = [y_low, y_high]

    myNAR = GAR(default_gar_config)
    myNAR.compute_loss([x_train], [y_train], to_fidelity_n=1)
    ypred, ypred_var = myNAR.forward([x_test])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.errorbar(x_test.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(x_test.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha=0.2)
    plt.plot(x_test.flatten(), y_test, 'k+')
    plt.show()