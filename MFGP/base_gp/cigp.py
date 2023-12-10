# import gpytorch
import torch

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

from MFGP.utils.gp_noise import GP_noise_box
from MFGP.kernel.kernel_utils import create_kernel
from MFGP.utils.dict_tools import update_dict_with_default
from MFGP.utils.mfgp_log import MFGP_LOG


default_config = {
    'noise': {'init_value': 1., 'format': 'exp'},
    'kernel': {'SE': {'noise_exp_format':True, 'length_scale':1., 'scale': 1.}},
}

class CIGP(torch.nn.Module):
    """
    Conditional Independent Gaussian Process (CIGP) module.

    Args:
        gp_model_config (dict): Configuration for the GP model.

    Attributes:
        gp_model_config (dict): Configuration for the GP model.
        noise_box (GP_noise_box): Noise box for the GP model.
        kernel_list (list): List of kernels for the GP model.
        train_x (torch.Tensor): Training input data.
        train_y (torch.Tensor): Training output data.
    """

    def __init__(self, gp_model_config=None) -> None:
        super().__init__()
        _final_config = update_dict_with_default(default_config, gp_model_config)
        self.gp_model_config = _final_config

        self.noise_box = GP_noise_box(self.gp_model_config['noise'])
        self.kernel = create_kernel(self.gp_model_config['kernel'])

        self.train_x = None
        self.train_y = None

    def check_single_tensor(self, t):
        """
        Check if the input tensor is a single tensor.

        Args:
            t (torch.Tensor or list): Input tensor or list of tensors.

        Returns:
            torch.Tensor: Single input tensor.
        """
        # cigp model only support one input and one output
        if isinstance(t, list):
            assert len(t) == 1, "CIGP model only support one input"
            t = t[0]
        return t

    def forward(self, x, x_var=0.):
        """
        Forward pass of the CIGP module.

        Args:
            x (torch.Tensor or list): Input tensor or list of tensors.
            x_var (float): Variance of the input tensor.

        Returns:
            tuple: Tuple containing the predicted mean and variance.
        """
        x = self.check_single_tensor(x)

        if self.train_x is None:
            MFGP_LOG.e("gp model model hasn't been trained. predict failed")
            return None
    
        with torch.no_grad():
            Sigma = self.kernel(self.train_x, self.train_x) + JITTER * torch.eye(self.train_x.size(0), device=list(self.parameters())[0].device)
            _noise = self.noise_box.get()
            Sigma = Sigma + _noise.pow(-1) * torch.eye(self.train_x.size(0), device=list(self.parameters())[0].device)

            L = torch.linalg.cholesky(Sigma)
            kx = self.kernel(self.train_x, x)
            # LinvKx, _ = torch.triangular_solve(kx, L, upper = False)
            LinvKx = torch.linalg.solve_triangular(kx, L, upper = False)
            torch.linalg.solve_triangular(kx, L, upper = False)

            u = kx.t() @ torch.cholesky_solve(self.train_y, L)

            var_diag = self.kernel(x, x).diag().view(-1, 1) - (LinvKx**2).sum(dim = 0).view(-1, 1)
            _noise = self.noise_box.get()
            var_diag = var_diag + _noise.pow(-1)
            var_diag = var_diag.expand_as(u)

            var_diag = var_diag + x_var

        return u, var_diag

    def compute_loss(self, x, y, x_var=0., y_var=0., update_data=False):
        """
        Compute the loss function for the CIGP module.

        Args:
            x (torch.Tensor or list): Input tensor or list of tensors.
            y (torch.Tensor or list): Output tensor or list of tensors.
            x_var (float): Variance of the input tensor.
            y_var (float): Variance of the output tensor.

        Returns:
            torch.Tensor: Loss value.
        """
        x = self.check_single_tensor(x)
        y = self.check_single_tensor(y)
        assert y.ndim == 2, "y should be 2d tensor"

        # TODO checking if inputs/outputs was changed
        if self.train_x is None:
            self.train_x = x
            self.train_y = y
        elif update_data:
            self.train_x = x
            self.train_y = y

        Sigma = self.kernel(x, x) + JITTER * torch.eye(x.size(0), device=list(self.parameters())[0].device)
        _noise = self.noise_box.get()
        Sigma = Sigma + _noise.pow(-1) * torch.eye(x.size(0), device=list(self.parameters())[0].device)
        Sigma = Sigma + y_var

        L = torch.linalg.cholesky(Sigma)

        gamma = L.inverse() @ y       # we can use this as an alternative because L is a lower triangular matrix.

        y_num, y_dimension = y.shape
        nll =  0.5 * (gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI, device=list(self.parameters())[0].device)) * y_dimension
        return nll

