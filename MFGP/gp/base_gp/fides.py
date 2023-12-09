# import gpytorch
import torch

from MFGP.utils.gp_noise import GP_noise_box
from MFGP.gp.kernel.kernel_utils import create_kernel
from MFGP.utils.dict_tools import update_dict_with_default
from MFGP.utils.mfgp_log import MFGP_LOG

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

default_config = {
    'noise': {'init_value': 1., 'format': 'exp'},
    'kernel': {
                'kernel_res': {'noise_exp_format':True, 'length_scale':1., 'scale': 1., 
                                      'length_scale_z':1.},
              },
}

class FIDES_MODULE(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        _final_config = update_dict_with_default(default_config, config)
        self.config = _final_config

        self.noise_box = GP_noise_box(self.config['noise'])

        self.train_x = None
        self.train_y = None

        self.kernel = create_kernel(self.config['kernel'])

        self.fi_define=False

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

    def set_fidelity(self, l1, h1, l2, h2):
        self.l1 = l1
        self.h1 = h1
        self.l2 = l2
        self.h2 = h2
        self.fi_define=True

    def forward(self, x, x_var=None):
        x = self.check_single_tensor(x)

        if self.train_x is None:
            MFGP_LOG.e("gp model model hasn't been trained. predict failed")
            return None
        
        l1, h1, l2, h2 = self.l1, self.h1, self.l2, self.h2
        _noise = self.noise_box.get()

        m_device = list(self.parameters())[0].device
        sigma = self.kernel(self.train_x, self.train_x, l1, h1, l2, h2) + _noise.pow(-1) * torch.eye(self.train_x.size(0), device=m_device)
        sigma = sigma + JITTER * torch.eye(self.train_x.size(0), device=m_device)

        kx = self.kernel(self.train_x, x, l1, h1, l2, h2)
        L = torch.cholesky(sigma)
        LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

        mean = kx.t() @ torch.cholesky_solve(self.train_y, L)  # torch.linalg.cholesky()
        
        var_diag = self.kernel(x, x, l1, h1, l2, h2).diag().view(-1, 1) \
            - (LinvKx**2).sum(dim = 0).view(-1, 1)
        # add the noise uncertainty
        var_diag = var_diag + _noise.pow(-1)

        return mean, var_diag


    def compute_loss(self, x, y, x_var=0., y_var=0., update_data=False):

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

        _noise = self.noise_box.get()

        l1, h1, l2, h2 = self.l1, self.h1, self.l2, self.h2

        m_device = list(self.parameters())[0].device
        sigma = self.kernel(x, x, l1, h1, l2, h2) + _noise.pow(-1) * torch.eye(x.size(0), device=m_device)
        sigma = sigma + JITTER * torch.eye(x.size(0), device=m_device)

        L = torch.linalg.cholesky(sigma)
        y_num, y_dimension = y.shape
        Gamma,_ = torch.triangular_solve(y, L, upper = False)
        nll =  0.5 * (Gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension

        return nll