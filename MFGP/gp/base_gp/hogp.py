# import gpytorch
import math
import torch

import tensorly
from tensorly import tucker_to_tensor
tensorly.set_backend('pytorch')

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

from MFGP.utils.gp_noise import GP_noise_box
from MFGP.gp.kernel.kernel_utils import create_kernels
from MFGP.utils.dict_tools import update_dict_with_default
from MFGP.utils.mfgp_log import MFGP_LOG

class eigen_pairs:
    def __init__(self, matrix) -> None:
        eigen_value, eigen_vector = torch.linalg.eigh(matrix, UPLO='U')
        self.value = eigen_value
        self.vector = eigen_vector

default_config = {
    'noise': {'init_value': 1., 'format': 'linear'},
    'kernel': [{'SE': {'noise_exp_format':True, 'length_scale':1., 'scale': 1.}}],

    'learnable_grid': False,
    'learnable_mapping': False,

    'fidelity_shapes': None,
}


class HOGP_MODULE(torch.nn.Module):
    """
    Higher Order Gaussian Process (HOGP) module.

    Args:
        gp_model_config (dict): Configuration for the GP model. Defaults to None.

    Attributes:
        gp_model_config (dict): Configuration for the GP model.
        noise_box (GP_noise_box): Noise box for GP model.
        train_x (torch.Tensor or None): Training input data.
        train_y (torch.Tensor or None): Training output data.
        n_dim (int): Number of dimensions in the output data.
        kernel_list (list): List of kernel functions.
        grid (torch.nn.ParameterList): Grid for mapping.
        mapping_vector (torch.nn.ParameterList): Mapping vectors.

    Methods:
        check_single_tensor(t): Check if the input tensor is a single tensor.
        compute_kernel_cache(): Compute kernel cache.
        compute_loss(x, y, x_var=0., y_var=0., update_data=False): Compute loss function.
        forward(x, x_vars=0.): Forward pass of the module.
    """

    def __init__(self, gp_model_config=None) -> None:
        super().__init__()
        _final_config = update_dict_with_default(default_config, gp_model_config)
        self.gp_model_config = _final_config

        y_shape = self.gp_model_config['fidelity_shapes']
        if y_shape is None:
            raise ValueError('y_shape must be set as list')
        
        if isinstance(y_shape[0], list) or isinstance(y_shape[0], torch.Size):
            y_shape = y_shape[0]

        for _d in y_shape:
            if _d == 1:
                MFGP_LOG.w('y_shape contains 1, which is invalid dim for HOGP model')

        self.noise_box = GP_noise_box(self.gp_model_config['noise'])

        self.train_x = None
        self.train_y = None
        self.n_dim = len(y_shape)

        repeat_k_config = self.gp_model_config['kernel']* (self.n_dim+1)
        self.kernel_list = create_kernels(repeat_k_config)

        # set grid
        self.grid = []
        for _value in y_shape:
            self.grid.append(torch.nn.Parameter(torch.tensor(range(_value)).reshape(-1,1).float()))
        if self.gp_model_config['learnable_grid'] is False:
            for i in range(len(self.grid)):
                self.grid[i].requires_grad = False
        self.grid = torch.nn.ParameterList(self.grid)

        # set mapping
        self.mapping_vector = []
        for _value in y_shape:
            self.mapping_vector.append(torch.nn.Parameter(torch.eye(_value)))
        if self.gp_model_config['learnable_mapping'] is False:
            for i in range(len(self.mapping_vector)):
                self.mapping_vector[i].requires_grad = False
        self.mapping_vector = torch.nn.ParameterList(self.mapping_vector)


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
            assert len(t) == 1, "HOGP model only support one input"
            t = t[0]
        return t
    

    def compute_kernel_cache(self):
        """
        Compute kernel cache.
        """
        kernel_result = []
        eigen_result = []
        # kernel on sample dim
        kernel_result.append(self.kernel_list[0](self.train_x, self.train_x))
        eigen_result.append(eigen_pairs(kernel_result[-1]))

        # kernel on ndim
        for i in range(0, self.n_dim):
            _in = tensorly.tenalg.mode_dot(self.grid[i], self.mapping_vector[i], 0)
            kernel_result.append(self.kernel_list[i+1](_in, _in))
            eigen_result.append(eigen_pairs(kernel_result[-1]))

        self.k_result_cache = kernel_result
        self.eigen_cache = eigen_result


    def compute_loss(self, x, y, x_var=0., y_var=0., update_data=False):
        """
        Compute the loss function.

        Args:
            x (torch.Tensor or list): Input data.
            y (torch.Tensor or list): Output data.
            x_var (float): Variance of the input data. Defaults to 0.
            y_var (float): Variance of the output data. Defaults to 0.
            update_data (bool): Whether to update the training data. Defaults to False.

        Returns:
            torch.Tensor: Loss value.
        """
        x = self.check_single_tensor(x)
        y = self.check_single_tensor(y)

        # TODO checking if inputs/outputs was changed
        if self.train_x is None:
            self.train_x = x
            self.train_y = y
        elif update_data:
            self.train_x = x
            self.train_y = y

        self.compute_kernel_cache()

        m_device = list(self.parameters())[0].device
        # compute log(|S|) = sum over the logarithm of all the elements in A. O(nd) complexity.
        _init_value = torch.tensor([1.0],  device=m_device).reshape(*[1 for i in range(self.n_dim+1)])

        # Kruskal operator
        lambda_list = [eigen.value.reshape(-1, 1) for eigen in self.eigen_cache]
        A = tucker_to_tensor((_init_value, lambda_list))

        _noise = self.noise_box.get()

        A = A + _noise.pow(-1)* tensorly.ones(A.shape,  device=m_device)
        A = A + y_var
        
        # vec(z).T@ S.inverse @ vec(z) = b.T @ b,  b = S.pow(-1/2) @ vec(z)
        T_1 = tensorly.tenalg.multi_mode_dot(self.train_y, [eigen.vector.T for eigen in self.eigen_cache])
        T_2 = T_1 * A.pow(-1/2)
        T_3 = tensorly.tenalg.multi_mode_dot(T_2, [eigen.vector for eigen in self.eigen_cache])
        b = tensorly.tensor_to_vec(T_3)

        # g = S.inverse@vec(z)
        g = tensorly.tenalg.multi_mode_dot(T_1 * A.pow(-1), [eigen.vector for eigen in self.eigen_cache])

        self.A = A
        self.g = g

        nd = torch.prod(torch.tensor([value for value in self.A.shape]))
        loss = -1/2* nd * torch.log(torch.tensor(2 * math.pi, device=m_device))
        loss += -1/2* torch.log(self.A).sum()
        loss += -1/2* b.t() @ b

        loss = -loss/nd
        return loss


    def forward(self, x, x_vars=0.):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor or list): Input data.
            x_vars (float): Variance of the input data. Defaults to 0.

        Returns:
            tuple: Tuple containing the predicted mean and variance.
        """
        x = self.check_single_tensor(x)

        with torch.no_grad():
            # Get predict mean
            K_star = self.kernel_list[0](x, self.train_x)
            predict_u = tensorly.tenalg.multi_mode_dot(self.g, [K_star] + self.k_result_cache[1:])

            # Get predict var
            _init_value = torch.tensor([1.0], device=x.device).reshape(*[1 for i in range(self.n_dim)])
            diag_K_dims = tucker_to_tensor(( _init_value, [_K.diag().reshape(-1,1) for _K in self.k_result_cache[1:]]))
            diag_K_dims = diag_K_dims.unsqueeze(0)
            diag_K_x = self.kernel_list[0](x, x).diag()
            for i in range(self.n_dim):
                diag_K_x = diag_K_x.unsqueeze(-1)
            diag_K = diag_K_x*diag_K_dims

            S = self.A * self.A.pow(-1/2)
            S_2 = S.pow(2)

            # S_product = tensorly.tenalg.multi_mode_dot(S_2, [(K_star@K_p.inverse()@eigen_vector_p).pow(2), eigen_vector_d1.pow(2), eigen_vector_d2.pow(2)])
            kernel_on_trainx = self.k_result_cache[0]
            eigen_vectors_x = K_star@kernel_on_trainx + JITTER*torch.eye(K_star.shape[0], kernel_on_trainx.shape[0], device=x.device).pow(2)
            eigen_vectors_dims = [self.eigen_cache[i+1].vector.pow(2) for i in range(self.n_dim)]
            eigen_vectors = [eigen_vectors_x] + eigen_vectors_dims
            S_product = tensorly.tenalg.multi_mode_dot(S_2, eigen_vectors)
            
            var_diag = diag_K + S_product

        return predict_u, var_diag


