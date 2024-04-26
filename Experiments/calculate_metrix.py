import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def _gaussian_loss(inputs, target, var):
    assert inputs.shape == target.shape
    # if inputs.shape == var.shape:
    #     pass
    # else:
    #     var = torch.diag_embed(torch.flatten(var))
    assert inputs.shape == var.shape
    if len(inputs.shape) > 2:
        sample = inputs.shape[0]
        inputs = inputs.reshape(-1, sample)
        target = target.reshape(-1, sample)
        var = var.reshape(-1, sample)
    with torch.no_grad():
        return torch.nn.functional.gaussian_nll_loss(inputs, target, var).item()
    
def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    tem = (actual - predicted) ** 2
    return np.average(tem, axis=0) / (tem.max() - tem.min())
    
# calculate r2 rmse nll
def calculate_metrix(**kwargs):
    """
        calculates r2, rmse and nll of model prediction.
        kwargs:
        :param y_test: ndarray or tensor
        :param y_mean_pre: ndarray or tensor
        :param y_var_pre: ndarray or tensor
    """
    # check if arguments is ndarray type
    for key, arg in kwargs.items():
        if type(arg) is torch.Tensor:
            kwargs[key] = kwargs[key].detach().numpy()
    # R2
    r2 = r2_score(kwargs['y_test'], kwargs['y_mean_pre'])
    # RMSE
    rmse = np.sqrt(mean_squared_error(kwargs['y_test'], kwargs['y_mean_pre']))
    # nll
    # The calculation of variance is not very feasible for GAR on large-scale data
    # nll = _gaussian_loss(torch.from_numpy(kwargs['y_test']), torch.from_numpy(kwargs['y_mean_pre']), torch.from_numpy(kwargs['y_var_pre']).diag().reshape(-1, 1))
    # NRMSE
    NRMSE = nrmse(kwargs['y_test'], kwargs['y_mean_pre'])[0]

    # return {'r2': r2, 'rmse': rmse, 'nll': nll, 'nrmse': NRMSE}
    return {'r2': r2, 'rmse': rmse, 'nrmse': NRMSE}