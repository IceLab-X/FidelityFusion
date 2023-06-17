import re
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def _reshape_as_2D(A, sample_last_dim):
    if len(A.shape) > 2:
        sample = A.shape[-1] if sample_last_dim is True else A.shape[0]
        A = A.reshape(-1, sample) if sample_last_dim is True else A.reshape(sample, -1)
    return A


def high_level_evaluator(predicts, target, method_list, sample_last_dim=False):
    # simple evaluator
    _method = []
    for _m in ['mae', 'r2', 'rmse']:
        if _m in method_list:
            _method.append(_m)
    result_dict = performance_evaluator(predicts[0], target, _method, sample_last_dim)

    if 'gaussian_loss' in method_list:
        result_dict['gaussian_loss'] = _gaussian_loss(predicts[0], target, predicts[1])
    return result_dict


def _gaussian_loss(inputs, target, var):
    assert inputs.shape == target.shape
    assert inputs.shape == var.shape
    if len(inputs.shape) > 2:
        sample = inputs.shape[0]
        inputs = inputs.reshape(-1, sample)
        target = target.reshape(-1, sample)
        var = var.reshape(-1, sample)
    with torch.no_grad():
        return torch.nn.functional.gaussian_nll_loss(inputs, target, var).item()


def performance_evaluator(A, B, method_list, sample_last_dim=False):
    if hasattr(A, 'numpy'):
        A = A.cpu().detach().numpy()
    if hasattr(B, 'numpy'):
        B = B.cpu().detach().numpy()
    A = _reshape_as_2D(A, sample_last_dim)
    B = _reshape_as_2D(B, sample_last_dim)

    result = {}
    for _method in method_list:
        if _method == 'mae':
            result['mae'] = mean_absolute_error(A, B)
        elif _method == 'r2':
            result['r2'] = r2_score(A, B)
        elif _method == 'rmse':
            result['rmse'] = np.sqrt(mean_squared_error(A, B))
    return result