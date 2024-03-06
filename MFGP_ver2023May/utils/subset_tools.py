from copy import deepcopy
from random import shuffle
import torch
import numpy as np

NP_ALLOW=True
def check_numpy(x):
    if isinstance(x, np.ndarray):
        if NP_ALLOW:
            return True
        else:
            assert False, "|error|: numpy is not allowed"
    else:
        return False

def numpy_compatible_decorator(function):    
    def numpy_compatible_wrapper(*args, **kwargs):
        args_len = len(args)
        args_represent = []
        numpy_exist = False
        for _a in args:
            if check_numpy(_a):
                args_represent.append(torch.from_numpy(_a))
                numpy_exist = True
            else:
                args_represent.append(_a)
        result = function(*args_represent, **kwargs)
        
        if numpy_exist is False or result is None:
            return result
        elif isinstance(result, torch.Tensor):
            return result.numpy()
        else:
            result_represent = []
            for _r in result:
                if isinstance(_r, torch.Tensor):
                    result_represent.append(_r.numpy())
                else:
                    result_represent.append(_r)
            return result_represent
    return numpy_compatible_wrapper


class Subset_checker():
    # @numpy_compatible_decorator
    # def __init__(self, sample_dim=0) -> None:
    #     self.sample_dim = sample_dim
    
    @staticmethod
    @numpy_compatible_decorator
    def unique_check(pt_tensor, sample_dim=0):
        u_tensor, index, count = pt_tensor.unique(sorted=False, return_inverse=True, return_counts=True, dim=sample_dim)
        if (count > 1).any():
            assert False, "|error|: tensor has duplicate samples"

    @numpy_compatible_decorator
    # @staticmethod
    def get_subset(data_a, data_b, subset_type='index', sample_dim=0):
        '''
        Return the subset of data_check, data_base
        
        Subset_type could be 'index' or 'mask'
        For 'index', it's ordered.
        For 'mask', it's not ordered.
        '''
        
        assert subset_type in ['index', 'mask'], "|error|: subset_type should be 'index' or 'mask'"
        Subset_checker.unique_check(data_a)
        Subset_checker.unique_check(data_b)
        t_device = data_a.device

        data_all = torch.cat([data_a, data_b], dim=sample_dim)
        unique_tensor, inverse_index, count = data_all.unique(sorted=False, return_inverse=True, return_counts=True, dim=sample_dim)
        repeat_unique_index = torch.arange(0, len(count), device=t_device)[count>1] 

        mask_matrix = (inverse_index.reshape(-1,1).repeat(1, repeat_unique_index.shape[0]) - repeat_unique_index) == 0
        index_metrix = torch.arange(0, mask_matrix.shape[0], device=t_device).reshape(-1, 1).repeat(1, mask_matrix.shape[1])* mask_matrix
        
        data_base_sample_size = data_a.shape[sample_dim]

        data_base_mask = mask_matrix.sum(1)[:data_base_sample_size]
        data_base_index = index_metrix[:data_base_sample_size, :].sum(0)
        
        data_check_mask = mask_matrix.sum(1)[data_base_sample_size:]
        data_check_index = index_metrix[data_base_sample_size:, :].sum(0) - data_base_sample_size
        
        if subset_type == 'index':
            return data_base_index, data_check_index
        elif subset_type == 'mask':
            return data_base_mask, data_check_mask
        
    @staticmethod
    def get_non_subset(data_a, data_b, non_subset_type='index', sample_dim=0):
        if non_subset_type == "mask":
            # TODO
            assert "not support"

        subset_base_index, subset_check_index = Subset_checker.get_subset(data_a, data_b, subset_type='index', sample_dim=sample_dim)

        nonsubset_base_index = []
        nonsubset_check_index = []
        for i in range(data_a.shape[sample_dim]):
            if i not in subset_base_index:
                nonsubset_base_index.append(i)

        for i in range(data_b.shape[sample_dim]):
            if i not in subset_check_index:
                nonsubset_check_index.append(i)

        return nonsubset_base_index, nonsubset_check_index

if __name__=='__main__':
    print('Testing subset tools')
    tensor_type = ['torch', 'numpy']
    for _tt in tensor_type:
        print('Testing with tensor type: ', _tt)
        x = torch.randn(100,50,50)
        if _tt == 'numpy':
            x = x.numpy()
        shuffle_set = list(range(x.shape[0]))
        shuffle(shuffle_set)
        set_0 = deepcopy(shuffle_set[0:20])
        set_1 = deepcopy(shuffle_set[10:40])
        shuffle(set_1)
        
        x_0 = x[set_0]
        x_1 = x[set_1]
                
        x_0_repeat_index, x_1_repeat_index = Subset_checker.get_subset(x_0, x_1, subset_type='index')
        
        diff = (x_0[x_0_repeat_index] - x_1[x_1_repeat_index])
        print("output index type", type(x_0_repeat_index))
        if (diff!=0).any():
            print("|error| Test not pass, diff: {}\n\n".format(diff))
        else:
            print("Test pass\n\n")