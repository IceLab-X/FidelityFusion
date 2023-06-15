import torch
import tensorly
import copy
import math

from tensorly import tucker_to_tensor
from utils.eigen import eigen_pairs
from utils.mlgp_log import mlgp_log
tensorly.set_backend('pytorch')

def mask_map_2_single_vector_index(mask_map):
    # mask_map: bool map
    return

def nvector_index_2_single_vector_index():
    return


def shape_index_flatten(index, shape):
    ind = 0
    base_set = 1
    for i in range(len(index)):
        assert index[-(i+1)] < shape[-(i+1)]
        ind += index[-(i+1)]*base_set
        base_set *= shape[-(i+1)]
    return ind

def index_as_shape(ind, shape):
    bin_set = [1]
    for i in range(len(shape)-1):
        bin_set.append(1*shape[-(i+1)]*bin_set[-1])
    bin_set.reverse()

    index = []
    _ind = ind
    for i, _b in enumerate(bin_set):
        index.append(_ind//_b)
        _ind = _ind%_b
    return index

def flatten_index_for_sample_last_dim(indexes, shape):
    sample_num = len(indexes)
    assert sample_num == shape[-1]
    new_index = []

    for i,_index in enumerate(indexes):
        for _ind in _index:
            shape_ind = index_as_shape(_ind, shape[:-1])
            shape_ind.append(i)
            fla_ind = shape_index_flatten(shape_ind, shape)
            new_index.append(fla_ind)
    return new_index


class posterior_output_decorator(torch.nn.Module):
    module_type_support = ['hogp']

    def __init__(self, module, module_type, mask_index, epoch=100, lr=0.001) -> None:
        super().__init__()
        if module_type not in self.module_type_support:
             mlgp_log.e("Module type support {}, but got".format(self.module_type_support, module_type))

        self.base_module = module
        self.m_type = module_type
        self.lr = lr
        self.epoch = epoch
        self.src_shape = module.predict_y.shape

        self.src_mask_indexes = mask_index
        self.mask_indexes_as_shape = flatten_index_for_sample_last_dim(mask_index, self.src_shape)
        for i, _ind in enumerate(self.mask_indexes_as_shape):
            self.mask_indexes_as_shape[i] = index_as_shape(_ind, self.src_shape)

        # set y for optimizer
        self.d_y = torch.nn.Parameter(module.predict_y.detach())
        self.optimizer = torch.optim.Adam([self.d_y], lr= lr)

        # ----------- FAILED TEST ------------
        # self.data_all = []
        # self.data_need_grad = []
        # y_flatten = module.predict_y.detach().flatten()
        # groundtrue_flatten = module.outputs_eval[0].detach().flatten()
        # _indexes = flatten_index_for_sample_last_dim(mask_index, self.src_shape)
        # for i in range(y_flatten.shape[0]):
        #     if i in _indexes:
        #         self.data_all.append(groundtrue_flatten[i])
        #     else:
        #         self.data_all.append(torch.nn.Parameter(y_flatten[i]))
        #         self.data_need_grad.append(self.data_all[-1])
        # self.optimizer = torch.optim.Adam(self.data_need_grad, lr= lr)
        # self.d_y = torch.stack(self.data_all, 0).reshape(self.src_shape)
        # ----------- FAILED ------------

        # get K star
        self.total_input = torch.cat([self.base_module.inputs_tr[0], self.base_module.inputs_eval[0]],0).detach()
        self.K_star = self.base_module.kernel_list[-1](self.total_input, self.total_input).detach()
        self.K_star_eigen = eigen_pairs(self.K_star)

        self.reset_groundtrue_point()


    def reset_groundtrue_point(self):
        import time
        starttime = time.perf_counter()

        if not hasattr(self, 'mask'):
            self.mask = torch.zeros(self.d_y.shape, device=self.d_y.device).bool()
            for index in self.mask_indexes_as_shape:
                exec("self.mask{} = True".format(index))

        with torch.no_grad():
            # ---------- low speed -----------
            # for index in self.mask_indexes_as_shape:
            #     # self.d_y[*index] = self.base_module.outputs_eval[0][*index]
            #     exec("self.d_y{} = self.base_module.outputs_eval[0]{}".format(index, index))

            self.d_y[self.mask] = self.base_module.outputs_eval[0][self.mask]

        endtime = time.perf_counter()
        # print('reset cost time:', (endtime - starttime))

    def train(self):
        new_y = torch.cat([self.base_module.outputs_tr[0], self.d_y], -1)

        _init_value = torch.tensor([1.0], device=self.d_y.device).reshape(*[1 for i in self.base_module.K])
        lambda_list = [eigen.value.reshape(-1, 1).detach() for eigen in self.base_module.K_eigen]
        lambda_list[-1] = self.K_star_eigen.value.reshape(-1,1)

        # # fast compute
        A = tucker_to_tensor((_init_value, lambda_list))
        A = self.base_module.noise.pow(-1)* tensorly.ones(A.shape, device=A.device)
    
        eigen_matix = [eigen.vector.detach() for eigen in self.base_module.K_eigen]
        eigen_matix[-1] = self.K_star_eigen.vector

        T_1 = tensorly.tenalg.multi_mode_dot(new_y, [_v.T for _v in eigen_matix])
        T_2 = T_1 * A.pow(-1/2)
        T_3 = tensorly.tenalg.multi_mode_dot(T_2, eigen_matix)
        b = tensorly.tensor_to_vec(T_3)

        # loss = b.t()@ b

        nd = torch.prod(torch.tensor([value for value in A.shape]))
        loss = -1/2* nd * torch.log(torch.tensor(2 * math.pi, device=list(self.parameters())[0].device))
        loss += -1/2* torch.log(A).sum()
        loss += -1/2* b.t()@ b
        loss = -loss/nd

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.reset_groundtrue_point()
        return

    def eval(self):
        return self.d_y.detach()