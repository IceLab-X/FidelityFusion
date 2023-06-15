# Base on hogp, train on x, y_res
# y_res = y_high - tensorly.mode.dot(y_low, metrix).


import os
import sys

import torch
import numpy as np

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from modules.gp_module.hogp import HOGP_MODULE
from modules.gp_module.cigp import CIGP_MODULE

def prepare_data():
    # prepare data
    x = np.load('./data/sample/input.npy')
    yl = np.load('./data/sample/output_fidelity_1.npy')
    yh = np.load('./data/sample/output_fidelity_2.npy')
    source_shape = [-1, *yh.shape[1:]]

    x = torch.tensor(x).float()
    yl = torch.tensor(yl).float()
    yh = torch.tensor(yh).float()

    train_inputs = [x[:128,:], yl[:128,:]]
    train_outputs = [yh[:128,:]]
    eval_inputs = [x[128:,:], yl[128:,:]]
    eval_outputs = [yh[128:,:]]
    return train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape


def plot_result(ground_true_y, predict_y, src_shape):
    # plot result
    from visualize_tools.plot_field import plot_container
    data_list = [ground_true_y, predict_y[0].get_mean(), (ground_true_y - predict_y[0].get_mean()).abs()]
    data_list = [_d.reshape(src_shape) for _d in data_list]
    label_list = ['groundtruth', 'predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()

def half(x):
    return x[:x.shape[0]//2]


def gp_model_block_test():
    # get dataset
    train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape = prepare_data()

    # prepare nonsubset
    cigp_train_inputs = [half(train_inputs[0])]
    cigp_train_outputs = [half(train_inputs[1]).flatten(1)]


    # normalizer now is outsider of the model
    from utils.normalizer import Dateset_normalize_manager
    data_norm_manager = Dateset_normalize_manager(cigp_train_inputs, cigp_train_outputs)


    # init low fidelity model
    cigp = CIGP_MODULE()

    # init gp_model_block
    from gp_model_block import GP_model_block
    cigp_model_block = GP_model_block()
    cigp_model_block.dnm = data_norm_manager
    cigp_model_block.gp_model = cigp

    # init optimizer, optimizer is also outsider of the model
    lr_dict = {'kernel': 0.01, 'noise': 0.01}
    params_dict = cigp_model_block.get_train_params()
    optimizer_dict = [{'params': params_dict[_key], 'lr': lr_dict[_key]} for _key in params_dict.keys()]
    optimizer = torch.optim.Adam(optimizer_dict)
    
    # start training
    max_epoch=100
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        loss = cigp_model_block.compute_loss(cigp_train_inputs, cigp_train_outputs)
        print('epoch {}/{}, loss_nll: {}'.format(epoch+1, max_epoch, loss.item()), end='\r')
        loss.backward()
        optimizer.step()
    print("low fidelity model training finished")


    # getting nonsubset dataset
    yl_with_nonsubset = cigp_model_block.predict_with_detecing_subset([train_inputs[0]])
    yl_with_nonsubset[0].reg_func(torch.reshape, source_shape)
    train_inputs[1] = yl_with_nonsubset[0]
    data_norm_manager = Dateset_normalize_manager(train_inputs, train_outputs)

    # init model
    test_config = {}
    test_config['output_shape'] = train_outputs[0][0,...].shape
    hogp = HOGP_MODULE(test_config)

    from modules.l2h_module.matrix import Matrix_l2h
    matrix_config = {}
    matrix_config['l_shape'] = train_inputs[1].mean[0,...].shape
    matrix_config['h_shape'] = train_outputs[0][0,...].shape
    matrix_l2h_modules = Matrix_l2h(matrix_config)


    # init gp_model_block
    from gp_model_block import GP_model_block
    gp_model_block = GP_model_block()
    gp_model_block.dnm = data_norm_manager
    gp_model_block.gp_model = hogp
    gp_model_block.pre_process_block = matrix_l2h_modules
    gp_model_block.post_process_block = matrix_l2h_modules

    # init optimizer, optimizer is also outsider of the model
    lr_dict = {'kernel': 0.01, 'noise': 0.01, 'others': 0.01, 'matrix': 0.01, 'rho': 0.01}
    params_dict = gp_model_block.get_train_params()
    optimizer_dict = [{'params': params_dict[_key], 'lr': lr_dict[_key]} for _key in params_dict.keys()]
    optimizer = torch.optim.Adam(optimizer_dict)
    
    # start training
    max_epoch=100
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        loss = gp_model_block.compute_loss(train_inputs, train_outputs)
        print('epoch {}/{}, loss_nll: {}'.format(epoch+1, max_epoch, loss.item()), end='\r')
        loss.backward()
        optimizer.step()

    # predict
    predict_yl = cigp_model_block.predict_with_detecing_subset([eval_inputs[0]])
    predict_yl[0].reg_func(torch.reshape, source_shape)
    predict_y = gp_model_block.predict_with_detecing_subset([eval_inputs[0], predict_yl[0]])

    # eval
    print('\n')
    gp_model_block.eval()
    predict_y = gp_model_block.predict(eval_inputs)
    plot_result(eval_outputs[0], predict_y, source_shape)



if __name__ == '__main__':
    gp_model_block_test()