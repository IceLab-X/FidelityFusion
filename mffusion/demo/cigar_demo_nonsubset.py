# AR for auto regression
# high - low = res

import os
import sys

import torch
import numpy as np

from mffusion.modules.gp_module.cigp import CIGP_MODULE

def prepare_data():
    # prepare data
    head_data_dir = lambda dfp: os.path.join('..', 'data', 'sample', dfp)
    x = np.load(head_data_dir('input.npy'))
    y_low = np.load(head_data_dir('output_fidelity_1.npy'))
    y_high = np.load(head_data_dir('output_fidelity_2.npy'))
    
    x = torch.tensor(x).float()
    y_low = torch.tensor(y_low).float()
    y_high = torch.tensor(y_high).float()

    data_len = x.shape[0]
    source_shape = [-1, *y_low.shape[1:]]

    # cigp only support 2d input (batch, dim)
    x = x.reshape(data_len, -1)
    y_low = y_low.reshape(data_len, -1)
    y_high = y_high.reshape(data_len, -1)
    train_inputs = [x[:128,:], y_low[:128,:]]
    train_outputs = [y_high[:128,:]]
    eval_inputs = [x[128:,:], y_low[128:,:]]
    eval_outputs = [y_high[128:,:]]
    return train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape


def plot_result(ground_true_y, predict_y, src_shape):
    # plot result
    from mffusion.visualize_tools.plot_field import plot_container
    data_list = [ground_true_y, predict_y[0], (ground_true_y - predict_y[0]).abs()]
    data_list = [_d.reshape(src_shape) for _d in data_list]
    label_list = ['groundtruth', 'predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()

def half(x):
    return x[:x.shape[0]//2]

def gp_model_block_test():
    # prepare data
    train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape = prepare_data()
    train_inputs[1] = train_inputs[1].flatten(1)
    eval_inputs[1] = eval_inputs[1].flatten(1)

    # prepare nonsubset
    cigp_train_inputs = [half(train_inputs[0])]
    cigp_train_outputs = [half(train_inputs[1])]

    # init low fidelity model
    cigp = CIGP_MODULE()

    # normalizer now is outsider of the model
    from mffusion.utils.normalizer import Dateset_normalize_manager
    data_norm_manager = Dateset_normalize_manager(cigp_train_inputs, cigp_train_outputs)

        # init gp_model_block
    from mffusion.gp_model_block import GP_model_block
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
    # yl_with_nonsubset[0].reg_func(torch.reshape, source_shape)
    train_inputs[1] = yl_with_nonsubset[0]
    data_norm_manager = Dateset_normalize_manager(train_inputs, train_outputs)

    # init model
    cigar = CIGP_MODULE()

    # init l2h modules
    from mffusion.modules.l2h_module.matrix import Matrix_l2h
    matrix_config = {}
    matrix_config['l_shape'] = train_inputs[1].mean[0,...].shape
    matrix_config['h_shape'] = train_outputs[0][0,...].shape
    matrix_l2h_modules = Matrix_l2h(matrix_config)

    # init gp_model_block
    gp_model_block = GP_model_block()
    gp_model_block.dnm = data_norm_manager
    gp_model_block.gp_model = cigar
    gp_model_block.pre_process_block = matrix_l2h_modules
    gp_model_block.post_process_block = matrix_l2h_modules

    # init optimizer, optimizer is also outsider of the model
    lr_dict = {'kernel': 0.01, 'noise': 0.01, 'others': 0.01, 'matrix': 0.01, 'rho': 0.01}
    params_dict = gp_model_block.get_train_params()
    optimizer_dict = [{'params': params_dict[_key], 'lr': lr_dict[_key]} for _key in params_dict.keys()]
    optimizer = torch.optim.Adam(optimizer_dict)
    
    max_epoch=300
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        loss = gp_model_block.compute_loss(train_inputs, train_outputs)
        print('epoch {}/{}, loss_nll: {}'.format(epoch+1, max_epoch, loss.item()), end='\r')
        loss.backward()
        optimizer.step()

    print('\n')
    gp_model_block.eval()
    # predict
    predict_yl = cigp_model_block.predict_with_detecing_subset([eval_inputs[0]])
    predict_y = gp_model_block.predict([eval_inputs[0], predict_yl[0]])


    from mffusion.utils.type_define import GP_val_with_var
    if isinstance(predict_y[0], GP_val_with_var):
        predict_y = [predict_y[0].mean]

    # plot result
    plot_result(eval_outputs[0], predict_y, source_shape)


if __name__ == '__main__':
    gp_model_block_test()
