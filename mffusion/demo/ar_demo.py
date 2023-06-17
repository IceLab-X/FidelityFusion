# AR for auto regression. Base on cigp. Train on x,y_res.
# y_res = (y_high - y_low*rho)

import os
import sys

import torch
import numpy as np


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
    train_inputs = [x[:128, :], y_low[:128, :]]
    train_outputs = [y_high[:128, :]]
    eval_inputs = [x[128:, :], y_low[128:, :]]
    eval_outputs = [y_high[128:, :]]
    return train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape


def plot_result(ground_true_y, predict_y, src_shape):
    # plot result
    from mffusion.visualize_tools.plot_field import plot_container
    data_list = [ground_true_y, predict_y[0], (ground_true_y - predict_y[0]).abs()]
    data_list = [_d.reshape(src_shape) for _d in data_list]
    label_list = ['groundtruth', 'predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()


def gp_model_block_test():
    # prepare data
    train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape = prepare_data()
    
    # normalizer now is outsider of the model
    from mffusion.utils.normalizer import Dateset_normalize_manager
    dateset_normalize_manager = Dateset_normalize_manager(train_inputs, train_outputs)
    
    # init model
    from mffusion.modules.gp_module.cigp import CIGP_MODULE
    cigp = CIGP_MODULE()
    
    # init l2h modules
    from mffusion.modules.l2h_module.rho import Res_rho_l2h
    rho_modules = Res_rho_l2h()
    
    # init gp_model_block
    from mffusion.gp_model_block import GP_model_block
    gp_model_block = GP_model_block()
    gp_model_block.dnm = dateset_normalize_manager
    gp_model_block.gp_model = cigp
    gp_model_block.pre_process_block = rho_modules
    gp_model_block.post_process_block = rho_modules
    
    # init optimizer, optimizer is also outsider of the model
    lr_dict = {'kernel': 0.01, 'noise': 0.01, 'rho': 0.01}
    params_dict = gp_model_block.get_train_params()
    optimizer_dict = [{'params': params_dict[_key], 'lr': lr_dict[_key]} for _key in params_dict.keys()]
    optimizer = torch.optim.Adam(optimizer_dict)
    
    print('rho value(init):', gp_model_block.pre_process_block.rho.data)
    max_epoch = 100
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        loss = gp_model_block.compute_loss(train_inputs, train_outputs)
        print('epoch {}/{}, loss_nll: {}'.format(epoch + 1, max_epoch, loss.item()), end='\r')
        loss.backward()
        optimizer.step()
    
    print('\n')
    gp_model_block.eval()
    predict_y = gp_model_block.predict(eval_inputs)
    print('rho value:', gp_model_block.pre_process_block.rho.data)
    
    # plot result
    plot_result(eval_outputs[0], predict_y, source_shape)


if __name__ == '__main__':
    gp_model_block_test()
