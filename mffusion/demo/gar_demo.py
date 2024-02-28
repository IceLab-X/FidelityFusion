# Base on hogp, train on x, y_res
# y_res = y_high - tensorly.mode.dot(y_low, metrix).


import os
import sys

import datetime
import time
import torch
import numpy as np

from mffusion.modules.gp_module.hogp import HOGP_MODULE
from mffusion.modules.gp_module.cigp import CIGP_MODULE

def prepare_data(_num):
    # prepare data
    head_data_dir = lambda dfp: os.path.join('..', 'data', 'sample', dfp)
    x = np.load(head_data_dir('input.npy'))
    yl = np.load(head_data_dir('output_fidelity_1.npy'))
    yh = np.load(head_data_dir('output_fidelity_2.npy'))
    source_shape = [-1, *yh.shape[1:]]

    x = torch.tensor(x).float()
    yl = torch.tensor(yl).float()
    yh = torch.tensor(yh).float()

    train_inputs = [x[:_num,:], yl[:_num,:]]
    train_outputs = [yh[:_num,:]]
    eval_inputs = [x[128:,:], yl[128:,:]]
    eval_outputs = [yh[128:,:]]
    return train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape


def plot_result(ground_true_y, predict_y, src_shape):
    # plot result
    from mffusion.visualize_tools.plot_field import plot_container
    data_list = [ground_true_y, predict_y[0].get_mean(), (ground_true_y - predict_y[0].get_mean()).abs()]
    data_list = [_d.reshape(src_shape) for _d in data_list]
    label_list = ['groundtruth', 'predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()


def gp_model_block_test(_num, exp_config):
    # setting record
    recorder = exp_config['recorder']
    start_time = time.time()

    # get dataset
    train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape = prepare_data(_num)

    # normalizer now is outsider of the model
    from mffusion.utils.normalizer import Dateset_normalize_manager
    data_norm_manager = Dateset_normalize_manager(train_inputs, train_outputs)

    # init model
    test_config = {}
    test_config['output_shape'] = train_outputs[0][0,...].shape
    hogp = HOGP_MODULE(test_config)

    from mffusion.modules.l2h_module.matrix import Matrix_l2h
    matrix_config = {}
    matrix_config['l_shape'] = train_inputs[1][0,...].shape
    matrix_config['h_shape'] = train_outputs[0][0,...].shape
    matrix_l2h_modules = Matrix_l2h(matrix_config)

    # init gp_model_block
    from mffusion.gp_model_block import GP_model_block
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

    # eval
    print('\n')
    gp_model_block.eval()
    predict_y = gp_model_block.predict(eval_inputs)
    plot_result(eval_outputs[0], predict_y, source_shape)

    from mffusion.utils.performance_evaluator import performance_evaluator
    eval_result = performance_evaluator(eval_outputs[0], predict_y[0].mean, ['rmse', 'r2'])
    eval_result['time'] = time.time() - start_time
    eval_result['train_sample_num'] = train_inputs[0].shape[0]
    recorder.record(eval_result)



if __name__ == '__main__':
    exp_name = os.path.join('mffusion', 'exp', 'gar', 'toy_data', str(datetime.date.today()), 'result.txt')

    from mffusion.utils.mlgp_result_record import MLGP_recorder
    recorder = MLGP_recorder(exp_name, overlap=True)
    recorder.register(['train_sample_num','rmse', 'r2', 'time'])
    exp_config = {
        'max_epoch': 100,
        'recorder': recorder,
    }
    
    train_sample_num = [4, 8, 16, 32]
    for _num in train_sample_num:
        # last one is shape
        gp_model_block_test(_num, exp_config)

    recorder.to_csv()
