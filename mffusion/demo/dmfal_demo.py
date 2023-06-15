import os
import sys

import time
import datetime
import torch
import numpy as np

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from modules.nn_net.dmfal.dmfal import DeepMFnet
from utils import *
from utils.mlgp_result_record import MLGP_recorder, MLGP_record_parser


def prepare_data():
    # prepare data
    x = np.load('./data/sample/input.npy')
    y0 = np.load('./data/sample/output_fidelity_0.npy')
    y1 = np.load('./data/sample/output_fidelity_1.npy')
    y2 = np.load('./data/sample/output_fidelity_2.npy')
    data_len = x.shape[0]
    source_shape = [-1, *y0.shape[1:]]

    x = torch.tensor(x).float()
    outputs = [torch.tensor(y0).float(), torch.tensor(y1).float(), torch.tensor(y2).float()]
    # outputs = [torch.tensor(y0).float(), torch.tensor(y2).float()]
    outputs = [y.reshape(data_len, -1) for y in outputs]

    train_inputs = [x[:128,:]]
    train_outputs = [y[:128,:] for y in outputs]
    eval_inputs = [x[128:,:]]
    eval_outputs = [y[128:,:] for y in outputs]
    return train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape


def plot_result(ground_true_y, predict_y, src_shape):
    # plot result
    from visualize_tools.plot_field import plot_container
    from utils.type_define import GP_val_with_var
    if isinstance(predict_y[0], GP_val_with_var):
        data_list = [ground_true_y, predict_y[0].get_mean(), (ground_true_y - predict_y[0].get_mean()).abs()]
    else:
        data_list = [ground_true_y, predict_y, (ground_true_y - predict_y).abs()]
    data_list = [_d.reshape(src_shape) for _d in data_list]
    label_list = ['groundtruth', 'predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()


def dmfal_test(dataset, exp_config):
    # setting record
    recorder = exp_config['recorder']
    start_time = time.time()

    # get dataset
    train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape = dataset

    from utils.normalizer import Dateset_normalize_manager
    data_norm_manager = Dateset_normalize_manager(train_inputs, train_outputs)
    train_inputs, train_outputs = data_norm_manager.normalize_all(train_inputs, train_outputs)
    eval_inputs, _ = data_norm_manager.normalize_all(eval_inputs, eval_outputs)

    dmfal_config = {
        # according to original inplement
        # h_w, h_d determine laten dim
        # net_param
        # 'M': 2,
        'nn_param': {
            'hlayers_w': [40],
            'hlayers_d': [2],
            'base_dim': [32],
            'activation': 'relu', # ['tanh','relu','sigmoid']
            # 'out_shape': [(100,1000), (100, 2000)],
            # 'in_shape': [(100, 5)],
        },
    }

    _in_shape = []
    for _i in range(len(train_inputs)):
        _in_shape.append(train_inputs[_i].shape)
    dmfal_config['nn_param']['in_shape'] = _in_shape
    
    _out_shape = []
    for _i in range(len(train_outputs)):
        _out_shape.append(train_outputs[_i].shape)
    dmfal_config['nn_param']['out_shape'] = _out_shape

    # extend nn_param as fidilety len
    dmfal_config['nn_param']['hlayers_w'] = dmfal_config['nn_param']['hlayers_w'] * len(train_outputs)
    dmfal_config['nn_param']['hlayers_d'] = dmfal_config['nn_param']['hlayers_d'] * len(train_outputs)
    dmfal_config['nn_param']['base_dim'] = dmfal_config['nn_param']['base_dim'] * len(train_outputs)

    dmfal_model = DeepMFnet(dmfal_config)

    # init optimizer, optimizer is also outsider of the model
    lr = 0.1
    params = dmfal_model.get_train_params()['params']
    optimizer = torch.optim.Adam([{'params': _v, 'lr': lr} for _v in params])
    
    max_epoch=exp_config['max_epoch']
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        loss = dmfal_model.compute_loss(train_inputs, train_outputs)
        print('epoch {}/{}, loss_nll: {}'.format(epoch+1, max_epoch, loss.item()), end='\r')
        loss.backward(retain_graph=True)
        optimizer.step()

    # predict
    predict_y = dmfal_model.predict(eval_inputs)
    predict_y = data_norm_manager.denormalize_output(predict_y, -1)
    # plot_result(eval_outputs[-1], predict_y, source_shape)

    from utils.performance_evaluator import performance_evaluator
    eval_result = performance_evaluator(eval_outputs[-1], predict_y, ['rmse', 'r2'])
    eval_result['time'] = time.time()-start_time
    eval_result['train_sample_num'] = train_inputs[0].shape[0]
    recorder.record(eval_result)    



if __name__ == '__main__':
    exp_name = os.path.join('exp', 'dmfal', 'toy_data', str(datetime.date.today()), 'result.txt')
    recorder = MLGP_recorder(exp_name, overlap=True)
    recorder.register(['train_sample_num','rmse', 'r2', 'time'])
    exp_config = {
        'max_epoch': 100,
        'recorder': recorder,
    }

    dataset = prepare_data()
    train_sample_num = [16,32,64,128]
    for _num in train_sample_num:
        sub_dataset = []
        # subset on train data
        for _data in dataset[:2]:
            data_list = []
            for _d in _data:
                data_list.append(_d[:_num])
            sub_dataset.append(data_list)

        sub_dataset.extend(dataset[2:])
        dmfal_test(sub_dataset, exp_config)

    recorder.to_csv()