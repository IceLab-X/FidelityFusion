import os
import sys

import datetime
import time
import torch
import numpy as np

realpath=os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('ML_gp')+1])
sys.path.append(realpath)

from modules.gp_module.fides import FIDES_MODULE
from modules.gp_module.cigp import CIGP_MODULE
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


def gp_model_block_test(dataset, exp_config):
    # setting record
    recorder = exp_config['recorder']
    start_time = time.time()

    # get dataset
    train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape = dataset

    # normalizer now is outsider of the model
    from utils.normalizer import Dateset_normalize_manager
    data_norm_manager = Dateset_normalize_manager(train_inputs, train_outputs)

    # lowest fidelity model
    # init model
    test_config = {
        'noise': 1.,
        'noise_exp_format': True,
        'kernel': {
                'K1': {'SE': {'noise_exp_format':True, 'length_scale':1., 'scale': 1.}},
              },}
    
    cigp = CIGP_MODULE(test_config)

    # init gp_model_block
    from gp_model_block import GP_model_block
    gp_model_block = GP_model_block()
    gp_model_block.dnm = data_norm_manager
    gp_model_block.gp_model = cigp

    # init optimizer, optimizer is also outsider of the model
    lr_dict = {'kernel': 0.01, 'noise': 0.01, 'others': 0.01}
    params_dict = gp_model_block.get_train_params()
    optimizer_dict = [{'params': params_dict[_key], 'lr': lr_dict[_key]} for _key in params_dict.keys()]
    optimizer = torch.optim.Adam(optimizer_dict)
    
    # start training
    max_epoch=exp_config['max_epoch']
    for epoch in range(max_epoch):
        optimizer.zero_grad()
        loss = gp_model_block.compute_loss(train_inputs, [train_outputs[0]])
        print('epoch {}/{}, loss_nll: {}'.format(epoch+1, max_epoch, loss.item()), end='\r')
        loss.backward()
        optimizer.step()

    # eval
    # print('\n')
    # gp_model_block.eval()
    # predict_y = gp_model_block.predict(eval_inputs)
    # plot_result(eval_outputs[0], predict_y, source_shape)


    '''
    training for high fidelity
    '''
    fidelity_num = len(train_outputs)
    fides = FIDES_MODULE({})

    fides_block = GP_model_block()
    fides_block.dnm = data_norm_manager
    fides_block.gp_model = fides

    # init l2h modules
    from modules.l2h_module.rho import Res_rho_l2h
    Res_rho_l2h_config = {
        'rho_value_init': 1.,
        'trainable': False,
    }
    rho_modules = Res_rho_l2h(Res_rho_l2h_config)
    fides_block.pre_process_block = rho_modules
    fides_block.post_process_block = rho_modules

    # init optimizer, optimizer is also outsider of the model
    lr_dict = {'kernel': 0.01, 'noise': 0.01, 'others': 0.01, 'rho': 0.}
    params_dict = fides_block.get_train_params()
    optimizer_dict = [{'params': params_dict[_key], 'lr': lr_dict[_key]} for _key in params_dict.keys()]
    optimizer = torch.optim.Adam(optimizer_dict)

    # start training
    max_epoch=exp_config['max_epoch']
    for _fi in range(1, fidelity_num):
        # reset normalizer
        data_norm_manager = Dateset_normalize_manager([train_inputs[0], train_outputs[_fi-1]], [train_outputs[_fi]])
        fides_block.dnm = data_norm_manager
        fides_block.gp_model.set_fidelity(_fi-1, _fi, _fi-1, _fi)
        for epoch in range(max_epoch):
            optimizer.zero_grad()
            loss = fides_block.compute_loss([train_inputs[0], train_outputs[_fi-1]], [train_outputs[_fi]])
            print('epoch {}/{}, loss_nll: {}'.format(epoch+1, max_epoch, loss.item()), end='\r')
            loss.backward()
            optimizer.step()


    # predict
    gp_model_block.eval()
    predict_y = gp_model_block.predict(eval_inputs)
    predict_y_mean = predict_y[0].mean
    # plot_result(eval_outputs[0], predict_y_mean, source_shape)

    # predict with high fidelity
    for _fi in range(1, fidelity_num):
        fides_block.gp_model.set_fidelity(_fi-1, _fi, _fi-1, _fi) 
        data_norm_manager = Dateset_normalize_manager([train_inputs[0], train_outputs[_fi-1]], [train_outputs[_fi]])
        fides_block.dnm = data_norm_manager

        fides_block.eval()
        predict_y = fides_block.predict([eval_inputs[0], predict_y_mean])
        predict_y_mean = predict_y[0]

    # plot_result(eval_outputs[len(train_outputs)-1], predict_y_mean, source_shape)

    from utils.performance_evaluator import performance_evaluator
    eval_result = performance_evaluator(eval_outputs[len(train_outputs)-1], predict_y_mean, ['rmse', 'r2'])
    eval_result['time'] = time.time()-start_time
    eval_result['train_sample_num'] = train_inputs[0].shape[0]
    recorder.record(eval_result)


if __name__ == '__main__':
    exp_name = os.path.join('exp', 'fides', 'toy_data', str(datetime.date.today()), 'result.txt')
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
        for _data in dataset[:-1]:
            data_list = []
            for _d in _data:
                data_list.append(_d[:_num])
            sub_dataset.append(data_list)
        sub_dataset.append(dataset[-1])         # last one is shape
        gp_model_block_test(sub_dataset, exp_config)

    recorder.to_csv()
