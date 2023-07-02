import os
import sys

import time
import datetime
import torch
import numpy as np

from mffusion.modules.nn_net.ifc.inf_fid_gpt_2d import InfFidNet2D
from mffusion.modules.nn_net.ifc.inf_dataset2D import MFData2D
from mffusion.utils import *
from mffusion.utils.mlgp_result_record import MLGP_recorder, MLGP_record_parser

def prepare_data():
    # prepare data
    head_data_dir = lambda dfp: os.path.join('..', 'data', 'sample', dfp)
    x = np.load(head_data_dir('input.npy'))
    y0 = np.load(head_data_dir('output_fidelity_0.npy'))[:, ::4, ::4]
    y1 = np.load(head_data_dir('output_fidelity_1.npy'))[:, ::2, ::2]
    y2 = np.load(head_data_dir('output_fidelity_2.npy'))[:, ::1, ::1]
    data_len = x.shape[0]
    source_shape = [-1, *y0.shape[1:]]

    x = torch.tensor(x).double()
    outputs = [torch.tensor(y0).double(), torch.tensor(y1).double(), torch.tensor(y2).double()]
    # outputs = [torch.tensor(y0).float(), torch.tensor(y2).float()]
    outputs = [y.reshape(data_len, -1) for y in outputs]

    train_outputs = [y[:128,:] for y in outputs]
    train_inputs = [x[:128,:]]* len(train_outputs)
    eval_outputs = [y[128:,:] for y in outputs]
    eval_inputs = [x[128:,:]]* len(eval_outputs)
    return train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape


def plot_result(ground_true_y, predict_y, src_shape):
    # plot result
    from mffusion.visualize_tools.plot_field import plot_container
    from mffusion.utils.type_define import GP_val_with_var
    if isinstance(predict_y[0], GP_val_with_var):
        data_list = [ground_true_y, predict_y[0].get_mean(), (ground_true_y - predict_y[0].get_mean()).abs()]
    else:
        data_list = [ground_true_y, predict_y, (ground_true_y - predict_y).abs()]
    data_list = [_d.reshape(src_shape) for _d in data_list]
    label_list = ['groundtruth', 'predict', 'diff']
    pc = plot_container(data_list, label_list, sample_dim=0)
    pc.plot()


class ExpConfigGPT():
    h_dim=5 

    int_steps=2
    solver='dopri5'

    g_width=40
    g_depth=2
    
    f_width=40
    f_depth=2
    
    kernel='RBF'
    max_lr=1e-2
    min_lr=1e-3
    test_interval=10
    device = 'cpu'
    verbose=False
    interp='bilinear'


def ifc_gpt_2d_test(dataset, exp_config):
    # setting record
    recorder = exp_config['recorder']
    start_time = time.time()

    # get dataset
    ifc_dataset = MFData2D(dataset)
    train_inputs, train_outputs, eval_inputs, eval_outputs, source_shape = dataset

    from mffusion.utils.normalizer import Dateset_normalize_manager
    data_norm_manager = Dateset_normalize_manager(train_inputs, train_outputs)
    train_inputs, train_outputs = data_norm_manager.normalize_all(train_inputs, train_outputs)
    eval_inputs, eval_outputs = data_norm_manager.normalize_all(eval_inputs, eval_outputs)

    ifc_config = ExpConfigGPT()
    ifc_model = InfFidNet2D(
        in_dim    = ifc_dataset.input_dim,
        h_dim     = ifc_config.h_dim,
        s_dim     = ifc_dataset.fid_max,
        int_steps = ifc_config.int_steps,
        solver    = ifc_config.solver,
        dataset   = ifc_dataset,
        g_width   = ifc_config.g_width,
        g_depth   = ifc_config.g_depth,
        f_width   = ifc_config.f_width,
        f_depth   = ifc_config.f_depth,
        ode_t     = True,
        interp    = ifc_config.interp,
    ).to(ifc_config.device)
    ifc_model.double()
    
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    optimizer = Adam(ifc_model.parameters(), lr=ifc_config.max_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=ifc_config.min_lr)
    
    max_epochs = exp_config['max_epoch']
    Xtr_list, ytr_list, t_list_tr = ifc_dataset.wrap(train_inputs, train_outputs, 'train')
    Xte_list, yte_list, t_list_te = ifc_dataset.wrap(eval_inputs, eval_outputs, 'eval')
    for ie in range(max_epochs):
        loss = ifc_model.eval_nelbo(Xtr_list, ytr_list, t_list_tr)
        print('epoch: {}    loss: {}'.format(ie, loss), end='\r')

        if ie % ifc_config.test_interval == 0 or (ie == max_epochs - 1):
            
            rmse_list_tr, adjust_rmse = ifc_model.eval_rmse(
                Xtr_list, ytr_list, t_list_tr, t_list_tr, return_adjust=True)
            
            # rmse_list_te = ifc_model.eval_rmse(Xte_list, yte_list, t_list_te, t_list_tr)
            
            # mae_list_tr = ifc_model.eval_mae(Xtr_list, ytr_list, t_list_tr, t_list_tr) 
            # mae_list_te = ifc_model.eval_mae(Xte_list, yte_list, t_list_te, t_list_tr)
            
            pred_dict = ifc_model.eval_pred(Xte_list, t_list_te, t_list_tr)
            # plot_result(eval_outputs[-1], predict_y, source_shape)
            

            from utils.performance_evaluator import performance_evaluator
            src_eval_output = data_norm_manager.denormalize_output(eval_outputs[-1], -1)
            pred_output = pred_dict[ifc_dataset.fid_list_tr[-1]]
            pred_output = data_norm_manager.denormalize_output(torch.tensor(pred_output).reshape_as(eval_outputs[-1]), -1)

            eval_result = performance_evaluator(src_eval_output, pred_output, ['rmse', 'r2'])
            eval_result['time'] = time.time()-start_time
            eval_result['train_sample_num'] = train_inputs[0].shape[0]
            eval_result['epoch'] = ie
            recorder.record(eval_result)  

            scheduler.step(adjust_rmse)

        #

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    return


if __name__ == "__main__":
    exp_name = os.path.join('exp', 'ifc_gpt_2d', 'toy_data', str(datetime.date.today()), 'result.txt')
    recorder = MLGP_recorder(exp_name, overlap=True)
    recorder.register(['train_sample_num','epoch','rmse', 'r2', 'time'])
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
        ifc_gpt_2d_test(sub_dataset, exp_config)

    recorder.to_csv()