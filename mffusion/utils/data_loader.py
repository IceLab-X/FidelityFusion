import numpy as np
from scipy.io import loadmat
import torch
from copy import deepcopy
torch.set_default_tensor_type(torch.DoubleTensor)


def _smart_path(path):
    # TODO : add real path parse
    return path

# fisrt for train, second for test
# int for exact number, point for percentage 
default_slice_param = [0.6, 0.4]

def np_list_to_tensor_list(np_list):
    return [torch.from_numpy(np_list[i]).float() for i in range(len(np_list))]

def dict_pattern(path, function, interp_available):
    return {'path': path, 'function': function, 'interp_available': interp_available}

def _concat_on_new_last_dim(arrays):
    # arr = []
    # for _array in arrays:
    #     tt = _array.reshape(_array.shape[0], -1)
    #     arr.append(tt)
    # tem = np.concatenate(arr, axis=-1)
    # return tem
    arrays = [_array.reshape(*_array.shape, 1) for _array in arrays]
    return np.concatenate(arrays, axis=-1)

def _force_2d(arrays):
    N = arrays.shape[0]
    n = arrays[0].shape[0] * arrays[0].shape[1]
    return np.reshape(arrays, (N, n))

def _get_format_slice_data(length, slice):
    if isinstance(slice[0], int):
        return slice[0], slice[1]

    elif isinstance(slice[0], float):
        assert slice[0] + slice[1] == 1, 'slice sum should be 1'
        _tr = int(length * slice[0])
        _te = int(length * slice[1])
        while _tr + _te > length:
            _tr -= 1
            _te -= 1
        return _tr, _te

def _flatten_inputs(inputs):
    _len_list = [len(_l) for _l in inputs]
    _temp_array_list = []
    [_temp_array_list.extend(deepcopy(_l)) for _l in inputs]
    return _temp_array_list, _len_list

def _reformat_inputs(array_list, _len_list):
    _index = 0
    outputs = []
    for _len in _len_list:
        outputs.append(array_list[_index:_index+_len])
        _index += _len
    return outputs


class SP_DataLoader(object):
    dataset_available = ['FlowMix3D_MF',
                         'MolecularDynamic_MF', 
                         'plasmonic2_MF', 
                         'SOFC_MF',
                         'NavierStock_mfGent_v1_02',
                         ]
    def __init__(self, dataset_name, need_interp=False) -> None:
        self.dataset_info = {
            'FlowMix3D_MF': 
                dict_pattern('data/MF_data/FlowMix3D_MF.mat', self._general, False),
            'MolecularDynamic_MF': 
                dict_pattern('data/MF_data/MolecularDynamic_MF.mat', self._general, False),
            'plasmonic2_MF': 
                dict_pattern('data/MF_data/plasmonic2_MF.mat', self._general, False),
            'SOFC_MF': 
                dict_pattern('data/MF_data/SOFC_MF.mat', self._SOFC_MF, False),
            'NavierStock_mfGent_v1_02':
                dict_pattern('data/NavierStock_mfGent_v1_02', self._NavierStock_mfGent_v1_02, False),
            } 
        # self.dataset_info = {
        #     'FlowMix3D_MF': 
        #         dict_pattern('data\MF_data\FlowMix3D_MF.mat', self._general, False),
        #     'MolecularDynamic_MF': 
        #         dict_pattern('data\MF_data\MolecularDynamic_MF.mat', self._general, False),
        #     'plasmonic2_MF': 
        #         dict_pattern('data\MF_data\plasmonic2_MF.mat', self._general, False),
        #     'SOFC_MF': 
        #         dict_pattern('data\MF_data\SOFC_MF.mat', self._SOFC_MF, False),
        #     }

        if dataset_name not in self.dataset_info:
            assert False
        if need_interp and self.dataset_info[dataset_name]['interp_available'] is False:
            assert False
        self.dataset_name = dataset_name
        self.need_interp = need_interp

    def _random_shuffle(self, inputs):
        _temp_array_list,_len_list = _flatten_inputs(inputs[0:2])
        _temp_array_list = self._random_shuffle_array_list(_temp_array_list)
        outputs = _reformat_inputs(_temp_array_list, _len_list)
        outputs.extend(inputs[2:4])
        return outputs

    def get_data(self):
        outputs = self.dataset_info[self.dataset_name]['function']()
        return outputs

    def _general(self):
        _data = loadmat(_smart_path(self.dataset_info[self.dataset_name]['path']))

        x = [torch.from_numpy(_data['X'])]
        xte = [torch.from_numpy(_data['X'][-int(x[0].shape[0] * 0.4) : -1])]
        y = []
        yte = []
        for i in range(len(_data['Y'][0])):
            y.append(torch.from_numpy(_data['Y'][0][i]))
            yte.append(torch.from_numpy(_data['Y'][0][i][-int(x[0].shape[0] * 0.4) : -1]))
        return x, y, xte, yte

    # def _FlowMix3D_MF(self):
    #     return self._general()

    # def _MolecularDynamic_MF(self):
    #     return self._general()

    # def _plasmonic2_MF(self):
    #     return self._general()

    def _NavierStock_mfGent_v1_02(self):
        import h5py
        field_need = ['PRec', 'URec', 'VRec']
        x = []
        y = []
        for i in range(5):
            _data = h5py.File(_smart_path(self.dataset_info[self.dataset_name]['path'] + '/Fidelity_' + str(i+1) + '.mat'), 'r')
            x.append(_data['X'].value.T)
            _temp_y = []
            for _field_name in field_need:
                _temp_y.append(_data[_field_name].value.T)
            if len(_temp_y) > 1:
                y.append(_concat_on_new_last_dim(_temp_y))
            elif len(_temp_y) == 1:
                y.append(_temp_y[0])
        return x, y, None, None

    def _SOFC_MF(self):
        _data = loadmat(_smart_path(self.dataset_info[self.dataset_name]['path']))
        x = [torch.from_numpy(_data['X'])]
        xte = [torch.from_numpy(_data['X'][0: int(x[0].shape[0] * 0.4)])]
        y = []
        yte = []
        for i in range(len(_data['Y1'][0])):
            y.append(torch.from_numpy(_concat_on_new_last_dim([_data['Y1'][0][i], _data['Y2'][0][i]])))
            yte.append(torch.from_numpy(_concat_on_new_last_dim([_data['Y1'][0][i], _data['Y2'][0][i]])[0: int(x[0].shape[0] * 0.4)]))
        
            # if i== 0:
            #     y.append(torch.from_numpy(_data['Y2'][0][i].reshape(_data['Y2'][0][i].shape[0], -1)))
            # else:
            #     y.append(_concat_on_new_last_dim([_data['Y1'][0][i], _data['Y2'][0][i]]))
        #     import matplotlib.pyplot as plt
        #     for j in range(128):
        #         fig, axs = plt.subplots(1, 2)
        #         # axs.plot(list(range(50)),_data['Y1'][0][i][j,::100])
        #         # axs.plot(list(range(5000)),_data['Y1'][0][i][0,:])
        #         axs[0].pcolor(_data['Y1'][0][1][j,...])
        #         axs[1].pcolor(_data['Y2'][0][1][j,...])
        #         plt.show()
        return x, y, xte, yte

    def _get_distribute(self):
        pass


class Standard_mat_DataLoader(object):
    dataset_available = ['poisson_v4_02',
                        'burger_v4_02',
                        'Burget_mfGent_v5',
                        'Burget_mfGent_v5_02',
                        'Heat_mfGent_v5',
                        'Piosson_mfGent_v5',
                        'Schroed2D_mfGent_v1',
                        'TopOP_mfGent_v5',]
    def __init__(self, dataset_name, need_interp=False) -> None:
        self.dataset_info = {
            'Burget_mfGent_v5_15': dict_pattern( 'data/Burget_mfGent_v5_15.mat', self._general, True),
            'Heat_mfGent_v5': dict_pattern( 'data/Heat_mfGent_v5.mat', self._general, True),
            'Heat_mfGent_v5_15': dict_pattern( 'data/Heat_mfGent_v5_15.mat', self._general, True),
            'Poisson_mfGent_v5': dict_pattern( 'data/Poisson_mfGent_v5.mat', self._general, True),
            'Poisson_mfGent_v5_15': dict_pattern('data/Poisson_mfGent_v5_15.mat', self._general, True),
            'Burget_mfGent_v5_02': dict_pattern('data/Burget_mfGent_v5_02.mat', self._general, True),
            'TopOP_mfGent_v5': dict_pattern('data/TopOP_mfGent_v5.mat', self._general, True),
            'TopOP_mfGent_v6': dict_pattern('data/TopOP_mfGent_v6.mat', self._general, True),
                   }
        if dataset_name not in self.dataset_info:
            assert False # dataset名字打错了
        if need_interp and self.dataset_info[dataset_name]['interp_available'] is False:
            assert False # 命令和设置相矛盾
        self.dataset_name = dataset_name
        self.need_interp = need_interp
    
    def _random_shuffle(self, inputs):
        _temp_array_list,_len_list = _flatten_inputs(inputs[0:2])
        _temp_array_list = self._random_shuffle_array_list(_temp_array_list)
        outputs = _reformat_inputs(_temp_array_list, _len_list)
        outputs.extend(inputs[2:4])
        return outputs

    def _general(self):
        _data = loadmat(_smart_path(self.dataset_info[self.dataset_name]['path']))

        x_tr = [torch.from_numpy(_data['xtr'])]
        x_te = [torch.from_numpy(_data['xte'])]
        if self.need_interp is False:
            y_tr = []
            for i in range(len(_data['Ytr'][0])):
                tem = _force_2d(_data['Ytr'][0][i])
                y_tr.append(torch.from_numpy(tem))
                # y_tr.append(torch.from_numpy(_data['Ytr'][0][i]))
            y_te = []
            for i in range(len(_data['Yte'][0])):
                tem = _force_2d(_data['Yte'][0][i])
                y_te.append(torch.from_numpy(tem))
                # y_tr.append(torch.from_numpy(_data['Yte'][0][i]))
        else:
            y_tr = []
            for i in range(len(_data['Ytr_interp'][0])):
                tem = _force_2d(_data['Ytr_interp'][0][i])
                y_tr.append(torch.from_numpy(tem))
                # y_tr.append(torch.from_numpy(_data['Ytr_interp'][0][i]))
            y_te = []
            for i in range(len(_data['Yte_interp'][0])):
                tem = _force_2d(_data['Yte_interp'][0][i])
                y_te.append(torch.from_numpy(tem))
                # y_tr.append(torch.from_numpy(_data['Yte_interp'][0][i]))
        return x_tr, y_tr, x_te, y_te

    def get_data(self):
        outputs = self.dataset_info[self.dataset_name]['function']()
        return outputs

    def missing_data(self, mis_index):
        # x_tr, y_tr, x_te, y_te = self.dataset_info[self.dataset_name]['function']()
        x_tr, y_tr, x_te, y_te = self.get_data()
        if len(mis_index) != len(y_tr):
            assert False

        xtr = []
        xte = []
        N = torch.tensor([np.NaN for i in range(x_tr[0].shape[1])])
        for i in range(1, len(mis_index)+1):
            missing_index = mis_index[i]
            if mis_index[i] == None:
                tem_tr = torch.clone(x_tr[0])
                xtr.append(tem_tr)
                tem_te = torch.clone(x_te[0])
                xte.append(tem_te)
            else:
                for j in missing_index:
                    tr = torch.clone(x_tr[0])
                    tr[j[0]: j[1]] = torch.stack([N for i in range(j[0], j[1])])

                    te = torch.clone(x_te[0])
                    te[j[0]: j[1]] = torch.stack([N for i in range(j[0], j[1])])
                xtr.append(tr)
                xte.append(te)

        ytr = []
        yte = []
        for i in range(1, len(mis_index)+1):
            missing_index = mis_index[i]
            N = torch.tensor([np.NaN for i in range(y_tr[i-1].shape[1])])
            if mis_index[i] == None:
                tem_tr = torch.clone(y_tr[i-1])
                ytr.append(tem_tr)
                tem_te = torch.clone(y_te[i-1])
                yte.append(tem_te)
            else:
                for j in missing_index:
                    tem_tr = torch.clone(y_tr[i-1])
                    tem_tr[j[0]: j[1]] = torch.stack([N for i in range(j[0], j[1])])
                    
                    tem_te = torch.clone(y_te[i-1])
                    tem_te[j[0]: j[1]] = torch.stack([N for i in range(j[0], j[1])])
                ytr.append(tem_tr)
                yte.append(tem_te)

        return xtr, ytr, xte, yte



if __name__ == '__main__':
    sp_data = SP_DataLoader('NavierStock_mfGent_v1_02', None)
    # sp_data = SP_DataLoader('SOFC_MF', None)
    print(sp_data.get_data())

    # stand_data = Standard_mat_DataLoader('poisson_v4_02')
    # print(stand_data.get_data())

    mat_data = Standard_mat_DataLoader('Poisson_mfGent_v5', True)
    mis_index = {1: None, 2: ((0, 14)), 3: ((7, 17)), 4:((0, 2),(7, 64)), 5:((16, 64))}
    xtr, ytr, xte, yte = mat_data.get_data()



    pass