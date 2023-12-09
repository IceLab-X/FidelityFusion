import os
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch

from MFGP.utils.mfgp_log import MFGP_LOG


class plot_container:
    def __init__(self, data_list, label_list, sample_dim) -> None:
        # data_list: containing data want to plot. suppose to be 3D(2D field + sample dim)
        # label_list: label attach to title
        # sample dim: set which dim is sample dim

        # check 2D, now support 2D plot
        shape=None
        for i,_d in enumerate(data_list):
            assert len(_d.shape)==3, "now support 2D field, but got {} for data-{}".format(_d.shape, i)
            if shape is not None:
                assert shape == _d.shape, "shape should be all same as {}".format(shape)
            else:
                shape = _d.shape

        self.data_list = data_list
        self.label_list = label_list
        self.sample_dim = sample_dim

        # compatability for torch tensor
        for i,_d in enumerate(self.data_list):
            if isinstance(_d, torch.Tensor):
                self.data_list[i] = _d.detach().cpu().numpy()

        self.sample_number = data_list[0].shape[sample_dim]
        self.data_number = len(data_list)
        self.dims = data_list[0].shape

        self.field_dims = [i for i in range(len(data_list[0].shape))]
        self.field_dims.remove(sample_dim)
        self.field_dims = tuple(self.field_dims)

        #['flat', 'nearest', 'gouraud', 'auto', 'jet']
        self.map_type = 'jet'
        #['flat', 'nearest', 'gouraud', 'auto']
        self.shading_method = 'gouraud'

        self.get_min_max()
        self.filter_outsider_value()


    def get_min_max(self):
        min_list = [_d.min(self.field_dims) for _d in self.data_list]
        max_list = [_d.max(self.field_dims) for _d in self.data_list]

        self.min_list = np.vstack(min_list).min(0)
        self.max_list = np.vstack(max_list).max(0)

        self.min_max_base_on_all_sample = False
        # if false, min_max get from everysample


    def filter_outsider_value(self):
        self.min_clamp = []
        self.max_clamp = []

        outsider_rate = 0.01
        for i in range(self.data_number):
            outsider_num = int(self.data_list[i].size * outsider_rate)
            default_bins = self.data_list[i].size//1000
            hist, bin_range = np.histogram(self.data_list[i],bins=default_bins)

            hist_sum = np.cumsum(hist)
            for j,_h in enumerate(hist_sum):
                if _h >= outsider_num:
                    self.min_clamp.append(bin_range[j])
                    break

            for j,_h in enumerate(hist_sum[::-1]):
                if (self.data_list[i].size - _h) >= outsider_num:
                    self.max_clamp.append(bin_range[min(-j,-1)])
                    break

            src_min = bin_range[0]
            src_max = bin_range[-1]
            # MFGP_LOG.w('data-{}: min {} clamp to {}, max {} clamp to {}'.format(i, src_min, self.min_clamp[-1], src_max, self.max_clamp[-1]))

        self.min_clamp = np.vstack(self.min_clamp).min(0)
        self.max_clamp = np.vstack(self.max_clamp).max(0)
        self.min_list = np.clip(self.min_list, self.min_clamp, self.max_clamp)
        self.max_list = np.clip(self.max_list, self.min_clamp, self.max_clamp)
            

    
    def plot(self, limit_frame=-1):
        # plot for each sample
        MFGP_LOG.i('Data get shape {}. Dim-{} is regarded as sample and the others are field.'.format(self.data_list[0].shape, self.sample_number))
        for j in range(self.sample_number):
            MFGP_LOG.i('now plot {}/{}'.format(j+1, self.sample_number), end='\r')
            fig, axs = plt.subplots(nrows=1, ncols=self.data_number, figsize = (10, 5.5))

            if limit_frame != -1:
                if j >= limit_frame:
                    break
    
            for i in range(self.data_number):

                _d = self.data_list[i]
                _l = self.label_list[i]
                ax = axs[i]

                get_sample_cmd = [':']* (len(self.dims)-1)
                get_sample_cmd.insert(self.sample_dim, 'j')
                get_sample_cmd = '_d[{}]'.format(','.join(get_sample_cmd))
                _d = eval(get_sample_cmd)
                
                if self.min_max_base_on_all_sample is True:
                    pcm = ax.pcolormesh(_d, cmap=self.map_type, shading=self.shading_method, vmin=self.min_list.min(), vmax=self.max_list.max())
                else:
                    pcm = ax.pcolormesh(_d, cmap=self.map_type, shading=self.shading_method, vmin=self.min_list[j], vmax=self.max_list[j])
                ax.tick_params(labelsize = 8)
                ax.set_title(str(self.label_list[i]), fontsize = 8)

            fig.tight_layout()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)
            plt.colorbar(pcm, cax=cax)
            # plt.rcParams.update({'font.size': 30})
            plt.tight_layout()
            plt.show()
            # input("AnyKey to continue")
            # plt.savefig(r'fig_new/' + f + '_' + str(sample_index) +'.eps', bbox_inches = 'tight')



if __name__ == '__main__':
    a = np.load('g.npy').reshape(-1,100,100)
    b = np.load('p.npy').reshape(-1,100,100)
    pc =plot_container([a,b,abs(a-b)], ['fidelity-0', 'fidelity-2', 'diff'], 0)
    
    pc.plot()
