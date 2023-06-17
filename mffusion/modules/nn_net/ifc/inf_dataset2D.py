import numpy as np
import copy
import torch
import math

dict_domains = {
   
    'Poisson' : {
        'fid_min':8,
        'fid_max':128,
        'interp':'linear',
    },
    
    'Heat' : {
        'fid_min':8,
        'fid_max':128,
        'interp':'linear',
    },
    
    'Burgers' : {
        'fid_min':16,
        'fid_max':128,
        'interp':'cubic',
    },
}

class MFData2D:
    def __init__(self,
                 dataset
                ):

        fidility_len = len(dataset[0])
        fid_list_tr = [math.sqrt(output.shape[1]) for output in dataset[1]]
        fid_list_te = [math.sqrt(output.shape[1]) for output in dataset[3]]
        # fid_list_tr = [output.shape[1] for output in dataset[1]]
        # fid_list_te = [output.shape[1] for output in dataset[3]]

        fid_min=int(min(fid_list_tr+fid_list_te))
        fid_max=int(max(fid_list_tr+fid_list_te))
        t_min=0.
        t_max=1.

        ns_list_tr=[_tr.shape[0] for _tr in dataset[0]]
        ns_list_te=[_tr.shape[0] for _tr in dataset[2]]
        
        self._init_mappings(t_min, t_max, fid_min, fid_max)
        
        self.input_dim = dataset[0][0].shape[1]
        
        self.fid_list_tr = copy.deepcopy(fid_list_tr)
        self.fid_list_te = copy.deepcopy(fid_list_te)

        self.ns_list_tr = copy.deepcopy(ns_list_tr)
        self.ns_list_te = copy.deepcopy(ns_list_te)
        
        self.dict_fid_to_ns_tr = {}
        self.dict_fid_to_ns_te = {}
        for fid, ns in zip(self.fid_list_tr, self.ns_list_tr):
            self.dict_fid_to_ns_tr[fid] = ns
        for fid, ns in zip(self.fid_list_te, self.ns_list_te):
            self.dict_fid_to_ns_te[fid] = ns

        self.t_list_tr = [self.func_fid_to_t(fid) for fid in self.fid_list_tr]
        self.t_list_te = [self.func_fid_to_t(fid) for fid in self.fid_list_te]

        #print(self.t_list_tr)
        #print(self.t_list_te)

        assert len(self.fid_list_tr) == len(self.ns_list_tr)
        assert len(self.fid_list_te) == len(self.ns_list_te)
        
    def _init_mappings(self, t_min, t_max, fid_min, fid_max):
        
        self.fid_min = fid_min
        self.fid_max = fid_max
        self.t_min = t_min
        self.t_max = t_max
        
        self.func_fid_to_t = lambda fid: \
            (fid-fid_min)*(t_max-t_min)/(fid_max-fid_min)
        
        self.func_t_to_fid = lambda t: \
            round((t-t_min)*(fid_max-fid_min)/(t_max-t_min) + fid_min)
        
        self.func_t_to_idx = lambda t: \
            round((t-t_min)*(fid_max-fid_min)/(t_max-t_min))
        
        fid_list_all = [fid for fid in range(self.fid_min, self.fid_max+1)]
        #cprint('r', self.fid_list)
        
        # sanity check 
        t_steps = 100
        t_span = np.linspace(t_min, t_max, t_steps)
        for i in range(t_span.size):
            t = t_span[i]
            fid = self.func_t_to_fid(t)
            idx = self.func_t_to_idx(t)
            t_rev = self.func_fid_to_t(fid)
            #cprint('r', '{:3f}-{}-{}'.format(t, fid, fid_list_all[idx]))
            err_t = np.abs(t-t_rev)
            #cprint('b', '{:.5f}-{:.5f}-{:.5f}'.format(t, t_rev, err_t))
            if fid != fid_list_all[idx]:
                raise Exception('Check the mappings of fids')
            #
            if err_t >= (t_max-t_min)/(fid_max-fid_min):
                raise Exception('Check the mappings of t')
            
    def wrap(self, inputs, outputs, type):
        if type.lower() == 'train':
            in_dict = {}
            out_dict = {}
            t_list = copy.deepcopy(self.t_list_tr)
            t_list = [torch.tensor(t, dtype=torch.double) for t in t_list]

            for i in range(len(t_list)):
                in_dict[self.fid_list_tr[i]] = inputs[i]
                out_dict[self.fid_list_tr[i]] = outputs[i]

        elif type.lower() in ['test', 'eval']:
            in_dict = {}
            out_dict = {}
            t_list = copy.deepcopy(self.t_list_te)
            t_list = [torch.tensor(t, dtype=torch.double) for t in t_list]

            for i in range(len(t_list)):
                in_dict[self.fid_list_te[i]] = inputs[i]
                out_dict[self.fid_list_te[i]] = outputs[i]

        else:
            raise Exception('Invalid type')

        return in_dict, out_dict, t_list