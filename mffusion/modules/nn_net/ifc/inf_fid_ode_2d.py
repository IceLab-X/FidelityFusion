import numpy as np
import random
import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchdiffeq import odeint

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# torch.set_default_tensor_type(torch.DoubleTensor)

class Net(nn.Module):
    def __init__(self, config, act=nn.Tanh()):
        
        super(Net, self).__init__()
 
        buff_layers = []

        for l in range(len(config)-2):
            in_dim = config[l]
            out_dim = config[l+1]
            buff_layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            buff_layers.append(nn.Tanh())
        #
        buff_layers.append(nn.Linear(in_features=config[-2], out_features=config[-1]))

        self.net = nn.ModuleList(buff_layers)
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, val=0)
            #
        #
        
    def forward(self, X):
        h = X
        for layer in self.net:
            h = layer(h)
        #
        return h
    
class ODEFunc(nn.Module):
    def __init__(self, config, act=nn.Tanh()):
        
        super(ODEFunc, self).__init__()
 
        buff_layers = []

        for l in range(len(config)-2):
            in_dim = config[l]
            out_dim = config[l+1]
            buff_layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            buff_layers.append(nn.Tanh())
        #
        buff_layers.append(nn.Linear(in_features=config[-2], out_features=config[-1]))

        self.net = nn.ModuleList(buff_layers)
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, val=0)
            #
        #

    def forward(self, t, y, X, t_align):
        # t is a scaler, access by ODE solver
        # y is N by d
        # x is N by p
        # t_align is N by 1
        
        assert y.ndim == X.ndim == t_align.ndim == 2

        aug_input = torch.hstack([t*t_align, y, X])
        
        h = aug_input
        for layer in self.net:
            h = layer(h)
        #
        
        h_align = t_align*h
        return h_align
    

class ODEFuncAt(nn.Module):
    def __init__(self, config, act=nn.Tanh()):
        
        super(ODEFuncAt, self).__init__()
 
        buff_layers = []

        for l in range(len(config)-2):
            in_dim = config[l]
            out_dim = config[l+1]
            buff_layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            buff_layers.append(nn.Tanh())
        #
        buff_layers.append(nn.Linear(in_features=config[-2], out_features=config[-1]))

        self.net = nn.ModuleList(buff_layers)
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, val=0)
            #
        #
        
        self.register_buffer('dummy', torch.tensor([]))

    def forward(self, t, y, t_align=None):
        
        if t_align is None:
            #cprint('r', 'solve in standard mode')
            shape_y = y.shape
            yvec = y.reshape([-1,1])
            tvec = t*torch.ones(yvec.shape).to(self.dummy.device)

            aug_input = torch.hstack([tvec, yvec])

            h = aug_input
            for layer in self.net:
                h = layer(h)
            #

            dyvec = h.reshape(shape_y)

            return dyvec
        
        else:
            #cprint('b', 'solve in alignment mode')
            shape_y = y.shape
            yvec = y.reshape([-1,1])
            tvec = t*torch.ones(yvec.shape).to(self.dummy.device)
            
            assert tvec.shape == t_align.shape
            
            t_vec_align = t_align*tvec

            aug_input = torch.hstack([t_vec_align, yvec])

            h = aug_input
            for layer in self.net:
                h = layer(h)
            #
            
            h_align = t_align*h

            dyvec = h_align.reshape(shape_y)

            return dyvec
        #

class InfFidNet2D(nn.Module):
    def __init__(self, 
                 in_dim,
                 h_dim,
                 s_dim,
                 dataset,
                 int_steps=2,
                 solver='dopri5',
                 g_width=40,
                 g_depth=2,
                 f_width=40,
                 f_depth=2,
                 A_width=40,
                 A_depth=2,
                 interp='bilinear',
        ):
        super(InfFidNet2D, self).__init__()
        
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.s_dim = s_dim
        
        self.int_steps = int_steps
        self.solver = solver
        self.dataset = dataset

        self.align = True
        
        self.interp = interp
        
        self.register_buffer('dummy', torch.tensor([]))
        
        g_config = [self.in_dim] + [g_width]*g_depth + [self.h_dim]
        f_config = [self.h_dim+self.in_dim+1] + [f_width]*f_depth + [self.h_dim]
        A_config = [2] + [A_width]*A_depth + [1]
        
        self.g_model = Net(g_config)
        self.f_model = ODEFunc(f_config)
        self.A_model = ODEFuncAt(A_config)

        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.log_nu = nn.Parameter(torch.tensor(0.0))

        self.A = nn.Parameter(torch.zeros(self.h_dim+1, self.s_dim**2))
        nn.init.xavier_normal_(self.A[:-1,:])

        
    def _single_solve_At(self, tt):
        if tt.item() == 0.0:
            tt = tt + 1e-8

        t_span = torch.linspace(torch.tensor(0.0), tt.item(), self.int_steps
                               ).to(self.dummy.device)
        
        dA = lambda t, y: self.A_model(t, y, t_align=None)
        
        soln_At = odeint(dA, self.A, t_span, method=self.solver)[-1,:]
        
        return soln_At
        
    def _signle_solve_At_align(self, tt):

        t_span = torch.linspace(torch.tensor(0.0), torch.tensor(1.0), self.int_steps
                               ).to(self.dummy.device)
        
        vec_t_align = tt*torch.ones(self.A.numel(), 1).to(self.dummy.device)
        
        dA_align = lambda t, y: self.A_model(t, y, t_align=vec_t_align)
        
        soln_At = odeint(dA_align, self.A, t_span, method=self.solver)[-1,:]
        
        return soln_At
    
    
    def _batch_solve_At_align(self, tt_list):
        
        t_span = torch.linspace(torch.tensor(0.0), torch.tensor(1.0), self.int_steps
                               ).to(self.dummy.device)
        ntt = len(tt_list)
        Astack = []
        vec_t_align_stack = []
        
        for tt in tt_list:
            vec_t_align = tt*torch.ones(self.A.numel(), 1).to(self.dummy.device)
            vec_t_align_stack.append(vec_t_align)
            Astack.append(self.A)
        #
        
        Astack = torch.vstack(Astack)
        vec_t_align_stack = torch.vstack(vec_t_align_stack)
      
        dA_align = lambda t, y: self.A_model(t, y, t_align=vec_t_align_stack)
        
        soln_At_stack = odeint(dA_align, Astack, t_span, method=self.solver)[-1,:]
        
        return soln_At_stack
    
    
    def _solve_At(self, tt_list, batch=True):
        
        A_list = {}
        
        if batch:
        
            bAt = self._batch_solve_At_align(tt_list)
            ib_A = 0

            for tt in tt_list:
                fid_t = self.dataset.func_t_to_fid(tt.item())
                At = bAt[ib_A:ib_A+self.h_dim+1, :]
                ib_A += (self.h_dim+1)
                A_list[fid_t] = At
            #

            assert ib_A == bAt.shape[0]
            
        else:
            for tt in tt_list:
                fid_t = self.dataset.func_t_to_fid(tt.item())
                At = self._single_solve_At(tt)
                A_list[fid_t] = At
            #
        #
        return A_list


    def _pack_batch_inputs(self, X_list, t_list):
        
        Nt = len(t_list) 
        tt_aug = []
        Xstack = []
        
        for tt in t_list:
            if tt.item() == 0.0:
                tt = tt + 1e-8
            X = X_list[self.dataset.func_t_to_fid(tt.item())]

            Xstack.append(X)
            
            N = X.shape[0]
            tt_aug.append(torch.ones([N,1]).to(self.dummy.device)*tt)
        #
        
        tt_aug = torch.vstack(tt_aug).to(self.dummy.device)
        Xstack = torch.vstack(Xstack).to(self.dummy.device)
        
        return Xstack, tt_aug
    
    def _batch_forward_h(self, X_list, t_list):
        
        bX, btt = self._pack_batch_inputs(X_list, t_list)
        bh0 = self.g_model(bX)

        t_span = torch.linspace(torch.tensor(0.0), torch.tensor(1.0), self.int_steps
                               ).to(self.dummy.device)
        
        df = lambda t, y: self.f_model(t, y, bX, btt)
        bht = odeint(df, bh0, t_span, method=self.solver)[-1,:,:]
        
        h_list = {}
        ib_h = 0
        
        for tt in t_list:
            fid_t = self.dataset.func_t_to_fid(tt.item())
            Nt = X_list[fid_t].shape[0]
            ht = bht[ib_h:ib_h+Nt, :]
            h_list[fid_t] = ht
            
            ib_h += Nt
        #
        assert ib_h == bht.shape[0]
        
        return h_list

        
    def _eval_llh(self, X_list, y_list, t_list):
        
        h_list = self._batch_forward_h(X_list, t_list)
        A_list = self._solve_At(t_list, batch=True)

        llh_list = []
        
        for tt in t_list:
            fid_t = self.dataset.func_t_to_fid(tt.item())

            yt = y_list[fid_t]
            ht = h_list[fid_t]
            At = A_list[fid_t]
            
            Nt = yt.shape[0]
            
            Am_t, Ab_t = At[:-1,:], At[-1,:]
            
            pred_yt = ht@Am_t + Ab_t
            
            pred_yt_2d = pred_yt.reshape([Nt, self.s_dim, self.s_dim])
            
            interp_pred_yt_2d = torch.nn.functional.interpolate(
                pred_yt_2d.unsqueeze(1), 
                size=fid_t, 
                mode=self.interp,
            ).squeeze(1)
            
            interp_pred_yt_1d = interp_pred_yt_2d.reshape([Nt, -1])

            dt = yt.shape[1]

            llh_t_n = 0.5*dt*self.log_tau - \
                0.5*dt*torch.log(2*torch.tensor(np.pi).to(self.dummy.device)) - \
                0.5*torch.exp(self.log_tau) * \
                (torch.square(interp_pred_yt_1d - yt).sum(1))
            
            llh_list.append(llh_t_n.sum())

        llh = sum(llh_list)

        return llh
    
    
    def _eval_prior(self,):
        
        param_list = []
        param_list += [torch.flatten(p) for p in self.g_model.parameters()]
        param_list += [torch.flatten(p) for p in self.f_model.parameters()]
        param_list += [torch.flatten(p) for p in self.A_model.parameters()]
        
        flat_ode_params = torch.cat(param_list)
        
        dim = flat_ode_params.shape[0]
        
        lprior = 0.5*dim*self.log_nu - \
                0.5*dim*torch.log(2*torch.tensor(np.pi)) - \
                0.5*torch.exp(self.log_nu)* \
                (torch.square(flat_ode_params).sum())
        
        return lprior
        
        
    def eval_loss(self, X_list, y_list, t_list):
        llh = self._eval_llh(X_list, y_list, t_list)
        lprior = self._eval_prior()
        nlogprob = -(llh+lprior)
        return nlogprob
    
    
    def predict(self, X_list, t_list):
        
        with torch.no_grad():
            h_list = self._batch_forward_h(X_list, t_list)
            A_list = self._solve_At(t_list, batch=True)

            pred_list = {}

            for tt in t_list:
                
                fid_t = self.dataset.func_t_to_fid(tt.item())
                ht = h_list[fid_t]
                At = A_list[fid_t]
                Xt = X_list[fid_t]

                Nt = Xt.shape[0]

                Am_t, Ab_t = At[:-1,:], At[-1,:]

                pred_yt = ht@Am_t + Ab_t

                pred_yt_2d = pred_yt.reshape([Nt, self.s_dim, self.s_dim])

                interp_pred_yt_2d = torch.nn.functional.interpolate(
                    pred_yt_2d.unsqueeze(1), 
                    size=fid_t, 
                    mode=self.interp,
                ).squeeze(1)

                interp_pred_yt_1d = interp_pred_yt_2d.reshape([Nt, -1])
                
                pred_list[fid_t] = interp_pred_yt_1d
            #
            
            return pred_list
        
    def eval_pred(self, X_list, t_list):
        
        pred_list = self.predict(X_list, t_list)
        np_pred_list = {}
        for tid, tt in enumerate(t_list):
            fid_t = self.dataset.func_t_to_fid(tt.item())
            pred_t = pred_list[fid_t]
            Nt = pred_t.shape[0]
            re_pred_t = pred_t.reshape([Nt, fid_t, fid_t])
            np_pred_list[fid_t] = re_pred_t.data.cpu().numpy()
        #
        return np_pred_list
        
    def eval_rmse(self, X_list, y_list, t_list, return_adjust=False):
        
        pred_list = self.predict(X_list, t_list)
        rmse_list = {}
        rmse_list_tune = []
        
        for tid, tt in enumerate(t_list):
            fid_t = self.dataset.func_t_to_fid(tt.item())
            pred_t = pred_list[fid_t]
            y_t = y_list[fid_t]
            rmse = torch.sqrt(torch.mean(torch.square(y_t-pred_t))) / \
                torch.sqrt(torch.square(y_t).mean())
            
            rmse_list[fid_t] = rmse.item()
            rmse_list_tune.append(rmse)
        #
        
        if return_adjust:
            return rmse_list, sum(rmse_list_tune)
        else:
            return rmse_list
            
    def eval_mae(self, X_list, y_list, t_list):
        
        pred_list = self.predict(X_list, t_list)
        mae_list = {}
        
        for tid, tt in enumerate(t_list):
            fid_t = self.dataset.func_t_to_fid(tt.item())
            pred_t = pred_list[fid_t]
            y_t = y_list[fid_t]
            mae = torch.abs(y_t-pred_t).mean() / torch.abs(y_t).mean() 
            mae_list[fid_t] = mae.item()
        #
        
        return mae_list
