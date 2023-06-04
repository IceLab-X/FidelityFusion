import numpy as np
import random
import time


import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchdiffeq import odeint

import tensorly as tl
tl.set_backend('pytorch')

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

torch.set_default_tensor_type(torch.DoubleTensor)

class InitNet(nn.Module):
    def __init__(self, config, act=nn.Tanh()):
        
        super(InitNet, self).__init__()
 
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
        
        self.register_buffer('dummy', torch.tensor([]))
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, val=0)
            #
        #

    def forward(self, t, y, X, t_align=None):

        if t_align is None:
            
            assert y.shape[0] == X.shape[0]
            
            aug_input = torch.hstack([y, X])

            h = aug_input
            for layer in self.net:
                h = layer(h)
            #

            yt = h

            return yt
        else:
            
            assert y.shape[0] == X.shape[0] == t_align.shape[0]
            
            aug_input = torch.hstack([y, X])

            h = aug_input
            for layer in self.net:
                h = layer(h)
            #
            
            h_align = t_align * h

            yt = h_align

            return yt
    
class ODEFuncTime(nn.Module):
    
    def __init__(self, config, act=nn.Tanh()):
        
        super(ODEFuncTime, self).__init__()
 
        buff_layers = []

        for l in range(len(config)-2):
            in_dim = config[l]
            out_dim = config[l+1]
            buff_layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            buff_layers.append(nn.Tanh())
        #
        buff_layers.append(nn.Linear(in_features=config[-2], out_features=config[-1]))

        self.net = nn.ModuleList(buff_layers)
        
        self.register_buffer('dummy', torch.tensor([]))
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, val=0)
            #
        #

    def forward(self, t, y, X, t_align=None):

        if t_align is None:
            
            assert y.shape[0] == X.shape[0]

            tvec = t*torch.ones([y.shape[0], 1]).to(self.dummy.device)
            aug_input = torch.hstack([tvec, y, X])

            h = aug_input
            for layer in self.net:
                h = layer(h)
            #

            yt = h

            return yt
        
        else:
            
            assert y.shape[0] == X.shape[0] == t_align.shape[0]

            tvec = t*torch.ones([y.shape[0], 1]).to(self.dummy.device)
            
            aug_input = torch.hstack([t_align*tvec, y, X])

            h = aug_input
            for layer in self.net:
                h = layer(h)
            #
            
            h_align = t_align * h

            yt = h_align
            
            return yt
        
class KernelRBF(nn.Module):
    def __init__(self, jitter=1e-5):
        super().__init__()     
        self.register_buffer('jitter', torch.tensor(jitter))
        self.log_amp = nn.Parameter(torch.tensor(0.0))
        self.log_ls = nn.Parameter(torch.tensor(0.0))
        
    def matrix(self, X):
        K = self.cross(X, X)
        Ijit = self.jitter*torch.eye(X.shape[0], device=self.jitter.device)
        K = K + Ijit
        return K

    def cross(self, X1, X2):
        norm1 = torch.reshape(torch.sum(torch.square(X1), dim=1), [-1,1])
        norm2 = torch.reshape(torch.sum(torch.square(X2), dim=1), [1,-1])        
        K = norm1-2.0*torch.matmul(X1,X2.T) + norm2
        K = torch.exp(-1.0*K/torch.exp(self.log_ls))
        return K
    
    
class KernelARD(nn.Module):
    def __init__(self, jitter=1e-5):
        super().__init__()     
        self.register_buffer('jitter', torch.tensor(jitter))
        self.log_amp = nn.Parameter(torch.tensor(0.0))
        self.log_ls = nn.Parameter(torch.tensor(0.0))
        
    def matrix(self, X):
        K = self.cross(X, X)
        Ijit = self.jitter*torch.eye(X.shape[0], device=self.jitter.device)
        K = K + Ijit
        return K

    def cross(self, X1, X2):
        norm1 = torch.reshape(torch.sum(torch.square(X1), dim=1), [-1,1])
        norm2 = torch.reshape(torch.sum(torch.square(X2), dim=1), [1,-1])        
        K = norm1-2.0*torch.matmul(X1,X2.T) + norm2
        K = torch.exp(self.log_amp)*torch.exp(-1.0*K/torch.exp(self.log_ls))
        return K


class HoA(nn.Module):
    
    def __init__(self, d, K, T):
        
        super(HoA, self).__init__()
        
        self.d = d
        self.K = K
        self.T = T
        
        self.register_buffer('dummy', torch.tensor([]))
        
        self.mu = nn.Parameter(torch.rand([d, K, T]))
        self.chol_d = nn.Parameter(torch.eye(d))
        self.chol_K = nn.Parameter(torch.eye(K))
        self.chol_T = nn.Parameter(torch.eye(T))


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
                 ode_t=True,
                 interp=None,
        ):
        
        super(InfFidNet2D, self).__init__()
        
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.s_dim = s_dim
        
        self.int_steps = int_steps
        self.solver = solver
        self.dataset = dataset
        
        self.interp = interp
        
        self.register_buffer('dummy', torch.tensor([]))
        
        g_config = [self.in_dim] + [g_width]*g_depth + [self.h_dim]
        self.g_model = InitNet(g_config)

        if ode_t:
            f_config = [self.h_dim+self.in_dim+1] + [f_width]*f_depth + [self.h_dim]
            self.f_model = ODEFuncTime(f_config)
        else:
            f_config = [self.h_dim+self.in_dim] + [f_width]*f_depth + [self.h_dim]
            self.f_model = ODEFunc(f_config)
        #
        
        self.log_tau = nn.Parameter(torch.tensor(0.0))        

        self.KerT = KernelRBF()
        #self.KerT = KernelARD()
        self.A = HoA(
            d=self.s_dim*self.s_dim, 
            K=self.h_dim, 
            T=len(self.dataset.ns_list_tr)
        )
        
        

    def _solve_ht(self, Xt, tt):
        
        t_span = torch.linspace(torch.tensor(0.0), tt.item(), self.int_steps
                               ).to(self.dummy.device)
        
        h0 = self.g_model(Xt)
        
        dh = lambda t, y: self.f_model(t, y, Xt, t_align=None)
        
        soln_ht = odeint(dh, h0, t_span, method=self.solver)
        
        return soln_ht
    
    def _solve_ht_align(self, Xt, tt):
        
        t_span = torch.linspace(torch.tensor(0.0), torch.tensor(1.0), self.int_steps
                               ).to(self.dummy.device)
        
        h0 = self.g_model(Xt)
        
        vec_t_align = tt*torch.ones(h0.shape[0], 1).to(self.dummy.device)
        
        dh = lambda t, y: self.f_model(t, y, Xt, t_align=vec_t_align)
        
        soln_ht = odeint(dh, h0, t_span, method=self.solver)
        
        return soln_ht
    
    
    def _pack_batch_inputs(self, X_list, t_list):
        
        Nt = len(t_list) 
        t_align_stack = []
        Xstack = []
        
        for tt in t_list:
            if tt.item() == 0.0:
                tt = tt + 1e-8
                
            fid_t = self.dataset.func_t_to_fid(tt.item())
            X = X_list[fid_t]

            Xstack.append(X)
            
            N = X.shape[0]
            t_align_stack.append(tt*torch.ones([N,1]).to(self.dummy.device))
        #
        
        t_align_stack = torch.vstack(t_align_stack).to(self.dummy.device)
        Xstack = torch.vstack(Xstack).to(self.dummy.device)
        
        return Xstack, t_align_stack

    def _forward_latent_single(self, X_list, t_list):
        
        ht_list = {}
        
        for tt in t_list:
            
            if tt.item() == 0.0:
                tt = tt + 1e-8
                
            fid_t = self.dataset.func_t_to_fid(tt.item())
            Xt = X_list[fid_t]
            
            soln_ht = self._solve_ht_align(Xt, tt)
            ht = soln_ht[-1,:]
            ht_list[fid_t] = ht
        #
        
        return ht_list
    
    def _forward_latent_batch(self, X_list, t_list):
        bX, b_t_align = self._pack_batch_inputs(X_list, t_list)
        soln_bht = self._solve_ht_align(bX, b_t_align)
        bht = soln_bht[-1,:]
        ht_list = {}
        ib = 0
        for tt in t_list:
            fid_t = self.dataset.func_t_to_fid(tt.item())
            Xt = X_list[fid_t]
            Nt = Xt.shape[0]
            
            ht = bht[ib:ib+Nt, :]
            ht_list[fid_t] = ht
            ib += Nt
        #
        assert ib == bht.shape[0]
            
        return ht_list
    
    
    def _eval_elbo_llh_t(self, ht, yt, tt):

        fid_t = self.dataset.func_t_to_fid(tt.item())
        tid = self.dataset.fid_list_tr.index(fid_t)
        
        N = yt.shape[0]
        
        yt_2d = yt.reshape([N, fid_t, -1]).unsqueeze(1)
        
        interp_yt_2d = \
            torch.nn.functional.interpolate(
                yt_2d, 
                size=self.s_dim,
                mode=self.interp,
            ).squeeze(1)
        
        re_yt = interp_yt_2d.reshape([N, -1])
        
        term1 = torch.sum(torch.square(re_yt), dim=1, keepdim=True)
        EE_A_t = self.A.mu[:,:,tid]
        
        term2 = torch.bmm(ht.unsqueeze(1) @ (EE_A_t.T), re_yt.unsqueeze(-1)).squeeze(-1)
        
        L_d = torch.tril(self.A.chol_d)
        L_K = torch.tril(self.A.chol_K)
        L_T = torch.tril(self.A.chol_T)
        
        Sd = torch.mm(L_d, L_d.T)
        SK = torch.mm(L_K, L_K.T)
        ST = torch.mm(L_T, L_T.T)
        
        V = SK * ST[tid, tid]
        U = Sd
        
        EE_A_t_T_A_t = V * torch.trace(U) + torch.mm(EE_A_t.T, EE_A_t)
        
        term3 = torch.bmm((ht.unsqueeze(1))@EE_A_t_T_A_t, ht.unsqueeze(-1)).squeeze(-1)
        
        tau = torch.exp(self.log_tau)
        y_dim = self.s_dim**2
        
        llh_n = -0.5*y_dim*np.log(2*np.pi) + 0.5*y_dim*self.log_tau - \
            0.5*tau*(term1 - 2*term2 + term3)
        
        llh_t = torch.sum(llh_n)
        
        return llh_t
    
    def _eval_elbo_llh(self, X_list, y_list, t_list):
        
        h_list = self._forward_latent_batch(X_list, t_list)
        llh_list = []
        
        for tid, tt in enumerate(t_list):
            fid_t = self.dataset.func_t_to_fid(tt.item())
            ht = h_list[fid_t]
            yt = y_list[fid_t]
            llh_t = self._eval_elbo_llh_t(ht, yt, tt)
            llh_list.append(llh_t)
        #
        
        llh = sum(llh_list)
        
        return llh
    
    
    def _eval_kl_div(self, t_list):
        
        tvec = torch.vstack(t_list)
        Kt = self.KerT.matrix(tvec)
        L_Kt = torch.linalg.cholesky(Kt)
        
        d, K, T = self.A.d, self.A.K, self.A.T
        
        # compute logdet pA
        log_det_pA = 2*d*K*torch.sum(torch.log(torch.diag(L_Kt)))
        
        L_d = torch.tril(self.A.chol_d)
        L_K = torch.tril(self.A.chol_K)
        L_T = torch.tril(self.A.chol_T)
        
        Sd = torch.mm(L_d, L_d.T)
        SK = torch.mm(L_K, L_K.T)
        ST = torch.mm(L_T, L_T.T)
        
        # compute logdet qA
        log_det_qA = 2*K*T*torch.sum(torch.log(torch.diag(L_d))) +\
            2*d*T*torch.sum(torch.log(torch.diag(L_K))) +\
            2*d*K*torch.sum(torch.log(torch.diag(L_T)))
        
        
        # trace term
        tr_term = torch.trace(Sd)*torch.trace(SK)*torch.trace(torch.linalg.inv(Kt)@ST)

        # quadratic term
        L_Kt_inv = torch.linalg.inv(L_Kt)
        mu = self.A.mu
        #print(mu.shape)
        
        Id = torch.eye(d).to(self.dummy.device)
        IK = torch.eye(K).to(self.dummy.device)
        
        quad_tensor = tl.tenalg.mode_dot(mu, Id, mode=0)
        quad_tensor = tl.tenalg.mode_dot(quad_tensor, IK, mode=1)
        quad_tensor = tl.tenalg.mode_dot(quad_tensor, L_Kt_inv, mode=2)
        
        quad_term = torch.sum(torch.square(quad_tensor))
        
        
        kld = 0.5*(log_det_pA - log_det_qA - d*K*T + tr_term + quad_term)
        
        return kld
    
    
    def eval_nelbo(self, X_list, y_list, t_list):
        
        llh = self._eval_elbo_llh(X_list, y_list, t_list)
        kld = self._eval_kl_div(t_list)
        
        nelbo = kld - llh
        
        return nelbo
    
    
    def _sample_At_star(self, t_list, t_list_star, num_samples):
        tvec = torch.vstack(t_list)
        tvec_star = torch.vstack(t_list_star)
        
        K_TT = self.KerT.matrix(tvec)
        K_TstarT = self.KerT.cross(tvec_star, tvec)
        
        As_list = []
        if num_samples == 0:
            Amu = self.A.mu
            kstar = K_TstarT@torch.linalg.inv(K_TT)
            As_tr = torch.tensordot(kstar, torch.permute(Amu,(2,0,1)), dims=1)
            As = torch.permute(As_tr, (1,2,0))
            As_list.append(As)
        else:
            Amu = self.A.mu
            L_d = torch.tril(self.A.chol_d)
            L_K = torch.tril(self.A.chol_K)
            L_T = torch.tril(self.A.chol_T)
            for s in range(ns):
                epsi = torch.randn(size=Amu.shape).to(self.dummy.device)
                epsi = tl.tenalg.mode_dot(epsi, L_d, mode=0)
                epsi = tl.tenalg.mode_dot(epsi, L_K, mode=1)
                epsi = tl.tenalg.mode_dot(epsi, L_T, mode=2)
                As = Amu + epsi
                As_list.append(As)
            #
            
        return As_list
    
    
    def predict(self, X_list, t_list, t_list_tr):
        
        with torch.no_grad():
            
            As = self._sample_At_star(t_list_tr, t_list, num_samples=0)[0]
            h_list = self._forward_latent_batch(X_list, t_list)
            pred_list = {}
            
            for tid, tt in enumerate(t_list):
                fid_t = self.dataset.func_t_to_fid(tt.item())
                ht = h_list[fid_t]
                At = As[:,:,tid]
                Nt = ht.shape[0]
                pred_t = torch.mm(ht, At.T).reshape([Nt, self.s_dim, self.s_dim])
                interp_pred_t_2d = \
                    torch.nn.functional.interpolate(
                        pred_t.unsqueeze(1), 
                        size=fid_t,
                        mode=self.interp,
                    ).squeeze(1)
                interp_pred_t_1d = interp_pred_t_2d.reshape([Nt, -1])
                pred_list[fid_t] = interp_pred_t_1d
            # 

            return pred_list
        
    def eval_pred(self, X_list, t_list, t_list_tr):
        
        pred_list = self.predict(X_list, t_list, t_list_tr)
        np_pred_list = {}
        for tid, tt in enumerate(t_list):
            fid_t = self.dataset.func_t_to_fid(tt.item())
            pred_t = pred_list[fid_t]
            Nt = pred_t.shape[0]
            re_pred_t = pred_t.reshape([Nt, fid_t, fid_t])
            np_pred_list[fid_t] = re_pred_t.data.cpu().numpy()
        #
        return np_pred_list
        
    def eval_rmse(self, X_list, y_list, t_list, t_list_tr, return_adjust=False):
        
        pred_list = self.predict(X_list, t_list, t_list_tr)
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
        
        
    def eval_mae(self, X_list, y_list, t_list, t_list_tr):
        
        pred_list = self.predict(X_list, t_list, t_list_tr)
        mae_list = {}
        
        for tid, tt in enumerate(t_list):
            fid_t = self.dataset.func_t_to_fid(tt.item())
            pred_t = pred_list[fid_t]
            y_t = y_list[fid_t]
            mae = torch.abs(y_t-pred_t).mean() / torch.abs(y_t).mean() 
            mae_list[fid_t] = mae.item()
        #

        return mae_list
       
