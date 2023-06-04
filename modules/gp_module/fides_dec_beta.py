import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

print(torch.__version__)

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # Fixing strange error if run in MacOS
JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

class fides_dec(nn.Module):
    def __init__(self, 
                xtr,
                ytr, 
                xte,
                train_begin_index, 
                train_num, 
                fidelity_num, 
                seed, 
                niteration,
                learning_rate,
                normal_y_mode=0):
        super(fides_dec, self).__init__()

        # initiate parameters
        self.train_begin_index = train_begin_index
        self.train_num = train_num
        self.fidelity_num = fidelity_num
        self.seed = seed
        self.niteration = niteration
        self.learning_rate = learning_rate

        x = xtr[0][train_begin_index:train_begin_index+train_num[0]]
        
        # normalize X independently for each dimension
        self.Xmean = x.mean(0)
        self.Xstd = x.std(0)
        self.X = (x - self.Xmean.expand_as(x)) / (self.Xstd.expand_as(x) + EPS)
        
        self.Y = ytr
        self.Ymean = []
        self.Ystd = []

        self.xte = xte

        # GP hyperparameters
        self.log_beta = nn.Parameter(torch.ones(1) * 0.0000001)
        self.log_length_scale = nn.Parameter(torch.zeros(xtr[0].shape[1]))    # ARD length scale
        self.log_length_scale_z = nn.Parameter(torch.zeros(1))    # ARD length scale for t
        self.log_scale = nn.Parameter(torch.zeros(1))   # kernel scale

        # Matern3 hyperparameters for x
        self.log_length_matern3 = torch.nn.Parameter(torch.zeros(xtr[0].shape[1]))  # Matern3 Kernel length
        self.log_coe_matern3 = torch.nn.Parameter(torch.zeros(1))  # Matern3 Kernel coefficient
        # Matern3 hyperparameters for z
        self.log_length_matern3_z = torch.nn.Parameter(torch.zeros(1)) 


        # parameters for residuals part
        # initialize the parameter of function G
        self.b = nn.Parameter(torch.ones(1))

    def y_norm(self, normal_y_mode = 1):
        Y = []
        if normal_y_mode == 0:
            # normalize y all together
            y_m = self.Y.mean()
            y_s = self.Y.std()
            self.Ymean.append(y_m)
            self.Ystd.append(y_s)
            Y = (self.Y - y_m.expand_as(self.Y)) / (y_s.expand_as(self.Y) + EPS)
        elif normal_y_mode == 1:
            # normalize y by each dimension
            for yy in self.Y:
                y_m = yy.mean(0)
                y_s = yy.std(0)
                self.Ymean.append(y_m)
                self.Ystd.append(y_s)
                Y.append((yy - y_m.expand_as(yy)) / (y_s.expand_as(yy) + EPS))
        return Y

    def y_denorm(self, y, index,normal_y_mode = 0):
        if normal_y_mode == 0:
            # normalize y all together
            y_m = self.Ymean[index]
            y_s = self.Ystd[index]
            Y = y * y_s + y_m
        elif normal_y_mode == 1:
            # normalize y by each dimension
            y_m = self.Ymean[index]
            y_s = self.Ystd[index]
            Y = y * y_s + y_m
        return Y
    
    # def kernel_matern3
    def kernel(self, x1, x2):
        """
        latex formula:
        \sigma ^2\left( 1+\frac{\sqrt{3}d}{\rho} \right) \exp \left( -\frac{\sqrt{3}d}{\rho} \right)
        :param x1: x_point1
        :param x2: x_point2
        :return: kernel matrix
        """
        const_sqrt_3 = torch.sqrt(torch.ones(1) * 3)
        x1 = x1 / self.log_length_matern3.exp()
        x2 = x2 / self.log_length_matern3.exp()
        distance = const_sqrt_3 * torch.cdist(x1, x2, p=2)
        k_matern3 = self.log_coe_matern3.exp() * (1 + distance) * (- distance).exp()
        return k_matern3

    def forward(self, x_tr, y_tr, index, Xte):
        n_test = Xte.size(0)
        Xte = ( Xte - self.Xmean.expand_as(Xte) ) / self.Xstd.expand_as(Xte)

        Sigma = self.kernel(x_tr, x_tr) + self.log_beta.exp().pow(-1) * torch.eye(x_tr.size(0)) \
            + JITTER * torch.eye(x_tr.size(0))

        kx = self.kernel(x_tr, Xte)
        L = torch.cholesky(Sigma)
        LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

        # option 1
        mean = kx.t() @ torch.cholesky_solve(y_tr, L)  # torch.linalg.cholesky()
        
        var_diag = self.kernel(Xte, Xte).diag().view(-1, 1) \
            - (LinvKx**2).sum(dim = 0).view(-1, 1)

        # add the noise uncertainty
        var_diag = var_diag + self.log_beta.exp().pow(-1)

        # de-normalized
        # mean = mean * self.Ystd[index].expand_as(mean) + self.Ymean[index].expand_as(mean)
        # var_diag = var_diag.expand_as(mean) * self.Ystd[index]**2

        return mean, var_diag

    def negative_log_likelihood(self, x, y):
        y_num, y_dimension = y.shape
        Sigma = self.kernel(x, x) + self.log_beta.exp().pow(-1) * torch.eye(
            x.size(0)) + JITTER * torch.eye(x.size(0))

        L = torch.linalg.cholesky(Sigma)
        #option 1 (use this if torch supports)
        Gamma,_ = torch.triangular_solve(y, L, upper = False)
        #option 2
        # gamma = L.inverse() @ Y       # we can use this as an alternative because L is a lower triangular matrix.

        nll =  0.5 * (Gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
        return nll

    def train_adam(self, x, y, niteration=10, lr=0.1):
        # adam optimizer
        # uncommont the following to enable
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            # self.update()
            loss = self.negative_log_likelihood(x, y)
            loss.backward()
            optimizer.step()
            # print('loss_nll:', loss.item())
            # print('iter', i, ' nll:', loss.item())
            print('iter', i, 'nll:{:.5f}'.format(loss.item()))

    def warp(self, lf1, hf1, lf2, hf2):
        '''
        l = [lf1, hf1, lf2, hf2]
        tem = []
        for i in range(4):
            tem.append(1 - pow((1 - pow(l[i], self.warp_a[i])), self.warp_b[i]) )
        
        l1 = tem[0]
        l2 = tem[2]
        h1 = tem[1]
        h2 = tem[3]

        return l1, h1, l2, h2
        '''
        return lf1, hf1, lf2, hf2

    def forward_res(self, x_tr, y_tr, l1, h1, l2, h2, index, Xte):
        n_test = Xte.size(0)
        Xte = ( Xte - self.Xmean.expand_as(Xte) ) / self.Xstd.expand_as(Xte)

        Sigma = self.kernel_res(x_tr, x_tr, l1, h1, l2, h2) + self.log_beta.exp().pow(-1) * torch.eye(x_tr.size(0)) \
            + JITTER * torch.eye(x_tr.size(0))

        kx = self.kernel_res(x_tr, Xte, l1, h1, l2, h2)
        L = torch.cholesky(Sigma)
        LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

        # option 1
        y_res = y_tr[0] - (-self.b * (h1 - l1)).exp() * y_tr[1]
        mean = kx.t() @ torch.cholesky_solve(y_res, L)  # torch.linalg.cholesky()
        
        var_diag = self.kernel_res(Xte, Xte, l1, h1, l2, h2).diag().view(-1, 1) \
            - (LinvKx**2).sum(dim = 0).view(-1, 1)

        # add the noise uncertainty
        var_diag = var_diag + self.log_beta.exp().pow(-1)

        # de-normalized
        # mean = mean * self.Ystd[index].expand_as(mean) + self.Ymean[index].expand_as(mean)
        # var_diag = var_diag.expand_as(mean) * self.Ystd[index]**2
        
        return mean, var_diag
    
    def kernel_res(self, X1, X2, l1, h1, l2, h2):
        lf1, hf1, lf2, hf2 = self.warp(l1, h1, l2, h2)

        N = 100
        torch.manual_seed(self.seed)
        # print(torch.rand(1))
        z1 = torch.rand(N) * (hf1 - lf1) + lf1 # 这块需要用来调整z选点的范围
        z2 = torch.rand(N) * (hf2 - lf2) + lf2

        X1 = X1 / self.log_length_scale.exp()
        X2 = X2 / self.log_length_scale.exp()
        # X1_norm2 = X1 * X1
        # X2_norm2 = X2 * X2
        X1_norm2 = torch.sum(X1 * X1, dim=1).view(-1, 1)
        X2_norm2 = torch.sum(X2 * X2, dim=1).view(-1, 1)

        K = -2.0 * X1 @ X2.t() + X1_norm2.expand(X1.size(0), X2.size(0)) + X2_norm2.t().expand(X1.size(0), X2.size(0))  
        # this is the effective Euclidean distance matrix between X1 and X2.
        K = self.log_scale.exp() * torch.exp(-0.5 * K)
        
        # z part use MCMC to calculate the integral
        dist_z = (z1 / self.log_length_scale_z.exp() - z2 / self.log_length_scale_z.exp()) ** 2
        z_part1 = -self.b * (z1 - hf1)
        z_part2 = -self.b * (z2 - hf2)
        z_part  = (z_part1 + z_part2 - 0.5 * dist_z).exp()
        z_part_mc = z_part.mean() * (hf1 - lf1) * (hf2 - lf2)
        # z_part_mc = z_part.mean()
        
        K_ard = z_part_mc * K
        return K_ard

    def negative_log_likelihood_res(self, x, y, l1, h1, l2, h2):
        # y = [ytr_h, ytr_l]
        y_num, y_dimension = y[0].shape
        Sigma = self.kernel_res(x, x, l1, h1, l2, h2) + self.log_beta.exp().pow(-1) * torch.eye(
        x.size(0)) + JITTER * torch.eye(x.size(0))

        L = torch.linalg.cholesky(Sigma)
        #option 1 (use this if torch supports)
        y_res = y[0] - (-self.b * (h1 - l1)).exp() * y[1]
        Gamma,_ = torch.triangular_solve(y_res, L, upper = False)
        #option 2
        # gamma = L.inverse() @ Y       # we can use this as an alternative because L is a lower triangular matrix.

        nll =  0.5 * (Gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension

        return nll

    def train_adam_res(self, x, y, l1, h1, l2, h2, niteration=10, lr=0.1):
        # adam optimizer
        # uncommont the following to enable
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()
        for i in range(niteration):
            optimizer.zero_grad()
            # self.update()
            loss = self.negative_log_likelihood_res(x, y, l1, h1, l2, h2)
            loss.backward()
            optimizer.step()
            # print('loss_nll:', loss.item())
            # print('iter', i, ' nll:', loss.item())
            print('iter', i, 'nll:{:.5f}'.format(loss.item()))
            print(self.b)
 
    def train_and_test_lowest_fidelity(self):
        x = self.X[self.train_begin_index:self.train_begin_index+self.train_num[0]]
        y = self.Y[0][self.train_begin_index:self.train_begin_index+self.train_num[0]]
        # y = self.y_norm(y)
        self.train_adam(x = x, y = y, niteration = self.niteration, lr = self.learning_rate)
        self.yte_mean, self.yte_var = self.forward(x, y, 0, self.xte)
        print("finish lowest fidelity")
        

    def train_mod(self):
        self.Y = self.y_norm(normal_y_mode = 1)
        self.train_and_test_lowest_fidelity()
        for fid in range(self.fidelity_num - 1):
            hf = fid + 1
            lf = fid
            # b = self.b.detach()
            
            # ytr_res = self.Y[hf][self.train_begin_index:self.train_begin_index + self.train_num[hf]] - (-b * (hf - lf)).exp() * self.Y[lf][self.train_begin_index:self.train_begin_index+self.train_num[hf]]
            # ytr_res = self.Y[hf][self.train_begin_index:self.train_begin_index + self.train_num[hf]] -  self.Y[lf][self.train_begin_index:self.train_begin_index+self.train_num[hf]]

            ytr_h = self.Y[hf][self.train_begin_index:self.train_begin_index + self.train_num[hf]]
            ytr_l = self.Y[lf][self.train_begin_index:self.train_begin_index + self.train_num[hf]]

            xtr_res = self.X[self.train_begin_index:self.train_begin_index + self.train_num[hf]]
            
            # ytr_res = self.y_norm(ytr_res)
            # Train the residual part model
            self.train_adam_res(x = xtr_res, y = [ytr_h, ytr_l], l1 = lf, h1 = hf, l2 = lf, h2 = hf, niteration = self.niteration, lr = self.learning_rate)
            # Predict the residual part
            yte_res_mean, yte_res_var = self.forward_res(xtr_res, [ytr_h, ytr_l], l1 = lf, h1 = hf, l2 = lf, h2 = hf, index = hf, Xte = self.xte)
            # Add on to get the highest fidelity prediction
            self.yte_mean = (- self.b * (hf - lf)).exp()* self.yte_mean + yte_res_mean
            self.yte_var = (- 2 * self.b * (hf - lf)).exp() * self.yte_var + yte_res_var

            print("finish", hf, "fidelity")

        # Denormalize
        mean = self.yte_mean * self.Ystd[self.fidelity_num - 1].expand_as(self.yte_mean) + self.Ymean[self.fidelity_num - 1].expand_as(self.yte_mean)
        var = self.yte_var.expand_as(self.yte_mean) * self.Ystd[self.fidelity_num - 1]**2

        return mean, var

        

        
if __name__ == '__main__':
    
    fides_dec("Heat_mfGent_v5",
                train_begin_index = 0, 
                test_begin_index = 0,
                train_samples_num = 64, 
                test_samples_num = 128, 
                # test_samples_num = 100,
                dec_rate = 0.5,
                fidelity_num = 5,
                seed = 0,
                need_inerp = True)
        