import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import GaussianProcess.kernel as kernel
from matplotlib import pyplot as plt
import time as time

print(torch.__version__)
# I use torch (1.11.0) for this work. lower version may not work.

JITTER = 1e-6
EPS = 1e-10
PI = 3.1415

class cigp(nn.Module):
    def __init__(self, kernel, log_beta):
        super(cigp, self).__init__()
        self.kernel = kernel
        self.log_beta = nn.Parameter(torch.tensor([log_beta]))


    def forward(self, x_train, y_train, x_test):
        
        if isinstance(y_train, list):
            y_train_var = y_train[1]
            y_train = y_train[0]
        else:
            y_train_var = None
        Sigma = self.kernel(x_train, x_train) + self.log_beta.exp().pow(-1) * torch.eye(x_train.size(0)).to(x_train.device) \
            + JITTER * torch.eye(x_train.size(0)).to(x_train.device)
        
        kx = self.kernel(x_train, x_test)
        L = torch.linalg.cholesky(Sigma)
        LinvKx,_ = torch.triangular_solve(kx, L, upper = False)

        # option 1
        mean = kx.t() @ torch.cholesky_solve(y_train, L)  # torch.linalg.cholesky()
        
        var = self.kernel(x_test, x_test) - LinvKx.t() @ LinvKx

        # add the noise uncertainty
        var = var + self.log_beta.exp().pow(-1)
        # if y_train_var is not None:
        #     var = var + y_train_var.diag()* torch.eye(x_test.size(0))

        return mean, var

    def negative_log_likelihood(self, x_train, y_train):
        if isinstance(y_train, list):
            y_train_var = y_train[1]
            y_train = y_train[0]
        else:
            y_train_var = None
        y_num, y_dimension = y_train.shape
        Sigma = self.kernel(x_train, x_train) + self.log_beta.exp().pow(-1) * torch.eye(
            x_train.size(0)).to(x_train.device) + JITTER * torch.eye(x_train.size(0)).to(x_train.device)
        if y_train_var is not None:
            Sigma = Sigma + y_train_var.diag()* torch.eye(x_train.size(0)).to(x_train.device)
        L = torch.linalg.cholesky(Sigma)
        #option 1 (use this if torch supports)
        Gamma,_ = torch.triangular_solve(y_train, L, upper = False)
        #option 2
        # gamma = L.inverse() @ Y       # we can use this as an alternative because L is a lower triangular matrix.

        nll =  0.5 * (Gamma ** 2).sum() +  L.diag().log().sum() * y_dimension  \
            + 0.5 * y_num * torch.log(2 * torch.tensor(PI)) * y_dimension
        return -nll


if __name__ == "__main__":

    # single output test 1
    xte = torch.linspace(0, 6, 100).view(-1, 1)
    yte = torch.sin(xte) + 10

    xtr = torch.rand(16, 1) * 6
    ytr = torch.sin(xtr) + torch.randn(16, 1) * 0.5 + 10

    kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
    model = cigp(kernel = kernel1, log_beta = 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)
    
    for i in range(100):
        startTime = time.time()
        optimizer.zero_grad()
        loss = -model.negative_log_likelihood(xtr, ytr)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        timeElapsed = time.time() - startTime
        print('time elapsed: {:.3f}s'.format(timeElapsed))
    with torch.no_grad():
        ypred, ypred_var = model.forward(xtr,ytr,xte)

    plt.figure()
    plt.errorbar(xte.flatten(), ypred.reshape(-1).detach(), ypred_var.diag().sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.fill_between(xte.flatten(), ypred.reshape(-1).detach() - ypred_var.diag().sqrt().squeeze().detach(), ypred.reshape(-1).detach() + ypred_var.diag().sqrt().squeeze().detach(), alpha = 0.2)
    plt.plot(xte.flatten(), yte, 'k+')
    plt.show()


    # single output test 2
    xte = torch.rand(128,2) * 2
    yte = torch.sin(xte.sum(1)).view(-1,1) + 10

    xtr = torch.rand(32, 2) * 2
    ytr = torch.sin(xtr.sum(1)).view(-1,1) + torch.randn(32, 1) * 0.5 + 10

    kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
    model = cigp(kernel = kernel1, log_beta = 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)
    
    for i in range(300):
        startTime = time.time()
        optimizer.zero_grad()
        loss = -model.negative_log_likelihood(xtr, ytr)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        timeElapsed = time.time() - startTime
        print('time elapsed: {:.3f}s'.format(timeElapsed))
    with torch.no_grad():
        ypred, ypred_var = model.forward(xtr,ytr,xte)

    # plt.errorbar(xte.sum(1), ypred.reshape(-1).detach(), ystd.sqrt().squeeze().detach(), fmt='r-.' ,alpha = 0.2)
    plt.plot(xte.sum(1), yte, 'b+')
    plt.plot(xte.sum(1), ypred.reshape(-1).detach(), 'r+')
    # plt.plot(xtr.sum(1), ytr, 'b+')
    plt.show()
    

    # multi output test
    xte = torch.linspace(0, 6, 100).view(-1, 1)
    yte = torch.hstack([torch.sin(xte),
                       torch.cos(xte),
                        xte.tanh()] )

    xtr = torch.rand(32, 1) * 6
    ytr = torch.sin(xtr) + torch.rand(32, 1) * 0.5
    ytr = torch.hstack([torch.sin(xtr),
                       torch.cos(xtr),
                        xtr.tanh()] )+ torch.randn(32, 3) * 0.2


    kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
    model = cigp(kernel = kernel1, log_beta = 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-1)
    
    for i in range(300):
        startTime = time.time()
        optimizer.zero_grad()
        loss = -model.negative_log_likelihood(xtr, ytr)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        timeElapsed = time.time() - startTime
        print('time elapsed: {:.3f}s'.format(timeElapsed))
    with torch.no_grad():
        ypred, ypred_var = model.forward(xtr,ytr,xte)

    # plt.errorbar(xte, ypred.detach(), ypred_var.sqrt().squeeze().detach(),fmt='r-.' ,alpha = 0.2)
    plt.plot(xte, ypred.detach(),'r-.')
    plt.plot(xtr, ytr, 'b+')
    plt.plot(xte, yte, 'k-')
    plt.show()

    # plt.close('all')
    plt.plot(xtr, ytr, 'b+')
    for i in range(3):
        plt.plot(xte, yte[:, i], label='truth', color='r')
        plt.plot(xte, ypred[:, i], label='prediction', color='navy')
        plt.fill_between(xte.squeeze(-1).detach().numpy(),
                         ypred[:, i].squeeze(-1).detach().numpy() + torch.sqrt(ypred_var[:, i].squeeze(-1)).detach().numpy(),
                         ypred[:, i].squeeze(-1).detach().numpy() - torch.sqrt(ypred_var[:, i].squeeze(-1)).detach().numpy(),
                         alpha=0.2)
    plt.show()

