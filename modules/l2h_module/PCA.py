import torch
from modules.l2h_module.base_l2h_module import Basic_l2h
from utils import smart_update
from utils.type_define import *


class PCA(object):
    def __init__(self, Y, r=0.99):
        # centerizing data
        self.mean = torch.mean(Y, axis=0)
        Y = Y - self.mean
        
        U, S, Vh = torch.linalg.svd(Y, full_matrices=True)
        cumuEnergy = S.cumsum(dim=0) / S.sum(dim=0)
        
        if r >= 1:
            rank = r 
        if r < 1:
            rank = (cumuEnergy>r).nonzero()[0][0]

        self.rank = rank
        # truncate the singular values and vector 
        U = U[:,0:rank]
        S = S[0:rank]
        Vh = Vh[0:rank,:]
        
        self.U = U
        self.S = S
        self.Vh = Vh
        
        self.Z = U @ S.diag_embed()

    def project(self, X):
        X = X - self.mean
        return X @ self.Vh.t()
        
    def recover(self, Z):
        Y = Z @ self.Vh
        return Y + self.mean.expand_as(Y)

class listPCA(object):
    # A PCA for a list of data
    def __init__(self, Ylist, r=0.99):
        # nData = len(Ylist)
        self.model_list = []
        self.Zlist =[]
        for Y in Ylist:
        # for i in range(len(Ylist)):
            model = PCA(Y, r=r)
            self.model_list.append(model)
            self.Zlist.append(model.Z)
    
    def project(self, Xlist):
        Zlist = []       
        for i in range(len(Xlist)):
            Zlist.append(self.model_list[i].project(Xlist[i]))
        return Zlist
    
    def recover(self, Zlist):
        Ylist = []
        for i in range(len(Zlist)):
            Ylist.append(self.model_list[i].recover(Zlist[i]))
        return Ylist


default_config = {
    'rho_value_init': 1.,
    'trainable': True,
}

