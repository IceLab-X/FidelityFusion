import sys
import numpy as np
import os
import random
import torch
from scipy.io import loadmat

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import Standard_mat_DataLoader
from utils.data_loader import SP_DataLoader

def w_h(t):
    w=t**2+0.1*torch.sin(10*torch.pi*t)
    #w=1-torch.exp(-3*t)-0.1*torch.exp(-t)*torch.sin(20*torch.pi*t)
    #w=1-0.2*torch.exp(-1.5*t)*torch.cos(8*torch.pi*t)+0.4*torch.log(0.5+t)/(2*t+0.5)
    return w

def w_l(t):
    w=1-w_h(t)
    #w=torch.exp(-3*t)-0.1*torch.exp(-t)*torch.cos(20*torch.pi*t)
    #w=torch.exp(-2.5*t)*(1-0.6*torch.sin(10*torch.pi*t))
    return w

def data_preparation(data_name,
                     fidelity_num,
                     seed,
                     train_samples_num):
    
    SP_DataLoader_available = ['FlowMix3D_MF',
                            'MolecularDynamic_MF', 
                            'plasmonic2_MF', 
                            'SOFC_MF',
                            #  'NavierStock_mfGent_v1_02', 
                            ]

    Standard_mat_DataLoader_available = ['Burget_mfGent_v5_02',
                            'Burget_mfGent_v5_15',
                            'Heat_mfGent_v5_15',
                            'Heat_mfGent_v5',
                            'Heat_mfGent_v5',
                            'Poisson_mfGent_v5_15',
                            'Poisson_mfGent_v5',
                            'Schroed2D_mfGent_v1',
                            'TopOP_mfGent_v5',
                            'TopOP_mfGent_v6',
                            ]
    
    load_data_certain_fi_available=['borehole','branin', 'currin', 'maolin1','maolin5','maolin6','maolin7', 'maolin8', 'maolin7', 'maolin8','maolin10','maolin12','maolin13','maolin15','maolin19','maolin20']


    if data_name in Standard_mat_DataLoader_available:
        mat_data = Standard_mat_DataLoader(data_name, True)
        xxtr, xytr, xte, yte = mat_data.get_data()
        random.seed(seed)
        ind = [random.randint(0,train_samples_num-1) for i in range(train_samples_num)] # generating the index of data for training
        xtr = [torch.stack([xxtr[0][j] for j in ind])]
        ytr = []
        for i in range(len(xytr)):
            ytr.append(torch.stack([xytr[i][j] for j in ind]))
            
    elif data_name in SP_DataLoader_available:
        mat_data = SP_DataLoader(data_name, None)
        xxtr, xytr, xte, yte = mat_data.get_data()

        random.seed(seed)
        ind = [random.randint(0,train_samples_num-1) for i in range(train_samples_num)] # generating the index of data for training
        if data_name == "TopOP_mfGent_v6":
            xtr = [torch.stack([torch.tensor(xxtr[0][j]) for j in ind])]
            ytr = []
            for i in range(len(xytr)):
                ytr.append(torch.stack([torch.tensor(xytr[i][j]) for j in ind]))
        else:
            xtr = [torch.stack([xxtr[0][j] for j in ind])]
            ytr = []
            for i in range(len(xytr)):
                ytr.append(torch.stack([xytr[i][j] for j in ind]))

        

    elif data_name in load_data_certain_fi_available:
        mat_data = loadmat(sys.path[-1] + "/data/gen_mf_data/" + str(data_name) + ".mat")

        l = 1
        h = fidelity_num
        fidelity_list=torch.linspace(l-1, l, h).view(-1,1)
        fidelity_list=fidelity_list[1:-1]

        xtr = [torch.tensor(mat_data['xtr'])]
        xte = [torch.tensor(mat_data['xte'])]
        Ytr_l = torch.tensor(mat_data['Ytr_l'])
        Ytr_h = torch.tensor(mat_data['Ytr_h'])
        Yte_l = torch.tensor(mat_data['Yte_l'])
        Yte_h = torch.tensor(mat_data['Yte_h'])
        yytr=[Ytr_l]
        yte=[Yte_l]
        for t in fidelity_list:
            ytr_fid=w_l(t)*Ytr_l+w_h(t)*Ytr_h
            yte_fid=w_l(t)*Yte_l+w_h(t)*Yte_h
            yytr.append(ytr_fid)
            yte.append(yte_fid)
        yytr.append(Ytr_h)
        yte.append(Yte_h)

        random.seed(seed)
        ind = [random.randint(0,train_samples_num-1) for i in range(train_samples_num)] # generating the index of data for training
        xtr = [torch.stack([xtr[0][j] for j in ind])]
        ytr = []
        for i in range(len(yytr)):
            ytr.append(torch.stack([yytr[i][j] for j in ind]))

    return xtr, ytr, xte, yte