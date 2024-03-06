# -*- coding = utf-8 -*-
# @Time : 9/10/23 10:30
# @Author : Alison_W
# @File : graphs.py
# @Software : PyCharm

import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import sys

realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
realpath = _sep.join(realpath[:realpath.index('MFBO') + 1])
sys.path.append(realpath)

def get_data(type, data_name, method_name, file_name):
    path = os.path.join(sys.path[-1], 'exp', type, data_name, method_name, file_name + '.csv')
    data = pd.DataFrame(pd.read_csv(path))
    return data


Dic = {'FiDEs_UCB': ['navy', "o", "FiDEs_UCB"],
       'FiDEs_EI': ['saddlebrown', "v", "FiDEs_EI"],
       'FiDEs_ES': ['green', "s", "FiDEs_ES"],
       'FiDEs_cfKG': ['orange', "*", "FiDEs_CFKG"],
       'ar_UCB': ['darkgreen', "+", "AR_UCB"],
       'ar_EI': ['green', "+", "AR_EI"],
       'ar_ES': ['lime', "+", "AR_ES"],
       'resgp_UCB': ['saddlebrown', "+", "ResGP_UCB"],
       'resgp_EI': ['chocolate', "+", "ResGP_EI"],
       'resgp_ES': ['sandybrown', "+", "ResGP_ES"],
       'smac': ['deeppink', "X", "SMAC"],
       'fabolas': ['fuchsia', "+", "FABOLAS"], }

data_name = 'forrester'
# methods_name_list = ['FiDEs_UCB', 'fides_EI', 'fides_ES', 'fides_CFKG', 'smac']
methods_name_list = ['FiDEs_UCB', 'FiDEs_EI', 'FiDEs_ES', 'FiDEs_cfKG']
line = []
for methods_name in methods_name_list:
    ct = []
    tem = []
    for seed in [0]:
        path = os.path.join(sys.path[-1], 'Experiment', 'Exp_discrete_1', 'Exp_results',
                            data_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        cost = data['cost'].to_numpy().reshape(-1, 1)
        incumbents = data['incumbents'].to_numpy().reshape(-1, 1)
        tem.append(incumbents)
        ct.append(cost)
    tem = np.array(tem)
    mean = np.mean(tem, axis=0)
    var = np.std(tem, axis=0)
    ll = plt.plot(ct[0].flatten(), mean.flatten(), ls='dashed', color=Dic[methods_name][0],
                   label=Dic[methods_name][2],
                   marker=Dic[methods_name][1], markersize=8)
    plt.fill_between(ct[0].flatten(),
                     mean.flatten() - 0.96 * var.flatten(),
                     mean.flatten() + 0.96 * var.flatten(),
                     alpha=0.1, color=Dic[methods_name][0])
    # line.append(ll)
    plt.xlabel("Cost", fontsize=25)
    plt.ylabel("Max_Value", fontsize=25)
    # plt.xticks(labelsize=20)
    # plt.yticks(labelsize=20)
label = [Dic[i][-1] for i in methods_name_list]
plt.legend(loc="upper right", fontsize=20)
plt.grid()

plt.tight_layout()
plt.savefig(os.path.join(sys.path[-1], 'Experiment', 'Exp_discrete_1', 'Graphs') + '/' + data_name + '.png',
            bbox_inches='tight')
