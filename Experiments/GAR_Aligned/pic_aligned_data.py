import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

Dic = {'ar': ['#ff7f0e', "o", "solid", "AR"],
       'nar': ['#708090', "o", "solid", "NAR"],
       'dc': ['#17becf', "o", "solid", "DC-I"],
       'resgp': ['#8c564b', "o", "solid", "ResGP"],
       'dmfal': ['#2ca02c', "o", "solid", "MF-BNN"],
       'gar': ['#DC143C', "o", "dashed", "GAR"],
       'cigar': ['#1f77b4', "o", "dashed", "CIGAR"],
        }

data_name = 'sample_data'
methods_name_list = ['nar', 'ar']

for methods_name in methods_name_list:
    ct = []
    tem = []
    for seed in [0, 1]:
        path = os.path.join(sys.path[0], 'exp_results', data_name, methods_name + '_seed_' + str(seed) + '.csv')
        data = pd.DataFrame(pd.read_csv(path))
        orders = data['train_sample_num'].to_numpy().reshape(-1, 1).flatten()
        rmse = data['rmse'].to_numpy().reshape(-1, 1)
        tem.append(rmse)

    tem = np.array(tem)
    mean = np.mean(tem, axis=0).flatten()
    var = np.std(tem, axis=0).flatten()
    plt.errorbar(orders, mean, yerr = var, ls = Dic[methods_name][2], linewidth=3.5, color=Dic[methods_name][0],
                label=Dic[methods_name][-1], marker=Dic[methods_name][1], fillstyle='full',
                elinewidth = 3 ,capsize = 8, markersize = 12, alpha = 0.8)

    plt.xlabel("#HF Samples", fontsize=20)
    plt.ylabel("RMSE", fontsize = 20)


plt.legend(loc="upper right", fontsize=20)
plt.grid()

plt.tight_layout()
fig_file = os.path.join(sys.path[0], 'pics')
if not os.path.exists(fig_file):
        os.makedirs(fig_file)
plt.savefig(fig_file  + '/' + data_name + '.png',
            bbox_inches='tight')
