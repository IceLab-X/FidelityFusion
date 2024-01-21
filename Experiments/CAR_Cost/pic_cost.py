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
       'car': ['#1f77b4', "o", "dashed", "Ours"],
        }

data_name = 'sample_data'
methods_name_list = ['car']
start_seed = 21
# cost实验一般不需要跑好几个seed取平均值 因为已经很随机了

for methods_name in methods_name_list:
    ct = []
    tem = []
    path = os.path.join(sys.path[0], 'exp_results', data_name, methods_name + '_seed_' + str(start_seed) + '.csv')
    data = pd.DataFrame(pd.read_csv(path))
    sorted_data = data.sort_values(by='cost', ascending=True)
    orders = sorted_data['cost'].to_numpy().reshape(-1, 1).flatten()
    rmse = sorted_data['rmse'].to_numpy().reshape(-1, 1)
    tem.append(rmse)

    plt.plot(orders, rmse, ls = Dic[methods_name][2], linewidth=3.5, color=Dic[methods_name][0], label=Dic[methods_name][-1], marker=Dic[methods_name][1], markersize = 12, alpha = 0.8)

    plt.xlabel("# Cost", fontsize=20)
    plt.ylabel("RMSE", fontsize = 20)


plt.legend(loc="upper right", fontsize=20)
plt.grid()

plt.tight_layout()
fig_file = os.path.join(sys.path[0], 'pics')
if not os.path.exists(fig_file):
        os.makedirs(fig_file)
plt.savefig(fig_file  + '/' + data_name + '.png',
            bbox_inches='tight')
