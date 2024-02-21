import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from FidelityFusion_Models.Load_Mfdata import get_full_name_list_with_fidelity

Dic = {'AR': ['#ff7f0e', "o", "solid", "AR"],
       'NAR': ['#708090', "o", "solid", "NAR"],
       'dc': ['#17becf', "o", "solid", "DC-I"],
       'ResGP': ['#8c564b', "o", "solid", "ResGP"],
       'dmfal': ['#2ca02c', "o", "solid", "MF-BNN"],
       'GAR': ['#DC143C', "o", "dashed", "GAR"],
       'CIGAR': ['#1f77b4', "o", "dashed", "CIGAR"],
        }

# data_name = 'forrester12'
data_name_list = ["colville", "nonlinearsin", "toal", "forrester",
                          "tl1", "tl2", "tl3", "tl4", "tl5", "tl6", "tl7", "tl8", "tl9", "tl10",
                          "p1", "p2", "p3", "p4", "p5",
                          "maolin1", "maolin5", "maolin6", "maolin7", "maolin8", "maolin10", "maolin12", "maolin13",
                          "maolin15",
                          "maolin19", "maolin20",
                          "shuo6", "shuo11", "shuo15", "shuo16",
                          "test3", "test4", "test5", "test6", "test7"]
methods_name_list = ['NAR', 'AR', 'ResGP', 'GAR', 'CIGAR']
all_data_name_with_fi_list = get_full_name_list_with_fidelity(data_name_list=data_name_list)   
for data_name in all_data_name_with_fi_list:
    plt.figure()
    for methods_name in methods_name_list:
        ct = []
        tem = []
        for seed in [0, 1, 2, 3, 4]:
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
