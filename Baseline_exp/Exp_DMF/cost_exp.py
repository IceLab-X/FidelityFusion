import torch
import sys
import pandas as pd
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import GaussianProcess.kernel as kernel
from FidelityFusion_Models import *
from FidelityFusion_Models.MF_data import MultiFidelityDataManager
from MF_BayesianOptimization.Discrete.DMF_acq import DiscreteAcquisitionFunction, optimize_acq_mf
import matplotlib.pyplot as plt
torch.manual_seed(1)

data_list = ["forrester"]
model_dic = {'AR': AR, 'ResGP': ResGP, 'NAR': NAR, 'CIGAR': CIGAR, 'GAR': GAR, 'CAR': ContinuousAutoRegression}
train_dic = {'AR': train_AR,'ResGP': train_ResGP, 'NAR': train_NAR,'CIGAR': train_CIGAR, 'GAR': train_GAR, 'CAR': train_CAR}

def DMF_BO(exp_config):
    pass
if __name__ == '__main__':
    data_name = "forrester"
    model_list = ['ResGP']
    acq_list = ['UCB', 'EI', 'PI', 'KG']
    for model in model_list:
        for acq in acq_list:
            exp_config = {
                'MF_model': model,
                'acq': acq,
                'model_iter': 200,
                'BO_iter': 10,
            }
        record = DMF_BO(exp_config)
        path_csv = os.path.join(sys.path[-1], 'Experiment', 'DMF', 'Exp_results',
                                        data_name)
        if not os.path.exists(path_csv):
            os.makedirs(path_csv)

        df = pd.DataFrame(record)
        df.to_csv(path_csv + '/'+ model +'_' + exp_config['Acq_function'] + '_seed_' + str(seed) + '.csv',
                    index=False)
