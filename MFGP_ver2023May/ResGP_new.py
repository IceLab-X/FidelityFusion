
import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)


import argparse
import time
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import torch
import torch.nn as nn
from MFGP_ver2023May.base_gp.cigp import CIGP
from MFGP_ver2023May.utils.dict_tools import update_dict_with_default

from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
# import MFGP_ver2023May.kernel.kernel as kernel


default_cigp_model_config = {
    'noise': {'init_value': 1., 'format': 'exp'},
    'kernel': {'SE': {'noise_exp_format':True, 'length_scale':1., 'scale': 1.}},
}

default_resgp_config = {
    'Residual': {'rho_value_init': 1., 'trainable': False},
    'cigp_model_config': default_cigp_model_config,
    'fidelity_shapes': [],
}


def arg_parse():
    parser = argparse.ArgumentParser(description="Residual GP")
    # parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    parser.add_argument(
        "--devices",
        default=1,
        help="gpu id(s) to use. int(0) for cpu. list[x,y] for xth, yth GPU."
        "str(x) for the first x GPUs. str(-1)/int(-1) for all available GPUs",
    )
    parser.add_argument("--ckpt_resume", default=None, help="path to train checkpoint file", type=str)
    parser.add_argument("--ckpt_test", default=None, help="path to test checkpoint file", type=str)
    args = parser.parse_args()
    return args



def encode_rows(matrix):
    """Encode rows of a matrix as strings for set operations."""
    return [','.join(map(str, row.tolist())) for row in matrix]

def find_matrix_row_overlap_and_indices(x_low, x_high):
    # Encode rows of both matrices
    encoded_x_low = encode_rows(x_low)
    encoded_x_high = encode_rows(x_high)

    # Find overlapping encoded rows and indices
    overlap_set = set(encoded_x_low).intersection(encoded_x_high)
    overlap_indices_low = [i for i, row in enumerate(encoded_x_low) if row in overlap_set]
    overlap_indices_high = [i for i, row in enumerate(encoded_x_high) if row in overlap_set]

    return overlap_set, overlap_indices_low, overlap_indices_high

def GP_train(GPmodel, xtr, ytr, lr=1e-1, max_iter=1000, verbose=True):
    optimizer = torch.optim.Adam(GPmodel.parameters(), lr)
    for i in range(max_iter):
        optimizer.zero_grad()
        loss = -GPmodel.log_likelihood(xtr, ytr)
        loss.backward()
        optimizer.step()
        if verbose:
            print('Iteration', i, 'Loss:', loss.item())

class ResGP(pl.LightningModule):
    def __init__(self, resgp_config):
        super(ResGP, self).__init__()
        # Model with custom kernels
        self.config = update_dict_with_default(default_resgp_config, resgp_config)
        self.fidelity_num = len(self.config['fidelity_shapes'])
        cigp_config_list = self.config['cigp_model_config']

        # kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
        self.low_fidelity_GP = CIGP(self.config)
        # kernel2 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
        self.high_fidelity_GP = CIGP(self.config)
        self.rho = nn.Parameter(torch.Tensor(1))


    def forward(self, x):
        """Forward pass for the model with a feature extractor and a classifier."""
        y_pred_low, _ = self.low_fidelity_GP.predict(x)
        y_pred_res, _ = self.high_fidelity_GP.predict(x)
        y_pred_high = y_pred_low + self.rho * y_pred_res
        return y_pred_high
    
    def compute_loss(self, x_list, y_list, to_fidelity_n=-1):
    
        """
        Compute loss for multiple fidelity levels.

        Args:
            x (torch.Tensor): Input tensor.
            y_list (list): List of output tensors for each fidelity level.
            to_fidelity_n (int, optional): Fidelity level to propagate to. Defaults to -1.

        Returns:
            torch.Tensor: Loss value.
        """
        loss = 0.
        for _fn in range(to_fidelity_n+1):
            if _fn == 0:
                loss += self.cigp_list[0].compute_loss(x_list[0], y_list[0])
            else:
                if self.nonsubset:
                    x, y_low, y_high = self._get_nonsubset_data(x_list[_fn-1], x_list[_fn], y_list[_fn-1], y_list[_fn], _fn)
                else:
                    x = x_list[0]
                    y_low = y_list[_fn-1]
                    y_high = y_list[_fn]

                res = self.residual_list[_fn-1].forward(y_low, y_high)
                loss += self.cigp_list[_fn].compute_loss(x, res, update_data=True)


        log_metrics = {
            f"loss": loss,
        }
        return torch.tensor(loss, requires_grad=True), log_metrics
    
    def training_step(self, train_batch, batch_idx) -> torch.Tensor:
        """Compute and return the training loss and metrics on one step. loss is to store the loss value. log_metrics
        is to store the metrics to be logged, including loss, top1 and/or top5 accuracies.

        Use self.log_dict(log_metrics, on_step, on_epoch, logger) to log the metrics on each step and each epoch. For
        training, log on each step and each epoch. For validation and testing, only log on each epoch. This way can
        avoid using on_training_epoch_end() and on_validation_epoch_end().
        """

        
        loss, metric = self.compute_loss(train_batch[0], train_batch[1])
        self.log_dict(metric, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def test_step(self, test_batch, batch_idx) -> torch.Tensor:
        """Compute and return the training loss and metrics on one step. loss is to store the loss value. log_metrics
        is to store the metrics to be logged, including loss, top1 and/or top5 accuracies.

        Use self.log_dict(log_metrics, on_step, on_epoch, logger) to log the metrics on each step and each epoch. For
        training, log on each step and each epoch. For validation and testing, only log on each epoch. This way can
        avoid using on_training_epoch_end() and on_validation_epoch_end().
        """

        
        loss, metric = self.compute_loss(test_batch[0], test_batch[1])
        self.log_dict(metric, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
        return [optimizer]

def get_testing_data(fidelity_num):
    x = np.load(r'../assets/MF_data/Poisson_data/input.npy')  
    y_list = [np.load(r'../assets/MF_data/Poisson_data/output_fidelity_{}.npy'.format(i)) for i in range(3)]
    y_list = y_list[:fidelity_num]

    x = torch.tensor(x)
    y_list = [torch.tensor(_) for _ in y_list]

    sample_num = x.shape[0]
    tr_x = x[:sample_num//2, ...].float()
    eval_x = x[sample_num//2:, ...].float()
    tr_y_list = [y[:sample_num//2, ...].float() for y in y_list]
    eval_y_list = [y[sample_num//2:, ...].float() for y in y_list]

    return tr_x, eval_x, tr_y_list, eval_y_list

def create_dataloaders(tr_x, tr_y_list, eval_x, eval_y_list):
    # Convert training data into a TensorDataset and then into a DataLoader
    train_data = TensorDataset(tr_x, *tr_y_list)
    train_loader = DataLoader(train_data, shuffle=True)

    # Similarly for validation data
    valid_data = TensorDataset(eval_x, *eval_y_list)
    valid_loader = DataLoader(valid_data, shuffle=False)

    return train_loader, valid_loader


def get_model(cfg):
    """
    Builds and returns a model according to the config object passed.

    Args:
        cfg: A YACS config object.
    """
    model = ResGP(cfg)

    return model




tr_x, eval_x, tr_y_list, eval_y_list = get_testing_data(1)
def main():
    args = arg_parse()

    # ---- setup config ----
    cfg = default_resgp_config
    
   

    # ---- setup dataset ----
    tr_x, eval_x, tr_y_list, eval_y_list = get_testing_data(1)
    train_loader, valid_loader = create_dataloaders(tr_x, tr_y_list, eval_x, eval_y_list)
    # ---- setup model ----
    print("==> Building model..")
    model = get_model(cfg)

    # ---- setup logger ----
    # Choose one logger (CometLogger or TensorBoardLogger) using cfg.COMET.ENABLE
    # if cfg.COMET.ENABLE:
    #     suffix = str(int(time.time() * 1000))[6:]
    #     logger = pl_loggers.CometLogger(
    #         api_key=cfg.COMET.API_KEY,
    #         project_name=cfg.COMET.PROJECT_NAME,
    #         save_dir=cfg.OUTPUT.OUT_DIR,
    #         experiment_name="{}_{}".format(cfg.COMET.EXPERIMENT_NAME, suffix),
    #     )
    # else:
    logger = pl_loggers.TensorBoardLogger("./outputs")

    # ---- setup callbacks ----
    # setup progress bar
    progress_bar = TQDMProgressBar(50)

    # setup learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ---- setup trainers ----
    trainer = pl.Trainer(

        max_epochs=100,
        accelerator="gpu" if args.devices != 0 else "cpu",
        devices=args.devices if args.devices != 0 else "auto",
        logger=logger,
        callbacks=[progress_bar, lr_monitor],
        strategy="ddp",  # comment this line on Windows, because Windows does not support CCL backend
        log_every_n_steps=1,

    )

    # ---- start training ----
    trainer.fit(model, train_loader, valid_loader)

    # ---- start testing ----
    trainer.test(model, valid_loader)


if __name__ == "__main__":
    main()












    # def train_model(self, x_train, y_train, max_iter=1000):
    #     x_low, y_low = x_train['low'], y_train['low']
    #     x_high, y_high = x_train['high'], y_train['high']

    #     # Train low fidelity GP
    #     GP_train(self.low_fidelity_GP, x_low, y_low, max_iter=max_iter)

    #     # Find overlap and train high fidelity GP
    #     _, indices_low, indices_high = find_matrix_row_overlap_and_indices(x_low, x_high)
    #     y_residual = y_high[indices_high] - self.rho * y_low[indices_low]
    #     GP_train(self.high_fidelity_GP, x_high[indices_high], y_residual, max_iter=max_iter)

    # rename to forward
        # def predict(self, x_test):
    #     y_pred_low, _ = self.low_fidelity_GP.predict(x_test)
    #     y_pred_res, _ = self.high_fidelity_GP.predict(x_test)
    #     y_pred_high = y_pred_low + self.rho * y_pred_res
    #     return y_pred_high

    # def compute_loss(self, x_list, y_list, to_fidelity_n=-1):
    #     """
    #     Compute loss for multiple fidelity levels.

    #     Args:
    #         x (torch.Tensor): Input tensor.
    #         y_list (list): List of output tensors for each fidelity level.
    #         to_fidelity_n (int, optional): Fidelity level to propagate to. Defaults to -1.

    #     Returns:
    #         torch.Tensor: Loss value.
    #     """
    #     loss = 0.
    #     for _fn in range(to_fidelity_n+1):
    #         if _fn == 0:
    #             loss += self.cigp_list[0].compute_loss(x_list[0], y_list[0])
    #         else:
    #             if self.nonsubset:
    #                 x, y_low, y_high = self._get_nonsubset_data(x_list[_fn-1], x_list[_fn], y_list[_fn-1], y_list[_fn], _fn)
    #             else:
    #                 x = x_list[0]
    #                 y_low = y_list[_fn-1]
    #                 y_high = y_list[_fn]

    #             res = self.residual_list[_fn-1].forward(y_low, y_high)
    #             loss += self.cigp_list[_fn].compute_loss(x, res, update_data=True)
    #     return loss


# # from MultiTaskGP_cigp import cigp

# # TODO: this codes needs to be improved for speed and memory usage
# def encode_rows(matrix):
#     """Encode rows of a matrix as strings for set operations."""
#     return [','.join(map(str, row.tolist())) for row in matrix]


# def find_matrix_row_overlap_and_indices(x_low, x_high):
#     # Encode rows of both matrices
#     encoded_x_low = encode_rows(x_low)
#     encoded_x_high = encode_rows(x_high)

#     # Find overlapping encoded rows
#     overlap_set = set(encoded_x_low).intersection(encoded_x_high)

#     # Get indices of overlapping rows
#     overlap_indices_low = [i for i, row in enumerate(encoded_x_low) if row in overlap_set]
#     overlap_indices_high = [i for i, row in enumerate(encoded_x_high) if row in overlap_set]

#     return overlap_set, overlap_indices_low, overlap_indices_high


# def GP_train(GPmodel, xtr, ytr, lr=1e-1, max_iter=1000, verbose=True):
#     optimizer = torch.optim.Adam(GPmodel.parameters(), lr=1e-1)
#     for i in range(max_iter):
#         optimizer.zero_grad()
#         loss = -GPmodel.log_likelihood(xtr, ytr)
#         loss.backward()
#         optimizer.step()
#         print('iter', i, 'nll:{:.5f}'.format(loss.item()))

    
# class ResGP(nn.Module):
#     # initialize the model
#     def __init__(self, gp):
#         super(ResGP, self).__init__()
        
#         # create the model
#         kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
#         self.low_fidelity_GP = CIGP(kernel=kernel1, noise_variance=1.0)
#         kernel2 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))
#         self.high_fidelity_GP = CIGP(kernel=kernel2, noise_variance=1.0)
#         self.rho = nn.Parameter(torch.Tensor(1))
    
#     # define the forward pass
#     def train(self, x_train, y_train, x_test):
#         # get the data
#         x_low = x_train[0]
#         y_low = y_train[0]
#         x_high = x_train[1]
#         y_high = y_train[1]
        
#         # train the low fidelity GP
#         GP_train(self.low_fidelity_GP, x_low, y_low, lr=1e-1, max_iter=1000, verbose=True)
        
#         # get the high fidelity part that is subset of the low fidelity part
#         overlap_set, overlap_indices_low, overlap_indices_high = find_matrix_row_overlap_and_indices(x_low, x_high)
        
#         # train the high fidelity GP
#         optimizer = torch.optim.Adam(self.high_fidelity_GP.parameters(), lr=1e-1)
#         for i in range(1000):
#             optimizer.zero_grad()
#             # y_residual = y_high[overlap_indices_high,:] - self.rho * self.low_fidelity_GP.predict(x_high[overlap_indices_high,:])
#             y_residual = y_high[overlap_indices_high,:] - self.rho * y_low[overlap_indices_low,:]
#             loss = -self.high_fidelity_GP.log_likelihood(overlap_set,y_residual)
#             loss.backward()
#             optimizer.step()
#             print('iter', i, 'nll:{:.5f}'.format(loss.item()))
        
#     def forward(self, x_test):
#         # predict the model
#         y_pred_low, cov_pred_low = self.low_fidelity_GP(x_test)
#         y_pred_res, cov_pred_res= self.high_fidelity_GP(x_test)
        
#         y_pred_high = y_pred_low + self.rho * y_pred_res
#         cov_pred_high = cov_pred_low + (self.rho **2) * cov_pred_res
        
#         # return the prediction
#         return y_pred_low, cov_pred_high