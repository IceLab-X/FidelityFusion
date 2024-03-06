import os
from torchvision.datasets import CIFAR100
from ISLP.torch import (SimpleDataModule,
                        SimpleModule,
                        ErrorTracker,
                        rec_num_workers)
from torchvision.transforms import ToTensor

import torch
from torch import nn
from torch.optim import RMSprop
from torch.utils.data import TensorDataset

# !pip install torchmetrics==0.11.4

# !pip install pytorch_lightning==1.7.7
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
seed_everything(0, workers=True)  # to reproduce results
# to use dterministic algorithms where possible
torch.use_deterministic_algorithms(True, warn_only=True)


class BuildingBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(BuildingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3),
                              padding='same')
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        return self.pool(self.activation(self.conv(x)))


class CIFARModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(CIFARModel, self).__init__()
        sizes = [(3, 32),
                 (32, 64),
                 (64, 128),
                 (128, 256)]
        self.conv = nn.Sequential(*[BuildingBlock(in_, out_)
                                    for in_, out_ in sizes])
        self.output = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(2*2*256, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 100))

    def forward(self, x):
        val = self.conv(x)
        val = torch.flatten(val, start_dim=1)
        return self.output(val)


class CnnCifar(nn.Module):
    def __init__(
            self,
            file: str = os.getcwd(),
            num_workers=rec_num_workers(),
            accelerator=None):
        '''
        Args:
            file:
                directory to store data
                default: current working directory
            num_workers:
                how many processes we will use for loading the data
                default: maximum number of processors on the computer
        '''

        cifar_train, cifar_test = [CIFAR100(root='./data/',
                                            train=train,
                                            download=True)
                                   for train in [True, False]]
        transform = ToTensor()
        cifar_train_X = torch.stack([transform(x)
                                     for x in cifar_train.data])
        cifar_test_X = torch.stack([transform(x)
                                    for x in cifar_test.data])

        self.cifar_train = TensorDataset(cifar_train_X,
                                         torch.tensor(cifar_train.targets))
        self.cifar_test = TensorDataset(cifar_test_X,
                                        torch.tensor(cifar_test.targets))

        self.num_workers = num_workers
        self.accelerator = accelerator

    def train(self, learning_rate=0.001, dropout_rate=0.5, batch_size=128, epochs=30):

        cifar_model = CIFARModel(dropout=dropout_rate)
        cifar_dm = SimpleDataModule(self.cifar_train,
                                    self.cifar_test,
                                    validation=self.cifar_test,
                                    num_workers=self.num_workers,
                                    batch_size=batch_size)
        cifar_optimizer = RMSprop(cifar_model.parameters(),
                                  lr=learning_rate)
        cifar_module = SimpleModule.classification(cifar_model,
                                                   num_classes=100,
                                                   optimizer=cifar_optimizer)
        self.cifar_trainer = Trainer(accelerator=self.accelerator,
                                     deterministic=True,
                                     max_epochs=epochs,
                                     callbacks=[ErrorTracker()])
        self.cifar_trainer.fit(cifar_module,
                               datamodule=cifar_dm
                               )
        
        return self.cifar_trainer.test(cifar_module,
                                       datamodule=cifar_dm,
                                       verbose=False)[0]['test_accuracy']
