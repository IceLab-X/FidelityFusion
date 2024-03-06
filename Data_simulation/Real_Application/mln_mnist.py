import os
from torchvision.datasets import MNIST  # , CIFAR100
from ISLP.torch import (SimpleDataModule,
                        SimpleModule,
                        ErrorTracker,
                        rec_num_workers)
from torchvision.transforms import ToTensor

import torch
from torch import nn

# !pip install torchmetrics==0.11.4


# !pip install pytorch_lightning==1.7.7
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
seed_everything(0, workers=True)  # to reproduce results
# to use dterministic algorithms where possible
torch.use_deterministic_algorithms(True, warn_only=True)


class MNISTModel(nn.Module):
    def __init__(self, dropout=0.4):
        super(MNISTModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self._forward = nn.Sequential(
            self.layer1,
            self.layer2,
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self._forward(x)


class MlnMnist(nn.Module):
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

        self.mnist_train, self.mnist_test = [MNIST(root='./data/',
                                                   train=train,
                                                   download=True,
                                                   transform=ToTensor())
                                             for train in [True, False]]
        self.num_workers = num_workers
        self.accelerator = accelerator

    def train(self, dropout_rate=0.4, batch_size=256, epochs=30):

        mnist_model = MNISTModel(dropout=dropout_rate)
        mnist_dm = SimpleDataModule(self.mnist_train,
                                    self.mnist_test,
                                    validation=self.mnist_test,
                                    num_workers=self.num_workers,
                                    batch_size=batch_size)
        mnist_module = SimpleModule.classification(mnist_model,
                                                   num_classes=10)
        self.mnist_trainer = Trainer(accelerator=self.accelerator,
                                     deterministic=True,
                                     max_epochs=epochs,
                                     callbacks=[ErrorTracker()])
        self.mnist_trainer.fit(mnist_module,
                               datamodule=mnist_dm)

        return self.mnist_trainer.test(mnist_module,
                                       datamodule=mnist_dm,
                                       verbose=False)[0]['test_accuracy']
