
import torch
from torch import nn
from torch.func import functional_call, grad
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms, models
import pytorch_lightning as pl
import os

from gc_module import ContNet

torch.set_float32_matmul_precision('high')


class CIFAR10Class(ContNet):
    def __init__(self, logcontvar0: float, loggradvar0: float, cont_lr: float, cont_reg: float, warmup_epochs: int):
        super().__init__(logcontvar0, loggradvar0, cont_lr, cont_reg, warmup_epochs)
        self.lossfunc = F.cross_entropy

        self.datasetname = 'CIFAR10_ResNet50'
        self.epochs = 20

        self.dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
        size_train = int(len(self.dataset)*0.9)
        self.data_train, self.data_val = random_split(self.dataset, [size_train, len(self.dataset) - size_train])

        self.net = models.resnet50()
        self.net.fc = nn.Linear(self.net.fc.in_features, 10)

        '''
        activ = nn.ReLU()
        self.net = nn.Sequential(
                                nn.Conv2d(3, 64, 3, padding=1), activ,
                                nn.Conv2d(64, 32, 3, padding=1), activ,
                                nn.Conv2d(32, 16, 3, padding=1), activ,
                                nn.Conv2d(16, 8, 3, padding=1), activ,
                                nn.Flatten(start_dim=1),
                                nn.Linear(8192, 256),
                                nn.Softmax(dim=1)
                                )'''
        
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                #nn.init.zeros_(layer.bias)

    def forward(self, x):
        ypred = self.net(x)
        return ypred
    
    def configure_optimizers(self):
        # include logcontvar in optimizer
        optimizer = torch.optim.Adam([{'params': self.net.parameters()},
                                      {'params': (self.logcontvar,), 'lr': self.cont_lr}],
                                     lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        self.log('param_norm', sum(p.pow(2.0).sum() for p in self.parameters()))
        
        opt = self.optimizers()

        # add gaussian noise to parameters
        self.perturb_params()
        
        # compute loss
        x, y = train_batch
        ypred = self.forward(x)
        loss = self.lossfunc(ypred, y)
        
        # calculate accuracy
        match = torch.eq(y, torch.argmax(ypred.detach(), dim=1))
        acc = torch.sum(match)/y.shape[0]

        opt.zero_grad()
        self.manual_backward(loss)

        # compute contvar gradient
        self.set_contvar_grad(self.calc_contvar_grad())
        
        # reload reference parameters
        self.unperturb_params()

        opt.step()
        self.log('train_loss', loss.detach())
        self.log('train_acc', acc)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        ypred = self.net(x)

        loss = self.lossfunc(ypred, y)
        self.log('val_loss', loss)
        
        match = torch.eq(y, torch.argmax(ypred, dim=1))
        acc = torch.sum(match)/y.shape[0]
        self.log('val_acc', acc)
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=128, shuffle=True, num_workers=16, persistent_workers=True)
        
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=128, num_workers=16, persistent_workers=True)
