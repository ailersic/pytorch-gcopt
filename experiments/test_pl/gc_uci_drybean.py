
import torch
from torch import nn
from torch.func import functional_call, grad
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import transforms, models
import pytorch_lightning as pl
import pandas as pd
import shutil
import numpy as np
import os
from sklearn import datasets
from ucimlrepo import fetch_ucirepo

from gc_module import ContNet

torch.set_float32_matmul_precision('high')

class DemoClassDataset(Dataset):
    def target_to_int(self, target):
        target_int = np.zeros_like(target)
        labels = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'HOROZ', 'SIRA', 'DERMASON']

        for i, label in enumerate(labels):
            target_int[target == label] = i
        return target_int

    def __init__(self, transform=None):
        # fetch dataset 
        uci_dataset = fetch_ucirepo(id=602) 

        # data
        self.X = uci_dataset.data.features.to_numpy()
        self.y = self.target_to_int(np.squeeze(uci_dataset.data.targets.to_numpy()))

        self.transform = transform

    def __len__(self):
        return len(self.y)
        #return len(self.sample_list)

    def __getitem__(self, idx):
        #sample = self.sample_list.iloc[idx, :].to_numpy()
        #return torch.from_numpy(sample[0:-1]).float(), torch.from_numpy(sample[-1:]).float()
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()

class DemoClass(ContNet):
    def __init__(self, loglambda0: float, cont_lr: float, cont_reg: float, warmup_epochs: int):
        super().__init__(loglambda0, cont_lr, cont_reg, warmup_epochs)
        self.lossfunc = nn.CrossEntropyLoss()

        self.datasetname = 'UCI_DryBean'
        self.epochs = 1000

        self.dataset = DemoClassDataset()#"class_data_n100_s50000.csv")
        size_train = int(len(self.dataset)*0.9)
        self.data_train, self.data_val = random_split(self.dataset, [size_train, len(self.dataset) - size_train])

        self.h_dim = 256
        self.n_layers = 5
        self.adam_lr = 1e-4

        activ = nn.ReLU()
        
        layers = []
        layers.extend([nn.Linear(16, self.h_dim), activ])
        for _ in range(self.n_layers - 2):
            layers.extend([nn.Linear(self.h_dim, self.h_dim), activ])
        layers.append(nn.Linear(self.h_dim, 7))

        self.net = nn.Sequential(*layers)
        
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        ypred = self.net(x)
        return ypred
    
    def configure_optimizers(self):
        # include logcontvar in optimizer
        optimizer = torch.optim.Adam([{'params': self.net.parameters()},
                                      {'params': (self.logcontvar,), 'lr': self.cont_lr}],
                                     lr=self.adam_lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        self.log('param_norm', sum(p.pow(2.0).sum() for p in self.parameters()))
        
        opt = self.optimizers()
        opt.zero_grad()

        # add gaussian noise to parameters
        self.perturb_params()
        
        # compute loss
        x, y = train_batch
        ypred = self.forward(x)
        loss = self.lossfunc(ypred, y)
        self.manual_backward(loss)

        # compute contvar gradient
        self.set_contvar_grad(self.calc_contvar_grad())
        
        # reload reference parameters
        self.unperturb_params()

        opt.step()
        self.log('train_loss', loss.detach())

        # compute accuracy
        match = torch.eq(y, torch.argmax(ypred, dim=1))
        acc = torch.sum(match.detach())/y.shape[0]

        self.log('train_acc', acc)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        ypred = self.net(x)

        loss = self.lossfunc(ypred, y)
        self.log('val_loss', loss)

        # compute accuracy
        match = torch.eq(y, torch.argmax(ypred, dim=1))
        acc = torch.sum(match.detach())/y.shape[0]

        self.log('val_acc', acc)
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=200, shuffle=True, num_workers=16, persistent_workers=True)
        
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=200, num_workers=16, persistent_workers=True)
