
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

from gc_module import ContNet

torch.set_float32_matmul_precision('high')

class DemoRegDataset(Dataset):
    def __init__(self, sample_file, transform=None):
        self.sample_list = pd.read_csv(sample_file)
        self.transform = transform

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list.iloc[idx, :].to_numpy()
        return torch.from_numpy(sample[0:-1]).float(), torch.from_numpy(sample[-1:]).float()

class DemoRegress(ContNet):
    def __init__(self, loglambda0: float, cont_lr: float, cont_reg: float, warmup_epochs: int):
        super().__init__(loglambda0, cont_lr, cont_reg, warmup_epochs)
        self.lossfunc = F.mse_loss

        self.datasetname = 'RGC_DemoRegSpiky'
        self.epochs = 50

        self.dataset = DemoRegDataset("reg_data_n10_s10000.csv")
        size_train = int(len(self.dataset)*0.9)
        self.data_train, self.data_val = random_split(self.dataset, [size_train, len(self.dataset) - size_train])

        self.h_dim = 256
        self.n_layers = 7
        self.adam_lr = 1e-4

        activ = nn.ReLU()
        
        layers = []
        layers.extend([nn.Linear(10, self.h_dim), activ])
        for _ in range(self.n_layers - 2):
            layers.extend([nn.Linear(self.h_dim, self.h_dim), activ])
        layers.append(nn.Linear(self.h_dim, 1))

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

        # add gaussian noise to parameters
        rand_samp, ref_params = self.perturb_params()
        
        # compute loss
        x, y = train_batch

        n_opt = int(0.9*y.shape[0])
        n_gen = y.shape[0] - n_opt
        x_opt, x_gen = torch.split(x, [n_opt, n_gen], dim=0)
        y_opt, y_gen = torch.split(y, [n_opt, n_gen], dim=0)

        # compute contvar gradient with gen data
        opt.zero_grad()

        ypred = self.forward(x_gen)
        loss_gen = self.lossfunc(ypred, y_gen)
        self.manual_backward(loss_gen)

        contvar_grad = self.calc_contvar_grad(rand_samp, loss_gen)

        # compute param gradient with opt data
        opt.zero_grad()

        ypred = self.forward(x_opt)
        loss_opt = self.lossfunc(ypred, y_opt)
        self.manual_backward(loss_opt)

        self.set_contvar_grad(contvar_grad)
        
        # reload reference parameters
        self.load_state_dict(ref_params)

        opt.step()
        self.log('train_loss', loss_opt.detach())

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        ypred = self.net(x)

        loss = self.lossfunc(ypred, y)
        self.log('val_loss', loss)
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=50, shuffle=False, num_workers=16, persistent_workers=True)
        
    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=50, num_workers=16, persistent_workers=True)
